"""Admission control: decide whether a pending job may start *now*.

Admission is the gate that keeps a single node from over-committing itself into
an OOM cascade. It is deliberately a stack of *simple* checks rather than a
clever NUMA-accounting scheme -- the lived lesson is that elaborate per-node
bookkeeping gets bypassed (by memory interleaving, by jobs launched outside the
scheduler) and a few blunt limits are what actually hold:

#. **Pause sentinel** -- a ``.pause`` file freezes all launches (manual brake).
#. **OOM cooldown** -- after a system OOM, no launches for a grace period so
   memory truly frees before retrying.
#. **Concurrency cap** -- a hard ceiling on simultaneous running jobs.
#. **Exclusive mutex** -- an ``exclusive`` job runs alone; nothing starts
   beside it and it does not start beside anything.
#. **Core budget** -- declared cores of running jobs + this job <= budget.
#. **RAM guard** -- the real one: if the cgroup slice is live, project
   ``slice.memory.current + ram_gb`` against ``slice.memory.max - safety``;
   otherwise fall back to declared-RAM accounting plus a live ``/proc/meminfo``
   free-memory check.
#. **maxcore sanity** -- warn (do not block) when ``ram_gb`` looks too small
   for ``maxcore * nprocs * 1.3``.

The public surface is :func:`can_launch`, returning ``(ok, reason)`` so the
scheduler can log *why* a job is waiting.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from nodepilot import cgroups
from nodepilot.config import Config, Job, JobStatus

__all__ = [
    "Decision",
    "AdmissionController",
    "running_cores",
    "running_ram_gb",
    "maxcore_sane",
]


@dataclass(frozen=True)
class Decision:
    """Result of an admission check.

    Attributes
    ----------
    ok
        Whether the job may launch now.
    reason
        Human-readable explanation (``"ok"`` on success; the failing rule
        otherwise). Always safe to log.
    """

    ok: bool
    reason: str

    def __bool__(self) -> bool:  # convenience: ``if decision:``
        return self.ok


def running_cores(jobs: list[Job]) -> int:
    """Total declared cores held by currently running jobs."""
    return sum(j.cores for j in jobs if j.status == JobStatus.RUNNING)


def running_ram_gb(jobs: list[Job]) -> float:
    """Total declared RAM (GiB) held by currently running jobs."""
    return sum(j.ram_gb for j in jobs if j.status == JobStatus.RUNNING)


def running_count(jobs: list[Job]) -> int:
    """Number of currently running jobs."""
    return sum(1 for j in jobs if j.status == JobStatus.RUNNING)


def maxcore_sane(job: Job, factor: float = 1.3) -> bool:
    """Whether ``ram_gb`` covers ``maxcore * nprocs * factor``.

    Mirrors the rule of thumb for codes that allocate ``maxcore`` MiB per
    process: the job's RAM budget should exceed the aggregate per-process
    allocation with headroom. Jobs with ``maxcore == 0`` are always "sane"
    (the hint is not provided). This is advisory only.
    """
    if job.maxcore <= 0:
        return True
    needed_gb = job.maxcore * job.effective_nprocs() * factor / 1024.0
    return job.ram_gb >= needed_gb


def _free_ram_gb() -> float | None:
    """Available system RAM in GiB from ``/proc/meminfo`` (``MemAvailable``)."""
    try:
        with open("/proc/meminfo", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024 * 1024)
    except (OSError, ValueError):
        pass
    return None


class AdmissionController:
    """Stateful admission gate for one orchestrator instance.

    Holds the small amount of mutable state admission needs (the OOM cooldown
    deadline) and a :class:`~nodepilot.cgroups.SliceMonitor` for the live RAM
    guard. Construct one per run and call :meth:`can_launch` each tick.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self._slice = cgroups.SliceMonitor(config.memory_slice) if config.memory_slice else None
        self._cooldown_until = 0.0
        #: Cached slice ceiling (GiB). Re-read lazily; an operator may change it
        #: at runtime, so :meth:`_slice_cap_gb` always re-queries.

    # -- cooldown management --------------------------------------------
    def trigger_oom_cooldown(self) -> None:
        """Freeze launches for ``config.oom_cooldown_seconds`` from now."""
        self._cooldown_until = time.time() + self.config.oom_cooldown_seconds

    def in_cooldown(self) -> float:
        """Seconds remaining in the OOM cooldown (0 if not cooling down)."""
        return max(0.0, self._cooldown_until - time.time())

    # -- RAM guard helpers ----------------------------------------------
    def _slice_used_gb(self) -> float | None:
        return self._slice.used_gb() if self._slice is not None else None

    def _slice_cap_gb(self) -> float | None:
        if self._slice is None:
            return None
        return self._slice.max_gb()

    # -- the decision ----------------------------------------------------
    def can_launch(self, job: Job, jobs: list[Job]) -> Decision:
        """Return whether *job* may start now given the state of *jobs*.

        ``jobs`` is the full job list (the controller filters for running ones
        itself). The checks run cheapest-first and short-circuit on the first
        failure.
        """
        cfg = self.config

        # 1. Manual pause sentinel.
        if cfg.pause_file and Path(cfg.pause_file).exists():
            return Decision(False, f"paused ({cfg.pause_file} present)")

        # 2. OOM cooldown.
        remaining = self.in_cooldown()
        if remaining > 0:
            return Decision(False, f"OOM cooldown ({int(remaining)}s remaining)")

        running = [j for j in jobs if j.status == JobStatus.RUNNING]

        # 3. Exclusive mutex (both directions).
        if any(j.exclusive for j in running):
            owner = next(j.id for j in running if j.exclusive)
            return Decision(False, f"exclusive job running ({owner})")
        if job.exclusive and running:
            return Decision(False, "job is exclusive but others are running")

        # 4. Concurrency cap.
        if len(running) >= cfg.max_concurrent:
            return Decision(
                False, f"concurrency cap reached ({len(running)}/{cfg.max_concurrent})"
            )

        # 5. Core budget.
        cores_used = sum(j.cores for j in running)
        if cores_used + job.cores > cfg.core_budget:
            free = cfg.core_budget - cores_used
            return Decision(
                False, f"core budget: need {job.cores}, {free} free of {cfg.core_budget}"
            )

        # 6. RAM guard -- cgroup truth first, declarative fallback otherwise.
        ram_decision = self._check_ram(job, running)
        if not ram_decision.ok:
            return ram_decision

        # 7. maxcore sanity (advisory; never blocks, surfaced in reason text).
        if not maxcore_sane(job):
            # Allowed through, but the reason notes the smell so the caller can
            # log it. The scheduler treats ok=True as launchable.
            return Decision(
                True,
                f"ok (warning: ram_gb={job.ram_gb:g} < maxcore*nprocs*1.3 "
                f"= {job.maxcore * job.effective_nprocs() * 1.3 / 1024:.1f}GB)",
            )

        return Decision(True, "ok")

    def _check_ram(self, job: Job, running: list[Job]) -> Decision:
        """RAM admission: cgroup ``memory.current`` if live, else declarative."""
        cfg = self.config
        used = self._slice_used_gb()
        if used is not None:
            cap = self._slice_cap_gb()
            if cap is None:
                cap = cfg.ram_budget_gb  # slice exists but no Max set
            safe_cap = cap - cfg.ram_safety_gb
            projected = used + job.ram_gb
            if projected > safe_cap:
                return Decision(
                    False,
                    f"slice RAM: {used:.0f}GB used + {job.ram_gb:g}GB > "
                    f"{safe_cap:.0f}GB safe cap (hard {cap:.0f}GB)",
                )
            return Decision(True, "ok")

        # Declarative fallback: no live cgroup accounting available.
        declared = sum(j.ram_gb for j in running)
        if declared + job.ram_gb > cfg.ram_budget_gb:
            free = cfg.ram_budget_gb - declared
            return Decision(
                False,
                f"declared RAM: need {job.ram_gb:g}GB, {free:.0f}GB of "
                f"{cfg.ram_budget_gb:.0f}GB free",
            )
        # Defense in depth: also respect live free memory so a job whose
        # neighbours under-declared cannot push the host into swap/OOM.
        free_now = _free_ram_gb()
        if free_now is not None and free_now < job.ram_gb + cfg.ram_safety_gb:
            return Decision(
                False,
                f"live RAM tight: {free_now:.0f}GB free, need "
                f"{job.ram_gb:g}+{cfg.ram_safety_gb:g}GB",
            )
        return Decision(True, "ok")
