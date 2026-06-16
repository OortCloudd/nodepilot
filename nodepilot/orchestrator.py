"""The scheduling loop: the watchdog that ties nodepilot together.

:class:`Orchestrator` owns the job list and runs a periodic *tick*. Each tick:

#. **Reap** -- for every running job whose process has exited, classify the
   outcome (done / failed / OOM) and, on a system OOM, arm the admission
   cooldown.
#. **Enforce pins** -- re-``taskset`` any job process that drifted off its
   assigned cores (MPI ranks escaping their cpuset).
#. **Launch** -- walk pending jobs in priority order; for each whose
   dependencies are satisfied and that admission control admits, find a
   NUMA-local core block, build the placed + cgroup-wrapped command, and start
   it.
#. **Persist** -- if anything changed, snapshot the state to JSON.

The loop exits when nothing is pending and nothing is running. On startup it
**reconciles zombies**: jobs left ``running`` in the state file by a crash but
with no live process/scope are marked failed so they free their resources and
do not block new launches.

This module is engine-agnostic. ORCA/CP2K/VASP/Gaussian/PyTorch are just
example *commands*; the orchestrator only ever sees a shell command string and
a resource request.
"""

from __future__ import annotations

import time
from pathlib import Path

from nodepilot import cgroups, numa, runner, state
from nodepilot.admission import AdmissionController
from nodepilot.config import Config, Job, JobStatus, load_queue
from nodepilot.logs import get_logger, kv

__all__ = ["Orchestrator"]


class Orchestrator:
    """Single-node job scheduler with cgroup containment and NUMA placement.

    Parameters
    ----------
    config
        Global tunables (see :class:`nodepilot.config.Config`).
    jobs
        The job list to schedule. Usually produced by :meth:`from_queue`.
    """

    def __init__(self, config: Config, jobs: list[Job]) -> None:
        self.config = config
        self.jobs = jobs
        self.log = get_logger(config.log_path)
        self.admission = AdmissionController(config)
        self.runner = runner.Runner(config)
        self._numa_nodes = config.numa_nodes_resolved
        self._reserved = numa.parse_cpu_list(config.reserved_cores)
        # SMT sibling threads, so the allocator can prefer distinct physical
        # cores; empty on hosts without SMT info (no effect there).
        self._smt_secondary = numa.detect_smt_secondary()

    # -- construction ----------------------------------------------------
    @classmethod
    def from_queue(cls, queue_path: str) -> "Orchestrator":
        """Build an orchestrator from a YAML queue, resuming prior state if any.

        If a state file (``config.state_path``) exists, the persisted job list
        is loaded *instead of* the YAML job list so progress is not lost; the
        YAML's ``global:`` block is still honoured for tunables. Delete the
        state file (or call :meth:`reset`) to start fresh from the YAML.
        """
        config, queue_jobs = load_queue(queue_path)
        if state.state_exists(config.state_path):
            jobs = state.load_state(config.state_path)
            _warn_if_queue_newer(queue_path, config.state_path, config)
        else:
            jobs = queue_jobs
        return cls(config, jobs)

    def reset(self) -> None:
        """Delete the persisted state file so the next run starts fresh."""
        p = Path(self.config.state_path)
        if p.exists():
            p.unlink()

    # -- the loop --------------------------------------------------------
    def run(self, *, max_ticks: int | None = None) -> None:
        """Run the scheduling loop until the queue drains.

        Parameters
        ----------
        max_ticks
            Stop after this many ticks regardless of queue state (used by tests
            and dry runs). ``None`` runs until completion.
        """
        cfg = self.config
        # Protect the scheduler itself from the OOM killer so it survives a
        # memory storm and can reap/sacrifice the jobs that caused it.
        cgroups.set_self_oom_score_adj(cfg.orchestrator_oom_score_adj)

        self._log_startup()
        self._reconcile_zombies()

        tick = 0
        while True:
            tick += 1
            changed = self._tick()
            if changed:
                state.save_state(cfg.state_path, self.jobs)

            if self._is_drained():
                self.log.info(
                    "queue complete %s",
                    kv(
                        done=self._count(JobStatus.DONE),
                        failed=self._count(JobStatus.FAILED),
                    ),
                )
                break
            if max_ticks is not None and tick >= max_ticks:
                break
            time.sleep(cfg.poll_interval)

    def _tick(self) -> bool:
        """Run one scheduling pass. Returns whether any state changed."""
        changed = False
        changed |= self._reap_finished()
        changed |= self._mark_dead_dependents()
        self._enforce_pins()  # side-effect only; not a state change to persist
        changed |= self._launch_pending()
        return changed

    # -- reaping ---------------------------------------------------------
    def _reap_finished(self) -> bool:
        changed = False
        for job in self.jobs:
            if job.status != JobStatus.RUNNING:
                continue
            if self.runner.is_alive(job):
                continue
            outcome = runner.reap(job, self.runner)
            job.status = outcome.status
            job.failure_reason = outcome.reason
            job.exit_code = outcome.exit_code
            job.end_time = time.time()
            changed = True
            if outcome.status == JobStatus.DONE:
                self.log.info(
                    "done %s", kv(job=job.id, hours=f"{job.runtime_hours():.2f}")
                )
            else:
                self.log.error(
                    "failed %s",
                    kv(
                        job=job.id,
                        reason=outcome.reason,
                        hours=f"{job.runtime_hours():.2f}",
                    ),
                )
                # An OOM kill means the host is under memory pressure: freeze
                # launches briefly so memory actually frees before we retry,
                # instead of relaunching straight into another OOM.
                if outcome.reason == "oom_killed":
                    self.admission.trigger_oom_cooldown()
                    self.log.warning(
                        "oom cooldown armed %s",
                        kv(seconds=self.config.oom_cooldown_seconds),
                    )
        return changed

    # -- pin enforcement -------------------------------------------------
    def _enforce_pins(self) -> None:
        for job in self.jobs:
            if job.status == JobStatus.RUNNING and job.cpu_list:
                try:
                    n = runner.enforce_pin(job)
                except OSError:
                    continue
                if n:
                    self.log.info(
                        "re-pinned drifted procs %s",
                        kv(job=job.id, count=n, cpu=job.cpu_list),
                    )

    # -- launching -------------------------------------------------------
    def _launch_pending(self) -> bool:
        changed = False
        pending = sorted(
            (j for j in self.jobs if j.status == JobStatus.PENDING),
            key=lambda j: (j.priority, j.id),
        )
        for job in pending:
            if not self._deps_satisfied(job):
                continue
            decision = self.admission.can_launch(job, self.jobs)
            if not decision.ok:
                continue  # waiting; reason available via decision.reason
            if self._launch(job):
                changed = True
        return changed

    def _launch(self, job: Job) -> bool:
        """Place and start one job. Returns ``True`` if it actually started."""
        placement = numa.allocate(
            job.cores,
            occupied=self._occupied_cores(),
            numa_nodes=self._numa_nodes,
            ram_gb=job.ram_gb,
            interleave_threshold_gb=self.config.interleave_threshold_gb,
            smt_secondary=self._smt_secondary,
        )
        if placement is None:
            # No NUMA-local block free right now; retry next tick.
            self.log.info(
                "defer %s", kv(job=job.id, why=f"no {job.cores}-core NUMA block free")
            )
            return False

        job.cpu_list = placement.cpu_list
        job.numa_node = placement.node
        if placement.smt_oversubscribed:
            # The job asked for more cores than the node has physical ones, so
            # some assigned ids are SMT siblings sharing a core -- not N
            # independent cores. Surface it rather than letting it pass silently.
            self.log.warning(
                "smt oversubscribed %s",
                kv(job=job.id, cores=job.cores, cpu=job.cpu_list, node=job.numa_node),
            )
        argv = runner.build_command(job, placement, self.config)
        try:
            self.runner.start(job, argv)
        except (OSError, RuntimeError) as exc:
            job.status = JobStatus.FAILED
            job.failure_reason = f"launch_error: {exc}"
            job.end_time = time.time()
            self.log.error("launch failed %s", kv(job=job.id, error=exc))
            return True  # state changed (job -> failed)

        job.status = JobStatus.RUNNING
        job.start_time = time.time()
        self.log.info(
            "launch %s",
            kv(
                job=job.id,
                cores=job.cores,
                ram_gb=f"{job.ram_gb:g}",
                cpu=job.cpu_list,
                node=job.numa_node,
                mem="interleave" if placement.interleave else f"membind:{job.numa_node}",
            ),
        )
        return True

    # -- helpers ---------------------------------------------------------
    def _occupied_cores(self) -> set[int]:
        """Cores currently held by running jobs, plus permanently reserved ones."""
        occ = set(self._reserved)
        for j in self.jobs:
            if j.status == JobStatus.RUNNING and j.cpu_list:
                occ |= numa.parse_cpu_list(j.cpu_list)
        return occ

    def _deps_satisfied(self, job: Job) -> bool:
        by_id = {j.id: j for j in self.jobs}
        for dep in job.depends_on:
            upstream = by_id.get(dep)
            if upstream is None or upstream.status != JobStatus.DONE:
                return False
        return True

    def _mark_dead_dependents(self) -> bool:
        """Skip pending jobs that can never run because a dependency failed.

        :meth:`_deps_satisfied` only advances a job once its dependencies are
        ``done``. A dependency that ends ``failed`` (or is itself ``skipped``,
        or does not exist) would otherwise leave the dependent ``pending``
        forever -- and since the run loop exits only when nothing is pending or
        running, the loop would spin without end. Marking such jobs terminal
        (``skipped``) lets the failure propagate and the queue drain. Iterates
        to a fixed point so an entire dependency chain collapses within one tick.
        """
        by_id = {j.id: j for j in self.jobs}
        dead = (JobStatus.FAILED, JobStatus.SKIPPED)
        changed = False
        while True:
            progressed = False
            for job in self.jobs:
                if job.status != JobStatus.PENDING:
                    continue
                blocker = None
                for dep in job.depends_on:
                    upstream = by_id.get(dep)
                    if upstream is None:
                        blocker = f"{dep} (unknown)"
                        break
                    if upstream.status in dead:
                        blocker = dep
                        break
                if blocker is not None:
                    job.status = JobStatus.SKIPPED
                    job.failure_reason = f"dependency_failed: {blocker}"
                    job.end_time = time.time()
                    changed = progressed = True
                    self.log.warning(
                        "skipped %s", kv(job=job.id, blocked_by=blocker)
                    )
            if not progressed:
                break
        return changed

    def _reconcile_zombies(self) -> None:
        """Mark crashed-but-``running`` jobs as failed at startup.

        A job is a zombie if the state file says ``running`` but neither its
        tracked process nor its cgroup scope is alive. Reconciling them frees
        their reserved cores/RAM and prevents them from blocking admission.
        """
        n = 0
        for job in self.jobs:
            if job.status != JobStatus.RUNNING:
                continue
            alive_proc = self.runner.is_alive(job)
            alive_scope = (
                cgroups.scope_is_active(job.id, self.config.memory_slice)
                if self.config.memory_slice
                else False
            )
            if not alive_proc and not alive_scope:
                job.status = JobStatus.FAILED
                job.failure_reason = "zombie_at_restart"
                job.end_time = time.time()
                # Release its core reservation so the allocator can reuse them.
                job.cpu_list = ""
                n += 1
                self.log.warning("zombie reconciled %s", kv(job=job.id))
        if n:
            state.save_state(self.config.state_path, self.jobs)

    def _is_drained(self) -> bool:
        return not any(
            j.status in (JobStatus.PENDING, JobStatus.RUNNING) for j in self.jobs
        )

    def _count(self, status: str) -> int:
        return sum(1 for j in self.jobs if j.status == status)

    def _log_startup(self) -> None:
        cfg = self.config
        monitor = cgroups.SliceMonitor(cfg.memory_slice) if cfg.memory_slice else None
        if monitor is not None and monitor.is_active():
            self.log.info(
                "slice active %s",
                kv(
                    slice=cfg.memory_slice,
                    used_gb=_fmt(monitor.used_gb()),
                    cap_gb=_fmt(monitor.max_gb()),
                ),
            )
        else:
            self.log.warning(
                "cgroup slice inactive %s",
                kv(slice=cfg.memory_slice, mode="declarative RAM accounting"),
            )
        self.log.info(
            "queue loaded %s",
            kv(
                jobs=len(self.jobs),
                pending=self._count(JobStatus.PENDING),
                done=self._count(JobStatus.DONE),
                cores=cfg.core_budget,
                ram_budget_gb=f"{cfg.ram_budget_gb:g}",
                max_concurrent=cfg.max_concurrent,
                nodes=len(self._numa_nodes),
            ),
        )

    # -- imperative controls (used by the CLI) ---------------------------
    def kill(self, job_id: str) -> bool:
        """Kill a running job by id. Returns ``True`` if it was running."""
        for job in self.jobs:
            if job.id == job_id and job.status == JobStatus.RUNNING:
                self.runner.kill(job)
                job.status = JobStatus.FAILED
                job.failure_reason = "killed_by_user"
                job.end_time = time.time()
                state.save_state(self.config.state_path, self.jobs)
                self.log.info("killed %s", kv(job=job_id))
                return True
        return False


def _fmt(value: float | None) -> str | None:
    return f"{value:.1f}" if value is not None else None


def _warn_if_queue_newer(queue_path: str, state_path: str, config: Config) -> None:
    """Warn when the YAML was edited after the last state save.

    Resuming loads jobs from the state file, so edits to the queue YAML are
    ignored until a ``reset``. This surfaces that gotcha rather than silently
    dropping the user's edits.
    """
    try:
        if Path(queue_path).stat().st_mtime > Path(state_path).stat().st_mtime:
            get_logger(config.log_path).warning(
                "queue edited after last state save %s",
                kv(queue=queue_path, hint="changes ignored until 'reset'"),
            )
    except OSError:
        pass
