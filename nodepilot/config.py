"""Configuration model and YAML queue parsing for nodepilot.

This module defines the two data classes that the rest of nodepilot revolves
around:

* :class:`Job` -- a single unit of work declared in a YAML queue (a command
  to run, plus its resource request and dependencies).
* :class:`Config` -- the global tunables (resource budgets, paths, NUMA
  topology) that govern admission and placement.

Both are plain ``@dataclass`` types with explicit types and sane defaults so a
queue file can be as small as ``id`` + ``command`` per job. Everything is
loaded from a single YAML document of the shape::

    global:
      max_concurrent: 4
      ram_budget_gb: 200
      state_path: ./nodepilot_state.json
    jobs:
      - id: hello
        command: "echo hi && sleep 5"
        cores: 4
        ram_gb: 8

The YAML schema is intentionally close to a job scheduler's: a ``global``
mapping plus a ``jobs`` list. A bare list of jobs is also accepted for quick
experiments.

Only the standard library and :mod:`yaml` (PyYAML) are used.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "Job",
    "Config",
    "load_queue",
    "JobStatus",
]


# Canonical job lifecycle states. Kept as plain strings (not an Enum) so they
# round-trip through JSON state files without custom encoders.
class JobStatus:
    """Namespace of the string constants a job's ``status`` field may hold."""

    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    #: Not run because a dependency failed (or was itself skipped, or does not
    #: exist). Terminal: the scheduler will never launch it. Without this state
    #: a job blocked by a failed dependency would sit ``PENDING`` forever and
    #: the run loop -- which exits only when nothing is pending or running --
    #: would never drain.
    SKIPPED = "skipped"
    #: Declared but not yet eligible: ignored by the scheduler until some
    #: external action (a hook, a manual edit) flips it to ``PENDING``.
    DORMANT = "dormant"

    #: States in which a job occupies resources and must be accounted for.
    ACTIVE = (RUNNING,)
    #: Terminal states.
    TERMINAL = (DONE, FAILED, SKIPPED)


@dataclass
class Job:
    """A single schedulable unit of work.

    The only required fields are :attr:`id` and :attr:`command`. Everything
    else has a default, so a minimal queue entry is two lines of YAML.

    Resource fields
    ---------------
    cores
        Number of *physical* cores the job needs. The NUMA allocator reserves a
        contiguous block of this many cores on one socket when possible.
    ram_gb
        Memory budget in GiB. Enforced as a hard cgroup ``MemoryMax`` on the
        job's scope (see :mod:`nodepilot.cgroups`) and used by admission
        control to avoid over-committing RAM.
    maxcore
        Optional per-process memory hint in MiB (mirrors the ``%maxcore``
        knob of several scientific codes). Used only by the admission sanity
        rule ``ram_gb >= maxcore * nprocs * 1.3``; never enforced.
    nprocs
        Number of MPI ranks the job will spawn. Defaults to ``cores`` (one rank
        per core). Drives both the ``maxcore`` sanity rule and the per-rank MPI
        binding plan in :mod:`nodepilot.mpi`.

    Scheduling fields
    -----------------
    depends_on
        Job ids that must reach ``done`` before this job is eligible.
    priority
        Lower runs first. Ties broken by id for determinism.
    exclusive
        If true, the job runs alone: it is admitted only when nothing else is
        running, and blocks every other launch while it runs. Use for memory-
        or bandwidth-bound phases that hate company.

    Runtime fields (filled in by the orchestrator; do not set by hand)
    ------------------------------------------------------------------
    status, session, cpu_list, numa_node, start_time, end_time,
    failure_reason, exit_code.
    """

    id: str
    command: str

    # --- resource request -------------------------------------------------
    cores: int = 1
    ram_gb: float = 4.0
    maxcore: int = 0
    nprocs: int = 0  # 0 -> defaults to ``cores`` via :meth:`effective_nprocs`

    # --- scheduling -------------------------------------------------------
    depends_on: list[str] = field(default_factory=list)
    priority: int = 100
    exclusive: bool = False

    # --- execution environment -------------------------------------------
    #: Working directory the command is run from. Defaults to CWD at launch.
    workdir: str = ""
    #: Extra environment variables merged into the child's environment.
    env: dict[str, str] = field(default_factory=dict)

    # --- runtime state (managed by the orchestrator) ---------------------
    status: str = JobStatus.PENDING
    session: str = ""
    cpu_list: str = ""
    numa_node: int = -1
    start_time: float = 0.0
    end_time: float = 0.0
    failure_reason: str = ""
    exit_code: int | None = None
    #: Free-form tags for downstream tooling (hooks, reporting). Never read by
    #: the core scheduler.
    metadata: dict[str, Any] = field(default_factory=dict)

    def effective_nprocs(self) -> int:
        """Number of MPI ranks, defaulting to one per core."""
        return self.nprocs if self.nprocs > 0 else self.cores

    def is_active(self) -> bool:
        """True while the job holds resources (currently: running)."""
        return self.status in JobStatus.ACTIVE

    def is_terminal(self) -> bool:
        """True once the job has finished (done or failed)."""
        return self.status in JobStatus.TERMINAL

    def runtime_hours(self) -> float:
        """Wall-clock hours between launch and finish (0 if not finished)."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time) / 3600.0
        return 0.0


@dataclass
class Config:
    """Global scheduler tunables.

    Defaults are conservative and host-agnostic: they are filled in from the
    live machine (physical core count, total RAM) when not given, then
    overridden by the ``global:`` block of the queue YAML.
    """

    # --- admission budgets ------------------------------------------------
    #: Hard cap on the number of simultaneously running jobs.
    max_concurrent: int = 4
    #: Total RAM (GiB) the scheduler may commit across running jobs. The cgroup
    #: containment cap (:attr:`memory_slice`) is the real OOM guard; this is the
    #: declarative pre-check.
    ram_budget_gb: float = 0.0  # 0 -> derived from host at load time
    #: Total physical cores the scheduler may hand out.
    core_budget: int = 0  # 0 -> derived from host at load time
    #: Safety margin (GiB) kept free below the cgroup slice cap so the kernel
    #: never has to oom-kill the slice pre-emptively.
    ram_safety_gb: float = 20.0

    # --- NUMA topology ----------------------------------------------------
    #: Mapping ``{node_index: "core-range"}`` describing which *physical* cores
    #: belong to each NUMA node, e.g. ``{0: "0-15", 1: "16-31"}``. Empty means
    #: "discover from the running machine" (see :func:`nodepilot.numa.detect`).
    numa_nodes: dict[int, str] = field(default_factory=dict)
    #: Cores never handed to jobs (e.g. SMT siblings, or cores reserved for the
    #: OS / a GPU lane). A core-range string like ``"96-191"``.
    reserved_cores: str = ""
    #: Spread a job's memory across all nodes (``--interleave=all``) once its
    #: ``ram_gb`` reaches this threshold, instead of binding to one node. Avoids
    #: saturating a single node with one large job. 0 disables interleaving.
    interleave_threshold_gb: float = 0.0

    # --- OOM safety -------------------------------------------------------
    #: systemd slice that contains every job scope. Its ``MemoryMax`` is the
    #: kernel-enforced ceiling and ``memory.current`` is the source of truth
    #: for committed RAM. Empty disables cgroup containment (declarative-only).
    memory_slice: str = "nodepilot.slice"
    #: ``oom_score_adj`` applied to the orchestrator itself (negative => protect
    #: from the OOM killer). Jobs get a positive score so they die first.
    orchestrator_oom_score_adj: int = -800
    #: ``oom_score_adj`` applied to each job's scope (positive => sacrifice
    #: first under memory pressure).
    job_oom_score_adj: int = 500
    #: Freeze new launches for this many seconds after a system OOM kill, to
    #: let memory actually free before retrying (anti-thrash).
    oom_cooldown_seconds: int = 300

    # --- execution --------------------------------------------------------
    #: How a job's command is executed: ``"subprocess"`` (default, detached
    #: child) or ``"tmux"`` (each job in its own tmux session for live attach).
    runner: str = "subprocess"
    #: Seconds between scheduler ticks.
    poll_interval: int = 10

    # --- paths ------------------------------------------------------------
    #: Where the JSON state snapshot is written (resume on restart).
    state_path: str = "nodepilot_state.json"
    #: Structured log file. Empty disables file logging (stderr only).
    log_path: str = "nodepilot.log"
    #: Touch this file to pause all new launches without stopping the daemon.
    pause_file: str = ".nodepilot.pause"

    def __post_init__(self) -> None:
        # Fill resource budgets from the live host when left at 0 so the
        # scheduler is usable out of the box on any machine.
        if self.core_budget <= 0:
            self.core_budget = _host_physical_cores()
        if self.ram_budget_gb <= 0:
            self.ram_budget_gb = round(_host_total_ram_gb() * 0.85, 1)

    # -- normalisation helpers ------------------------------------------
    @property
    def numa_nodes_resolved(self) -> dict[int, str]:
        """NUMA map with discovery applied when none was configured.

        Imported lazily to avoid a config<->numa import cycle.
        """
        if self.numa_nodes:
            return {int(k): str(v) for k, v in self.numa_nodes.items()}
        from nodepilot import numa  # local import breaks the cycle

        return numa.detect()


# ---------------------------------------------------------------------------
# Host probing helpers (best-effort; never raise)
# ---------------------------------------------------------------------------
def _host_physical_cores() -> int:
    """Best-effort physical (non-SMT) core count, falling back to logical."""
    try:
        # os.sched_getaffinity gives logical CPUs available to this process.
        logical = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        logical = os.cpu_count() or 1
    # Without a hardware probe we cannot tell SMT siblings apart, so assume the
    # affinity set is logical and halve only if it looks SMT-doubled is unsafe;
    # return logical and let the user pin reserved_cores for SMT exclusion.
    return logical


def _host_total_ram_gb() -> float:
    """Total system RAM in GiB read from ``/proc/meminfo`` (0 on failure)."""
    try:
        with open("/proc/meminfo", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return kb / (1024 * 1024)
    except (OSError, ValueError):
        pass
    return 0.0


# ---------------------------------------------------------------------------
# Queue loading
# ---------------------------------------------------------------------------
def _job_from_dict(data: dict[str, Any]) -> Job:
    """Build a :class:`Job` from a mapping, ignoring unknown keys.

    Tolerating unknown keys lets queue files carry annotations (``notes:``,
    custom tags) without breaking, and lets state files written by a newer
    version load in an older one.
    """
    known = {f.name for f in fields(Job)}
    filtered = {k: v for k, v in data.items() if k in known}
    if "id" not in filtered or "command" not in filtered:
        raise ValueError(
            f"job entry missing required 'id' and/or 'command': {data!r}"
        )
    return Job(**filtered)


def _config_from_global(block: dict[str, Any] | None) -> Config:
    """Build a :class:`Config` from a ``global:`` mapping (may be ``None``)."""
    block = block or {}
    known = {f.name for f in fields(Config)}
    filtered = {k: v for k, v in block.items() if k in known}
    return Config(**filtered)


def load_queue(path: str | os.PathLike[str]) -> tuple[Config, list[Job]]:
    """Parse a YAML queue file into a :class:`Config` and a list of :class:`Job`.

    Accepts either the structured form (a mapping with ``global`` and ``jobs``)
    or a bare list of job mappings (in which case defaults apply to the config).

    Parameters
    ----------
    path
        Filesystem path to the YAML queue.

    Returns
    -------
    (config, jobs)
        The resolved global config and the declared jobs, in file order.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the document is malformed or a job lacks ``id``/``command``.
    """
    text = Path(path).read_text(encoding="utf-8")
    doc = yaml.safe_load(text) or {}

    if isinstance(doc, list):
        jobs_raw: list[Any] = doc
        config = Config()
    elif isinstance(doc, dict):
        config = _config_from_global(doc.get("global"))
        jobs_raw = doc.get("jobs", [])
        if not isinstance(jobs_raw, list):
            raise ValueError("'jobs' must be a list")
    else:
        raise ValueError(
            f"queue must be a mapping or a list, got {type(doc).__name__}"
        )

    jobs = [_job_from_dict(j) for j in jobs_raw]
    _validate_unique_ids(jobs)
    return config, jobs


def _validate_unique_ids(jobs: list[Job]) -> None:
    seen: set[str] = set()
    for job in jobs:
        if job.id in seen:
            raise ValueError(f"duplicate job id in queue: {job.id!r}")
        seen.add(job.id)


def job_to_dict(job: Job) -> dict[str, Any]:
    """Serialise a :class:`Job` to a plain dict (for JSON state snapshots)."""
    return asdict(job)
