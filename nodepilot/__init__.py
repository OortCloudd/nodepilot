"""nodepilot -- a single-node, OOM-safe HPC job orchestrator.

nodepilot schedules a queue of compute jobs on one big machine *without* a
cluster scheduler. It is built for the case where SLURM is overkill but the OS
default scheduler is not enough: you have a workstation or a fat server, you
run ORCA / CP2K / VASP / Gaussian / PyTorch jobs back to back, and you keep
getting bitten by out-of-memory kills and NUMA-scattered threads.

What it gives you
-----------------
* **A YAML queue** of jobs with cores / RAM requests, dependencies, and
  priorities (:mod:`nodepilot.config`).
* **cgroups v2 memory containment** -- every job runs in a systemd scope with a
  hard ``MemoryMax`` under a parent slice whose ceiling sits below physical RAM,
  so an overshoot kills the *job*, not the host (:mod:`nodepilot.cgroups`).
* **NUMA-aware placement** -- each job gets a contiguous block of physical cores
  on one socket, pinned with ``numactl --physcpubind --membind``
  (:mod:`nodepilot.numa`).
* **Admission control** -- concurrency cap, core/RAM budgets, an exclusive
  mutex, a ``.pause`` sentinel, and a post-OOM cooldown
  (:mod:`nodepilot.admission`).
* **MPI binding helpers** -- stop launchers from scattering ranks across sockets
  and spilling memory cross-node (:mod:`nodepilot.mpi`).
* **A watchdog loop** with JSON state, crash-safe resume, and zombie
  reconciliation (:mod:`nodepilot.orchestrator`).

Quick start
-----------
>>> from nodepilot import Orchestrator
>>> orch = Orchestrator.from_queue("queue.yaml")
>>> orch.run()  # doctest: +SKIP

Or from the command line::

    nodepilot run queue.yaml
    nodepilot status
    nodepilot kill <job-id>

ORCA/CP2K/VASP/Gaussian/PyTorch are only *example* workloads; nodepilot itself
sees nothing but a shell command and a resource request.
"""

from __future__ import annotations

from nodepilot.admission import AdmissionController, Decision
from nodepilot.config import Config, Job, JobStatus, load_queue
from nodepilot.numa import Placement, allocate, detect, placement_prefix
from nodepilot.orchestrator import Orchestrator

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Orchestrator",
    "Config",
    "Job",
    "JobStatus",
    "load_queue",
    "AdmissionController",
    "Decision",
    "Placement",
    "allocate",
    "detect",
    "placement_prefix",
]
