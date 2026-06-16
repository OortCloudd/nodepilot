"""MPI rank binding helpers to stop cross-socket DRAM spill.

The problem (learned the hard way)
----------------------------------
When you pin a job with ``numactl --physcpubind --membind`` and then launch MPI
inside it, the MPI launcher's *own* affinity logic can override the parent
cpuset: ranks get scattered across sockets and, worse, their first-touch memory
pages land on a remote node. Once allocated, those pages stay remote even if a
watchdog later re-pins the threads -- you eat a 1.5-1.8x latency penalty for the
life of the job, and you risk a single-node OOM while the other node sits free.

The cure has two parts:

1. **Tell the launcher to keep its hands off.** Disable the launcher's binding
   and map ranks by NUMA node so memory stays local. For Open MPI that is a set
   of ``OMPI_MCA_*`` environment variables (binding policy ``none``, mapping
   policy by node); the equivalent ``mpirun`` flags are ``--bind-to none
   --map-by node``. The outer ``numactl`` then owns all placement.

2. **Lay out ranks explicitly when you do want per-rank pinning.** Give rank *i*
   the cores ``base + i*stride .. +width``. :func:`rank_binding_plan` computes
   that layout from a job's core block.

This module is launcher-agnostic data + Open MPI conveniences; it builds env
dicts and argv fragments but never executes anything.
"""

from __future__ import annotations

from nodepilot.numa import format_cpu_list, parse_cpu_list

__all__ = [
    "RankBinding",
    "rank_binding_plan",
    "openmpi_no_bind_env",
    "openmpi_no_bind_flags",
    "openmpi_rankfile",
    "omp_thread_env",
]


class RankBinding:
    """The core assignment for a single MPI rank.

    Attributes
    ----------
    rank
        Rank index (0-based).
    cores
        Sorted list of physical core ids assigned to this rank.
    """

    __slots__ = ("rank", "cores")

    def __init__(self, rank: int, cores: list[int]) -> None:
        self.rank = rank
        self.cores = cores

    @property
    def cpu_list(self) -> str:
        """Compact CPU-list string for this rank's cores."""
        return format_cpu_list(self.cores)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"RankBinding(rank={self.rank}, cores={self.cpu_list!r})"


def rank_binding_plan(
    cpu_list: str,
    nprocs: int,
    *,
    threads_per_rank: int | None = None,
) -> list[RankBinding]:
    """Lay ``nprocs`` ranks across the cores in ``cpu_list``.

    Rank *i* gets the slice ``cores[i*stride : i*stride + width]`` where
    ``stride = floor(len(cores) / nprocs)`` and ``width = threads_per_rank or
    stride``. With the default (``threads_per_rank=None``) the cores are
    partitioned evenly and contiguously, e.g. 16 cores / 4 ranks ->
    rank0=0-3, rank1=4-7, rank2=8-11, rank3=12-15.

    Any leftover cores (when the count does not divide evenly) are appended to
    the last rank so no core is wasted.

    Parameters
    ----------
    cpu_list
        The job's whole physical-core block (``"0-15"``).
    nprocs
        Number of MPI ranks to place. Must be >= 1.
    threads_per_rank
        Cores to assign per rank (for hybrid MPI+OpenMP). Defaults to the even
        stride. If larger than the stride, ranks' core ranges may overlap --
        which the caller may want for oversubscribed OpenMP, but is usually a
        mistake; the function does not forbid it.

    Returns
    -------
    list[RankBinding]
        One entry per rank, in rank order.

    Raises
    ------
    ValueError
        If ``nprocs < 1`` or there are fewer cores than ranks.
    """
    if nprocs < 1:
        raise ValueError("nprocs must be >= 1")
    cores = sorted(parse_cpu_list(cpu_list))
    if len(cores) < nprocs:
        raise ValueError(
            f"cannot place {nprocs} ranks on {len(cores)} cores "
            f"({cpu_list!r}): need at least one core per rank"
        )
    stride = len(cores) // nprocs
    width = threads_per_rank if threads_per_rank is not None else stride
    width = max(1, width)

    plan: list[RankBinding] = []
    for i in range(nprocs):
        start = i * stride
        if i == nprocs - 1:
            # Last rank absorbs any remainder cores.
            chunk = cores[start:]
        else:
            chunk = cores[start : start + width]
        plan.append(RankBinding(i, chunk))
    return plan


def openmpi_no_bind_env() -> dict[str, str]:
    """Open MPI environment that defers all binding to the outer ``numactl``.

    Sets the MCA parameters that make ``mpirun`` *not* bind ranks and map them
    by NUMA node, so first-touch pages stay local and the parent cpuset/membind
    is respected:

    * ``OMPI_MCA_hwloc_base_binding_policy=none`` -- do not bind ranks.
    * ``OMPI_MCA_rmaps_base_mapping_policy=node`` -- distribute ranks by node.
    * ``OMPI_MCA_rmaps_base_ranking_policy=core`` -- rank within a node by core.

    Merge the result into the job's environment before launch.
    """
    return {
        "OMPI_MCA_hwloc_base_binding_policy": "none",
        "OMPI_MCA_rmaps_base_mapping_policy": "node",
        "OMPI_MCA_rmaps_base_ranking_policy": "core",
    }


def openmpi_no_bind_flags() -> list[str]:
    """``mpirun`` flags equivalent to :func:`openmpi_no_bind_env`.

    Returns ``["--bind-to", "none", "--map-by", "node"]`` for callers that
    prefer command-line flags over MCA environment variables.
    """
    return ["--bind-to", "none", "--map-by", "node"]


def openmpi_rankfile(plan: list[RankBinding]) -> str:
    """Render an Open MPI rankfile from a :func:`rank_binding_plan` result.

    Each line maps a rank to a slot list, e.g.::

        rank 0=localhost slot=0-3
        rank 1=localhost slot=4-7

    Use with ``mpirun --rankfile <path>`` when you want explicit per-rank
    pinning instead of the hands-off ``--bind-to none`` approach.
    """
    lines = [
        f"rank {b.rank}=localhost slot={b.cpu_list}"
        for b in plan
        if b.cores
    ]
    return "\n".join(lines) + ("\n" if lines else "")


def omp_thread_env(threads: int) -> dict[str, str]:
    """OpenMP thread-pinning environment for hybrid jobs.

    Sets ``OMP_NUM_THREADS`` and the standard close/cores binding so OpenMP
    threads stay packed on their rank's cores rather than wandering.
    """
    return {
        "OMP_NUM_THREADS": str(max(1, threads)),
        "OMP_PROC_BIND": "close",
        "OMP_PLACES": "cores",
    }
