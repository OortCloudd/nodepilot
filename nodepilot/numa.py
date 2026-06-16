"""NUMA topology discovery, core allocation, and placement command building.

Why this exists
---------------
On a multi-socket box, letting the kernel scatter a job's threads across every
CCD/socket is a performance disaster: a couple of complexes (CCDs) absorb all
the work, effective throughput collapses, and memory accesses go cross-socket.
The fix is to give each job a *contiguous block of physical cores on a single
NUMA node* and pin both its CPUs and its memory there with ``numactl``.

This module provides three things:

1. :func:`detect` -- read the machine's NUMA topology from sysfs.
2. :func:`allocate` -- find a free, preferably contiguous, NUMA-local block of
   N cores given what is already occupied.
3. :func:`placement_prefix` -- build the ``numactl --physcpubind ... --membind``
   (or ``--interleave=all``) command prefix for a placed job.

Plus small helpers to parse/format Linux CPU-list strings (``"0-3,8"``) and to
read per-node memory pressure from sysfs.

Everything degrades gracefully: on a machine without ``/sys`` NUMA nodes (a
laptop, a container), :func:`detect` returns a single synthetic node spanning
all available CPUs and the rest still works.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "parse_cpu_list",
    "format_cpu_list",
    "detect",
    "detect_smt_secondary",
    "node_memory_gb",
    "Placement",
    "allocate",
    "placement_prefix",
    "has_numactl",
]

_SYS_NODE = Path("/sys/devices/system/node")
_SYS_CPU = Path("/sys/devices/system/cpu")


# ---------------------------------------------------------------------------
# CPU-list string handling
# ---------------------------------------------------------------------------
def parse_cpu_list(spec: str) -> set[int]:
    """Parse a Linux CPU-list string into a set of ints.

    Accepts the kernel's standard format: comma-separated singletons and
    inclusive ``a-b`` ranges, e.g. ``"0-3,8,12-13"`` -> ``{0,1,2,3,8,12,13}``.
    An empty or whitespace-only string yields an empty set.

    Raises
    ------
    ValueError
        On a malformed range (non-integer bounds, or ``b < a``).
    """
    out: set[int] = set()
    if not spec or not spec.strip():
        return out
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            if hi < lo:
                raise ValueError(f"inverted CPU range: {part!r}")
            out.update(range(lo, hi + 1))
        else:
            out.add(int(part))
    return out


def format_cpu_list(cores: set[int] | list[int]) -> str:
    """Format a set/list of CPU ids into a compact CPU-list string.

    Consecutive runs collapse to ``a-b``; isolated ids stay singletons.
    ``{0,1,2,3,8}`` -> ``"0-3,8"``. Empty input yields ``""``.
    """
    ordered = sorted(set(cores))
    if not ordered:
        return ""
    parts: list[str] = []
    run_start = prev = ordered[0]
    for c in ordered[1:]:
        if c == prev + 1:
            prev = c
            continue
        parts.append(_run(run_start, prev))
        run_start = prev = c
    parts.append(_run(run_start, prev))
    return ",".join(parts)


def _run(lo: int, hi: int) -> str:
    return str(lo) if lo == hi else f"{lo}-{hi}"


# ---------------------------------------------------------------------------
# Topology discovery
# ---------------------------------------------------------------------------
def detect() -> dict[int, str]:
    """Discover NUMA nodes -> physical-core ranges from sysfs.

    Reads ``/sys/devices/system/node/node*/cpulist``. The returned mapping is
    ``{node_index: "cpu-list-string"}`` using each node's *online* CPUs.

    Falls back to a single synthetic node ``{0: <all online cpus>}`` when sysfs
    NUMA information is unavailable (containers, non-NUMA hardware), so callers
    never have to special-case "no NUMA".

    Note
    ----
    This reports *logical* CPUs as the kernel groups them per node; it does not
    attempt to strip SMT siblings. Exclude siblings via
    ``Config.reserved_cores`` if your workload should avoid them.
    """
    nodes: dict[int, str] = {}
    if _SYS_NODE.is_dir():
        for child in sorted(_SYS_NODE.glob("node[0-9]*")):
            name = child.name  # e.g. "node0"
            try:
                idx = int(name[4:])
            except ValueError:
                continue
            cpulist_file = child / "cpulist"
            try:
                spec = cpulist_file.read_text(encoding="ascii").strip()
            except OSError:
                continue
            if spec:
                nodes[idx] = spec
    if nodes:
        return nodes
    # Fallback: one node spanning every CPU this process may run on.
    try:
        cpus = os.sched_getaffinity(0)
    except (AttributeError, OSError):
        cpus = set(range(os.cpu_count() or 1))
    return {0: format_cpu_list(cpus)}


def detect_smt_secondary() -> set[int]:
    """Return the *secondary* SMT thread ids on this machine.

    For each physical core the kernel lists its hardware threads in
    ``/sys/devices/system/cpu/cpu<N>/topology/thread_siblings_list``. The lowest
    id in a sibling group is the *primary* (physical) thread; the rest are
    *secondary* (SMT) threads. :func:`allocate` prefers primaries so a job gets
    distinct physical cores until it genuinely needs more than the node has.

    Returns an empty set when sysfs is unavailable (containers) or the machine
    has no SMT, so callers degrade cleanly to "no SMT info".
    """
    secondary: set[int] = set()
    if not _SYS_CPU.is_dir():
        return secondary
    seen: set[int] = set()
    for child in _SYS_CPU.glob("cpu[0-9]*"):
        sib_file = child / "topology" / "thread_siblings_list"
        try:
            group = parse_cpu_list(sib_file.read_text(encoding="ascii").strip())
        except (OSError, ValueError):
            continue
        if not group:
            continue
        primary = min(group)
        if primary in seen:
            continue  # this sibling group was already accounted for
        seen |= group
        secondary |= group - {primary}
    return secondary


def node_memory_gb(node: int) -> tuple[float | None, float | None]:
    """Return ``(anon_used_gb, free_gb)`` for a NUMA node, or ``(None, None)``.

    ``anon_used`` is ``Active(anon) + Inactive(anon)`` from
    ``/sys/devices/system/node/nodeN/meminfo`` -- the *real* process memory on
    the node, excluding reclaimable page cache. Using anonymous pages (rather
    than ``MemUsed``) prevents a node that merely holds a lot of file cache from
    looking saturated and needlessly blocking placement there.
    """
    path = _SYS_NODE / f"node{node}" / "meminfo"
    try:
        active_anon = inactive_anon = free_kb = 0
        with path.open(encoding="ascii") as fh:
            for line in fh:
                # Format: "Node 0 MemFree:  12345 kB"
                parts = line.split()
                if len(parts) < 4:
                    continue
                key = parts[2].rstrip(":")
                val = int(parts[3])
                if key == "MemFree":
                    free_kb = val
                elif key == "Active(anon)":
                    active_anon = val
                elif key == "Inactive(anon)":
                    inactive_anon = val
        gib = 1024 * 1024
        return (active_anon + inactive_anon) / gib, free_kb / gib
    except (OSError, ValueError):
        return None, None


# ---------------------------------------------------------------------------
# Allocation
# ---------------------------------------------------------------------------
class Placement:
    """A resolved core/memory placement for one job.

    Attributes
    ----------
    cpu_list
        Compact CPU-list string of the chosen physical cores (``"0-15"``).
    node
        NUMA node the cores belong to.
    interleave
        If true, memory should be spread across all nodes
        (``--interleave=all``) rather than bound to :attr:`node`. Set when the
        job's RAM request crosses the interleave threshold, to avoid one big
        job saturating a single node.
    contiguous
        Whether the chosen cores form a single contiguous run (informational;
        a non-contiguous fallback is still a valid placement).
    smt_oversubscribed
        True when the job requested more cores than the node has *distinct
        physical* cores, so SMT sibling threads had to be included. The job then
        does not get N independent cores; the orchestrator surfaces this.
    """

    __slots__ = ("cpu_list", "node", "interleave", "contiguous", "smt_oversubscribed")

    def __init__(
        self,
        cpu_list: str,
        node: int,
        interleave: bool = False,
        contiguous: bool = True,
        smt_oversubscribed: bool = False,
    ) -> None:
        self.cpu_list = cpu_list
        self.node = node
        self.interleave = interleave
        self.contiguous = contiguous
        self.smt_oversubscribed = smt_oversubscribed

    def cores(self) -> set[int]:
        """The set of physical core ids in this placement."""
        return parse_cpu_list(self.cpu_list)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        mem = "interleave=all" if self.interleave else f"membind={self.node}"
        return f"Placement(cpu={self.cpu_list!r}, {mem}, contig={self.contiguous})"


def _largest_contiguous_run(free_sorted: list[int], n: int) -> int | None:
    """Return the start of the first contiguous run of length >= ``n``."""
    i = 0
    length = len(free_sorted)
    while i < length:
        j = i
        while j + 1 < length and free_sorted[j + 1] == free_sorted[j] + 1:
            j += 1
        if j - i + 1 >= n:
            return free_sorted[i]
        i = j + 1
    return None


def allocate(
    n_cores: int,
    occupied: set[int],
    numa_nodes: dict[int, str],
    *,
    ram_gb: float = 0.0,
    interleave_threshold_gb: float = 0.0,
    node_ram_cap_gb: float = 0.0,
    node_ram_safety_gb: float = 0.0,
    smt_secondary: set[int] | None = None,
) -> Placement | None:
    """Find a NUMA-local block of ``n_cores`` free physical cores.

    Strategy
    --------
    For each node, take the cores not in ``occupied`` and prefer the largest
    contiguous run; if none is long enough, fall back to any ``n_cores`` free
    cores on that node (still NUMA-local, just not contiguous). Among candidate
    nodes, prefer a contiguous placement, then the node with the most free
    cores, then the node under least memory pressure.

    Optional per-node RAM gating (``node_ram_cap_gb`` > 0) skips a node whose
    current anonymous memory plus this job's ``ram_gb`` would exceed
    ``node_ram_cap_gb - node_ram_safety_gb`` -- a guard against the
    single-node-OOM failure mode where one node fills up while the other is
    free. Skipped when the job will interleave its memory anyway.

    Parameters
    ----------
    n_cores
        Number of physical cores required.
    occupied
        Cores already handed out (and any reserved cores), to be avoided.
    numa_nodes
        ``{node: "cpu-list"}`` topology (typically ``Config.numa_nodes_resolved``).
    ram_gb
        The job's memory request; only used to decide interleaving and gating.
    interleave_threshold_gb
        Spread memory across nodes once ``ram_gb`` reaches this (0 disables).
    node_ram_cap_gb, node_ram_safety_gb
        Per-node RAM gating bounds (0 disables gating).

    Returns
    -------
    Placement or None
        ``None`` when no node can satisfy the request right now (the caller
        should defer and retry on the next tick).
    """
    if n_cores <= 0:
        # Degenerate request: place on the first node with no CPU binding.
        first = next(iter(numa_nodes), 0)
        return Placement("", first, interleave=False, contiguous=True)

    will_interleave = (
        interleave_threshold_gb > 0 and ram_gb >= interleave_threshold_gb
    )
    gate_ram = node_ram_cap_gb > 0 and not will_interleave
    safe_cap = node_ram_cap_gb - node_ram_safety_gb
    secondary = smt_secondary or set()

    candidates: list[tuple[bool, int, float, int, list[int], bool]] = []
    # tuple = (not_contiguous, -free_count, node_used_gb, node, cores, oversub)
    for node, spec in numa_nodes.items():
        node_cores = sorted(parse_cpu_list(spec))
        free = [c for c in node_cores if c not in occupied]
        if len(free) < n_cores:
            continue

        used_gb = 0.0
        if gate_ram:
            anon, _ = node_memory_gb(node)
            if anon is not None:
                used_gb = anon
                if used_gb + ram_gb > safe_cap:
                    continue  # node would be over-committed; skip it

        # Draw from distinct *physical* cores (primary SMT threads) first, and
        # only fall back to sibling threads when the job needs more cores than
        # the node has physical ones -- otherwise a job could silently get two
        # threads of one core instead of two separate cores. With no SMT info
        # (``secondary`` empty) ``primaries == free`` and this is a no-op.
        primaries = [c for c in free if c not in secondary]
        siblings = [c for c in free if c in secondary]
        oversub = len(primaries) < n_cores

        if not oversub:
            start = _largest_contiguous_run(primaries, n_cores)
            if start is not None:
                chosen = list(range(start, start + n_cores))
                contiguous = True
            else:
                chosen = primaries[:n_cores]
                contiguous = False
        else:
            # More cores requested than physical cores on the node: take every
            # physical core, then fill the remainder with sibling threads.
            chosen = primaries + siblings[: n_cores - len(primaries)]
            contiguous = False
        candidates.append(
            (not contiguous, -len(free), used_gb, node, chosen, oversub)
        )

    if not candidates:
        return None

    candidates.sort(key=lambda t: (t[0], t[1], t[2], t[3]))
    not_contig, _neg_free, _used, node, chosen, oversub = candidates[0]
    return Placement(
        format_cpu_list(chosen),
        node,
        interleave=will_interleave,
        contiguous=not not_contig,
        smt_oversubscribed=oversub,
    )


# ---------------------------------------------------------------------------
# Command building
# ---------------------------------------------------------------------------
def has_numactl() -> bool:
    """Whether the ``numactl`` binary is on PATH."""
    from shutil import which

    return which("numactl") is not None


def placement_prefix(placement: Placement, *, prefer_numactl: bool = True) -> list[str]:
    """Build the command-prefix argv that pins a job to its placement.

    Returns a list suitable for prepending to the job's argv:

    * With ``numactl`` available (and ``prefer_numactl``)::

          numactl --physcpubind=<cpu_list> --membind=<node>
          # or, when placement.interleave is set:
          numactl --physcpubind=<cpu_list> --interleave=all

    * Otherwise falls back to ``taskset -c <cpu_list>`` (CPU pinning only; no
      memory binding). Returns ``[]`` when there is nothing to pin (empty
      cpu_list and no numactl).

    The caller is responsible for actually prepending this to the command and
    for running it (see :mod:`nodepilot.orchestrator`).
    """
    if not placement.cpu_list:
        return []
    if prefer_numactl and has_numactl():
        mem = (
            ["--interleave=all"]
            if placement.interleave
            else [f"--membind={placement.node}"]
        )
        return ["numactl", f"--physcpubind={placement.cpu_list}", *mem]
    # Fallback: taskset pins CPUs but cannot bind memory.
    from shutil import which

    if which("taskset"):
        return ["taskset", "-c", placement.cpu_list]
    return []
