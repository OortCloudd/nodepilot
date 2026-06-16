"""Unit tests for :mod:`nodepilot.numa`.

Covers the three responsibilities of the NUMA module:

* CPU-list parsing/formatting (``parse_cpu_list`` / ``format_cpu_list``),
  including round-tripping and inverted-range rejection.
* Core allocation (``allocate``): contiguous-preferred placement, occupancy
  forcing a job onto the other node, the non-contiguous fallback, the no-fit
  case, the interleave flag tripping at its threshold, and the per-node RAM
  gate skipping a saturated node.
* Placement command building (``placement_prefix``): the ``numactl`` form with
  ``--membind`` vs ``--interleave=all`` and the ``taskset`` fallback.

Plus a sanity check that topology discovery (``detect``) never returns an empty
mapping. All tests are hermetic: no network, no root, no real cluster. Sysfs and
PATH probing are monkeypatched so the suite behaves identically on a laptop, a
container, or a 192-thread server.
"""

from __future__ import annotations

import pytest

from nodepilot import numa
from nodepilot.numa import (
    Placement,
    allocate,
    detect,
    format_cpu_list,
    parse_cpu_list,
    placement_prefix,
)

# A fixed, synthetic two-node topology used by most allocation tests. Keeping it
# small and explicit means the assertions never depend on the host's real CPU
# layout.
TWO_NODES = {0: "0-15", 1: "16-31"}


# ---------------------------------------------------------------------------
# parse_cpu_list / format_cpu_list
# ---------------------------------------------------------------------------
class TestCpuListParsing:
    def test_parse_mixed_singletons_and_ranges(self) -> None:
        assert parse_cpu_list("0-3,8,12-13") == {0, 1, 2, 3, 8, 12, 13}

    def test_parse_single_value(self) -> None:
        assert parse_cpu_list("7") == {7}

    def test_parse_single_range(self) -> None:
        assert parse_cpu_list("4-6") == {4, 5, 6}

    def test_parse_collapsed_range_endpoints_are_inclusive(self) -> None:
        # "5-5" is a degenerate but legal range covering exactly {5}.
        assert parse_cpu_list("5-5") == {5}

    @pytest.mark.parametrize("spec", ["", "   ", "\t"])
    def test_parse_empty_or_blank_yields_empty_set(self, spec: str) -> None:
        assert parse_cpu_list(spec) == set()

    def test_parse_tolerates_whitespace_around_parts(self) -> None:
        assert parse_cpu_list(" 0 , 2-3 , 5 ") == {0, 2, 3, 5}

    def test_parse_skips_empty_fields(self) -> None:
        # Trailing/empty comma fields are ignored rather than raising.
        assert parse_cpu_list("0,,2,") == {0, 2}

    def test_parse_inverted_range_raises_valueerror(self) -> None:
        with pytest.raises(ValueError):
            parse_cpu_list("8-4")

    def test_parse_non_integer_bound_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_cpu_list("a-b")

    def test_format_collapses_runs(self) -> None:
        assert format_cpu_list({0, 1, 2, 3, 8}) == "0-3,8"

    def test_format_isolated_ids_stay_singletons(self) -> None:
        assert format_cpu_list({0, 2, 4}) == "0,2,4"

    def test_format_accepts_list_and_deduplicates(self) -> None:
        assert format_cpu_list([3, 1, 2, 2, 0]) == "0-3"

    def test_format_empty_yields_empty_string(self) -> None:
        assert format_cpu_list(set()) == ""
        assert format_cpu_list([]) == ""

    def test_roundtrip_parse_format_parse(self) -> None:
        spec = "0-3,8,12-13"
        cores = parse_cpu_list(spec)
        # format then re-parse must reproduce the same id set...
        assert parse_cpu_list(format_cpu_list(cores)) == cores
        # ...and on this already-canonical input, the string round-trips too.
        assert format_cpu_list(cores) == spec

    @pytest.mark.parametrize(
        "spec",
        ["0", "0-31", "0-3,8,12-13", "1,3,5,7", "16-19,48-51"],
    )
    def test_roundtrip_canonical_specs(self, spec: str) -> None:
        assert format_cpu_list(parse_cpu_list(spec)) == spec


# ---------------------------------------------------------------------------
# allocate
# ---------------------------------------------------------------------------
class TestAllocate:
    def test_contiguous_block_preferred(self) -> None:
        placed = allocate(4, set(), TWO_NODES)
        assert placed is not None
        assert placed.contiguous is True
        assert placed.cores() == {0, 1, 2, 3}
        # Lowest node index wins when nodes are otherwise equivalent.
        assert placed.node == 0
        assert placed.interleave is False

    def test_zero_cores_is_unpinned_placement(self) -> None:
        # A 0-core request is degenerate: a placement with no CPU binding on the
        # first node, so the orchestrator can still run the command unpinned.
        placed = allocate(0, set(), TWO_NODES)
        assert placed is not None
        assert placed.cpu_list == ""
        assert placed.cores() == set()
        assert placed.node == 0

    def test_occupancy_forces_other_node(self) -> None:
        # Node 0 is fully occupied; the only fit is node 1.
        occupied = parse_cpu_list("0-15")
        placed = allocate(4, occupied, TWO_NODES)
        assert placed is not None
        assert placed.node == 1
        assert placed.cores() <= parse_cpu_list("16-31")
        assert placed.cores().isdisjoint(occupied)

    def test_allocation_never_overlaps_occupied(self) -> None:
        occupied = {0, 1, 2}
        placed = allocate(4, occupied, TWO_NODES)
        assert placed is not None
        assert placed.cores().isdisjoint(occupied)
        assert len(placed.cores()) == 4

    def test_non_contiguous_fallback(self) -> None:
        # Checkerboard node 0 (every odd core taken) and fully occupy node 1, so
        # the only way to seat 4 cores is a non-contiguous block on node 0.
        occupied = {1, 3, 5, 7, 9, 11, 13, 15} | parse_cpu_list("16-31")
        placed = allocate(4, occupied, TWO_NODES)
        assert placed is not None
        assert placed.node == 0
        assert placed.contiguous is False
        assert len(placed.cores()) == 4
        assert placed.cores() <= {0, 2, 4, 6, 8, 10, 12, 14}

    def test_no_fit_returns_none(self) -> None:
        # More cores than any single node can offer -> no placement.
        assert allocate(40, set(), TWO_NODES) is None

    def test_no_fit_when_everything_occupied(self) -> None:
        occupied = parse_cpu_list("0-31")
        assert allocate(1, occupied, TWO_NODES) is None

    def test_prefers_contiguous_node_over_fragmented_one(self) -> None:
        # Node 0 is fragmented (can only seat 4 non-contiguously); node 1 is
        # pristine. A contiguous placement must win, i.e. land on node 1.
        occupied = {1, 3, 5, 7, 9, 11, 13, 15}
        placed = allocate(4, occupied, TWO_NODES)
        assert placed is not None
        assert placed.node == 1
        assert placed.contiguous is True

    def test_interleave_flag_set_at_threshold(self) -> None:
        placed = allocate(
            4, set(), TWO_NODES, ram_gb=64, interleave_threshold_gb=64
        )
        assert placed is not None
        # ram_gb >= threshold -> spread memory across nodes.
        assert placed.interleave is True

    def test_interleave_flag_clear_below_threshold(self) -> None:
        placed = allocate(
            4, set(), TWO_NODES, ram_gb=63.9, interleave_threshold_gb=64
        )
        assert placed is not None
        assert placed.interleave is False

    def test_interleave_disabled_when_threshold_zero(self) -> None:
        placed = allocate(
            4, set(), TWO_NODES, ram_gb=10_000, interleave_threshold_gb=0
        )
        assert placed is not None
        assert placed.interleave is False

    def test_per_node_ram_gate_skips_saturated_node(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Node 0 reports nearly-full anonymous memory; node 1 is empty. With a
        # per-node cap of 120 GiB and 20 GiB safety (safe cap = 100 GiB), a
        # 10 GiB job would push node 0 to 110 GiB and must be sent to node 1.
        def fake_mem(node: int) -> tuple[float, float]:
            return (100.0, 10.0) if node == 0 else (0.0, 200.0)

        monkeypatch.setattr(numa, "node_memory_gb", fake_mem)
        placed = allocate(
            4,
            set(),
            TWO_NODES,
            ram_gb=10,
            node_ram_cap_gb=120,
            node_ram_safety_gb=20,
        )
        assert placed is not None
        assert placed.node == 1

    def test_per_node_ram_gate_can_exhaust_all_nodes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Every node saturated -> the gate refuses all of them -> no placement.
        monkeypatch.setattr(numa, "node_memory_gb", lambda node: (100.0, 1.0))
        placed = allocate(
            4,
            set(),
            TWO_NODES,
            ram_gb=10,
            node_ram_cap_gb=120,
            node_ram_safety_gb=20,
        )
        assert placed is None

    def test_ram_gate_bypassed_when_interleaving(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # When the job will interleave its memory across all nodes, the
        # per-node cap is moot: placement must succeed even though both nodes
        # look saturated to the (here, deliberately exploding) gate probe.
        def boom(node: int) -> tuple[float, float]:  # pragma: no cover
            raise AssertionError("node_memory_gb must not be consulted")

        monkeypatch.setattr(numa, "node_memory_gb", boom)
        placed = allocate(
            4,
            set(),
            TWO_NODES,
            ram_gb=200,
            interleave_threshold_gb=64,
            node_ram_cap_gb=120,
            node_ram_safety_gb=20,
        )
        assert placed is not None
        assert placed.interleave is True

    def test_ram_gate_disabled_when_cap_zero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # node_ram_cap_gb == 0 disables gating entirely; the memory probe must
        # not even be called.
        def boom(node: int) -> tuple[float, float]:  # pragma: no cover
            raise AssertionError("gating disabled; probe must not run")

        monkeypatch.setattr(numa, "node_memory_gb", boom)
        placed = allocate(4, set(), TWO_NODES, ram_gb=10, node_ram_cap_gb=0)
        assert placed is not None
        assert placed.node == 0


# ---------------------------------------------------------------------------
# placement_prefix
# ---------------------------------------------------------------------------
class TestPlacementPrefix:
    def test_numactl_membind_form(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(numa, "has_numactl", lambda: True)
        prefix = placement_prefix(Placement("0-3", 1, interleave=False))
        assert prefix == [
            "numactl",
            "--physcpubind=0-3",
            "--membind=1",
        ]

    def test_numactl_interleave_form(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(numa, "has_numactl", lambda: True)
        prefix = placement_prefix(Placement("0-3", 1, interleave=True))
        assert prefix == [
            "numactl",
            "--physcpubind=0-3",
            "--interleave=all",
        ]

    def test_empty_cpu_list_yields_no_prefix(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Nothing to pin -> empty prefix, regardless of numactl availability.
        monkeypatch.setattr(numa, "has_numactl", lambda: True)
        assert placement_prefix(Placement("", 0)) == []

    def test_taskset_fallback_when_numactl_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # No numactl, but taskset on PATH -> CPU-only pinning fallback.
        monkeypatch.setattr(numa, "has_numactl", lambda: False)
        monkeypatch.setattr(
            "shutil.which",
            lambda name: "/usr/bin/taskset" if name == "taskset" else None,
        )
        prefix = placement_prefix(Placement("0-3", 1))
        assert prefix == ["taskset", "-c", "0-3"]

    def test_no_tool_available_yields_empty_prefix(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Neither numactl nor taskset present -> run unpinned (empty prefix).
        monkeypatch.setattr(numa, "has_numactl", lambda: False)
        monkeypatch.setattr("shutil.which", lambda name: None)
        assert placement_prefix(Placement("0-3", 1)) == []

    def test_prefer_numactl_false_uses_taskset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Even if numactl exists, prefer_numactl=False forces the taskset path.
        monkeypatch.setattr(numa, "has_numactl", lambda: True)
        monkeypatch.setattr(
            "shutil.which",
            lambda name: "/usr/bin/taskset" if name == "taskset" else None,
        )
        prefix = placement_prefix(
            Placement("16-19", 1), prefer_numactl=False
        )
        assert prefix == ["taskset", "-c", "16-19"]


# ---------------------------------------------------------------------------
# detect
# ---------------------------------------------------------------------------
class TestDetect:
    def test_detect_returns_non_empty_mapping(self) -> None:
        nodes = detect()
        assert isinstance(nodes, dict)
        assert len(nodes) >= 1

    def test_detect_values_are_parseable_cpu_lists(self) -> None:
        # Every reported node must map to a non-empty, parseable CPU-list.
        for node, spec in detect().items():
            assert isinstance(node, int)
            assert isinstance(spec, str)
            cores = parse_cpu_list(spec)
            assert cores, f"node {node} has no cores: {spec!r}"

    def test_detect_falls_back_to_single_node_without_sysfs(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        # Point the sysfs root at an empty directory so no node*/cpulist exists;
        # detect() must still synthesize a single node spanning the affinity set
        # rather than returning an empty mapping.
        empty_sysfs = tmp_path / "no_numa_here"
        empty_sysfs.mkdir()
        monkeypatch.setattr(numa, "_SYS_NODE", empty_sysfs)
        monkeypatch.setattr(numa.os, "sched_getaffinity", lambda pid: {0, 1, 2, 3})
        nodes = detect()
        assert nodes == {0: "0-3"}

    def test_detect_fallback_handles_missing_affinity(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        # Some platforms lack sched_getaffinity; the fallback must still produce
        # a single node (driven by os.cpu_count) and never raise or return {}.
        empty_sysfs = tmp_path / "void"
        empty_sysfs.mkdir()
        monkeypatch.setattr(numa, "_SYS_NODE", empty_sysfs)

        def no_affinity(pid: int):
            raise OSError("sched_getaffinity unavailable")

        monkeypatch.setattr(numa.os, "sched_getaffinity", no_affinity)
        monkeypatch.setattr(numa.os, "cpu_count", lambda: 2)
        nodes = detect()
        assert len(nodes) == 1
        assert parse_cpu_list(nodes[0]) == {0, 1}


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
