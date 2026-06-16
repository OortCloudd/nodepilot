"""Unit tests for :mod:`nodepilot.mpi`.

The MPI module is *pure data*: every function here builds a dict, a string, or
a list of :class:`~nodepilot.mpi.RankBinding` objects and executes nothing. No
launcher is spawned, no environment is mutated, no file is written. That makes
these tests fully hermetic -- no MPI install, no cluster, no root.

Coverage:

* :func:`rank_binding_plan` -- the contiguous even partition, the
  last-rank-absorbs-the-remainder rule when cores do not divide by ranks, the
  ``threads_per_rank`` width override, the single-rank degenerate case, and the
  two ``ValueError`` guards (``nprocs < 1`` and fewer cores than ranks).
* :func:`openmpi_no_bind_env` / :func:`openmpi_no_bind_flags` -- the exact MCA
  env dict and the equivalent ``mpirun`` flag list.
* :func:`openmpi_rankfile` -- the ``rank N=localhost slot=...`` text format,
  including the trailing newline and the empty-plan case.
* :func:`omp_thread_env` -- the OpenMP pinning env, including the ``threads`` <= 0
  clamp to 1.

Assertions pin exact expected outputs rather than loose shapes, so a behaviour
change in the layout maths or the rendered text will surface here.
"""

from __future__ import annotations

import pytest

from nodepilot.mpi import (
    RankBinding,
    omp_thread_env,
    openmpi_no_bind_env,
    openmpi_no_bind_flags,
    openmpi_rankfile,
    rank_binding_plan,
)


def _layout(plan: list[RankBinding]) -> list[tuple[int, list[int]]]:
    """Flatten a plan into ``(rank, cores)`` tuples for compact assertions."""
    return [(b.rank, b.cores) for b in plan]


# ---------------------------------------------------------------------------
# rank_binding_plan
# ---------------------------------------------------------------------------
class TestRankBindingPlan:
    def test_even_partition_contiguous_slices(self) -> None:
        # 16 cores / 4 ranks -> four contiguous 4-core slices, in rank order.
        plan = rank_binding_plan("0-15", 4)
        assert _layout(plan) == [
            (0, [0, 1, 2, 3]),
            (1, [4, 5, 6, 7]),
            (2, [8, 9, 10, 11]),
            (3, [12, 13, 14, 15]),
        ]
        # Slices are contiguous and disjoint, covering every core exactly once.
        covered = [c for b in plan for c in b.cores]
        assert covered == list(range(16))

    def test_per_rank_cpu_list_is_compact(self) -> None:
        # The RankBinding.cpu_list property collapses each slice into a range.
        plan = rank_binding_plan("0-15", 4)
        assert [b.cpu_list for b in plan] == ["0-3", "4-7", "8-11", "12-15"]

    def test_last_rank_absorbs_remainder_when_not_divisible(self) -> None:
        # 10 cores / 3 ranks: stride = 10 // 3 = 3, so ranks 0 and 1 get three
        # cores each and the last rank swallows the leftover (cores[6:]).
        plan = rank_binding_plan("0-9", 3)
        assert _layout(plan) == [
            (0, [0, 1, 2]),
            (1, [3, 4, 5]),
            (2, [6, 7, 8, 9]),  # remainder {9} appended to the last rank
        ]
        # No core is wasted: the union is the whole block.
        covered = sorted(c for b in plan for c in b.cores)
        assert covered == list(range(10))

    def test_single_rank_takes_all_cores(self) -> None:
        # nprocs == 1: rank 0 is the last rank, so it gets cores[0:] = everything.
        plan = rank_binding_plan("0-7", 1)
        assert _layout(plan) == [(0, [0, 1, 2, 3, 4, 5, 6, 7])]

    def test_one_core_per_rank_exact_fit(self) -> None:
        # cores == ranks: stride 1, one core each, last rank has no remainder.
        plan = rank_binding_plan("0-3", 4)
        assert _layout(plan) == [(0, [0]), (1, [1]), (2, [2]), (3, [3])]

    def test_non_contiguous_input_block_is_partitioned_in_sorted_order(
        self,
    ) -> None:
        # A fragmented block is sorted first, then sliced; the last rank still
        # absorbs the remainder.
        plan = rank_binding_plan("0,2,4,6,8", 2)
        # 5 cores / 2 ranks -> stride 2: rank0 = {0,2}, rank1 = {4,6,8}.
        assert _layout(plan) == [(0, [0, 2]), (1, [4, 6, 8])]

    def test_threads_per_rank_overrides_width_for_non_last_ranks(self) -> None:
        # 16 cores / 2 ranks: stride 8, but width pinned to 2. Non-last ranks
        # take exactly `width` cores at their stride offset; the last rank still
        # runs to the end of the block (cores[8:]).
        plan = rank_binding_plan("0-15", 2, threads_per_rank=2)
        assert _layout(plan) == [
            (0, [0, 1]),
            (1, [8, 9, 10, 11, 12, 13, 14, 15]),
        ]

    def test_threads_per_rank_zero_is_clamped_to_one(self) -> None:
        # width = max(1, 0) = 1, so each non-last rank gets a single core.
        plan = rank_binding_plan("0-7", 4, threads_per_rank=0)
        assert _layout(plan) == [
            (0, [0]),
            (1, [2]),  # stride is still 2 (8 // 4); offset = i * stride
            (2, [4]),
            (3, [6, 7]),  # last rank from its offset to the end
        ]

    def test_nprocs_below_one_raises(self) -> None:
        with pytest.raises(ValueError):
            rank_binding_plan("0-7", 0)

    def test_fewer_cores_than_ranks_raises(self) -> None:
        with pytest.raises(ValueError):
            rank_binding_plan("0-3", 8)


# ---------------------------------------------------------------------------
# openmpi_no_bind_env / openmpi_no_bind_flags
# ---------------------------------------------------------------------------
class TestOpenMpiNoBind:
    def test_env_is_exact_mca_mapping(self) -> None:
        assert openmpi_no_bind_env() == {
            "OMPI_MCA_hwloc_base_binding_policy": "none",
            "OMPI_MCA_rmaps_base_mapping_policy": "node",
            "OMPI_MCA_rmaps_base_ranking_policy": "core",
        }

    def test_env_values_are_all_strings(self) -> None:
        # The env is merged into os.environ-style dicts, so values must be str.
        env = openmpi_no_bind_env()
        assert all(isinstance(v, str) for v in env.values())

    def test_flags_are_exact_mpirun_equivalent(self) -> None:
        assert openmpi_no_bind_flags() == [
            "--bind-to",
            "none",
            "--map-by",
            "node",
        ]


# ---------------------------------------------------------------------------
# openmpi_rankfile
# ---------------------------------------------------------------------------
class TestOpenMpiRankfile:
    def test_renders_one_line_per_rank_with_trailing_newline(self) -> None:
        plan = rank_binding_plan("0-7", 2)
        text = openmpi_rankfile(plan)
        assert text == (
            "rank 0=localhost slot=0-3\n"
            "rank 1=localhost slot=4-7\n"
        )

    def test_uses_compact_cpu_list_per_rank(self) -> None:
        # A non-divisible plan exercises the remainder slot on the last line.
        plan = rank_binding_plan("0-9", 3)
        assert openmpi_rankfile(plan) == (
            "rank 0=localhost slot=0-2\n"
            "rank 1=localhost slot=3-5\n"
            "rank 2=localhost slot=6-9\n"
        )

    def test_empty_plan_yields_empty_string(self) -> None:
        assert openmpi_rankfile([]) == ""

    def test_ranks_with_no_cores_are_skipped(self) -> None:
        # A coreless rank contributes no line (and an all-empty plan -> "").
        plan = [RankBinding(0, [0, 1]), RankBinding(1, [])]
        assert openmpi_rankfile(plan) == "rank 0=localhost slot=0-1\n"


# ---------------------------------------------------------------------------
# omp_thread_env
# ---------------------------------------------------------------------------
class TestOmpThreadEnv:
    def test_sets_count_and_pinning(self) -> None:
        assert omp_thread_env(8) == {
            "OMP_NUM_THREADS": "8",
            "OMP_PROC_BIND": "close",
            "OMP_PLACES": "cores",
        }

    def test_single_thread(self) -> None:
        assert omp_thread_env(1)["OMP_NUM_THREADS"] == "1"

    @pytest.mark.parametrize("threads", [0, -1, -16])
    def test_non_positive_thread_count_clamped_to_one(self, threads: int) -> None:
        # max(1, threads) guards against a zero/negative OMP_NUM_THREADS.
        env = omp_thread_env(threads)
        assert env["OMP_NUM_THREADS"] == "1"

    def test_values_are_all_strings(self) -> None:
        env = omp_thread_env(4)
        assert all(isinstance(v, str) for v in env.values())


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
