"""Unit tests for the admission decision stack (:mod:`nodepilot.admission`).

Admission control is the gate that keeps a single fat node from over-committing
itself into an OOM cascade. It is a short stack of *simple* checks evaluated
cheapest-first; :func:`nodepilot.admission.AdmissionController.can_launch`
returns a :class:`~nodepilot.admission.Decision` (an ``(ok, reason)`` pair) so
the scheduler can log *why* a job is waiting.

These tests exercise each rule in isolation. To keep them deterministic and
hermetic they:

* force the **declarative RAM path** with ``Config(memory_slice="")`` so no
  live cgroup ``memory.current`` reading is involved (the slice monitor is
  never constructed);
* set **explicit** ``core_budget`` / ``ram_budget_gb`` / ``max_concurrent`` so
  the auto-from-host defaults in ``Config.__post_init__`` never leak the test
  machine's real topology into an assertion;
* point ``pause_file`` at a path *inside the test's ``tmp_path``* that does not
  exist, so a stray ``.nodepilot.pause`` in the working directory can never
  make an unrelated test fail;
* build neighbour jobs with ``status=JobStatus.RUNNING`` to simulate occupancy
  (admission counts only running jobs).

Every workload here is synthetic (``echo`` / ``sleep`` / ``true``); nothing is
launched -- ``can_launch`` is pure decision logic.
"""

from __future__ import annotations

import time

import pytest

from nodepilot.admission import (
    AdmissionController,
    Decision,
    maxcore_sane,
    running_cores,
    running_count,
    running_ram_gb,
)
from nodepilot.config import Config, Job, JobStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_config(tmp_path, **overrides) -> Config:
    """A fully *declarative* config: no cgroup slice, explicit budgets.

    ``memory_slice=""`` disables the live ``SliceMonitor`` so the RAM guard
    falls back to declared-RAM accounting (the path these tests target).
    ``pause_file`` defaults to a non-existent file under ``tmp_path`` so the
    pause sentinel is *off* unless a test deliberately creates it.

    Any keyword overrides the corresponding :class:`Config` field, e.g.
    ``make_config(tmp_path, max_concurrent=2)``.
    """
    params = dict(
        memory_slice="",  # force declarative RAM path (no SliceMonitor)
        core_budget=64,
        ram_budget_gb=200.0,
        ram_safety_gb=20.0,
        max_concurrent=8,
        oom_cooldown_seconds=300,
        # A path that does not exist -> pause sentinel inactive by default.
        pause_file=str(tmp_path / "absent.nodepilot.pause"),
    )
    params.update(overrides)
    return Config(**params)


def running_job(job_id: str, *, cores: int = 1, ram_gb: float = 1.0,
                exclusive: bool = False) -> Job:
    """A job already marked ``RUNNING`` (occupies cores/RAM in admission)."""
    return Job(
        id=job_id,
        command="sleep 1",
        cores=cores,
        ram_gb=ram_gb,
        exclusive=exclusive,
        status=JobStatus.RUNNING,
    )


def pending_job(job_id: str = "candidate", *, cores: int = 1, ram_gb: float = 1.0,
                exclusive: bool = False, maxcore: int = 0, nprocs: int = 0) -> Job:
    """A fresh candidate job (default ``pending``) to feed to ``can_launch``."""
    return Job(
        id=job_id,
        command="echo hi",
        cores=cores,
        ram_gb=ram_gb,
        exclusive=exclusive,
        maxcore=maxcore,
        nprocs=nprocs,
    )


# ---------------------------------------------------------------------------
# Decision dataclass: __bool__ must track .ok
# ---------------------------------------------------------------------------
class TestDecision:
    def test_bool_matches_ok_true(self):
        d = Decision(True, "ok")
        assert d.ok is True
        assert bool(d) is True
        assert d  # truthy in an ``if`` context

    def test_bool_matches_ok_false(self):
        d = Decision(False, "nope")
        assert d.ok is False
        assert bool(d) is False
        assert not d

    def test_reason_is_preserved(self):
        d = Decision(False, "concurrency cap reached (8/8)")
        assert d.reason == "concurrency cap reached (8/8)"

    @pytest.mark.parametrize("ok", [True, False])
    def test_bool_is_exactly_ok(self, ok):
        # __bool__ should mirror .ok for every value, not merely be truthy.
        assert bool(Decision(ok, "r")) == ok


# ---------------------------------------------------------------------------
# Concurrency cap
# ---------------------------------------------------------------------------
class TestConcurrencyCap:
    def test_allows_below_cap(self, tmp_path):
        cfg = make_config(tmp_path, max_concurrent=3)
        ac = AdmissionController(cfg)
        running = [running_job("a"), running_job("b")]  # 2 < 3
        decision = ac.can_launch(pending_job(), running)
        assert decision.ok
        assert decision.reason == "ok"

    def test_blocks_at_cap(self, tmp_path):
        cfg = make_config(tmp_path, max_concurrent=2)
        ac = AdmissionController(cfg)
        running = [running_job("a"), running_job("b")]  # 2 >= 2
        decision = ac.can_launch(pending_job(), running)
        assert not decision.ok
        assert "concurrency cap" in decision.reason
        assert "2/2" in decision.reason

    def test_only_running_jobs_count_toward_cap(self, tmp_path):
        # Pending / done / failed neighbours must not consume a concurrency slot.
        cfg = make_config(tmp_path, max_concurrent=1)
        ac = AdmissionController(cfg)
        neighbours = [
            Job(id="p", command="echo", status=JobStatus.PENDING),
            Job(id="d", command="echo", status=JobStatus.DONE),
            Job(id="f", command="echo", status=JobStatus.FAILED),
        ]
        decision = ac.can_launch(pending_job(), neighbours)
        assert decision.ok, decision.reason


# ---------------------------------------------------------------------------
# Exclusive mutex (both directions)
# ---------------------------------------------------------------------------
class TestExclusiveMutex:
    def test_running_exclusive_blocks_new_normal_job(self, tmp_path):
        # Direction 1: an exclusive job is running -> nothing may start beside it.
        cfg = make_config(tmp_path)
        ac = AdmissionController(cfg)
        running = [running_job("solo", exclusive=True)]
        decision = ac.can_launch(pending_job(), running)
        assert not decision.ok
        assert "exclusive" in decision.reason
        assert "solo" in decision.reason  # names the blocking owner

    def test_new_exclusive_blocked_by_any_running_job(self, tmp_path):
        # Direction 2: the candidate is exclusive but something is already running.
        cfg = make_config(tmp_path)
        ac = AdmissionController(cfg)
        running = [running_job("worker")]
        decision = ac.can_launch(pending_job(exclusive=True), running)
        assert not decision.ok
        assert "exclusive" in decision.reason

    def test_exclusive_admitted_on_empty_node(self, tmp_path):
        # An exclusive job *is* allowed when nothing else runs.
        cfg = make_config(tmp_path)
        ac = AdmissionController(cfg)
        decision = ac.can_launch(pending_job(exclusive=True), [])
        assert decision.ok, decision.reason

    def test_normal_job_admitted_when_no_exclusive_running(self, tmp_path):
        cfg = make_config(tmp_path)
        ac = AdmissionController(cfg)
        running = [running_job("a"), running_job("b")]
        decision = ac.can_launch(pending_job(), running)
        assert decision.ok, decision.reason


# ---------------------------------------------------------------------------
# Core budget
# ---------------------------------------------------------------------------
class TestCoreBudget:
    def test_blocks_when_cores_exceed_budget(self, tmp_path):
        # 12 cores running + 8 requested > 16 budget. Keep max_concurrent high so
        # the concurrency cap (checked earlier) does not pre-empt this rule.
        cfg = make_config(tmp_path, core_budget=16, max_concurrent=100)
        ac = AdmissionController(cfg)
        running = [running_job("big", cores=12)]
        decision = ac.can_launch(pending_job(cores=8), running)
        assert not decision.ok
        assert "core budget" in decision.reason
        assert "16" in decision.reason  # surfaces the budget

    def test_allows_when_cores_fit_exactly(self, tmp_path):
        # 8 running + 8 requested == 16 budget: fits (strict ``>`` is the gate).
        cfg = make_config(tmp_path, core_budget=16, max_concurrent=100)
        ac = AdmissionController(cfg)
        running = [running_job("half", cores=8)]
        decision = ac.can_launch(pending_job(cores=8), running)
        assert decision.ok, decision.reason

    def test_core_budget_counts_only_running(self, tmp_path):
        # A pending neighbour reserving many cores must not block the candidate.
        cfg = make_config(tmp_path, core_budget=16, max_concurrent=100)
        ac = AdmissionController(cfg)
        neighbours = [Job(id="pend", command="echo", cores=64,
                          status=JobStatus.PENDING)]
        decision = ac.can_launch(pending_job(cores=8), neighbours)
        assert decision.ok, decision.reason


# ---------------------------------------------------------------------------
# Declarative RAM guard (memory_slice="" path)
# ---------------------------------------------------------------------------
class TestDeclarativeRam:
    def test_blocks_when_declared_ram_over_budget(self, tmp_path):
        # 180 GB running + 40 GB requested > 200 GB budget. Generous core budget
        # and concurrency keep earlier rules from masking the RAM verdict.
        cfg = make_config(
            tmp_path, ram_budget_gb=200.0, core_budget=1000, max_concurrent=100
        )
        ac = AdmissionController(cfg)
        running = [running_job("hog", cores=1, ram_gb=180.0)]
        decision = ac.can_launch(pending_job(cores=1, ram_gb=40.0), running)
        assert not decision.ok
        assert "declared RAM" in decision.reason

    def test_allows_when_declared_ram_fits(self, tmp_path):
        # 80 GB running + 40 GB requested == 120 GB <= 200 GB budget. Well clear
        # of budget, so the (live ``/proc/meminfo``) defence-in-depth check is
        # the only thing that could intervene; on any sane CI box 40+20 GB is
        # available, but we keep the request modest to stay robust.
        cfg = make_config(
            tmp_path, ram_budget_gb=200.0, core_budget=1000, max_concurrent=100
        )
        ac = AdmissionController(cfg)
        running = [running_job("warm", cores=1, ram_gb=80.0)]
        decision = ac.can_launch(pending_job(cores=1, ram_gb=4.0), running)
        assert decision.ok, decision.reason

    def test_uses_declarative_path_when_slice_disabled(self, tmp_path):
        # Sanity: memory_slice="" means no SliceMonitor is constructed, so the
        # RAM guard is purely declarative (no cgroup read attempted).
        cfg = make_config(tmp_path)
        ac = AdmissionController(cfg)
        assert ac._slice is None


# ---------------------------------------------------------------------------
# Pause sentinel
# ---------------------------------------------------------------------------
class TestPauseSentinel:
    def test_pause_file_blocks_all_launches(self, tmp_path):
        pause = tmp_path / "paused.nodepilot.pause"
        pause.write_text("", encoding="ascii")  # presence is the signal
        cfg = make_config(tmp_path, pause_file=str(pause))
        ac = AdmissionController(cfg)
        decision = ac.can_launch(pending_job(), [])
        assert not decision.ok
        assert "paused" in decision.reason
        assert str(pause) in decision.reason

    def test_no_pause_file_allows_launch(self, tmp_path):
        # Default make_config points pause_file at a path that does not exist.
        cfg = make_config(tmp_path)
        ac = AdmissionController(cfg)
        decision = ac.can_launch(pending_job(), [])
        assert decision.ok, decision.reason

    def test_pause_takes_priority_over_other_rules(self, tmp_path):
        # Pause is the first check: it fires even when the node is otherwise full
        # (here also over the concurrency cap), proving precedence.
        pause = tmp_path / "brake.pause"
        pause.write_text("", encoding="ascii")
        cfg = make_config(tmp_path, pause_file=str(pause), max_concurrent=1)
        ac = AdmissionController(cfg)
        running = [running_job("a"), running_job("b")]  # already over cap
        decision = ac.can_launch(pending_job(), running)
        assert not decision.ok
        assert "paused" in decision.reason  # pause, not the cap, is reported


# ---------------------------------------------------------------------------
# OOM cooldown
# ---------------------------------------------------------------------------
class TestOomCooldown:
    def test_not_in_cooldown_initially(self, tmp_path):
        ac = AdmissionController(make_config(tmp_path))
        assert ac.in_cooldown() == 0.0

    def test_trigger_blocks_launches(self, tmp_path):
        cfg = make_config(tmp_path, oom_cooldown_seconds=300)
        ac = AdmissionController(cfg)
        ac.trigger_oom_cooldown()
        # Remaining time is positive and capped by the configured window.
        remaining = ac.in_cooldown()
        assert remaining > 0
        assert remaining <= 300
        decision = ac.can_launch(pending_job(), [])
        assert not decision.ok
        assert "cooldown" in decision.reason

    def test_cooldown_takes_priority_over_capacity_rules(self, tmp_path):
        # Cooldown is checked before exclusive/cap/budgets: even on an empty node
        # it blocks, and it would block regardless of how much capacity is free.
        cfg = make_config(tmp_path, oom_cooldown_seconds=300, max_concurrent=100)
        ac = AdmissionController(cfg)
        ac.trigger_oom_cooldown()
        decision = ac.can_launch(pending_job(cores=1, ram_gb=1.0), [])
        assert not decision.ok
        assert "cooldown" in decision.reason

    def test_cooldown_decays_with_time(self, tmp_path, monkeypatch):
        # Drive wall-clock forward via the module's time source rather than
        # sleeping: trigger now, then jump past the window and confirm the gate
        # reopens. Patch ``time.time`` as seen inside nodepilot.admission.
        import nodepilot.admission as admission_mod

        base = 1_000_000.0
        clock = {"now": base}
        monkeypatch.setattr(admission_mod.time, "time", lambda: clock["now"])

        cfg = make_config(tmp_path, oom_cooldown_seconds=300)
        ac = AdmissionController(cfg)
        ac.trigger_oom_cooldown()  # deadline = base + 300
        assert ac.in_cooldown() == pytest.approx(300.0)
        assert not ac.can_launch(pending_job(), [])

        # Halfway through: still cooling down, with less time remaining.
        clock["now"] = base + 150
        assert ac.in_cooldown() == pytest.approx(150.0)
        assert not ac.can_launch(pending_job(), [])

        # Past the window: cooldown has fully decayed and launches resume.
        clock["now"] = base + 301
        assert ac.in_cooldown() == 0.0
        decision = ac.can_launch(pending_job(), [])
        assert decision.ok, decision.reason

    def test_in_cooldown_never_negative(self, tmp_path, monkeypatch):
        import nodepilot.admission as admission_mod

        clock = {"now": 500.0}
        monkeypatch.setattr(admission_mod.time, "time", lambda: clock["now"])
        ac = AdmissionController(make_config(tmp_path, oom_cooldown_seconds=10))
        ac.trigger_oom_cooldown()
        clock["now"] += 9999  # long past expiry
        assert ac.in_cooldown() == 0.0  # clamped to zero, not negative


# ---------------------------------------------------------------------------
# maxcore sanity (advisory: warns, never blocks)
# ---------------------------------------------------------------------------
class TestMaxcoreAdvisory:
    def test_insane_maxcore_is_admitted_with_warning(self, tmp_path):
        # ram_gb=8 but maxcore=4000 MiB over nprocs=8 needs ~40.6 GB: the sanity
        # rule fails, yet admission still returns ok=True with a 'warning' note.
        cfg = make_config(tmp_path, core_budget=1000, ram_budget_gb=1000.0,
                          max_concurrent=100)
        ac = AdmissionController(cfg)
        job = pending_job(cores=8, ram_gb=8.0, maxcore=4000)
        decision = ac.can_launch(job, [])
        assert decision.ok  # advisory -> never blocks
        assert "warning" in decision.reason

    def test_sane_maxcore_returns_plain_ok(self, tmp_path):
        # ram_gb=64 comfortably covers maxcore=4000 * nprocs=8 * 1.3 / 1024.
        cfg = make_config(tmp_path, core_budget=1000, ram_budget_gb=1000.0,
                          max_concurrent=100)
        ac = AdmissionController(cfg)
        job = pending_job(cores=8, ram_gb=64.0, maxcore=4000)
        decision = ac.can_launch(job, [])
        assert decision.ok
        assert decision.reason == "ok"
        assert "warning" not in decision.reason

    def test_no_maxcore_hint_is_always_sane(self, tmp_path):
        cfg = make_config(tmp_path, core_budget=1000, ram_budget_gb=1000.0)
        ac = AdmissionController(cfg)
        job = pending_job(cores=4, ram_gb=4.0, maxcore=0)
        decision = ac.can_launch(job, [])
        assert decision.ok
        assert decision.reason == "ok"


# ---------------------------------------------------------------------------
# maxcore_sane() in isolation
# ---------------------------------------------------------------------------
class TestMaxcoreSaneFunction:
    def test_zero_maxcore_is_sane(self):
        assert maxcore_sane(Job(id="j", command="echo", maxcore=0)) is True

    def test_ram_below_requirement_is_insane(self):
        # need = 4000 * 8 * 1.3 / 1024 ~= 40.6 GB; 8 GB is short.
        job = Job(id="j", command="echo", cores=8, ram_gb=8.0, maxcore=4000)
        assert maxcore_sane(job) is False

    def test_ram_above_requirement_is_sane(self):
        job = Job(id="j", command="echo", cores=8, ram_gb=64.0, maxcore=4000)
        assert maxcore_sane(job) is True

    def test_nprocs_overrides_cores_in_requirement(self):
        # effective_nprocs() uses explicit nprocs when given: 1000 MiB * 16 * 1.3
        # / 1024 ~= 20.3 GB needed, so 16 GB is insufficient despite cores=1.
        job = Job(id="j", command="echo", cores=1, nprocs=16, ram_gb=16.0,
                  maxcore=1000)
        assert maxcore_sane(job) is False

    def test_custom_factor_is_respected(self):
        # need = 1000 * 4 * 2.0 / 1024 ~= 7.81 GB; 7 GB fails at factor 2.0 but
        # the same job is sane at the default 1.3 factor (~5.08 GB).
        job = Job(id="j", command="echo", cores=4, ram_gb=7.0, maxcore=1000)
        assert maxcore_sane(job, factor=2.0) is False
        assert maxcore_sane(job) is True


# ---------------------------------------------------------------------------
# Occupancy accounting helpers
# ---------------------------------------------------------------------------
class TestOccupancyHelpers:
    def _mixed(self) -> list[Job]:
        return [
            running_job("r1", cores=4, ram_gb=8.0),
            running_job("r2", cores=2, ram_gb=4.0),
            Job(id="p", command="echo", cores=16, ram_gb=64.0,
                status=JobStatus.PENDING),
            Job(id="d", command="echo", cores=8, ram_gb=32.0,
                status=JobStatus.DONE),
        ]

    def test_running_cores_sums_only_running(self):
        assert running_cores(self._mixed()) == 6  # 4 + 2

    def test_running_ram_gb_sums_only_running(self):
        assert running_ram_gb(self._mixed()) == pytest.approx(12.0)  # 8 + 4

    def test_running_count_counts_only_running(self):
        assert running_count(self._mixed()) == 2

    def test_helpers_on_empty_list(self):
        assert running_cores([]) == 0
        assert running_ram_gb([]) == 0
        assert running_count([]) == 0


# ---------------------------------------------------------------------------
# Rule precedence on a fully-loaded node (an integration-flavoured smoke test)
# ---------------------------------------------------------------------------
class TestRulePrecedence:
    def test_first_failing_rule_short_circuits(self, tmp_path):
        # Build a node that violates *several* rules at once and confirm the
        # cheapest check wins. Pause is first, so its reason must be the one
        # reported even though the cap and budgets are also blown.
        pause = tmp_path / "halt.pause"
        pause.write_text("", encoding="ascii")
        cfg = make_config(
            tmp_path,
            pause_file=str(pause),
            max_concurrent=1,
            core_budget=2,
            ram_budget_gb=2.0,
        )
        ac = AdmissionController(cfg)
        running = [running_job("a", cores=4, ram_gb=8.0),
                   running_job("b", cores=4, ram_gb=8.0)]
        decision = ac.can_launch(pending_job(cores=8, ram_gb=64.0), running)
        assert not decision.ok
        assert "paused" in decision.reason

    def test_clean_node_admits(self, tmp_path):
        # No pause, no cooldown, comfortable budgets, nothing exclusive running.
        cfg = make_config(tmp_path)
        ac = AdmissionController(cfg)
        running = [running_job("a", cores=2, ram_gb=4.0)]
        decision = ac.can_launch(pending_job(cores=2, ram_gb=4.0), running)
        assert decision.ok
        assert decision.reason == "ok"
        assert bool(decision) is decision.ok
