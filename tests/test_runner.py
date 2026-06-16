"""Unit tests for :mod:`nodepilot.runner` outcome classification and handle GC.

Two behaviours are pinned here without launching real processes:

* :func:`nodepilot.runner.reap` consults the kernel log (``journal_reports_oom``)
  *only* for an ambiguous death -- a ``137``/SIGKILL exit or an unrecoverable
  (``None``) status -- and never on a clean or plainly non-zero exit, so a
  long-lived scheduler does not fork ``journalctl`` on every successful reap.
* :meth:`nodepilot.runner.Runner._forget` drops a finished job's retained
  ``Popen`` handle, and ``reap`` calls it, so ``_procs`` stays bounded.

``returncode`` and ``journal_reports_oom`` are monkeypatched, so the tests are
hermetic: no child processes, no systemd, no real kernel log.
"""

from __future__ import annotations

import pytest

import nodepilot.runner as runner_mod
from nodepilot.config import Config, Job
from nodepilot.runner import Runner, reap


def _runner() -> Runner:
    return Runner(Config())


def _job(session: str = "pid:0") -> Job:
    job = Job(id="j", command="x")
    job.session = session
    return job


def _patch(monkeypatch: pytest.MonkeyPatch, *, code, oom: bool) -> list[int]:
    """Stub returncode->*code* and a call-counting journal->*oom*. Returns the
    list the journal stub appends to (its length = number of journal calls)."""
    calls: list[int] = []

    def journal(**_kwargs: object) -> bool:
        calls.append(1)
        return oom

    monkeypatch.setattr(runner_mod.cgroups, "journal_reports_oom", journal)
    return calls


class TestReapJournalIsConditional:
    def test_clean_exit_does_not_consult_journal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = _patch(monkeypatch, code=0, oom=False)
        r = _runner()
        monkeypatch.setattr(r, "returncode", lambda job: 0)
        out = reap(_job(), r)
        assert (out.status, out.exit_code) == ("done", 0)
        assert calls == [], "journalctl must not run on a clean exit"

    def test_nonzero_exit_does_not_consult_journal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = _patch(monkeypatch, code=3, oom=False)
        r = _runner()
        monkeypatch.setattr(r, "returncode", lambda job: 3)
        out = reap(_job(), r)
        assert (out.status, out.reason) == ("failed", "exit_3")
        assert calls == [], "journalctl must not run on a plain non-zero exit"

    def test_signal_exit_does_not_consult_journal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = _patch(monkeypatch, code=139, oom=False)  # 128 + SIGSEGV
        r = _runner()
        monkeypatch.setattr(r, "returncode", lambda job: 139)
        out = reap(_job(), r)
        assert out.reason == "signal_11"
        assert calls == []

    def test_137_with_oom_hint_is_oom_killed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = _patch(monkeypatch, code=137, oom=True)
        r = _runner()
        monkeypatch.setattr(r, "returncode", lambda job: 137)
        out = reap(_job(), r)
        assert (out.status, out.reason) == ("failed", "oom_killed")
        assert calls == [1], "137 must consult the journal"

    def test_137_without_oom_hint_is_killed_signal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch(monkeypatch, code=137, oom=False)
        r = _runner()
        monkeypatch.setattr(r, "returncode", lambda job: 137)
        out = reap(_job(), r)
        assert out.reason == "killed_signal"

    def test_none_with_oom_hint_is_oom_killed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = _patch(monkeypatch, code=None, oom=True)
        r = _runner()
        monkeypatch.setattr(r, "returncode", lambda job: None)
        out = reap(_job("tmux:j"), r)
        assert (out.status, out.reason) == ("failed", "oom_killed")
        assert calls == [1]

    def test_none_without_oom_hint_is_done(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch(monkeypatch, code=None, oom=False)
        r = _runner()
        monkeypatch.setattr(r, "returncode", lambda job: None)
        out = reap(_job("tmux:j"), r)
        assert out.status == "done"


class TestForgetReleasesHandles:
    def test_forget_pops_the_pid(self) -> None:
        r = _runner()
        r._procs[4242] = object()  # type: ignore[assignment]
        r._forget(_job("pid:4242"))
        assert 4242 not in r._procs

    def test_forget_is_a_noop_for_tmux_jobs(self) -> None:
        r = _runner()
        r._procs[7] = object()  # type: ignore[assignment]
        r._forget(_job("tmux:session"))
        assert 7 in r._procs  # untouched

    def test_reap_forgets_the_handle(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch(monkeypatch, code=0, oom=False)
        r = _runner()
        r._procs[55] = object()  # type: ignore[assignment]
        monkeypatch.setattr(r, "returncode", lambda job: 0)
        reap(_job("pid:55"), r)
        assert 55 not in r._procs


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
