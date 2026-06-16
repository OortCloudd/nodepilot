"""Unit tests for :mod:`nodepilot.cgroups`.

Only the *unprivileged*, pure-logic surface is exercised here -- nothing that
needs systemd, a user bus, root, or a real cgroup v2 hierarchy:

* :func:`scope_unit_name` -- the deterministic unit-name construction and the
  sanitisation of unsafe characters in a job id.
* :func:`wrap_scope_command` -- the *shape* of the ``systemd-run`` argv: the
  ``--user --scope`` mode, the joined slice, the ``MemoryMax=<n>G`` ceiling,
  ``MemorySwapMax=0``, and a ``MemoryHigh`` strictly below ``MemoryMax``. The
  wrapped command is appended verbatim.
* :func:`set_oom_score_adj` / :func:`set_self_oom_score_adj` -- the clamp to
  ``[-1000, 1000]`` and the contract that they return ``False`` (never raise)
  when the write cannot happen.
* :class:`SliceMonitor` -- accounting reads against a *fake* cgroup directory
  built under ``tmp_path`` (``memory.current`` / ``memory.max`` files), the
  ``"max"`` sentinel mapping to ``None``, and the absent-cgroup -> ``None``
  fall-through.

The slice directory lookup is monkeypatched (``_slice_dir_candidates``) so the
monitor reads our temp files instead of the host's ``/sys/fs/cgroup``; that
keeps every test hermetic and root-free.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nodepilot import cgroups
from nodepilot.cgroups import (
    SliceMonitor,
    scope_unit_name,
    set_oom_score_adj,
    set_self_oom_score_adj,
    wrap_scope_command,
)

_GIB = 1024 ** 3


# ---------------------------------------------------------------------------
# scope_unit_name
# ---------------------------------------------------------------------------
class TestScopeUnitName:
    def test_plain_id_is_prefixed_and_suffixed(self) -> None:
        assert scope_unit_name("job1") == "nodepilot-job1.scope"

    def test_hyphen_and_underscore_are_preserved(self) -> None:
        assert scope_unit_name("a-b_c") == "nodepilot-a-b_c.scope"

    def test_unsafe_characters_are_sanitised_to_underscore(self) -> None:
        # Slashes, dots, spaces and colons are not valid in the id portion and
        # each becomes a single '_'.
        assert scope_unit_name("a/b c.d:e") == "nodepilot-a_b_c_d_e.scope"

    def test_is_deterministic(self) -> None:
        # Same id -> same unit name (so the scope is re-identifiable after a
        # restart).
        assert scope_unit_name("orca-7") == scope_unit_name("orca-7")


# ---------------------------------------------------------------------------
# wrap_scope_command
# ---------------------------------------------------------------------------
class TestWrapScopeCommand:
    def test_argv_shape_and_memory_properties(self) -> None:
        argv = wrap_scope_command(["echo", "hi"], "job1", 64.0)

        # systemd-run --user --scope mode.
        assert argv[0] == "systemd-run"
        assert "--user" in argv
        assert "--scope" in argv

        # Joined to the default slice and named after the job.
        assert "--slice=nodepilot.slice" in argv
        assert f"--unit={scope_unit_name('job1')}" in argv

        # Memory properties are passed as `-p KEY=VALUE` pairs.
        props = _properties(argv)
        assert props["MemoryMax"] == "64G"
        assert props["MemorySwapMax"] == "0"
        # MemoryHigh sits below MemoryMax (64 - 4 headroom = 60G).
        assert props["MemoryHigh"] == "60G"
        assert _gib(props["MemoryHigh"]) < _gib(props["MemoryMax"])

        # The wrapped command is appended verbatim at the tail.
        assert argv[-2:] == ["echo", "hi"]

    def test_custom_slice_and_swap_and_headroom(self) -> None:
        argv = wrap_scope_command(
            ["sleep", "1"],
            "j2",
            32.0,
            slice_name="work.slice",
            swap_max=0,
            high_headroom_gb=8.0,
        )
        assert "--slice=work.slice" in argv
        props = _properties(argv)
        assert props["MemoryMax"] == "32G"
        assert props["MemoryHigh"] == "24G"  # 32 - 8
        assert props["MemorySwapMax"] == "0"

    def test_memory_high_clamped_to_at_least_one_gib(self) -> None:
        # ram_gb (2) - headroom (4) would be negative; MemoryHigh floors at 1G.
        argv = wrap_scope_command(["true"], "tiny", 2.0)
        props = _properties(argv)
        assert props["MemoryMax"] == "2G"
        assert props["MemoryHigh"] == "1G"
        assert _gib(props["MemoryHigh"]) < _gib(props["MemoryMax"])

    def test_swap_max_zero_disables_swap(self) -> None:
        argv = wrap_scope_command(["true"], "j", 16.0)
        assert _properties(argv)["MemorySwapMax"] == "0"


# ---------------------------------------------------------------------------
# oom_score_adj helpers
# ---------------------------------------------------------------------------
class _RecordingPath:
    """Stand-in for ``pathlib.Path`` that records the written oom value."""

    written: list[str] = []

    def __init__(self, *_args: object) -> None:
        pass

    def write_text(self, data: str, *_a: object, **_k: object) -> int:
        type(self).written.append(data)
        return len(data)


class TestOomScoreAdjClamp:
    def test_value_clamped_to_upper_bound(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _RecordingPath.written = []
        monkeypatch.setattr(cgroups, "Path", _RecordingPath)
        assert set_oom_score_adj(1234, 5000) is True
        assert _RecordingPath.written == ["1000"]

    def test_value_clamped_to_lower_bound(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _RecordingPath.written = []
        monkeypatch.setattr(cgroups, "Path", _RecordingPath)
        assert set_oom_score_adj(1234, -5000) is True
        assert _RecordingPath.written == ["-1000"]

    def test_in_range_value_written_unchanged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _RecordingPath.written = []
        monkeypatch.setattr(cgroups, "Path", _RecordingPath)
        assert set_oom_score_adj(1234, -250) is True
        assert _RecordingPath.written == ["-250"]


class TestOomScoreAdjFailureIsSoft:
    def test_returns_false_for_nonexistent_pid(self) -> None:
        # PID 2**31-1 is effectively impossible; the /proc path is missing, so
        # the write raises OSError internally and the function reports False.
        assert set_oom_score_adj(2_147_483_647, -100) is False

    def test_returns_false_when_write_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Simulate an unwritable /proc/<pid>/oom_score_adj (e.g. another user's
        # process): the helper must swallow the error and return False.
        class _Unwritable:
            def __init__(self, *_a: object) -> None:
                pass

            def write_text(self, *_a: object, **_k: object) -> int:
                raise PermissionError("operation not permitted")

        monkeypatch.setattr(cgroups, "Path", _Unwritable)
        assert set_oom_score_adj(1, -1000) is False

    def test_self_helper_returns_false_when_write_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # set_self_oom_score_adj delegates to set_oom_score_adj(getpid(), ...);
        # it must likewise never raise.
        class _Unwritable:
            def __init__(self, *_a: object) -> None:
                pass

            def write_text(self, *_a: object, **_k: object) -> int:
                raise OSError("read-only")

        monkeypatch.setattr(cgroups, "Path", _Unwritable)
        assert set_self_oom_score_adj(-998) is False


# ---------------------------------------------------------------------------
# SliceMonitor against a fake cgroup directory
# ---------------------------------------------------------------------------
class TestSliceMonitor:
    def _point_at(
        self, monkeypatch: pytest.MonkeyPatch, directory: Path | None
    ) -> SliceMonitor:
        """Make the monitor's only candidate dir be ``directory`` (or none)."""
        candidates = [directory] if directory is not None else []
        monkeypatch.setattr(
            cgroups, "_slice_dir_candidates", lambda slice_name: candidates
        )
        return SliceMonitor("nodepilot.slice")

    def test_reads_used_and_max_from_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cg = tmp_path / "nodepilot.slice"
        cg.mkdir()
        # 8 GiB committed, 64 GiB ceiling.
        (cg / "memory.current").write_text(str(8 * _GIB), encoding="ascii")
        (cg / "memory.max").write_text(str(64 * _GIB), encoding="ascii")

        mon = self._point_at(monkeypatch, cg)
        assert mon.is_active() is True
        assert mon.cgroup_dir() == cg
        assert mon.used_gb() == pytest.approx(8.0)
        assert mon.max_gb() == pytest.approx(64.0)

    def test_max_sentinel_maps_to_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # An unbounded slice writes the literal "max" in memory.max.
        cg = tmp_path / "nodepilot.slice"
        cg.mkdir()
        (cg / "memory.current").write_text(str(2 * _GIB), encoding="ascii")
        (cg / "memory.max").write_text("max", encoding="ascii")

        mon = self._point_at(monkeypatch, cg)
        assert mon.used_gb() == pytest.approx(2.0)
        assert mon.max_gb() is None  # "max" -> no enforced ceiling

    def test_absent_cgroup_directory_reads_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # No candidate directory exists at all.
        mon = self._point_at(monkeypatch, None)
        assert mon.is_active() is False
        assert mon.cgroup_dir() is None
        assert mon.used_gb() is None
        assert mon.max_gb() is None

    def test_candidate_present_but_not_a_directory_is_inactive(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # is_dir() guards against a stray regular file masquerading as the slice.
        not_a_dir = tmp_path / "nodepilot.slice"
        not_a_dir.write_text("", encoding="ascii")
        mon = self._point_at(monkeypatch, not_a_dir)
        assert mon.is_active() is False
        assert mon.used_gb() is None

    def test_missing_memory_file_reads_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The cgroup dir exists but the specific accounting file does not; the
        # OSError on read is swallowed and the reader returns None.
        cg = tmp_path / "nodepilot.slice"
        cg.mkdir()  # no memory.current / memory.max written
        mon = self._point_at(monkeypatch, cg)
        assert mon.is_active() is True
        assert mon.used_gb() is None
        assert mon.max_gb() is None

    def test_non_integer_memory_value_reads_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A malformed memory.current (not "max", not an int) must not raise.
        cg = tmp_path / "nodepilot.slice"
        cg.mkdir()
        (cg / "memory.current").write_text("garbage", encoding="ascii")
        mon = self._point_at(monkeypatch, cg)
        assert mon.used_gb() is None


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _properties(argv: list[str]) -> dict[str, str]:
    """Collect ``-p KEY=VALUE`` pairs from a systemd-run argv into a dict."""
    props: dict[str, str] = {}
    for flag, val in zip(argv, argv[1:]):
        if flag == "-p" and "=" in val:
            key, _, v = val.partition("=")
            props[key] = v
    return props


def _gib(spec: str) -> float:
    """Parse a ``"<n>G"`` systemd memory spec into a float GiB count."""
    assert spec.endswith("G"), spec
    return float(spec[:-1])


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
