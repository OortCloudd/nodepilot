"""cgroups v2 memory containment via systemd scopes.

The OOM guard at the heart of nodepilot is *not* the declarative ``ram_gb``
sum -- real footprints routinely overshoot or undershoot the declared value by
tens of percent. The reliable guard is the kernel: run every job inside a
cgroup v2 scope with a hard ``MemoryMax``, gather those scopes under one parent
*slice* whose own ``MemoryMax`` sits safely below physical RAM, and read that
slice's ``memory.current`` as the single source of truth for committed memory.

This module wraps that pattern over **systemd user scopes**:

* :func:`wrap_scope_command` turns a job's argv into a ``systemd-run --user
  --scope`` invocation that sets ``MemoryMax`` / ``MemorySwapMax=0`` and joins
  the configured slice.
* :class:`SliceMonitor` locates the slice's cgroup directory under
  ``/sys/fs/cgroup`` and reads ``memory.current`` / ``memory.max``.
* helpers detect whether a given job's scope is still alive (for zombie
  reconciliation) and apply ``oom_score_adj``.

If systemd or cgroup v2 is unavailable, callers fall back to declarative
accounting; nothing here raises on a missing cgroup -- the probes return
``None`` and the orchestrator copes.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from shutil import which

__all__ = [
    "systemd_run_available",
    "wrap_scope_command",
    "scope_unit_name",
    "SliceMonitor",
    "scope_is_active",
    "set_oom_score_adj",
    "set_self_oom_score_adj",
]

_CGROUP_ROOT = Path("/sys/fs/cgroup")
_GIB = 1024 ** 3


def systemd_run_available() -> bool:
    """Whether ``systemd-run`` exists and a user bus is reachable."""
    if which("systemd-run") is None:
        return False
    # A user manager must be running for ``--user`` scopes to work.
    return bool(os.environ.get("XDG_RUNTIME_DIR")) or _user_bus_socket_exists()


def _user_bus_socket_exists() -> bool:
    runtime = os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}"
    return Path(runtime, "bus").exists()


def scope_unit_name(job_id: str) -> str:
    """Deterministic systemd scope unit name for a job.

    Sanitises the job id into a valid unit name and suffixes ``.scope`` so the
    same job always maps to the same unit -- which lets
    :func:`scope_is_active` recognise it after an orchestrator restart.
    """
    safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in job_id)
    return f"nodepilot-{safe}.scope"


def wrap_scope_command(
    argv: list[str],
    job_id: str,
    ram_gb: float,
    *,
    slice_name: str = "nodepilot.slice",
    swap_max: int = 0,
    high_headroom_gb: float = 4.0,
) -> list[str]:
    """Wrap ``argv`` in a ``systemd-run --user --scope`` with a memory cap.

    The produced command runs ``argv`` inside a transient scope joined to
    ``slice_name`` with:

    * ``MemoryMax=<ram_gb>G`` -- hard ceiling; the kernel oom-kills the scope
      (not the whole host) on overshoot.
    * ``MemorySwapMax=0`` -- no silent swapping that would mask a runaway.
    * ``MemoryHigh`` a few GiB below ``Max`` -- throttle before the hard kill.

    Parameters
    ----------
    argv
        The already-built command to run (e.g. a NUMA-prefixed job command).
    job_id
        Used to name the scope (:func:`scope_unit_name`).
    ram_gb
        Hard memory ceiling in GiB.
    slice_name
        Parent slice to contain the scope under.
    swap_max
        ``MemorySwapMax`` in bytes (default 0 = no swap).
    high_headroom_gb
        Gap between ``MemoryHigh`` and ``MemoryMax``.

    Returns
    -------
    list[str]
        The full ``systemd-run`` argv. ``--collect`` removes the scope when the
        process exits; ``--no-block`` avoids stalling on a slow user bus.
    """
    high_gb = max(1.0, ram_gb - high_headroom_gb)
    return [
        "systemd-run",
        "--user",
        "--scope",
        "--quiet",
        "--collect",
        "--no-block",
        f"--slice={slice_name}",
        f"--unit={scope_unit_name(job_id)}",
        "-p",
        f"MemoryMax={ram_gb:g}G",
        "-p",
        f"MemorySwapMax={swap_max}",
        "-p",
        f"MemoryHigh={high_gb:g}G",
        *argv,
    ]


def _slice_dir_candidates(slice_name: str) -> list[Path]:
    """Plausible cgroup v2 directories for a user slice.

    systemd nests user units under ``user.slice/user-UID.slice/user@UID.service``
    and, depending on version, an intermediate ``app.slice``.
    """
    uid = os.getuid()
    base = _CGROUP_ROOT / "user.slice" / f"user-{uid}.slice" / f"user@{uid}.service"
    return [
        base / slice_name,
        base / "app.slice" / slice_name,
        # System-scope fallback if the slice was created system-wide.
        _CGROUP_ROOT / slice_name,
    ]


class SliceMonitor:
    """Reads live memory accounting from a systemd slice's cgroup.

    The slice is the parent of every job scope; its ``memory.current`` is the
    authoritative committed-RAM figure and ``memory.max`` its kernel-enforced
    ceiling (which an operator may change at runtime with
    ``systemctl --user set-property``).

    All readers return ``None`` when the slice cgroup is absent (e.g. the slice
    has never been started, or this host has no cgroup v2), so the orchestrator
    can fall back to declarative accounting.
    """

    def __init__(self, slice_name: str) -> None:
        self.slice_name = slice_name

    # -- locating the cgroup directory ----------------------------------
    def cgroup_dir(self) -> Path | None:
        """The slice's cgroup v2 directory, or ``None`` if not present."""
        for cand in _slice_dir_candidates(self.slice_name):
            if cand.is_dir():
                return cand
        return None

    def is_active(self) -> bool:
        """Whether the slice cgroup currently exists."""
        return self.cgroup_dir() is not None

    # -- memory accounting ----------------------------------------------
    def used_gb(self) -> float | None:
        """``memory.current`` of the slice in GiB (sum of all job scopes)."""
        return self._read_mem_file("memory.current")

    def max_gb(self) -> float | None:
        """``memory.max`` of the slice in GiB; ``None`` if unset or ``max``."""
        raw = self._read_mem_raw("memory.max")
        if raw is None or raw == "max":
            return None
        try:
            return int(raw) / _GIB
        except ValueError:
            return None

    def _read_mem_raw(self, fname: str) -> str | None:
        cg = self.cgroup_dir()
        if cg is None:
            return None
        try:
            return (cg / fname).read_text(encoding="ascii").strip()
        except OSError:
            return None

    def _read_mem_file(self, fname: str) -> float | None:
        raw = self._read_mem_raw(fname)
        if raw is None or raw == "max":
            return None
        try:
            return int(raw) / _GIB
        except ValueError:
            return None


def scope_is_active(job_id: str, slice_name: str) -> bool:
    """Whether a job's scope cgroup directory still exists under the slice.

    Used at orchestrator startup to tell a genuinely-running job from a
    "zombie" left ``running`` in the state file by a crash.
    """
    monitor = SliceMonitor(slice_name)
    cg = monitor.cgroup_dir()
    if cg is None:
        return False
    return (cg / scope_unit_name(job_id)).is_dir()


# ---------------------------------------------------------------------------
# oom_score_adj helpers
# ---------------------------------------------------------------------------
def set_oom_score_adj(pid: int, value: int) -> bool:
    """Write ``oom_score_adj`` for *pid* (clamped to the valid range).

    Negative values protect a process from the OOM killer; positive values
    make it a preferred victim. Returns ``True`` on success, ``False`` if the
    process is gone or the write is not permitted.
    """
    value = max(-1000, min(1000, value))
    try:
        Path(f"/proc/{pid}/oom_score_adj").write_text(str(value), encoding="ascii")
        return True
    except (OSError, ProcessLookupError):
        return False


def set_self_oom_score_adj(value: int) -> bool:
    """Protect (or expose) the *current* process via ``oom_score_adj``.

    Call this once at orchestrator startup with a negative value so the
    scheduler survives a memory storm and can clean up after the jobs it
    sacrifices.
    """
    return set_oom_score_adj(os.getpid(), value)


def journal_reports_oom(within_minutes: int = 30) -> bool:
    """Best-effort check of the kernel log for a recent OOM kill.

    Reads ``journalctl -k`` over the last *within_minutes*. Returns ``False``
    if journalctl is unavailable or the call fails -- this is a hint used to
    label ambiguous failures, never a hard signal.
    """
    if which("journalctl") is None:
        return False
    try:
        result = subprocess.run(
            ["journalctl", "-k", "--since", f"{within_minutes} min ago"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (subprocess.SubprocessError, OSError):
        return False
    return "oom-kill" in result.stdout.lower() or "out of memory" in result.stdout.lower()
