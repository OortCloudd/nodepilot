"""Job execution backends and post-launch supervision.

Two ways to start a job's command, selected by ``Config.runner``:

* ``"subprocess"`` (default) -- the command runs as a detached child via
  ``setsid``; the orchestrator tracks it by PID and process-group.
* ``"tmux"`` -- the command runs in a named tmux session you can ``tmux
  attach`` to for live output; liveness is checked with ``tmux has-session``.

The runner is also where two gritty realities live:

* :func:`reap` -- decide a finished job's outcome (success / OOM / nonzero
  exit) from its exit status and, as a hint, the kernel log.
* :func:`enforce_pin` -- re-apply ``taskset`` to any of a job's processes whose
  CPU affinity has drifted outside its assigned core block (MPI ranks love to
  escape). Crucially it **skips the shared tmux server process**, which pgrep
  matches by session name but which must never be narrowed to one job's cores.

The runner builds the *final* command (placement prefix + cgroup scope wrapper)
and hands execution back; it does not make admission or placement decisions.
"""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path
from shutil import which

from nodepilot import cgroups
from nodepilot.config import Config, Job
from nodepilot.numa import Placement, parse_cpu_list, placement_prefix

__all__ = [
    "Outcome",
    "Runner",
    "build_command",
    "reap",
    "enforce_pin",
]


@dataclass(frozen=True)
class Outcome:
    """Classified result of a finished job."""

    status: str  # "done" or "failed"
    reason: str  # "" on success; a short code otherwise
    exit_code: int | None


def build_command(
    job: Job,
    placement: Placement,
    config: Config,
) -> list[str]:
    """Assemble the full argv to launch *job*, inside-out.

    Layering (outermost first):

    #. ``systemd-run --user --scope`` with the memory cap, if cgroup
       containment is enabled and available -- this is the OOM guard.
    #. ``numactl --physcpubind --membind`` (or ``taskset`` fallback) -- CPU and
       memory placement from :mod:`nodepilot.numa`.
    #. ``bash -lc "<job.command>"`` -- the user's command, run through a shell
       so pipelines / ``&&`` / env expansion work as written in the YAML.

    Returns the argv list ready for :class:`Runner` to execute.
    """
    inner = ["bash", "-lc", job.command]
    placed = [*placement_prefix(placement), *inner]
    if config.memory_slice and cgroups.systemd_run_available():
        return cgroups.wrap_scope_command(
            placed,
            job_id=job.id,
            ram_gb=job.ram_gb,
            slice_name=config.memory_slice,
        )
    return placed


class Runner:
    """Starts and supervises job processes for a chosen backend.

    Parameters
    ----------
    config
        Provides ``runner`` (``"subprocess"`` / ``"tmux"``) and the
        ``job_oom_score_adj`` to apply to launched processes.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        if config.runner == "tmux" and which("tmux") is None:
            raise RuntimeError("runner='tmux' but tmux is not on PATH")
        # Retain Popen handles for jobs we launched this process, keyed by PID.
        # Keeping the handle prevents the child being auto-reaped by Popen's
        # finalizer (which would lose the exit status) and lets us read the
        # exit code via ``returncode`` instead of a racy bare ``waitpid``.
        # Jobs inherited from a state file after a restart have no handle here;
        # those are handled by the orchestrator's zombie reconciliation.
        self._procs: dict[int, subprocess.Popen] = {}

    # -- launching -------------------------------------------------------
    def start(self, job: Job, argv: list[str]) -> None:
        """Launch *job* with the fully-built *argv*, recording how to track it.

        Sets ``job.session`` to either a PID (subprocess backend) or the tmux
        session name. Applies ``oom_score_adj`` to the new process so jobs are
        sacrificed before the orchestrator under memory pressure.
        """
        workdir = job.workdir or os.getcwd()
        env = {**os.environ, **job.env}
        if self.config.runner == "tmux":
            self._start_tmux(job, argv, workdir, env)
        else:
            self._start_subprocess(job, argv, workdir, env)

    def _start_subprocess(
        self, job: Job, argv: list[str], workdir: str, env: dict[str, str]
    ) -> None:
        # ``start_new_session`` puts the child in its own process group so we
        # can signal the whole tree on kill and so it survives our restarts.
        proc = subprocess.Popen(
            argv,
            cwd=workdir,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        job.session = f"pid:{proc.pid}"
        self._procs[proc.pid] = proc
        cgroups.set_oom_score_adj(proc.pid, self.config.job_oom_score_adj)

    def _start_tmux(
        self, job: Job, argv: list[str], workdir: str, env: dict[str, str]
    ) -> None:
        session = f"nodepilot_{job.id}"
        # Kill any stale session of the same name first (idempotent relaunch).
        subprocess.run(
            ["tmux", "kill-session", "-t", session],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        # tmux runs a login shell; cd into workdir then exec the command line.
        command_line = "cd {} && exec {}".format(
            shlex.quote(workdir), " ".join(shlex.quote(a) for a in argv)
        )
        subprocess.run(
            ["tmux", "new-session", "-d", "-s", session, "bash", "-lc", command_line],
            env=env,
            check=True,
        )
        job.session = f"tmux:{session}"

    # -- liveness --------------------------------------------------------
    def is_alive(self, job: Job) -> bool:
        """Whether *job*'s tracked process/session is still running."""
        if job.session.startswith("tmux:"):
            return _tmux_alive(job.session[len("tmux:") :])
        if job.session.startswith("pid:"):
            pid = int(job.session[len("pid:") :])
            proc = self._procs.get(pid)
            if proc is not None:
                # ``poll`` reaps the child if it has exited and caches the exit
                # status on the handle (read later by ``reap``); returns None
                # while still running.
                return proc.poll() is None
            # No handle (job inherited after a restart): probe by PID.
            return _pid_alive(pid)
        return False

    def returncode(self, job: Job) -> int | None:
        """Exit status of a finished subprocess job, or ``None`` if unknown.

        Reads the cached ``Popen.returncode`` for jobs we launched. Returns
        ``None`` for tmux jobs and for inherited jobs without a retained handle.
        Negative codes (signal deaths) are normalised to ``128 + signo`` to
        match the shell convention used in :func:`reap`.
        """
        if not job.session.startswith("pid:"):
            return None
        pid = int(job.session[len("pid:") :])
        proc = self._procs.get(pid)
        if proc is None:
            return None
        rc = proc.poll()
        if rc is None:
            return None
        # Popen reports signal deaths as negative numbers; normalise.
        return rc if rc >= 0 else 128 - rc

    # -- killing ---------------------------------------------------------
    def kill(self, job: Job) -> None:
        """Terminate *job*'s process group / tmux session."""
        if job.session.startswith("tmux:"):
            subprocess.run(
                ["tmux", "kill-session", "-t", job.session[len("tmux:") :]],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        elif job.session.startswith("pid:"):
            pid = int(job.session[len("pid:") :])
            try:
                os.killpg(os.getpgid(pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
            # Drain the handle so the killed child is reaped, not left a zombie.
            proc = self._procs.get(pid)
            if proc is not None:
                try:
                    proc.wait(timeout=5)
                except (subprocess.TimeoutExpired, OSError):
                    pass


# ---------------------------------------------------------------------------
# Liveness primitives
# ---------------------------------------------------------------------------
def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but not ours
    # Treat zombies as gone: a reaped-but-not-waited child is finished.
    try:
        state = Path(f"/proc/{pid}/stat").read_text(encoding="ascii").split()[2]
        return state != "Z"
    except (OSError, IndexError):
        return True


def _tmux_alive(session: str) -> bool:
    if which("tmux") is None:
        return False
    result = subprocess.run(
        ["tmux", "has-session", "-t", session],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


# ---------------------------------------------------------------------------
# Outcome classification
# ---------------------------------------------------------------------------
def reap(job: Job, runner: Runner) -> Outcome:
    """Determine the outcome of a job whose process is no longer alive.

    For the subprocess backend the exit code is the primary signal. For tmux
    (where the wrapper's exit status is not easily recovered) we rely on the
    kernel-log OOM hint and default to success when nothing indicates failure.

    The kernel-log OOM check (:func:`nodepilot.cgroups.journal_reports_oom`) is
    only a *hint* used to label an otherwise-ambiguous death; it is not
    authoritative and never gates anything by itself.
    """
    exit_code = runner.returncode(job)

    # An explicit OOM kill shows up as SIGKILL (137) and/or in the kernel log.
    oom_hint = cgroups.journal_reports_oom(within_minutes=5)
    if exit_code == 137 or (exit_code is None and oom_hint):
        if oom_hint:
            return Outcome("failed", "oom_killed", exit_code)
        return Outcome("failed", "killed_signal", exit_code)

    if exit_code is None:
        # tmux backend with no recoverable status: assume success unless the
        # kernel log flagged an OOM (handled above).
        return Outcome("done", "", None)

    if exit_code == 0:
        return Outcome("done", "", 0)
    if exit_code > 128:
        return Outcome("failed", f"signal_{exit_code - 128}", exit_code)
    return Outcome("failed", f"exit_{exit_code}", exit_code)


# ---------------------------------------------------------------------------
# Pin enforcement
# ---------------------------------------------------------------------------
def enforce_pin(job: Job) -> int:
    """Re-pin any of *job*'s processes that drifted off its core block.

    Walks the PIDs matching the job (by command substring), reads each one's
    ``Cpus_allowed_list``, and runs ``taskset`` on those whose affinity is not
    a subset of ``job.cpu_list``. This counters MPI launchers that re-affine
    ranks via ``sched_setaffinity`` after launch, and catches helper processes
    spawned mid-run.

    The shared tmux **server** is explicitly skipped: pgrep matches it by the
    session name embedded in its argv, but narrowing the server to one job's
    cores would break every other session it hosts.

    Returns the number of processes re-pinned this call (0 if none drifted or
    the job has no assigned cores).
    """
    if not job.cpu_list or not job.id:
        return 0
    if which("taskset") is None:
        return 0
    expected = parse_cpu_list(job.cpu_list)
    repinned = 0
    for pid in _pids_for_job(job):
        try:
            comm = Path(f"/proc/{pid}/comm").read_text(encoding="ascii").strip()
        except OSError:
            continue
        # Never re-pin the shared tmux server (see docstring).
        if comm.startswith("tmux"):
            continue
        current = _cpus_allowed(pid)
        if current is None:
            continue
        if not current or not current.issubset(expected):
            subprocess.run(
                ["taskset", "-cp", job.cpu_list, str(pid)],
                capture_output=True,
                check=False,
            )
            repinned += 1
    return repinned


def _pids_for_job(job: Job) -> list[int]:
    """PIDs whose command line references this job.

    Uses the tracked PID's process group for the subprocess backend (precise),
    and a ``pgrep`` substring match for tmux (best-effort).
    """
    if job.session.startswith("pid:"):
        root = int(job.session[len("pid:") :])
        try:
            pgid = os.getpgid(root)
        except ProcessLookupError:
            return []
        pids: list[int] = []
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                stat = (entry / "stat").read_text(encoding="ascii").split()
                if int(stat[4]) == pgid:  # field 5 = process group id
                    pids.append(int(entry.name))
            except (OSError, IndexError, ValueError):
                continue
        return pids
    # tmux backend: match by job id substring.
    if which("pgrep") is None:
        return []
    result = subprocess.run(
        ["pgrep", "-f", job.id],
        capture_output=True,
        text=True,
        check=False,
    )
    return [int(p) for p in result.stdout.split() if p.isdigit()]


def _cpus_allowed(pid: int) -> set[int] | None:
    """Parse ``Cpus_allowed_list`` for *pid* from ``/proc/<pid>/status``."""
    try:
        with open(f"/proc/{pid}/status", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("Cpus_allowed_list:"):
                    return parse_cpu_list(line.split(":", 1)[1].strip())
    except OSError:
        return None
    return None
