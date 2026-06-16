"""End-to-end tests for the scheduling loop (:mod:`nodepilot.orchestrator`).

These exercise the orchestrator the way it actually runs in production, with
real child processes -- not mocks. A temporary YAML queue is written to
``tmp_path``, cgroup containment is disabled (``memory_slice=''``) so the tests
need no systemd user bus, the subprocess runner is used, and an explicit NUMA
topology is supplied so placement is deterministic on any host.

What is verified here mirrors the manual smoke test performed during the build:

* a job with ``depends_on`` starts only *after* its dependency reaches ``done``;
* a command that exits non-zero is classified ``failed`` with the right
  ``exit_code`` / ``failure_reason`` (``exit_<n>``);
* successful commands are classified ``done`` with ``exit_code == 0``;
* concurrently-running jobs receive NUMA core blocks that never overlap, and
  each placement stays inside its declared node;
* a job left ``running`` in a crash-recovered state with a dead PID is
  reconciled to ``failed`` / ``zombie_at_restart`` and has its core reservation
  released.

The example commands are deliberately trivial (``sleep`` / ``echo`` / ``exit``):
nodepilot only ever sees a shell command string, so synthetic commands give the
same coverage as any real ORCA/CP2K/VASP workload without the weight.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from nodepilot.config import JobStatus, load_queue
from nodepilot.numa import parse_cpu_list
from nodepilot.orchestrator import Orchestrator
from nodepilot.state import load_state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _write_queue(tmp_path: Path, jobs_yaml: str, **global_overrides: object) -> Path:
    """Write a queue YAML into *tmp_path* and return its path.

    The ``global:`` block is fixed up for a hermetic test run:

    * ``memory_slice: ''`` -- no cgroup scope wrapping, so jobs run as plain
      children and no systemd user bus is required.
    * ``runner: subprocess`` -- track jobs by PID (no tmux dependency).
    * ``poll_interval: 0`` -- the loop never sleeps between ticks, so a bounded
      ``max_ticks`` run finishes effectively instantly.
    * explicit ``numa_nodes`` -- two synthetic nodes of four cores each, so
      :func:`nodepilot.numa.allocate` is deterministic regardless of the host's
      real topology.
    * ``state_path`` / ``log_path`` under ``tmp_path`` -- nothing escapes the
      temp dir.

    Callers may override any global via *global_overrides*.
    """
    state_path = tmp_path / "state.json"
    log_path = tmp_path / "nodepilot.log"
    pause_file = tmp_path / ".nodepilot.pause"

    base_globals = {
        "memory_slice": "''",  # quoted so YAML yields the empty string
        "runner": "subprocess",
        "poll_interval": 0,
        "max_concurrent": 4,
        "core_budget": 8,
        "ram_budget_gb": 64,
        "ram_safety_gb": 0,
        "numa_nodes": "{0: '0-3', 1: '4-7'}",
        "state_path": repr(str(state_path)),
        "log_path": repr(str(log_path)),
        "pause_file": repr(str(pause_file)),
    }
    base_globals.update({k: _yaml_scalar(v) for k, v in global_overrides.items()})

    global_block = "\n".join(f"  {k}: {v}" for k, v in base_globals.items())
    # Dedent first (triple-quoted blocks carry their source indentation, which
    # varies with how deeply the call is nested) then re-indent uniformly under
    # ``jobs:`` so the document is always well-formed YAML.
    jobs_block = textwrap.indent(textwrap.dedent(jobs_yaml).strip(), "  ")
    doc = f"global:\n{global_block}\njobs:\n{jobs_block}\n"

    queue_path = tmp_path / "queue.yaml"
    queue_path.write_text(doc, encoding="utf-8")
    return queue_path


def _yaml_scalar(value: object) -> str:
    """Render an override value as a YAML scalar (quote strings)."""
    if isinstance(value, str):
        return repr(value)
    return str(value)


class _RecordingOrchestrator(Orchestrator):
    """Orchestrator that snapshots the running set at the end of every tick.

    After each tick it records ``{job_id: cpu_list}`` for jobs that are
    ``running`` at that moment. Tests inspect these snapshots to assert
    properties that only hold *while jobs are concurrent* (e.g. non-overlapping
    core blocks), which a post-run state inspection could not prove.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.running_snapshots: list[dict[str, str]] = []

    def _tick(self) -> bool:
        changed = super()._tick()
        snapshot = {
            j.id: j.cpu_list
            for j in self.jobs
            if j.status == JobStatus.RUNNING and j.cpu_list
        }
        self.running_snapshots.append(snapshot)
        return changed


# ---------------------------------------------------------------------------
# End-to-end loop with real subprocesses
# ---------------------------------------------------------------------------
def test_run_dependencies_failures_and_success(tmp_path: Path) -> None:
    """A full run: dependency ordering + failure classification + success.

    Queue shape::

        first  (sleep briefly, exit 0)            -- a successful root job
        second (depends on first, exit 0)         -- must wait for `first`
        boom   (exit 3)                           -- independent failure

    Assertions:

    * every job reaches a terminal state;
    * ``first`` and ``second`` are ``done`` with ``exit_code == 0``;
    * ``second`` did not *start* until ``first`` had *finished*
      (``second.start_time >= first.end_time``) -- the dependency gate held;
    * ``boom`` is ``failed`` with ``exit_code == 3`` and
      ``failure_reason == 'exit_3'``.
    """
    queue_path = _write_queue(
        tmp_path,
        """
        - id: first
          command: "sleep 0.3; exit 0"
          cores: 2
          ram_gb: 1
        - id: second
          command: "echo done"
          cores: 2
          ram_gb: 1
          depends_on: [first]
        - id: boom
          command: "exit 3"
          cores: 2
          ram_gb: 1
        """,
    )

    orch = Orchestrator.from_queue(str(queue_path))
    # A generous tick bound: the loop exits early as soon as the queue drains,
    # and with poll_interval=0 it never sleeps, so this stays fast.
    orch.run(max_ticks=200)

    by_id = {j.id: j for j in orch.jobs}

    # Everything finished.
    assert all(j.is_terminal() for j in orch.jobs), {
        j.id: j.status for j in orch.jobs
    }

    # Successful jobs.
    first, second, boom = by_id["first"], by_id["second"], by_id["boom"]
    assert first.status == JobStatus.DONE
    assert first.exit_code == 0
    assert second.status == JobStatus.DONE
    assert second.exit_code == 0

    # Dependency ordering: the dependent must not have launched before its
    # dependency completed. Both timestamps are set by the loop (start at
    # launch, end at reap), so this is the real ordering guarantee.
    assert second.start_time > 0.0
    assert first.end_time > 0.0
    assert second.start_time >= first.end_time, (
        f"second started at {second.start_time} before first ended "
        f"at {first.end_time}"
    )

    # Failure classification: a non-zero exit becomes failed / exit_<n>.
    assert boom.status == JobStatus.FAILED
    assert boom.exit_code == 3
    assert boom.failure_reason == "exit_3"


def test_failed_dependency_skips_dependents_and_drains(tmp_path: Path) -> None:
    """A failed dependency must drain the queue, not hang the loop.

    ``blocker`` exits non-zero, so ``downstream`` (which depends on it) can
    never satisfy its dependency, and ``grandchild`` (which depends on
    ``downstream``) likewise. Without failure propagation both dependents would
    sit ``pending`` forever and ``run()`` -- which exits only when nothing is
    pending or running -- would never return.

    The run is invoked with **no** ``max_ticks``: the test itself only
    terminates if the loop drains on its own. Both dependents must end
    ``skipped`` (transitively), and the failure reason must name the blocker.
    """
    queue_path = _write_queue(
        tmp_path,
        """
        - id: blocker
          command: "exit 1"
          cores: 2
          ram_gb: 1
        - id: downstream
          command: "echo should-never-run"
          cores: 2
          ram_gb: 1
          depends_on: [blocker]
        - id: grandchild
          command: "echo should-never-run"
          cores: 2
          ram_gb: 1
          depends_on: [downstream]
        """,
    )

    orch = Orchestrator.from_queue(str(queue_path))
    orch.run()  # no max_ticks: only returns if the queue actually drains

    by_id = {j.id: j for j in orch.jobs}
    assert by_id["blocker"].status == JobStatus.FAILED
    assert by_id["downstream"].status == JobStatus.SKIPPED
    assert by_id["grandchild"].status == JobStatus.SKIPPED
    assert by_id["downstream"].failure_reason == "dependency_failed: blocker"
    assert by_id["downstream"].command  # job object intact, just never launched
    assert all(j.is_terminal() for j in orch.jobs)


def test_run_persists_terminal_state(tmp_path: Path) -> None:
    """After draining, the state file reflects the terminal outcomes.

    Confirms the loop snapshots progress to ``state_path`` (atomic JSON) and
    that a fresh :func:`nodepilot.state.load_state` reads back the same final
    statuses -- the basis for crash-safe resume.
    """
    queue_path = _write_queue(
        tmp_path,
        """
        - id: ok
          command: "true"
          cores: 1
          ram_gb: 1
        - id: bad
          command: "exit 7"
          cores: 1
          ram_gb: 1
        """,
    )
    config, _ = load_queue(str(queue_path))

    Orchestrator.from_queue(str(queue_path)).run(max_ticks=100)

    assert Path(config.state_path).is_file()
    persisted = {j.id: j for j in load_state(config.state_path)}
    assert persisted["ok"].status == JobStatus.DONE
    assert persisted["ok"].exit_code == 0
    assert persisted["bad"].status == JobStatus.FAILED
    assert persisted["bad"].exit_code == 7
    assert persisted["bad"].failure_reason == "exit_7"


def test_concurrent_jobs_get_nonoverlapping_numa_blocks(tmp_path: Path) -> None:
    """Concurrently-running jobs are pinned to disjoint, in-node core blocks.

    Two independent jobs each request two cores and sleep long enough to be
    co-resident for at least one tick. With a two-node 4+4 topology and a
    concurrency cap of 2, both run at once. Using a recording orchestrator we
    capture the running set per tick and assert:

    * at least one tick had both jobs running simultaneously;
    * whenever 2+ jobs run together, their assigned cpu sets are pairwise
      disjoint (no core double-booked);
    * every assigned core belongs to one of the configured NUMA nodes.
    """
    queue_path = _write_queue(
        tmp_path,
        """
        - id: alpha
          command: "sleep 0.4"
          cores: 2
          ram_gb: 1
        - id: beta
          command: "sleep 0.4"
          cores: 2
          ram_gb: 1
        """,
        max_concurrent=2,
    )

    config, _ = load_queue(str(queue_path))
    orch = _RecordingOrchestrator.from_queue(str(queue_path))
    orch.run(max_ticks=200)

    # Both jobs completed successfully.
    by_id = {j.id: j for j in orch.jobs}
    assert by_id["alpha"].status == JobStatus.DONE
    assert by_id["beta"].status == JobStatus.DONE

    # There was a moment where both were running concurrently.
    concurrent_snaps = [s for s in orch.running_snapshots if len(s) >= 2]
    assert concurrent_snaps, (
        "expected a tick with >=2 jobs running; snapshots="
        f"{orch.running_snapshots}"
    )

    # In every concurrent moment the core blocks were disjoint.
    for snap in concurrent_snaps:
        cpu_sets = [parse_cpu_list(cl) for cl in snap.values()]
        union_size = len(set().union(*cpu_sets))
        total_size = sum(len(s) for s in cpu_sets)
        assert union_size == total_size, (
            f"overlapping NUMA blocks in concurrent snapshot {snap}"
        )

    # Each final placement stayed within the declared NUMA topology.
    all_node_cores = parse_cpu_list("0-3") | parse_cpu_list("4-7")
    for jid in ("alpha", "beta"):
        cores = parse_cpu_list(by_id[jid].cpu_list)
        assert cores, f"{jid} got no cores"
        assert cores <= all_node_cores, f"{jid} placed outside topology: {cores}"
        assert by_id[jid].numa_node in (0, 1)


def test_exit_code_zero_when_command_succeeds(tmp_path: Path) -> None:
    """A plain successful command yields done / exit_code 0 / empty reason."""
    queue_path = _write_queue(
        tmp_path,
        """
        - id: hello
          command: "echo hi && true"
          cores: 1
          ram_gb: 1
        """,
    )
    orch = Orchestrator.from_queue(str(queue_path))
    orch.run(max_ticks=50)

    job = orch.jobs[0]
    assert job.status == JobStatus.DONE
    assert job.exit_code == 0
    assert job.failure_reason == ""


# ---------------------------------------------------------------------------
# Zombie reconciliation
# ---------------------------------------------------------------------------
def test_reconcile_zombies_flips_dead_running_job(tmp_path: Path) -> None:
    """A crash-recovered ``running`` job with a dead PID becomes a zombie fail.

    We simulate a state file written just before a crash: one job left
    ``running``, its ``session`` pointing at a PID that does not exist, and a
    core reservation still held. On startup ``_reconcile_zombies`` must:

    * flip it to ``failed`` with ``failure_reason == 'zombie_at_restart'``;
    * clear ``cpu_list`` so the allocator can reuse those cores.

    ``memory_slice`` is empty, so the scope-liveness probe short-circuits to
    "not alive" and the only liveness signal is the (absent) PID.
    """
    # Build a config + a single zombie job directly, bypassing YAML so we can
    # plant the exact runtime fields a crash would have left behind.
    config, _ = load_queue(
        str(
            _write_queue(
                tmp_path,
                """
                - id: placeholder
                  command: "true"
                """,
            )
        )
    )
    from nodepilot.config import Job  # local import: only the test needs it

    # PID chosen to be implausibly live; os.kill(pid, 0) raises ProcessLookupError.
    zombie = Job(
        id="ghost",
        command="sleep 999",
        cores=2,
        ram_gb=1,
        status=JobStatus.RUNNING,
        session="pid:999999",
        cpu_list="0-1",
        numa_node=0,
        start_time=1.0,
    )

    orch = Orchestrator(config, [zombie])
    orch._reconcile_zombies()

    assert zombie.status == JobStatus.FAILED
    assert zombie.failure_reason == "zombie_at_restart"
    assert zombie.cpu_list == ""
    assert zombie.end_time > 0.0


def test_zombie_reconciliation_runs_on_resume_from_state(tmp_path: Path) -> None:
    """``from_queue`` + ``run`` reconciles a zombie persisted in the state file.

    End-to-end variant: a state file with a dead ``running`` job is written to
    ``state_path``; ``Orchestrator.from_queue`` resumes from it (not the YAML),
    and the first thing ``run`` does is reconcile zombies. The job ends
    ``failed`` / ``zombie_at_restart`` and the queue drains.
    """
    queue_path = _write_queue(
        tmp_path,
        """
        - id: real
          command: "true"
          cores: 1
          ram_gb: 1
        """,
    )
    config, _ = load_queue(str(queue_path))

    # Hand-write a state file: one zombie running job with a dead PID.
    from nodepilot.config import Job
    from nodepilot.state import save_state

    zombie = Job(
        id="orphan",
        command="sleep 999",
        cores=2,
        ram_gb=1,
        status=JobStatus.RUNNING,
        session="pid:999999",
        cpu_list="4-5",
        numa_node=1,
        start_time=1.0,
    )
    save_state(config.state_path, [zombie])
    assert Path(config.state_path).is_file()

    # from_queue resumes from the state file (zombie), ignoring the YAML job.
    orch = Orchestrator.from_queue(str(queue_path))
    assert {j.id for j in orch.jobs} == {"orphan"}, "should have resumed from state"

    orch.run(max_ticks=20)

    reconciled = {j.id: j for j in orch.jobs}["orphan"]
    assert reconciled.status == JobStatus.FAILED
    assert reconciled.failure_reason == "zombie_at_restart"
    assert reconciled.cpu_list == ""


# ---------------------------------------------------------------------------
# Bounded-loop / reset behaviour
# ---------------------------------------------------------------------------
def test_reset_removes_state_file(tmp_path: Path) -> None:
    """``reset`` deletes the persisted state so the next run starts fresh."""
    queue_path = _write_queue(
        tmp_path,
        """
        - id: only
          command: "true"
          cores: 1
          ram_gb: 1
        """,
    )
    config, _ = load_queue(str(queue_path))

    orch = Orchestrator.from_queue(str(queue_path))
    orch.run(max_ticks=50)
    assert Path(config.state_path).is_file()

    orch.reset()
    assert not Path(config.state_path).exists()


def test_finished_jobs_release_their_process_handles(tmp_path: Path) -> None:
    """Reaped jobs must not accumulate ``Popen`` handles (bounded memory).

    The runner retains a handle per launched job to read its exit code; once a
    job is reaped the handle is dropped. After the queue drains the retained-set
    must be empty -- otherwise it would grow for the life of the scheduler.
    """
    queue_path = _write_queue(
        tmp_path,
        """
        - id: ok
          command: "true"
          cores: 1
          ram_gb: 1
        - id: bad
          command: "exit 5"
          cores: 1
          ram_gb: 1
        """,
    )
    orch = Orchestrator.from_queue(str(queue_path))
    orch.run(max_ticks=100)

    assert all(j.is_terminal() for j in orch.jobs)
    assert orch.runner._procs == {}, "finished jobs left dangling Popen handles"


if __name__ == "__main__":  # pragma: no cover - allow ``python test_orchestrator.py``
    raise SystemExit(pytest.main([__file__, "-v"]))
