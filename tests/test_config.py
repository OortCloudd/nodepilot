"""Unit tests for queue loading, ``Job``/``Config`` defaults, and state I/O.

These cover the parsing and persistence layer that the rest of nodepilot is
built on:

* :func:`nodepilot.config.load_queue` -- both the structured ``global``/``jobs``
  form and the bare-list shorthand, plus its error contract (missing
  ``id``/``command`` -> ``ValueError``, duplicate ids -> ``ValueError``, missing
  file -> ``FileNotFoundError``) and forward-compatible tolerance of unknown
  keys.
* :class:`nodepilot.config.Job` -- field defaults, ``effective_nprocs`` falling
  back to ``cores``, and ``runtime_hours`` arithmetic.
* :class:`nodepilot.config.Config` -- ``__post_init__`` filling ``core_budget``
  and ``ram_budget_gb`` from the host when left at 0, while honouring explicit
  values.
* :func:`nodepilot.state.save_state` / :func:`nodepilot.state.load_state` --
  a round trip that preserves runtime fields, plus the atomic-write guarantee
  (no leftover ``*.tmp`` sidecar).

Everything is self-contained: queue and state files are written under pytest's
``tmp_path`` fixture, so no real machine state is touched and tests can run
fully in parallel.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nodepilot.config import (
    Config,
    Job,
    JobStatus,
    job_to_dict,
    load_queue,
)
from nodepilot.state import load_state, save_state, state_exists


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _write(path: Path, text: str) -> Path:
    """Write *text* to *path* and return *path* (for inline fixture files)."""
    path.write_text(text, encoding="utf-8")
    return path


# ===========================================================================
# Job defaults and derived properties
# ===========================================================================
class TestJobDefaults:
    def test_minimal_job_only_requires_id_and_command(self) -> None:
        job = Job(id="hello", command="echo hi")
        assert job.id == "hello"
        assert job.command == "echo hi"

    def test_resource_and_scheduling_defaults(self) -> None:
        job = Job(id="j", command="true")
        assert job.cores == 1
        assert job.ram_gb == 4.0
        assert job.maxcore == 0
        assert job.nprocs == 0
        assert job.priority == 100
        assert job.exclusive is False
        assert job.depends_on == []
        assert job.env == {}
        assert job.metadata == {}

    def test_runtime_state_defaults(self) -> None:
        job = Job(id="j", command="true")
        assert job.status == JobStatus.PENDING
        assert job.session == ""
        assert job.cpu_list == ""
        assert job.numa_node == -1
        assert job.start_time == 0.0
        assert job.end_time == 0.0
        assert job.failure_reason == ""
        assert job.exit_code is None

    def test_mutable_defaults_are_not_shared(self) -> None:
        # field(default_factory=...) must give each instance its own container,
        # not a shared class-level list/dict.
        a = Job(id="a", command="x")
        b = Job(id="b", command="y")
        a.depends_on.append("upstream")
        a.env["KEY"] = "val"
        a.metadata["tag"] = 1
        assert b.depends_on == []
        assert b.env == {}
        assert b.metadata == {}

    def test_effective_nprocs_defaults_to_cores(self) -> None:
        assert Job(id="j", command="x", cores=8).effective_nprocs() == 8
        # The default cores=1 implies a single rank.
        assert Job(id="j", command="x").effective_nprocs() == 1

    def test_effective_nprocs_honours_explicit_value(self) -> None:
        job = Job(id="j", command="x", cores=16, nprocs=4)
        assert job.effective_nprocs() == 4

    @pytest.mark.parametrize(
        "start, end, expected",
        [
            (0.0, 0.0, 0.0),          # never started/finished
            (100.0, 0.0, 0.0),        # started but not finished
            (0.0, 100.0, 0.0),        # end without start is not a duration
            (100.0, 100.0 + 3600, 1.0),
            (100.0, 100.0 + 1800, 0.5),
            (1000.0, 1000.0 + 3600 * 2.5, 2.5),
        ],
    )
    def test_runtime_hours(self, start: float, end: float, expected: float) -> None:
        job = Job(id="j", command="x", start_time=start, end_time=end)
        assert job.runtime_hours() == pytest.approx(expected)

    def test_is_active_and_is_terminal(self) -> None:
        running = Job(id="r", command="x", status=JobStatus.RUNNING)
        done = Job(id="d", command="x", status=JobStatus.DONE)
        failed = Job(id="f", command="x", status=JobStatus.FAILED)
        pending = Job(id="p", command="x", status=JobStatus.PENDING)

        assert running.is_active() and not running.is_terminal()
        assert done.is_terminal() and not done.is_active()
        assert failed.is_terminal() and not failed.is_active()
        assert not pending.is_active() and not pending.is_terminal()


# ===========================================================================
# Config defaults / host backfill
# ===========================================================================
class TestConfigDefaults:
    def test_host_budgets_filled_when_zero(self) -> None:
        # core_budget/ram_budget_gb default to 0 in the dataclass and are
        # replaced by host-derived values in __post_init__.
        cfg = Config()
        assert cfg.core_budget > 0
        assert cfg.ram_budget_gb > 0

    def test_explicit_budgets_are_preserved(self) -> None:
        cfg = Config(core_budget=8, ram_budget_gb=64.0)
        assert cfg.core_budget == 8
        assert cfg.ram_budget_gb == 64.0

    def test_unset_scalar_defaults(self) -> None:
        cfg = Config()
        assert cfg.max_concurrent == 4
        assert cfg.ram_safety_gb == 20.0
        assert cfg.memory_slice == "nodepilot.slice"
        assert cfg.orchestrator_oom_score_adj == -800
        assert cfg.job_oom_score_adj == 500
        assert cfg.oom_cooldown_seconds == 300
        assert cfg.runner == "subprocess"
        assert cfg.poll_interval == 10
        assert cfg.state_path == "nodepilot_state.json"
        assert cfg.log_path == "nodepilot.log"
        assert cfg.pause_file == ".nodepilot.pause"
        assert cfg.numa_nodes == {}

    def test_numa_nodes_resolved_uses_explicit_map(self) -> None:
        cfg = Config(numa_nodes={0: "0-15", 1: "16-31"})
        resolved = cfg.numa_nodes_resolved
        assert resolved == {0: "0-15", 1: "16-31"}
        # Keys/values are normalised to int/str.
        assert all(isinstance(k, int) and isinstance(v, str) for k, v in resolved.items())


# ===========================================================================
# load_queue -- structured form
# ===========================================================================
class TestLoadQueueStructured:
    def test_parses_global_and_jobs(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path / "queue.yaml",
            """
            global:
              max_concurrent: 7
              ram_budget_gb: 123
              core_budget: 16
              poll_interval: 30
            jobs:
              - id: a
                command: "echo a"
                cores: 4
                ram_gb: 8
              - id: b
                command: "sleep 1"
            """,
        )
        config, jobs = load_queue(path)

        assert config.max_concurrent == 7
        assert config.ram_budget_gb == 123
        assert config.core_budget == 16
        assert config.poll_interval == 30

        assert [j.id for j in jobs] == ["a", "b"]
        assert jobs[0].command == "echo a"
        assert jobs[0].cores == 4
        assert jobs[0].ram_gb == 8
        # Second job falls back to dataclass defaults.
        assert jobs[1].cores == 1
        assert jobs[1].ram_gb == 4.0

    def test_jobs_preserve_file_order(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path / "queue.yaml",
            """
            jobs:
              - id: third
                command: "true"
              - id: first
                command: "true"
              - id: second
                command: "true"
            """,
        )
        _, jobs = load_queue(path)
        assert [j.id for j in jobs] == ["third", "first", "second"]

    def test_missing_global_block_uses_config_defaults(self, tmp_path: Path) -> None:
        # A document with only `jobs:` (no `global:`) must still build a usable
        # Config with host-filled budgets.
        path = _write(
            tmp_path / "queue.yaml",
            """
            jobs:
              - id: a
                command: "true"
            """,
        )
        config, jobs = load_queue(path)
        assert config.max_concurrent == 4
        assert config.core_budget > 0
        assert config.ram_budget_gb > 0
        assert [j.id for j in jobs] == ["a"]

    def test_empty_jobs_list(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "queue.yaml", "global:\n  max_concurrent: 2\njobs: []\n")
        config, jobs = load_queue(path)
        assert config.max_concurrent == 2
        assert jobs == []


# ===========================================================================
# load_queue -- bare-list form
# ===========================================================================
class TestLoadQueueBareList:
    def test_bare_list_of_jobs(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path / "queue.yaml",
            """
            - id: solo
              command: "hostname"
              cores: 2
            - id: solo2
              command: "echo hi"
            """,
        )
        config, jobs = load_queue(path)
        # No global block: config is all defaults (host budgets filled).
        assert isinstance(config, Config)
        assert config.max_concurrent == 4
        assert config.core_budget > 0
        assert [j.id for j in jobs] == ["solo", "solo2"]
        assert jobs[0].cores == 2


# ===========================================================================
# load_queue -- forward compatibility (unknown keys ignored)
# ===========================================================================
class TestLoadQueueUnknownKeys:
    def test_unknown_job_keys_are_ignored(self, tmp_path: Path) -> None:
        # A queue written by a newer nodepilot (or carrying human annotations)
        # must load without error; unrecognised keys are dropped.
        path = _write(
            tmp_path / "queue.yaml",
            """
            jobs:
              - id: a
                command: "true"
                notes: "free-form annotation"
                future_field: 42
                cores: 3
            """,
        )
        _, jobs = load_queue(path)
        assert jobs[0].id == "a"
        assert jobs[0].cores == 3
        # The unknown keys did not sneak onto the dataclass.
        assert not hasattr(jobs[0], "notes")
        assert not hasattr(jobs[0], "future_field")

    def test_unknown_global_keys_are_ignored(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path / "queue.yaml",
            """
            global:
              max_concurrent: 3
              experimental_knob: "ignore me"
            jobs:
              - id: a
                command: "true"
            """,
        )
        config, _ = load_queue(path)
        assert config.max_concurrent == 3
        assert not hasattr(config, "experimental_knob")


# ===========================================================================
# load_queue -- error contract
# ===========================================================================
class TestLoadQueueErrors:
    def test_missing_file_raises_filenotfound(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_queue(tmp_path / "does_not_exist.yaml")

    def test_job_missing_command_raises_valueerror(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "queue.yaml", "jobs:\n  - id: x\n")
        with pytest.raises(ValueError):
            load_queue(path)

    def test_job_missing_id_raises_valueerror(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "queue.yaml", "jobs:\n  - command: 'echo hi'\n")
        with pytest.raises(ValueError):
            load_queue(path)

    def test_duplicate_ids_raise_valueerror(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path / "queue.yaml",
            """
            jobs:
              - id: dup
                command: "true"
              - id: dup
                command: "false"
            """,
        )
        with pytest.raises(ValueError) as excinfo:
            load_queue(path)
        # The offending id is surfaced to help the user fix the queue.
        assert "dup" in str(excinfo.value)

    def test_jobs_not_a_list_raises_valueerror(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "queue.yaml", "jobs:\n  id: a\n  command: b\n")
        with pytest.raises(ValueError):
            load_queue(path)

    def test_scalar_document_raises_valueerror(self, tmp_path: Path) -> None:
        # A YAML scalar is neither a mapping nor a list of jobs.
        path = _write(tmp_path / "queue.yaml", "just-a-string\n")
        with pytest.raises(ValueError):
            load_queue(path)


# ===========================================================================
# state round-trip and atomicity
# ===========================================================================
class TestStateRoundTrip:
    def test_state_exists_reports_presence(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"
        assert state_exists(path) is False
        save_state(path, [Job(id="a", command="true")])
        assert state_exists(path) is True

    def test_round_trip_preserves_runtime_fields(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"
        original = [
            Job(
                id="done-job",
                command="echo ok",
                status=JobStatus.DONE,
                exit_code=0,
                cpu_list="0-3",
                numa_node=0,
                start_time=1000.0,
                end_time=1000.0 + 3600,
            ),
            Job(
                id="oom-job",
                command="stress-ng --vm 1",
                status=JobStatus.FAILED,
                exit_code=137,
                cpu_list="4-7",
                numa_node=1,
                failure_reason="oom_killed",
            ),
        ]
        save_state(path, original)
        restored = load_state(path)

        assert [j.id for j in restored] == ["done-job", "oom-job"]

        done = restored[0]
        assert done.status == JobStatus.DONE
        assert done.exit_code == 0
        assert done.cpu_list == "0-3"
        assert done.numa_node == 0
        assert done.runtime_hours() == pytest.approx(1.0)

        oom = restored[1]
        assert oom.status == JobStatus.FAILED
        assert oom.exit_code == 137
        assert oom.cpu_list == "4-7"
        assert oom.failure_reason == "oom_killed"

    def test_round_trip_preserves_none_exit_code(self, tmp_path: Path) -> None:
        # A still-pending job has exit_code=None; JSON null must survive the
        # round trip rather than becoming 0 or a string.
        path = tmp_path / "state.json"
        save_state(path, [Job(id="p", command="true", status=JobStatus.PENDING)])
        restored = load_state(path)
        assert restored[0].exit_code is None
        assert restored[0].status == JobStatus.PENDING

    def test_round_trip_preserves_collections(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"
        job = Job(
            id="j",
            command="true",
            depends_on=["a", "b"],
            env={"OMP_NUM_THREADS": "4"},
            metadata={"group": "phase1", "retries": 2},
        )
        save_state(path, [job])
        restored = load_state(path)[0]
        assert restored.depends_on == ["a", "b"]
        assert restored.env == {"OMP_NUM_THREADS": "4"}
        assert restored.metadata == {"group": "phase1", "retries": 2}

    def test_saved_payload_shape(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"
        save_state(path, [Job(id="a", command="true")])
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert set(payload) == {"saved_at", "jobs"}
        assert isinstance(payload["saved_at"], (int, float))
        assert isinstance(payload["jobs"], list)
        assert payload["jobs"][0]["id"] == "a"

    def test_save_is_atomic_no_tmp_left_behind(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"
        save_state(path, [Job(id="a", command="true")])
        # The sibling temp file used for the atomic os.replace must be gone.
        assert (tmp_path / "state.json.tmp").exists() is False
        # Nothing but the final state file should remain in the directory.
        assert sorted(p.name for p in tmp_path.iterdir()) == ["state.json"]

    def test_save_overwrites_in_place(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"
        save_state(path, [Job(id="first", command="true")])
        save_state(path, [Job(id="second", command="true")])
        restored = load_state(path)
        assert [j.id for j in restored] == ["second"]
        assert (tmp_path / "state.json.tmp").exists() is False

    def test_load_state_accepts_bare_list(self, tmp_path: Path) -> None:
        # load_state tolerates a bare list payload (not just the wrapped form).
        path = tmp_path / "state.json"
        path.write_text(
            json.dumps([job_to_dict(Job(id="a", command="true"))]),
            encoding="utf-8",
        )
        restored = load_state(path)
        assert [j.id for j in restored] == ["a"]

    def test_load_state_ignores_unknown_keys(self, tmp_path: Path) -> None:
        # Forward compat: a state file from a newer version may carry per-job
        # keys this version doesn't know about.
        path = tmp_path / "state.json"
        record = job_to_dict(Job(id="a", command="true"))
        record["future_runtime_field"] = "whatever"
        path.write_text(json.dumps({"jobs": [record]}), encoding="utf-8")
        restored = load_state(path)
        assert restored[0].id == "a"
        assert not hasattr(restored[0], "future_runtime_field")

    def test_load_state_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_state(tmp_path / "absent.json")
