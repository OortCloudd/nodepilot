"""Command-line interface for nodepilot.

Exposes four subcommands::

    nodepilot run    <queue.yaml> [--max-ticks N]   # start/resume the loop
    nodepilot status <queue.yaml>                    # print queue state
    nodepilot kill   <queue.yaml> <job-id>           # kill a running job
    nodepilot reset  <queue.yaml>                     # discard saved state

``run`` resumes from the JSON state file if present (see
:meth:`nodepilot.orchestrator.Orchestrator.from_queue`); ``reset`` deletes it so
the next ``run`` starts fresh from the YAML.

The entry point is wired in ``pyproject.toml`` as ``nodepilot = nodepilot.cli:main``.
"""

from __future__ import annotations

import argparse
import sys

from nodepilot.config import Job, JobStatus, load_queue
from nodepilot.orchestrator import Orchestrator
from nodepilot.state import load_state, state_exists

__all__ = ["main", "build_parser"]


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the ``nodepilot`` command."""
    parser = argparse.ArgumentParser(
        prog="nodepilot",
        description="Single-node, OOM-safe HPC job orchestrator.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="start or resume the scheduling loop")
    p_run.add_argument("queue", help="path to the YAML queue file")
    p_run.add_argument(
        "--max-ticks",
        type=int,
        default=None,
        help="stop after N ticks (default: run until the queue drains)",
    )

    p_status = sub.add_parser("status", help="print the current queue state")
    p_status.add_argument("queue", help="path to the YAML queue file")

    p_kill = sub.add_parser("kill", help="kill a running job by id")
    p_kill.add_argument("queue", help="path to the YAML queue file")
    p_kill.add_argument("job_id", help="id of the job to kill")

    p_reset = sub.add_parser("reset", help="delete the saved state file")
    p_reset.add_argument("queue", help="path to the YAML queue file")

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)

    if args.command == "run":
        Orchestrator.from_queue(args.queue).run(max_ticks=args.max_ticks)
        return 0

    if args.command == "status":
        return _cmd_status(args.queue)

    if args.command == "kill":
        orch = Orchestrator.from_queue(args.queue)
        ok = orch.kill(args.job_id)
        if not ok:
            print(f"no running job with id {args.job_id!r}", file=sys.stderr)
            return 1
        return 0

    if args.command == "reset":
        Orchestrator.from_queue(args.queue).reset()
        print("state cleared")
        return 0

    return 2  # unreachable: argparse enforces a valid subcommand


def _cmd_status(queue_path: str) -> int:
    """Print a compact table of job statuses, preferring saved state."""
    config, queue_jobs = load_queue(queue_path)
    if state_exists(config.state_path):
        jobs = load_state(config.state_path)
    else:
        jobs = queue_jobs

    _print_table(jobs)
    return 0


def _print_table(jobs: list[Job]) -> None:
    header = f"{'ID':<24} {'STATUS':<9} {'PRIO':>4} {'CORES':>5} {'RAM_GB':>7} {'CPU':<12} REASON"
    print(header)
    print("-" * len(header))
    counts: dict[str, int] = {}
    for job in sorted(jobs, key=lambda j: (j.priority, j.id)):
        counts[job.status] = counts.get(job.status, 0) + 1
        print(
            f"{job.id:<24} {job.status:<9} {job.priority:>4} {job.cores:>5} "
            f"{job.ram_gb:>7g} {job.cpu_list or '-':<12} {job.failure_reason}"
        )
    print("-" * len(header))
    summary = " ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    print(f"total={len(jobs)}  {summary}")


if __name__ == "__main__":
    raise SystemExit(main())
