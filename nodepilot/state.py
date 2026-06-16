"""JSON state persistence and resume.

The scheduler snapshots the full job list to a JSON file after every tick that
changes something, so a restart (planned or after a crash) resumes exactly
where it left off: finished jobs stay finished, running jobs are reconciled,
pending jobs are retried.

Writes are atomic (write to a temp file, then ``os.replace``) so a crash mid-
write can never leave a truncated state file. Loading tolerates unknown fields
so a state written by a newer version still loads.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from nodepilot.config import Job, job_to_dict
from nodepilot.config import _job_from_dict  # reuse the tolerant constructor

__all__ = ["save_state", "load_state", "state_exists"]


def save_state(path: str | os.PathLike[str], jobs: list[Job]) -> None:
    """Atomically write the job list to *path* as JSON.

    The payload is ``{"saved_at": <epoch>, "jobs": [...]}``. The write goes to
    a sibling ``*.tmp`` file first and is then ``os.replace``-d into place, so
    readers never observe a partial file.
    """
    target = Path(path)
    payload = {
        "saved_at": time.time(),
        "jobs": [job_to_dict(j) for j in jobs],
    }
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, target)


def load_state(path: str | os.PathLike[str]) -> list[Job]:
    """Load jobs from a JSON state file written by :func:`save_state`.

    Accepts both the wrapped form (``{"jobs": [...]}``) and a bare list, and
    ignores unknown per-job keys for forward compatibility.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        raw = data.get("jobs", [])
    elif isinstance(data, list):
        raw = data
    else:
        raise ValueError(f"unexpected state format in {path}")
    return [_job_from_dict(j) for j in raw]


def state_exists(path: str | os.PathLike[str]) -> bool:
    """Whether a state file is present at *path*."""
    return Path(path).is_file()
