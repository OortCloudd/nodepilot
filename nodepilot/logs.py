"""Structured, dependency-free logging for nodepilot.

A thin wrapper over :mod:`logging` that emits one line per event with a stable,
greppable shape::

    2026-01-01 12:00:00 INFO  launch job=hello cores=4 ram_gb=8 cpu=0-3 node=0

Key/value context is appended as ``key=value`` pairs so logs are easy to filter
with ``grep`` or parse downstream, without pulling in a JSON-logging library.
Output goes to stderr and, optionally, to a file.

The module name is ``logs`` (not ``logging``) to avoid shadowing the stdlib
module within the package.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

__all__ = ["get_logger", "kv"]

_CONFIGURED = False


def get_logger(log_path: str | None = None, *, level: int = logging.INFO) -> logging.Logger:
    """Return the package logger, configuring handlers once.

    Parameters
    ----------
    log_path
        If given, log lines are also appended to this file. Passing ``None`` or
        an empty string logs to stderr only.
    level
        Logging level (default :data:`logging.INFO`).
    """
    global _CONFIGURED
    logger = logging.getLogger("nodepilot")
    if not _CONFIGURED:
        logger.setLevel(level)
        fmt = logging.Formatter(
            "%(asctime)s %(levelname)-5s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        stream = logging.StreamHandler(sys.stderr)
        stream.setFormatter(fmt)
        logger.addHandler(stream)
        if log_path:
            try:
                file_handler = logging.FileHandler(log_path)
                file_handler.setFormatter(fmt)
                logger.addHandler(file_handler)
            except OSError:
                # A bad log path must never crash the scheduler; stderr remains.
                logger.warning("could not open log file %s", log_path)
        logger.propagate = False
        _CONFIGURED = True
    return logger


def kv(**pairs: Any) -> str:
    """Render keyword arguments as a ``key=value`` suffix string.

    ``kv(job="hello", cores=4)`` -> ``"job=hello cores=4"``. Values that are
    ``None`` are skipped so optional context does not clutter the line.
    """
    return " ".join(f"{k}={v}" for k, v in pairs.items() if v is not None)
