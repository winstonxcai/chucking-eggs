"""Logging configuration for GuanZero training runs.

One logger per run: ``guanzero``. The file handler writes DEBUG-and-above to
``<run_dir>/train.log`` (or ``learner.log`` when called from the learner
subprocess). The optional stream handler, gated on
``GUANZERO_STREAM_LOGS=1``, duplicates INFO-and-above to stdout.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path


def setup_run_logging(
    run_dir: Path,
    log_filename: str = "train.log",
    stream_to_stdout: bool | None = None,
) -> tuple[logging.Logger, Path]:
    """Configure the ``guanzero`` logger for a run.

    Returns the logger and the resolved log file path.

    ``stream_to_stdout`` defaults to the ``GUANZERO_STREAM_LOGS=1``
    env-var when not explicitly set.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / log_filename

    if stream_to_stdout is None:
        stream_to_stdout = os.environ.get("GUANZERO_STREAM_LOGS") == "1"

    logger = logging.getLogger("guanzero")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-5s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_path, mode="a")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    if stream_to_stdout:
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(logging.INFO)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    return logger, log_path


__all__ = ["setup_run_logging"]
