"""Logging configuration for GuanZero training runs.

One namespaced logger per process. The file handler writes DEBUG-and-above
to ``<run_dir>/<log_filename>`` with a full timestamped format. The optional
stream handler — installed when ``stream_to_stdout`` is true (or the
``GUANZERO_STREAM_LOGS=1`` env-var is set) — writes INFO-and-above through
``tqdm.write`` so an active progress bar is not corrupted.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path


_FILE_FORMATTER = logging.Formatter(
    "%(asctime)s [%(levelname)-5s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_TERM_FORMATTER = logging.Formatter("%(message)s")


class TqdmLoggingHandler(logging.Handler):
    """Stream handler that writes via ``tqdm.write``.

    Falls back gracefully to plain stdout when no bar is active — ``tqdm.write``
    detects this case itself.
    """

    def emit(self, record: logging.LogRecord) -> None:
        from tqdm import tqdm
        try:
            tqdm.write(self.format(record))
        except Exception:
            self.handleError(record)


def setup_run_logging(
    run_dir: Path,
    log_filename: str = "train.log",
    name: str = "guanzero",
    stream_to_stdout: bool | None = None,
) -> tuple[logging.Logger, Path]:
    """Configure a namespaced logger for one process of a run.

    Returns the logger and the resolved log file path. Existing handlers on
    the logger are cleared so re-entry from a re-imported module (spawn) does
    not double-attach.

    ``stream_to_stdout`` defaults to the ``GUANZERO_STREAM_LOGS=1`` env-var
    when not explicitly set.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / log_filename

    if stream_to_stdout is None:
        stream_to_stdout = os.environ.get("GUANZERO_STREAM_LOGS") == "1"

    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    fh = logging.FileHandler(log_path, mode="a")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(_FILE_FORMATTER)
    logger.addHandler(fh)

    if stream_to_stdout:
        sh = TqdmLoggingHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(_TERM_FORMATTER)
        logger.addHandler(sh)

    return logger, log_path


__all__ = ["TqdmLoggingHandler", "setup_run_logging"]
