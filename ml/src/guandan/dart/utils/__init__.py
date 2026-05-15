"""Cross-cutting utilities for the dart training stack.

Pure infrastructure (logging, profiling, metric emission, schedules,
filesystem layout, ad-hoc helpers) — no dependencies on model, data, or
runtime modules.
"""

from .legal_utils import dedup_strategic, select_legal
from .logging_setup import TqdmLoggingHandler, setup_run_logging
from .metrics import (
    JsonlBackend,
    METRICS_SCHEMA_VERSION,
    MetricsBackend,
    MetricsWriter,
    jsonl_writer,
)
from .profiler import PhaseProfiler, k_bucket_label
from .run_layout import RunLayout
from .schedules import epsilon_linear

__all__ = [
    "dedup_strategic",
    "select_legal",
    "TqdmLoggingHandler",
    "setup_run_logging",
    "JsonlBackend",
    "METRICS_SCHEMA_VERSION",
    "MetricsBackend",
    "MetricsWriter",
    "jsonl_writer",
    "PhaseProfiler",
    "k_bucket_label",
    "RunLayout",
    "epsilon_linear",
]
