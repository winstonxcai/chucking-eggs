"""Cross-cutting utilities for the guanzero training stack.

Pure infrastructure (logging, profiling, metric emission, schedules,
filesystem layout, ad-hoc helpers) — no dependencies on model, data, or
runtime modules.
"""

from .legal_utils import dedup_strategic, strategic_key
from .logging_setup import TqdmLoggingHandler, setup_run_logging
from .metrics import (
    JsonlBackend,
    METRICS_SCHEMA_VERSION,
    MetricsBackend,
    MetricsWriter,
    jsonl_writer,
)
from .profiler import PhaseProfiler, _K_BUCKETS, _k_bucket
from .run_layout import RunLayout
from .schedules import epsilon_linear

__all__ = [
    "dedup_strategic",
    "strategic_key",
    "TqdmLoggingHandler",
    "setup_run_logging",
    "JsonlBackend",
    "METRICS_SCHEMA_VERSION",
    "MetricsBackend",
    "MetricsWriter",
    "jsonl_writer",
    "PhaseProfiler",
    "_K_BUCKETS",
    "_k_bucket",
    "RunLayout",
    "epsilon_linear",
]
