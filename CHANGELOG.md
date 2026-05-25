# Changelog

## Unreleased

## v0.1.0 - 2026-05-25

- Published the DART 1.25M checkpoint and release evidence bundle.
- Added raw DART-vs-bot eval JSON, Glicko injection inputs, derived leaderboard output, checksum manifest, and compact training metrics summaries.
- Added a GitHub Release artifact for the checkpoint/evaluation bundle.
- Added README demo imagery, architecture figures, local DART web quick start, model card, evaluation guide, and release documentation.
- Hardened WebSocket reconnect-token enforcement and expanded backend/frontend E2E coverage.
- Aligned Docker/Fly deploy paths, locked runtimes, tightened production CORS, and disabled hosted DART serving for cost control.
- Added portable DART CPU smoke coverage and early device validation for CUDA/MPS configs.

- Added actor RNG state capture/restore for full DART checkpoint resume.
- Made actor queue backpressure fail fast instead of silently dropping sample batches.
- Added checkpoint-eval timeout support via `eval.max_wait_s`.
- Added pure-Python `guandan_rs.select_legal` fallback coverage and native/fallback parity tests.
- Added root documentation for development, config, evaluation, architecture, debugging, profiling, API, release, and contributing.
- Added portable Modal volume defaults with `DART_MODAL_VOL` and `DART_HF_CACHE` overrides.
- Added CI lint/type-check/test stages for Python, ML tests, and web backend tests.
- Added frontend project README and `ml/src/guandan_rs/README.md`.
