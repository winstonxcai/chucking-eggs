# Changelog

## Unreleased

- Added actor RNG state capture/restore for full DART checkpoint resume.
- Made actor queue backpressure fail fast instead of silently dropping sample batches.
- Added checkpoint-eval timeout support via `eval.max_wait_s`.
- Added pure-Python `guandan_rs.select_legal` fallback coverage and native/fallback parity tests.
- Added root documentation for development, config, evaluation, architecture, debugging, profiling, API, release, and contributing.
- Added portable Modal volume defaults with `DART_MODAL_VOL` and `DART_HF_CACHE` overrides.
- Added CI lint/type-check/test stages for Python, ML tests, and web backend tests.
- Added frontend project README and `ml/src/guandan_rs/README.md`.
