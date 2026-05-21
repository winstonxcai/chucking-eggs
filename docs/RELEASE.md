# Release Process

This repo does not yet publish formal binary artifacts. Use this checklist for
GitHub releases and public result updates.

## Before Tagging

1. Run CI locally:

   ```bash
   uv run ruff check ml/src/guandan/dart web/backend/app examples
   uv run mypy
   uv run pytest -q ml/tests
   uv run pytest -q web/backend/tests
   ```

2. Rebuild the native move generator locally if you changed `guandan_rs`:

   ```bash
   uv run maturin develop --release --manifest-path ml/src/guandan_rs/Cargo.toml
   ```

3. For ML-result releases, record:
   - config file
   - seed
   - checkpoint path or artifact URL
   - eval command
   - raw eval JSON
   - Glicko-2 leaderboard JSON, if applicable

4. Confirm `README.md`, `docs/CONFIG.md`, and `docs/EVALUATION.md` match the
   released checkpoint/config.

## Suggested GitHub Release Contents

- Source tag.
- Checkpoint artifact or external artifact link.
- Config file used for the run.
- Evaluation commands and raw output.
- Known limitations, including single-seed status and CUDA determinism notes.

## Versioning

Until the package has external consumers, use lightweight semantic tags:

- `v0.x.0`: user-visible training/eval/web changes.
- `v0.x.y`: docs, tests, small bug fixes.

Update `CHANGELOG.md` before creating the tag.
