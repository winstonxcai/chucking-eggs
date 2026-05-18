# Project Overview

Guan Dan (掼蛋) RL agent — full game engine (Python + Rust extension) + DART, a distributed Deep Monte Carlo training pipeline (LSTM Q-network, trick-position-relative shared heads, parallel actors + GPU learner). See `ml/src/guandan/dart/` for the active pipeline.

# Repo Structure

```
ml/src/guandan/         — Python package (hatchling-built)
  cards.py, combos.py, game.py  — Core game engine (Python wrapper)
  agents/               — 13 bots (random → jidan), plus _vendor/ for competition entries
  dart/                 — Active training stack
                            model/    — q_network, encoder, encoding/, checkpoint
                            data/     — buffer, returns, sample_tags
                            runtime/  — actor, worker, learner, inference_server, train
                            utils/    — logging_setup, profiler, schedules, metrics
                            configs/  — YAML training recipes (dart_l4.yaml, dart_mps.yaml, …)
                            agent.py, config.py — public surface
ml/src/guandan_rs/      — Rust extension (pyo3 + maturin) for legal-move generation,
                            episode rollouts, and the role-aware encoder
ml/scripts/eval/        — wr_matrix.py, eval_dart.py, _eval_worker.py, plot_wr_v5_baseline.py
ml/scripts/modal/       — train_dart_modal.py (Modal launcher)
ml/tests/               — Engine + dart pytest tests
ml/runs/                — Experiment outputs (gitignored)
ml/data/, ml/checkpoints/ — Training data + checkpoints (gitignored)
web/                    — Next.js frontend + FastAPI backend
web/backend/tests/      — Backend API pytest tests
docs/                   — Architecture and design docs
```

# Setup

After `uv sync`, the Rust extension must be built into the venv:

```
uv pip install maturin
uv run maturin develop --release --manifest-path ml/src/guandan_rs/Cargo.toml
```

Re-running `uv sync` wipes the editable `guandan_rs` install — redo maturin develop afterward.

# Common Commands

- Run tests: `uv run pytest ml/tests/`
- Train (local MPS): `uv run python -m guandan.dart --config ml/src/guandan/dart/configs/dart_mps.yaml --updates 10000 --run-name my_run`
- Train (Modal GPU, detached): `modal run --detach ml/scripts/modal/train_dart_modal.py --config-path /root/ml/src/guandan/dart/configs/dart_l4.yaml`
- Modal dry-run (no training, image build only): `modal run ml/scripts/modal/train_dart_modal.py --dry-run --config-path /root/ml/src/guandan/dart/configs/dart_l4.yaml`
- Evaluate checkpoint vs one bot: `uv run ml/scripts/eval/eval_dart.py --checkpoint <path> --opponent strategic --games 1000`
- WR matrix (Glicko-2, all bots): `uv run ml/scripts/eval/wr_matrix.py --games 200`

# Training Constraints

- **6-hour hard cutoff** for local runs. Estimate `episodes / eps_per_sec / 3600` before launching and verify < 6h.
- Modal L4 actual cost: ~$2.89/hr (memory file). Full 0→200k baseline is ~30 hours, ~$75 across multiple resumes.

# Conventions

- **Virtual env**: root `.venv/` (created by `uv sync`). All Python invocations should go through `uv run` (or `.venv/bin/python` directly).
- Python 3.10+. Runtime deps: numpy, torch, tqdm, pyyaml, matplotlib. Optional extras: `modal`, `web` (FastAPI stack).
- Snake_case everywhere, `_bot` suffix for agent classes (e.g. `JidanBot` in `jidan_bot.py`).
- All agents implement `Agent.act(env, player) -> Combo`.
- Training outputs go to `ml/runs/<run_name>/` with `config.json`, `metrics_learner.jsonl`, `train.log`, `checkpoints/`.
- DART training is configured via YAML (`ml/src/guandan/dart/configs/*.yaml`); the entry point is `python -m guandan.dart`.
