# Project Overview

Guan Dan (掼蛋) RL agent — full game engine + paper-spec Deep Monte Carlo training (LSTM Q-network, distributed actors + learner, optional shared-memory inference server). See `ml/src/guandan/guanzero/` for the active pipeline.

# Repo Structure

```
ml/src/guandan/       — Python package (pip-installable via hatch)
  cards.py, combos.py, game.py  — Core game engine
  agents/             — Rule-based agent hierarchy (random, greedy, heuristic, strategic, yaoji, jidan, …)
  guanzero/           — Active training stack (actor, learner, buffer, encoder, q_network, train)
ml/scripts/           — eval/ and modal/ subfolders (training entry is `python -m guandan.guanzero.train`)
ml/tests/             — Engine + guanzero pytest tests
ml/runs/              — Experiment outputs (gitignored)
ml/data/              — Training data (gitignored)
ml/checkpoints/       — Model checkpoints (gitignored)
web/                  — Next.js frontend + FastAPI backend
web/backend/tests/    — Backend API pytest tests
docs/                 — Architecture and design docs
```

# Common Commands

- Run tests: `uv run pytest`
- Train (local): `PYTHONPATH=ml/src .venv/bin/python -m guandan.guanzero.train --config <config.yaml>`
- Train (Modal GPU): `modal run --detach ml/scripts/modal/train_guanzero_modal.py`
- Evaluate checkpoint: `PYTHONPATH=ml/src python ml/scripts/eval/eval_guanzero.py --checkpoint <path>`
- Head-to-head eval: `PYTHONPATH=ml/src python ml/scripts/eval/bots.py --agent1 <a> --agent2 <b> --games 200`
- WR matrix (Glicko-2): `PYTHONPATH=ml/src python ml/scripts/eval/wr_matrix.py --games 200`

# Training Constraints

- **6-hour hard cutoff**: No local training run should exceed 6 hours wall time. If estimated runtime exceeds this, reduce episodes, increase batch parallelism, or pre-compute expensive operations. Always estimate runtime BEFORE launching.
- Before launching any training run, calculate: `episodes / eps_per_sec / 3600` and verify < 6h.

# Conventions

- **Virtual env**: Use the root `.venv/` for all Python commands (e.g. `.venv/bin/python`, or `uv pip install` which targets it automatically).
- Python 3.10+, dependencies: numpy, torch. Dev: pytest. Optional: modal.
- Snake_case everywhere, `_bot` suffix for agent classes.
- All agents implement `Agent.act(env, player) -> Combo`.
- Training outputs go to `ml/runs/<run_name>/` with `config.json`, `metrics.jsonl`, `train.log`.
- `uv run` is the standard runner; `PYTHONPATH=ml/src` needed when invoking modules directly.
- Guanzero training is configured via YAML (`ml/src/guandan/guanzero/config/*.yaml`).
