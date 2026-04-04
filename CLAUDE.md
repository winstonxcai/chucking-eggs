# Project Overview

Guan Dan (掼蛋) RL agent — full game engine + Deep Monte Carlo training with LSTM Q-network and curriculum learning (random -> greedy -> heuristic opponents).

# Repo Structure

```
src/guandan/          — Python package (pip-installable via hatch)
  cards.py, combos.py, game.py  — Core game engine
  agents/             — Agent hierarchy (random, greedy, heuristic, strategic, MC, RL)
  training/           — RL training (encoding, q_network, replay, train)
ml/scripts/           — eval/, train/, util/, modal/ subfolders + run_e2e.sh entry point
ml/tests/             — ML + game engine pytest tests
web/                  — Next.js frontend + FastAPI backend
web/backend/tests/    — Backend API pytest tests
paper/                — Writeups and archived dev logs
runs/                 — Experiment outputs (gitignored)
```

# Common Commands

- Run tests: `uv run pytest`
- Smoke train: `./ml/scripts/run_e2e.sh --train-only`
- Full train: `PYTHONPATH=src python -m guandan.training.train --episodes 30000`
- Quick validation: `PYTHONPATH=src python -m guandan.training.train --quick`
- Evaluate: `PYTHONPATH=src python ml/scripts/eval/checkpoint.py --checkpoint <path> --opponent heuristic --games 500`
- Ladder eval: `PYTHONPATH=src python ml/scripts/eval/ladder.py --checkpoint <path>`
- Modal GPU train: `./ml/scripts/run_e2e.sh --modal`

# Training Constraints

- **6-hour hard cutoff**: No local training run should exceed 6 hours wall time. If estimated runtime exceeds this, reduce episodes, increase batch parallelism, or pre-compute expensive operations. Always estimate runtime BEFORE launching.
- Before launching any training run, calculate: `episodes / eps_per_sec / 3600` and verify < 6h.

# Conventions

- Python 3.10+, dependencies: numpy, torch. Dev: pytest. Optional: modal.
- Snake_case everywhere, `_bot` suffix for agent classes.
- All agents implement `Agent.act(env, player) -> Combo`.
- Training outputs go to `runs/<run_name>/` with `config.json`, `metrics.jsonl`, `train.log`.
- Use `--quick` flag for smoke testing (~4-5 hours on M1 Pro, 8000 episodes).
- Encoding dimensions: state=417, action=160, history_move=83.
- Curriculum stages: random (65% win gate) -> greedy (60% win gate) -> heuristic (terminal).
- `uv run` is the standard runner; `PYTHONPATH=src` needed when invoking modules directly.
- E2E script (`ml/scripts/run_e2e.sh`) is the single entry point with skip flags.
