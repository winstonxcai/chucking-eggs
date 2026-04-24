# ml/_archive/

Snapshot of every ml/ file not used by the website runtime as of **2026-04-24**. Tracked in git; recoverable on every clone.

## Why archive

The website runtime needs only the engine (`cards.py`, `combos.py`, `game.py`), the rule-based agents wired into the 11 difficulty tiers, and the Rust movegen. Every other file under `ml/` — all training scripts, the entire `training/` package (including the unfinished Tier 1 visibility prototype), all RL/LLM/MC agents, all 47 checkpoints (~1.2 GB), the `search/` module, configs, data, and tests for archived modules — is preserved here for history and recoverability.

This was a strict-scope archive: the goal was a clean tree before designing the next perfect-info partner-visibility agent. None of the archived training infrastructure produced a checkpoint that beats Jidan, so nothing of operational value was lost. See `ml/LOGBOOK.md` for the full chronology of attempts and why each failed.

## Layout

```
_archive/
  scripts/{train,eval,modal,util}/   — every previous training/eval entry point
  configs/                           — tier1.json
  src/guandan/training/              — entire training package (encoding, q_network, replay, qmix, gnn, guanzero_*, visibility/*)
  src/guandan/agents/                — impossible_bot, rl_agent, rl_recommender, monte_carlo_bot, guanzero_bot, llm_bot, llm_prompts, search_agent
  src/guandan/search/                — determinize, search, simulate
  src/guandan/rating.py
  checkpoints/                       — all 47 .pt files (~1.2 GB)
  tests/                             — test_hand_gnn, test_guanzero_encoding, test_llm_bot, test_encoding
  data/                              — human_games.jsonl
```

## Restoring a file

```bash
git log --follow ml/_archive/<path>          # find the commit before the move
git checkout HEAD~1 -- ml/<original-path>    # restore from prior history
# or just `git mv` it back if the imports still resolve
```

Specific resurrections likely for the next plan:
- `training/visibility/encoding.py` (the 477-dim and 480-dim partner-hand encoders)
- `training/replay.py` (replay buffer)
- `training/game_runner.py` (batched-inference episode collection)
- `eval/checkpoint.py` and `eval/wr_matrix.py`

## What is NOT here

The current website runtime files stay in their original locations:

- `ml/src/guandan/__init__.py`, `cards.py`, `combos.py`, `game.py`
- `ml/src/guandan/agents/{base, greedy, random, heuristic, strategic, xingdream, jidan, yaoji, noai, lalala, hulalala, liuzha, wjsd, ez}_bot.py` + `_vendor/`
- `ml/src/guandan_rs/` (Rust movegen)
- `ml/tests/test_{cards,combos,game,agents,heuristic,rust_fuzz}.py`
- `ml/scripts/run_e2e.sh`
- `ml/LOGBOOK.md`
