#!/usr/bin/env python
"""Generate MC teacher data for distillation.

MC bot plays seats 0,2 vs opponent at seats 1,3.
Records all non-trivial MC decisions as (state, action_encs, history, expert_idx).
Parallelizes at the game level (n_workers game processes, each with MC n_workers=1)
to avoid nested Pool overhead.

Usage:
  # Smoke test (fast)
  PYTHONPATH=src python scripts/generate_mc_data.py \\
      --n-games 10 --n-sims 5 --n-workers 2 --output /tmp/mc_test.pt

  # Full run (~15-20min on 8 cores)
  PYTHONPATH=src python scripts/generate_mc_data.py \\
      --n-games 1000 --n-sims 20 --n-workers 8 \\
      --output data/mc_vs_strategic_1000.pt
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import time
from pathlib import Path
from queue import Empty

import numpy as np
import torch
from tqdm import tqdm

from guandan.agents import make_agent
from guandan.agents.monte_carlo_bot import MonteCarloBot
from guandan.cards import ComboType, Rank
from guandan.game import GuanDanEnv
from guandan.training.encoding import encode_action, encode_history, encode_state

log = logging.getLogger(__name__)


# ── Worker (runs in subprocess) ──────────────────────────────────────────────

def _mc_data_worker(
    queue: mp.Queue,
    n_games: int,
    n_sims: int,
    opponent_name: str,
    level_rank: int,
) -> None:
    """Play games with MC(n_workers=1) at seats 0,2 vs opponent at 1,3.

    Uses n_workers=1 to avoid nested multiprocessing (Pool inside subprocess).
    Pushes encoded decisions and GAME_DONE sentinels to queue.
    """
    mc = MonteCarloBot(n_sims=n_sims, n_workers=1, level_rank=level_rank)
    opp = make_agent(opponent_name, level_rank=level_rank)
    env = GuanDanEnv(level_rank)

    for _ in range(n_games):
        env.reset()
        while not env.done:
            p = env.current_player
            legal = env.legal_moves(p)

            if len(legal) <= 1:
                env.step(legal[0])
                continue

            if p in (0, 2):
                # MC team — evaluate all pruned candidates, record scores
                is_leading = env.current_trick is None
                candidates = (
                    mc._prune_lead(legal, env, p)
                    if is_leading
                    else mc._prune_follow(legal, env, p)
                )
                real = [m for m in candidates if m.type != ComboType.PASS]

                if len(real) <= 1:
                    # Trivial: no multi-candidate decision to record
                    env.step(mc.act(env, p))
                    continue

                # Evaluate every pruned candidate (n_sims rollouts each)
                scores = []
                best_score = -float("inf")
                best_action = real[0]
                for move in candidates:
                    s = mc._evaluate_move(env, p, move)
                    scores.append(s)
                    if s > best_score:
                        best_score = s
                        best_action = move

                history, hist_len = encode_history(env, p, level_rank)
                queue.put({
                    "state": encode_state(env, p),
                    "action_encs": np.array(
                        [encode_action(m, env.hands[p], level_rank) for m in candidates],
                        dtype=np.float32,
                    ),
                    "history": history,
                    "hist_len": hist_len,
                    "scores": np.array(scores, dtype=np.float32),
                    "is_leading": is_leading,
                })
                env.step(best_action)
            else:
                env.step(opp.act(env, p))

        queue.put("GAME_DONE")

    queue.put(None)  # sentinel


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MC teacher data")
    parser.add_argument("--n-games",   type=int, default=1000)
    parser.add_argument("--n-sims",    type=int, default=20,
                        help="MC rollouts per candidate (default 20 for speed; 50 for quality)")
    parser.add_argument("--n-workers", type=int, default=8,
                        help="Parallel game processes (each MC uses n_workers=1 internally)")
    parser.add_argument("--opponent",  type=str, default="strategic",
                        choices=["random", "greedy", "heuristic", "strategic", "jidan"])
    parser.add_argument("--output",    type=str, required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    level_rank = Rank.TWO
    n_workers = min(args.n_workers, args.n_games)

    log.info(
        "MC data gen: %d games, n_sims=%d, %d workers, opponent=%s → %s",
        args.n_games, args.n_sims, n_workers, args.opponent, output_path,
    )

    # Distribute games evenly across workers
    games_per_worker = [args.n_games // n_workers] * n_workers
    for i in range(args.n_games % n_workers):
        games_per_worker[i] += 1

    queue: mp.Queue = mp.Queue(maxsize=2000)
    workers = []
    for g in games_per_worker:
        if g == 0:
            continue
        p = mp.Process(
            target=_mc_data_worker,
            args=(queue, g, args.n_sims, args.opponent, level_rank),
        )
        p.start()
        workers.append(p)

    decisions: list[dict] = []
    sentinels = 0
    games_done = 0
    t0 = time.time()

    with tqdm(total=args.n_games, unit="game", desc="Generating") as pbar:
        while sentinels < len(workers):
            try:
                item = queue.get(timeout=120)
            except Empty:
                alive = sum(1 for p in workers if p.is_alive())
                if alive == 0 and sentinels < len(workers):
                    log.warning("All workers exited before sending sentinels")
                    break
                continue

            if item is None:
                sentinels += 1
            elif item == "GAME_DONE":
                games_done += 1
                pbar.update(1)
                elapsed = time.time() - t0
                pbar.set_postfix(
                    decisions=len(decisions),
                    eps=f"{games_done / elapsed:.1f}" if elapsed > 0 else "?",
                )
            else:
                decisions.append(item)

    for p in workers:
        p.join(timeout=10)

    elapsed = time.time() - t0
    lead = sum(1 for d in decisions if d["is_leading"])
    follow = len(decisions) - lead

    log.info(
        "Done. %d decisions from %d games (%.0fs, %.1f games/min)",
        len(decisions), games_done, elapsed, games_done / elapsed * 60,
    )
    log.info("  Lead: %d  Follow: %d  avg candidates: %.1f",
             lead, follow,
             sum(len(d["action_encs"]) for d in decisions) / max(len(decisions), 1))

    torch.save(decisions, output_path)
    log.info("Saved → %s  (%.1f MB)", output_path,
             output_path.stat().st_size / 1e6)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
