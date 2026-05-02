"""Eval a GuanZero checkpoint against a fixed opponent.

Team 0/2 = GuanZero, Team 1/3 = opponent.

Usage:
    PYTHONPATH=ml/src python ml/scripts/eval/eval_guanzero.py \\
        --checkpoint ml/runs/guanzero_m0_20260502_0120/checkpoints/update_00015000.pt \\
        --opponent random --games 200
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from tqdm import tqdm

from guandan.agents import make_agent
from guandan.game import GuanDanEnv
from guandan.guanzero.agent import GuanZeroBot


def run_eval(checkpoint: Path, opponent_name: str, n_games: int, device: str) -> dict:
    gz_bot  = GuanZeroBot.load(checkpoint, device=device)
    opp_bot = make_agent(opponent_name)

    wins = 0  # GuanZero team wins (team 0/2 finishes first)
    for _ in tqdm(range(n_games), desc=f"vs {opponent_name}", unit="game"):
        env = GuanDanEnv()
        env.reset()
        while not env.done:
            p = env.current_player
            if p % 2 == 0:
                action = gz_bot.act(env, p)
            else:
                action = opp_bot.act(env, p)
            env.step(action)
        rewards = env.get_rewards()
        # Win = team 0/2 gets positive reward
        if rewards[0] > 0:
            wins += 1

    win_rate = wins / n_games
    se = math.sqrt(win_rate * (1 - win_rate) / n_games)
    return {"games": n_games, "wins": wins, "win_rate": win_rate, "se": se}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--opponent", default="random",
                   choices=["random", "greedy", "heuristic", "strategic"])
    p.add_argument("--games", type=int, default=200)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    result = run_eval(args.checkpoint, args.opponent, args.games, args.device)
    wr, se = result["win_rate"], result["se"]
    print(f"\nGuanZero vs {args.opponent}: {wr:.1%} ± {se:.1%}  ({result['wins']}/{result['games']})")


if __name__ == "__main__":
    main()
