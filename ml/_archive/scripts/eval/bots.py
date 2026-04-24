"""Evaluate two heuristic agents head-to-head (no RL checkpoint needed)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from tqdm import tqdm

from guandan.agents import make_agent
from guandan.game import GuanDanEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Head-to-head agent eval")
    parser.add_argument("--agent1", required=True, help="Team 0/2 agent")
    parser.add_argument("--agent2", required=True, help="Team 1/3 agent")
    parser.add_argument("--games", type=int, default=500)
    args = parser.parse_args()

    a1 = make_agent(args.agent1)
    a2 = make_agent(args.agent2)
    env = GuanDanEnv()
    wins = 0

    for _ in tqdm(range(args.games), desc=f"{args.agent1} vs {args.agent2}"):
        env.reset()
        while not env.done:
            p = env.current_player
            agent = a1 if p % 2 == 0 else a2
            env.step(agent.act(env, p))
        if env.finish_order[0] % 2 == 0:
            wins += 1

    print(f"\n{args.agent1} win rate vs {args.agent2}: {wins / args.games:.1%} ({wins}/{args.games})")


if __name__ == "__main__":
    main()
