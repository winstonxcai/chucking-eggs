"""Play one complete hand between bundled rule-based bots.

Usage:
    uv run python examples/play_rule_bot_hand.py --agents strategic,yaoji
"""

from __future__ import annotations

import argparse

from guandan.agents import make_agent
from guandan.game import GuanDanEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play one Guan Dan hand")
    parser.add_argument(
        "--agents",
        default="strategic,yaoji",
        help="Comma-separated bot names; one name means all seats, two means teams",
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--max-moves", default=400, type=int)
    return parser.parse_args()


def _seat_names(spec: str) -> list[str]:
    names = [name.strip() for name in spec.split(",") if name.strip()]
    if len(names) == 1:
        return names * 4
    if len(names) == 2:
        return [names[0], names[1], names[0], names[1]]
    if len(names) == 4:
        return names
    raise ValueError("--agents must contain 1, 2, or 4 names")


def main() -> None:
    args = parse_args()
    env = GuanDanEnv(seed=args.seed)
    names = _seat_names(args.agents)
    agents = [make_agent(name) for name in names]

    moves = 0
    while not env.done and moves < args.max_moves:
        seat = env.current_player
        combo = agents[seat].act(env, seat)
        print(f"{moves:03d} seat={seat} bot={names[seat]} move={combo}")
        env.step(combo)
        moves += 1

    print(f"done={env.done} moves={moves} finish_order={env.finish_order}")
    if env.done:
        print(f"rewards={env.get_rewards()}")


if __name__ == "__main__":
    main()
