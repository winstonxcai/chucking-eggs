"""Load a DART checkpoint and choose one legal move from a fresh game.

Usage:
    uv run python examples/dart_inference.py \
      --checkpoint ml/runs/my_run/checkpoints/final.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from guandan.dart.agent import DartBot
from guandan.game import GuanDanEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one DART checkpoint inference step")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", default=0, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = GuanDanEnv(seed=args.seed)
    bot = DartBot.load(args.checkpoint, device=args.device)
    player = env.current_player
    move = bot.act(env, player)
    print(f"player={player} move={move}")


if __name__ == "__main__":
    main()
