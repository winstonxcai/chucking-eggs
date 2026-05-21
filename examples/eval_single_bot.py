"""Evaluate a DART checkpoint against one rule-based bot.

Usage:
    uv run python examples/eval_single_bot.py \
      --checkpoint ml/runs/my_run/checkpoints/final.pt \
      --opponent strategic --games 200
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DART evaluation against one bundled bot",
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--opponent", default="strategic")
    parser.add_argument("--games", default=200, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--workers", default=1, type=int)
    parser.add_argument("--lanes", default=1, type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cmd = [
        sys.executable,
        "-m",
        "guandan.scripts.eval.eval_dart",
        "--checkpoint",
        str(args.checkpoint),
        "--opponent",
        args.opponent,
        "--games",
        str(args.games),
        "--seed",
        str(args.seed),
        "--workers",
        str(args.workers),
        "--lanes",
        str(args.lanes),
        "--device",
        args.device,
    ]
    if args.out is not None:
        cmd.extend(["--out", str(args.out)])
    raise SystemExit(subprocess.run(cmd, check=False).returncode)


if __name__ == "__main__":
    main()
