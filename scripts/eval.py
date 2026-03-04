"""Evaluate a trained model against baselines.

Usage:
    python scripts/eval.py --checkpoint model_final.pt --opponent random --games 1000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from guandan.q_network import QNetwork, get_device
from guandan.train import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Guan Dan agent")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--opponent", type=str, default="random", choices=["random", "heuristic"])
    parser.add_argument("--games", type=int, default=1000)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    q_net = QNetwork().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    q_net.load_state_dict(ckpt["model_state_dict"])
    q_net.eval()

    print(f"Loaded checkpoint: {args.checkpoint}")
    if "episode" in ckpt:
        print(f"  Trained for {ckpt['episode']} episodes")

    wr = evaluate(q_net, device, n_games=args.games, opponent=args.opponent)
    print(f"\nWin rate vs {args.opponent}: {wr:.1%} ({args.games} games)")


if __name__ == "__main__":
    main()
