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

from guandan.q_network import QNetworkLSTM, get_device
from guandan.train import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Guan Dan agent")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--opponent", type=str, default="random", choices=["random", "greedy", "heuristic", "strategic"])
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--lstm-hidden", type=int, default=128)
    parser.add_argument("--mlp-hidden", type=int, default=512)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    q_lead = QNetworkLSTM(lstm_hidden=args.lstm_hidden, hidden=args.mlp_hidden).to(device)
    q_follow = QNetworkLSTM(lstm_hidden=args.lstm_hidden, hidden=args.mlp_hidden).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    q_lead.load_state_dict(ckpt["lead_state_dict"])
    q_follow.load_state_dict(ckpt["follow_state_dict"])
    q_lead.eval()
    q_follow.eval()

    print(f"Loaded checkpoint: {args.checkpoint}")
    if "episode" in ckpt:
        print(f"  Trained for {ckpt['episode']} episodes")

    result = evaluate(q_lead, q_follow, device, n_games=args.games, opponent=args.opponent)
    print(f"\nResults vs {args.opponent} ({args.games} games):")
    print(f"  Win rate:   {result['winrate']:.1%}")
    print(f"  Avg reward: {result['avg_reward']:+.2f}")
    print(f"  Finish 1-2: {result['finish_12']}")
    print(f"  Finish 1-3: {result['finish_13']}")
    print(f"  Finish 1-4: {result['finish_14']}")


if __name__ == "__main__":
    main()
