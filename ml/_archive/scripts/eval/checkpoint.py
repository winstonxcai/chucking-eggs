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
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from guandan.training.q_network import QNetworkLSTM, get_device
from guandan.training.train import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Guan Dan agent")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--opponent", type=str, default="random", choices=["random", "greedy", "heuristic", "strategic", "monte_carlo", "xingdream", "noai", "lalala"])
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--lstm-hidden", type=int, default=128)
    parser.add_argument("--mlp-hidden", type=int, default=512)
    parser.add_argument("--use-gnn", action="store_true")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    q_lead = QNetworkLSTM(lstm_hidden=args.lstm_hidden, hidden=args.mlp_hidden, use_gnn=args.use_gnn, gnn_out=128).to(device)
    q_follow = QNetworkLSTM(lstm_hidden=args.lstm_hidden, hidden=args.mlp_hidden, use_gnn=args.use_gnn, gnn_out=128).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    lead_key = "lead_state_dict" if "lead_state_dict" in ckpt else "lead"
    follow_key = "follow_state_dict" if "follow_state_dict" in ckpt else "follow"
    q_lead.load_state_dict(ckpt[lead_key], strict=False)
    q_follow.load_state_dict(ckpt[follow_key], strict=False)
    q_lead.eval()
    q_follow.eval()

    print(f"Loaded checkpoint: {args.checkpoint}")
    if "episode" in ckpt:
        print(f"  Trained for {ckpt['episode']} episodes")

    result = evaluate(q_lead, q_follow, device, n_games=args.games, opponent=args.opponent)
    n = args.games
    f12 = result['finish_12']
    f13 = result['finish_13']
    f14 = result['finish_14']
    f23 = result['finish_23']
    f24 = result['finish_24']
    f34 = result['finish_34']
    level_eff = (3*f12 + 2*f13 + f14 - f23 - 2*f24 - 3*f34) / n

    print(f"\nResults vs {args.opponent} ({n} games):")
    print(f"  Win rate:        {result['winrate']:.1%}")
    print(f"  Avg reward:      {result['avg_reward']:+.2f}")
    print(f"  --- Wins ---")
    print(f"  1-2 finish rate: {f12/n:.1%}  ({f12})  +3 levels")
    print(f"  1-3 finish rate: {f13/n:.1%}  ({f13})  +2 levels")
    print(f"  1-4 finish rate: {f14/n:.1%}  ({f14})  +1 level")
    print(f"  --- Losses ---")
    print(f"  2-3 finish rate: {f23/n:.1%}  ({f23})  -1 level")
    print(f"  2-4 finish rate: {f24/n:.1%}  ({f24})  -2 levels")
    print(f"  3-4 finish rate: {f34/n:.1%}  ({f34})  -3 levels")
    print(f"  --- Summary ---")
    print(f"  Net levels/game: {level_eff:+.3f}")


if __name__ == "__main__":
    main()
