#!/usr/bin/env python
"""CLI entry point for Tier 1 (partner visibility) training.

Usage:
    PYTHONPATH=ml/src python ml/scripts/train/train_tier1.py \
        --base-checkpoint ml/checkpoints/prod_03_29_11_51.pt \
        --config ml/configs/tier1.json \
        --run-name tier1_v1

    # Verify expansion only (no training):
    PYTHONPATH=ml/src python ml/scripts/train/train_tier1.py \
        --base-checkpoint ml/checkpoints/prod_03_29_11_51.pt \
        --verify-only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from guandan.training.q_network import QNetworkLSTM, get_device
from guandan.training.visibility.encoding import STATE_DIM_TIER1
from guandan.training.visibility.expand import expand_checkpoint
from guandan.training.visibility.train import evaluate_tier1, run_training


def main():
    parser = argparse.ArgumentParser(description="Tier 1 training with partner visibility")
    parser.add_argument(
        "--base-checkpoint", type=str, required=True,
        help="Path to standard-rules checkpoint to expand",
    )
    parser.add_argument(
        "--config", type=str, default="ml/configs/tier1.json",
        help="Path to training config JSON",
    )
    parser.add_argument(
        "--run-name", type=str, default="tier1",
        help="Name for the run directory under ml/runs/",
    )
    parser.add_argument(
        "--verify-only", action="store_true",
        help="Expand checkpoint and verify WR matches base, then exit",
    )
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    # Load config
    config_path = Path(args.config)
    with open(config_path) as f:
        config = json.load(f)
    print(f"Config: {config_path}")

    # Expand checkpoint
    run_dir = Path("ml/runs") / args.run_name
    expanded_path = run_dir / "checkpoint_expanded.pt"

    if expanded_path.exists():
        print(f"Using existing expanded checkpoint: {expanded_path}")
        ckpt = torch.load(expanded_path, map_location="cpu", weights_only=True)
    else:
        print(f"Expanding {args.base_checkpoint} → {expanded_path}")
        ckpt = expand_checkpoint(args.base_checkpoint, expanded_path)

    # Build Tier 1 networks
    q_lead = QNetworkLSTM(
        d_state=STATE_DIM_TIER1,
        lstm_hidden=config["lstm_hidden"],
        hidden=config["mlp_hidden"],
    ).to(device)
    q_follow = QNetworkLSTM(
        d_state=STATE_DIM_TIER1,
        lstm_hidden=config["lstm_hidden"],
        hidden=config["mlp_hidden"],
    ).to(device)

    q_lead.load_state_dict(ckpt["lead"])
    q_follow.load_state_dict(ckpt["follow"])
    print(f"Loaded expanded weights. d_state={STATE_DIM_TIER1}")

    # Verify expansion
    print("\nVerifying expanded checkpoint vs jidan (500 games)...")
    q_lead.eval()
    q_follow.eval()
    result = evaluate_tier1(q_lead, q_follow, device, n_games=500, opponent="jidan")
    print(
        f"  WR vs jidan: {result['winrate']:.1%} "
        f"(1-2: {result['finish_12']}, 1-3: {result['finish_13']}, "
        f"1-4: {result['finish_14']})"
    )

    if args.verify_only:
        print("\n--verify-only: done.")
        return

    # Train
    run_training(q_lead, q_follow, config, run_dir, device)


if __name__ == "__main__":
    main()
