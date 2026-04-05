#!/usr/bin/env python
"""GuanZero training: Jidan distillation + DMC self-play.

Usage:
  # Local smoke test (~10 min, correctness check only):
  python ml/scripts/train/guanzero.py --distill-games 500 --distill-epochs 1 --selfplay-episodes 200

  # Modal benchmark (measure speed, stops before full training):
  python ml/scripts/train/guanzero.py --distill-games 500 --selfplay-episodes 1000 --benchmark

  # Full Modal run (run via modal launcher, requires --detach):
  python ml/scripts/train/guanzero.py --distill-games 8000 --selfplay-episodes 100000

  # Resume from distilled checkpoint, self-play only:
  python ml/scripts/train/guanzero.py --phase selfplay --checkpoint checkpoints/jidan_distill.pt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from guandan.cards import Rank
from guandan.training.guanzero_network import GuanZeroNetwork
from guandan.training.guanzero_selfplay import ReplayBuffer


_PROD_CKPT = Path("ml/checkpoints/prod_03_29_11_51.pt")


def _get_device(args) -> torch.device:
    if args.device == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(args.device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GuanZero agent")
    parser.add_argument(
        "--phase",
        choices=["distill", "selfplay", "both"],
        default="both",
        help="Training phase (default: both)",
    )
    parser.add_argument(
        "--distill-games",
        type=int,
        default=5000,
        help="Jidan self-play games for Stage 1 distillation (default: 5000)",
    )
    parser.add_argument(
        "--distill-epochs",
        type=int,
        default=3,
        help="Epochs per distillation stage (default: 3)",
    )
    parser.add_argument(
        "--selfplay-episodes",
        type=int,
        default=15000,
        help="Self-play episodes (default: 15000 for local; use 100000 on Modal)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Training batch size (default: 512)",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=5000,
        help="Evaluate every N self-play episodes (default: 5000)",
    )
    parser.add_argument(
        "--eval-games",
        type=int,
        default=200,
        help="Games per eval match (default: 200)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Resume from checkpoint (skip distillation if --phase=selfplay)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="checkpoints",
        help="Directory for saving checkpoints (default: checkpoints/)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: auto, cpu, mps, cuda (default: auto)",
    )
    parser.add_argument(
        "--buffer-capacity",
        type=int,
        default=200_000,
        help="Replay buffer capacity (default: 200000)",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run a small benchmark and print estimated full-run time, then exit",
    )
    args = parser.parse_args()

    device = _get_device(args)
    save_dir = Path(args.save_dir)
    print(f"[GuanZero] device={device}  phase={args.phase}")

    # Build or load network
    net = GuanZeroNetwork().to(device)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        net.load_state_dict(ckpt.get("state_dict", ckpt))
        print(f"[GuanZero] Loaded checkpoint: {args.checkpoint}")
    else:
        n_params = sum(p.numel() for p in net.parameters())
        print(f"[GuanZero] Fresh network  params={n_params:,}")

    level_rank = Rank.TWO

    # ── Distillation ─────────────────────────────────────────────────────────
    if args.phase in ("distill", "both"):
        from guandan.training.guanzero_distill import run_distillation

        distill_path = save_dir / "jidan_distill.pt"
        run_distillation(
            net=net,
            device=device,
            level_rank=level_rank,
            distill_games=args.distill_games,
            distill_epochs=args.distill_epochs,
            vs_games=args.distill_games // 2,
            vs_epochs=args.distill_epochs,
            save_path=distill_path,
        )

    # ── Self-play ─────────────────────────────────────────────────────────────
    if args.phase in ("selfplay", "both"):
        from guandan.training.guanzero_selfplay import train_selfplay

        buffer = ReplayBuffer(capacity=args.buffer_capacity)
        prod_ckpt = _PROD_CKPT if _PROD_CKPT.exists() else None

        if args.benchmark:
            _run_benchmark(net, buffer, device, level_rank, args.batch_size)
            return

        train_selfplay(
            net=net,
            buffer=buffer,
            device=device,
            level_rank=level_rank,
            total_episodes=args.selfplay_episodes,
            batch_size=args.batch_size,
            eval_interval=args.eval_interval,
            eval_games=args.eval_games,
            save_dir=save_dir,
            prod_ckpt_path=prod_ckpt,
        )


def _run_benchmark(net, buffer, device, level_rank, batch_size: int) -> None:
    """Time 100 self-play episodes and extrapolate to full run."""
    from guandan.training.guanzero_selfplay import (
        play_selfplay_episode, train_dmc_step,
    )
    from guandan.game import GuanDanEnv

    print("[Benchmark] Running 100 episodes to measure speed...")
    env = GuanDanEnv(level_rank=level_rank)
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-5)

    N = 100
    t0 = time.perf_counter()
    for i in range(N):
        transitions = play_selfplay_episode(net, env, epsilon=0.1, device=device, level_rank=level_rank)
        for nh, hist, hl, a_enc, G in transitions:
            buffer.push(nh, hist, hl, a_enc, G)
        for _ in range(4):
            train_dmc_step(net, optimizer, buffer, batch_size, device)
        if (i + 1) % 20 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  {i+1}/100 episodes  {elapsed:.1f}s elapsed  ({elapsed/(i+1):.2f}s/ep)")

    total = time.perf_counter() - t0
    sec_per_ep = total / N
    for target_eps in [15_000, 50_000, 100_000]:
        hrs = sec_per_ep * target_eps / 3600
        print(f"  Extrapolated: {target_eps:,} episodes = {hrs:.1f} hrs at {sec_per_ep:.2f}s/ep")

    print(f"\n[Benchmark] STOP — report these numbers before launching full training.")


if __name__ == "__main__":
    main()
