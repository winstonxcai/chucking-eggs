#!/usr/bin/env python
"""Supervised distillation: train Q-networks by imitating teacher bots.

Usage:
  python scripts/distill.py --stage 1          # Heuristic only
  python scripts/distill.py --stage 2          # Both stages
  python scripts/distill.py --stage 2 --resume checkpoints/stage1_heuristic.pt
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

import torch

from guandan.agents import GreedyBot, HeuristicBot, RandomBot, RLAgentLSTM, StrategicBot
from guandan.cards import Rank
from guandan.game import GuanDanEnv
from guandan.training.q_network import QNetworkLSTM, get_device
from guandan.training.supervised import train_supervised


def _setup_logging(log_path: Path) -> None:
    """Configure root logger: INFO to both file and stderr."""
    fmt = "%(asctime)s %(levelname)s %(message)s"
    datefmt = "%H:%M:%S"

    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        datefmt=datefmt,
        handlers=[
            logging.FileHandler(log_path, mode="a"),
            logging.StreamHandler(),
        ],
    )


log = logging.getLogger(__name__)


def _run_eval(q_lead, q_follow, device, level_rank) -> None:
    """Ladder eval against all opponents."""
    rl = RLAgentLSTM(q_lead, q_follow, device, level_rank)

    opponents = [
        ("Random",    RandomBot(),              200),
        ("Greedy",    GreedyBot(level_rank),    200),
        ("Heuristic", HeuristicBot(level_rank), 300),
        ("Strategic", StrategicBot(level_rank), 300),
    ]

    log.info("%-12s | %5s | %6s | %5s %5s %5s", "Opponent", "Games", "WR", "1-2", "1-3", "1-4")
    log.info("-" * 52)

    for name, opp, n_games in opponents:
        wins = 0
        finish_12 = finish_13 = finish_14 = 0
        env = GuanDanEnv(level_rank)
        for _ in range(n_games):
            env.reset()
            while not env.done:
                p = env.current_player
                if p in (0, 2):
                    env.step(rl.act(env, p))
                else:
                    env.step(opp.act(env, p))
            rewards = env.get_rewards()
            if rewards[0] + rewards[2] > 0:
                wins += 1
            order = env.finish_order
            rl_positions = sorted(order.index(p) + 1 for p in (0, 2) if p in order)
            if rl_positions == [1, 2]:
                finish_12 += 1
            elif 1 in rl_positions and 3 in rl_positions:
                finish_13 += 1
            elif 1 in rl_positions and 4 in rl_positions:
                finish_14 += 1

        log.info(
            "%-12s | %5d | %5.1f%% | %5d %5d %5d",
            name, n_games, 100 * wins / n_games, finish_12, finish_13, finish_14,
        )


def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        parser = argparse.ArgumentParser(description="Supervised distillation")
        parser.add_argument("--stage", type=int, default=2, choices=[1, 2])
        parser.add_argument("--resume", type=str, default=None)
        parser.add_argument("--games", type=int, default=5000)
        parser.add_argument("--epochs", type=int, default=3)
        parser.add_argument("--batch-size", type=int, default=32)
        parser.add_argument("--lr", type=float, default=1e-4)
        parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
        parser.add_argument("--run-name", type=str, default="distill")
        parser.add_argument("--workers", type=int, default=0,
                            help="Parallel CPU workers (0=single-threaded)")
        args = parser.parse_args()

    run_dir = Path("ml/runs") / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(run_dir / "train.log")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = get_device()
    t0 = time.time()

    log.info("Device: %s", device)
    log.info("Run dir: %s", run_dir)
    n_workers = getattr(args, "workers", 0)
    log.info(
        "Stages: 1-%d | Games/stage: %d | Epochs: %d | LR: %g | Workers: %d",
        args.stage, args.games, args.epochs, args.lr, n_workers,
    )

    level_rank = Rank.TWO
    q_lead = QNetworkLSTM(lstm_hidden=256, hidden=1024).to(device)
    q_follow = QNetworkLSTM(lstm_hidden=256, hidden=1024).to(device)
    n_params = sum(p.numel() for p in q_lead.parameters())
    log.info("Network: %s params per head (%s total)", f"{n_params:,}", f"{2 * n_params:,}")

    opt_lead = torch.optim.Adam(q_lead.parameters(), lr=args.lr)
    opt_follow = torch.optim.Adam(q_follow.parameters(), lr=args.lr)

    start_stage = 1
    if args.resume:
        log.info("Resuming from %s", args.resume)
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        q_lead.load_state_dict(ckpt["lead"], strict=False)
        q_follow.load_state_dict(ckpt["follow"], strict=False)
        if "stage1" in args.resume:
            start_stage = 2

    # ── Stage 1: Distill HeuristicBot ──
    if start_stage <= 1:
        log.info("=" * 60)
        log.info("STAGE 1: Distill HeuristicBot")
        log.info("=" * 60)
        t_stage = time.time()

        teacher = HeuristicBot(level_rank)
        log.info("Training on %d heuristic self-play games x %d epochs...", args.games, args.epochs)
        train_supervised(
            q_lead, q_follow, opt_lead, opt_follow,
            teacher, args.games, level_rank,
            epochs=args.epochs, batch_size=args.batch_size, device=str(device),
            n_workers=n_workers,
        )

        log.info("Stage 1 complete (%.0fs)", time.time() - t_stage)
        log.info("Ladder eval after Stage 1:")
        _run_eval(q_lead, q_follow, device, level_rank)

        path = os.path.join(args.checkpoint_dir, "stage1_heuristic.pt")
        torch.save({"lead": q_lead.state_dict(), "follow": q_follow.state_dict()}, path)
        log.info("Saved: %s", path)

    # ── Stage 2: Distill StrategicBot ──
    if args.stage >= 2 and start_stage <= 2:
        log.info("=" * 60)
        log.info("STAGE 2: Distill StrategicBot")
        log.info("=" * 60)
        t_stage = time.time()

        for pg in opt_lead.param_groups:
            pg["lr"] = args.lr * 0.5
        for pg in opt_follow.param_groups:
            pg["lr"] = args.lr * 0.5
        log.info("LR reduced to %g for fine-tuning", args.lr * 0.5)

        teacher = StrategicBot(level_rank)
        log.info("Training on %d strategic self-play games x %d epochs...", args.games, args.epochs)
        train_supervised(
            q_lead, q_follow, opt_lead, opt_follow,
            teacher, args.games, level_rank,
            epochs=args.epochs, batch_size=args.batch_size, device=str(device),
            n_workers=n_workers,
        )

        log.info("Stage 2 complete (%.0fs)", time.time() - t_stage)
        log.info("Ladder eval after Stage 2:")
        _run_eval(q_lead, q_follow, device, level_rank)

        path = os.path.join(args.checkpoint_dir, "stage2_strategic.pt")
        torch.save({"lead": q_lead.state_dict(), "follow": q_follow.state_dict()}, path)
        log.info("Saved: %s", path)

    log.info("Done. Total time: %.0fs (%.1fh)", time.time() - t0, (time.time() - t0) / 3600)


if __name__ == "__main__":
    main()
