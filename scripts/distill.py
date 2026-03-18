#!/usr/bin/env python
"""Supervised distillation: train Q-networks by imitating teacher bots.

Usage:
  python scripts/distill.py --stage 1          # Heuristic only
  python scripts/distill.py --stage 2          # Both stages
  python scripts/distill.py --stage 2 --resume checkpoints/stage1_heuristic.pt
"""

from __future__ import annotations

import argparse
import os

import torch

from guandan.agents import HeuristicBot, RLAgentLSTM, StrategicBot
from guandan.agents import RandomBot, GreedyBot
from guandan.cards import Rank
from guandan.game import GuanDanEnv
from guandan.training.q_network import QNetworkLSTM, get_device
from guandan.training.supervised import generate_teacher_data, train_supervised


def _run_eval(q_lead, q_follow, device, level_rank):
    """Quick ladder eval."""
    rl = RLAgentLSTM(q_lead, q_follow, device, level_rank)

    opponents = [
        ("Random", RandomBot(), 200),
        ("Greedy", GreedyBot(level_rank), 200),
        ("Heuristic", HeuristicBot(level_rank), 300),
        ("Strategic", StrategicBot(level_rank), 300),
    ]

    print(f"\n  {'Opponent':>12} | {'Games':>5} | {'WR':>6}")
    print("  " + "-" * 32)

    for name, opp, n_games in opponents:
        wins = 0
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
        print(f"  {name:>12} | {n_games:>5} | {wins / n_games:>5.1%}")


def main():
    parser = argparse.ArgumentParser(description="Supervised distillation")
    parser.add_argument(
        "--stage", type=int, default=2, choices=[1, 2],
        help="How many stages to run (1=heuristic, 2=heuristic+strategic)",
    )
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--games", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = get_device()
    print(f"Device: {device}")
    level_rank = Rank.TWO

    q_lead = QNetworkLSTM().to(device)
    q_follow = QNetworkLSTM().to(device)
    opt_lead = torch.optim.Adam(q_lead.parameters(), lr=args.lr)
    opt_follow = torch.optim.Adam(q_follow.parameters(), lr=args.lr)

    start_stage = 1

    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        q_lead.load_state_dict(ckpt["lead"])
        q_follow.load_state_dict(ckpt["follow"])
        if "stage1" in args.resume:
            start_stage = 2

    # Stage 1: Distill HeuristicBot
    if start_stage <= 1:
        print("\n" + "=" * 60)
        print("STAGE 1: Distill HeuristicBot")
        print("=" * 60)

        teacher = HeuristicBot(level_rank)
        data = generate_teacher_data(teacher, args.games, level_rank)
        train_supervised(
            q_lead, q_follow, opt_lead, opt_follow,
            data, epochs=args.epochs, batch_size=args.batch_size,
            device=str(device),
        )

        print("\nLadder eval after Stage 1:")
        _run_eval(q_lead, q_follow, device, level_rank)

        path = os.path.join(args.checkpoint_dir, "stage1_heuristic.pt")
        torch.save({"lead": q_lead.state_dict(), "follow": q_follow.state_dict()}, path)
        print(f"\nSaved: {path}")

    # Stage 2: Distill StrategicBot
    if args.stage >= 2 and start_stage <= 2:
        print("\n" + "=" * 60)
        print("STAGE 2: Distill StrategicBot")
        print("=" * 60)

        teacher = StrategicBot(level_rank)

        # Lower LR for fine-tuning
        for pg in opt_lead.param_groups:
            pg["lr"] = args.lr * 0.5
        for pg in opt_follow.param_groups:
            pg["lr"] = args.lr * 0.5

        data = generate_teacher_data(teacher, args.games, level_rank)
        train_supervised(
            q_lead, q_follow, opt_lead, opt_follow,
            data, epochs=args.epochs, batch_size=args.batch_size,
            device=str(device),
        )

        print("\nLadder eval after Stage 2:")
        _run_eval(q_lead, q_follow, device, level_rank)

        path = os.path.join(args.checkpoint_dir, "stage2_strategic.pt")
        torch.save({"lead": q_lead.state_dict(), "follow": q_follow.state_dict()}, path)
        print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
