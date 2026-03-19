#!/usr/bin/env python
"""Self-play fine-tuning from distilled weights.

Usage:
  python scripts/selfplay.py --resume checkpoints/stage2_strategic.pt
  python scripts/selfplay.py --resume checkpoints/stage2_strategic.pt --episodes 2000 --run-name selfplay_smoke
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

import torch
from tqdm import tqdm

from guandan.agents import GreedyBot, HeuristicBot, RandomBot, RLAgentLSTM, StrategicBot
from guandan.cards import Rank
from guandan.game import GuanDanEnv
from guandan.training.game_runner import GameRunner
from guandan.training.q_network import QNetworkLSTM, get_device
from guandan.training.replay import ReplayBuffer
from guandan.training.train import train_step


def _setup_logging(log_path: Path) -> None:
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


def get_epsilon(episode: int, total: int, start: float, end: float, decay_frac: float) -> float:
    decay_episodes = total * decay_frac
    return max(end, start - (start - end) * episode / max(decay_episodes, 1))


def _run_eval(q_lead, q_follow, device, level_rank) -> dict:
    """Ladder eval against all opponents. Returns dict of win rates."""
    rl = RLAgentLSTM(q_lead, q_follow, device, level_rank)

    opponents = [
        ("Random",    RandomBot(),              200),
        ("Greedy",    GreedyBot(level_rank),    200),
        ("Heuristic", HeuristicBot(level_rank), 300),
        ("Strategic", StrategicBot(level_rank), 300),
    ]

    log.info("%-12s | %5s | %6s | %5s %5s %5s", "Opponent", "Games", "WR", "1-2", "1-3", "1-4")
    log.info("-" * 52)

    results = {}
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

        wr = wins / n_games
        log.info(
            "%-12s | %5d | %5.1f%% | %5d %5d %5d",
            name, n_games, 100 * wr, finish_12, finish_13, finish_14,
        )
        results[name.lower()] = wr

    return results


def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        parser = argparse.ArgumentParser(description="Self-play fine-tuning")
        parser.add_argument("--resume", type=str, required=True)
        parser.add_argument("--episodes", type=int, default=20000)
        parser.add_argument("--n-envs", type=int, default=64)
        parser.add_argument("--train-steps", type=int, default=4)
        parser.add_argument("--batch-size", type=int, default=1024)
        parser.add_argument("--lr", type=float, default=3e-5)
        parser.add_argument("--buffer-size", type=int, default=250000)
        parser.add_argument("--eval-interval", type=int, default=2000)
        parser.add_argument("--save-interval", type=int, default=5000)
        parser.add_argument("--epsilon-start", type=float, default=0.15)
        parser.add_argument("--epsilon-end", type=float, default=0.03)
        parser.add_argument("--epsilon-decay-frac", type=float, default=0.80)
        parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
        parser.add_argument("--run-name", type=str, default="selfplay")
        args = parser.parse_args()

    run_dir = Path("runs") / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(run_dir / "train.log")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = get_device()
    t0 = time.time()

    log.info("Device: %s", device)
    log.info("Run dir: %s", run_dir)
    log.info(
        "Episodes: %d | N-envs: %d | Train-steps: %d | LR: %g | Batch: %d",
        args.episodes, args.n_envs, args.train_steps, args.lr, args.batch_size,
    )

    level_rank = Rank.TWO
    q_lead = QNetworkLSTM(lstm_hidden=256, hidden=1024).to(device)
    q_follow = QNetworkLSTM(lstm_hidden=256, hidden=1024).to(device)

    # Load distilled weights
    log.info("Loading checkpoint: %s", args.resume)
    ckpt = torch.load(args.resume, map_location=device, weights_only=True)
    q_lead.load_state_dict(ckpt["lead"])
    q_follow.load_state_dict(ckpt["follow"])

    n_params = sum(p.numel() for p in q_lead.parameters())
    log.info("Network: %s params per head (%s total)", f"{n_params:,}", f"{2 * n_params:,}")

    # Baseline eval
    log.info("=" * 60)
    log.info("BASELINE EVAL (before self-play)")
    log.info("=" * 60)
    baseline = _run_eval(q_lead, q_follow, device, level_rank)
    log.info("Baseline: %.1f%% vs Heuristic, %.1f%% vs Strategic",
             baseline.get("heuristic", 0) * 100, baseline.get("strategic", 0) * 100)

    if args.episodes == 0:
        log.info("Episodes=0, exiting after baseline eval.")
        return

    # Setup
    opt_lead = torch.optim.Adam(q_lead.parameters(), lr=args.lr)
    opt_follow = torch.optim.Adam(q_follow.parameters(), lr=args.lr)
    buffer = ReplayBuffer(capacity=args.buffer_size)

    runner = GameRunner(
        n_envs=args.n_envs,
        q_lead=q_lead,
        q_follow=q_follow,
        device=device,
        level_rank=level_rank,
        epsilon=args.epsilon_start,
    )

    best_wr_heuristic = baseline.get("heuristic", 0)
    episodes_done = 0

    log.info("=" * 60)
    log.info("SELF-PLAY TRAINING")
    log.info("=" * 60)

    with tqdm(total=args.episodes, desc="Self-play", unit="ep") as pbar:
        while episodes_done < args.episodes:
            # Update epsilon
            eps = get_epsilon(
                episodes_done, args.episodes,
                args.epsilon_start, args.epsilon_end, args.epsilon_decay_frac,
            )
            runner.epsilon = eps

            # Generate a batch of episodes
            batch_target = min(args.n_envs, args.episodes - episodes_done)
            for transitions in runner.generate_episodes(batch_target):
                for (state, action, history, hist_len, mc_return) in transitions:
                    buffer.push(state, action, history, hist_len, mc_return)
                episodes_done += 1
                pbar.update(1)

            # Train
            for _ in range(args.train_steps):
                train_step(q_lead, buffer, opt_lead, args.batch_size, device)
                train_step(q_follow, buffer, opt_follow, args.batch_size, device)

            pbar.set_postfix(
                eps=f"{eps:.3f}",
                buf=f"{len(buffer):,}",
            )

            # Eval
            if episodes_done > 0 and episodes_done % args.eval_interval == 0:
                log.info("=" * 60)
                log.info(
                    "EVAL @ episode %d/%d | ε=%.3f | buffer=%d",
                    episodes_done, args.episodes, eps, len(buffer),
                )
                log.info("=" * 60)
                results = _run_eval(q_lead, q_follow, device, level_rank)

                wr_h = results.get("heuristic", 0)
                if wr_h > best_wr_heuristic:
                    best_wr_heuristic = wr_h
                    path = os.path.join(args.checkpoint_dir, "selfplay_best.pt")
                    torch.save({"lead": q_lead.state_dict(), "follow": q_follow.state_dict(),
                                "episode": episodes_done, "wr_heuristic": wr_h}, path)
                    log.info("★ New best: %.1f%% vs Heuristic → %s", wr_h * 100, path)

            # Save periodic checkpoint
            if episodes_done > 0 and episodes_done % args.save_interval == 0:
                path = os.path.join(args.checkpoint_dir, f"selfplay_ep{episodes_done}.pt")
                torch.save({"lead": q_lead.state_dict(), "follow": q_follow.state_dict(),
                            "episode": episodes_done}, path)
                log.info("Saved: %s", path)

    # Final save + eval
    path = os.path.join(args.checkpoint_dir, "selfplay_final.pt")
    torch.save({"lead": q_lead.state_dict(), "follow": q_follow.state_dict(),
                "episode": episodes_done}, path)
    log.info("Saved: %s", path)

    log.info("=" * 60)
    log.info("FINAL EVAL")
    log.info("=" * 60)
    _run_eval(q_lead, q_follow, device, level_rank)

    log.info("Done. Total time: %.0fs (%.1fh)", time.time() - t0, (time.time() - t0) / 3600)
    log.info("Best WR vs Heuristic: %.1f%%", best_wr_heuristic * 100)


if __name__ == "__main__":
    main()
