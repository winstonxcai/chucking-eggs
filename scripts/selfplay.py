#!/usr/bin/env python
"""Self-play fine-tuning from distilled weights.

Usage:
  # Single-threaded (GameRunner batched inference):
  python scripts/selfplay.py --resume checkpoints/selfplay_best.pt

  # Multi-process (CPU workers, lean eval):
  python scripts/selfplay.py --resume checkpoints/selfplay_best.pt --workers 10 --eval-games 100
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
import time
from pathlib import Path
from queue import Empty

import numpy as np
import torch
from datetime import datetime
from tqdm import tqdm

from guandan.agents import (GreedyBot, HeuristicBot, JidanBot, NoAIBot,
                             RandomBot, RLAgentLSTM, StrategicBot, YaojiBot,
                             make_agent)
from guandan.cards import Rank
from guandan.game import GuanDanEnv
from guandan.training.game_runner import GameRunner
from guandan.training.q_network import QNetworkLSTM, get_device
from guandan.training.replay import ReplayBuffer
from guandan.training.train import play_episode, train_step


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


# ── Multi-process worker ───────────────────────────────────────────────────

def _selfplay_worker(
    queue: mp.Queue,
    lead_sd: dict,
    follow_sd: dict,
    level_rank,
    n_games: int,
    epsilon: float,
    opp_lead_sd: dict | None = None,
    opp_follow_sd: dict | None = None,
    use_gnn: bool = False,
    opp_name: str | None = None,
) -> None:
    """CPU worker: self-play games with own model copies, push transitions."""
    from guandan.training.q_network import load_compat

    device = torch.device("cpu")
    q_lead = QNetworkLSTM(lstm_hidden=256, hidden=1024, use_gnn=use_gnn).to(device)
    q_follow = QNetworkLSTM(lstm_hidden=256, hidden=1024, use_gnn=use_gnn).to(device)
    load_compat(q_lead, lead_sd)
    load_compat(q_follow, follow_sd)
    q_lead.eval()
    q_follow.eval()

    # Opponent: named agent > pool checkpoint > self-play
    import random as _random
    opponent = None
    if opp_name == "competition":
        opponent = make_agent(_random.choice(["yaoji", "jidan", "noai"]), level_rank)
    elif opp_name:
        opponent = make_agent(opp_name, level_rank)
    elif opp_lead_sd is not None:
        q_opp_lead = QNetworkLSTM(lstm_hidden=256, hidden=1024, use_gnn=use_gnn).to(device)
        q_opp_follow = QNetworkLSTM(lstm_hidden=256, hidden=1024, use_gnn=use_gnn).to(device)
        load_compat(q_opp_lead, opp_lead_sd)
        load_compat(q_opp_follow, opp_follow_sd)
        q_opp_lead.eval()
        q_opp_follow.eval()
        opponent = RLAgentLSTM(q_opp_lead, q_opp_follow, device, level_rank)

    env = GuanDanEnv(level_rank)

    for _ in range(n_games):
        transitions = play_episode(env, q_lead, q_follow, epsilon, device, opponent=opponent)
        for t in transitions:
            queue.put(t)
        queue.put("GAME_DONE")
    queue.put(None)  # sentinel


# ── Eval ───────────────────────────────────────────────────────────────────

def _run_eval(q_lead, q_follow, device, level_rank, n_games_per_opp: int = 200) -> dict:
    """Ladder eval against all opponents. Returns dict of win rates."""
    rl = RLAgentLSTM(q_lead, q_follow, device, level_rank)

    opponents = [
        ("Random",    RandomBot(),              min(n_games_per_opp, 200)),
        ("Greedy",    GreedyBot(level_rank),    min(n_games_per_opp, 200)),
        ("Heuristic", HeuristicBot(level_rank), n_games_per_opp),
        ("Strategic", StrategicBot(level_rank), n_games_per_opp),
        ("Yaoji",     YaojiBot(level_rank),     n_games_per_opp),
        ("Jidan",     JidanBot(level_rank),     n_games_per_opp),
        ("NoAI",      NoAIBot(level_rank),      n_games_per_opp),
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


# ── Training loops ─────────────────────────────────────────────────────────

def _train_multiprocess(q_lead, q_follow, opt_lead, opt_follow, buffer,
                        args, device, level_rank, best_wr):
    """Multi-process: CPU workers generate self-play, GPU trains."""
    from guandan.training.opponent_pool import OpponentPool

    n_workers = args.workers
    train_every = 64  # train after every ~64 episodes
    sync_interval = getattr(args, "sync_interval", 20000)  # weight sync interval
    pool_add_interval = getattr(args, "pool_add_interval", 2000)

    pool = OpponentPool(
        max_size=getattr(args, "pool_size", 10),
        self_play_prob=1.0 - getattr(args, "pool_prob", 0.0),
    )

    episodes_done = 0
    last_sync = 0

    while episodes_done < args.episodes:
        # Determine batch size for this round
        batch_episodes = min(args.episodes - episodes_done, sync_interval - (episodes_done - last_sync))
        eps = get_epsilon(
            episodes_done, args.episodes,
            args.epsilon_start, args.epsilon_end, args.epsilon_decay_frac,
        )

        # Get current weights for workers
        lead_sd = {k: v.cpu() for k, v in q_lead.state_dict().items()}
        follow_sd = {k: v.cpu() for k, v in q_follow.state_dict().items()}

        # Add to opponent pool periodically
        if episodes_done % pool_add_interval == 0 and episodes_done > 0:
            pool.add(q_lead.state_dict(), q_follow.state_dict())
            log.info("Added weights to opponent pool (size=%d)", len(pool))

        # Sample pool opponent for this batch (all workers use same opponent)
        opp_sample = pool.sample()
        opp_lead_sd = opp_sample[0] if opp_sample else None
        opp_follow_sd = opp_sample[1] if opp_sample else None

        # Distribute games across workers
        games_per_worker = [batch_episodes // n_workers] * n_workers
        for i in range(batch_episodes % n_workers):
            games_per_worker[i] += 1

        queue = mp.Queue(maxsize=2000)
        workers = []
        for i in range(n_workers):
            if games_per_worker[i] == 0:
                continue
            p = mp.Process(
                target=_selfplay_worker,
                args=(queue, lead_sd, follow_sd, level_rank, games_per_worker[i], eps,
                      opp_lead_sd, opp_follow_sd, getattr(args, 'use_gnn', False),
                      getattr(args, 'train_opponent', None)),
            )
            p.start()
            workers.append(p)

        sentinels = 0
        batch_ep_done = 0

        with tqdm(total=batch_episodes, desc=f"Self-play (ε={eps:.3f})", unit="ep", leave=False) as pbar:
            while sentinels < len(workers):
                try:
                    item = queue.get(timeout=120)
                except Empty:
                    alive = sum(1 for p in workers if p.is_alive())
                    if alive == 0:
                        log.warning("All workers died")
                        break
                    continue

                if item is None:
                    sentinels += 1
                    continue
                if item == "GAME_DONE":
                    batch_ep_done += 1
                    episodes_done += 1
                    pbar.update(1)
                    pbar.set_postfix(buf=f"{len(buffer):,}")

                    # Train periodically
                    if batch_ep_done % train_every == 0 and len(buffer) >= args.batch_size:
                        for _ in range(args.train_steps):
                            train_step(q_lead, buffer, opt_lead, args.batch_size, device)
                            train_step(q_follow, buffer, opt_follow, args.batch_size, device)

                    # Eval
                    if episodes_done > 0 and episodes_done % args.eval_interval == 0:
                        log.info("=" * 60)
                        log.info("EVAL @ episode %d/%d | ε=%.3f | buffer=%d",
                                 episodes_done, args.episodes, eps, len(buffer))
                        log.info("=" * 60)
                        eval_games = getattr(args, "eval_games", 200)
                        results = _run_eval(q_lead, q_follow, device, level_rank, eval_games)
                        wr_h = results.get("heuristic", 0)
                        wr_comp = (results.get("yaoji", 0) + results.get("jidan", 0) + results.get("noai", 0)) / 3
                        wr_gate = (wr_h + wr_comp) / 2
                        if wr_gate > best_wr:
                            best_wr = wr_gate
                            ckpt_dict = {"lead": q_lead.state_dict(), "follow": q_follow.state_dict(),
                                         "episode": episodes_done, "wr_gate": wr_gate,
                                         "wr_heuristic": wr_h, "wr_competition": wr_comp}
                            path = os.path.join(args.checkpoint_dir, "selfplay_best.pt")
                            torch.save(ckpt_dict, path)
                            prod_ts = datetime.now().strftime("%m_%d_%H_%M")
                            prod_path = os.path.join(args.checkpoint_dir, f"prod_{prod_ts}.pt")
                            torch.save(ckpt_dict, prod_path)
                            log.info("★ New best: gate=%.1f%% (h=%.1f%%, comp=%.1f%%) → %s",
                                     wr_gate * 100, wr_h * 100, wr_comp * 100, prod_path)

                    # Save checkpoint
                    if episodes_done > 0 and episodes_done % args.save_interval == 0:
                        path = os.path.join(args.checkpoint_dir, f"selfplay_ep{episodes_done}.pt")
                        torch.save({"lead": q_lead.state_dict(), "follow": q_follow.state_dict(),
                                    "episode": episodes_done}, path)
                        log.info("Saved: %s", path)

                    continue

                # Push transition to buffer
                buffer.push(*item)

        for p in workers:
            p.join(timeout=10)

        last_sync = episodes_done
        log.info("Weight sync @ episode %d", episodes_done)

        # Add to pool after weight sync
        if episodes_done % pool_add_interval == 0 and episodes_done > 0 and len(pool) < pool.max_size:
            pool.add(q_lead.state_dict(), q_follow.state_dict())
            log.info("Added weights to opponent pool (size=%d)", len(pool))

    # Final train flush
    if len(buffer) >= args.batch_size:
        for _ in range(args.train_steps * 4):
            train_step(q_lead, buffer, opt_lead, args.batch_size, device)
            train_step(q_follow, buffer, opt_follow, args.batch_size, device)

    return best_wr


class _CompetitionPool:
    """Randomly picks from {Yaoji, Jidan, NoAI} on each act() call."""
    import random as _r

    def __init__(self, level_rank):
        self._agents = [YaojiBot(level_rank), JidanBot(level_rank), NoAIBot(level_rank)]

    def act(self, env, player):
        return self._r.choice(self._agents).act(env, player)


def _train_gamerunner(q_lead, q_follow, opt_lead, opt_follow, buffer,
                      args, device, level_rank, best_wr):
    """Single-threaded GameRunner with batched GPU inference."""
    runner = GameRunner(
        n_envs=args.n_envs,
        q_lead=q_lead,
        q_follow=q_follow,
        device=device,
        level_rank=level_rank,
        epsilon=args.epsilon_start,
    )
    runner.team_spirit = getattr(args, 'team_spirit', 0.0)

    train_opp_name = getattr(args, "train_opponent", None)
    if train_opp_name == "competition":
        runner.opponent = _CompetitionPool(level_rank)
    elif train_opp_name:
        runner.opponent = make_agent(train_opp_name, level_rank=level_rank)
    if runner.opponent is not None:
        log.info("Training opponent: %s", train_opp_name)

    episodes_done = 0

    with tqdm(total=args.episodes, desc="Self-play", unit="ep") as pbar:
        while episodes_done < args.episodes:
            eps = get_epsilon(
                episodes_done, args.episodes,
                args.epsilon_start, args.epsilon_end, args.epsilon_decay_frac,
            )
            runner.epsilon = eps

            batch_target = min(args.n_envs, args.episodes - episodes_done)
            for transitions in runner.generate_episodes(batch_target):
                for trans in transitions:
                    buffer.push(*trans)
                episodes_done += 1
                pbar.update(1)

            for _ in range(args.train_steps):
                train_step(q_lead, buffer, opt_lead, args.batch_size, device)
                train_step(q_follow, buffer, opt_follow, args.batch_size, device)

            pbar.set_postfix(eps=f"{eps:.3f}", buf=f"{len(buffer):,}")

            if episodes_done > 0 and episodes_done % args.eval_interval == 0:
                log.info("=" * 60)
                log.info("EVAL @ episode %d/%d | ε=%.3f | buffer=%d",
                         episodes_done, args.episodes, eps, len(buffer))
                log.info("=" * 60)
                eval_games = getattr(args, "eval_games", 200)
                results = _run_eval(q_lead, q_follow, device, level_rank, eval_games)
                wr_h = results.get("heuristic", 0)
                wr_comp = (results.get("yaoji", 0) + results.get("jidan", 0) + results.get("noai", 0)) / 3
                wr_gate = (wr_h + wr_comp) / 2
                if wr_gate > best_wr:
                    best_wr = wr_gate
                    ckpt_dict = {"lead": q_lead.state_dict(), "follow": q_follow.state_dict(),
                                 "episode": episodes_done, "wr_gate": wr_gate,
                                 "wr_heuristic": wr_h, "wr_competition": wr_comp}
                    path = os.path.join(args.checkpoint_dir, "selfplay_best.pt")
                    torch.save(ckpt_dict, path)
                    prod_ts = datetime.now().strftime("%m_%d_%H_%M")
                    prod_path = os.path.join(args.checkpoint_dir, f"prod_{prod_ts}.pt")
                    torch.save(ckpt_dict, prod_path)
                    log.info("★ New best: gate=%.1f%% (h=%.1f%%, comp=%.1f%%) → %s",
                             wr_gate * 100, wr_h * 100, wr_comp * 100, prod_path)

            if episodes_done > 0 and episodes_done % args.save_interval == 0:
                path = os.path.join(args.checkpoint_dir, f"selfplay_ep{episodes_done}.pt")
                torch.save({"lead": q_lead.state_dict(), "follow": q_follow.state_dict(),
                            "episode": episodes_done}, path)
                log.info("Saved: %s", path)

    return best_wr


# ── Main ───────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        parser = argparse.ArgumentParser(description="Self-play fine-tuning")
        parser.add_argument("--resume", type=str, required=True)
        parser.add_argument("--episodes", type=int, default=20000)
        parser.add_argument("--workers", type=int, default=0,
                            help="CPU workers for parallel game gen (0=GameRunner)")
        parser.add_argument("--n-envs", type=int, default=64,
                            help="Parallel envs for GameRunner (when workers=0)")
        parser.add_argument("--train-steps", type=int, default=4)
        parser.add_argument("--batch-size", type=int, default=1024)
        parser.add_argument("--lr", type=float, default=3e-5)
        parser.add_argument("--buffer-size", type=int, default=250000)
        parser.add_argument("--eval-interval", type=int, default=2000)
        parser.add_argument("--eval-games", type=int, default=200,
                            help="Games per opponent during eval")
        parser.add_argument("--save-interval", type=int, default=5000)
        parser.add_argument("--epsilon-start", type=float, default=0.15)
        parser.add_argument("--epsilon-end", type=float, default=0.03)
        parser.add_argument("--epsilon-decay-frac", type=float, default=0.80)
        parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
        parser.add_argument("--run-name", type=str, default="selfplay")
        parser.add_argument("--team-spirit", type=float, default=0.0,
                            help="Mix partner reward into mc_return (0=individual, 0.5=half-shared, 1=fully shared)")
        parser.add_argument("--no-baseline", action="store_true",
                            help="Skip baseline eval at startup")
        parser.add_argument("--pool-size", type=int, default=10,
                            help="Max checkpoints in opponent pool")
        parser.add_argument("--pool-prob", type=float, default=0.0,
                            help="Prob of sampling a pool opponent (0=disabled, 0.3=recommended)")
        parser.add_argument("--pool-add-interval", type=int, default=2000,
                            help="Add current weights to pool every N episodes")
        parser.add_argument("--use-gnn", action="store_true",
                            help="Enable GNN hand structure encoding")
        parser.add_argument("--train-opponent", type=str, default=None,
                            choices=["yaoji", "jidan", "noai", "heuristic", "strategic", "competition"],
                            help="Named opponent for training episodes seats 1&3 (None=self-play)")
        args = parser.parse_args()

    run_dir = Path("runs") / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(run_dir / "train.log")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = get_device()
    t0 = time.time()

    n_workers = getattr(args, "workers", 0)
    eval_games = getattr(args, "eval_games", 200)
    no_baseline = getattr(args, "no_baseline", False)

    log.info("Device: %s", device)
    log.info("Run dir: %s", run_dir)
    log.info(
        "Episodes: %d | Workers: %d | Train-steps: %d | LR: %g | Batch: %d | Eval-games: %d",
        args.episodes, n_workers, args.train_steps, args.lr, args.batch_size, eval_games,
    )

    level_rank = Rank.TWO
    use_gnn = getattr(args, "use_gnn", False)
    q_lead = QNetworkLSTM(lstm_hidden=256, hidden=1024, use_gnn=use_gnn).to(device)
    q_follow = QNetworkLSTM(lstm_hidden=256, hidden=1024, use_gnn=use_gnn).to(device)

    log.info("Loading checkpoint: %s", args.resume)
    ckpt = torch.load(args.resume, map_location=device, weights_only=True)
    from guandan.training.q_network import load_compat, load_with_gnn_expansion
    if use_gnn:
        log.info("GNN enabled — loading with MLP expansion (zero-init GNN columns)")
        load_with_gnn_expansion(q_lead, ckpt["lead"])
        load_with_gnn_expansion(q_follow, ckpt["follow"])
    else:
        load_compat(q_lead, ckpt["lead"])
        load_compat(q_follow, ckpt["follow"])

    n_params = sum(p.numel() for p in q_lead.parameters())
    log.info("Network: %s params per head (%s total)", f"{n_params:,}", f"{2 * n_params:,}")
    if use_gnn:
        log.info("GNN enabled: hand structure graph with amortized inference")

    # Baseline eval
    best_wr = 0.0
    if not no_baseline:
        log.info("=" * 60)
        log.info("BASELINE EVAL (before self-play)")
        log.info("=" * 60)
        baseline = _run_eval(q_lead, q_follow, device, level_rank, eval_games)
        wr_h0 = baseline.get("heuristic", 0)
        wr_comp0 = (baseline.get("yaoji", 0) + baseline.get("jidan", 0) + baseline.get("noai", 0)) / 3
        best_wr = (wr_h0 + wr_comp0) / 2
        log.info("Baseline gate=%.1f%% (h=%.1f%%, comp=%.1f%%, strategic=%.1f%%)",
                 best_wr * 100, wr_h0 * 100, wr_comp0 * 100, baseline.get("strategic", 0) * 100)

    if args.episodes == 0:
        log.info("Episodes=0, exiting after baseline eval.")
        return

    # Setup
    opt_lead = torch.optim.Adam(q_lead.parameters(), lr=args.lr)
    opt_follow = torch.optim.Adam(q_follow.parameters(), lr=args.lr)
    buffer = ReplayBuffer(capacity=args.buffer_size)

    log.info("=" * 60)
    log.info("SELF-PLAY TRAINING (%s)", "multi-process" if n_workers > 0 else "GameRunner")
    log.info("=" * 60)

    if n_workers > 0:
        best_wr = _train_multiprocess(
            q_lead, q_follow, opt_lead, opt_follow, buffer,
            args, device, level_rank, best_wr,
        )
    else:
        best_wr = _train_gamerunner(
            q_lead, q_follow, opt_lead, opt_follow, buffer,
            args, device, level_rank, best_wr,
        )

    # Final save + eval
    path = os.path.join(args.checkpoint_dir, "selfplay_final.pt")
    torch.save({"lead": q_lead.state_dict(), "follow": q_follow.state_dict()}, path)
    log.info("Saved: %s", path)

    log.info("=" * 60)
    log.info("FINAL EVAL")
    log.info("=" * 60)
    _run_eval(q_lead, q_follow, device, level_rank, eval_games)

    log.info("Done. Total time: %.0fs (%.1fh)", time.time() - t0, (time.time() - t0) / 3600)
    log.info("Best gate WR: %.1f%%", best_wr * 100)


if __name__ == "__main__":
    main()
