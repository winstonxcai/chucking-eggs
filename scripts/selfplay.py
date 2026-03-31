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
import dataclasses
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
from guandan.training.train import play_episode, pretrain_from_heuristic, train_step


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
    opp_name: str | None = None,
) -> None:
    """CPU worker: self-play games with own model copies, push transitions."""
    from guandan.training.q_network import load_compat

    device = torch.device("cpu")
    q_lead = QNetworkLSTM(lstm_hidden=256, hidden=1024).to(device)
    q_follow = QNetworkLSTM(lstm_hidden=256, hidden=1024).to(device)
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
        q_opp_lead = QNetworkLSTM(lstm_hidden=256, hidden=1024).to(device)
        q_opp_follow = QNetworkLSTM(lstm_hidden=256, hidden=1024).to(device)
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
    next_eval_at = args.eval_interval
    next_save_at = args.save_interval

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
                      opp_lead_sd, opp_follow_sd,
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
                    if episodes_done >= next_eval_at:
                        next_eval_at += args.eval_interval
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
                    if episodes_done >= next_save_at:
                        next_save_at += args.save_interval
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


# ── Curriculum ──────────────────────────────────────────────────────────────

@dataclasses.dataclass
class CurriculumStage:
    name: str
    opponent: str | None       # "random", "greedy", "heuristic", "strategic", "competition", or None
    wr_gate: float | None      # None = terminal (no gate check)
    max_episodes: int
    lr: float
    epsilon_start: float
    epsilon_end: float
    eval_opponent: str         # agent name or "ladder" for full 7-bot eval
    eval_interval: int
    eval_games: int
    clear_buffer: bool


CURRICULUM_STAGES: list[CurriculumStage] = [
    CurriculumStage(
        name="random",
        opponent="random",
        wr_gate=0.90,
        max_episodes=3000,
        lr=1e-4,
        epsilon_start=0.30,
        epsilon_end=0.10,
        eval_opponent="random",
        eval_interval=500,
        eval_games=100,
        clear_buffer=True,
    ),
    CurriculumStage(
        name="greedy",
        opponent="greedy",
        wr_gate=0.85,
        max_episodes=5000,
        lr=1e-4,
        epsilon_start=0.25,
        epsilon_end=0.08,
        eval_opponent="greedy",
        eval_interval=1000,
        eval_games=100,
        clear_buffer=True,
    ),
    CurriculumStage(
        name="heuristic",
        opponent="heuristic",
        wr_gate=0.70,
        max_episodes=8000,
        lr=5e-5,
        epsilon_start=0.20,
        epsilon_end=0.05,
        eval_opponent="heuristic",
        eval_interval=1000,
        eval_games=100,
        clear_buffer=True,
    ),
    CurriculumStage(
        name="strategic",
        opponent="strategic",
        wr_gate=0.60,
        max_episodes=8000,
        lr=3e-5,
        epsilon_start=0.15,
        epsilon_end=0.05,
        eval_opponent="strategic",
        eval_interval=1000,
        eval_games=100,
        clear_buffer=True,
    ),
    CurriculumStage(
        name="competition",
        opponent="competition",
        wr_gate=None,
        max_episodes=30000,
        lr=3e-5,
        epsilon_start=0.15,
        epsilon_end=0.03,
        eval_opponent="ladder",
        eval_interval=5000,
        eval_games=200,
        clear_buffer=False,
    ),
]


def _run_single_eval(q_lead, q_follow, device, level_rank,
                     opponent_name: str, n_games: int = 100) -> float:
    """Eval against one named opponent. Returns win rate."""
    rl = RLAgentLSTM(q_lead, q_follow, device, level_rank)
    opp = make_agent(opponent_name, level_rank=level_rank)
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
    return wins / n_games


def _run_stage(
    stage: CurriculumStage,
    runner: GameRunner,
    q_lead, q_follow,
    opt_lead, opt_follow,
    buffer: ReplayBuffer,
    args: argparse.Namespace,
    device,
    level_rank,
    checkpoint_dir: str,
) -> tuple[int, float]:
    """Run one curriculum stage. Returns (episodes_consumed, best_wr_this_stage)."""
    # Update LR (preserves Adam momentum)
    for opt in (opt_lead, opt_follow):
        for g in opt.param_groups:
            g["lr"] = stage.lr

    # Set opponent
    if stage.opponent == "competition":
        runner.opponent = _CompetitionPool(level_rank)
    elif stage.opponent:
        runner.opponent = make_agent(stage.opponent, level_rank=level_rank)
    else:
        runner.opponent = None

    if stage.clear_buffer:
        buffer.clear()

    log.info("=" * 60)
    log.info("STAGE: %s | opp=%s | gate=%s | max=%d | LR=%g | ε %.2f→%.2f",
             stage.name,
             stage.opponent or "self-play",
             f"{stage.wr_gate * 100:.0f}%" if stage.wr_gate is not None else "terminal",
             stage.max_episodes,
             stage.lr,
             stage.epsilon_start,
             stage.epsilon_end)
    log.info("=" * 60)

    episodes_done = 0
    best_wr = 0.0
    next_eval_at = stage.eval_interval

    with tqdm(total=stage.max_episodes, desc=f"Stage:{stage.name}", unit="ep") as pbar:
        while episodes_done < stage.max_episodes:
            eps = get_epsilon(
                episodes_done, stage.max_episodes,
                stage.epsilon_start, stage.epsilon_end, 0.80,
            )
            runner.epsilon = eps

            batch_target = min(args.n_envs, stage.max_episodes - episodes_done)
            for transitions in runner.generate_episodes(batch_target):
                for trans in transitions:
                    buffer.push(*trans)
                episodes_done += 1
                pbar.update(1)

            if len(buffer) >= args.batch_size:
                for _ in range(args.train_steps):
                    train_step(q_lead, buffer, opt_lead, args.batch_size, device)
                    train_step(q_follow, buffer, opt_follow, args.batch_size, device)

            pbar.set_postfix(eps=f"{eps:.3f}", buf=f"{len(buffer):,}")

            if episodes_done >= next_eval_at:
                next_eval_at += stage.eval_interval
                log.info("EVAL [%s] @ %d eps | ε=%.3f", stage.name, episodes_done, eps)

                if stage.eval_opponent == "ladder":
                    results = _run_eval(q_lead, q_follow, device, level_rank, stage.eval_games)
                    wr_h = results.get("heuristic", 0)
                    wr_comp = (results.get("yaoji", 0) + results.get("jidan", 0) + results.get("noai", 0)) / 3
                    wr = (wr_h + wr_comp) / 2
                    log.info("gate=%.1f%% (h=%.1f%%, comp=%.1f%%)", wr * 100, wr_h * 100, wr_comp * 100)
                    if wr > best_wr:
                        best_wr = wr
                        ckpt_dict = {
                            "lead": q_lead.state_dict(),
                            "follow": q_follow.state_dict(),
                            "episode": episodes_done,
                            "wr_gate": wr,
                            "wr_heuristic": wr_h,
                            "wr_competition": wr_comp,
                        }
                        torch.save(ckpt_dict, os.path.join(checkpoint_dir, "selfplay_best.pt"))
                        prod_ts = datetime.now().strftime("%m_%d_%H_%M")
                        prod_path = os.path.join(checkpoint_dir, f"prod_{prod_ts}.pt")
                        torch.save(ckpt_dict, prod_path)
                        log.info("★ New best: gate=%.1f%% → %s", wr * 100, prod_path)
                else:
                    wr = _run_single_eval(
                        q_lead, q_follow, device, level_rank,
                        stage.eval_opponent, stage.eval_games,
                    )
                    log.info("WR vs %s: %.1f%%", stage.eval_opponent, wr * 100)
                    if wr > best_wr:
                        best_wr = wr

                # Check gate
                if stage.wr_gate is not None and best_wr >= stage.wr_gate:
                    log.info("PASSED gate %.0f%% (best=%.1f%%) — advancing",
                             stage.wr_gate * 100, best_wr * 100)
                    return episodes_done, best_wr

    if stage.wr_gate is not None:
        log.warning("FAILED gate %.0f%% (best=%.1f%%) — advancing anyway (max_episodes hit)",
                    stage.wr_gate * 100, best_wr * 100)

    return episodes_done, best_wr


def _train_curriculum(args: argparse.Namespace, device, level_rank) -> None:
    """Curriculum training from random weights. Exp 1 of isolated pipeline."""
    run_dir = Path("runs") / args.run_name
    t0 = time.time()

    log.info("=" * 60)
    log.info("CURRICULUM TRAINING — fresh random weights")
    log.info("Architecture: 417D state, 3×1024 MLP, LSTM 256")
    log.info("=" * 60)

    q_lead = QNetworkLSTM(lstm_hidden=256, hidden=1024).to(device)
    q_follow = QNetworkLSTM(lstm_hidden=256, hidden=1024).to(device)
    n_params = sum(p.numel() for p in q_lead.parameters())
    log.info("Network: %s params per head (%s total)", f"{n_params:,}", f"{2 * n_params:,}")

    opt_lead = torch.optim.Adam(q_lead.parameters(), lr=1e-4)
    opt_follow = torch.optim.Adam(q_follow.parameters(), lr=1e-4)
    buffer = ReplayBuffer(capacity=args.buffer_size)

    # Stage 0: Heuristic imitation pretrain
    log.info("=" * 60)
    log.info("STAGE: pretrain (heuristic imitation, %d games)", args.pretrain_games)
    log.info("=" * 60)
    pretrain_from_heuristic(
        q_lead, q_follow, opt_lead, opt_follow, buffer, device,
        n_games=args.pretrain_games,
        batch_size=args.batch_size,
        train_steps_per_game=args.train_steps,
        eval_interval=500,
        eval_games=100,
        run_dir=run_dir,
    )

    # Validate pretrain WR
    log.info("Post-pretrain eval vs heuristic (100 games):")
    wr_pre = _run_single_eval(q_lead, q_follow, device, level_rank, "heuristic", 100)
    log.info("Pretrain WR vs heuristic: %.1f%%", wr_pre * 100)

    # Create GameRunner (reused across all stages)
    runner = GameRunner(
        n_envs=args.n_envs,
        q_lead=q_lead,
        q_follow=q_follow,
        device=device,
        level_rank=level_rank,
        epsilon=0.20,
    )

    # Stages 1-5
    total_episodes = 0
    for stage in CURRICULUM_STAGES:
        eps_consumed, best_wr = _run_stage(
            stage, runner, q_lead, q_follow, opt_lead, opt_follow,
            buffer, args, device, level_rank, args.checkpoint_dir,
        )
        total_episodes += eps_consumed
        log.info("Stage %s complete: %d eps, best_wr=%.1f%%",
                 stage.name, eps_consumed, best_wr * 100)

    # Final ladder eval
    log.info("=" * 60)
    log.info("FINAL EVAL (full ladder, 200 games each)")
    log.info("=" * 60)
    results = _run_eval(q_lead, q_follow, device, level_rank, 200)
    wr_h = results.get("heuristic", 0)
    wr_comp = (results.get("yaoji", 0) + results.get("jidan", 0) + results.get("noai", 0)) / 3
    log.info("Final gate=%.1f%% (h=%.1f%%, comp=%.1f%%)",
             (wr_h + wr_comp) / 2 * 100, wr_h * 100, wr_comp * 100)
    log.info("Total curriculum time: %.0fs (%.1fh)", time.time() - t0, (time.time() - t0) / 3600)
    log.info("Total RL episodes: %d", total_episodes)

    if wr_comp > 0.55:
        log.info("comp avg %.1f%% > 55%% → Experiment 2 (behavior regulation) is next", wr_comp * 100)
    elif wr_comp <= 0.51:
        log.info("comp avg %.1f%% ≤ 51%% → problem is scale, skip to Modal", wr_comp * 100)
    else:
        log.info("comp avg %.1f%% — borderline, investigate before Experiment 2", wr_comp * 100)


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
    next_eval_at = args.eval_interval
    next_save_at = args.save_interval

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

            if episodes_done >= next_eval_at:
                next_eval_at += args.eval_interval
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

            if episodes_done >= next_save_at:
                next_save_at += args.save_interval
                path = os.path.join(args.checkpoint_dir, f"selfplay_ep{episodes_done}.pt")
                torch.save({"lead": q_lead.state_dict(), "follow": q_follow.state_dict(),
                            "episode": episodes_done}, path)
                log.info("Saved: %s", path)

    return best_wr


# ── Main ───────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        parser = argparse.ArgumentParser(description="Self-play fine-tuning")
        parser.add_argument("--resume", type=str, default=None)
        parser.add_argument("--mode", type=str, default="selfplay",
                            choices=["selfplay", "curriculum"],
                            help="selfplay=fine-tune from checkpoint; curriculum=train from scratch")
        parser.add_argument("--pretrain-games", type=int, default=2000,
                            help="Heuristic imitation games for curriculum stage 0")
        parser.add_argument("--episodes", type=int, default=20000)
        parser.add_argument("--workers", type=int, default=0,
                            help="CPU workers for parallel game gen (0=GameRunner)")
        parser.add_argument("--n-envs", type=int, default=128,
                            help="Parallel envs for GameRunner (when workers=0)")
        parser.add_argument("--train-steps", type=int, default=2)
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
        parser.add_argument("--train-opponent", type=str, default=None,
                            choices=["yaoji", "jidan", "noai", "heuristic", "strategic", "competition"],
                            help="Named opponent for training episodes seats 1&3 (None=self-play)")
        args = parser.parse_args()

    run_dir = Path("runs") / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(run_dir / "train.log")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = get_device()

    log.info("Device: %s", device)
    log.info("Run dir: %s", run_dir)
    log.info("Mode: %s", getattr(args, "mode", "selfplay"))

    level_rank = Rank.TWO

    # Curriculum mode: fresh weights, no checkpoint needed
    if getattr(args, "mode", "selfplay") == "curriculum":
        _train_curriculum(args, device, level_rank)
        return

    # Selfplay mode: requires --resume
    if not args.resume:
        raise ValueError("--resume is required for selfplay mode")

    t0 = time.time()
    n_workers = getattr(args, "workers", 0)
    eval_games = getattr(args, "eval_games", 200)
    no_baseline = getattr(args, "no_baseline", False)

    log.info(
        "Episodes: %d | Workers: %d | Train-steps: %d | LR: %g | Batch: %d | Eval-games: %d",
        args.episodes, n_workers, args.train_steps, args.lr, args.batch_size, eval_games,
    )

    q_lead = QNetworkLSTM(lstm_hidden=256, hidden=1024).to(device)
    q_follow = QNetworkLSTM(lstm_hidden=256, hidden=1024).to(device)

    log.info("Loading checkpoint: %s", args.resume)
    ckpt = torch.load(args.resume, map_location=device, weights_only=True)
    lead_key = "lead_state_dict" if "lead_state_dict" in ckpt else "lead"
    follow_key = "follow_state_dict" if "follow_state_dict" in ckpt else "follow"
    from guandan.training.q_network import load_compat, load_strip_gnn, checkpoint_has_gnn
    if checkpoint_has_gnn(ckpt[lead_key]):
        log.info("Detected GNN-expanded checkpoint — stripping dead GNN columns")
        load_strip_gnn(q_lead, ckpt[lead_key])
        load_strip_gnn(q_follow, ckpt[follow_key])
    else:
        load_compat(q_lead, ckpt[lead_key])
        load_compat(q_follow, ckpt[follow_key])

    n_params = sum(p.numel() for p in q_lead.parameters())
    log.info("Network: %s params per head (%s total)", f"{n_params:,}", f"{2 * n_params:,}")

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
