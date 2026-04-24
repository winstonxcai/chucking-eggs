#!/usr/bin/env python
"""Benchmark: GameRunner (batched GPU) vs multiprocess workers (parallel CPU).

Tests pure episode generation throughput — no training steps, no eval.
Use this to decide which path to use for the 100K self-play run.

Usage:
  PYTHONPATH=src python scripts/benchmark_selfplay.py \\
      --resume checkpoints/newrewards_ts05.pt \\
      --n-episodes 200 --n-workers 4
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from queue import Empty

import torch

from guandan.cards import Rank
from guandan.training.game_runner import GameRunner
from guandan.training.q_network import QNetworkLSTM, get_device
from guandan.training.train import play_episode
from guandan.game import GuanDanEnv


# ── Worker (mirrors selfplay.py _selfplay_worker) ───────────────────────────

def _bench_worker(
    queue: mp.Queue,
    lead_sd: dict,
    follow_sd: dict,
    level_rank: int,
    n_games: int,
    epsilon: float,
) -> None:
    device = torch.device("cpu")
    q_lead  = QNetworkLSTM(lstm_hidden=256, hidden=1024).to(device)
    q_follow = QNetworkLSTM(lstm_hidden=256, hidden=1024).to(device)
    q_lead.load_state_dict(lead_sd, strict=False)
    q_follow.load_state_dict(follow_sd, strict=False)
    q_lead.eval()
    q_follow.eval()

    env = GuanDanEnv(level_rank)
    for _ in range(n_games):
        play_episode(env, q_lead, q_follow, epsilon, device)
        queue.put("GAME_DONE")
    queue.put(None)


# ── Benchmark helpers ────────────────────────────────────────────────────────

def bench_gamerunner(q_lead, q_follow, device, level_rank, n_episodes, n_envs) -> float:
    """Time GameRunner episode generation. Returns episodes/sec."""
    runner = GameRunner(
        n_envs=n_envs,
        q_lead=q_lead,
        q_follow=q_follow,
        device=device,
        level_rank=level_rank,
        epsilon=0.1,
    )

    t0 = time.time()
    count = 0
    for _ in runner.generate_episodes(n_episodes):
        count += 1
    elapsed = time.time() - t0
    return count / elapsed


def bench_workers(q_lead, q_follow, level_rank, n_episodes, n_workers, epsilon=0.1) -> float:
    """Time multiprocess worker episode generation. Returns episodes/sec."""
    lead_sd  = {k: v.cpu() for k, v in q_lead.state_dict().items()}
    follow_sd = {k: v.cpu() for k, v in q_follow.state_dict().items()}

    n_workers = min(n_workers, n_episodes)
    games_per_worker = [n_episodes // n_workers] * n_workers
    for i in range(n_episodes % n_workers):
        games_per_worker[i] += 1

    queue: mp.Queue = mp.Queue(maxsize=1000)
    workers = []
    for g in games_per_worker:
        if g == 0:
            continue
        p = mp.Process(
            target=_bench_worker,
            args=(queue, lead_sd, follow_sd, level_rank, g, epsilon),
        )
        p.start()
        workers.append(p)

    t0 = time.time()
    sentinels = 0
    count = 0

    while sentinels < len(workers):
        try:
            item = queue.get(timeout=120)
        except Empty:
            alive = sum(1 for p in workers if p.is_alive())
            if alive == 0:
                break
            continue
        if item is None:
            sentinels += 1
        elif item == "GAME_DONE":
            count += 1

    elapsed = time.time() - t0

    for p in workers:
        p.join(timeout=10)

    return count / elapsed


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark GameRunner vs workers")
    parser.add_argument("--resume",     type=str, required=True)
    parser.add_argument("--n-episodes", type=int, default=200,
                        help="Episodes per benchmark run")
    parser.add_argument("--n-workers",  type=int, default=4,
                        help="Number of CPU workers to benchmark")
    parser.add_argument("--n-envs",     type=int, default=128,
                        help="Parallel envs for GameRunner")
    parser.add_argument("--skip-gamerunner", action="store_true")
    parser.add_argument("--skip-workers",    action="store_true")
    args = parser.parse_args()

    device = get_device()
    level_rank = Rank.TWO

    print(f"Device: {device}")
    print(f"Benchmark: {args.n_episodes} episodes each\n")

    q_lead  = QNetworkLSTM(lstm_hidden=256, hidden=1024).to(device)
    q_follow = QNetworkLSTM(lstm_hidden=256, hidden=1024).to(device)
    ckpt = torch.load(args.resume, map_location=device, weights_only=True)
    lead_key = "lead_state_dict" if "lead_state_dict" in ckpt else "lead"
    follow_key = "follow_state_dict" if "follow_state_dict" in ckpt else "follow"
    from guandan.training.q_network import load_strip_gnn, checkpoint_has_gnn, load_compat
    if checkpoint_has_gnn(ckpt[lead_key]):
        print("Detected GNN-expanded checkpoint — stripping GNN columns")
        load_strip_gnn(q_lead, ckpt[lead_key])
        load_strip_gnn(q_follow, ckpt[follow_key])
    else:
        load_compat(q_lead, ckpt[lead_key])
        load_compat(q_follow, ckpt[follow_key])
    q_lead.eval()
    q_follow.eval()

    results: dict[str, float] = {}

    # GameRunner
    if not args.skip_gamerunner:
        print(f"[1/2] GameRunner (n_envs={args.n_envs})... ", end="", flush=True)
        eps = bench_gamerunner(q_lead, q_follow, device, level_rank,
                               args.n_episodes, args.n_envs)
        results["gamerunner"] = eps
        print(f"{eps:.1f} eps/sec")

    # Workers
    if not args.skip_workers:
        print(f"[2/2] Workers (n={args.n_workers})... ", end="", flush=True)
        eps = bench_workers(q_lead, q_follow, level_rank,
                            args.n_episodes, args.n_workers)
        results["workers"] = eps
        print(f"{eps:.1f} eps/sec")

    # Summary
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    for name, val in results.items():
        print(f"  {name:20s}: {val:.1f} eps/sec")

    if "gamerunner" in results and "workers" in results:
        speedup = results["workers"] / results["gamerunner"]
        verdict = "FASTER" if speedup >= 1.5 else "NOT faster"
        print(f"\n  Workers speedup: {speedup:.2f}x ({verdict})")
        if speedup >= 2.0:
            n_episodes_100k_gamerunner = 100_000 / results["gamerunner"] / 3600
            n_episodes_100k_workers = 100_000 / results["workers"] / 3600
            print(f"\n  100K episodes estimate:")
            print(f"    GameRunner: {n_episodes_100k_gamerunner:.1f}h")
            print(f"    Workers:    {n_episodes_100k_workers:.1f}h")
        print()
        if speedup >= 2.0:
            print("→ USE --workers for self-play (update selfplay_launch.py)")
        else:
            print("→ KEEP GameRunner (workers not significantly faster)")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
