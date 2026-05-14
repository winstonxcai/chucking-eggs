#!/usr/bin/env python3
"""Throughput benchmark: Python episode loop vs Rust episode loop.

Usage:
    PYTHONPATH=ml/src python ml/scripts/bench_rollout.py [--n-episodes 200] [--epsilon 0.05]

Reports episodes/sec, decisions/sec, and mean ms/decision for each path at
two epsilon levels:
  - epsilon=0.0  (full greedy — every decision calls the NN)
  - epsilon=1.0  (full random — measures game-loop overhead without NN forward)

Results are written to:
  ml/runs/bench/baseline.json  if --phase baseline  (default when Rust not yet integrated)
  ml/runs/bench/phase1.json    if --phase phase1

Compare the two files to see the speedup from Phase 1.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[2] / "ml" / "src"))

from guandan.guanzero.model.encoding.role_encoder import RoleAwareStateActionEncoder
from guandan.guanzero.model.q_network import SharedHeadQNet, SharedHeadQNetConfig
from guandan.guanzero.runtime.actor import play_episode, play_episode_rust

# Match prod config: single-threaded CPU inference per actor process.
torch.set_num_threads(1)

WARMUP_EPISODES = 10


@dataclass
class PathResult:
    path: str
    n_episodes: int
    epsilon: float
    wall_sec: float
    total_decisions: int
    episodes_per_sec: float
    decisions_per_sec: float
    mean_ms_per_decision: float


def _run_path(
    name: str,
    fn,
    q_net: SharedHeadQNet,
    encoder: RoleAwareStateActionEncoder,
    n_episodes: int,
    epsilon: float,
) -> PathResult:
    """Time n_episodes through fn and return a PathResult."""
    common = dict(q_nets=q_net, encoder=encoder, epsilon=epsilon, device="cpu")

    for i in range(WARMUP_EPISODES):
        fn(**common, seed=10_000 + i)

    total_decisions = 0
    t0 = time.perf_counter()
    for i in range(n_episodes):
        samples = fn(**common, seed=i)
        # samples contains only team-0 steps; multiply by 2 for all 4 players.
        total_decisions += len(samples) * 2
    wall = time.perf_counter() - t0

    eps_per_sec = n_episodes / wall
    dec_per_sec = total_decisions / wall
    mean_ms = wall * 1000 / total_decisions if total_decisions else float("nan")

    return PathResult(
        path=name,
        n_episodes=n_episodes,
        epsilon=epsilon,
        wall_sec=round(wall, 3),
        total_decisions=total_decisions,
        episodes_per_sec=round(eps_per_sec, 2),
        decisions_per_sec=round(dec_per_sec, 1),
        mean_ms_per_decision=round(mean_ms, 3),
    )


def _print_result(r: PathResult) -> None:
    print(
        f"  [{r.path:6s}] ε={r.epsilon:.2f}  "
        f"{r.episodes_per_sec:6.1f} eps/s  "
        f"{r.decisions_per_sec:7.1f} dec/s  "
        f"{r.mean_ms_per_decision:6.2f} ms/dec  "
        f"(wall {r.wall_sec:.1f}s, {r.total_decisions} decisions)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Guanzero rollout throughput benchmark")
    parser.add_argument("--n-episodes", type=int, default=200)
    parser.add_argument(
        "--phase",
        choices=["baseline", "phase1", "phase2"],
        default="phase2",
        help="Tag for output file",
    )
    args = parser.parse_args()

    print(f"Building SharedHeadQNet (default config) on CPU ...")
    cfg = SharedHeadQNetConfig()
    net = SharedHeadQNet(cfg).eval()
    encoder = RoleAwareStateActionEncoder(head_scheme="absolute_seat")

    out_dir = Path(__file__).parents[1] / "runs" / "bench"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.phase}.json"

    results: list[PathResult] = []

    print(f"\nBenchmarking {args.n_episodes} episodes (+{WARMUP_EPISODES} warmup) ...")
    print(f"{'':>10}  {'eps/s':>8}  {'dec/s':>9}  {'ms/dec':>8}")

    for epsilon in (0.0, 1.0):
        label = "full-NN" if epsilon == 0.0 else "random"
        print(f"\n  ε={epsilon:.1f} ({label})")

        py_r = _run_path("python", play_episode, net, encoder, args.n_episodes, epsilon)
        _print_result(py_r)
        results.append(py_r)

        rs_r = _run_path("rust", play_episode_rust, net, encoder, args.n_episodes, epsilon)
        _print_result(rs_r)
        results.append(rs_r)

        if py_r.decisions_per_sec > 0:
            speedup = rs_r.decisions_per_sec / py_r.decisions_per_sec
            print(f"  {'':6}  speedup: {speedup:.3f}x")

    out_path.write_text(json.dumps([asdict(r) for r in results], indent=2))
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
