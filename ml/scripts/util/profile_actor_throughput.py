"""Profile Dart actor rollout throughput with Modal-sized worker/lanes settings.

This is intentionally a raw actor benchmark: no learner, replay buffer, queue, or
replay-ratio throttle. It answers where actor CPU time goes for the configured
``workers × lanes`` rollout shape.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import torch
from guandan.dart.config import MODEL_TYPE_DART, TrainConfig, dart_qnet_config
from guandan.dart.model.encoding.role_encoder import RoleAwareStateActionEncoder
from guandan.dart.model.q_network import DartQNet
from guandan.dart.runtime.actor import (
    LaneConfig,
    all_latest_seats,
    play_episodes_batched,
)
from guandan.dart.utils.profiler import PhaseProfiler


@dataclass(frozen=True)
class WorkerProfile:
    worker_id: int
    wall_s: float
    episodes: int
    samples: int
    times: dict[str, float]
    counts: dict[str, int]


def _build_net() -> DartQNet:
    cfg = TrainConfig(model_type=MODEL_TYPE_DART)
    torch.manual_seed(cfg.seed)
    return DartQNet(dart_qnet_config(cfg)).eval()


def _run_batched(
    net: DartQNet,
    encoder: RoleAwareStateActionEncoder,
    *,
    episodes: int,
    lanes: int,
    epsilon: float,
    seed_base: int,
    prof: PhaseProfiler,
    max_forced_pass_replay_frac: float,
) -> tuple[float, int]:
    samples = 0
    completed = 0
    t0 = time.perf_counter()
    while completed < episodes:
        n = min(lanes, episodes - completed)
        samples_per_lane = play_episodes_batched(
            q_nets=net,
            encoder=encoder,
            lanes=[
                LaneConfig(seed=seed_base + completed + i, seats=all_latest_seats(epsilon))
                for i in range(n)
            ],
            device="cpu",
            profiler=prof,
            record_forced_k1_samples=max_forced_pass_replay_frac > 0.0,
        )
        samples += sum(len(xs) for xs in samples_per_lane)
        completed += n
    return time.perf_counter() - t0, samples


def _worker_main(
    worker_id: int,
    episodes: int,
    lanes: int,
    epsilon: float,
    seed_base: int,
    threads: int,
    max_forced_pass_replay_frac: float,
    out_q: mp.Queue,
) -> None:
    torch.set_num_threads(threads)
    net = _build_net()
    encoder = RoleAwareStateActionEncoder(is_partner_visible=True)
    base = seed_base + worker_id * 1_000_000

    # Warm the same batched path so first-use initialization does not dominate.
    play_episodes_batched(
        q_nets=net,
        encoder=encoder,
        lanes=[
            LaneConfig(seed=base - 10 + i, seats=all_latest_seats(epsilon))
            for i in range(min(4, lanes))
        ],
        device="cpu",
        record_forced_k1_samples=max_forced_pass_replay_frac > 0.0,
    )

    prof = PhaseProfiler(enabled=True)
    wall_s, samples = _run_batched(
        net,
        encoder,
        episodes=episodes,
        lanes=lanes,
        epsilon=epsilon,
        seed_base=base,
        prof=prof,
        max_forced_pass_replay_frac=max_forced_pass_replay_frac,
    )
    out_q.put(WorkerProfile(
        worker_id=worker_id,
        wall_s=wall_s,
        episodes=episodes,
        samples=samples,
        times=dict(prof._times),
        counts=dict(prof._counts),
    ))


def _sum_dicts(items: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = defaultdict(float)
    for item in items:
        for k, v in item.items():
            out[k] += v
    return dict(out)


def _fmt_pct(num: float, den: float) -> str:
    return f"{100.0 * num / den:5.1f}%" if den > 0 else "  n/a"


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile Modal-shaped Dart actor rollout")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--episodes", type=int, default=64, help="Episodes per worker")
    parser.add_argument("--lanes", type=int, default=40)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--max-forced-pass-replay-frac", type=float, default=0.0)
    parser.add_argument("--seed-base", type=int, default=10_000)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    ctx = mp.get_context("spawn")
    out_q = ctx.Queue()
    procs = [
        ctx.Process(
            target=_worker_main,
            args=(
                w,
                args.episodes,
                args.lanes,
                args.epsilon,
                args.seed_base,
                args.threads,
                args.max_forced_pass_replay_frac,
                out_q,
            ),
            name=f"actor-profile-{w}",
        )
        for w in range(args.workers)
    ]

    t0 = time.perf_counter()
    for p in procs:
        p.start()

    profiles: list[WorkerProfile] = []
    try:
        for _ in procs:
            profiles.append(out_q.get())
    finally:
        for p in procs:
            p.join()

    failures = [p for p in procs if p.exitcode != 0]
    if failures:
        raise RuntimeError(
            "worker failure(s): "
            + ", ".join(f"{p.name} exitcode={p.exitcode}" for p in failures)
        )

    wall_s = time.perf_counter() - t0
    profiles.sort(key=lambda p: p.worker_id)
    total_episodes = sum(p.episodes for p in profiles)
    total_samples = sum(p.samples for p in profiles)
    worker_wall_sum = sum(p.wall_s for p in profiles)
    worker_wall_max = max(p.wall_s for p in profiles)
    times = _sum_dicts([p.times for p in profiles])
    counts = _sum_dicts([p.counts for p in profiles])
    total_timed = sum(times.values())
    q_forward_s = sum(v for k, v in times.items() if k.startswith("q_net_forward"))

    print(
        f"workers={args.workers} lanes={args.lanes} episodes_per_worker={args.episodes} "
        f"epsilon={args.epsilon} max_forced_pass_replay_frac={args.max_forced_pass_replay_frac} "
        f"torch_threads={args.threads}",
        flush=True,
    )
    print(
        f"wall_s={wall_s:.3f} worker_wall_max_s={worker_wall_max:.3f} "
        f"worker_wall_sum_s={worker_wall_sum:.3f}",
        flush=True,
    )
    print(
        f"episodes={total_episodes} samples={total_samples} "
        f"episodes_s={total_episodes / wall_s:.3f} samples_s={total_samples / wall_s:.1f}",
        flush=True,
    )
    print(
        f"total_timed_cpu_s={total_timed:.3f} timed_over_worker_wall="
        f"{total_timed / worker_wall_sum:.3f}",
        flush=True,
    )
    print(
        f"q_forward_cpu_s={q_forward_s:.3f} q_forward_share_timed={_fmt_pct(q_forward_s, total_timed)} "
        f"q_forward_share_worker_wall={_fmt_pct(q_forward_s, worker_wall_sum)}",
        flush=True,
    )

    print("\nphase,cpu_s,pct_timed,calls,ms_per_call", flush=True)
    for name, sec in sorted(times.items(), key=lambda kv: -kv[1]):
        calls = int(counts.get(name, 0))
        ms = 1000.0 * sec / calls if calls > 0 else 0.0
        print(f"{name},{sec:.6f},{_fmt_pct(sec, total_timed).strip()},{calls},{ms:.6f}", flush=True)

    decisions = int(counts.get("num_decisions", 0))
    action_rows = int(counts.get("q_forward_action_rows", 0))
    q_groups = int(counts.get("q_forward_groups", 0))
    q_calls = sum(int(counts.get(k, 0)) for k in counts if k.startswith("q_net_forward"))
    print("\ncounts", flush=True)
    print(f"decisions={decisions}", flush=True)
    print(f"num_legal_actions={int(counts.get('num_legal_actions', 0))}", flush=True)
    print(f"shortcut_K1={int(counts.get('shortcut_K1', 0))}", flush=True)
    print(f"epsilon_random={int(counts.get('epsilon_random', 0))}", flush=True)
    print(f"q_forward_calls={q_calls}", flush=True)
    print(f"q_forward_groups={q_groups}", flush=True)
    print(f"q_forward_action_rows={action_rows}", flush=True)
    if q_calls:
        print(f"groups_per_forward={q_groups / q_calls:.3f}", flush=True)
        print(f"actions_per_forward={action_rows / q_calls:.3f}", flush=True)
    if q_groups:
        print(f"actions_per_forward_group={action_rows / q_groups:.3f}", flush=True)


if __name__ == "__main__":
    main()
