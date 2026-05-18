"""Benchmark Dart actor rollout throughput.

Measures the existing single-episode actor loop against the batched self-play
lane path. Use this from the repo root:

    .venv/bin/python ml/scripts/util/bench_actor_throughput.py --episodes 64
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time

import torch

from guandan.dart.config import MODEL_TYPE_DART, TrainConfig, dart_qnet_config
from guandan.dart.model.encoding.role_encoder import RoleAwareStateActionEncoder
from guandan.dart.model.q_network import DartQNet
from guandan.dart.runtime.actor import LaneConfig, all_latest_seats, play_episode, play_episodes_batched


def _build_net() -> DartQNet:
    cfg = TrainConfig(model_type=MODEL_TYPE_DART)
    torch.manual_seed(cfg.seed)
    return DartQNet(dart_qnet_config(cfg)).eval()


def _bench_sequential(net, encoder, episodes: int, epsilon: float, seed_base: int) -> tuple[float, int]:
    samples = 0
    t0 = time.perf_counter()
    for i in range(episodes):
        samples += len(play_episode(
            q_nets=net,
            encoder=encoder,
            seats=all_latest_seats(epsilon),
            seed=seed_base + i,
            device="cpu",
        ))
    return time.perf_counter() - t0, samples


def _bench_batched(
    net,
    encoder,
    episodes: int,
    lanes: int,
    epsilon: float,
    seed_base: int,
) -> tuple[float, int]:
    samples = 0
    completed = 0
    t0 = time.perf_counter()
    while completed < episodes:
        n = min(lanes, episodes - completed)
        seeds = list(range(seed_base + completed, seed_base + completed + n))
        samples += sum(len(xs) for xs in play_episodes_batched(
            q_nets=net,
            encoder=encoder,
            lanes=[
                LaneConfig(seed=seed, seats=all_latest_seats(epsilon))
                for seed in seeds
            ],
            device="cpu",
        ))
        completed += n
    return time.perf_counter() - t0, samples


def _worker_main(
    worker_id: int,
    episodes: int,
    lanes: int,
    epsilon: float,
    seed_base: int,
    threads: int,
    out_q: mp.Queue,
) -> None:
    torch.set_num_threads(threads)
    net = _build_net()
    encoder = RoleAwareStateActionEncoder(is_partner_visible=True)
    base = seed_base + worker_id * 1_000_000
    # Warm both code paths inside each spawned process.
    play_episode(
        net,
        encoder,
        seats=all_latest_seats(epsilon),
        seed=base - 2,
        device="cpu",
    )
    play_episodes_batched(
        net,
        encoder,
        lanes=[
            LaneConfig(seed=base - 1, seats=all_latest_seats(epsilon)),
            LaneConfig(seed=base, seats=all_latest_seats(epsilon)),
        ],
        device="cpu",
    )
    if lanes == 1:
        wall, samples = _bench_sequential(net, encoder, episodes, epsilon, base)
    else:
        wall, samples = _bench_batched(net, encoder, episodes, lanes, epsilon, base)
    out_q.put((worker_id, wall, samples))


def _bench_workers(
    workers: int,
    episodes_per_worker: int,
    lanes: int,
    epsilon: float,
    seed_base: int,
    threads: int,
) -> tuple[float, int, list[float]]:
    ctx = mp.get_context("spawn")
    out_q = ctx.Queue()
    procs = [
        ctx.Process(
            target=_worker_main,
            args=(w, episodes_per_worker, lanes, epsilon, seed_base, threads, out_q),
            name=f"actor-bench-{w}",
        )
        for w in range(workers)
    ]
    t0 = time.perf_counter()
    for p in procs:
        p.start()

    results = []
    try:
        for _ in procs:
            results.append(out_q.get())
    finally:
        for p in procs:
            p.join()
    wall = time.perf_counter() - t0
    failures = [p for p in procs if p.exitcode != 0]
    if failures:
        raise RuntimeError(
            "worker failure(s): "
            + ", ".join(f"{p.name} exitcode={p.exitcode}" for p in failures)
        )
    return wall, sum(samples for _, _, samples in results), [w for _, w, _ in results]


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark local Dart actor throughput")
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--lanes", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--seed-base", type=int, default=10_000)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    net = _build_net()
    encoder = RoleAwareStateActionEncoder(is_partner_visible=True)

    # Warm both code paths.
    play_episode(
        net,
        encoder,
        seats=all_latest_seats(args.epsilon),
        seed=args.seed_base - 2,
        device="cpu",
    )
    play_episodes_batched(
        net,
        encoder,
        lanes=[
            LaneConfig(seed=args.seed_base - 1, seats=all_latest_seats(args.epsilon)),
            LaneConfig(seed=args.seed_base, seats=all_latest_seats(args.epsilon)),
        ],
        device="cpu",
    )

    print(
        f"workers={args.workers} episodes_per_worker={args.episodes} "
        f"epsilon={args.epsilon} torch_threads={args.threads}"
    )
    print("lanes,wall_s,episodes_s,samples_s,samples,worker_wall_max_s")
    for lanes in args.lanes:
        if args.workers == 1:
            if lanes == 1:
                wall, samples = _bench_sequential(
                    net, encoder, args.episodes, args.epsilon, args.seed_base
                )
            else:
                wall, samples = _bench_batched(
                    net, encoder, args.episodes, lanes, args.epsilon, args.seed_base
                )
            worker_wall_max = wall
        else:
            wall, samples, worker_walls = _bench_workers(
                args.workers,
                args.episodes,
                lanes,
                args.epsilon,
                args.seed_base,
                args.threads,
            )
            worker_wall_max = max(worker_walls)
        episodes = args.episodes * args.workers
        print(
            f"{lanes},{wall:.3f},{episodes / wall:.3f},"
            f"{samples / wall:.1f},{samples},{worker_wall_max:.3f}"
        )


if __name__ == "__main__":
    main()
