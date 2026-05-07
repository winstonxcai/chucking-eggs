"""Multi-actor scaling benchmark.

Spawns N independent actor processes, each running R episodes with epsilon=0
(so every decision goes through Q-forward). Reports total ep/s and per-actor
ep/s so we can see where we hit CPU saturation on M1.

No queue, no learner, no weight sync — pure rollout-throughput-vs-N test.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import random
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch


def _worker(actor_id: int, args_dict: dict, return_q: mp.Queue) -> None:
    import sys, os
    sys.path.insert(0, str(Path(args_dict["repo_root"]) / "ml" / "src"))
    from guandan.game import GuanDanEnv
    from guandan.guanzero.encoder import StateActionEncoder, _multi_hot
    from guandan.guanzero.q_network import init_seat_nets
    from guandan.guanzero.config import QNetConfig
    from guandan.guanzero.returns import compute_mc_returns
    from guandan.guanzero.buffer import collate_encoded
    from guandan.azguan.behavior_flags import compute_behavior_flags
    from guandan.pvguan.legal_utils import dedup_strategic
    from guandan.pvguan.rollout import _cap_legal

    torch.set_num_threads(1)
    torch.manual_seed(args_dict["seed"] + actor_id)
    rng = random.Random(args_dict["seed"] + actor_id * 10_000)

    encoder = StateActionEncoder(use_oracle_others_hand=True)
    nets = init_seat_nets(QNetConfig(hidden_lstm=args_dict["hidden_lstm"], hidden_mlp=args_dict["hidden_mlp"], n_mlp_layers=args_dict["n_mlp_layers"]))
    for n in nets.values():
        n.eval()

    n_decisions = 0
    t0 = time.perf_counter()
    for _ in range(args_dict["episodes"]):
        env = GuanDanEnv()
        env.reset(seed=rng.randint(0, 10_000_000))
        traj = []
        while not env.done:
            p = env.current_player
            legal = env.legal_moves(p)
            legal = dedup_strategic(legal)
            if args_dict["max_legal"] and len(legal) > args_dict["max_legal"]:
                keep = _cap_legal(env, p, legal, args_dict["max_legal"])
                legal = [legal[i] for i in keep]
            if not legal:
                break
            n_decisions += 1
            shared = encoder._encode_state(env, p)
            encoded_list = []
            for action in legal:
                enc = dict(shared)
                enc["behavior"] = compute_behavior_flags(env, p, action, legal)
                enc["candidate_action"] = _multi_hot(action.cards)
                encoded_list.append(enc)
            if random.random() < args_dict["epsilon"]:
                idx = random.randrange(len(legal))
            else:
                batch = collate_encoded(encoded_list, device="cpu")
                with torch.no_grad():
                    q = nets[p](batch)
                    idx = int(q.argmax().item())
            traj.append({"player": p, "encoded": encoded_list[idx]})
            env.step(legal[idx])
        rewards = env.get_rewards() if env.done else {p: 0.0 for p in range(4)}
        compute_mc_returns(traj, rewards, gamma=1.0)
    elapsed = time.perf_counter() - t0
    return_q.put((actor_id, elapsed, n_decisions))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes-per-actor", type=int, default=50)
    ap.add_argument("--n-actors-list", default="1,2,4,6,8")
    ap.add_argument("--epsilon", type=float, default=0.0)
    ap.add_argument("--max-legal", type=int, default=128)
    ap.add_argument("--hidden-lstm", type=int, default=256)
    ap.add_argument("--hidden-mlp", type=int, default=1024)
    ap.add_argument("--n-mlp-layers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    common = dict(
        episodes=args.episodes_per_actor,
        epsilon=args.epsilon,
        max_legal=args.max_legal,
        hidden_lstm=args.hidden_lstm,
        hidden_mlp=args.hidden_mlp,
        n_mlp_layers=args.n_mlp_layers,
        seed=args.seed,
        repo_root=str(repo_root),
    )

    print(f"Multi-actor scaling  |  {args.episodes_per_actor} ep/actor  "
          f"|  net=LSTM{args.hidden_lstm}+MLP{args.hidden_mlp}x{args.n_mlp_layers}")
    print(f"{'N':>3s} {'wall':>8s} {'total_ep/s':>12s} {'per_actor':>10s} "
          f"{'speedup':>8s} {'efficiency':>10s}")

    baseline = None
    ctx = mp.get_context("spawn")
    for n in [int(s) for s in args.n_actors_list.split(",")]:
        return_q = ctx.Queue()
        procs = [
            ctx.Process(target=_worker, args=(i, common, return_q))
            for i in range(n)
        ]
        t0 = time.perf_counter()
        for p in procs:
            p.start()
        results = [return_q.get() for _ in range(n)]
        for p in procs:
            p.join()
        wall = time.perf_counter() - t0
        total_eps = n * args.episodes_per_actor
        total_eps_per_s = total_eps / wall
        per_actor = (args.episodes_per_actor / max(r[1] for r in results))
        if baseline is None:
            baseline = total_eps_per_s
        speedup = total_eps_per_s / baseline
        eff = speedup / n
        print(f"{n:>3d} {wall:>8.2f} {total_eps_per_s:>12.2f} "
              f"{per_actor:>10.2f} {speedup:>8.2f}x {eff*100:>9.0f}%")


if __name__ == "__main__":
    main()
