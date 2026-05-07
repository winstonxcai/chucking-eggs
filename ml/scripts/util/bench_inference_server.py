"""Cross-actor inference batching benchmark.

Compares two architectures on M1 MPS:

  baseline    — N rollout workers, each holds its own q_nets on the inference
                device, forwards locally per decision.
  server      — N rollout workers (no q_nets), 1 inference-server process holds
                the 4 q_nets on the inference device. Workers send candidate
                sets via mp.Queue, server batches across workers per position,
                runs one forward per position-group, replies via per-worker
                reply queue.

Each worker plays R full episodes with epsilon=0 (every decision goes through
inference). We report total ep/s, per-worker ep/s, and inference-side stats
(mean batch size, forwards/sec).

Usage:
    PYTHONPATH=ml/src .venv/bin/python ml/scripts/util/bench_inference_server.py \\
        --device mps --episodes-per-worker 30 --n-workers-list 1,2,4,8
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


_KEYS = (
    "own_hand", "others_hand", "recent_action_each_player",
    "played_cards_others", "remaining_counts_others",
    "level", "history", "behavior", "candidate_action",
)


# ─── BASELINE worker (q_nets local, like today's actor) ────────────


def _baseline_worker(worker_id: int, args_dict: dict, return_q: mp.Queue) -> None:
    import sys
    sys.path.insert(0, str(Path(args_dict["repo_root"]) / "ml" / "src"))
    from guandan.game import GuanDanEnv
    from guandan.guanzero.encoder import StateActionEncoder, _multi_hot
    from guandan.guanzero.q_network import init_seat_nets
    from guandan.guanzero.buffer import collate_encoded
    from guandan.azguan.behavior_flags import compute_behavior_flags
    from guandan.pvguan.legal_utils import dedup_strategic
    from guandan.pvguan.rollout import _cap_legal

    torch.set_num_threads(1)
    rng = random.Random(args_dict["seed"] + worker_id * 10_000)
    encoder = StateActionEncoder(use_oracle_others_hand=True)
    nets = init_seat_nets(
        hidden_lstm=args_dict["hidden_lstm"],
        hidden_mlp=args_dict["hidden_mlp"],
        n_mlp_layers=args_dict["n_mlp_layers"],
        dropout=0.0, use_oracle_others_hand=True,
    )
    device = torch.device(args_dict["device"])
    for n in nets.values():
        n.to(device); n.eval()

    n_dec = 0
    t0 = time.perf_counter()
    for _ in range(args_dict["episodes"]):
        env = GuanDanEnv(); env.reset(seed=rng.randint(0, 10_000_000))
        while not env.done:
            p = env.current_player
            legal = env.legal_moves(p)
            legal = dedup_strategic(legal)
            if args_dict["max_legal"] and len(legal) > args_dict["max_legal"]:
                keep = _cap_legal(env, p, legal, args_dict["max_legal"])
                legal = [legal[i] for i in keep]
            if not legal:
                break
            n_dec += 1
            shared = encoder._encode_state(env, p)
            encoded_list = []
            for action in legal:
                enc = dict(shared)
                enc["behavior"] = compute_behavior_flags(env, p, action, legal)
                enc["candidate_action"] = _multi_hot(action.cards)
                encoded_list.append(enc)
            batch = collate_encoded(encoded_list, device=device)
            with torch.no_grad():
                q = nets[p](batch)
                idx = int(q.argmax().item())
            env.step(legal[idx])
    return_q.put((worker_id, time.perf_counter() - t0, n_dec))


# ─── SERVER architecture ───────────────────────────────────────────


_SHAPES = {
    "own_hand": (108,),
    "others_hand": (108,),
    "recent_action_each_player": (4, 108),
    "played_cards_others": (3, 108),
    "remaining_counts_others": (3, 27),
    "level": (13,),
    "history": (20, 108),
    "behavior": (9,),
    "candidate_action": (108,),
}


def _server_worker(worker_id: int, args_dict: dict,
                   req_q: mp.Queue, reply_q: mp.Queue,
                   return_q: mp.Queue) -> None:
    import sys
    sys.path.insert(0, str(Path(args_dict["repo_root"]) / "ml" / "src"))
    from guandan.game import GuanDanEnv
    from guandan.guanzero.encoder import StateActionEncoder, _multi_hot
    from guandan.azguan.behavior_flags import compute_behavior_flags
    from guandan.pvguan.legal_utils import dedup_strategic
    from guandan.pvguan.rollout import _cap_legal

    torch.set_num_threads(1)
    rng = random.Random(args_dict["seed"] + worker_id * 10_000)
    encoder = StateActionEncoder(use_oracle_others_hand=True)

    n_dec = 0
    t0 = time.perf_counter()
    for _ in range(args_dict["episodes"]):
        env = GuanDanEnv(); env.reset(seed=rng.randint(0, 10_000_000))
        while not env.done:
            p = env.current_player
            legal = env.legal_moves(p)
            legal = dedup_strategic(legal)
            if args_dict["max_legal"] and len(legal) > args_dict["max_legal"]:
                keep = _cap_legal(env, p, legal, args_dict["max_legal"])
                legal = [legal[i] for i in keep]
            if not legal:
                break
            n_dec += 1
            shared = encoder._encode_state(env, p)
            encoded_list = []
            for action in legal:
                enc = dict(shared)
                enc["behavior"] = compute_behavior_flags(env, p, action, legal)
                enc["candidate_action"] = _multi_hot(action.cards)
                encoded_list.append(enc)
            req_q.put({
                "worker_id": worker_id,
                "player":    p,
                "encoded":   encoded_list,
            })
            q = reply_q.get()                # blocks until server replies
            idx = int(np.argmax(q))
            env.step(legal[idx])
    return_q.put((worker_id, time.perf_counter() - t0, n_dec))


def _server_loop(args_dict: dict, req_q: mp.Queue, reply_qs: list,
                 stop_event, stats_q: mp.Queue) -> None:
    import sys
    sys.path.insert(0, str(Path(args_dict["repo_root"]) / "ml" / "src"))
    from guandan.guanzero.q_network import init_seat_nets
    from guandan.guanzero.buffer import collate_encoded

    torch.set_num_threads(1)
    device = torch.device(args_dict["device"])
    nets = init_seat_nets(
        hidden_lstm=args_dict["hidden_lstm"],
        hidden_mlp=args_dict["hidden_mlp"],
        n_mlp_layers=args_dict["n_mlp_layers"],
        dropout=0.0, use_oracle_others_hand=True,
    )
    for n in nets.values():
        n.to(device); n.eval()

    n_forwards = 0
    sum_batch  = 0
    n_requests = 0

    while not stop_event.is_set():
        # Block on first request, then drain greedily up to K_max.
        try:
            first = req_q.get(timeout=0.05)
        except Exception:
            continue
        pending = [first]
        # Drain any other requests already waiting (non-blocking).
        while len(pending) < args_dict["max_concurrent"]:
            try:
                pending.append(req_q.get_nowait())
            except Exception:
                break

        # Group by position
        by_pos: dict[int, list] = defaultdict(list)
        for r in pending:
            by_pos[r["player"]].append(r)

        for p, reqs in by_pos.items():
            # Build a single big encoded_list by concatenating each request's candidates
            offsets = [0]
            big = []
            for r in reqs:
                big.extend(r["encoded"])
                offsets.append(len(big))
            batch = collate_encoded(big, device=device)
            with torch.no_grad():
                q_all = nets[p](batch).detach().cpu().numpy()
            n_forwards += 1
            sum_batch  += len(big)
            n_requests += len(reqs)
            for i, r in enumerate(reqs):
                lo, hi = offsets[i], offsets[i + 1]
                reply_qs[r["worker_id"]].put(q_all[lo:hi])

    stats_q.put((n_forwards, sum_batch, n_requests))


# ─── Drivers ───────────────────────────────────────────────────────


def run_baseline(N: int, common: dict, ctx) -> dict:
    return_q = ctx.Queue()
    procs = [
        ctx.Process(target=_baseline_worker, args=(i, common, return_q))
        for i in range(N)
    ]
    t0 = time.perf_counter()
    for p in procs: p.start()
    results = [return_q.get() for _ in range(N)]
    for p in procs: p.join()
    wall = time.perf_counter() - t0
    total_eps = N * common["episodes"]
    return {
        "wall": wall,
        "total_eps_per_s": total_eps / wall,
        "per_worker_eps": common["episodes"] / max(r[1] for r in results),
    }


def run_server(N: int, common: dict, ctx) -> dict:
    req_q     = ctx.Queue(maxsize=N * 2)
    reply_qs  = [ctx.Queue() for _ in range(N)]
    return_q  = ctx.Queue()
    stats_q   = ctx.Queue()
    stop_evt  = ctx.Event()

    server = ctx.Process(target=_server_loop,
                         args=(common, req_q, reply_qs, stop_evt, stats_q),
                         daemon=True)
    server.start()

    procs = [
        ctx.Process(target=_server_worker,
                    args=(i, common, req_q, reply_qs[i], return_q))
        for i in range(N)
    ]
    t0 = time.perf_counter()
    for p in procs: p.start()
    results = [return_q.get() for _ in range(N)]
    for p in procs: p.join()
    wall = time.perf_counter() - t0

    stop_evt.set()
    server.join(timeout=5)
    if server.is_alive():
        server.terminate()
        server.join(timeout=2)

    n_fwd = sum_b = n_req = 0
    try:
        n_fwd, sum_b, n_req = stats_q.get(timeout=1)
    except Exception:
        pass

    total_eps = N * common["episodes"]
    return {
        "wall":           wall,
        "total_eps_per_s": total_eps / wall,
        "per_worker_eps": common["episodes"] / max(r[1] for r in results),
        "forwards":       n_fwd,
        "mean_batch":     (sum_b / n_fwd) if n_fwd else 0,
        "reqs_per_fwd":   (n_req / n_fwd) if n_fwd else 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--episodes-per-worker", type=int, default=30)
    ap.add_argument("--n-workers-list", default="1,2,4,8")
    ap.add_argument("--max-concurrent", type=int, default=64,
                    help="Server batches up to this many candidate sets per forward.")
    ap.add_argument("--max-legal", type=int, default=128)
    ap.add_argument("--hidden-lstm", type=int, default=256)
    ap.add_argument("--hidden-mlp", type=int, default=1024)
    ap.add_argument("--n-mlp-layers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=("both", "baseline", "server"), default="both")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    common = dict(
        device=args.device, episodes=args.episodes_per_worker,
        max_concurrent=args.max_concurrent, max_legal=args.max_legal,
        hidden_lstm=args.hidden_lstm, hidden_mlp=args.hidden_mlp,
        n_mlp_layers=args.n_mlp_layers, seed=args.seed,
        repo_root=str(repo_root),
    )

    print(f"Inference-server bench  device={args.device}  "
          f"net=LSTM{args.hidden_lstm}+MLP{args.hidden_mlp}x{args.n_mlp_layers}  "
          f"R={args.episodes_per_worker} ep/worker")

    Ns = [int(s) for s in args.n_workers_list.split(",")]
    ctx = mp.get_context("spawn")

    print(f"\n  {'mode':>10s} {'N':>3s} {'wall':>7s} {'tot ep/s':>9s} "
          f"{'per-w':>6s} {'fwds':>6s} {'mean_B':>7s} {'reqs/fwd':>9s}")
    print(f"  {'-'*72}")
    for N in Ns:
        if args.mode in ("both", "baseline"):
            r = run_baseline(N, common, ctx)
            print(f"  {'baseline':>10s} {N:>3d} {r['wall']:>7.2f} "
                  f"{r['total_eps_per_s']:>9.2f} {r['per_worker_eps']:>6.2f} "
                  f"{'—':>6s} {'—':>7s} {'—':>9s}")
        if args.mode in ("both", "server"):
            r = run_server(N, common, ctx)
            print(f"  {'server':>10s} {N:>3d} {r['wall']:>7.2f} "
                  f"{r['total_eps_per_s']:>9.2f} {r['per_worker_eps']:>6.2f} "
                  f"{r['forwards']:>6d} {r['mean_batch']:>7.1f} "
                  f"{r['reqs_per_fwd']:>9.2f}")


if __name__ == "__main__":
    main()
