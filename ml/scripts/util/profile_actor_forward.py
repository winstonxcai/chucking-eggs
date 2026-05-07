"""torch.profiler decomposition of the actor's local-CPU forward pass.

We want to know, for one decision through the paper-spec q-net at typical
K (legal-action count), how the wall time splits across:

  - collate_encoded     (np.stack + torch.from_numpy + .to(device))
  - LSTM history        (history channel only: (B, 20, 108) → (B, 256))
  - tensor concat       (cat of 9 channels → (B, in_dim))
  - MLP                 (6 × (Linear + ReLU) + final Linear)
  - argmax + .item()    (cpu sync)
  - device casts        (any sneaky .to() or implicit casts)

We use torch.profiler with `record_function` blocks for logical phases and
the per-aten-op breakdown for fine-grained attribution. Output is the
standard `key_averages().table()` plus our own logical-phase summary.

Usage:
    PYTHONPATH=ml/src .venv/bin/python ml/scripts/util/profile_actor_forward.py \\
        --device cpu --K 5 --repeats 500 --warmup 50

    # Compare with torch.compile on the q-net:
    PYTHONPATH=ml/src .venv/bin/python ml/scripts/util/profile_actor_forward.py \\
        --device cpu --K 5 --compile

    # Realistic K distribution (heavy-tailed) instead of fixed K:
    PYTHONPATH=ml/src .venv/bin/python ml/scripts/util/profile_actor_forward.py \\
        --device cpu --K-dist heavy --repeats 1000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile, record_function

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))


# Match encoder.py shapes exactly so the q-net forward sees realistic input
_KEYS = (
    "own_hand", "others_hand", "recent_action_each_player",
    "played_cards_others", "remaining_counts_others",
    "level", "history", "behavior", "candidate_action",
)
_SHAPES = {
    "own_hand":                  (108,),
    "others_hand":               (108,),
    "recent_action_each_player": (4, 108),
    "played_cards_others":       (3, 108),
    "remaining_counts_others":   (3, 27),
    "level":                     (13,),
    "history":                   (20, 108),
    "behavior":                  (9,),
    "candidate_action":          (108,),
}


def make_encoded_list(K: int, rng: np.random.Generator) -> list[dict]:
    """Build a fake encoded_list of K decisions sharing one synthetic state.
    Mirrors what `StateActionEncoder.encode_all` returns for one decision."""
    state = {
        "own_hand":                  rng.integers(0, 2, _SHAPES["own_hand"], np.uint8).astype(np.float32),
        "others_hand":               rng.integers(0, 2, _SHAPES["others_hand"], np.uint8).astype(np.float32),
        "recent_action_each_player": rng.integers(0, 2, _SHAPES["recent_action_each_player"], np.uint8).astype(np.float32),
        "played_cards_others":       rng.integers(0, 2, _SHAPES["played_cards_others"], np.uint8).astype(np.float32),
        "remaining_counts_others":   _onehot_axis_last(_SHAPES["remaining_counts_others"], rng),
        "level":                     _onehot_axis_last(_SHAPES["level"], rng),
        "history":                   rng.integers(0, 2, _SHAPES["history"], np.uint8).astype(np.float32),
    }
    out = []
    for _ in range(K):
        enc = dict(state)   # shallow copy — encoder does this too
        enc["behavior"]         = _onehot_axis_last(_SHAPES["behavior"], rng)
        enc["candidate_action"] = rng.integers(0, 2, _SHAPES["candidate_action"], np.uint8).astype(np.float32)
        out.append(enc)
    return out


def _onehot_axis_last(shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    out = np.zeros(shape, dtype=np.float32)
    if len(shape) == 1:
        out[rng.integers(0, shape[0])] = 1.0
    else:
        for i in range(shape[0]):
            out[i, rng.integers(0, shape[1])] = 1.0
    return out


def _draw_K(K_dist: str, fixed_K: int, rng: np.random.Generator, max_K: int = 320) -> int:
    if K_dist == "heavy":
        r = rng.random()
        if   r < 0.6:  return int(rng.integers(2, 10))
        elif r < 0.95: return int(rng.integers(10, 40))
        else:          return int(rng.integers(40, min(120, max_K)))
    return fixed_K


def collate_encoded_local(encoded_list, device):
    """Inline copy of buffer.collate_encoded so we can wrap it in record_function
    without pulling buffer.py into the trace."""
    keys = encoded_list[0].keys()
    batch = {}
    for k in keys:
        arr = np.stack([e[k] for e in encoded_list], axis=0)
        batch[k] = torch.from_numpy(arr).to(device, non_blocking=True)
    return batch


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--hidden-lstm", type=int, default=256)
    ap.add_argument("--hidden-mlp", type=int, default=1024)
    ap.add_argument("--n-mlp-layers", type=int, default=6)
    ap.add_argument("--K", type=int, default=5,
                    help="Fixed batch size (legal-action count). Ignored if --K-dist=heavy.")
    ap.add_argument("--K-dist", choices=["fixed", "heavy"], default="fixed")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--repeats", type=int, default=500)
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile(net, dynamic=True) before profiling")
    ap.add_argument("--row-limit", type=int, default=20)
    ap.add_argument("--trace-out", type=str, default=None,
                    help="Save chrome trace to this path for visualization")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    rng = np.random.default_rng(0)

    # Build paper-spec net
    from guandan.guanzero.q_network import init_seat_nets
    from guandan.guanzero.config import QNetConfig
    nets = init_seat_nets(QNetConfig(hidden_lstm=args.hidden_lstm, hidden_mlp=args.hidden_mlp, n_mlp_layers=args.n_mlp_layers))
    net = nets[0].to(device)
    net.eval()
    if args.compile:
        net = torch.compile(net, dynamic=True)
        print(f"[INFO] torch.compile(dynamic=True) enabled — first-call warmup may be slow")

    # ── Warmup ──────────────────────────────────────────────────
    print(f"\nWarming up ({args.warmup} iters at K={args.K})...")
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(args.warmup):
            K = _draw_K(args.K_dist, args.K, rng)
            encoded = make_encoded_list(K, rng)
            batch = collate_encoded_local(encoded, device)
            _ = net(batch)
    print(f"Warmup wall: {time.perf_counter()-t0:.2f}s")

    # ── Total wall measurement (no profiler, for accurate timing) ──
    print(f"\nMeasuring wall time over {args.repeats} iters (no profiler)...")
    t_collate = t_forward = t_argmax = 0.0
    n_K_seen = 0
    sum_K = 0
    with torch.no_grad():
        for _ in range(args.repeats):
            K = _draw_K(args.K_dist, args.K, rng)
            encoded = make_encoded_list(K, rng)
            n_K_seen += 1
            sum_K += K

            t0 = time.perf_counter()
            batch = collate_encoded_local(encoded, device)
            t_collate += time.perf_counter() - t0

            t0 = time.perf_counter()
            q = net(batch)
            t_forward += time.perf_counter() - t0

            t0 = time.perf_counter()
            _ = int(q.argmax().item())
            t_argmax += time.perf_counter() - t0

    total_ms_call = 1000 * (t_collate + t_forward + t_argmax) / args.repeats
    print(f"{'phase':22s} {'ms/call':>10s} {'%':>7s}")
    print("-" * 45)
    total = t_collate + t_forward + t_argmax
    for name, t in [("collate_encoded", t_collate),
                    ("net(batch) forward", t_forward),
                    ("argmax + .item()", t_argmax)]:
        ms = 1000 * t / args.repeats
        pct = 100 * t / total if total else 0
        print(f"{name:22s} {ms:>10.3f} {pct:>7.1f}")
    print("-" * 45)
    print(f"{'TOTAL':22s} {total_ms_call:>10.3f}")
    print(f"  mean K = {sum_K / n_K_seen:.1f}")

    # ── torch.profiler decomposition ────────────────────────────
    print(f"\nRunning torch.profiler over {min(args.repeats, 100)} iters...")
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    n_prof_iters = min(args.repeats, 100)
    with torch.no_grad(), profile(
        activities=activities,
        record_shapes=False,
        with_stack=False,
    ) as prof:
        for _ in range(n_prof_iters):
            K = _draw_K(args.K_dist, args.K, rng)
            encoded = make_encoded_list(K, rng)
            with record_function("p_collate"):
                batch = collate_encoded_local(encoded, device)
            with record_function("p_net_forward"):
                q = net(batch)
            with record_function("p_argmax_item"):
                _ = int(q.argmax().item())

    print("\n=== Per-op self-time table (top {}) ===".format(args.row_limit))
    sort_key = "cpu_time_total" if device.type != "cuda" else "cuda_time_total"
    print(prof.key_averages().table(
        sort_by=sort_key,
        row_limit=args.row_limit,
        max_name_column_width=60,
    ))

    # Build a logical-phase summary by walking the events
    print("\n=== Logical phase summary (record_function blocks) ===")
    events = prof.key_averages()
    phase_names = ("p_collate", "p_net_forward", "p_argmax_item")
    print(f"{'phase':22s} {'count':>8s} {'cpu_us/call':>14s} {'cpu_total_ms':>14s}")
    print("-" * 65)
    for ev in events:
        if ev.key in phase_names:
            print(f"{ev.key:22s} {ev.count:>8d} {ev.cpu_time:>14.1f} {ev.cpu_time_total/1000:>14.2f}")

    # Look for specific ops we care about
    print("\n=== Ops of interest (LSTM / MLP / cat / cast / from_numpy) ===")
    interesting_ops = (
        "aten::lstm", "aten::cudnn_rnn", "aten::_lstm_mps", "aten::linear",
        "aten::addmm", "aten::matmul", "aten::cat", "aten::reshape",
        "aten::view", "aten::contiguous", "aten::to", "aten::copy_",
        "aten::stack", "aten::relu", "aten::relu_", "aten::tanh",
        "aten::sigmoid", "aten::argmax", "aten::_to_copy",
    )
    print(f"{'op':50s} {'count':>8s} {'us/call':>10s} {'total_ms':>12s}")
    print("-" * 85)
    for ev in sorted(events, key=lambda e: -e.cpu_time_total):
        if ev.key in interesting_ops:
            print(f"{ev.key:50s} {ev.count:>8d} {ev.cpu_time:>10.1f} {ev.cpu_time_total/1000:>12.2f}")

    if args.trace_out:
        prof.export_chrome_trace(args.trace_out)
        print(f"\nChrome trace saved → {args.trace_out}")


if __name__ == "__main__":
    main()
