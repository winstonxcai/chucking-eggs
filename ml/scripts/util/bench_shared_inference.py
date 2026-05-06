"""Phase 3 smoke: run the shared-memory inference server with N synthetic
actors and measure GPU-side throughput.

Independent of the learner — validates that the server alone can handle the
target load (≥50k candidate rows/sec at 32 actors with paper-spec net) and
that latency stays under a few ms.

Synthetic actors generate decisions with realistic-shaped K (legal-action
count) drawn from a heavy-tailed distribution that approximates real Guan
Dan play (most decisions K=4-20, occasional K up to ~100).

Usage on Modal (CUDA):
    modal run --detach ml/scripts/modal/bench_shared_inference_modal.py \\
        --n-actors 32 --duration-s 30

Or locally on CPU / MPS:
    PYTHONPATH=ml/src .venv/bin/python ml/scripts/util/bench_shared_inference.py \\
        --n-actors 16 --device cpu --duration-s 30
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))


# ─── Synthetic-actor process ─────────────────────────────────


def synthetic_actor_loop(
    actor_id:        int,
    meta,                                          # SharedBufferMeta
    free_slots,
    request_queue,
    events,
    stop_event,
    duration_s:      float,
    rng_seed:        int,
    K_dist:          str = "heavy",
) -> None:
    """Generate random decisions and submit them until stop_event.
    Writes per-actor latency stats to /tmp for parent-side aggregation."""
    import torch
    torch.set_num_threads(1)

    from guandan.guanzero.encoder import (
        CARD_ID_DIM, HISTORY_LEN, RANK_BUCKETS, LEVEL_DIM, FLAG_DIM,
    )
    from guandan.guanzero.inference_server import (
        SharedInferenceClient,
        attach_shared_buffers,
    )

    rng = np.random.default_rng(rng_seed)
    bufs = attach_shared_buffers(
        meta=meta,
        free_slots=free_slots,
        request_queue=request_queue,
        events=events,
    )
    client = SharedInferenceClient(actor_id=actor_id, bufs=bufs,
                                   timeout_s=10.0,
                                   max_actions=meta.max_actions)

    def _draw_K() -> int:
        """Approx real Guan Dan K distribution: heavy-tailed, mean ~10."""
        if K_dist == "heavy":
            r = rng.random()
            if   r < 0.6:  return int(rng.integers(2, 10))
            elif r < 0.95: return int(rng.integers(10, 40))
            else:          return int(rng.integers(40, min(120, meta.max_actions)))
        return int(rng.integers(2, min(20, meta.max_actions)))

    def _onehot(shape):
        out = np.zeros(shape, dtype=np.float32)
        if len(shape) == 1:
            out[rng.integers(0, shape[0])] = 1.0
        else:
            for i in range(shape[0]):
                out[i, rng.integers(0, shape[1])] = 1.0
        return out

    def _make_encoded(K: int) -> list[dict]:
        # Shared state (encoder._encode_state shape)
        state = {
            "own_hand":                  rng.integers(0, 2, size=(CARD_ID_DIM,), dtype=np.uint8).astype(np.float32),
            "others_hand":               rng.integers(0, 2, size=(CARD_ID_DIM,), dtype=np.uint8).astype(np.float32),
            "recent_action_each_player": rng.integers(0, 2, size=(4, CARD_ID_DIM), dtype=np.uint8).astype(np.float32),
            "played_cards_others":       rng.integers(0, 2, size=(3, CARD_ID_DIM), dtype=np.uint8).astype(np.float32),
            "remaining_counts_others":   _onehot((3, RANK_BUCKETS)),
            "level":                     _onehot((LEVEL_DIM,)),
            "history":                   rng.integers(0, 2, size=(HISTORY_LEN, CARD_ID_DIM), dtype=np.uint8).astype(np.float32),
        }
        out = []
        for _ in range(K):
            enc = dict(state)
            enc["behavior"]         = _onehot((FLAG_DIM,))
            enc["candidate_action"] = rng.integers(0, 2, size=(CARD_ID_DIM,), dtype=np.uint8).astype(np.float32)
            out.append(enc)
        return out

    n_decisions = 0
    n_action_rows = 0
    wait_ms_log: list[float] = []
    seat_cycle = 0
    deadline = time.perf_counter() + duration_s

    while not stop_event.is_set() and time.perf_counter() < deadline:
        seat = seat_cycle & 3
        seat_cycle += 1
        K = _draw_K()
        encoded = _make_encoded(K)

        t0 = time.perf_counter()
        client.submit(seat, encoded)
        wait_ms = (time.perf_counter() - t0) * 1000.0

        n_decisions += 1
        n_action_rows += K
        wait_ms_log.append(wait_ms)

    summary = {
        "actor_id":      actor_id,
        "n_decisions":   n_decisions,
        "n_action_rows": n_action_rows,
        "wait_ms_p50":   statistics.median(wait_ms_log) if wait_ms_log else 0.0,
        "wait_ms_p95":   _p95(wait_ms_log),
        "wait_ms_max":   max(wait_ms_log) if wait_ms_log else 0.0,
    }
    Path(f"/tmp/bench_shared_inference_actor_{actor_id}.json").write_text(json.dumps(summary))


def _p95(xs: list[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, int(len(s) * 0.95))]


# ─── Driver ──────────────────────────────────────────────────


def run_bench(args) -> dict:
    from guandan.guanzero.inference_server import (
        allocate_shared_buffers,
        release_shared_buffers,
        shared_server_loop_entry,
    )

    ctx = mp.get_context("spawn")
    bufs, meta = allocate_shared_buffers(
        num_slots=args.num_slots,
        max_actions=args.max_actions,
        n_actors=args.n_actors,
        ctx=ctx,
    )
    stop_event = ctx.Event()

    cfg_dict = {
        "inference_device":                args.device,
        "inference_batch_max_requests":    args.max_requests,
        "inference_batch_max_action_rows": args.max_action_rows,
        "inference_batch_timeout_ms":      args.timeout_ms,
        "use_bf16_learner":                args.use_bf16,
    }
    q_net_kwargs = {
        "hidden_lstm":            args.hidden_lstm,
        "hidden_mlp":              args.hidden_mlp,
        "n_mlp_layers":            args.n_mlp_layers,
        "dropout":                 0.0,
        "use_oracle_others_hand":  True,
    }

    server_proc = ctx.Process(
        target=shared_server_loop_entry,
        args=(
            cfg_dict, q_net_kwargs, meta,
            bufs.free_slots, bufs.request_queue, bufs.events, stop_event,
        ),
        daemon=True, name="inference_server",
    )
    server_proc.start()

    print(f"Server starting on {args.device}; warming up...", flush=True)
    time.sleep(3.0 if args.device != "cpu" else 1.0)

    actor_procs = []
    for i in range(args.n_actors):
        p = ctx.Process(
            target=synthetic_actor_loop,
            args=(
                i, meta,
                bufs.free_slots, bufs.request_queue, bufs.events, stop_event,
                args.duration_s, 1234 + i, "heavy",
            ),
            daemon=True, name=f"synth_actor_{i}",
        )
        p.start()
        actor_procs.append(p)

    print(f"Running {args.n_actors} synthetic actors for {args.duration_s}s...", flush=True)
    t0 = time.perf_counter()

    for p in actor_procs:
        p.join(timeout=args.duration_s + 30.0)

    wall_s = time.perf_counter() - t0

    stop_event.set()
    server_proc.join(timeout=10.0)

    summaries = []
    for i in range(args.n_actors):
        path = Path(f"/tmp/bench_shared_inference_actor_{i}.json")
        if path.exists():
            summaries.append(json.loads(path.read_text()))
            path.unlink()

    total_decisions   = sum(s["n_decisions"] for s in summaries)
    total_action_rows = sum(s["n_action_rows"] for s in summaries)

    result = {
        "device":            args.device,
        "n_actors":          args.n_actors,
        "wall_s":            round(wall_s, 2),
        "total_decisions":   total_decisions,
        "total_action_rows": total_action_rows,
        "decisions_per_s":   round(total_decisions / wall_s, 1) if wall_s else 0.0,
        "rows_per_s":        round(total_action_rows / wall_s, 1) if wall_s else 0.0,
        "wait_ms_p50_med":   round(statistics.median([s["wait_ms_p50"] for s in summaries]), 2) if summaries else 0.0,
        "wait_ms_p95_med":   round(statistics.median([s["wait_ms_p95"] for s in summaries]), 2) if summaries else 0.0,
        "wait_ms_max":       round(max([s["wait_ms_max"] for s in summaries]), 2) if summaries else 0.0,
    }

    release_shared_buffers(bufs, unlink=True)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-actors",         type=int,   default=16)
    ap.add_argument("--device",           type=str,   default="cpu")
    ap.add_argument("--duration-s",       type=float, default=20.0)
    ap.add_argument("--num-slots",        type=int,   default=512)
    ap.add_argument("--max-actions",      type=int,   default=320)
    ap.add_argument("--max-requests",     type=int,   default=32)
    ap.add_argument("--max-action-rows",  type=int,   default=4096)
    ap.add_argument("--timeout-ms",       type=float, default=1.0)
    ap.add_argument("--hidden-lstm",      type=int,   default=256)
    ap.add_argument("--hidden-mlp",       type=int,   default=1024)
    ap.add_argument("--n-mlp-layers",     type=int,   default=6)
    ap.add_argument("--use-bf16",         action="store_true")
    ap.add_argument("--out",              type=str,   default=None,
                    help="Optional JSON path to save the result.")
    args = ap.parse_args()

    result = run_bench(args)

    print()
    print("=" * 60)
    print("Phase 3 shared-inference smoke")
    print("=" * 60)
    for k, v in result.items():
        print(f"  {k:20s} = {v}")
    print()

    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"saved → {args.out}")


if __name__ == "__main__":
    main()
