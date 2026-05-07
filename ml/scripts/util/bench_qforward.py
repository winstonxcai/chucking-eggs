"""Q-network forward throughput vs batch size.

Answers: at what batch size does the paper-spec network amortize torch
dispatch overhead? If t(batch=64) ≈ t(batch=4) × 16 the model is compute-bound
and small batches are unavoidable. If t(batch=64) ≈ t(batch=4) the model is
dispatch-bound and cross-decision batching has high leverage.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from guandan.guanzero.q_network import init_seat_nets
from guandan.guanzero.config import QNetConfig
    from guandan.guanzero.config import QNetConfig

_KEYS = (
    "own_hand", "others_hand", "recent_action_each_player",
    "played_cards_others", "remaining_counts_others",
    "level", "history", "behavior", "candidate_action",
)
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


def make_batch(batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
    out = {}
    for k in _KEYS:
        shape = (batch_size,) + _SHAPES[k]
        out[k] = torch.zeros(shape, dtype=torch.float32, device=device)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--hidden-lstm", type=int, default=256)
    ap.add_argument("--hidden-mlp", type=int, default=1024)
    ap.add_argument("--n-mlp-layers", type=int, default=6)
    ap.add_argument("--repeats", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--batch-sizes", default="1,2,4,8,16,32,64,128,256")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    device = torch.device(args.device)

    nets = init_seat_nets(QNetConfig(hidden_lstm=args.hidden_lstm, hidden_mlp=args.hidden_mlp, n_mlp_layers=args.n_mlp_layers))
    net = nets[0].to(device)
    net.eval()

    print(f"Q-forward throughput  device={args.device} threads={args.threads}")
    print(f"  net: LSTM {args.hidden_lstm} + MLP {args.hidden_mlp}x{args.n_mlp_layers}")
    print(f"{'batch':>6s} {'sec_total':>10s} {'ms/call':>10s} {'us/sample':>10s} {'samples/s':>12s}")

    sizes = [int(s) for s in args.batch_sizes.split(",")]
    for bs in sizes:
        batch = make_batch(bs, device)
        def _sync():
            if device.type == "cuda":
                torch.cuda.synchronize()
            elif device.type == "mps":
                torch.mps.synchronize()
        with torch.no_grad():
            for _ in range(args.warmup):
                _ = net(batch)
            _sync()
            t0 = time.perf_counter()
            for _ in range(args.repeats):
                _ = net(batch)
            _sync()
            t = time.perf_counter() - t0
        ms_call = 1000 * t / args.repeats
        us_sample = 1e6 * t / (args.repeats * bs)
        sps = args.repeats * bs / t
        print(f"{bs:>6d} {t:>10.3f} {ms_call:>10.3f} {us_sample:>10.1f} {sps:>12.0f}")


if __name__ == "__main__":
    main()
