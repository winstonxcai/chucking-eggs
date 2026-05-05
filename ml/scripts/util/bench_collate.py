"""Microbench: collate_encoded variants on MPS / CPU.

Variants:
  v0_current   — np.stack per key, then .to(device) per key (today's code)
  v1_nonblock  — same, but .to(device, non_blocking=True)
  v2_flatone   — np.stack per key, np.concatenate to (B, FLAT_TOTAL), single .to,
                 split + reshape on-device
  v3_flatone_contig — like v2, but slice + .contiguous().reshape (forces clean view)

Picks the winner before we modify production code.
"""

from __future__ import annotations

import argparse
import time
import numpy as np
import torch


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


def make_encoded(B: int) -> list[dict]:
    return [
        {k: np.zeros(_SHAPES[k], dtype=np.float32) for k in _KEYS}
        for _ in range(B)
    ]


def v0_current(encoded_list, device):
    keys = encoded_list[0].keys()
    out = {}
    for k in keys:
        arr = np.stack([e[k] for e in encoded_list], axis=0)
        out[k] = torch.from_numpy(arr).to(device)
    return out


def v1_nonblock(encoded_list, device):
    keys = encoded_list[0].keys()
    out = {}
    for k in keys:
        arr = np.stack([e[k] for e in encoded_list], axis=0)
        out[k] = torch.from_numpy(arr).to(device, non_blocking=True)
    return out


def v2_flatone(encoded_list, device):
    B = len(encoded_list)
    arrays = {k: np.stack([e[k] for e in encoded_list], axis=0) for k in _KEYS}
    flat = np.concatenate([arrays[k].reshape(B, -1) for k in _KEYS], axis=1)
    flat_t = torch.from_numpy(flat).to(device)
    out = {}
    off = 0
    for k in _KEYS:
        sz = int(np.prod(_SHAPES[k]))
        out[k] = flat_t[:, off:off+sz].reshape((B,) + _SHAPES[k])
        off += sz
    return out


def v3_flatone_contig(encoded_list, device):
    B = len(encoded_list)
    arrays = {k: np.stack([e[k] for e in encoded_list], axis=0) for k in _KEYS}
    flat = np.concatenate([arrays[k].reshape(B, -1) for k in _KEYS], axis=1)
    flat_t = torch.from_numpy(flat).to(device)
    out = {}
    off = 0
    for k in _KEYS:
        sz = int(np.prod(_SHAPES[k]))
        out[k] = flat_t[:, off:off+sz].contiguous().reshape((B,) + _SHAPES[k])
        off += sz
    return out


def v4_persample_pack(encoded_list, device):
    """Pack each sample into one flat buffer, np.stack across, single H2D."""
    B = len(encoded_list)
    sizes = [int(np.prod(_SHAPES[k])) for k in _KEYS]
    total = sum(sizes)
    flat_np = np.empty((B, total), dtype=np.float32)
    for i, e in enumerate(encoded_list):
        off = 0
        for k_idx, k in enumerate(_KEYS):
            sz = sizes[k_idx]
            flat_np[i, off:off+sz] = e[k].ravel()
            off += sz
    flat_t = torch.from_numpy(flat_np).to(device)
    out = {}
    off = 0
    for k_idx, k in enumerate(_KEYS):
        sz = sizes[k_idx]
        out[k] = flat_t[:, off:off+sz].reshape((B,) + _SHAPES[k])
        off += sz
    return out


VARIANTS = {
    "v0_current": v0_current,
    "v1_nonblock": v1_nonblock,
    "v2_flatone": v2_flatone,
    "v3_flatone_contig": v3_flatone_contig,
    "v4_persample_pack": v4_persample_pack,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--repeats", type=int, default=2000)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--batches", default="2,5,8,16,32",
                    help="batch sizes (i.e. # candidate actions)")
    args = ap.parse_args()

    torch.set_num_threads(1)
    device = torch.device(args.device)

    def _sync():
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

    print(f"collate_encoded variants  device={args.device}  repeats={args.repeats}")
    for B in [int(x) for x in args.batches.split(",")]:
        enc = make_encoded(B)
        print(f"\n  B = {B:>3d}")
        print(f"  {'variant':22s} {'ms/call':>10s} {'us/sample':>10s}")
        for name, fn in VARIANTS.items():
            for _ in range(args.warmup):
                _ = fn(enc, device)
            _sync()
            t0 = time.perf_counter()
            for _ in range(args.repeats):
                _ = fn(enc, device)
            _sync()
            t = time.perf_counter() - t0
            ms = 1000 * t / args.repeats
            us = 1e6 * t / (args.repeats * B)
            print(f"  {name:22s} {ms:>10.3f} {us:>10.1f}")


if __name__ == "__main__":
    main()
