"""Train QValueNet (policy + value heads) on AZ self-play data (Direction D, Phase 2).

Loss:
  policy_loss = CrossEntropy(softmax(Q[legal]), pi_search)   -- soft target
  value_loss  = MSE(V(state), z)
  total       = policy_loss + 0.5 * value_loss

Initialises from an existing checkpoint (e.g. jidan_policy.pt) with strict=False
so the Q-trunk restores exactly and the V-trunk (new) starts from zero.

Usage:
    PYTHONPATH=ml/src python ml/scripts/train/train_az.py \\
        --data ml/data/selfplay_gen1.npz \\
        --init-checkpoint ml/checkpoints/jidan_policy.pt \\
        --out ml/checkpoints/az_gen1.pt \\
        --epochs 10 --hidden 512

    # smoke test
    PYTHONPATH=ml/src python ml/scripts/train/train_az.py \\
        --data /tmp/selfplay_smoke.npz \\
        --init-checkpoint ml/checkpoints/jidan_policy.pt \\
        --out /tmp/az_smoke.pt \\
        --epochs 1 --hidden 512
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from guandan.training import ACTION_DIM, QValueNet, get_device
from guandan.training.visibility import STATE_DIM_TIER1_TEAM


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data(path: str) -> dict[str, np.ndarray]:
    d = np.load(path)
    return {k: d[k] for k in d.files}


# ── Training step ──────────────────────────────────────────────────────────────

def _step(
    net: QValueNet,
    states: torch.Tensor,      # [B, 480]
    actions: torch.Tensor,     # [B, K, 160]
    pi_search: torch.Tensor,   # [B, K] soft target
    n_cands: torch.Tensor,     # [B] int
    z: torch.Tensor,           # [B]
    train: bool,
    opt=None,
) -> tuple[float, float, float]:
    B, K, _ = actions.shape
    device = states.device

    # Build validity mask from n_cands
    mask = torch.arange(K, device=device).unsqueeze(0) < n_cands.unsqueeze(1)  # [B, K]

    # Policy head: Q(s, a) for each candidate
    states_exp = states.unsqueeze(1).expand(B, K, -1).reshape(B * K, -1)
    actions_flat = actions.reshape(B * K, -1)
    q_flat = net(states_exp, actions_flat).reshape(B, K)           # [B, K]
    q_flat = q_flat.masked_fill(~mask, -1e9)

    # Soft cross-entropy: -sum(pi_search * log_softmax(q))
    log_probs = F.log_softmax(q_flat, dim=-1)                      # [B, K]
    policy_loss = -(pi_search * log_probs).sum(dim=-1).mean()      # scalar

    # Value head
    v_pred = net.value(states)                                      # [B]
    value_loss = F.mse_loss(v_pred, z)                             # scalar

    total_loss = policy_loss + 0.5 * value_loss

    if train:
        opt.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()

    return total_loss.item(), policy_loss.item(), value_loss.item()


# ── Training loop ──────────────────────────────────────────────────────────────

def train(
    data: dict[str, np.ndarray],
    init_checkpoint: str | None,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    hidden: int,
    device: torch.device,
    val_split: float = 0.10,
    early_stop_patience: int = 3,
    seed: int = 0,
) -> tuple[QValueNet, dict]:
    states_np = data["states"]       # [N, 480]
    actions_np = data["actions"]     # [N, K, 160]
    pi_np = data["pi_search"]        # [N, K]
    n_cands_np = data["n_cands"]     # [N]
    z_np = data["z"]                 # [N]
    N = len(states_np)

    net = QValueNet(d_state=STATE_DIM_TIER1_TEAM, d_action=ACTION_DIM, hidden=hidden).to(device)

    if init_checkpoint:
        ckpt = torch.load(init_checkpoint, map_location=device, weights_only=True)
        missing, unexpected = net.load_state_dict(ckpt["state_dict"], strict=False)
        n_params = sum(p.numel() for p in net.parameters())
        print(f"  loaded {init_checkpoint}  (missing={len(missing)}, unexpected={len(unexpected)})")
        print(f"  net: {n_params:,} params on {device} (hidden={hidden})")
    else:
        n_params = sum(p.numel() for p in net.parameters())
        print(f"  net: {n_params:,} params on {device} (hidden={hidden}, no init checkpoint)")

    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.05)

    rng = random.Random(seed)
    indices = list(range(N))
    rng.shuffle(indices)
    n_val = int(N * val_split)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    print(f"  split: {len(train_idx)} train / {len(val_idx)} val")
    print(f"  z stats: mean={z_np.mean():.3f} std={z_np.std():.3f}")

    def to_tensors(idx_list):
        return (
            torch.from_numpy(states_np[idx_list]).to(device),
            torch.from_numpy(actions_np[idx_list]).to(device),
            torch.from_numpy(pi_np[idx_list]).to(device),
            torch.from_numpy(n_cands_np[idx_list].astype(np.int64)).to(device),
            torch.from_numpy(z_np[idx_list]).to(device),
        )

    metrics = {"epochs": []}
    t0 = time.time()
    best_val_loss = float("inf")
    best_state = None
    patience_count = 0

    for epoch in range(epochs):
        rng.shuffle(train_idx)
        net.train()
        ep_total = ep_policy = ep_value = 0.0
        n_batches = 0

        for bs in range(0, len(train_idx), batch_size):
            batch_idx = train_idx[bs: bs + batch_size]
            st, ac, pi, nc, zb = to_tensors(batch_idx)
            tl, pl, vl = _step(net, st, ac, pi, nc, zb, train=True, opt=opt)
            ep_total += tl
            ep_policy += pl
            ep_value += vl
            n_batches += 1

        net.eval()
        val_total = val_policy = val_value = 0.0
        v_batches = 0
        with torch.no_grad():
            for bs in range(0, len(val_idx), batch_size):
                batch_idx = val_idx[bs: bs + batch_size]
                st, ac, pi, nc, zb = to_tensors(batch_idx)
                tl, pl, vl = _step(net, st, ac, pi, nc, zb, train=False)
                val_total += tl
                val_policy += pl
                val_value += vl
                v_batches += 1

        cur_lr = opt.param_groups[0]["lr"]
        sched.step()
        elapsed = time.time() - t0

        tl_avg = ep_total / max(1, n_batches)
        vl_avg = val_total / max(1, v_batches)
        improved = vl_avg < best_val_loss - 1e-4
        if improved:
            best_val_loss = vl_avg
            best_state = deepcopy(net.state_dict())
            patience_count = 0
            tag = "  (best)"
        else:
            patience_count += 1
            tag = f"  (no-improve {patience_count}/{early_stop_patience})"

        print(
            f"  epoch {epoch+1:2d}/{epochs}  "
            f"train={tl_avg:.4f} (p={ep_policy/max(1,n_batches):.4f} v={ep_value/max(1,n_batches):.4f})  "
            f"val={vl_avg:.4f} (p={val_policy/max(1,v_batches):.4f} v={val_value/max(1,v_batches):.4f})  "
            f"lr={cur_lr:.2e}  ({elapsed:.0f}s){tag}",
            flush=True,
        )

        metrics["epochs"].append({
            "epoch": epoch + 1,
            "train_loss": tl_avg, "train_policy": ep_policy / max(1, n_batches),
            "train_value": ep_value / max(1, n_batches),
            "val_loss": vl_avg, "val_policy": val_policy / max(1, v_batches),
            "val_value": val_value / max(1, v_batches),
            "lr": cur_lr, "elapsed_s": elapsed,
        })

        if patience_count >= early_stop_patience:
            print(f"  early stop at epoch {epoch+1}")
            break

    if best_state is not None:
        net.load_state_dict(best_state)

    metrics["n_train"] = len(train_idx)
    metrics["n_val"] = len(val_idx)
    metrics["best_val_loss"] = best_val_loss
    metrics["total_train_time_s"] = time.time() - t0
    return net, metrics


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="AZ training: policy + value heads")
    p.add_argument("--data", required=True, help="selfplay .npz file")
    p.add_argument("--init-checkpoint", default=None,
                   help="Checkpoint to initialise Q-trunk from (e.g. jidan_policy.pt)")
    p.add_argument("--out", required=True, help="Output checkpoint path")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    print("AZ training: policy + value heads")
    print(f"  data: {args.data}")
    print(f"  init: {args.init_checkpoint or '(none)'}")
    print(f"  epochs={args.epochs}  batch={args.batch_size}  lr={args.lr}  "
          f"hidden={args.hidden}", flush=True)

    data = load_data(args.data)
    N = len(data["states"])
    K = data["actions"].shape[1]
    print(f"  loaded {N} decisions (K={K})")

    device = get_device()
    net, metrics = train(
        data=data,
        init_checkpoint=args.init_checkpoint,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden=args.hidden,
        device=device,
        early_stop_patience=args.patience,
        seed=args.seed,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": net.state_dict(),
        "config": {
            "d_state": STATE_DIM_TIER1_TEAM,
            "d_action": ACTION_DIM,
            "hidden": args.hidden,
        },
        "metrics": metrics,
        "args": vars(args),
    }, out)
    print(f"\nSaved → {out}")
    final = metrics["epochs"][-1]
    print(f"Final: val={final['val_loss']:.4f}  "
          f"(policy={final['val_policy']:.4f}  value={final['val_value']:.4f})  "
          f"(best val={metrics['best_val_loss']:.4f})")


if __name__ == "__main__":
    main()
