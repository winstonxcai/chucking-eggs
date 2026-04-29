"""Distill JidanBot into a partner-visible Q-network policy (Direction C, Stage 1).

Pipeline:
  1. Generate (state, legal_actions, jidan_pick_idx) samples by self-playing
     Jidan-vs-Jidan games in parallel worker processes. State uses
     encode_state_team (480-dim, both teammate hands visible). Only
     collect decisions from team {0,2}; seat reflection happens at inference.
  2. Train QNetwork(state ⊕ action → Q) with cross-entropy loss over softmax
     of Q-values per legal-action set, target = one-hot of Jidan's pick.
     AdamW + cosine LR + weight decay. Early stopping on val loss.
  3. Save best checkpoint to ml/checkpoints/jidan_policy.pt.

Usage:
    PYTHONPATH=ml/src python ml/scripts/train/distill_jidan.py \
        --decisions 100000 --epochs 10 --workers 4

    # smoke test (~1 min)
    PYTHONPATH=ml/src python ml/scripts/train/distill_jidan.py \
        --decisions 2000 --epochs 2 --workers 2
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from guandan.agents import JidanBot
from guandan.cards import Rank
from guandan.combos import Combo
from guandan.game import GuanDanEnv
from guandan.pvguan.legal_utils import dedup_strategic, strategic_key
from guandan.azguan import (
    ACTION_DIM,
    QNetwork,
    STATE_DIM_TEAM_WITH_FLAGS,
    encode_action,
    encode_state_team_with_flags,
    get_device,
)

# Cap legal moves per sample to bound batch padding / memory.
# Top-K kept by rank_sum heuristic; tail dropped. Jidan rarely picks deep tail.
MAX_LEGAL_PER_SAMPLE = 64


def _combo_key(combo: Combo) -> tuple:
    return (
        int(combo.type),
        tuple(sorted((c.rank, c.suit, c.deck) for c in combo.cards)),
    )


def _gen_worker(args: tuple) -> list[dict]:
    """Worker: generate `n_decisions` samples in this process."""
    n_decisions, level_rank, seed = args
    random.seed(seed)
    env = GuanDanEnv(level_rank=level_rank)
    jidan = JidanBot(level_rank=level_rank)

    samples: list[dict] = []
    while len(samples) < n_decisions:
        env.reset()
        while not env.done:
            player = env.current_player
            legal = dedup_strategic(env.legal_moves(player))
            pick = jidan.act(env, player)

            if player in (0, 2) and len(legal) > 1:
                pick_key = strategic_key(pick)
                pick_idx_full = next(
                    i for i, a in enumerate(legal) if strategic_key(a) == pick_key
                )

                # Cap legal to top-K by rank_sum, but always keep Jidan's pick.
                if len(legal) > MAX_LEGAL_PER_SAMPLE:
                    scored = sorted(
                        range(len(legal)),
                        key=lambda i: sum(c.rank for c in legal[i].cards),
                    )
                    keep_idx = scored[:MAX_LEGAL_PER_SAMPLE]
                    if pick_idx_full not in keep_idx:
                        keep_idx[-1] = pick_idx_full  # swap in Jidan's pick
                    keep_idx = sorted(keep_idx)
                    capped_legal = [legal[i] for i in keep_idx]
                    pick_idx = keep_idx.index(pick_idx_full)
                else:
                    capped_legal = legal
                    pick_idx = pick_idx_full

                states = np.stack([
                    encode_state_team_with_flags(env, player, a, capped_legal)
                    for a in capped_legal
                ]).astype(np.float32)
                actions = np.stack([
                    encode_action(a, env.hands[player], env.level_rank)
                    for a in capped_legal
                ]).astype(np.float32)
                samples.append({
                    "states": states,
                    "actions": actions,
                    "pick_idx": pick_idx,
                })

            env.step(pick)

    return samples[:n_decisions]


def generate_data_parallel(
    n_decisions: int,
    workers: int = 4,
    level_rank: int = Rank.TWO,
    seed: int = 0,
) -> list[dict]:
    """Spawn `workers` worker procs, each generating n_decisions/workers samples."""
    per_worker = math.ceil(n_decisions / workers)
    print(f"  data gen: {workers} workers × {per_worker} samples each "
          f"(target {n_decisions})", flush=True)

    t0 = time.time()
    args_list = [(per_worker, level_rank, seed + w) for w in range(workers)]

    if workers == 1:
        all_samples = _gen_worker(args_list[0])
    else:
        with mp.get_context("spawn").Pool(workers) as pool:
            results = pool.map(_gen_worker, args_list)
        all_samples = [s for chunk in results for s in chunk]

    elapsed = time.time() - t0
    print(f"  data gen: {len(all_samples)} samples in {elapsed:.1f}s "
          f"({len(all_samples)/elapsed:.0f} samples/s)", flush=True)
    return all_samples[:n_decisions]


def _step(net, batch, device, train: bool, opt=None) -> tuple[float, int, int]:
    B = len(batch)
    max_a = max(s["actions"].shape[0] for s in batch)
    d_state = batch[0]["states"].shape[1]

    states = torch.zeros(B, max_a, d_state, dtype=torch.float32, device=device)
    actions = torch.zeros(B, max_a, ACTION_DIM, dtype=torch.float32, device=device)
    mask = torch.zeros(B, max_a, dtype=torch.bool, device=device)
    targets = torch.zeros(B, dtype=torch.long, device=device)

    for i, s in enumerate(batch):
        k = s["actions"].shape[0]
        states[i, :k] = torch.from_numpy(s["states"])
        actions[i, :k] = torch.from_numpy(s["actions"])
        mask[i, :k] = True
        targets[i] = s["pick_idx"]

    states_flat = states.reshape(B * max_a, d_state)
    actions_flat = actions.reshape(B * max_a, ACTION_DIM)
    q_flat = net(states_flat, actions_flat).reshape(B, max_a)
    q_flat = q_flat.masked_fill(~mask, -1e9)

    loss = F.cross_entropy(q_flat, targets)

    if train:
        opt.zero_grad()
        loss.backward()
        opt.step()

    pred = q_flat.argmax(dim=-1)
    correct = (pred == targets).sum().item()
    return loss.item(), correct, B


def train(
    samples: list[dict],
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    hidden: int,
    device: torch.device,
    val_split: float = 0.10,
    early_stop_patience: int = 2,
    seed: int = 0,
) -> tuple[QNetwork, dict]:
    net = QNetwork(
        d_state=STATE_DIM_TEAM_WITH_FLAGS, d_action=ACTION_DIM, hidden=hidden
    ).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.05)

    n_params = sum(p.numel() for p in net.parameters())
    print(f"  net: {n_params:,} params on {device} (hidden={hidden})")

    rng = random.Random(seed)
    n = len(samples)
    indices = list(range(n))
    rng.shuffle(indices)
    n_val = int(n * val_split)
    val_idx = set(indices[:n_val])
    train_idx = [i for i in indices if i not in val_idx]
    val_idx_list = list(val_idx)
    print(f"  split: {len(train_idx)} train / {len(val_idx_list)} val")

    metrics = {"epochs": []}
    t0 = time.time()
    best_val_loss = float("inf")
    best_state = None
    epochs_since_improve = 0

    for epoch in range(epochs):
        rng.shuffle(train_idx)
        net.train()
        ep_loss = 0.0
        ep_correct = 0
        ep_total = 0
        n_batches = 0

        for bs in range(0, len(train_idx), batch_size):
            batch = [samples[i] for i in train_idx[bs : bs + batch_size]]
            loss, correct, total = _step(net, batch, device, train=True, opt=opt)
            ep_loss += loss
            ep_correct += correct
            ep_total += total
            n_batches += 1

        train_loss = ep_loss / n_batches
        train_acc = ep_correct / ep_total

        net.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        v_batches = 0
        with torch.no_grad():
            for bs in range(0, len(val_idx_list), batch_size):
                batch = [samples[i] for i in val_idx_list[bs : bs + batch_size]]
                if not batch:
                    continue
                loss, correct, total = _step(net, batch, device, train=False)
                val_loss += loss
                val_correct += correct
                val_total += total
                v_batches += 1
        val_loss /= max(1, v_batches)
        val_acc = val_correct / max(1, val_total)
        cur_lr = opt.param_groups[0]["lr"]
        sched.step()

        elapsed = time.time() - t0
        improved = val_loss < best_val_loss - 1e-4
        if improved:
            best_val_loss = val_loss
            best_state = deepcopy(net.state_dict())
            epochs_since_improve = 0
            tag = "  (best)"
        else:
            epochs_since_improve += 1
            tag = f"  (no-improve {epochs_since_improve}/{early_stop_patience})"

        print(f"  epoch {epoch+1:2d}/{epochs}  "
              f"train_loss={train_loss:.4f} acc={train_acc:.3f}  "
              f"val_loss={val_loss:.4f} acc={val_acc:.3f}  "
              f"lr={cur_lr:.2e}  ({elapsed:.0f}s){tag}", flush=True)

        metrics["epochs"].append({
            "epoch": epoch + 1,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
            "lr": cur_lr, "elapsed_s": elapsed,
        })

        if epochs_since_improve >= early_stop_patience:
            print(f"  early stop at epoch {epoch+1}", flush=True)
            break

    if best_state is not None:
        net.load_state_dict(best_state)

    metrics["n_train"] = len(train_idx)
    metrics["n_val"] = len(val_idx_list)
    metrics["best_val_loss"] = best_val_loss
    metrics["total_train_time_s"] = time.time() - t0
    return net, metrics


def main() -> None:
    p = argparse.ArgumentParser(description="Distill Jidan policy (Direction C, Stage 1)")
    p.add_argument("--decisions", type=int, default=100000)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="ml/checkpoints/jidan_policy.pt")
    args = p.parse_args()

    print(f"Direction C Stage 1: distill Jidan → policy net")
    print(f"  config: decisions={args.decisions} epochs={args.epochs} "
          f"batch_size={args.batch_size} lr={args.lr} wd={args.weight_decay} "
          f"hidden={args.hidden} workers={args.workers}")

    print("\n[1/2] Generating data ...")
    samples = generate_data_parallel(
        args.decisions, workers=args.workers, seed=args.seed
    )

    print("\n[2/2] Training ...")
    device = get_device()
    net, metrics = train(
        samples, args.epochs, args.batch_size, args.lr, args.weight_decay,
        args.hidden, device,
        early_stop_patience=args.patience, seed=args.seed,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": net.state_dict(),
        "config": {
            "d_state": STATE_DIM_TEAM_WITH_FLAGS,
            "d_action": ACTION_DIM,
            "hidden": args.hidden,
            "level_rank": int(Rank.TWO),
            "max_legal_per_sample": MAX_LEGAL_PER_SAMPLE,
        },
        "metrics": metrics,
        "args": vars(args),
    }, out_path)
    print(f"\nSaved → {out_path}")
    final = metrics["epochs"][-1]
    print(f"Final: val_loss={final['val_loss']:.4f}  val_acc={final['val_acc']:.3f}  "
          f"(best val_loss={metrics['best_val_loss']:.4f})")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
