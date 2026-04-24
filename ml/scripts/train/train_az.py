"""Train QValueNet (policy + value heads) on AZ self-play data (Direction D, Phase 2).

Loss:
  policy_loss = soft cross-entropy(softmax(Q[legal]), pi_search target)
  value_loss  = MSE(V(state), z)
  total       = policy_loss + 0.5 * value_loss

Per-epoch diagnostics:
  • train/val total/policy/value losses
  • policy entropy (val) — catches confidence collapse
  • policy argmax accuracy vs π_search argmax (val)
  • value MAE (val)
  • inline 50-game policy-only eval vs Jidan (real-task signal)

Initialises from an existing checkpoint with strict=False so the Q-trunk
restores exactly and the V-trunk (new) starts zero-initialised.

Usage:
    PYTHONPATH=ml/src python ml/scripts/train/train_az.py \\
        --data ml/data/selfplay_gen1.npz \\
        --init-checkpoint ml/checkpoints/jidan_policy.pt \\
        --out ml/checkpoints/az_gen1.pt \\
        --epochs 10 --hidden 512 --eval-games 50
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from guandan.training import ACTION_DIM, QValueNet, encode_action, get_device
from guandan.training.visibility import STATE_DIM_TIER1_TEAM, encode_state_tier1_team


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data(path: str) -> dict[str, np.ndarray]:
    d = np.load(path)
    return {k: d[k] for k in d.files}


# ── Training step ──────────────────────────────────────────────────────────────

def _train_step(
    net: QValueNet,
    states: torch.Tensor,      # [B, 480]
    actions: torch.Tensor,     # [B, K, 160]
    pi_search: torch.Tensor,   # [B, K] soft target
    n_cands: torch.Tensor,     # [B] int
    z: torch.Tensor,           # [B]
    opt: torch.optim.Optimizer,
) -> tuple[float, float, float]:
    B, K, _ = actions.shape
    mask = torch.arange(K, device=states.device).unsqueeze(0) < n_cands.unsqueeze(1)

    states_exp = states.unsqueeze(1).expand(B, K, -1).reshape(B * K, -1)
    q_flat = net(states_exp, actions.reshape(B * K, -1)).reshape(B, K)
    q_flat = q_flat.masked_fill(~mask, -1e9)
    log_probs = F.log_softmax(q_flat, dim=-1)
    policy_loss = -(pi_search * log_probs).sum(dim=-1).mean()

    v_pred = net.value(states)
    value_loss = F.mse_loss(v_pred, z)

    total = policy_loss + 0.5 * value_loss
    opt.zero_grad()
    total.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    opt.step()
    return total.item(), policy_loss.item(), value_loss.item()


@torch.no_grad()
def _eval_step(
    net: QValueNet,
    states: torch.Tensor,
    actions: torch.Tensor,
    pi_search: torch.Tensor,
    n_cands: torch.Tensor,
    z: torch.Tensor,
) -> dict:
    """Returns batched val metrics (sums, divide by N at end)."""
    B, K, _ = actions.shape
    mask = torch.arange(K, device=states.device).unsqueeze(0) < n_cands.unsqueeze(1)

    states_exp = states.unsqueeze(1).expand(B, K, -1).reshape(B * K, -1)
    q_flat = net(states_exp, actions.reshape(B * K, -1)).reshape(B, K)
    q_flat = q_flat.masked_fill(~mask, -1e9)
    log_probs = F.log_softmax(q_flat, dim=-1)
    probs = log_probs.exp()

    policy_loss = -(pi_search * log_probs).sum(dim=-1).sum().item()  # sum over batch

    # Policy entropy on each sample (mean over batch later)
    entropy_per = -(probs * log_probs).masked_fill(~mask, 0).sum(dim=-1)  # [B]
    entropy_sum = entropy_per.sum().item()

    # Policy argmax accuracy: predicted argmax matches π_search argmax?
    pred = q_flat.argmax(dim=-1)
    target = pi_search.argmax(dim=-1)
    correct = (pred == target).sum().item()

    v_pred = net.value(states)
    value_loss = F.mse_loss(v_pred, z, reduction="sum").item()
    value_mae = (v_pred - z).abs().sum().item()

    return {
        "policy_loss_sum": policy_loss,
        "value_loss_sum": value_loss,
        "value_mae_sum": value_mae,
        "entropy_sum": entropy_sum,
        "correct": correct,
        "n": B,
    }


# ── Inline eval vs Jidan (policy-only argmax) ──────────────────────────────────

def _eval_vs_jidan(
    net: QValueNet,
    device: torch.device,
    n_games: int,
    seed: int,
) -> tuple[float, float]:
    """Play n_games policy-only argmax vs JidanBot, alternating seats. Returns (WR, sec)."""
    from guandan.agents import JidanBot
    from guandan.agents.partner_oracle_bot import _reflect_env
    from guandan.cards import Rank
    from guandan.game import GuanDanEnv

    jidan = JidanBot(level_rank=Rank.TWO)
    env = GuanDanEnv(level_rank=Rank.TWO)
    rng = random.Random(seed)

    @torch.no_grad()
    def policy_act(env, player):
        legal = env.legal_moves(player)
        if len(legal) == 1:
            return legal[0]
        if player in (0, 2):
            enc_env, enc_player = env, player
        else:
            enc_env, enc_player = _reflect_env(env), player ^ 1
        state = encode_state_tier1_team(enc_env, enc_player).astype(np.float32)
        actions = np.stack([
            encode_action(a, enc_env.hands[enc_player], env.level_rank)
            for a in legal
        ]).astype(np.float32)
        st = torch.from_numpy(state).to(device).unsqueeze(0).expand(len(legal), -1)
        at = torch.from_numpy(actions).to(device)
        q = net(st, at).cpu().numpy()
        return legal[int(np.argmax(q))]

    wins = 0
    t0 = time.time()
    for i in range(n_games):
        env.reset(seed=rng.randint(0, 1 << 30))
        a1_seats = {0, 2} if i % 2 == 0 else {1, 3}
        while not env.done:
            p = env.current_player
            move = policy_act(env, p) if p in a1_seats else jidan.act(env, p)
            env.step(move)
        if env.get_rewards()[next(iter(a1_seats))] > 0:
            wins += 1
    elapsed = time.time() - t0
    return wins / n_games, elapsed


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
    eval_games: int,
    eval_every: int,
    val_split: float = 0.10,
    early_stop_patience: int = 3,
    seed: int = 0,
    out_path: str | None = None,
) -> tuple[QValueNet, dict]:
    states_np = data["states"]
    actions_np = data["actions"]
    pi_np = data["pi_search"]
    n_cands_np = data["n_cands"]
    z_np = data["z"]
    N = len(states_np)

    net = QValueNet(d_state=STATE_DIM_TIER1_TEAM, d_action=ACTION_DIM, hidden=hidden).to(device)
    if init_checkpoint:
        ckpt = torch.load(init_checkpoint, map_location=device, weights_only=True)
        missing, unexpected = net.load_state_dict(ckpt["state_dict"], strict=False)
        print(f"  loaded {init_checkpoint}  (missing={len(missing)}, unexpected={len(unexpected)})")
    n_params = sum(p.numel() for p in net.parameters())
    print(f"  net: {n_params:,} params on {device} (hidden={hidden})")

    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.05)

    rng = random.Random(seed)
    indices = list(range(N))
    rng.shuffle(indices)
    n_val = int(N * val_split)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    print(f"  split: {len(train_idx)} train / {len(val_idx)} val")
    print(f"  z stats: mean={z_np.mean():+.3f}  std={z_np.std():.3f}  "
          f"  baseline_value_loss={(z_np**2).mean():.3f} (V≡0)")

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

    # Initial eval (epoch 0): how good is the prior?
    if eval_games > 0:
        net.eval()
        wr0, t_eval = _eval_vs_jidan(net, device, eval_games, seed=seed)
        print(f"  baseline (epoch 0): WR vs Jidan (policy-only, {eval_games} games) = "
              f"{wr0:.1%}  ({t_eval:.0f}s)", flush=True)

    for epoch in range(epochs):
        rng.shuffle(train_idx)
        net.train()
        ep_total = ep_policy = ep_value = 0.0
        n_batches = 0
        for bs in range(0, len(train_idx), batch_size):
            st, ac, pi, nc, zb = to_tensors(train_idx[bs: bs + batch_size])
            tl, pl, vl = _train_step(net, st, ac, pi, nc, zb, opt)
            ep_total += tl
            ep_policy += pl
            ep_value += vl
            n_batches += 1
        train_total = ep_total / max(1, n_batches)
        train_policy = ep_policy / max(1, n_batches)
        train_value = ep_value / max(1, n_batches)

        # Val
        net.eval()
        agg = {"policy_loss_sum": 0, "value_loss_sum": 0, "value_mae_sum": 0,
               "entropy_sum": 0, "correct": 0, "n": 0}
        for bs in range(0, len(val_idx), batch_size):
            st, ac, pi, nc, zb = to_tensors(val_idx[bs: bs + batch_size])
            m = _eval_step(net, st, ac, pi, nc, zb)
            for k in agg:
                agg[k] += m[k]
        n = max(1, agg["n"])
        val_policy = agg["policy_loss_sum"] / n
        val_value = agg["value_loss_sum"] / n
        val_total = val_policy + 0.5 * val_value
        val_mae = agg["value_mae_sum"] / n
        val_entropy = agg["entropy_sum"] / n
        val_acc = agg["correct"] / n

        # Inline eval vs Jidan
        eval_str = ""
        wr_jidan = None
        if eval_games > 0 and (epoch + 1) % eval_every == 0:
            wr_jidan, t_eval = _eval_vs_jidan(net, device, eval_games, seed=seed + epoch + 1)
            eval_str = f"  WR_jidan={wr_jidan:.1%} ({t_eval:.0f}s)"

        cur_lr = opt.param_groups[0]["lr"]
        sched.step()
        elapsed = time.time() - t0

        improved = val_total < best_val_loss - 1e-4
        if improved:
            best_val_loss = val_total
            best_state = deepcopy(net.state_dict())
            patience_count = 0
            tag = "  (best)"
        else:
            patience_count += 1
            tag = f"  (no-improve {patience_count}/{early_stop_patience})"

        print(
            f"  ep {epoch+1:2d}/{epochs}  "
            f"tr={train_total:.3f}(p={train_policy:.3f} v={train_value:.3f})  "
            f"val={val_total:.3f}(p={val_policy:.3f} v={val_value:.3f})  "
            f"H={val_entropy:.3f} acc={val_acc:.2f} v_mae={val_mae:.2f}  "
            f"lr={cur_lr:.1e}  ({elapsed:.0f}s){eval_str}{tag}",
            flush=True,
        )

        metrics["epochs"].append({
            "epoch": epoch + 1,
            "train_total": train_total, "train_policy": train_policy, "train_value": train_value,
            "val_total": val_total, "val_policy": val_policy, "val_value": val_value,
            "val_entropy": val_entropy, "val_acc": val_acc, "val_mae": val_mae,
            "wr_jidan": wr_jidan,
            "lr": cur_lr, "elapsed_s": elapsed,
        })

        # Save latest checkpoint (so we can recover if interrupted)
        if out_path:
            latest_path = Path(out_path).with_name(Path(out_path).stem + "_latest.pt")
            torch.save({
                "state_dict": net.state_dict(),
                "config": {"d_state": STATE_DIM_TIER1_TEAM, "d_action": ACTION_DIM, "hidden": hidden},
                "metrics": metrics, "epoch": epoch + 1,
            }, latest_path)

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
    p.add_argument("--data", required=True)
    p.add_argument("--init-checkpoint", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-games", type=int, default=50,
                   help="Inline policy-only Jidan eval games per epoch (0 = disabled)")
    p.add_argument("--eval-every", type=int, default=1,
                   help="Run inline Jidan eval every N epochs (default 1)")
    args = p.parse_args()

    print("AZ training: policy + value heads")
    print(f"  data: {args.data}")
    print(f"  init: {args.init_checkpoint or '(none)'}")
    print(f"  epochs={args.epochs}  batch={args.batch_size}  lr={args.lr}  "
          f"hidden={args.hidden}  eval_games={args.eval_games}", flush=True)

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
        eval_games=args.eval_games,
        eval_every=args.eval_every,
        early_stop_patience=args.patience,
        seed=args.seed,
        out_path=args.out,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": net.state_dict(),
        "config": {"d_state": STATE_DIM_TIER1_TEAM, "d_action": ACTION_DIM, "hidden": args.hidden},
        "metrics": metrics,
        "args": vars(args),
    }, out)
    print(f"\nSaved → {out}")
    final = metrics["epochs"][-1]
    print(f"Final ep: val={final['val_total']:.3f}  "
          f"acc={final['val_acc']:.2f}  H={final['val_entropy']:.3f}  "
          f"WR_jidan={final.get('wr_jidan')}  (best val={metrics['best_val_loss']:.3f})")


if __name__ == "__main__":
    main()
