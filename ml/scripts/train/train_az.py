"""Train QValueNet (policy + value heads) on AZ self-play data (Direction D, Phase 2).

Loss:
  policy_loss = soft cross-entropy(softmax(Q[legal]), pi_smooth target)
  value_loss  = MSE(V(state), z)
  total       = policy_loss + 0.5 * value_loss
  where pi_smooth = (1-ε)*pi_search + ε/n_cands  (label smoothing, default ε=0.1)

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
import json
import logging
import math
import random
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from guandan.azguan import (
    ACTION_DIM,
    QValueNet,
    STATE_DIM_TEAM_WITH_FLAGS,
    encode_action,
    encode_state_team_with_flags,
    get_device,
)


# ── Logging setup ──────────────────────────────────────────────────────────────

def _setup_logging(run_dir: Path) -> logging.Logger:
    run_dir.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s  %(levelname)-8s %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    log = logging.getLogger("train_az")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fh = logging.FileHandler(run_dir / "train.log")
    fh.setFormatter(logging.Formatter(fmt, datefmt))
    log.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter(fmt, datefmt))
    log.addHandler(sh)
    return log


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data(path: str) -> dict[str, np.ndarray]:
    d = np.load(path)
    return {k: d[k] for k in d.files}


# ── Training step ──────────────────────────────────────────────────────────────

RANK_MARGIN = 1.0   # Q(top-K min) must exceed Q(neg max) by this margin
RANK_WEIGHT = 0.5   # weight of ranking loss in total
LABEL_SMOOTH = 0.1  # label smoothing ε: prevents policy collapse from peaked V-at-leaf targets


def _smooth_targets(pi: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Apply label smoothing: π_smooth = (1-ε)*π + ε/n_cands over valid candidates."""
    n_cands = mask.sum(dim=-1, keepdim=True).float().clamp(min=1)
    uniform = mask.float() / n_cands
    return (1 - LABEL_SMOOTH) * pi + LABEL_SMOOTH * uniform


def _train_step(
    net: QValueNet,
    base_states: torch.Tensor,   # [B, 480] — for V head
    cand_states: torch.Tensor,   # [B, K, 489] — per-candidate state for Q
    actions: torch.Tensor,       # [B, K, 160]
    pi_search: torch.Tensor,     # [B, K] soft target
    n_cands: torch.Tensor,       # [B] int
    z: torch.Tensor,             # [B]
    opt: torch.optim.Optimizer,
    neg_states: torch.Tensor | None = None,   # [B, K_neg, 489]
    neg_actions: torch.Tensor | None = None,  # [B, K_neg, 160]
    n_neg: torch.Tensor | None = None,        # [B] int
) -> tuple[float, float, float, float]:
    B, K, d_state = cand_states.shape
    mask = torch.arange(K, device=cand_states.device).unsqueeze(0) < n_cands.unsqueeze(1)

    q_flat = net(cand_states.reshape(B * K, d_state),
                 actions.reshape(B * K, -1)).reshape(B, K)
    q_flat = q_flat.masked_fill(~mask, -1e9)
    log_probs = F.log_softmax(q_flat, dim=-1)
    pi_smooth = _smooth_targets(pi_search, mask)
    policy_loss = -(pi_smooth * log_probs).sum(dim=-1).mean()

    # V head sees the base 480-dim state (no action-conditional flags).
    # Pad with zeros to match d_state if needed (V net was built with d_state).
    pad_w = d_state - base_states.shape[1]
    v_input = (
        torch.cat(
            [base_states, torch.zeros(B, pad_w, device=base_states.device)], dim=-1
        )
        if pad_w > 0
        else base_states
    )
    v_pred = net.value(v_input)
    value_loss = F.mse_loss(v_pred, z)

    # Ranking loss: enforce Q(top-K) > Q(non-candidates) + margin.
    rank_loss = torch.tensor(0.0, device=cand_states.device)
    if neg_states is not None and neg_actions is not None and n_neg is not None:
        K_neg = neg_actions.shape[1]
        neg_mask = torch.arange(K_neg, device=cand_states.device).unsqueeze(0) < n_neg.unsqueeze(1)
        has_neg = neg_mask.any(dim=1)
        if has_neg.any():
            q_neg = net(neg_states.reshape(B * K_neg, d_state),
                        neg_actions.reshape(B * K_neg, -1)).reshape(B, K_neg)
            q_neg = q_neg.masked_fill(~neg_mask, -1e9)
            q_neg_max = q_neg.max(dim=-1).values

            q_pos = q_flat.masked_fill(~mask, 1e9)
            q_pos_min = q_pos.min(dim=-1).values

            margin_loss = F.relu(q_neg_max - q_pos_min + RANK_MARGIN)
            rank_loss = margin_loss[has_neg].mean()

    total = policy_loss + 0.5 * value_loss + RANK_WEIGHT * rank_loss
    opt.zero_grad()
    total.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    opt.step()
    return total.item(), policy_loss.item(), value_loss.item(), rank_loss.item()


@torch.no_grad()
def _eval_step(
    net: QValueNet,
    base_states: torch.Tensor,
    cand_states: torch.Tensor,
    actions: torch.Tensor,
    pi_search: torch.Tensor,
    n_cands: torch.Tensor,
    z: torch.Tensor,
) -> dict:
    """Returns batched val metrics (sums, divide by N at end)."""
    B, K, d_state = cand_states.shape
    mask = torch.arange(K, device=cand_states.device).unsqueeze(0) < n_cands.unsqueeze(1)

    q_flat = net(cand_states.reshape(B * K, d_state),
                 actions.reshape(B * K, -1)).reshape(B, K)
    q_flat = q_flat.masked_fill(~mask, -1e9)
    log_probs = F.log_softmax(q_flat, dim=-1)
    probs = log_probs.exp()

    policy_loss = -(pi_search * log_probs).sum(dim=-1).sum().item()

    entropy_per = -(probs * log_probs).masked_fill(~mask, 0).sum(dim=-1)
    entropy_sum = entropy_per.sum().item()

    pred = q_flat.argmax(dim=-1)
    target = pi_search.argmax(dim=-1)
    correct = (pred == target).sum().item()

    pad_w = d_state - base_states.shape[1]
    v_input = (
        torch.cat(
            [base_states, torch.zeros(B, pad_w, device=base_states.device)], dim=-1
        )
        if pad_w > 0
        else base_states
    )
    v_pred = net.value(v_input)
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


# ── Inline eval (policy-only argmax) ──────────────────────────────────────────

def _eval_vs_opponent(
    net: QValueNet,
    device: torch.device,
    opponent_name: str,
    n_games: int,
    seed: int,
) -> tuple[float, float]:
    """Play n_games policy-only argmax vs named opponent, alternating seats. Returns (WR, sec)."""
    from guandan.agents import JidanBot, StrategicBot
    from guandan.agents.yaoji_bot import YaojiBot
    from guandan.agents.partner_oracle_bot import _reflect_env
    from guandan.cards import Rank
    from guandan.game import GuanDanEnv

    _AGENT_CLS = {"jidan": JidanBot, "yaoji": YaojiBot, "strategic": StrategicBot}
    opp = _AGENT_CLS[opponent_name](level_rank=Rank.TWO)
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
        states = np.stack([
            encode_state_team_with_flags(enc_env, enc_player, a, legal)
            for a in legal
        ]).astype(np.float32)
        actions = np.stack([
            encode_action(a, enc_env.hands[enc_player], env.level_rank)
            for a in legal
        ]).astype(np.float32)
        st = torch.from_numpy(states).to(device)
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
            move = policy_act(env, p) if p in a1_seats else opp.act(env, p)
            env.step(move)
        if env.get_rewards()[next(iter(a1_seats))] > 0:
            wins += 1
    return wins / n_games, time.time() - t0


def _eval_vs_opponents(
    net: QValueNet,
    device: torch.device,
    opponents: list[str],
    n_games: int,
    seed: int,
    log: logging.Logger,
) -> tuple[float, dict[str, float]]:
    """Eval vs each opponent sequentially. Returns (avg_WR, per_opp_WR_dict)."""
    per_opp: dict[str, float] = {}
    for opp_name in opponents:
        wr, t_opp = _eval_vs_opponent(net, device, opp_name, n_games, seed)
        per_opp[opp_name] = wr
        log.info(f"    vs {opp_name:<12} WR={wr:.1%}  ({n_games} games, {t_opp:.0f}s)")
    avg_wr = sum(per_opp.values()) / len(per_opp)
    opps_str = "  ".join(f"{k}={v:.1%}" for k, v in per_opp.items())
    log.info(f"    avg_WR={avg_wr:.1%}  [{opps_str}]")
    return avg_wr, per_opp


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
    eval_opponents: list[str] | None = None,
    val_split: float = 0.10,
    early_stop_patience: int = 3,
    seed: int = 0,
    out_path: str | None = None,
    log: logging.Logger | None = None,
    run_dir: Path | None = None,
) -> tuple[QValueNet, dict]:
    if log is None:
        log = logging.getLogger("train_az")
    if eval_opponents is None:
        eval_opponents = ["jidan", "yaoji", "strategic"]
    base_states_np = data["base_states"]
    cand_states_np = data["cand_states"]
    actions_np = data["actions"]
    pi_np = data["pi_search"]
    n_cands_np = data["n_cands"]
    z_np = data["z"]
    neg_states_np = data.get("neg_states")
    neg_actions_np = data.get("neg_actions")
    n_neg_np = data.get("n_neg")
    N = len(base_states_np)

    net = QValueNet(d_state=STATE_DIM_TEAM_WITH_FLAGS, d_action=ACTION_DIM, hidden=hidden).to(device)
    if init_checkpoint:
        ckpt = torch.load(init_checkpoint, map_location=device, weights_only=True)
        missing, unexpected = net.load_state_dict(ckpt["state_dict"], strict=False)
        log.info(f"  loaded {init_checkpoint}  (missing={len(missing)}, unexpected={len(unexpected)})")
    n_params = sum(p.numel() for p in net.parameters())
    log.info(f"  net: {n_params:,} params on {device} (hidden={hidden})")

    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.05)

    rng = random.Random(seed)
    indices = list(range(N))
    rng.shuffle(indices)
    n_val = int(N * val_split)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    log.info(f"  split: {len(train_idx):,} train / {len(val_idx):,} val")
    log.info(f"  z stats: mean={z_np.mean():+.3f}  std={z_np.std():.3f}  "
             f"baseline_V_loss={(z_np**2).mean():.3f} (V≡0)")

    has_negatives = (
        neg_states_np is not None and neg_actions_np is not None and n_neg_np is not None
    )

    def to_tensors(idx_list):
        bs = torch.from_numpy(base_states_np[idx_list]).to(device)
        cs = torch.from_numpy(cand_states_np[idx_list]).to(device)
        ac = torch.from_numpy(actions_np[idx_list]).to(device)
        pi = torch.from_numpy(pi_np[idx_list]).to(device)
        nc = torch.from_numpy(n_cands_np[idx_list].astype(np.int64)).to(device)
        zb = torch.from_numpy(z_np[idx_list]).to(device)
        if has_negatives:
            ns = torch.from_numpy(neg_states_np[idx_list]).to(device)
            na = torch.from_numpy(neg_actions_np[idx_list]).to(device)
            nn_ = torch.from_numpy(n_neg_np[idx_list].astype(np.int64)).to(device)
        else:
            ns, na, nn_ = None, None, None
        return bs, cs, ac, pi, nc, zb, ns, na, nn_

    metrics = {"epochs": []}
    t0 = time.time()
    best_val_loss = float("inf")
    best_wr = -1.0
    best_wr_state = None
    patience_count = 0

    # Initial eval (epoch 0): baseline quality of the prior
    if eval_games > 0:
        net.eval()
        log.info("  baseline ep 0:")
        _eval_vs_opponents(net, device, eval_opponents, eval_games, seed=seed, log=log)

    log.info("─" * 64)
    log.info("Training ...")

    with logging_redirect_tqdm(loggers=[log]):
        pbar = tqdm(range(epochs), unit="ep", desc="train_az", dynamic_ncols=True)
        for epoch in pbar:
            rng.shuffle(train_idx)
            net.train()
            ep_total = ep_policy = ep_value = ep_rank = 0.0
            n_batches = 0
            for bs in range(0, len(train_idx), batch_size):
                bst, cst, ac, pi, nc, zb, ns, na, nn_ = to_tensors(train_idx[bs: bs + batch_size])
                tl, pl, vl, rl = _train_step(net, bst, cst, ac, pi, nc, zb, opt, ns, na, nn_)
                ep_total += tl
                ep_policy += pl
                ep_value += vl
                ep_rank += rl
                n_batches += 1
            train_total = ep_total / max(1, n_batches)
            train_policy = ep_policy / max(1, n_batches)
            train_value = ep_value / max(1, n_batches)
            train_rank = ep_rank / max(1, n_batches)

            # Val
            net.eval()
            agg = {"policy_loss_sum": 0, "value_loss_sum": 0, "value_mae_sum": 0,
                   "entropy_sum": 0, "correct": 0, "n": 0}
            for bs in range(0, len(val_idx), batch_size):
                bst, cst, ac, pi, nc, zb, _ns, _na, _nn = to_tensors(val_idx[bs: bs + batch_size])
                m = _eval_step(net, bst, cst, ac, pi, nc, zb)
                for k in agg:
                    agg[k] += m[k]
            n = max(1, agg["n"])
            val_policy = agg["policy_loss_sum"] / n
            val_value = agg["value_loss_sum"] / n
            val_total = val_policy + 0.5 * val_value
            val_mae = agg["value_mae_sum"] / n
            val_entropy = agg["entropy_sum"] / n
            val_acc = agg["correct"] / n

            # Inline eval vs all opponents — avg WR drives patience + best checkpoint
            avg_wr = None
            per_opp_wr: dict[str, float] = {}
            do_eval = eval_games > 0 and (epoch + 1) % eval_every == 0
            if do_eval:
                log.info(f"  ep {epoch+1} eval:")
                avg_wr, per_opp_wr = _eval_vs_opponents(
                    net, device, eval_opponents, eval_games, seed=seed + epoch + 1, log=log
                )

            cur_lr = opt.param_groups[0]["lr"]
            sched.step()
            elapsed = time.time() - t0

            # Val loss tracked for diagnostics only (not used for stopping)
            if val_total < best_val_loss - 1e-4:
                best_val_loss = val_total

            # WR-based patience + best-checkpoint selection
            if do_eval:
                if avg_wr > best_wr + 1e-4:
                    best_wr = avg_wr
                    best_wr_state = deepcopy(net.state_dict())
                    patience_count = 0
                    tag = "  ★ best_wr"
                else:
                    patience_count += 1
                    tag = f"  (no-improve {patience_count}/{early_stop_patience})"
            else:
                tag = ""

            wr_str = f" | avg_WR={avg_wr:.1%}" if avg_wr is not None else ""
            rank_str = f" r={train_rank:.3f}" if has_negatives else ""
            log.info(
                f"ep {epoch+1:2d}/{epochs} | "
                f"val={val_total:.3f}(p={val_policy:.3f} v={val_value:.3f}) | "
                f"tr={train_total:.3f}(p={train_policy:.3f} v={train_value:.3f}{rank_str}) | "
                f"H={val_entropy:.3f} acc={val_acc:.2f} v_mae={val_mae:.2f} | "
                f"lr={cur_lr:.1e} t={elapsed:.0f}s"
                f"{wr_str}{tag}"
            )

            row = {
                "epoch": epoch + 1,
                "train_total": round(train_total, 4), "train_policy": round(train_policy, 4),
                "train_value": round(train_value, 4), "train_rank": round(train_rank, 4),
                "val_total": round(val_total, 4), "val_policy": round(val_policy, 4),
                "val_value": round(val_value, 4), "val_entropy": round(val_entropy, 4),
                "val_acc": round(val_acc, 4), "val_mae": round(val_mae, 4),
                "avg_wr": round(avg_wr, 4) if avg_wr is not None else None,
                **{f"wr_{k}": round(v, 4) for k, v in per_opp_wr.items()},
                "lr": cur_lr, "elapsed_s": round(elapsed, 1),
            }
            metrics["epochs"].append(row)
            if run_dir:
                with open(run_dir / "metrics.jsonl", "a") as f:
                    f.write(json.dumps(row) + "\n")

            # Save latest checkpoint (recoverable if interrupted)
            if out_path:
                latest_path = Path(out_path).with_name(Path(out_path).stem + "_latest.pt")
                torch.save({
                    "state_dict": net.state_dict(),
                    "config": {"d_state": STATE_DIM_TEAM_WITH_FLAGS, "d_action": ACTION_DIM, "hidden": hidden},
                    "metrics": metrics, "epoch": epoch + 1,
                }, latest_path)

            if do_eval and patience_count >= early_stop_patience:
                log.info(f"  early stop at epoch {epoch + 1}")
                pbar.close()
                break

    if best_wr_state is not None:
        net.load_state_dict(best_wr_state)

    # Save best-WR checkpoint
    if best_wr_state is not None and out_path:
        wr_path = Path(out_path).with_name(Path(out_path).stem + "_bestwr.pt")
        torch.save({
            "state_dict": best_wr_state,
            "config": {"d_state": STATE_DIM_TEAM_WITH_FLAGS, "d_action": ACTION_DIM, "hidden": hidden},
            "metrics": metrics,
        }, wr_path)
        log.info(f"  ★ best-WR checkpoint → {wr_path}  (WR={best_wr:.1%})")

    metrics["n_train"] = len(train_idx)
    metrics["n_val"] = len(val_idx)
    metrics["best_val_loss"] = best_val_loss
    metrics["best_wr"] = best_wr
    metrics["total_train_time_s"] = round(time.time() - t0, 1)
    return net, metrics


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="AZ training: policy + value heads")
    ap.add_argument("--data", required=True)
    ap.add_argument("--init-checkpoint", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-games", type=int, default=50,
                    help="Inline policy-only eval games per opponent per epoch (0 = disabled)")
    ap.add_argument("--eval-every", type=int, default=1,
                    help="Run inline eval every N epochs (default 1)")
    ap.add_argument("--eval-opponents", default="jidan,yaoji,strategic",
                    help="Comma-separated opponents for inline eval (default: jidan,yaoji,strategic)")
    ap.add_argument("--run-dir", default=None,
                    help="Run directory for logs/config (default: ml/runs/az_{stem}_{ts}).")
    args = ap.parse_args()

    out = Path(args.out)
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        run_dir = Path("ml/runs") / f"az_{out.stem}_{ts}"

    log = _setup_logging(run_dir)
    device = get_device()

    log.info("=" * 64)
    log.info(f"[train_az]  run_dir={run_dir}  device={device}")
    log.info(f"  data           = {args.data}")
    log.info(f"  init_ckpt      = {args.init_checkpoint or '(none)'}")
    log.info(f"  epochs={args.epochs}  batch={args.batch_size}  lr={args.lr}  "
             f"wd={args.weight_decay}  hidden={args.hidden}  patience={args.patience}")
    eval_opponents = [o.strip() for o in args.eval_opponents.split(",") if o.strip()]
    log.info(f"  eval_games={args.eval_games}  eval_every={args.eval_every}  "
             f"eval_opponents={eval_opponents}  seed={args.seed}")
    log.info("=" * 64)

    config = vars(args)
    config["run_dir"] = str(run_dir)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    data = load_data(args.data)
    N = len(data["base_states"])
    K = data["actions"].shape[1]
    log.info(f"  loaded {N:,} decisions (K={K})")

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
        eval_opponents=eval_opponents,
        early_stop_patience=args.patience,
        seed=args.seed,
        out_path=args.out,
        log=log,
        run_dir=run_dir,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": net.state_dict(),
        "config": {"d_state": STATE_DIM_TEAM_WITH_FLAGS, "d_action": ACTION_DIM, "hidden": args.hidden},
        "metrics": metrics,
        "args": vars(args),
    }, out)

    final = metrics["epochs"][-1]
    log.info("─" * 64)
    log.info(
        f"[train_az]  done  {metrics['total_train_time_s']:.0f}s  "
        f"best_val={metrics['best_val_loss']:.3f}  best_WR={metrics['best_wr']:.1%}  "
        f"→  {out}"
    )


if __name__ == "__main__":
    main()
