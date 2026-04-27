"""PPO training for the partner-visible PTIE experiment.

Two ablations selected by --critic {pv,ptie}.
Trains to a fixed --total-decisions budget (default 6M). No early stopping.

Usage:
    # smoke test (~10 min on M1, both ablations)
    PYTHONPATH=ml/src python ml/scripts/train/train_pvguan.py \\
        --critic pv --seed 0 \\
        --warmstart ml/checkpoints/pvguan_distilled_oracle.pt \\
        --total-decisions 4096 --iter-decisions 1024 \\
        --critic-warmup 2 --run-dir ml/runs/smoke_pv

    PYTHONPATH=ml/src python ml/scripts/train/train_pvguan.py \\
        --critic ptie --seed 0 \\
        --warmstart ml/checkpoints/pvguan_distilled_oracle.pt \\
        --total-decisions 4096 --iter-decisions 1024 \\
        --critic-warmup 2 --run-dir ml/runs/smoke_ptie
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Literal

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from guandan.pvguan.actor_critic import ActorCriticNet, get_device, load_warmstart
from guandan.pvguan.buffer import RolloutBuffer
from guandan.pvguan.diagnostics import (
    RunLogger,
    compute_policy_diagnostics,
    compute_value_diagnostics,
    critic_priv_weight_norms,
    pv_ac_priv_slots_zero,
)
from guandan.pvguan.encoders import ACTOR_DIM, ACTION_DIM, CRITIC_DIM, CRITIC_PRIV
from guandan.pvguan.ppo import PPOConfig, PPOTrainer
from guandan.pvguan.rollout import HandScheduler, RolloutConfig, collect_rollout


SNAPSHOT_BUDGETS = [1_000_000, 3_000_000, 6_000_000]


def _ckpt_name(run_name: str, tag: str) -> str:
    return f"pvguan_{run_name}_{tag}.pt"


def _save_checkpoint(
    path: Path,
    net: ActorCriticNet,
    metadata: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "actor_state_dict":  net.actor_head.state_dict(),
        "critic_state_dict": net.critic_head.state_dict(),
        "full_state_dict":   net.state_dict(),
        "metadata":          metadata,
    }, path)


def _checkpoint_hash(path: Path) -> str:
    if not path.exists():
        return "none"
    h = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    return h


def main():
    parser = argparse.ArgumentParser("train_pvguan")
    parser.add_argument("--critic",    required=True, choices=["pv", "ptie"])
    parser.add_argument("--seed",      type=int, default=0)
    parser.add_argument("--warmstart", default=None,
                        help="Path to pvguan_distilled_*.pt checkpoint")
    parser.add_argument("--total-decisions",  type=int, default=6_000_000)
    parser.add_argument("--iter-decisions",   type=int, default=4096)
    parser.add_argument("--temperature",      type=float, default=1.0)
    parser.add_argument("--init-temperature", type=float, default=None)
    parser.add_argument("--kl-init",          type=float, default=0.05)
    parser.add_argument("--kl-decay-actor-iters", type=int, default=100)
    parser.add_argument("--critic-warmup",    type=int,  default=30)
    parser.add_argument("--clip-eps",         type=float, default=0.2)
    parser.add_argument("--c-value",          type=float, default=0.5)
    parser.add_argument("--c-entropy",        type=float, default=0.01)
    parser.add_argument("--lr",               type=float, default=3e-4)
    parser.add_argument("--hidden",           type=int,  default=256)
    parser.add_argument("--n-epochs",         type=int,  default=4)
    parser.add_argument("--mini-batch",       type=int,  default=256)
    parser.add_argument("--gamma",            type=float, default=1.0)
    parser.add_argument("--lam",              type=float, default=0.95)
    parser.add_argument("--run-dir",          default=None)
    args = parser.parse_args()

    device = get_device()
    init_temp = args.init_temperature or args.temperature

    run_name = f"{args.critic}_seed{args.seed}"
    run_dir  = Path(args.run_dir) if args.run_dir else \
               Path(f"ml/runs/pvguan_{run_name}")

    config = {
        "critic_mode": args.critic,
        "seed": args.seed,
        "total_decisions": args.total_decisions,
        "iter_decisions": args.iter_decisions,
        "temperature": args.temperature,
        "kl_init": args.kl_init,
        "kl_decay_actor_iters": args.kl_decay_actor_iters,
        "critic_warmup": args.critic_warmup,
        "clip_eps": args.clip_eps,
        "c_value": args.c_value,
        "c_entropy": args.c_entropy,
        "lr": args.lr,
        "hidden": args.hidden,
        "n_epochs": args.n_epochs,
        "mini_batch": args.mini_batch,
        "gamma": args.gamma,
        "lam": args.lam,
        "warmstart": args.warmstart,
        "device": str(device),
    }
    logger = RunLogger(run_dir, config)
    print(f"[train_pvguan] run_dir={run_dir} critic={args.critic} device={device}")

    # ── Build network ──────────────────────────────────────────────────────────
    net = ActorCriticNet(
        d_state_actor=ACTOR_DIM,
        d_action=ACTION_DIM,
        d_state_critic=CRITIC_DIM,
        hidden=args.hidden,
        temperature=init_temp,
    ).to(device)

    warmstart_hash = "none"
    warmstart_meta: dict = {}
    if args.warmstart:
        warmstart_hash = _checkpoint_hash(Path(args.warmstart))
        warmstart_meta = load_warmstart(net, args.warmstart, map_location=device)
        print(f"  loaded warmstart from {args.warmstart} (hash={warmstart_hash})")
    else:
        print("  no warmstart — training from scratch")

    # Capture initial privileged-column weights for delta_norm diagnostic
    with torch.no_grad():
        init_priv_w = net.critic_head.net[0].weight[:, CRITIC_PRIV].clone()

    # ── PPO trainer ───────────────────────────────────────────────────────────
    ppo_cfg = PPOConfig(
        clip_eps=args.clip_eps,
        c_value=args.c_value,
        c_entropy=args.c_entropy,
        kl_init=args.kl_init,
        kl_decay_actor_iters=args.kl_decay_actor_iters,
        critic_warmup_iters=args.critic_warmup,
        lr=args.lr,
        weight_decay=0.0,     # intentionally zero — see plan §5
        n_epochs=args.n_epochs,
        mini_batch_size=args.mini_batch,
        gamma=args.gamma,
        lam=args.lam,
        temperature=args.temperature,
    )
    trainer = PPOTrainer(net, ppo_cfg, device)

    rollout_cfg = RolloutConfig(
        target_decisions=args.iter_decisions,
        critic_mode=args.critic,
        temperature=args.temperature,
    )
    scheduler = HandScheduler(global_run_seed=args.seed * 100_000)

    cumulative_decisions = 0
    cumulative_hands = 0
    best_val_metric = float("-inf")
    snapshot_done: set[int] = set()
    iter_idx = 0
    t_run_start = time.time()

    # Iter-0 entropy guard
    if args.warmstart:
        from guandan.pvguan.encoders import encode_actor_pair_features, encode_action
        from guandan.game import GuanDanEnv
        from guandan.cards import Rank
        from guandan.combos import generate_all_leads
        probe_env = GuanDanEnv(Rank.TWO)
        probe_env.reset(seed=9999)
        legal = probe_env.legal_moves(0)
        if len(legal) > 1:
            hand = list(probe_env.hands[0])
            sa = torch.tensor(np.array([
                encode_actor_pair_features(probe_env, 0, a, legal) for a in legal
            ], dtype=np.float32), device=device)
            ac = torch.tensor(np.array([
                encode_action(a, hand, probe_env.level_rank) for a in legal
            ], dtype=np.float32), device=device)
            mask = torch.ones(len(legal), dtype=torch.bool, device=device)
            with torch.no_grad():
                logits = net.policy_logits(sa, ac, mask)
                lp = torch.nn.functional.log_softmax(logits, dim=0)
                p = lp.exp()
                entropy = -(p * lp).sum().item()
            print(f"  iter-0 entropy_legal = {entropy:.3f}")
            if entropy < 0.5 and args.init_temperature is None:
                print("WARN: iter-0 entropy < 0.5 nat — consider --init-temperature 1.5")

    # ── Training loop ─────────────────────────────────────────────────────────
    while cumulative_decisions < args.total_decisions:
        t_iter_start = time.time()

        # 1. Rollout
        t_roll_start = time.time()
        buf = collect_rollout(net, scheduler, rollout_cfg, device)
        roll_wall = time.time() - t_roll_start
        iter_decisions = buf.size()

        # 2. Build batch (freeze V_old, returns, advantages)
        batch = buf.compute_batch(device=device)
        buf.clear()

        # 3. PPO update
        t_update_start = time.time()
        update_metrics = trainer.update(batch)
        update_wall = time.time() - t_update_start

        # 4. Diagnostics
        val_diag  = compute_value_diagnostics(batch)
        pol_diag  = compute_policy_diagnostics(net, batch, temperature=args.temperature)
        priv_norms = critic_priv_weight_norms(net, init_priv_w)

        cumulative_decisions += iter_decisions
        cumulative_hands     += sum(1 for t in batch.sampled_idx)  # proxy
        iter_wall = time.time() - t_iter_start

        metrics = {
            "iter": iter_idx,
            "cumulative_decisions": cumulative_decisions,
            "iter_decisions": iter_decisions,
            "iter_wall_s": iter_wall,
            "rollout_wall_s": roll_wall,
            "update_wall_s": update_wall,
            "rollout_dec_per_s": iter_decisions / max(roll_wall, 0.001),
            "actor_grad_norm": trainer.actor_grad_norm(),
            "critic_grad_norm": trainer.critic_grad_norm(),
            **update_metrics,
            **val_diag,
            **pol_diag,
            **priv_norms,
        }

        # Alarm checks
        if args.critic == "pv":
            if not pv_ac_priv_slots_zero(batch):
                metrics["pv_ac_priv_slots_zero"] = False
                print("ABORT: PV-AC privileged slots non-zero — information leakage!")
                break
            metrics["pv_ac_priv_slots_zero"] = True

        logger.log_iter(metrics)

        # Snapshot checkpoints at fixed decision budgets
        for budget in SNAPSHOT_BUDGETS:
            if budget not in snapshot_done and cumulative_decisions >= budget:
                snapshot_path = run_dir / "checkpoints" / _ckpt_name(run_name, f"{budget//1_000_000}M")
                ck_meta = {**config,
                           "cumulative_decisions": cumulative_decisions,
                           "warmstart_hash": warmstart_hash,
                           "warmstart_supervisor": warmstart_meta.get("supervisor", "?"),
                           "critic_mode": args.critic}
                _save_checkpoint(snapshot_path, net, ck_meta)
                print(f"  snapshot @ {budget//1_000_000}M → {snapshot_path}")
                snapshot_done.add(budget)

        iter_idx += 1

    # Final checkpoint
    final_path = run_dir / "checkpoints" / _ckpt_name(run_name, "final")
    final_meta = {**config,
                  "cumulative_decisions": cumulative_decisions,
                  "warmstart_hash": warmstart_hash}
    _save_checkpoint(final_path, net, final_meta)
    print(f"\n[train_pvguan] done. {cumulative_decisions} decisions. "
          f"final checkpoint → {final_path}")


if __name__ == "__main__":
    main()
