"""PPO training for the partner-visible PTIE experiment.

Two ablations selected by --critic {pv,ptie}.
Trains to a fixed --total-decisions budget (default 6M). Optional early
stopping via --val-patience (stop after N consecutive non-improving evals).

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
import logging
import random
import sys
import time
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))

from guandan.pvguan.actor_critic import ActorCriticNet, get_device, load_warmstart
from guandan.pvguan.agent import PVGuanBot
from guandan.pvguan.config import PPORunConfig
from guandan.pvguan.buffer import RolloutBuffer
from guandan.pvguan.diagnostics import (
    RunLogger,
    build_priv_probe_set,
    compute_policy_diagnostics,
    compute_value_diagnostics,
    critic_priv_sensitivity,
    critic_priv_weight_norms,
    pv_ac_priv_slots_zero,
)
from guandan.pvguan.encoders import ACTOR_DIM, ACTION_DIM, CRITIC_DIM, CRITIC_PRIV
from guandan.pvguan.plot_run import plot_ppo
from guandan.pvguan.ppo import PPOConfig, PPOTrainer
from guandan.pvguan.rollout import (
    HandScheduler,
    RolloutConfig,
    collect_rollout,
    collect_rollout_parallel,
)

# paired_eval is a script under ml/scripts/eval; sys.path was extended above
from paired_eval import run_paired_eval, get_deal_seeds, _build_bot


def _ckpt_name(run_name: str, tag: str) -> str:
    return f"pvguan_{run_name}_{tag}.pt"


def _save_checkpoint(
    path: Path,
    net: ActorCriticNet,
    metadata: dict,
    *,
    trainer: "PPOTrainer | None" = None,
    scheduler: "HandScheduler | None" = None,
    loop_state: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "actor_state_dict":  net.actor_head.state_dict(),
        "critic_state_dict": net.critic_head.state_dict(),
        "full_state_dict":   net.state_dict(),
        "metadata":          metadata,
    }
    if trainer is not None:
        payload["optimizer_state_dict"] = trainer.optimizer.state_dict()
        payload["trainer_iter"]         = trainer._iter
        payload["actor_unfrozen_at"]    = trainer._actor_unfrozen_at
    if scheduler is not None:
        payload["scheduler_counter"] = scheduler._counter
    if loop_state is not None:
        payload["loop_state"] = loop_state
    torch.save(payload, path)


def _checkpoint_hash(path: Path) -> str:
    if not path.exists():
        return "none"
    h = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    return h


def _git_sha() -> str:
    """Best-effort git HEAD SHA for reproducibility."""
    import subprocess
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
            cwd=Path(__file__).resolve().parent,
            timeout=2,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _run_validation(
    net: ActorCriticNet,
    iter_idx: int,
    cumulative_decisions: int,
    cfg: PPORunConfig,
    log,
) -> dict:
    """Run paired eval against each opponent. Returns the row written to JSONL."""
    from guandan.cards import Rank

    net.eval()
    try:
        bot = PVGuanBot.from_net(net, level_rank=Rank.TWO, sample=False)
        # Each deck plays 4 seat rotations → games = decks × 4
        n_decks = max(cfg.val_games // 4, 1)
        deal_seeds = get_deal_seeds("validation", n_decks)

        row: dict = {
            "iter":                 iter_idx,
            "cumulative_decisions": cumulative_decisions,
        }
        rank_advs: list[float] = []
        for opp_name in cfg.val_opponents:
            opp_name = opp_name.strip()
            if not opp_name:
                continue
            t0 = time.time()
            opp = _build_bot(opp_name, Rank.TWO)
            results = run_paired_eval(
                bot, opp, deal_seeds, n_bootstrap=cfg.val_bootstrap
            )
            wall = time.time() - t0
            row[f"vs_{opp_name}"] = {
                "promotion_diff_mean": results["promotion_diff_mean"],
                "promotion_diff_ci":   results["promotion_diff_ci"],
                "win_rate":            results["win_rate"],
                "tie_rate":            results["tie_rate"],
                "finish_split":        results["finish_split"],
                "n_hands":             results["n_hands"],
                "wall_s":              wall,
            }
            rank_advs.append(results["promotion_diff_mean"])
            log.info(
                f"  val vs {opp_name:<10} "
                f"Δ={results['promotion_diff_mean']:+.3f} "
                f"CI=[{results['promotion_diff_ci'][0]:+.3f}, "
                f"{results['promotion_diff_ci'][1]:+.3f}]  "
                f"wr={results['win_rate']:.3f}  "
                f"n={results['n_hands']}  t={wall:.1f}s"
            )

        # Combined metric drives best-checkpoint selection
        row["validation_metric"] = float(np.mean(rank_advs)) if rank_advs else 0.0
        return row
    finally:
        net.train()


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
    parser.add_argument("--rollout-workers",  type=int,   default=1,
                        help="Number of multiprocess rollout workers (1 = single-threaded)")
    parser.add_argument("--resume",           default=None,
                        help="Path to checkpoint to resume from (loads optimizer + scheduler counter)")
    parser.add_argument("--snapshot-interval", type=int,  default=1_000_000,
                        help="Save a snapshot checkpoint every N decisions")
    parser.add_argument("--val-every",        type=int,   default=50,
                        help="Iters between paired-eval validations (0 to disable)")
    parser.add_argument("--val-games",        type=int,   default=200,
                        help="Games per opponent per validation "
                             "(= n_decks × 4 seat rotations; rounded down to multiple of 4)")
    parser.add_argument("--val-opponents",    default="jidan,yaoji",
                        help="Comma-separated opponent names")
    parser.add_argument("--val-bootstrap",    type=int,   default=200)
    parser.add_argument("--val-patience",     type=int,   default=0,
                        help="Early stopping: stop if val metric doesn't improve "
                             "for this many consecutive evals (0 = disabled)")
    parser.add_argument("--opponent-mix",     default="",
                        help="Comma-separated opponent bot names mixed into rollout "
                             "(e.g. 'jidan,yaoji,strategic'). Empty = pure self-play.")
    parser.add_argument("--selfplay-frac",    type=float, default=1.0,
                        help="Fraction of hands that are pure self-play. "
                             "1.0 = legacy (no mixed opponents); "
                             "0.25 = 75%% mixed-opponent hands. Ignored if "
                             "--opponent-mix is empty.")
    args = parser.parse_args()

    device  = get_device()
    run_cfg = PPORunConfig.from_args(args)
    run_dir = Path(run_cfg.resolved_run_dir)

    run_name = f"{run_cfg.critic}_seed{run_cfg.seed}"
    config   = {**run_cfg.to_dict(), "device": str(device), "git_sha": _git_sha()}
    run_logger = RunLogger(run_dir, config)   # writes config.json with git_sha + device
    log = run_logger.log

    log.info("=" * 64)
    log.info(f"[train_pvguan]  run_dir={run_dir}  device={device}")
    log.info(str(run_cfg))
    log.info("=" * 64)

    # ── Build network ──────────────────────────────────────────────────────────
    net = ActorCriticNet(
        d_state_actor=ACTOR_DIM,
        d_action=ACTION_DIM,
        d_state_critic=CRITIC_DIM,
        hidden=run_cfg.hidden,
        temperature=run_cfg.resolved_init_temperature,
    ).to(device)

    warmstart_hash = "none"
    warmstart_meta: dict = {}
    resume_ckpt: dict | None = None
    if run_cfg.resume:
        if run_cfg.warmstart:
            raise SystemExit("--resume and --warmstart are mutually exclusive")
        resume_ckpt = torch.load(
            run_cfg.resume, map_location=device, weights_only=False
        )
        net.load_state_dict(resume_ckpt["full_state_dict"])
        warmstart_hash = _checkpoint_hash(Path(run_cfg.resume))
        log.info(f"  resuming from {run_cfg.resume}  hash={warmstart_hash}")
    elif run_cfg.warmstart:
        warmstart_hash = _checkpoint_hash(Path(run_cfg.warmstart))
        warmstart_meta = load_warmstart(net, run_cfg.warmstart, map_location=device)
        log.info(f"  warmstart loaded from {run_cfg.warmstart}  hash={warmstart_hash}")
        log.info(f"  warmstart supervisor={warmstart_meta.get('supervisor', '?')}")
    else:
        log.info("  no warmstart — training from scratch")

    # Capture initial privileged-column weights for delta_norm diagnostic
    with torch.no_grad():
        init_priv_w = net.critic_head.net[0].weight[:, CRITIC_PRIV].clone()

    # ── PPO trainer ───────────────────────────────────────────────────────────
    ppo_cfg = PPOConfig(
        clip_eps=run_cfg.clip_eps,
        c_value=run_cfg.c_value,
        c_entropy=run_cfg.c_entropy,
        kl_init=run_cfg.c_kl_init,
        kl_decay_actor_iters=run_cfg.kl_decay_actor_iters,
        critic_warmup_iters=run_cfg.critic_warmup_iters,
        lr=run_cfg.lr,
        weight_decay=run_cfg.weight_decay,
        n_epochs=run_cfg.n_epochs,
        mini_batch_size=run_cfg.mini_batch,
        gamma=run_cfg.gamma,
        lam=run_cfg.lam,
        temperature=run_cfg.temperature,
    )
    trainer = PPOTrainer(net, ppo_cfg, device)

    rollout_cfg = RolloutConfig(
        target_decisions=run_cfg.iter_decisions,
        critic_mode=run_cfg.critic,
        temperature=run_cfg.temperature,
        opponent_mix=tuple(run_cfg.opponent_mix),
        selfplay_frac=run_cfg.selfplay_frac,
    )
    scheduler = HandScheduler(global_run_seed=run_cfg.seed * 100_000)

    cumulative_decisions = 0
    cumulative_hands = 0
    best_val_metric = float("-inf")
    val_no_improve_count = 0
    iter_idx = 0
    t_run_start = time.time()

    # Restore trainer/scheduler/loop state if resuming
    if resume_ckpt is not None:
        if "optimizer_state_dict" in resume_ckpt:
            trainer.optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        if "trainer_iter" in resume_ckpt:
            trainer._iter = resume_ckpt["trainer_iter"]
        if "actor_unfrozen_at" in resume_ckpt:
            trainer._actor_unfrozen_at = resume_ckpt["actor_unfrozen_at"]
        if "scheduler_counter" in resume_ckpt:
            scheduler._counter = resume_ckpt["scheduler_counter"]
        ls = resume_ckpt.get("loop_state", {})
        cumulative_decisions = ls.get("cumulative_decisions", 0)
        cumulative_hands     = ls.get("cumulative_hands", 0)
        iter_idx             = ls.get("iter_idx", 0)
        best_val_metric      = ls.get("best_val_metric", float("-inf"))
        log.info(
            f"  resumed: cumulative_decisions={cumulative_decisions:,}  "
            f"iter={iter_idx}  trainer_iter={trainer._iter}  "
            f"scheduler_counter={scheduler._counter}  "
            f"best_val_metric={best_val_metric:+.3f}"
        )

    # Iter-0 entropy guard (skip on resume — already past iter 0)
    if run_cfg.warmstart and resume_ckpt is None:
        from guandan.pvguan.encoders import encode_actor_pair_features, encode_action
        from guandan.game import GuanDanEnv
        from guandan.cards import Rank
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
            log.info(f"  iter-0 entropy_legal = {entropy:.3f}")
            if entropy < 0.5 and run_cfg.init_temperature is None:
                log.warning("iter-0 entropy < 0.5 nat — consider --init-temperature 1.5")

    # ── Multiprocess rollout pool ─────────────────────────────────────────────
    rollout_pool = None
    rollout_state_path: str | None = None
    if run_cfg.rollout_workers > 1:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        rollout_pool = ctx.Pool(processes=run_cfg.rollout_workers)
        rollout_state_path = (
            f"/tmp/pvguan_rollout_state_{run_name}_{int(time.time())}.pt"
        )
        log.info(
            f"  rollout_pool: {run_cfg.rollout_workers} workers  "
            f"state_path={rollout_state_path}"
        )

    # ── Training loop ─────────────────────────────────────────────────────────
    log.info("─" * 64)
    log.info("Training ...")

    # Probe set for critic_priv_sensitivity (built lazily from the first batch)
    probe_with_priv: torch.Tensor | None = None
    probe_zero_priv: torch.Tensor | None = None

    # Periodic snapshot tracker — derived from cumulative_decisions on resume
    next_snapshot = (
        (cumulative_decisions // run_cfg.snapshot_interval) + 1
    ) * run_cfg.snapshot_interval

    with logging_redirect_tqdm(loggers=[log]):
        pbar = tqdm(
            total=run_cfg.total_decisions,
            unit="dec",
            desc=f"{run_cfg.critic}/s{run_cfg.seed}",
            dynamic_ncols=True,
        )

        while cumulative_decisions < run_cfg.total_decisions:
            t_iter_start = time.time()

            # 1. Rollout
            t_roll_start = time.time()
            if rollout_pool is not None:
                buf = collect_rollout_parallel(
                    net, scheduler, rollout_cfg,
                    n_workers=run_cfg.rollout_workers,
                    pool=rollout_pool,
                    state_path=rollout_state_path,
                    hidden=run_cfg.hidden,
                )
            else:
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
            val_diag   = compute_value_diagnostics(batch)
            pol_diag   = compute_policy_diagnostics(net, batch, temperature=run_cfg.temperature)
            priv_norms = critic_priv_weight_norms(net, init_priv_w)
            roll_stats = buf.stats.summary()

            # Build the priv-sensitivity probe set on the first batch, then
            # reuse the same set every iter so the metric is comparable
            if probe_with_priv is None:
                probe_with_priv, probe_zero_priv = build_priv_probe_set(
                    batch, n_probe=100, seed=run_cfg.seed
                )
            priv_sens = critic_priv_sensitivity(net, probe_with_priv, probe_zero_priv)

            cumulative_decisions += iter_decisions
            cumulative_hands     += sum(1 for _ in batch.sampled_idx)
            iter_wall = time.time() - t_iter_start

            metrics = {
                "iter":                  iter_idx,
                "cumulative_decisions":  cumulative_decisions,
                "total_decisions":       run_cfg.total_decisions,
                "iter_decisions":        iter_decisions,
                "iter_wall_s":           iter_wall,
                "rollout_wall_s":        roll_wall,
                "update_wall_s":         update_wall,
                "rollout_dec_per_s":     iter_decisions / max(roll_wall, 0.001),
                "actor_grad_norm":       trainer.actor_grad_norm(),
                "critic_grad_norm":      trainer.critic_grad_norm(),
                "critic_priv_sensitivity": priv_sens,
                **update_metrics,
                **val_diag,
                **pol_diag,
                **priv_norms,
                **roll_stats,
            }

            # Leakage alarm (PV-AC only)
            if run_cfg.critic == "pv":
                if not pv_ac_priv_slots_zero(batch):
                    metrics["pv_ac_priv_slots_zero"] = False
                    log.error("ABORT: PV-AC privileged slots non-zero — information leakage!")
                    break
                metrics["pv_ac_priv_slots_zero"] = True

            # Log to file + console summary line
            run_logger.log_iter(metrics)

            # Update progress bar postfix (live dashboard)
            pbar.update(iter_decisions)
            pbar.set_postfix(
                r=f"{metrics.get('terminal_reward_mean', 0):+.3f}",
                H=f"{metrics.get('entropy_legal', 0):.2f}",
                EV=f"{metrics.get('explained_variance_old', 0):.2f}",
                clip=f"{metrics.get('clip_fraction', 0):.2f}",
                KL=f"{metrics.get('approx_kl', 0):.4f}",
                frozen="Y" if metrics.get("actor_frozen") else "N",
            )

            # Live figures (every 10 iters)
            if iter_idx % 10 == 0:
                plot_ppo(run_dir)

            # ── Inline validation ─────────────────────────────────────────────
            if (run_cfg.val_every > 0
                    and iter_idx > 0
                    and iter_idx % run_cfg.val_every == 0):
                val_row = _run_validation(
                    net, iter_idx, cumulative_decisions, run_cfg, log
                )
                run_logger.log_val(val_row)

                # Best-checkpoint selection and early stopping
                if val_row["validation_metric"] > best_val_metric:
                    best_val_metric = val_row["validation_metric"]
                    val_no_improve_count = 0
                    best_path = run_dir / "checkpoints" / _ckpt_name(run_name, "best")
                    _save_checkpoint(best_path, net, {
                        **config,
                        "cumulative_decisions": cumulative_decisions,
                        "warmstart_hash": warmstart_hash,
                        "validation_metric": best_val_metric,
                        "iter": iter_idx,
                        "critic_mode": run_cfg.critic,
                    }, trainer=trainer, scheduler=scheduler, loop_state={
                        "cumulative_decisions": cumulative_decisions,
                        "cumulative_hands":     cumulative_hands,
                        "iter_idx":             iter_idx + 1,
                        "best_val_metric":      best_val_metric,
                    })
                    log.info(f"  ★ new best: validation_metric={best_val_metric:+.3f} → {best_path.name}")
                else:
                    val_no_improve_count += 1
                    log.info(f"  no improvement ({val_no_improve_count}/{run_cfg.val_patience or '∞'})")

                if run_cfg.val_patience > 0 and val_no_improve_count >= run_cfg.val_patience:
                    log.info(
                        f"  ★ early stopping: no improvement for {run_cfg.val_patience} "
                        f"consecutive evals (best={best_val_metric:+.3f})"
                    )
                    break

            # Periodic snapshot checkpoints (every snapshot_interval decisions)
            while cumulative_decisions >= next_snapshot:
                if next_snapshot >= 1_000_000:
                    snap_tag = f"{next_snapshot // 1_000_000}M"
                else:
                    snap_tag = f"{next_snapshot}"
                snap_path = run_dir / "checkpoints" / _ckpt_name(run_name, snap_tag)
                ck_meta = {
                    **config,
                    "cumulative_decisions": cumulative_decisions,
                    "warmstart_hash": warmstart_hash,
                    "warmstart_supervisor": warmstart_meta.get("supervisor", "?"),
                    "critic_mode": run_cfg.critic,
                }
                _save_checkpoint(
                    snap_path, net, ck_meta,
                    trainer=trainer, scheduler=scheduler,
                    loop_state={
                        "cumulative_decisions": cumulative_decisions,
                        "cumulative_hands":     cumulative_hands,
                        "iter_idx":             iter_idx + 1,
                        "best_val_metric":      best_val_metric,
                    },
                )
                log.info(f"  snapshot @ {snap_tag} → {snap_path}")
                next_snapshot += run_cfg.snapshot_interval

            iter_idx += 1

        pbar.close()

    # Final checkpoint
    log.info("─" * 64)
    final_path = run_dir / "checkpoints" / _ckpt_name(run_name, "final")
    final_meta = {
        **config,
        "cumulative_decisions": cumulative_decisions,
        "warmstart_hash": warmstart_hash,
    }
    _save_checkpoint(
        final_path, net, final_meta,
        trainer=trainer, scheduler=scheduler,
        loop_state={
            "cumulative_decisions": cumulative_decisions,
            "cumulative_hands":     cumulative_hands,
            "iter_idx":             iter_idx,
            "best_val_metric":      best_val_metric,
        },
    )
    log.info(
        f"[train_pvguan]  done  {cumulative_decisions:,} decisions  "
        f"wall={run_logger.elapsed():.0f}s  →  {final_path}"
    )

    # Cleanup multiprocess pool + temp state file
    if rollout_pool is not None:
        rollout_pool.close()
        rollout_pool.join()
    if rollout_state_path is not None:
        try:
            Path(rollout_state_path).unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    main()
