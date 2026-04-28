"""Live training figures for pvguan runs.

Writes PNGs to run_dir/figures/. Designed to be called every N iters from
the training loop so figures stay current without a dashboard server.
macOS Preview auto-refreshes open PNGs.

Usage (standalone):
    PYTHONPATH=ml/src python -m guandan.pvguan.plot_run \\
        --run-dir ml/runs/pvguan_pv_seed0_20260427_2038
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


# ─── EMA smoothing ────────────────────────────────────────────────────────────

def _ema(values: list[float], alpha: float = 0.1) -> list[float]:
    if not values:
        return []
    out = [values[0]]
    for v in values[1:]:
        out.append(alpha * v + (1 - alpha) * out[-1])
    return out


# ─── PPO figure ───────────────────────────────────────────────────────────────

def plot_ppo(run_dir: str | Path) -> Path:
    """Read metrics.jsonl and write figures/training.png. Returns the path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir)
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return run_dir / "figures" / "training.png"

    rows = [json.loads(l) for l in metrics_path.read_text().splitlines() if l.strip()]
    if not rows:
        return run_dir / "figures" / "training.png"

    # Validation eval rows (sparse)
    val_path = run_dir / "validation_eval.jsonl"
    val_rows: list[dict] = []
    if val_path.exists():
        val_rows = [
            json.loads(l) for l in val_path.read_text().splitlines() if l.strip()
        ]

    def col(key: str, default: float = 0.0) -> list[float]:
        return [r.get(key, default) for r in rows]

    dec = [r.get("cumulative_decisions", i) / 1e3 for i, r in enumerate(rows)]
    total_k = rows[-1].get("total_decisions", 0) / 1e3

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    fig.suptitle(
        f"{run_dir.name}  —  {dec[-1]:.0f}k / {total_k:.0f}k decisions",
        fontsize=11, fontweight="bold",
    )

    # ── [0,0] Reward + validation overlay ─────────────────────────────────────
    ax = axes[0, 0]
    r_raw = col("terminal_reward_mean")
    r_smooth = _ema(r_raw, alpha=0.15)
    ax.plot(dec, r_raw, color="steelblue", alpha=0.25, linewidth=0.8)
    ax.plot(dec, r_smooth, color="steelblue", linewidth=1.8, label="self-play r̄ (EMA)")
    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
    ax.set_ylabel("terminal reward")
    ax.set_title("Reward + Validation")

    if val_rows:
        val_dec = [r["cumulative_decisions"] / 1e3 for r in val_rows]
        val_opps = sorted({
            k.removeprefix("vs_") for r in val_rows for k in r if k.startswith("vs_")
        })
        opp_colors = {"jidan": "darkorange", "yaoji": "crimson",
                      "strategic": "purple", "heuristic": "olive"}
        for opp in val_opps:
            ys = [r.get(f"vs_{opp}", {}).get("promotion_diff_mean") for r in val_rows]
            color = opp_colors.get(opp, "gray")
            ax.plot(val_dec, ys, color=color, marker="o", markersize=4,
                    linewidth=1.2, label=f"vs {opp}")
    ax.legend(fontsize=8)

    # ── [0,1] Explained variance ──────────────────────────────────────────────
    ax = axes[0, 1]
    ev = col("explained_variance_old")
    ax.plot(dec, ev, color="darkorange", linewidth=1.5)
    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
    ax.set_ylabel("explained variance")
    ax.set_title("Critic Quality  (EV_old)")
    ax.set_ylim(-0.2, 1.05)

    # critic-warmup shading
    frozen = [r.get("actor_frozen", False) for r in rows]
    if any(frozen):
        last_frozen = max(i for i, f in enumerate(frozen) if f)
        ax.axvspan(dec[0], dec[last_frozen], alpha=0.08, color="purple",
                   label="critic warmup")
        ax.legend(fontsize=8)

    # ── [1,0] Policy entropy + top-1 prob ────────────────────────────────────
    ax = axes[1, 0]
    H   = col("entropy_legal")
    top1 = col("top1_prob")
    ax.plot(dec, H, color="seagreen", linewidth=1.5, label="entropy (nat)")
    ax2 = ax.twinx()
    ax2.plot(dec, top1, color="tomato", linewidth=1.2, alpha=0.7, label="top-1 prob")
    ax2.set_ylabel("top-1 prob", color="tomato")
    ax2.tick_params(axis="y", labelcolor="tomato")
    ax.set_ylabel("entropy (nat)")
    ax.set_title("Policy Distribution")
    lines1, lab1 = ax.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lab1 + lab2, fontsize=8)

    # ── [1,1] KL + clip fraction ─────────────────────────────────────────────
    ax = axes[1, 1]
    kl   = col("approx_kl")
    clip = col("clip_fraction")
    ax.plot(dec, kl,   color="purple",    linewidth=1.5, label="approx KL")
    ax2 = ax.twinx()
    ax2.plot(dec, clip, color="goldenrod", linewidth=1.2, alpha=0.8, label="clip frac")
    ax2.set_ylabel("clip fraction", color="goldenrod")
    ax2.tick_params(axis="y", labelcolor="goldenrod")
    ax.set_ylabel("approx KL")
    ax.set_title("Trust Region")
    lines1, lab1 = ax.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lab1 + lab2, fontsize=8)

    # ── [2,0] Gradient norms ─────────────────────────────────────────────────
    ax = axes[2, 0]
    ag = col("actor_grad_norm")
    cg = col("critic_grad_norm")
    ax.plot(dec, ag, color="royalblue", linewidth=1.2, label="actor")
    ax.plot(dec, cg, color="coral",     linewidth=1.2, label="critic")
    ax.set_ylabel("grad norm (pre-clip)")
    ax.set_title("Gradient Norms")
    ax.legend(fontsize=8)

    # ── [2,1] Privileged channel / loss breakdown ─────────────────────────────
    ax = axes[2, 1]
    delta = col("critic_priv_weight_delta_norm", default=-1.0)
    if any(d >= 0 for d in delta):
        ax.plot(dec, delta, color="mediumorchid", linewidth=1.5)
        ax.set_ylabel("||W_priv − W_priv_init||")
        ax.set_title("Privileged Channel Activation")
        ax.text(0.05, 0.92, "PV-PTIE: should rise\nPV-AC: should stay ≈0",
                transform=ax.transAxes, fontsize=7.5, verticalalignment="top",
                color="gray")
    else:
        # Fall back to loss components
        loss_pol = col("loss_policy")
        loss_val = col("loss_value")
        loss_ent = col("loss_entropy")
        ax.plot(dec, loss_pol, label="policy", linewidth=1.2)
        ax.plot(dec, loss_val, label="value",  linewidth=1.2)
        ax.plot(dec, loss_ent, label="entropy", linewidth=1.2)
        ax.set_ylabel("loss")
        ax.set_title("Loss Components")
        ax.legend(fontsize=8)

    for ax_row in axes:
        for ax in ax_row:
            ax.set_xlabel("decisions (k)")
            ax.grid(True, alpha=0.25)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    fig.tight_layout()

    fig_dir = run_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    out = fig_dir / "training.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


# ─── Distillation figure ──────────────────────────────────────────────────────

def plot_distill(history: list[dict], run_dir: str | Path) -> Path:
    """Write figures/distill.png from the epoch history list."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir)
    if not history:
        return run_dir / "figures" / "distill.png"

    epochs   = [r["epoch"]   for r in history]
    tr_loss  = [r["tr_loss"] for r in history]
    tr_acc   = [r["tr_acc"]  for r in history]
    val_acc  = [r["val_acc"] for r in history]
    lrs      = [r["lr"]      for r in history]
    best_ep  = max(history, key=lambda r: r["val_acc"])["epoch"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(
        f"{run_dir.name}  —  distillation  "
        f"(best val_acc={max(val_acc):.4f} @ ep {best_ep})",
        fontsize=10, fontweight="bold",
    )

    # Loss
    ax = axes[0]
    ax.plot(epochs, tr_loss, color="steelblue", marker="o", markersize=4, label="train loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("Training Loss")
    ax.legend(fontsize=8)
    ax2 = ax.twinx()
    ax2.plot(epochs, lrs, color="gray", linewidth=0.8, linestyle="--", label="lr")
    ax2.set_ylabel("learning rate", color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")

    # Accuracy
    ax = axes[1]
    ax.plot(epochs, tr_acc,  color="steelblue", marker="o", markersize=4, label="train acc")
    ax.plot(epochs, val_acc, color="darkorange", marker="s", markersize=4, label="val acc")
    ax.axhline(0.95, color="green",  linewidth=0.8, linestyle="--", label="95% gate")
    ax.axhline(0.90, color="orange", linewidth=0.8, linestyle="--", label="90% gate")
    ax.axvline(best_ep, color="darkorange", linewidth=0.6, linestyle=":", alpha=0.6)
    ax.set_xlabel("epoch")
    ax.set_ylabel("argmax accuracy")
    ax.set_title("Accuracy")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)

    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()

    fig_dir = run_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    out = fig_dir / "distill.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser("plot_run")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--watch", type=int, default=0,
                        help="Re-plot every N seconds (0 = once)")
    args = parser.parse_args()

    import time
    while True:
        out = plot_ppo(args.run_dir)
        print(f"updated → {out}")
        if args.watch <= 0:
            break
        time.sleep(args.watch)
