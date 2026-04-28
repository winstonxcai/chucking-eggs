"""Per-encoder-group weight delta-norm visualization.

Reads metrics.jsonl from a pvguan PPO run and produces:
  1. Line plot  — cumulative normalized delta-norm per group over training
  2. Heatmap    — learning velocity (delta gained per eval interval)

Usage:
    python ml/scripts/util/plot_weight_norms.py ml/runs/<run>/metrics.jsonl
    python ml/scripts/util/plot_weight_norms.py ml/runs/<run>/metrics.jsonl --out norms.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


GROUP_ORDER = [
    "g1_own_hand",
    "g1_partner_hand",
    "g1_team_played",
    "g1_opp_played",
    "g1_unknown",
    "g2_seat_status",
    "g3_acting_ctx",
    "g4_active_trick",
    "g5_last_action",
    "g6_move_history",
    "g7_opp_style",
    "g8_behavior",
    "action",
]

GROUP_LABELS = {
    "g1_own_hand":     "G1 own hand (60)",
    "g1_partner_hand": "G1 partner hand (60)",
    "g1_team_played":  "G1 team played (120)",
    "g1_opp_played":   "G1 opp played (120)",
    "g1_unknown":      "G1 unknown (60)",
    "g2_seat_status":  "G2 seat status (40)",
    "g3_acting_ctx":   "G3 acting ctx (18)",
    "g4_active_trick": "G4 active trick (98)",
    "g5_last_action":  "G5 last action (96)",
    "g6_move_history": "G6 move history (83)",
    "g7_opp_style":    "G7 opp style (12) ★",
    "g8_behavior":     "G8 behavior flags (9)",
    "action":          "Action features (198)",
}

# Colours for the line plot — roughly ordered light→dark for visual separation
LINE_COLORS = [
    "#4e9af1", "#2166ac",           # G1 own/partner hand (blues)
    "#74c476", "#238b45",           # G1 team/opp played (greens)
    "#9ecae1",                      # G1 unknown (light blue)
    "#fd8d3c",                      # G2
    "#e6550d",                      # G3
    "#756bb1",                      # G4
    "#9e9ac8",                      # G5
    "#bcbddc",                      # G6
    "#d62728",                      # G7 opp-style (red — highlight)
    "#8c564b",                      # G8
    "#7f7f7f",                      # action
]


def load_metrics(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open()]


def extract_matrix(
    rows: list[dict],
    stride: int = 5,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (matrix [n_groups, n_time], decisions_M [n_time], present_groups)."""
    present = [g for g in GROUP_ORDER if f"{g}_wdelta" in rows[0]]
    sampled = rows[::stride]
    decisions = np.array([r["cumulative_decisions"] / 1e6 for r in sampled])
    mat = np.array(
        [[r[f"{g}_wdelta"] for g in present] for r in sampled]
    ).T  # [n_groups, n_time]
    return mat, decisions, present


def plot(metrics_path: Path, out_path: Path | None = None, stride: int = 5) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    rows = load_metrics(metrics_path)
    if not any(k.endswith("_wdelta") for k in rows[0]):
        print("No _wdelta keys found — run was started before group tracking was added.")
        return

    mat, decisions, groups = extract_matrix(rows, stride=stride)
    labels = [GROUP_LABELS.get(g, g) for g in groups]
    colors = [LINE_COLORS[GROUP_ORDER.index(g)] if g in GROUP_ORDER else "#aaa" for g in groups]

    fig = plt.figure(figsize=(15, 11))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1.4, 1], hspace=0.45)

    # ── Top: line plot (cumulative delta) ────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    for i, (g, label, color) in enumerate(zip(groups, labels, colors)):
        lw = 2.2 if g in ("g7_opp_style", "action") else 1.2
        ax1.plot(decisions, mat[i], label=label, color=color, linewidth=lw)
    ax1.set_xlabel("Decisions (M)", fontsize=11)
    ax1.set_ylabel("Δ norm / √group_size", fontsize=11)
    ax1.set_title("Encoder group weight delta — cumulative", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=7.5, ncol=3, loc="upper left")
    ax1.grid(alpha=0.25)

    # ── Bottom: velocity heatmap ───────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    velocity = np.diff(mat, axis=1)         # [n_groups, n_time-1]
    vmax = np.percentile(np.abs(velocity), 98)

    im = ax2.imshow(
        velocity,
        aspect="auto",
        cmap="YlOrRd",
        vmin=0,
        vmax=vmax,
        interpolation="nearest",
    )
    ax2.set_yticks(range(len(groups)))
    ax2.set_yticklabels(labels, fontsize=8)

    # x-ticks: show ~10 evenly spaced decision labels
    n_ticks = min(10, velocity.shape[1])
    tick_idx = np.linspace(0, velocity.shape[1] - 1, n_ticks, dtype=int)
    ax2.set_xticks(tick_idx)
    ax2.set_xticklabels([f"{decisions[i+1]:.1f}M" for i in tick_idx], fontsize=8)
    ax2.set_xlabel("Decisions (M)", fontsize=11)
    ax2.set_title("Learning velocity (Δ per interval)", fontsize=13, fontweight="bold")

    plt.colorbar(im, ax=ax2, label="Δ norm / √group_size per interval", pad=0.01)

    run_name = metrics_path.parent.name
    fig.suptitle(f"Weight-norm diagnostics — {run_name}", fontsize=14, y=1.01)

    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {out_path}")
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser("plot_weight_norms")
    parser.add_argument("metrics", type=Path, help="Path to metrics.jsonl")
    parser.add_argument("--out", type=Path, default=None, help="Output PNG path")
    parser.add_argument("--stride", type=int, default=5,
                        help="Sample every N iters (default 5)")
    args = parser.parse_args()
    plot(args.metrics, args.out, stride=args.stride)


if __name__ == "__main__":
    main()
