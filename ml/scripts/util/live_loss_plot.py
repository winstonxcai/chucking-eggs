"""Live-updating loss plot from metrics_learner.jsonl.

Usage:
    ~/miniconda3/bin/python ml/scripts/util/live_loss_plot.py \\
        --metrics ml/runs/guanzero_m0_20260502_0120/metrics_learner.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.animation as animation


def load(path: Path) -> tuple[list, dict]:
    updates = []
    losses = {p: [] for p in range(4)}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            updates.append(row["updates"])
            for p in range(4):
                losses[p].append(row["loss"].get(str(p), float("nan")))
        except Exception:
            continue
    return updates, losses


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", required=True, type=Path)
    p.add_argument("--interval", type=int, default=5000, help="Refresh interval ms.")
    args = p.parse_args()

    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
    labels = ["p0", "p1", "p2", "p3"]

    fig, ax = plt.subplots(figsize=(11, 5))
    lines = [ax.plot([], [], color=c, label=l, linewidth=1.5, alpha=0.85)[0]
             for c, l in zip(colors, labels)]
    ax.set_xlabel("Learner updates")
    ax.set_ylabel("MSE loss")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    title = ax.set_title("")

    def update(_frame):
        updates, losses = load(args.metrics)
        if not updates:
            return lines
        for i, line in enumerate(lines):
            line.set_data(updates, losses[i])
        ax.relim()
        ax.autoscale_view()
        last = updates[-1]
        last_loss = {i: losses[i][-1] for i in range(4) if losses[i]}
        loss_str = "  ".join(f"p{i}={v:.4f}" for i, v in last_loss.items())
        title.set_text(f"GuanZero M0 loss — update {last:,}    {loss_str}")
        fig.canvas.draw_idle()
        return lines

    ani = animation.FuncAnimation(fig, update, interval=args.interval, cache_frame_data=False)
    update(None)  # draw immediately on open
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
