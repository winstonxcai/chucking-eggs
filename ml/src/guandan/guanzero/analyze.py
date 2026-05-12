"""Static analysis: plot per-position learner loss + throughput + replay
control for a GuanZero run.

Reads ``<run_dir>/metrics_learner.jsonl`` and writes ``<run_dir>/analysis.png``
by default. The figure is a 2x2 panel:

  (top-left)     loss   linear-y
  (top-right)    loss   log-y
  (bottom-left)  throughput timeline (actor + learner samples/s,
                 throttle-sleep fraction on a twin axis)
  (bottom-right) replay-ratio timeline (interval + cumulative,
                 with the configured max_replay cap as a horizontal line)

Usage:
    PYTHONPATH=ml/src ~/miniconda3/bin/python -m guandan.guanzero.analyze \
        --run ml/runs/guanzero_m0_20260503_1337
    PYTHONPATH=ml/src ~/miniconda3/bin/python -m guandan.guanzero.analyze \
        --run <dir> --out /tmp/loss.png
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
SMOOTH_WINDOW = 5


@dataclass
class RunMetrics:
    updates:                np.ndarray  # int64
    losses:                 dict[int, np.ndarray]
    upd_per_sec:            np.ndarray
    samples_per_sec:        np.ndarray  # learner consumption (interval)
    actor_rate_samp_per_s:  np.ndarray  # actor production (interval)
    elapsed_s:              np.ndarray  # cumulative wall
    throttle_sleep_s:       np.ndarray  # cumulative sleep
    replay_interval:        np.ndarray
    replay_cumulative:      np.ndarray


def load_metrics(metrics_path: Path) -> RunMetrics:
    rows: list[dict] = []
    for line in metrics_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    def col(key: str, default=float("nan")) -> np.ndarray:
        return np.asarray([r.get(key, default) for r in rows], dtype=np.float64)

    def _seat_loss(r: dict, p: int) -> float:
        d = r["loss"]
        # M0/M1: keys are "0","1","2","3"
        # M3 shared-head: keys are "loss_seat_0","loss_seat_1",...
        # M3 trick-head: keys are "loss_trick_head_0","loss_trick_head_1",...
        return d.get(str(p), d.get(f"loss_seat_{p}", d.get(f"loss_trick_head_{p}", float("nan"))))

    losses = {
        p: np.asarray([_seat_loss(r, p) for r in rows], dtype=np.float64)
        for p in range(4)
    }
    return RunMetrics(
        updates=               np.asarray([r["updates"] for r in rows], dtype=np.int64),
        losses=                losses,
        upd_per_sec=           col("upd_per_sec"),
        samples_per_sec=       col("samples_per_sec"),
        actor_rate_samp_per_s= col("actor_rate_samp_per_sec"),
        elapsed_s=             col("elapsed_s"),
        throttle_sleep_s=      col("throttle_sleep_s"),
        replay_interval=       col("replay_interval"),
        replay_cumulative=     col("replay_cumulative"),
    )


def causal_rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    """Right-aligned rolling mean — value at index i averages [max(0,i-w+1), i]."""
    out = np.empty_like(x)
    for i in range(len(x)):
        lo = max(0, i - window + 1)
        out[i] = np.nanmean(x[lo : i + 1])
    return out


def _interval_throttle_fraction(elapsed_s: np.ndarray, throttle_sleep_s: np.ndarray) -> np.ndarray:
    """Per-interval fraction of wall time spent in throttle sleep.

    Both inputs are cumulative across the run; we diff to get per-interval
    values then divide. First sample uses cumulative-from-zero values.
    """
    wall_diff = np.diff(elapsed_s, prepend=0.0)
    sleep_diff = np.diff(throttle_sleep_s, prepend=0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(wall_diff > 0, sleep_diff / wall_diff, 0.0)
    return np.clip(frac, 0.0, 1.0)


def _plot_loss(ax: plt.Axes, m: RunMetrics, log_y: bool) -> None:
    """Plot per-position loss vs updates. If ``log_y``, plot log10(loss) on a
    linear axis (so y-tick labels are the log values directly)."""
    transform = (lambda y: np.log10(y)) if log_y else (lambda y: y)
    avg = np.nanmean(np.stack([m.losses[p] for p in range(4)], axis=0), axis=0)
    for p in range(4):
        ax.plot(m.updates, transform(m.losses[p]),
                color=COLORS[p], alpha=0.20, linewidth=0.8)
        ax.plot(m.updates, transform(causal_rolling_mean(m.losses[p], SMOOTH_WINDOW)),
                color=COLORS[p], linewidth=1.6, label=f"p{p}")
    ax.plot(m.updates, transform(causal_rolling_mean(avg, SMOOTH_WINDOW)),
            color="black", linewidth=2.0, linestyle="--", label="avg")
    ax.set_xlabel("Learner updates")
    ax.set_ylabel("log10(MSE loss)" if log_y else "MSE loss")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)


def _plot_throughput(ax: plt.Axes, m: RunMetrics) -> None:
    ax.plot(m.updates, causal_rolling_mean(m.actor_rate_samp_per_s, SMOOTH_WINDOW),
            color="#377eb8", linewidth=1.6, label="actor produce (samp/s)")
    ax.plot(m.updates, causal_rolling_mean(m.samples_per_sec, SMOOTH_WINDOW),
            color="#e41a1c", linewidth=1.6, label="learner consume (samp/s)")
    ax.set_xlabel("Learner updates")
    ax.set_ylabel("samples / s")
    ax.grid(True, alpha=0.3)

    # Throttle-sleep fraction on twin axis (0–100%).
    ax2 = ax.twinx()
    frac = _interval_throttle_fraction(m.elapsed_s, m.throttle_sleep_s) * 100.0
    ax2.plot(m.updates, causal_rolling_mean(frac, SMOOTH_WINDOW),
             color="#4daf4a", linewidth=1.2, linestyle=":", label="throttle %")
    ax2.set_ylabel("learner throttle (% wall)", color="#4daf4a")
    ax2.tick_params(axis="y", colors="#4daf4a")
    ax2.set_ylim(0, 100)

    # Combined legend.
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)


def _plot_replay(ax: plt.Axes, m: RunMetrics, max_replay_cap: float | None) -> None:
    ax.plot(m.updates, causal_rolling_mean(m.replay_interval, SMOOTH_WINDOW),
            color="#377eb8", linewidth=1.6, label="replay (interval)")
    ax.plot(m.updates, m.replay_cumulative,
            color="black", linewidth=1.6, linestyle="--", label="replay (cumulative)")
    if max_replay_cap is not None:
        ax.axhline(max_replay_cap, color="#e41a1c", linewidth=1.0, linestyle=":",
                   label=f"max_replay = {max_replay_cap:g}")
    ax.set_xlabel("Learner updates")
    ax.set_ylabel("replay ratio")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)


def _read_max_replay(run_dir: Path) -> float | None:
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        return None
    try:
        cfg = json.loads(cfg_path.read_text())
        v = cfg.get("max_replay_ratio")
        return float(v) if v is not None else None
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def plot(run_dir: Path, out_path: Path) -> None:
    metrics_path = run_dir / "metrics_learner.jsonl"
    if not metrics_path.exists():
        raise FileNotFoundError(f"no metrics file at {metrics_path}")

    m = load_metrics(metrics_path)
    if len(m.updates) == 0:
        raise RuntimeError(f"{metrics_path} is empty")

    avg = np.nanmean(np.stack([m.losses[p] for p in range(4)], axis=0), axis=0)
    final_avg = float(np.nanmean(avg[-min(20, len(avg)) :]))
    final_rate = float(m.upd_per_sec[-1]) if len(m.upd_per_sec) and not np.isnan(m.upd_per_sec[-1]) else float("nan")
    max_replay_cap = _read_max_replay(run_dir)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    _plot_loss(axes[0, 0], m, log_y=False)
    _plot_loss(axes[0, 1], m, log_y=True)
    _plot_throughput(axes[1, 0], m)
    _plot_replay(axes[1, 1], m, max_replay_cap)

    fig.suptitle(
        f"{run_dir.name}  ·  {m.updates[-1]:,} updates  ·  "
        f"avg loss (last 20) = {final_avg:.4f}  ·  {final_rate:.2f} upd/s"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"wrote {out_path}  ({len(m.updates)} rows, final avg loss {final_avg:.4f})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, type=Path,
                    help="Run directory (must contain metrics_learner.jsonl).")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output PNG path. Default: <run>/analysis.png")
    args = ap.parse_args()

    out_path = args.out or (args.run / "analysis.png")
    plot(args.run, out_path)


if __name__ == "__main__":
    main()
