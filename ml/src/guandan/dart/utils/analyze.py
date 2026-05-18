"""Static analysis: plot per-position learner loss + throughput + replay
control for a Dart run.

Reads ``<run_dir>/metrics_learner.jsonl`` and writes ``<run_dir>/analysis.png``
by default. The figure is a 2x2 panel:

  (top-left)     loss   linear-y
  (top-right)    loss   log-y
  (bottom-left)  throughput timeline (actor + learner samples/s,
                 throttle-sleep fraction on a twin axis)
  (bottom-right) replay-ratio timeline (interval + cumulative,
                 with the configured max_replay cap as a horizontal line)

Usage:
    uv run python -m guandan.dart.analyze --run ml/runs/dart_20260503_1337
    uv run python -m guandan.dart.analyze --run <dir> --out /tmp/loss.png
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
        # GuanZero logs use "0".."3"; Dart logs use "loss_trick_head_0".."..._3".
        return d.get(str(p), d.get(f"loss_trick_head_{p}", float("nan")))

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


def _load_phase_rows(metrics_path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in metrics_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(r.get("phase"), dict) and r["phase"]:
            rows.append(r)
    return rows


def _grid_loss_frac(phase: dict, prefix: str, n_cells: int) -> tuple[np.ndarray, np.ndarray]:
    losses = np.full(n_cells, np.nan, dtype=np.float64)
    fracs = np.zeros(n_cells, dtype=np.float64)
    for c in range(n_cells):
        loss = phase.get(f"{prefix}_{c}_loss")
        frac = phase.get(f"{prefix}_{c}_frac", 0.0)
        if loss is not None:
            losses[c] = float(loss)
        fracs[c] = float(frac or 0.0)
    return losses, fracs


def _stacked_grad_share(phase_rows: list[dict], prefix: str, n_cells: int) -> np.ndarray:
    """Compute per-row gradient-share (loss * frac) per cell. NaN losses → 0."""
    out = np.zeros((len(phase_rows), n_cells), dtype=np.float64)
    for i, r in enumerate(phase_rows):
        phase = r["phase"]
        losses, fracs = _grid_loss_frac(phase, prefix, n_cells)
        losses = np.where(np.isnan(losses), 0.0, losses)
        out[i] = losses * fracs
    return out


def _panel_stacked(ax, updates, share, labels, title):
    n_cells = share.shape[1]
    cmap = plt.get_cmap("viridis", n_cells)
    colors = [cmap(i) for i in range(n_cells)]
    ax.stackplot(updates, share.T, labels=labels, colors=colors, alpha=0.85)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("updates")
    ax.set_ylabel("loss × frac")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=6, ncol=2)


def plot_phase_analysis(run_dir: Path, out_path: Path) -> None:
    metrics_path = run_dir / "metrics_learner.jsonl"
    if not metrics_path.exists():
        raise FileNotFoundError(f"no metrics file at {metrics_path}")
    phase_rows = _load_phase_rows(metrics_path)
    if not phase_rows:
        print(f"no phase sub-dicts in {metrics_path}; skipping phase_analysis.png")
        return

    updates = np.asarray([r["updates"] for r in phase_rows], dtype=np.int64)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Panel 1: phase_role 3x3 (phase_self × trick_role)
    share = _stacked_grad_share(phase_rows, "phase_role", 9)
    labels = [f"phase{a}_role{b}" for a in range(3) for b in range(3)]
    _panel_stacked(axes[0, 0], updates, share, labels, "phase × trick_role")

    # Panel 2: phase_pair 3x4 (phase_self × phase_partner)
    share = _stacked_grad_share(phase_rows, "phase_pair", 12)
    labels = [f"phase{a}_part{b}" for a in range(3) for b in range(4)]
    _panel_stacked(axes[0, 1], updates, share, labels, "phase × phase_partner")

    # Panel 3: source_phase 3x3 (episode_mode × phase_self)
    share = _stacked_grad_share(phase_rows, "source_phase", 9)
    labels = [f"src{a}_phase{b}" for a in range(3) for b in range(3)]
    _panel_stacked(axes[0, 2], updates, share, labels, "episode_mode × phase")

    # Panel 4: opp_phase 6x3 (opp_id-in-top × phase_self)
    share = _stacked_grad_share(phase_rows, "opp_phase", 18)
    labels = [f"opp{a}_phase{b}" for a in range(6) for b in range(3)]
    _panel_stacked(axes[1, 0], updates, share, labels, "opponent × phase")

    # Panel 5: action_phase 6x3
    share = _stacked_grad_share(phase_rows, "action_phase", 18)
    labels = [f"act{a}_phase{b}" for a in range(6) for b in range(3)]
    _panel_stacked(axes[1, 1], updates, share, labels, "action_class × phase")

    # Panel 6: marginal bars at final row
    last = phase_rows[-1]["phase"]
    ax = axes[1, 2]
    marginals = [
        ("epsilon", 2), ("is_pass", 2), ("is_bomb", 2),
        ("k_bucket", 4), ("q_gap", 4), ("team", 2), ("reward", 4),
    ]
    bar_labels: list[str] = []
    bar_vals: list[float] = []
    for prefix, n in marginals:
        for c in range(n):
            v = last.get(f"{prefix}_{c}_loss")
            bar_labels.append(f"{prefix}{c}")
            bar_vals.append(float(v) if v is not None else float("nan"))
    xs = np.arange(len(bar_vals))
    ax.bar(xs, bar_vals, color="#4daf4a")
    ax.set_xticks(xs)
    ax.set_xticklabels(bar_labels, rotation=90, fontsize=6)
    ax.set_title("marginal losses (final row)", fontsize=10)
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"{run_dir.name} · phase diagnostics ({len(phase_rows)} rows)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, type=Path,
                    help="Run directory (must contain metrics_learner.jsonl).")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output PNG path. Default: <run>/analysis.png")
    args = ap.parse_args()

    out_path = args.out or (args.run / "analysis.png")
    plot(args.run, out_path)
    phase_out = args.run / "phase_analysis.png"
    try:
        plot_phase_analysis(args.run, phase_out)
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    main()
