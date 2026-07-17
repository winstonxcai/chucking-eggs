"""Static analysis for a Dart training run.

Reads ``<run_dir>/metrics_learner.jsonl`` and writes a diagnostics bundle:
``training_health.png``, ``training_health_recent_50k.png``,
``systems_health.png``, ``q_diagnostics.png``, ``bucket_diagnostics.*``,
``bucket_shift.*``, and ``health_summary.md``.

  (top-left)     MSE loss by trick-relative head
  (top-right)    gradient norm
  (bottom-left)  actor/learner throughput plus queue depth
  (bottom-right) replay-ratio timeline

Usage:
    uv run guandan-analyze --run ml/runs/dart_20260503_1337
    uv run python -m guandan.scripts.analyze --run <dir> --out /tmp/loss.png
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np

from ..dart.runtime.learners.loss_bucket_schema import (
    LOSS_BUCKET_GRIDS,
    LOSS_BUCKET_MARGINALS,
)

COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
HEAD_LABELS = ("leader", "first responder", "across", "last responder")
SMOOTH_WINDOW = 5


@dataclass
class MetricSegment:
    start_idx: int
    end_idx: int
    start_update: int
    end_update: int
    rows: list[dict]

    @property
    def n_rows(self) -> int:
        return len(self.rows)


@dataclass
class RunMetrics:
    updates:                np.ndarray  # int64
    losses:                 dict[int, np.ndarray]
    upd_per_sec:            np.ndarray
    samples_per_sec:        np.ndarray  # learner consumption (interval)
    actor_rate_samp_per_s:  np.ndarray  # actor production (interval)
    queue_depth:            np.ndarray
    drained_since_log:      np.ndarray
    fresh_samples_total:    np.ndarray
    elapsed_s:              np.ndarray  # cumulative wall
    throttle_sleep_s:       np.ndarray  # cumulative sleep
    replay_interval:        np.ndarray
    replay_cumulative:      np.ndarray
    total_loss:             np.ndarray
    grad_norm:              np.ndarray
    grad_norm_trunk:        np.ndarray
    sample_counts:          dict[int, np.ndarray]
    q_mean:                 dict[int, np.ndarray]
    q_std:                  dict[int, np.ndarray]


def load_metrics(metrics_path: Path) -> RunMetrics:
    rows: list[dict] = []
    for line in metrics_path.open():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    def col(key: str, default=float("nan")) -> np.ndarray:
        return np.asarray([r.get(key, default) for r in rows], dtype=np.float64)

    def loss_col(key: str, default=float("nan")) -> np.ndarray:
        vals = []
        for r in rows:
            d = r.get("loss", {})
            vals.append(d.get(key, default) if isinstance(d, dict) else default)
        return np.asarray(vals, dtype=np.float64)

    def wall_clock_col() -> np.ndarray:
        return np.asarray(
            [r.get("wall_clock_s", r.get("elapsed_s", float("nan"))) for r in rows],
            dtype=np.float64,
        )

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
        queue_depth=           col("queue_depth"),
        drained_since_log=     col("drained_since_last_log"),
        fresh_samples_total=   col("fresh_samples_total"),
        elapsed_s=             wall_clock_col(),
        throttle_sleep_s=      col("throttle_sleep_s"),
        replay_interval=       col("replay_interval"),
        replay_cumulative=     col("replay_cumulative"),
        total_loss=            loss_col("loss"),
        grad_norm=             loss_col("grad_norm"),
        grad_norm_trunk=       loss_col("grad_norm_trunk"),
        sample_counts={
            p: loss_col(f"sample_count_trick_head_{p}") for p in range(4)
        },
        q_mean={
            p: loss_col(f"q_mean_trick_head_{p}") for p in range(4)
        },
        q_std={
            p: loss_col(f"q_std_trick_head_{p}") for p in range(4)
        },
    )


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.open():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _counter_decreased(row: dict, prev: dict, key: str) -> bool:
    current = row.get(key)
    previous = prev.get(key)
    if not isinstance(current, (int, float)) or not isinstance(previous, (int, float)):
        return False
    return current < previous


def _split_resume_segments(rows: list[dict], *, max_update_gap: int = 1000) -> list[MetricSegment]:
    segments: list[MetricSegment] = []
    current: list[dict] = []
    start_idx = 1
    prev: dict | None = None

    for idx, row in enumerate(rows, start=1):
        update = int(row.get("updates", 0))
        boundary = False
        if prev is not None:
            prev_update = int(prev.get("updates", update))
            update_delta = update - prev_update
            boundary = (
                update_delta <= 0
                or update_delta > max_update_gap
                or _counter_decreased(row, prev, "elapsed_s")
                or _counter_decreased(row, prev, "fresh_samples_total")
                or _counter_decreased(row, prev, "cumulative_drained")
                or _counter_decreased(row, prev, "throttle_sleep_s")
            )

        if boundary and current:
            segments.append(MetricSegment(
                start_idx=start_idx,
                end_idx=idx - 1,
                start_update=int(current[0]["updates"]),
                end_update=int(current[-1]["updates"]),
                rows=current,
            ))
            current = []
            start_idx = idx

        current.append(row)
        prev = row

    if current:
        segments.append(MetricSegment(
            start_idx=start_idx,
            end_idx=start_idx + len(current) - 1,
            start_update=int(current[0]["updates"]),
            end_update=int(current[-1]["updates"]),
            rows=current,
        ))
    return segments


def _drop_contained_segments(segments: list[MetricSegment]) -> list[MetricSegment]:
    kept: list[MetricSegment] = []
    for i, segment in enumerate(segments):
        contained = False
        for j, other in enumerate(segments):
            if i == j:
                continue
            if (
                other.start_update <= segment.start_update
                and other.end_update >= segment.end_update
                and other.n_rows > segment.n_rows
            ):
                contained = True
                break
        if not contained:
            kept.append(segment)
    return kept


def _apply_cumulative_offsets(rows: list[dict], cumulative_keys: tuple[str, ...]) -> None:
    offsets = {key: 0.0 for key in cumulative_keys}
    segment_start_values: dict[str, float] = {}
    current_segment: int | None = None
    previous_row: dict | None = None

    for row in rows:
        segment_id = int(row["_postprocess"]["segment_id"])
        if segment_id != current_segment:
            if previous_row is not None:
                for key in cumulative_keys:
                    value = previous_row.get(key)
                    if isinstance(value, (int, float)):
                        offsets[key] = float(value)
            current_segment = segment_id
            segment_start_values = {
                key: float(row.get(key, 0.0))
                for key in cumulative_keys
                if isinstance(row.get(key), (int, float))
            }

        for key in cumulative_keys:
            value = row.get(key)
            if not isinstance(value, (int, float)):
                continue
            start_value = segment_start_values.get(key, 0.0)
            row[key] = round(offsets[key] + max(0.0, float(value) - start_value), 6)
        previous_row = row


def postprocess_resume_metrics(
    metrics_path: Path,
    out_path: Path,
    *,
    warmup_rows: int = 20,
) -> dict:
    """Clean append-only Modal metrics produced across multiple resumed jobs.

    The raw log is preserved. The derived file removes superseded duplicate
    resume branches, cuts each retained segment at the next resume start, drops
    a tiny warmup after each resume, and offsets cumulative counters so systems
    plots do not show artificial resets.
    """
    rows = _read_jsonl(metrics_path)
    raw_segments = _split_resume_segments(rows)
    kept_segments = _drop_contained_segments(raw_segments)

    selected: list[dict] = []
    segment_summaries: list[dict] = []
    duplicate_updates = 0
    seen_updates: set[int] = set()

    for i, segment in enumerate(kept_segments):
        next_start = (
            kept_segments[i + 1].start_update
            if i + 1 < len(kept_segments)
            else None
        )
        segment_rows = [
            dict(row)
            for row in segment.rows
            if next_start is None or int(row["updates"]) < next_start
        ]
        dropped_for_next = segment.n_rows - len(segment_rows)
        warmup_drop = min(warmup_rows if i > 0 else 0, max(0, len(segment_rows) - 1))
        segment_rows = segment_rows[warmup_drop:]

        kept_count = 0
        for row in segment_rows:
            update = int(row["updates"])
            if update in seen_updates:
                duplicate_updates += 1
                continue
            seen_updates.add(update)
            row["_postprocess"] = {
                "segment_id": i,
                "source_start_idx": segment.start_idx,
                "source_end_idx": segment.end_idx,
            }
            selected.append(row)
            kept_count += 1

        segment_summaries.append({
            "segment_id": i,
            "source_start_idx": segment.start_idx,
            "source_end_idx": segment.end_idx,
            "source_start_update": segment.start_update,
            "source_end_update": segment.end_update,
            "source_rows": segment.n_rows,
            "cut_at_next_start_update": next_start,
            "dropped_for_next_resume": dropped_for_next,
            "dropped_warmup_rows": warmup_drop,
            "kept_rows": kept_count,
        })

    selected.sort(key=lambda row: int(row["updates"]))
    _apply_cumulative_offsets(
        selected,
        (
            "elapsed_s",
            "fresh_samples_total",
            "cumulative_drained",
            "throttle_sleep_s",
            "throttle_sleeps",
        ),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for row in selected:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")

    summary = {
        "raw_metrics": str(metrics_path),
        "clean_metrics": str(out_path),
        "raw_rows": len(rows),
        "clean_rows": len(selected),
        "raw_segments": [
            {
                "start_idx": s.start_idx,
                "end_idx": s.end_idx,
                "start_update": s.start_update,
                "end_update": s.end_update,
                "rows": s.n_rows,
            }
            for s in raw_segments
        ],
        "kept_segments": segment_summaries,
        "dropped_contained_segments": len(raw_segments) - len(kept_segments),
        "duplicate_updates_skipped": duplicate_updates,
        "warmup_rows_per_resume": warmup_rows,
        "update_start": int(selected[0]["updates"]) if selected else None,
        "update_end": int(selected[-1]["updates"]) if selected else None,
    }
    return summary


def postprocess_run_metrics(run_dir: Path, out_dir: Path, *, warmup_rows: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics_learner.jsonl"
    out_metrics = out_dir / "metrics_learner.jsonl"
    summary = postprocess_resume_metrics(metrics_path, out_metrics, warmup_rows=warmup_rows)
    summary_path = out_dir / "postprocess_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        shutil.copy2(cfg_path, out_dir / "config.json")
    print(f"wrote {out_metrics}")
    print(f"wrote {summary_path}")
    return out_dir


def _slice_metrics(m: RunMetrics, mask: np.ndarray) -> RunMetrics:
    return RunMetrics(
        updates=m.updates[mask],
        losses={p: v[mask] for p, v in m.losses.items()},
        upd_per_sec=m.upd_per_sec[mask],
        samples_per_sec=m.samples_per_sec[mask],
        actor_rate_samp_per_s=m.actor_rate_samp_per_s[mask],
        queue_depth=m.queue_depth[mask],
        drained_since_log=m.drained_since_log[mask],
        fresh_samples_total=m.fresh_samples_total[mask],
        elapsed_s=m.elapsed_s[mask],
        throttle_sleep_s=m.throttle_sleep_s[mask],
        replay_interval=m.replay_interval[mask],
        replay_cumulative=m.replay_cumulative[mask],
        total_loss=m.total_loss[mask],
        grad_norm=m.grad_norm[mask],
        grad_norm_trunk=m.grad_norm_trunk[mask],
        sample_counts={p: v[mask] for p, v in m.sample_counts.items()},
        q_mean={p: v[mask] for p, v in m.q_mean.items()},
        q_std={p: v[mask] for p, v in m.q_std.items()},
    )


def causal_rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    """Right-aligned rolling mean — value at index i averages [max(0,i-w+1), i]."""
    out = np.empty_like(x)
    for i in range(len(x)):
        lo = max(0, i - window + 1)
        chunk = x[lo : i + 1]
        out[i] = np.nan if np.isnan(chunk).all() else np.nanmean(chunk)
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
                color=COLORS[p], linewidth=1.6, label=HEAD_LABELS[p])
    ax.plot(m.updates, transform(causal_rolling_mean(avg, SMOOTH_WINDOW)),
            color="black", linewidth=2.0, linestyle="--", label="avg")
    ax.set_xlabel("Learner updates")
    ax.set_ylabel("log10(MSE loss)" if log_y else "MSE loss")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)


def _plot_grad_norm(ax: plt.Axes, m: RunMetrics) -> None:
    ax.plot(
        m.updates,
        causal_rolling_mean(m.grad_norm, SMOOTH_WINDOW),
        color="#984ea3",
        linewidth=1.8,
        label="total",
    )
    if not np.isnan(m.grad_norm_trunk).all():
        ax.plot(
            m.updates,
            causal_rolling_mean(m.grad_norm_trunk, SMOOTH_WINDOW),
            color="#4daf4a",
            linewidth=1.4,
            label="trunk",
        )
    ax.set_xlabel("Learner updates")
    ax.set_ylabel("grad norm")
    if np.nanmax(m.grad_norm) > 0:
        ax.set_yscale("log")
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

    h1, l1 = ax.get_legend_handles_labels()
    queue = np.asarray(m.queue_depth, dtype=np.float64)
    valid_queue = queue[np.isfinite(queue) & (queue >= 0)]
    if len(valid_queue):
        ax2 = ax.twinx()
        ax2.plot(m.updates, causal_rolling_mean(queue, SMOOTH_WINDOW),
                 color="#4daf4a", linewidth=1.2, linestyle=":", label="queue depth")
        ax2.set_ylabel("queue depth", color="#4daf4a")
        ax2.tick_params(axis="y", colors="#4daf4a")
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)
    else:
        ax.legend(h1, l1, loc="upper right", fontsize=8)


def _plot_replay(ax: plt.Axes, m: RunMetrics, max_replay_cap: float | None) -> None:
    interval = causal_rolling_mean(m.replay_interval, SMOOTH_WINDOW)
    finite_interval = interval[np.isfinite(interval)]
    clip_label = "replay (interval)"
    if len(finite_interval):
        p99 = float(np.nanpercentile(finite_interval, 99))
        y_cap = max(
            p99,
            float(max_replay_cap or 0.0),
            float(np.nanmax(m.replay_cumulative[np.isfinite(m.replay_cumulative)])),
        )
        if np.nanmax(finite_interval) > y_cap * 1.5:
            interval = np.minimum(interval, y_cap)
            clip_label = "replay (interval, clipped)"
    ax.plot(m.updates, interval,
            color="#377eb8", linewidth=1.6, label=clip_label)
    ax.plot(m.updates, m.replay_cumulative,
            color="black", linewidth=1.6, linestyle="--", label="replay (cumulative)")
    if max_replay_cap is not None:
        ax.axhline(max_replay_cap, color="#e41a1c", linewidth=1.0, linestyle=":",
                   label=f"max_replay = {max_replay_cap:g}")
    ax.set_xlabel("Learner updates")
    ax.set_ylabel("replay ratio")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)


def _plot_sample_counts(ax: plt.Axes, m: RunMetrics) -> None:
    total = np.zeros_like(m.updates, dtype=np.float64)
    for p in range(4):
        total += np.nan_to_num(m.sample_counts[p], nan=0.0)
    for p in range(4):
        ax.plot(
            m.updates,
            causal_rolling_mean(m.sample_counts[p], SMOOTH_WINDOW),
            color=COLORS[p],
            linewidth=1.5,
            label=HEAD_LABELS[p],
        )
    ax.set_xlabel("Learner updates")
    ax.set_ylabel("samples / head / batch")
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


def plot(run_dir: Path, out_path: Path, *, recent_updates: int | None = None) -> None:
    metrics_path = run_dir / "metrics_learner.jsonl"
    if not metrics_path.exists():
        raise FileNotFoundError(f"no metrics file at {metrics_path}")

    m = load_metrics(metrics_path)
    if len(m.updates) == 0:
        raise RuntimeError(f"{metrics_path} is empty")
    if recent_updates is not None and len(m.updates):
        start = m.updates[-1] - recent_updates
        m = _slice_metrics(m, m.updates >= start)

    avg = np.nanmean(np.stack([m.losses[p] for p in range(4)], axis=0), axis=0)
    final_avg = float(np.nanmean(avg[-min(20, len(avg)) :]))
    final_rate = float(m.upd_per_sec[-1]) if len(m.upd_per_sec) and not np.isnan(m.upd_per_sec[-1]) else float("nan")
    max_replay_cap = _read_max_replay(run_dir)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    _plot_loss(axes[0, 0], m, log_y=False)
    _plot_grad_norm(axes[0, 1], m)
    _plot_throughput(axes[1, 0], m)
    _plot_replay(axes[1, 1], m, max_replay_cap)

    fig.suptitle(
        f"{run_dir.name}  ·  {m.updates[0]:,}-{m.updates[-1]:,} updates  ·  "
        f"avg loss (last 20) = {final_avg:.4f}  ·  {final_rate:.2f} upd/s"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"wrote {out_path}  ({len(m.updates)} rows, final avg loss {final_avg:.4f})")


def plot_systems_health(run_dir: Path, out_path: Path) -> None:
    metrics_path = run_dir / "metrics_learner.jsonl"
    m = load_metrics(metrics_path)
    if len(m.updates) == 0:
        raise RuntimeError(f"{metrics_path} is empty")
    max_replay_cap = _read_max_replay(run_dir)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    _plot_throughput(axes[0, 0], m)
    _plot_replay(axes[0, 1], m, max_replay_cap)

    queue = m.queue_depth
    valid_queue = queue[np.isfinite(queue) & (queue >= 0)]
    if len(valid_queue):
        axes[1, 0].plot(m.updates, causal_rolling_mean(queue, SMOOTH_WINDOW), color="#4daf4a")
        axes[1, 0].set_ylabel("queue depth")
    else:
        axes[1, 0].plot(m.updates, causal_rolling_mean(m.drained_since_log, SMOOTH_WINDOW), color="#4daf4a")
        axes[1, 0].set_ylabel("drained queue batches / log")
    axes[1, 0].set_xlabel("Learner updates")
    axes[1, 0].grid(True, alpha=0.3)

    throttle = _interval_throttle_fraction(m.elapsed_s, m.throttle_sleep_s) * 100.0
    axes[1, 1].plot(m.updates, causal_rolling_mean(throttle, SMOOTH_WINDOW), color="#ff7f00")
    axes[1, 1].set_xlabel("Learner updates")
    axes[1, 1].set_ylabel("throttle sleep (% wall)")
    axes[1, 1].set_ylim(bottom=0)
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(f"{run_dir.name} · systems health ({m.updates[-1]:,} updates)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_q_diagnostics(run_dir: Path, out_path: Path) -> None:
    metrics_path = run_dir / "metrics_learner.jsonl"
    m = load_metrics(metrics_path)
    if len(m.updates) == 0:
        raise RuntimeError(f"{metrics_path} is empty")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    for p in range(4):
        axes[0, 0].plot(
            m.updates,
            causal_rolling_mean(m.q_mean[p], SMOOTH_WINDOW),
            color=COLORS[p],
            linewidth=1.4,
            label=HEAD_LABELS[p],
        )
        axes[0, 1].plot(
            m.updates,
            causal_rolling_mean(m.q_std[p], SMOOTH_WINDOW),
            color=COLORS[p],
            linewidth=1.4,
            label=HEAD_LABELS[p],
        )
    axes[0, 0].set_title("Q mean by head")
    axes[0, 0].set_ylabel("Q mean")
    axes[0, 1].set_title("Q std by head")
    axes[0, 1].set_ylabel("Q std")
    for ax in axes[0]:
        ax.set_xlabel("Learner updates")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    _plot_sample_counts(axes[1, 0], m)
    _plot_loss(axes[1, 1], m, log_y=False)
    axes[1, 1].set_title("Loss by head")

    fig.suptitle(f"{run_dir.name} · Q diagnostics ({m.updates[-1]:,} updates)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"wrote {out_path}")


def _load_phase_rows(metrics_path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in metrics_path.open():
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


def _grid_loss_frac(phase: dict, spec) -> tuple[np.ndarray, np.ndarray]:
    n_cells = len(spec.axis_a.labels) * len(spec.axis_b.labels)
    losses = np.full(n_cells, np.nan, dtype=np.float64)
    fracs = np.zeros(n_cells, dtype=np.float64)
    i = 0
    for a in range(len(spec.axis_a.labels)):
        for b in range(len(spec.axis_b.labels)):
            loss = phase.get(spec.key(a, b, "loss"))
            frac = phase.get(spec.key(a, b, "frac"), 0.0)
            if loss is not None:
                losses[i] = float(loss)
            fracs[i] = float(frac or 0.0)
            i += 1
    return losses, fracs


def _marginal_loss(phase: dict, spec) -> tuple[list[str], list[float]]:
    labels: list[str] = []
    vals: list[float] = []
    for level, label in enumerate(spec.axis.labels):
        loss = phase.get(spec.key(level, "loss"))
        labels.append(f"{spec.name}:{label}")
        vals.append(float(loss) if loss is not None else float("nan"))
    return labels, vals


class BucketRecord(NamedTuple):
    kind: str
    family: str
    label: str
    mean_loss: float
    mean_frac: float
    mean_n: float
    mean_contribution: float


class BucketShiftRecord(NamedTuple):
    kind: str
    family: str
    label: str
    first_loss: float
    last_loss: float
    delta_loss: float
    first_frac: float
    last_frac: float
    delta_frac: float
    first_n: float
    last_n: float


def _clean_label(label: str) -> str:
    return label.replace("_", " ")


def _grid_labels(spec) -> list[str]:
    return [
        f"{a_label}/{b_label}"
        for a_label in spec.axis_a.labels
        for b_label in spec.axis_b.labels
    ]


def _stacked_grad_share(phase_rows: list[dict], spec) -> np.ndarray:
    """Compute per-row gradient-share (loss * frac) per cell. NaN losses → 0."""
    n_cells = len(spec.axis_a.labels) * len(spec.axis_b.labels)
    out = np.zeros((len(phase_rows), n_cells), dtype=np.float64)
    for i, r in enumerate(phase_rows):
        phase = r["phase"]
        losses, fracs = _grid_loss_frac(phase, spec)
        losses = np.where(np.isnan(losses), 0.0, losses)
        out[i] = losses * fracs
    return out


def _legacy_grid_loss_frac(phase: dict, prefix: str, n_cells: int) -> tuple[np.ndarray, np.ndarray]:
    losses = np.full(n_cells, np.nan, dtype=np.float64)
    fracs = np.zeros(n_cells, dtype=np.float64)
    for c in range(n_cells):
        loss = phase.get(f"{prefix}_{c}_loss")
        frac = phase.get(f"{prefix}_{c}_frac", 0.0)
        if loss is not None:
            losses[c] = float(loss)
        fracs[c] = float(frac or 0.0)
    return losses, fracs


def _legacy_stacked_grad_share(phase_rows: list[dict], prefix: str, n_cells: int) -> np.ndarray:
    """Compute per-row gradient-share (loss * frac) per cell. NaN losses → 0."""
    out = np.zeros((len(phase_rows), n_cells), dtype=np.float64)
    for i, r in enumerate(phase_rows):
        phase = r["phase"]
        losses, fracs = _legacy_grid_loss_frac(phase, prefix, n_cells)
        losses = np.where(np.isnan(losses), 0.0, losses)
        out[i] = losses * fracs
    return out


_LEGACY_GRID_DIMS = {
    "phase_role": (3, 3, "phase{a}_role{b}"),
    "phase_pair": (3, 4, "phase{a}_part{b}"),
    "source_phase": (3, 3, "src{a}_phase{b}"),
    "opp_phase": (6, 3, "opp{a}_phase{b}"),
    "action_phase": (6, 3, "act{a}_phase{b}"),
}


def _share_and_labels(
    phase_rows: list[dict],
    spec,
    use_legacy_keys: bool,
) -> tuple[np.ndarray, list[str]]:
    if not use_legacy_keys:
        return _stacked_grad_share(phase_rows, spec), _grid_labels(spec)

    n_a, n_b, label_fmt = _LEGACY_GRID_DIMS[spec.name]
    labels = [label_fmt.format(a=a, b=b) for a in range(n_a) for b in range(n_b)]
    return _legacy_stacked_grad_share(phase_rows, spec.name, n_a * n_b), labels


def _finite_mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if len(arr) else float("nan")


def _collect_grid_records(phase_rows: list[dict], use_legacy_keys: bool) -> list[BucketRecord]:
    records: list[BucketRecord] = []
    for spec in LOSS_BUCKET_GRIDS:
        if use_legacy_keys:
            n_a, n_b, label_fmt = _LEGACY_GRID_DIMS[spec.name]
            cell_iter = [
                (
                    a,
                    b,
                    label_fmt.format(a=a, b=b),
                    f"{spec.name}_{a * n_b + b}_loss",
                    f"{spec.name}_{a * n_b + b}_frac",
                    f"{spec.name}_{a * n_b + b}_n",
                )
                for a in range(n_a)
                for b in range(n_b)
            ]
        else:
            cell_iter = [
                (
                    a,
                    b,
                    f"{_clean_label(spec.axis_a.labels[a])}/{_clean_label(spec.axis_b.labels[b])}",
                    spec.key(a, b, "loss"),
                    spec.key(a, b, "frac"),
                    spec.key(a, b, "n"),
                )
                for a in range(len(spec.axis_a.labels))
                for b in range(len(spec.axis_b.labels))
            ]

        for _a, _b, label, loss_key, frac_key, n_key in cell_iter:
            losses: list[float] = []
            fracs: list[float] = []
            ns: list[float] = []
            contributions: list[float] = []
            for row in phase_rows:
                phase = row["phase"]
                loss = phase.get(loss_key)
                frac = float(phase.get(frac_key, 0.0) or 0.0)
                n = float(phase.get(n_key, 0.0) or 0.0)
                fracs.append(frac)
                ns.append(n)
                if loss is None:
                    losses.append(float("nan"))
                    contributions.append(0.0)
                else:
                    loss_f = float(loss)
                    losses.append(loss_f)
                    contributions.append(loss_f * frac if np.isfinite(loss_f) else 0.0)
            records.append(BucketRecord(
                kind="grid",
                family=spec.name,
                label=f"{spec.name}: {label}",
                mean_loss=_finite_mean(losses),
                mean_frac=float(np.mean(fracs)) if fracs else 0.0,
                mean_n=float(np.mean(ns)) if ns else 0.0,
                mean_contribution=float(np.mean(contributions)) if contributions else 0.0,
            ))
    return records


def _collect_marginal_records(
    phase_rows: list[dict],
    use_legacy_keys: bool,
) -> list[BucketRecord]:
    records: list[BucketRecord] = []
    if use_legacy_keys:
        legacy_specs = [
            ("epsilon", 2), ("is_pass", 2), ("is_bomb", 2),
            ("k_bucket", 4), ("q_gap", 4), ("team", 2), ("reward", 4),
        ]
        for family, n_levels in legacy_specs:
            for level in range(n_levels):
                label = f"{family}{level}"
                loss_key = f"{family}_{level}_loss"
                frac_key = f"{family}_{level}_frac"
                n_key = f"{family}_{level}_n"
                records.append(_collect_one_marginal(phase_rows, family, label, loss_key, frac_key, n_key))
        return records

    for spec in LOSS_BUCKET_MARGINALS:
        for level, level_label in enumerate(spec.axis.labels):
            records.append(_collect_one_marginal(
                phase_rows,
                spec.name,
                f"{spec.name}: {_clean_label(level_label)}",
                spec.key(level, "loss"),
                spec.key(level, "frac"),
                spec.key(level, "n"),
            ))
    return records


def _collect_one_marginal(
    phase_rows: list[dict],
    family: str,
    label: str,
    loss_key: str,
    frac_key: str,
    n_key: str,
) -> BucketRecord:
    losses: list[float] = []
    fracs: list[float] = []
    ns: list[float] = []
    contributions: list[float] = []
    for row in phase_rows:
        phase = row["phase"]
        loss = phase.get(loss_key)
        frac = float(phase.get(frac_key, 0.0) or 0.0)
        n = float(phase.get(n_key, 0.0) or 0.0)
        fracs.append(frac)
        ns.append(n)
        if loss is None:
            losses.append(float("nan"))
            contributions.append(0.0)
        else:
            loss_f = float(loss)
            losses.append(loss_f)
            contributions.append(loss_f * frac if np.isfinite(loss_f) else 0.0)
    return BucketRecord(
        kind="marginal",
        family=family,
        label=label,
        mean_loss=_finite_mean(losses),
        mean_frac=float(np.mean(fracs)) if fracs else 0.0,
        mean_n=float(np.mean(ns)) if ns else 0.0,
        mean_contribution=float(np.mean(contributions)) if contributions else 0.0,
    )


def _top_records(
    records: list[BucketRecord],
    field: str,
    *,
    min_mean_n: float = 32.0,
    max_items: int = 12,
) -> list[BucketRecord]:
    return sorted(
        [
            r for r in records
            if r.mean_n >= min_mean_n and np.isfinite(getattr(r, field))
        ],
        key=lambda r: getattr(r, field),
        reverse=True,
    )[:max_items]


def _records_for_families(
    records: list[BucketRecord],
    families: set[str],
) -> list[BucketRecord]:
    return [record for record in records if record.family in families]


def _barh_records(
    ax: plt.Axes,
    records: list[BucketRecord],
    field: str,
    title: str,
    xlabel: str,
    color: str,
) -> None:
    if not records:
        ax.text(0.5, 0.5, "no supported buckets", ha="center", va="center")
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        return
    labels = [r.label for r in records]
    values = [getattr(r, field) for r in records]
    y = np.arange(len(records))
    ax.barh(y, values, color=color)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.grid(True, alpha=0.3, axis="x")
    x_max = max(values) if values else 0.0
    for yi, record, value in zip(y, records, values, strict=False):
        ax.text(
            value + x_max * 0.01 if x_max else value,
            yi,
            f"n={record.mean_n:.0f}, f={record.mean_frac:.2f}",
            va="center",
            fontsize=6,
        )


def _write_bucket_csv(out_path: Path, records: list[BucketRecord]) -> None:
    csv_path = out_path.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(BucketRecord._fields))
        writer.writeheader()
        for record in sorted(records, key=lambda r: (r.kind, r.family, r.label)):
            writer.writerow(record._asdict())
    print(f"wrote {csv_path}")


def _write_shift_csv(out_path: Path, records: list[BucketShiftRecord]) -> None:
    csv_path = out_path.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(BucketShiftRecord._fields))
        writer.writeheader()
        for record in sorted(records, key=lambda r: (r.kind, r.family, r.label)):
            writer.writerow(record._asdict())
    print(f"wrote {csv_path}")


def _bucket_records_for_window(
    phase_rows: list[dict],
    use_legacy_keys: bool,
) -> list[BucketRecord]:
    return (
        _collect_grid_records(phase_rows, use_legacy_keys)
        + _collect_marginal_records(phase_rows, use_legacy_keys)
    )


def _bucket_shift_records(
    phase_rows: list[dict],
    use_legacy_keys: bool,
    window_rows: int = 20,
) -> list[BucketShiftRecord]:
    if len(phase_rows) < 2:
        return []
    first_window = phase_rows[:min(window_rows, len(phase_rows))]
    last_window = phase_rows[-min(window_rows, len(phase_rows)):]
    first = {
        (r.kind, r.family, r.label): r
        for r in _bucket_records_for_window(first_window, use_legacy_keys)
    }
    last = {
        (r.kind, r.family, r.label): r
        for r in _bucket_records_for_window(last_window, use_legacy_keys)
    }
    out: list[BucketShiftRecord] = []
    for key in sorted(set(first) & set(last)):
        a = first[key]
        b = last[key]
        if not (np.isfinite(a.mean_loss) and np.isfinite(b.mean_loss)):
            continue
        out.append(BucketShiftRecord(
            kind=b.kind,
            family=b.family,
            label=b.label,
            first_loss=a.mean_loss,
            last_loss=b.mean_loss,
            delta_loss=b.mean_loss - a.mean_loss,
            first_frac=a.mean_frac,
            last_frac=b.mean_frac,
            delta_frac=b.mean_frac - a.mean_frac,
            first_n=a.mean_n,
            last_n=b.mean_n,
        ))
    return out


def _top_shift(
    records: list[BucketShiftRecord],
    field: str,
    *,
    positive: bool = True,
    min_n: float = 32.0,
    max_items: int = 12,
) -> list[BucketShiftRecord]:
    rows = [
        r for r in records
        if min(r.first_n, r.last_n) >= min_n and np.isfinite(getattr(r, field))
    ]
    return sorted(rows, key=lambda r: getattr(r, field), reverse=positive)[:max_items]


def _barh_shift(
    ax: plt.Axes,
    records: list[BucketShiftRecord],
    field: str,
    title: str,
    xlabel: str,
    color: str,
) -> None:
    if not records:
        ax.text(0.5, 0.5, "no supported buckets", ha="center", va="center")
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        return
    labels = [r.label for r in records]
    values = [getattr(r, field) for r in records]
    y = np.arange(len(records))
    ax.barh(y, values, color=color)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.grid(True, alpha=0.3, axis="x")
    span = max(abs(float(np.nanmin(values))), abs(float(np.nanmax(values))), 1e-9)
    for yi, record, value in zip(y, records, values, strict=False):
        ax.text(
            value + np.sign(value or 1.0) * span * 0.02,
            yi,
            f"n={record.last_n:.0f}, f={record.last_frac:.2f}",
            va="center",
            fontsize=6,
        )


def plot_bucket_diagnostics(run_dir: Path, out_path: Path) -> None:
    metrics_path = run_dir / "metrics_learner.jsonl"
    if not metrics_path.exists():
        raise FileNotFoundError(f"no metrics file at {metrics_path}")
    phase_rows = _load_phase_rows(metrics_path)
    if not phase_rows:
        print(f"no phase sub-dicts in {metrics_path}; skipping bucket_diagnostics.png")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    use_legacy_keys = "phase_role_opening__leading_loss" not in phase_rows[-1]["phase"]

    window = phase_rows[-min(20, len(phase_rows)) :]
    grid_records = _collect_grid_records(window, use_legacy_keys)
    marginal_records = _collect_marginal_records(window, use_legacy_keys)
    all_records = grid_records + marginal_records

    _barh_records(
        axes[0, 0],
        _top_records(_records_for_families(grid_records, {"phase_role", "phase_pair"}), "mean_loss"),
        "mean_loss",
        "highest phase/role losses",
        "mean loss",
        "#e41a1c",
    )
    _barh_records(
        axes[0, 1],
        _top_records(_records_for_families(grid_records, {"action_phase"}), "mean_loss"),
        "mean_loss",
        "highest action losses",
        "mean loss",
        "#377eb8",
    )
    _barh_records(
        axes[0, 2],
        _top_records(
            _records_for_families(grid_records, {"source_phase", "opp_phase"}),
            "mean_frac",
            min_mean_n=1.0,
        ),
        "mean_frac",
        "source/opponent mix",
        "mean sample fraction",
        "#4daf4a",
    )
    _barh_records(
        axes[1, 0],
        _top_records(marginal_records, "mean_loss"),
        "mean_loss",
        "highest marginal losses",
        "mean loss",
        "#984ea3",
    )
    _barh_records(
        axes[1, 1],
        _top_records(marginal_records, "mean_contribution"),
        "mean_contribution",
        "largest marginal contribution",
        "mean loss x sample fraction",
        "#ff7f00",
    )
    _barh_records(
        axes[1, 2],
        _top_records(marginal_records, "mean_frac", min_mean_n=1.0),
        "mean_frac",
        "dominant marginal buckets",
        "mean sample fraction",
        "#a6cee3",
    )

    fig.suptitle(
        f"{run_dir.name} · bucket diagnostics "
        f"(last {len(window)} metric rows, through update {phase_rows[-1]['updates']:,})"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"wrote {out_path}")
    _write_bucket_csv(out_path, all_records)


def plot_bucket_shift(run_dir: Path, out_path: Path, *, window_rows: int = 20) -> None:
    metrics_path = run_dir / "metrics_learner.jsonl"
    if not metrics_path.exists():
        raise FileNotFoundError(f"no metrics file at {metrics_path}")
    phase_rows = _load_phase_rows(metrics_path)
    if len(phase_rows) < 2:
        print(f"not enough phase rows in {metrics_path}; skipping bucket_shift.png")
        return
    use_legacy_keys = "phase_role_opening__leading_loss" not in phase_rows[-1]["phase"]
    records = _bucket_shift_records(phase_rows, use_legacy_keys, window_rows=window_rows)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    _barh_shift(
        axes[0],
        _top_shift(records, "delta_loss", positive=True),
        "delta_loss",
        "loss increased most",
        "last loss - first loss",
        "#e41a1c",
    )
    _barh_shift(
        axes[1],
        _top_shift(records, "delta_loss", positive=False),
        "delta_loss",
        "loss decreased most",
        "last loss - first loss",
        "#377eb8",
    )
    _barh_shift(
        axes[2],
        _top_shift(records, "delta_frac", positive=True, min_n=1.0),
        "delta_frac",
        "sample fraction increased most",
        "last frac - first frac",
        "#4daf4a",
    )

    fig.suptitle(
        f"{run_dir.name} · bucket shift "
        f"(first {min(window_rows, len(phase_rows))} rows vs last {min(window_rows, len(phase_rows))} rows)"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"wrote {out_path}")
    _write_shift_csv(out_path, records)


def write_health_summary(run_dir: Path, out_path: Path) -> None:
    metrics_path = run_dir / "metrics_learner.jsonl"
    m = load_metrics(metrics_path)
    if len(m.updates) == 0:
        raise RuntimeError(f"{metrics_path} is empty")

    warnings: list[str] = []
    avg = np.nanmean(np.stack([m.losses[p] for p in range(4)], axis=0), axis=0)
    recent_n = min(20, len(avg))
    final_avg = float(np.nanmean(avg[-recent_n:]))
    recent_upd_per_sec = float(np.nanmedian(m.upd_per_sec[-recent_n:]))
    recent_samples_per_sec = float(np.nanmedian(m.samples_per_sec[-recent_n:]))
    recent_actor_rate = float(np.nanmedian(m.actor_rate_samp_per_s[-recent_n:]))
    recent_queue_depth = float(np.nanmedian(m.queue_depth[-recent_n:]))
    max_replay_cap = _read_max_replay(run_dir)
    finite_numbers = np.concatenate([
        avg[np.isfinite(avg)],
        m.grad_norm[np.isfinite(m.grad_norm)],
        m.replay_interval[np.isfinite(m.replay_interval)],
    ])
    if len(finite_numbers) == 0:
        warnings.append("No finite learner scalar metrics found.")
    if not np.isfinite(final_avg):
        warnings.append("Final average loss is non-finite.")

    if max_replay_cap is not None:
        finite_replay = m.replay_interval[np.isfinite(m.replay_interval)]
        if len(finite_replay):
            # The logged interval ratio is rounded and often hovers at 1.25-1.27
            # around a 1.25 cap. Warn only on material overshoot.
            exceed = float(np.mean(finite_replay > max_replay_cap + 0.05))
            if exceed > 0.05:
                warnings.append(
                    f"Replay interval exceeded max_replay_ratio={max_replay_cap:g} on {exceed:.1%} of logged intervals."
                )

    queue = m.queue_depth[np.isfinite(m.queue_depth) & (m.queue_depth >= 0)]
    if len(queue) and np.nanmedian(queue) <= 1:
        warnings.append("Queue depth is often near empty; actor throughput may be limiting learner utilization.")

    counts = np.stack([m.sample_counts[p] for p in range(4)], axis=0)
    mean_counts = np.nanmean(counts[:, -min(20, counts.shape[1]) :], axis=1)
    if np.nanmin(mean_counts) < 0.5 * np.nanmax(mean_counts):
        warnings.append("Recent per-head sample counts are imbalanced by more than 2x.")

    grad = m.grad_norm[np.isfinite(m.grad_norm)]
    if len(grad) >= 40:
        early = float(np.nanmean(grad[:20]))
        late = float(np.nanmean(grad[-20:]))
        if late > early * 1.5:
            warnings.append("Recent gradient norm is >1.5x the early-run mean; check for late instability.")

    lines = [
        f"# {run_dir.name} Health Summary",
        "",
        f"- Rows: {len(m.updates):,}",
        f"- Update range: {int(m.updates[0]):,} to {int(m.updates[-1]):,}",
        f"- Adjusted wall time: {float(m.elapsed_s[-1]) / 3600:.2f} hours",
        f"- Final average loss over last {recent_n} rows: {final_avg:.6g}",
        f"- Recent median update/sec: {recent_upd_per_sec:.3g}",
        f"- Recent median learner samples/sec: {recent_samples_per_sec:.3g}",
        f"- Recent median actor samples/sec: {recent_actor_rate:.3g}",
        f"- Recent median queue depth: {recent_queue_depth:.3g}",
        f"- Final cumulative replay ratio: {float(m.replay_cumulative[-1]):.3g}",
        "",
        "## Recent Head Sample Counts",
        "",
    ]
    for label, value in zip(HEAD_LABELS, mean_counts, strict=False):
        lines.append(f"- {label}: {value:.1f} samples/batch")
    lines.extend(["", "## Warnings", ""])
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- None.")
    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_path}")


def _format_update_window(updates: int) -> str:
    if updates >= 1_000_000 and updates % 1_000_000 == 0:
        return f"{updates // 1_000_000}m"
    if updates >= 1_000 and updates % 1_000 == 0:
        return f"{updates // 1_000}k"
    return f"{updates}u"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, type=Path,
                    help="Run directory (must contain metrics_learner.jsonl).")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output PNG path. Default: <run>/training_health.png")
    ap.add_argument("--recent-updates", type=int, default=50_000,
                    help="Window for training_health_recent_<N>.png.")
    ap.add_argument("--postprocess-resumes", action="store_true",
                    help=(
                        "Create a postprocessed metrics file that removes superseded "
                        "resume branches and offsets cumulative counters, then analyze it."
                    ))
    ap.add_argument("--postprocess-dir", type=Path, default=None,
                    help="Directory for cleaned metrics and plots. Default: <run>/postprocessed.")
    ap.add_argument("--resume-warmup-rows", type=int, default=20,
                    help="Rows to drop after each retained resume segment when postprocessing.")
    args = ap.parse_args()

    run_dir = args.run
    if args.postprocess_resumes:
        run_dir = postprocess_run_metrics(
            args.run,
            args.postprocess_dir or (args.run / "postprocessed"),
            warmup_rows=args.resume_warmup_rows,
        )

    out_path = args.out or (run_dir / "training_health.png")
    plot(run_dir, out_path)
    if args.recent_updates > 0:
        recent_out = run_dir / f"training_health_recent_{_format_update_window(args.recent_updates)}.png"
        plot(run_dir, recent_out, recent_updates=args.recent_updates)
    plot_systems_health(run_dir, run_dir / "systems_health.png")
    plot_q_diagnostics(run_dir, run_dir / "q_diagnostics.png")
    phase_out = run_dir / "bucket_diagnostics.png"
    try:
        plot_bucket_diagnostics(run_dir, phase_out)
        plot_bucket_shift(run_dir, run_dir / "bucket_shift.png")
    except FileNotFoundError:
        pass
    write_health_summary(run_dir, run_dir / "health_summary.md")


if __name__ == "__main__":
    main()
