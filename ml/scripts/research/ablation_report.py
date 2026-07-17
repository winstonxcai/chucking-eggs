"""Build the curated report for the four-cell L4 ablation study.

The input directory is expected to contain one subdirectory per cell with the
final evaluation, periodic checkpoint evaluations, the merged config, and,
when available, the raw learner metrics JSONL. The output is intentionally
small enough to commit: it contains evaluation JSONs, metric rows aligned to
evaluations, a summary table, a provenance manifest, and plots.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

OPPONENTS = ("yaoji", "ez", "jidan", "strategic")
CELLS = {
    "dart_visible": {
        "model_type": "Dart",
        "is_partner_visible": True,
        "parameter_count": 4_723_716,
    },
    "dart_hidden": {
        "model_type": "Dart",
        "is_partner_visible": False,
        "parameter_count": 4_723_716,
    },
    "guanzero_visible": {
        "model_type": "GuanZero",
        "is_partner_visible": True,
        "parameter_count": 28_393_476,
    },
    "guanzero_hidden": {
        "model_type": "GuanZero",
        "is_partner_visible": False,
        "parameter_count": 28_393_476,
    },
}
UPDATE_RE = re.compile(r"update_(\d+)\.json$")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _macro(eval_data: dict[str, Any]) -> float:
    return sum(float(eval_data["results"][opp]["wr"]) for opp in OPPONENTS) / len(OPPONENTS)


def _metric_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line, strict=False)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def _metric_at_or_before(rows: list[dict[str, Any]], updates: int) -> dict[str, Any]:
    candidates = [row for row in rows if int(row.get("updates", 0)) <= updates]
    if not candidates:
        return {}
    return max(candidates, key=lambda row: int(row.get("updates", 0)))


def _eval_files(cell_dir: Path) -> list[tuple[int, Path]]:
    files: list[tuple[int, Path]] = []
    for path in cell_dir.glob("update_*.json"):
        match = UPDATE_RE.search(path.name)
        if match:
            files.append((int(match.group(1)), path))
    return sorted(files)


def _trajectory(cell_dir: Path) -> list[dict[str, Any]]:
    rows = _metric_rows(cell_dir / "metrics_learner.jsonl")
    trajectory: list[dict[str, Any]] = []
    for updates, eval_path in _eval_files(cell_dir):
        eval_data = _read_json(eval_path)
        metric = _metric_at_or_before(rows, updates)
        trajectory.append(
            {
                "updates": updates,
                "wall_clock_s": metric.get("wall_clock_s"),
                "learner_samples_total": metric.get("learner_samples_total"),
                "samples_per_sec": metric.get("samples_per_sec"),
                "macro_win_rate": round(_macro(eval_data), 6),
                "eval_file": eval_path.name,
            }
        )

    final_eval = _read_json(cell_dir / "final.json")
    final_metric = rows[-1] if rows else {}
    trajectory.append(
        {
            "updates": int(final_metric.get("updates", 0)),
            "wall_clock_s": final_metric.get("wall_clock_s"),
            "learner_samples_total": final_metric.get("learner_samples_total"),
            "samples_per_sec": final_metric.get("samples_per_sec"),
            "macro_win_rate": round(_macro(final_eval), 6),
            "eval_file": "final.json",
        }
    )
    return trajectory


def _copy_source_artifacts(source: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "final.json"):
        shutil.copy2(source / name, output / name)
    for _updates, path in _eval_files(source):
        shutil.copy2(path, output / path.name)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _plot(summary: dict[str, Any], plot_dir: Path) -> None:
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)
    colors = {
        "dart_visible": "#1769aa",
        "dart_hidden": "#5b9bd5",
        "guanzero_visible": "#c44e52",
        "guanzero_hidden": "#dd8452",
    }
    labels = {
        "dart_visible": "DART · visible",
        "dart_hidden": "DART · hidden",
        "guanzero_visible": "GuanZero · visible",
        "guanzero_hidden": "GuanZero · hidden",
    }

    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=180)
    for cell, data in summary["runs"].items():
        points = data["trajectory"]
        real_x = [float(p["wall_clock_s"]) / 3600 for p in points if p["wall_clock_s"] is not None]
        real_y = [float(p["macro_win_rate"]) * 100 for p in points if p["wall_clock_s"] is not None]
        ax.plot(real_x, real_y, marker="o", linewidth=2, markersize=4, label=labels[cell], color=colors[cell])
        ax.scatter([0], [0], marker="x", s=42, linewidths=1.5, color=colors[cell], alpha=0.65)
    ax.set_xlim(left=0)
    ax.set_xlabel("Wall-clock time (hours)")
    ax.set_ylabel("Macro win rate (%)")
    ax.set_title("Four-cell ablation: performance over wall-clock time")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    ax.annotate("visual origin; not evaluated", (0, 0), xytext=(10, 8), textcoords="offset points", fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "ablation_l4_winrate_wallclock.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=180)
    for cell, data in summary["runs"].items():
        points = [p for p in data["trajectory"] if p["learner_samples_total"] is not None]
        x = [float(p["learner_samples_total"]) / 1e6 for p in points]
        y = [float(p["macro_win_rate"]) * 100 for p in points]
        ax.plot(x, y, marker="o", linewidth=2, markersize=4, label=labels[cell], color=colors[cell])
    ax.set_xlabel("Effective learner samples (millions)")
    ax.set_ylabel("Macro win rate (%)")
    ax.set_title("Four-cell ablation: performance over effective samples")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(plot_dir / "ablation_l4_winrate_effective_samples.png", bbox_inches="tight")
    plt.close(fig)

    final = {
        cell: data["final_macro_win_rate"] * 100
        for cell, data in summary["runs"].items()
    }
    contrasts = {
        "DART visibility effect": final["dart_visible"] - final["dart_hidden"],
        "GuanZero visibility effect": final["guanzero_visible"] - final["guanzero_hidden"],
        "DART advantage, visible": final["dart_visible"] - final["guanzero_visible"],
        "DART advantage, hidden": final["dart_hidden"] - final["guanzero_hidden"],
        "Interaction": (final["dart_visible"] - final["dart_hidden"])
        - (final["guanzero_visible"] - final["guanzero_hidden"]),
    }
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=180)
    names = list(contrasts)
    values = list(contrasts.values())
    ax.bar(names, values, color=["#5b9bd5", "#dd8452", "#1769aa", "#c44e52", "#8172b2"])
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_ylabel("Difference in macro win rate (percentage points)")
    ax.set_title("Final-time factorial contrasts")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_dir / "ablation_l4_factorial_contrasts.png", bbox_inches="tight")
    plt.close(fig)


def build(source_root: Path, output_root: Path, plot_dir: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "study": "ablation_l4_10h",
        "training_budget_s": 36_000,
        "seed": 0,
        "opponents": list(OPPONENTS),
        "source_commit": None,
        "provenance_note": (
            "Modal runs were launched from the working tree before commit 7638c6e; "
            "7638c6e records the secured equivalent ablation implementation."
        ),
        "runs": {},
    }
    for cell, metadata in CELLS.items():
        source = source_root / cell
        output = output_root / cell
        _copy_source_artifacts(source, output)
        final_eval = _read_json(source / "final.json")
        trajectory = _trajectory(source)
        final_metric = trajectory[-1]
        summary["runs"][cell] = {
            **metadata,
            "seed": 0,
            "final_macro_win_rate": round(_macro(final_eval), 6),
            "final_results": final_eval["results"],
            "learner_updates": final_metric["updates"],
            "wall_clock_s": final_metric["wall_clock_s"],
            "learner_samples_total": final_metric["learner_samples_total"],
            "samples_per_sec": final_metric["samples_per_sec"],
            "average_samples_per_sec": (
                float(final_metric["learner_samples_total"])
                / float(final_metric["wall_clock_s"])
                if final_metric["learner_samples_total"] is not None
                and final_metric["wall_clock_s"]
                else None
            ),
            "trajectory": trajectory,
        }
        _write_json(output / "metrics_summary.json", {"cell": cell, "trajectory": trajectory})

    _write_json(output_root / "summary.json", summary)
    with (output_root / "summary.csv").open("w", newline="") as handle:
        fields = ["cell", "model_type", "is_partner_visible", "parameter_count", "final_macro_win_rate", "learner_updates", "wall_clock_s", "learner_samples_total", "average_samples_per_sec"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for cell, data in summary["runs"].items():
            writer.writerow({"cell": cell, **{field: data[field] for field in fields[1:]}})

    _plot(summary, plot_dir)

    manifest: dict[str, Any] = {
        "study": summary["study"],
        "artifact_commit": "7638c6e",
        "source_commit": summary["source_commit"],
        "provenance_note": summary["provenance_note"],
        "top_level_artifacts": {},
        "runs": {},
    }
    for name in ("README.md", "summary.json", "summary.csv"):
        manifest["top_level_artifacts"][name] = _sha256(output_root / name)
    for name in (
        "ablation_l4_winrate_wallclock.png",
        "ablation_l4_winrate_effective_samples.png",
        "ablation_l4_factorial_contrasts.png",
    ):
        plot_path = plot_dir / name
        manifest["top_level_artifacts"][str(plot_path)] = _sha256(plot_path)
    for cell, data in summary["runs"].items():
        run_dir = output_root / cell
        artifacts = {
            str(path.relative_to(output_root)): _sha256(path)
            for path in sorted(run_dir.glob("*.json"))
        }
        manifest["runs"][cell] = {
            "model_type": data["model_type"],
            "is_partner_visible": data["is_partner_visible"],
            "seed": data["seed"],
            "hardware": "Modal L4 + 32 vCPU actors",
            "training_budget_s": summary["training_budget_s"],
            "evaluation": {"opponents": list(OPPONENTS), "games_per_opponent": 1000},
            "learner_updates": data["learner_updates"],
            "wall_clock_s": data["wall_clock_s"],
            "learner_samples_total": data["learner_samples_total"],
            "average_samples_per_sec": data["average_samples_per_sec"],
            "artifacts": artifacts,
        }
    _write_json(output_root / "manifest.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--plot-dir", type=Path, required=True)
    args = parser.parse_args()
    build(args.source_root, args.output_root, args.plot_dir)


if __name__ == "__main__":
    main()
