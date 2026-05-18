"""Plot win rate by opponent from checkpoint-sweep eval JSON files.

This script is archival in name only: it was originally written for the M5/V5
baseline plots, but now accepts explicit eval directories instead of hard-coded
run names.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_OPPONENTS = [
    "random",
    "greedy",
    "heuristic",
    "xingdream",
    "strategic",
    "yaoji",
    "jidan",
    "lalala",
    "liuzha",
    "hulalala",
    "ez",
    "wjsd",
]

COLORS = {
    "random": "#aaaaaa",
    "greedy": "#1f77b4",
    "heuristic": "#2ca02c",
    "xingdream": "#9467bd",
    "strategic": "#ff7f0e",
    "yaoji": "#d62728",
    "jidan": "#8c564b",
    "lalala": "#e377c2",
    "liuzha": "#7f7f7f",
    "hulalala": "#bcbd22",
    "ez": "#17becf",
    "wjsd": "#aec7e8",
}


def _update_from_path(path: Path) -> int:
    return int(path.stem.replace("update_", ""))


def _load_eval_dirs(eval_dirs: list[Path]) -> dict[int, dict[str, float]]:
    data: dict[int, dict[str, float]] = {}
    for eval_dir in eval_dirs:
        if not eval_dir.exists():
            raise FileNotFoundError(f"eval dir does not exist: {eval_dir}")
        for path in sorted(eval_dir.glob("update_*.json")):
            payload = json.loads(path.read_text())
            results = payload.get("results", {})
            data[_update_from_path(path)] = {
                opponent: float(row["wr"]) * 100.0
                for opponent, row in results.items()
                if isinstance(row, dict) and "wr" in row
            }
    if not data:
        raise ValueError(f"no update_*.json files found under: {eval_dirs}")
    return data


def plot(eval_dirs: list[Path], out: Path, title: str, opponents: list[str]) -> None:
    data = _load_eval_dirs(eval_dirs)
    xs = sorted(data)
    xs_k = [x / 1000 for x in xs]

    fig, ax = plt.subplots(figsize=(13, 7))
    for opponent in opponents:
        valid = [
            (xk, data[x].get(opponent))
            for x, xk in zip(xs, xs_k)
            if data[x].get(opponent) is not None
        ]
        if not valid:
            continue
        vx, vy = zip(*valid)
        ax.plot(
            vx,
            vy,
            label=opponent,
            color=COLORS.get(opponent),
            linewidth=1.5,
            marker="o",
            markersize=2.5,
        )

    ax.set_xlabel("Updates (x1000)")
    ax.set_ylabel("Win rate (%)")
    ax.set_title(title)
    ax.set_ylim(0, 101)
    ax.set_xlim(0, max(xs_k) + 5)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", ncol=2, fontsize=8.5)
    plt.tight_layout()

    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=140)
    print(f"saved {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot WR by opponent over training.")
    parser.add_argument(
        "--eval-dir",
        type=Path,
        action="append",
        default=[Path("ml/runs/dart_v5_baseline_400k/eval")],
        help="Directory containing update_*.json eval files. Repeat to merge runs.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("ml/runs/dart_v5_baseline_400k/wr_plot.png"),
    )
    parser.add_argument(
        "--title",
        default="DART V5 baseline - WR per opponent",
    )
    parser.add_argument(
        "--opponents",
        default=",".join(DEFAULT_OPPONENTS),
        help="Comma-separated opponent order to plot.",
    )
    args = parser.parse_args()

    opponents = [name.strip() for name in args.opponents.split(",") if name.strip()]
    plot(args.eval_dir, args.out, args.title, opponents)


if __name__ == "__main__":
    main()
