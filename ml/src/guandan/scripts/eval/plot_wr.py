"""Plot win rate by opponent from checkpoint-sweep eval JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from guandan.agents import AGENT_REGISTRY

DEFAULT_OPPONENTS = list(AGENT_REGISTRY)


def _update_from_path(path: Path) -> int:
    return int(path.stem.replace("update_", ""))


def _load_eval_dir(eval_dir: Path) -> dict[int, dict[str, float]]:
    data: dict[int, dict[str, float]] = {}
    for path in sorted(eval_dir.glob("update_*.json")):
        payload = json.loads(path.read_text())
        results = payload.get("results", {})
        data[_update_from_path(path)] = {
            opponent: float(row["wr"]) * 100.0
            for opponent, row in results.items()
            if isinstance(row, dict) and "wr" in row
        }
    if not data:
        raise ValueError(f"no update_*.json files found under {eval_dir}")
    return data


def _format_update_k(update: int) -> str:
    if update % 1000 == 0:
        return f"{update // 1000}k"
    return f"{update / 1000:.1f}k"


def _parse_opponents(values: list[str] | None) -> list[str]:
    if not values:
        return DEFAULT_OPPONENTS
    opponents = []
    for value in values:
        opponents.extend(name.strip() for name in value.split(",") if name.strip())
    return opponents or DEFAULT_OPPONENTS


def _agent_color(name: str) -> str | None:
    cls = AGENT_REGISTRY.get(name)
    return None if cls is None else cls.color


def _validate_complete_results(
    data: dict[int, dict[str, float]],
    opponents: list[str],
) -> None:
    missing = {
        update: [opponent for opponent in opponents if opponent not in results]
        for update, results in sorted(data.items())
    }
    missing = {update: names for update, names in missing.items() if names}
    if not missing:
        return

    examples = [
        f"update_{update:08d}.json missing {', '.join(names)}"
        for update, names in list(missing.items())[:5]
    ]
    suffix = "" if len(missing) <= 5 else f"; ... {len(missing) - 5} more files"
    raise ValueError("incomplete eval results: " + "; ".join(examples) + suffix)


def plot_wr(
    eval_dir: Path,
    out: Path,
    title: str,
    opponents: list[str],
    resume_updates: list[int],
    y_min: float,
) -> None:
    if not eval_dir.exists():
        raise FileNotFoundError(f"eval dir does not exist: {eval_dir}")

    data = _load_eval_dir(eval_dir)
    _validate_complete_results(data, opponents)
    xs = sorted(data)
    xs_k = [x / 1000 for x in xs]

    fig, ax = plt.subplots(figsize=(13, 7))
    for opponent in opponents:
        valid = [
            (xk, data[x].get(opponent))
            for x, xk in zip(xs, xs_k, strict=False)
            if data[x].get(opponent) is not None
        ]
        if not valid:
            continue
        vx, vy = zip(*valid, strict=False)
        ax.plot(
            vx,
            vy,
            label=opponent,
            color=_agent_color(opponent),
            linewidth=1.5,
            marker="o",
            markersize=2.5,
        )

    for update in resume_updates:
        x = update / 1000
        ax.axvline(x, color="black", linestyle=":", linewidth=0.8, alpha=0.4)
        ax.text(
            x + 0.5,
            y_min + 3,
            _format_update_k(update),
            fontsize=7.5,
            color="black",
            alpha=0.55,
        )

    ax.set_xlabel("Updates (x1000)")
    ax.set_ylabel("Win rate (%)")
    ax.set_title(title)
    ax.set_ylim(y_min, 101)
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
        default=Path("ml/runs/dart_v5_baseline_400k/eval"),
        help="Directory containing existing update_*.json eval files.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("ml/runs/dart_v5_baseline_400k/wr_plot.png"),
    )
    parser.add_argument("--title", default="DART - WR per opponent over training")
    parser.add_argument(
        "--opponents",
        nargs="*",
        default=None,
        help=(
            "Opponent names to plot. Defaults to all registered agents. "
            "Accepts either space-separated names or comma-separated groups."
        ),
    )
    parser.add_argument(
        "--resume-updates",
        type=int,
        nargs="*",
        default=[],
        help="Update counts where training resumed, e.g. --resume-updates 20000 60000.",
    )
    parser.add_argument("--y-min", type=float, default=25.0)
    args = parser.parse_args()

    plot_wr(
        eval_dir=args.eval_dir,
        out=args.out,
        title=args.title,
        opponents=_parse_opponents(args.opponents),
        resume_updates=args.resume_updates,
        y_min=args.y_min,
    )


if __name__ == "__main__":
    main()
