"""Plot WR per opponent over training from dart_l4/eval/*.json."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

OPPONENTS = [
    "random", "greedy", "heuristic", "xingdream",
    "strategic", "yaoji", "jidan",
    "lalala", "liuzha", "hulalala", "ez", "wjsd",
]

COLORS = {
    "random":    "#aaaaaa",
    "greedy":    "#1f77b4",
    "heuristic": "#2ca02c",
    "xingdream": "#9467bd",
    "strategic": "#ff7f0e",
    "yaoji":     "#d62728",
    "jidan":     "#8c564b",
    "lalala":    "#e377c2",
    "liuzha":    "#7f7f7f",
    "hulalala":  "#bcbd22",
    "ez":        "#17becf",
    "wjsd":      "#aec7e8",
}

eval_dir = Path("ml/runs/dart_l4/eval")
data: dict[int, dict[str, float]] = {}

for f in sorted(eval_dir.glob("update_*.json")):
    d = json.loads(f.read_text())
    n = int(f.stem.replace("update_", ""))
    r = d["results"]
    data[n] = {opp: r[opp]["wr"] * 100 for opp in OPPONENTS if opp in r}

xs = sorted(data.keys())
xs_k = [x / 1000 for x in xs]

fig, ax = plt.subplots(figsize=(13, 7))

for opp in OPPONENTS:
    ys = [data[x].get(opp) for x in xs]
    valid = [(xk, y) for xk, y in zip(xs_k, ys) if y is not None]
    if not valid:
        continue
    vx, vy = zip(*valid)
    ax.plot(vx, vy, label=opp, color=COLORS[opp], linewidth=1.5, marker="o", markersize=2.5)

for x_res, label in [(20, "20k"), (60, "60k"), (70, "70k"), (125, "125k"), (135, "135k"), (200, "200k")]:
    ax.axvline(x_res, color="black", linestyle=":", linewidth=0.8, alpha=0.4)
    ax.text(x_res + 0.5, 28, label, fontsize=7.5, color="black", alpha=0.5)

ax.set_xlabel("Updates (×1000)")
ax.set_ylabel("Win rate (%)")
ax.set_title("DART — WR per opponent over training (1000 games / opp / ckpt)")
ax.set_ylim(25, 101)
ax.set_xlim(0, max(xs_k) + 5)
ax.grid(True, alpha=0.3)
ax.legend(loc="lower right", ncol=2, fontsize=8.5)
plt.tight_layout()

out = Path("ml/runs/dart_l4/wr_over_training.png")
plt.savefig(out, dpi=140)
print(f"saved {out}")
