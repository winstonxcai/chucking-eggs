"""Plot M5 baseline WR per opponent from 0 to 200k updates.

0-70k data is hand-curated from LOGBOOK; 72.5k+ is loaded from
ml/runs/guanzero_v5_baseline_150k/eval/*.json,
ml/runs/guanzero_v5_baseline_135k/eval/*.json, and
ml/runs/dart_v5_baseline_200k/eval/*.json.
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt

OPPONENTS = ["random", "greedy", "heuristic", "xingdream", "strategic", "yaoji", "jidan"]

# LOGBOOK rows 2.5k → 70k (combined WR).
EARLY = {
    2500:  [98.7, 96.8, 81.3, 79.7, 59.5, 34.6, 35.5],
    5000:  [98.9, 98.2, 86.2, 84.9, 66.3, 39.6, 45.0],
    7500:  [99.5, 98.2, 88.1, 85.5, 63.7, 45.1, 44.3],
    10000: [99.4, 98.1, 85.4, 85.5, 62.4, 40.4, 45.2],
    12500: [99.2, 98.6, 87.5, 87.6, 67.1, 39.9, 45.5],
    15000: [99.5, 98.0, 88.5, 89.3, 68.3, 43.0, 47.5],
    17500: [99.3, 98.6, 89.8, 88.8, 66.4, 44.5, 46.9],
    20000: [99.4, 97.8, 88.0, 88.9, 65.8, 40.5, 46.9],
    22500: [99.4, 98.2, 86.9, 89.8, 61.3, 40.0, 48.5],
    25000: [99.7, 98.7, 89.6, 92.1, 67.4, 42.3, 52.2],
    27500: [99.4, 98.4, 88.9, 89.7, 65.1, 41.2, 53.2],
    30000: [99.5, 97.9, 85.0, 89.7, 62.0, 39.4, 49.8],
    32500: [99.6, 98.8, 89.7, 91.6, 69.0, 42.7, 50.5],
    35000: [99.3, 98.4, 90.1, 89.4, 69.2, 46.4, 52.6],
    37500: [99.6, 98.9, 85.9, 89.3, 68.5, 43.9, 52.8],
    40000: [99.6, 98.4, 87.0, 89.1, 69.8, 45.7, 53.0],
    42500: [99.2, 98.1, 88.8, 89.9, 66.7, 45.7, 51.1],
    45000: [99.1, 98.2, 86.9, 90.6, 66.3, 47.4, 53.4],
    47500: [99.6, 98.1, 86.1, 90.5, 67.6, 44.9, 53.3],
    50000: [99.8, 97.9, 82.6, 89.8, 62.4, 41.5, 50.9],
    52500: [99.5, 98.1, 85.0, 91.8, 65.4, 42.6, 54.3],
    55000: [99.1, 97.0, 81.5, 88.9, 61.7, 43.7, 50.7],
    57500: [99.2, 98.4, 87.8, 90.5, 66.0, 43.5, 52.0],
    60000: [99.6, 97.8, 84.2, 89.0, 62.0, 40.2, 48.8],
    62500: [99.5, 99.0, 86.8, 91.1, 68.1, 47.3, 55.3],
    65000: [99.8, 98.6, 88.3, 91.0, 66.3, 48.0, 56.5],
    67500: [99.8, 98.7, 89.4, 91.5, 68.7, 50.6, 54.8],
    70000: [99.5, 98.9, 87.7, 92.7, 68.5, 46.8, 55.6],
}

EVAL_DIRS = [
    Path("ml/runs/guanzero_v5_baseline_150k/eval"),
    Path("ml/runs/guanzero_v5_baseline_135k/eval"),
    Path("ml/runs/dart_v5_baseline_200k/eval"),
]
data: dict[int, list[float]] = dict(EARLY)
for eval_dir in EVAL_DIRS:
    for f in sorted(eval_dir.glob("update_*.json")):
        d = json.loads(f.read_text())
        n = int(os.path.basename(d["checkpoint"]).replace("update_", "").replace(".pt", ""))
        r = d["results"]
        row = [r[o]["wr"] * 100 for o in OPPONENTS]
        if n not in data:
            data[n] = row

xs = sorted(data.keys())
ys_per_opp = {opp: [data[x][i] for x in xs] for i, opp in enumerate(OPPONENTS)}

fig, ax = plt.subplots(figsize=(11, 6.5))
colors = {
    "random":    "#888888",
    "greedy":    "#1f77b4",
    "heuristic": "#2ca02c",
    "xingdream": "#9467bd",
    "strategic": "#ff7f0e",
    "yaoji":     "#d62728",
    "jidan":     "#8c564b",
}
for opp in OPPONENTS:
    xs_k = [x / 1000 for x in xs]
    ax.plot(xs_k, ys_per_opp[opp], label=opp, color=colors[opp], linewidth=1.6, marker="o", markersize=3)

for x_res, label in [(20, "20k"), (60, "60k"), (70, "70k"), (125, "125k"), (135, "135k")]:
    ax.axvline(x_res, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.text(x_res, 32, f" {label} resume", fontsize=8, color="black", alpha=0.6)
ax.set_xlabel("Updates (×1000)")
ax.set_ylabel("Win rate (%)")
ax.set_title("M5 baseline — WR per opponent (1000 games / opp / ckpt)")
ax.set_ylim(30, 101)
ax.set_xlim(0, 205)
ax.grid(True, alpha=0.3)
ax.legend(loc="lower right", ncol=2, fontsize=9)
plt.tight_layout()

out = Path("ml/runs/dart_v5_baseline_200k/wr_plot_200k.png")
plt.savefig(out, dpi=130)
print(f"saved {out}")
