# Four-cell L4 ablation artifact

This directory contains the curated evidence for the ten-hour DART/GuanZero
comparison. It intentionally excludes full checkpoints and raw learner JSONL
logs.

## Contents

- `summary.json`: final results and evaluation-aligned learning trajectories;
- `summary.csv`: compact table for downstream analysis;
- `manifest.json`: study provenance, protocol, and SHA-256 hashes;
- one directory per cell with the fully merged config, final evaluation, and
  periodic evaluation JSONs;
- `metrics_summary.json` per cell with wall-clock and effective-sample rows
  joined to the corresponding evaluations.

The runs used seed 0, Modal L4 hardware, 32 CPU actors, a ten-hour duration
budget, and 1,000 paired fixed-deck games per opponent for Yaoji, EZ, Jidan,
and Strategic. DART and GuanZero both consumed 4,096 effective learner
samples per update; GuanZero sampled 1,024 per seat.

The runs were launched from the working tree before commit `7638c6e`. That
commit records the secured equivalent ablation implementation and is listed as
the artifact commit in `manifest.json`; it is not asserted to be the original
training source commit.

Regenerate the directory and plots from raw Modal artifacts with:

```bash
uv run python ml/scripts/research/ablation_report.py \
  --source-root /path/to/raw/ablation_l4_10h \
  --output-root ml/results/ablation_l4_10h \
  --plot-dir docs/assets
```
