# Training Metrics Summary

This summary is derived from the postprocessed Modal learner log for the released `update_01250000.pt` checkpoint. The raw log is append-only across multiple resumes, so release plots and scalar summaries should use the cleaned metrics described in `metrics_postprocess_summary.json`.

## Release Checkpoint

- Checkpoint: `update_01250000.pt`
- Target update: `1,250,000`
- Modal source: `pvguan-runs:dart/dart_v5_baseline_400k/metrics_learner.jsonl`
- Cleaned rows: `114,840` from `119,004` raw rows
- Adjusted logged wall time through target: `55.80` hours since the first retained 200k-row log
- Adjusted fresh samples through target: `3,435,606,016` since the first retained 200k-row log
- Replay cumulative at target: `1.25`

## Recommended Scalars

Use the last 100 learner metric rows ending at update `1,250,000`:

- Mean learner loss: `0.2483`
- Median grad norm: `0.5483`
- Median learner update rate: `4.91` updates/sec
- Median learner sample consumption: `20116` samples/sec
- Median actor sample production: `16046` samples/sec
- Median queue depth: `64`
- Median replay interval: `1.25`

The final 1.35M rows are useful for training-health plots, but release checkpoint claims should be anchored at update `1,250,000`.
