# Evaluation Protocol

DART evaluation uses paired fixed decks so each matchup plays both
partnerships on the same shuffled deals. A reported win rate is the fraction of
games won by the evaluated agent's team; an even game count makes the two deck
parities directly comparable.

The release leaderboard evidence lives in `ml/results/release_1_25m/`. That
directory includes the released checkpoint, raw 5000-game DART eval JSON, the
base rule-bot matrix, the explicit Glicko injections, and the derived ratings.

## Single Opponent

```bash
uv run guandan-eval-dart \
  --checkpoint ml/results/release_1_25m/update_01250000.pt \
  --opponent strategic \
  --games 1000 \
  --out results.json
```

Use an even `--games` value. For headline numbers, run enough games to report
a confidence interval. The four-cell ablation uses 1,000 games per opponent;
the release tables use 5,000 games or more.

For a win rate \(\hat p\) over \(n\) games, the reported 95% interval is the
normal approximation \(\hat p \pm 1.96\sqrt{\hat p(1-\hat p)/n}\). These
intervals quantify evaluation-game uncertainty only; they do not measure
variation across independent training seeds.

## Leaderboard

```bash
uv run guandan-wr-matrix --games 200
```

The output `elos.json` can be converted into the Glicko-2 table shown in the README.
For the released table, see `ml/results/release_1_25m/glicko_results.json` and
`ml/results/release_1_25m/glicko_injections.json`.

## Custom Opponents

Add a bot to `AGENT_REGISTRY` first, then pass its registry key as `--opponent`.
See [Reproducibility](REPRODUCIBILITY.md) for the required bot interface and
validation commands.

## Checkpoint Choice

- `final.pt` is best for resume because it is always a full checkpoint on clean shutdown.
- `update_*.pt` may be weight-only when `checkpoint_save_type: weight`; use those for evaluation, not exact training resume.
