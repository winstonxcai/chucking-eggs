# Evaluation

DART evaluation uses paired fixed decks so each matchup plays both partnerships on the same shuffled deals.

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

Use an even `--games` value. For headline numbers, run enough games to report a confidence interval; the README tables use 5000-game or larger evals.

## Leaderboard

```bash
uv run guandan-wr-matrix --games 200
```

The output `elos.json` can be converted into the Glicko-2 table shown in the README.
For the released table, see `ml/results/release_1_25m/glicko_results.json` and
`ml/results/release_1_25m/glicko_injections.json`.

## Custom Opponents

Add a bot to `AGENT_REGISTRY` first, then pass its registry key as `--opponent`. See `docs/CONTRIBUTING.md` for the required bot interface.

## Checkpoint Choice

- `final.pt` is best for resume because it is always a full checkpoint on clean shutdown.
- `update_*.pt` may be weight-only when `checkpoint_save_type: weight`; use those for evaluation, not exact training resume.
