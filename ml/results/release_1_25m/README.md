# DART 1.25M Release Evidence

This directory contains the reproducibility evidence for the README leaderboard and the packaged `ml/src/guandan/elos.json` defaults.

## Files

- `update_01250000.pt`: released DART checkpoint.
- `train_config.yaml`: training config used for the L4 run.
- `eval_5k_all_bots.json`: curated 5000-game paired fixed-deck DART-vs-bot results.
- `eval_5k_hard4.json`: refreshed 5000-game paired fixed-deck DART eval against Yaoji, EZ, Jidan, and Strategic.
- `wr_matrix_rulebots_5k.json`: refreshed 5000-game rule-bot round-robin matrix used as the Glicko base.
- `glicko_injections.json`: explicit DART matchup rows injected into the matrix.
- `glicko_results.json`: full Glicko result after loading the base matrix and injecting DART rows.
- `elos.json`: rounded release Elo table copied into the package runtime defaults.
- `manifest.json`: checksums, commands, seed, games, and source metadata.

The DART eval uses seed `0`; with paired fixed decks, games `0..2499` are played twice, once with DART on seats `(0, 2)` and once with DART on seats `(1, 3)`.
