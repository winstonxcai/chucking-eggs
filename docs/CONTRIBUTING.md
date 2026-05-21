# Contributing

This project has two main surfaces: the Guan Dan game/agent library under `ml/src/guandan` and the web app under `web/`.

## Before a PR

- Run the focused tests for the area you touched.
- Keep generated training artifacts, checkpoints, and run logs out of git.
- Prefer small PRs with one behavioral change at a time.
- For ML changes, include the config, seed, checkpoint path, and eval command used to validate the result.

## Adding a Rule Bot

1. Implement `Agent.act(env, player)` in `ml/src/guandan/agents/`.
2. Add the bot to `AGENT_REGISTRY` in `ml/src/guandan/agents/__init__.py`.
3. Add a smoke test that the bot returns legal moves over a short rollout.
4. If the bot comes from an external competition submission, keep vendored code under `ml/src/guandan/agents/_vendor/`.

## Adding DART Training Changes

- Update `docs/CONFIG.md` when adding or renaming config fields.
- Add or update a smoke config if the change affects actor/learner throughput.
- Preserve checkpoint compatibility in `DartBot.load()` when practical, and log a warning for deprecated checkpoint keys.

## Reporting Results

Use paired fixed-deck evaluation for win-rate claims:

```bash
uv run guandan-eval-dart \
  --checkpoint ml/runs/my_run/checkpoints/update_00050000.pt \
  --opponent strategic --games 1000 --out results.json
```

For leaderboard changes, include the `guandan-wr-matrix` command and the resulting `elos.json`.
