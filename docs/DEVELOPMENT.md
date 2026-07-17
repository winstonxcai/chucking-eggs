# Development

This document is the engineering playbook for the research codebase. Scientific
protocol and result provenance live in [Reproducibility](REPRODUCIBILITY.md).

## Setup

```bash
uv sync --group dev
uv run maturin develop --release --manifest-path ml/src/guandan_rs/Cargo.toml
```

The Rust move generator is optional for correctness tests. Re-run the
`maturin` command after `uv sync` if the editable extension is no longer
visible.

## Validation matrix

```bash
uv run ruff check ml/src/guandan web/backend/app examples
uv run mypy
uv run pytest -q ml/tests
uv run pytest -q web/backend/tests

cd web/frontend
npm ci
npm run lint
npm run test:unit
npm run build
npm run e2e
npm run e2e:integration
```

The ML suite covers the engine, rule bots, encoders, replay, checkpoints, the
actor/learner runtime, evaluation handoff, and an end-to-end CPU smoke. Backend
and frontend tests are the compatibility gate for the secondary application
surface. Mongo and slow full-game tests remain explicit release checks because
they require external services or substantially more time.

## Local training and debugging

Use the CPU smoke for setup and runtime changes:

```bash
uv run python -m guandan.dart \
  --config ml/src/guandan/dart/configs/dart_cpu_smoke.yaml \
  --updates 5 --run-dir ml/runs/dev_smoke
```

For a longer Apple Silicon smoke:

```bash
uv run python -m guandan.dart \
  --config ml/src/guandan/dart/configs/dart_mps.yaml \
  --updates 100 --run-dir ml/runs/mps_smoke
```

`DART_TQDM=0` disables progress-bar redraws in collected logs.
`DART_ACTOR_PROFILE=1` and `DART_LEARNER_PROFILE=1` enable actor and learner
phase timing. Raw actor throughput can be measured with
`ml/scripts/util/profile_actor_throughput.py`.

Exact resume requires a full checkpoint. Weight-only checkpoints are intended
for evaluation and lightweight progress inspection.

## Change discipline

- Keep one behavioral change per commit where practical.
- Add or update a focused test for configuration, model, runtime, or protocol
  changes.
- Preserve checkpoint compatibility when changing serialized fields.
- Record config, seed, hardware, checkpoint, evaluation command, and result for
  new ML experiments.
- Do not commit raw Modal volumes, credentials, or generated multi-gigabyte
  training logs.
- Keep external rule-bot attribution and license scope in `NOTICE`.

## Rule-bot interface

Implement `Agent.act(env, player) -> Combo`, register the bot in
`ml/src/guandan/agents/__init__.py`, and add a legal-move smoke test. Vendored
competition submissions remain under `ml/src/guandan/agents/_vendor/` with
their original attribution.

## Web application

The application surface lives under `web/`. Its API and WebSocket contract are
documented in [API](API.md). Use the backend and frontend test commands above;
research changes should not require the deployment stack to run a CPU smoke.

## Release gate

Before publishing a checkpoint or result, run the validation matrix, verify the
merged configuration and artifact manifest, update the model card, and confirm
that README tables and plots agree with the raw evaluation JSONs. Release
bundles should include the source tag, checkpoint or artifact link, config,
evaluation output, and known limitations.
