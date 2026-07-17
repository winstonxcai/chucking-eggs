# Reproducibility

This repository separates three claims: the game engine can be installed and
tested locally, the training pipeline can be smoke-tested without a GPU, and
the reported L4 studies can be reconstructed from pinned configurations and
curated artifacts.

## Environment

The Python environment is defined by `pyproject.toml` and `uv.lock`. Use the
checked-in lockfile for local work. The native Rust move generator is optional
for tests and local play; large training and evaluation runs should build it
with `maturin` for the intended throughput.

Exact bitwise CUDA reproducibility is not promised. Checkpoints preserve the
learner and actor RNG state needed for process-level resume consistency, but
CUDA/cuDNN kernels are not forced into deterministic mode because doing so can
reduce throughput or reject supported kernels.

## Minimal validation

```bash
uv sync --group dev
uv run pytest -q ml/tests
uv run python -m guandan.dart \
  --config ml/src/guandan/dart/configs/dart_cpu_smoke.yaml \
  --updates 5 --run-dir ml/runs/cpu_smoke
```

For the native move generator:

```bash
uv run maturin develop --release --manifest-path ml/src/guandan_rs/Cargo.toml
```

## Engineering validation

The repository's validation matrix is intentionally split by surface:

```bash
uv run ruff check ml/src/guandan web/backend/app examples
uv run mypy
uv run pytest -q ml/tests
uv run pytest -q web/backend/tests
```

For the secondary web application:

```bash
cd web/frontend
npm ci
npm run lint
npm run test:unit
npm run build
npm run e2e
npm run e2e:integration
```

The ML tests cover the engine, rule bots, encoders, replay, checkpointing,
actor–learner lifecycle, evaluation handoff, and CPU smoke. Backend and
frontend checks protect the application surface without being prerequisites for
the research pipeline. MongoDB- and slow-game-dependent checks are explicit
release checks rather than hidden setup requirements.

For collected logs, set `DART_TQDM=0` to disable progress-bar redraws. Set
`DART_ACTOR_PROFILE=1` or `DART_LEARNER_PROFILE=1` to enable phase timing. The
actor throughput benchmark is available through
`ml/scripts/util/profile_actor_throughput.py`.

## Four-cell ablation

The controlled study uses one shared L4 baseline and four minimal overrides:

- [shared baseline](../ml/src/guandan/dart/configs/ablation_l4_common.yaml);
- [DART, partner visible](../ml/src/guandan/dart/configs/ablation_l4_dart_partner_visible_10h.yaml);
- [DART, partner hidden](../ml/src/guandan/dart/configs/ablation_l4_dart_partner_hidden_10h.yaml);
- [GuanZero, partner visible](../ml/src/guandan/dart/configs/ablation_l4_guanzero_partner_visible_10h.yaml);
- [GuanZero, partner hidden](../ml/src/guandan/dart/configs/ablation_l4_guanzero_partner_hidden_10h.yaml).

The common baseline fixes the Modal L4 runtime, actor/queue behavior, replay,
optimizer, exploration, evaluation opponents, and ten-hour duration. The only
experimental fields in the overrides are algorithm type and partner
visibility.

`batch_size: 4096` means total learner samples per update in this study. DART
uses one shared 4,096-sample batch. GuanZero samples 1,024 examples for each
of four seats. Legacy GuanZero configs explicitly use
`batch_size_semantics: per_seat` and are not silently reinterpreted.

YAML files may inherit a relative `base_config`. Bases are recursively loaded,
nested mappings are merged, and child values take precedence. The fully merged
configuration is serialized into `config.json` and checkpoints. Older field
names for forced-pass replay and learner-team mixing are migrated on load so
legacy MPS configs and checkpoints remain interpretable.

## Modal reproduction

The Modal launcher accepts workspace placeholders so the public instructions
do not expose personal account names. Run one command per workspace, using the
same seed and the corresponding configuration:

```bash
MODAL_PROFILE=WORKSPACE .venv/bin/modal run --detach \
  ml/scripts/modal/train_dart_modal.py::train_remote \
  --run-name RUN_NAME \
  --seed 0 \
  --config-path /root/ml/src/guandan/dart/configs/CONFIG.yaml \
  --device cuda
```

Do not pass `--updates` for the production ablation. The configuration's
duration limit controls stopping. The Modal timeout provides startup and
finalization headroom beyond the ten-hour training budget.

Posthoc evaluation should use the final checkpoint in the Modal volume. The
four opponents are `yaoji`, `ez`, `jidan`, and `strategic`, with 1,000 paired
fixed-deck games per opponent. The final checkpoint is evaluated separately
from periodic checkpoint evaluation so the training and analysis stages are
unambiguous.

## Artifacts

The curated [ablation artifact directory](../ml/results/ablation_l4_10h/)
contains:

- fully merged configurations;
- final and periodic evaluation JSONs;
- metric summaries aligned to evaluation update counts;
- `summary.json` and `summary.csv`;
- `manifest.json` with provenance and SHA-256 hashes.

Raw ten-hour learner JSONL files and full Modal volumes are intentionally not
committed. The report builder can consume those raw run directories and
recreate the curated artifact directory and plots:

```bash
uv run python ml/scripts/research/ablation_report.py \
  --source-root /path/to/raw/ablation_l4_10h \
  --output-root ml/results/ablation_l4_10h \
  --plot-dir docs/assets
```

The analysis joins each checkpoint evaluation to the nearest learner-metrics
row at or before the checkpoint update. It reports both wall-clock time and
effective learner samples so algorithm comparisons cannot be reduced to raw
update count when batch semantics differ.

## Checkpoints and release evidence

Periodic `update_*.pt` files may be weight-only. A clean shutdown writes
`final.pt` as a full resume checkpoint containing model, optimizer, replay, and
RNG state. The release evidence for the larger DART run lives under
`ml/results/release_1_25m/`; its manifest records the checkpoint and evaluation
hashes.

When publishing a new result, record the source commit, merged config, seed,
hardware, budget, evaluation command, raw JSON, derived table, and artifact
hashes. If the run predates a commit, record that fact explicitly rather than
assigning a later commit retroactively.

## Development and release discipline

Keep one behavioral change per commit where practical and add a focused test for
configuration, model, runtime, or protocol changes. Preserve checkpoint
compatibility when changing serialized fields. Do not commit raw Modal volumes,
credentials, or generated multi-gigabyte logs; retain compact summaries and
manifests instead.

Exact resume requires a full checkpoint. Weight-only checkpoints are suitable
for evaluation and lightweight progress inspection, but cannot restore the
optimizer, replay, or RNG state. Before publishing a result, verify the merged
configuration, artifact manifest, SHA-256 values, README tables, and plots. A
release bundle should identify the source commit, checkpoint or artifact link,
configuration, evaluation output, and known limitations.

The application surface lives under `web/`. Its backend API and WebSocket
contract are documented in the [backend README](../web/backend/README.md).

### Adding a rule bot

Implement `Agent.act(env, player) -> Combo`, register the bot in
`ml/src/guandan/agents/__init__.py`, and add a legal-move smoke test. Vendored
competition submissions remain under
`ml/src/guandan/agents/_vendor/` with their original attribution.
