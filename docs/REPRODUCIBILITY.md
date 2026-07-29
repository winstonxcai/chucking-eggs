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

## Local Docker training on A800-SXM4-80GB

The dedicated training image targets a Docker-enabled Linux x86-64 host with
one visible `NVIDIA A800-SXM4-80GB`. It installs the environment from the root
`pyproject.toml` and `uv.lock`, builds the native Rust move generator, and pins
PyTorch 2.10.0 with its CUDA 12.8 runtime. A host driver that advertises CUDA
13.0 is expected to run this older container runtime through NVIDIA's driver
backward compatibility; the container deliberately does not replace the
locked PyTorch build with a CUDA 13 build.

These commands are for a standalone Docker host. They do not submit Slurm jobs
and should not be run on a shared login node.

Prepare a writable run directory and preserve host ownership of artifacts:

```bash
mkdir -p ml/runs
export TRAIN_UID="$(id -u)"
export TRAIN_GID="$(id -g)"
export TRAIN_COMPOSE="docker-compose.training.yml"
```

Build the image and run the exact-hardware preflight:

```bash
docker compose -f "$TRAIN_COMPOSE" build
docker compose -f "$TRAIN_COMPOSE" run --rm \
  --entrypoint python dart-train \
  /workspace/ml/scripts/util/check_a800_training.py
```

The preflight requires one visible CUDA device at index 0, an
`A800-SXM4-80GB` with at least 80,000 MiB, compute capability 8.0 or newer,
BF16, driver 580.65.06 or newer, persistence mode, default compute mode, MIG
disabled, 128 CPUs, 128 GiB RAM, native `guandan_rs`, and a writable `/runs`.
It prints the complete detected report before returning success or failure.

Run the ML test suite inside the built image:

```bash
docker compose -f "$TRAIN_COMPOSE" run --rm \
  --entrypoint python dart-train -m pytest -q ml/tests
```

Run the portable actor–learner smoke on CPU:

```bash
docker compose -f "$TRAIN_COMPOSE" run --rm dart-train \
  --config /workspace/ml/src/guandan/dart/configs/dart_cpu_smoke.yaml \
  --device cpu --updates 5 --run-dir /runs/docker_cpu_smoke
```

Then run 100 learner updates through CUDA. The final checkpoint is always a
full checkpoint even though periodic checkpoints use the lighter weight-only
format:

```bash
docker compose -f "$TRAIN_COMPOSE" run --rm dart-train \
  --config /workspace/ml/src/guandan/dart/configs/dart_a800.yaml \
  --updates 100 --run-dir /runs/a800_cuda_smoke --seed 0
```

### Host calibration

The checked-in A800 default is 112 actors with 32 lanes per actor. Before a
long run, measure the four candidate shapes in separate run directories:

```bash
docker compose -f "$TRAIN_COMPOSE" run --rm dart-train \
  --config /workspace/ml/src/guandan/dart/configs/dart_a800.yaml \
  --n-actors 64 --actor-batch-lanes 32 --updates 2000 \
  --run-dir /runs/probe_64x32 --seed 0

docker compose -f "$TRAIN_COMPOSE" run --rm dart-train \
  --config /workspace/ml/src/guandan/dart/configs/dart_a800.yaml \
  --n-actors 96 --actor-batch-lanes 32 --updates 2000 \
  --run-dir /runs/probe_96x32 --seed 0

docker compose -f "$TRAIN_COMPOSE" run --rm dart-train \
  --config /workspace/ml/src/guandan/dart/configs/dart_a800.yaml \
  --n-actors 112 --actor-batch-lanes 32 --updates 2000 \
  --run-dir /runs/probe_112x32 --seed 0

docker compose -f "$TRAIN_COMPOSE" run --rm dart-train \
  --config /workspace/ml/src/guandan/dart/configs/dart_a800.yaml \
  --n-actors 112 --actor-batch-lanes 64 --updates 2000 \
  --run-dir /runs/probe_112x64 --seed 0
```

Compare the post-warmup `samples_per_sec`, `actor_rate_samp_per_sec`, replay
ratio, queue depth, and GPU allocation in each `metrics_learner.jsonl`. Use the
fastest shape with no actor failure or OOM, cumulative replay near 1.0, and at
least 20% host RAM free. Do not infer a problem from low VRAM use alone: this
small learner can remain limited by CPU self-play.

### Scratch, warm-start, and full resume

`--updates` is an **absolute total update target**, including updates loaded
from a checkpoint. It is not a number of additional updates.

Start a fresh 250,000-update run with the default Compose command:

```bash
DART_RUN_NAME=a800_scratch_250k DART_UPDATES=250000 \
  docker compose -f "$TRAIN_COMPOSE" up dart-train
```

Warm-start the packaged 1.25M weights and train to 1.50M total updates. This
performs 250,000 new updates, but optimizer, replay, and actor RNG state start
fresh because the packaged checkpoint is weight-only:

```bash
docker compose -f "$TRAIN_COMPOSE" run --rm --name dart-a800-warm dart-train \
  --config /workspace/ml/src/guandan/dart/configs/dart_a800.yaml \
  --resume /workspace/ml/results/release_1_25m/update_01250000.pt \
  --updates 1500000 --run-dir /runs/a800_warm_1500k --seed 0
```

Resume a locally produced full checkpoint in place and raise the absolute
target, for example from 250k to 500k:

```bash
docker compose -f "$TRAIN_COMPOSE" run --rm --name dart-a800-resume dart-train \
  --config /workspace/ml/src/guandan/dart/configs/dart_a800.yaml \
  --resume /runs/a800_scratch_250k/checkpoints/final.pt \
  --updates 500000 --run-dir /runs/a800_scratch_250k --seed 0
```

Use Ctrl-C for an attached run or allow the Compose service's ten-minute stop
grace period when stopping a detached run. SIGINT and SIGTERM both enter the
same clean shutdown path and write `checkpoints/final.pt` with model,
optimizer, replay, and RNG state.

Monitor the run from separate terminals:

```bash
nvidia-smi -l 2
docker stats
tail -f ml/runs/a800_scratch_250k/learner.log
tail -f ml/runs/a800_scratch_250k/metrics_learner.jsonl
```

Evaluate a final checkpoint with 1,000 paired fixed-deck games per opponent:

```bash
docker compose -f "$TRAIN_COMPOSE" run --rm \
  --entrypoint python dart-train \
  -m guandan.scripts.eval.eval_dart \
  --checkpoint /runs/a800_scratch_250k/checkpoints/final.pt \
  --opponent yaoji ez jidan strategic --games 1000 \
  --workers 32 --lanes 64 --device cpu --seed 0 \
  --out /runs/a800_scratch_250k/eval/final_1k.json
```

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
