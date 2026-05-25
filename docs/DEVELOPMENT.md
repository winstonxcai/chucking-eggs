# Development

## Setup

```bash
git clone https://github.com/PoohTheWinnie/chucking-eggs
cd chucking-eggs
uv sync --group dev
```

The pure-Python move generator works without Rust. For faster local development, build the native extension:

```bash
uv run maturin develop --release --manifest-path ml/src/guandan_rs/Cargo.toml
```

If you run `uv sync` again, rerun the `maturin develop` command because the editable environment can stop seeing `_guandan_rs`.

## Useful Commands

```bash
uv run pytest -q ml/tests
uv run pytest -q ml/tests/dart/runtime/test_train_smoke.py
uv run pytest -q web/backend/tests
```

Local DART CPU smoke:

```bash
uv run python -m guandan.dart \
  --config ml/src/guandan/dart/configs/dart_cpu_smoke.yaml \
  --updates 5 --run-dir ml/runs/dev_smoke
```

Apple Silicon longer smoke:

```bash
uv run python -m guandan.dart \
  --config ml/src/guandan/dart/configs/dart_mps.yaml \
  --updates 100 --run-dir ml/runs/mps_smoke
```

Modal dry run:

```bash
modal run ml/scripts/modal/train_dart_modal.py --dry-run \
  --config-path /root/ml/src/guandan/dart/configs/dart_l4.yaml
```

## Runtime Notes

- `DART_WEIGHT_DIR` can point weight polling at local disk while checkpoints remain in the run directory.
- `DART_TQDM=0` disables progress-bar redraws in cloud log collectors.
- `DART_ACTOR_PROFILE=1`, `true`, `yes`, or `on` enables actor profiling.
- Exact resume requires full checkpoints. Weight-only checkpoints are for evaluation and lightweight progress snapshots.
