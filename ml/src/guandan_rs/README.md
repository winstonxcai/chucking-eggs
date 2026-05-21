# guandan_rs

`guandan_rs` is the optional native move-generation backend for Guan Dan.

The Python package works without compiling Rust by falling back to a pure-Python implementation. The native `_guandan_rs` extension is much faster and is recommended for training.

## Build

```bash
uv run maturin develop --release --manifest-path ml/src/guandan_rs/Cargo.toml
```

The Modal training image builds the wheel during image creation. Local users should rerun `maturin develop` after recreating or resyncing the Python environment.

## Public API

- `generate_all_leads(hand, level_rank)`
- `generate_responses(hand, level_rank, trick)`
- `dedup_strategic(combos)`
- `select_legal(hand, level_rank, trick)`
- `mc_rollout(...)`
- `mc_rollout_batch(...)`

The Python fallback implements legal move generation and strategic deduplication. Monte Carlo rollout helpers require the native extension.
