"""Phase 2 throughput benchmark on Modal: Python episode loop vs Rust episode loop.

Runs two training jobs (100 updates each, 32 actors, m3 shared-heads config) in
parallel and reports actor samples/sec and learner updates/sec for each.

Usage:
    modal run --detach ml/scripts/modal/bench_phase2_modal.py

Cost estimate: ~2 × $2.89/hr × (100-update wall time ~3 min) ≈ $0.30
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import modal

app = modal.App("guanzero-bench-phase2")
vol = modal.Volume.from_name("pvguan-runs", create_if_missing=True)
RUN_VOL = "/runs"

_root = Path(__file__).resolve().parent.parent.parent.parent  # repo root

# ── Image with Rust toolchain + guandan_rs ───────────────────────────────────
# The build step compiles the Rust PyO3 extension so actors can use
# the Rust episode loop and encoder (Phase 2).
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "gcc", "libssl-dev", "pkg-config", "build-essential")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable",
    )
    .pip_install("torch", "numpy", "pyyaml", "tqdm", "maturin==1.7.0")
    .add_local_dir(str(_root / "ml" / "src"), remote_path="/root/ml/src", copy=True)
    .add_local_dir(str(_root / "ml" / "scripts"), remote_path="/root/ml/scripts", copy=True)
    .run_commands(
        ". $HOME/.cargo/env && cd /root/ml/src/guandan_rs"
        " && maturin build --release -i python3"
        " && pip install target/wheels/*.whl",
    )
)

_CONFIG = "/root/ml/src/guandan/guanzero/configs/m3_l4_no_server.yaml"
_UPDATES = 100


@app.function(
    image=image,
    gpu="L4",
    cpu=32,
    memory=20 * 1024,
    timeout=1800,
    volumes={RUN_VOL: vol},
)
def run_variant(use_rust: bool, run_name: str) -> dict:
    import os
    import subprocess

    env = os.environ.copy()
    env["PYTHONPATH"] = "/root/ml/src:" + env.get("PYTHONPATH", "")
    env["GUANZERO_STREAM_LOGS"] = "1"
    env["GUANZERO_WEIGHT_DIR"] = "/tmp/guanzero_weights"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    if not use_rust:
        env["GUANZERO_NO_RUST"] = "1"

    run_dir = f"{RUN_VOL}/guanzero/bench_phase2/{run_name}"
    cmd = [
        "python", "-m", "guandan.guanzero",
        "--config", _CONFIG,
        "--updates", str(_UPDATES),
        "--device", "cuda",
        "--seed", "42",
        "--run-dir", run_dir,
    ]
    print(f"[{run_name}] Starting {'Rust' if use_rust else 'Python'} path: {' '.join(cmd)}")
    subprocess.run(cmd, env=env, check=True)
    vol.commit()

    # Parse metrics_learner.jsonl
    metrics_path = Path(run_dir) / "metrics_learner.jsonl"
    rows = [json.loads(l) for l in metrics_path.read_text().splitlines() if l.strip()]
    # Skip first entry (warmup / buffer fill), use steady-state rows
    steady = rows[1:] if len(rows) > 1 else rows
    actor_rates = [r["actor_rate_samp_per_sec"] for r in steady if r.get("actor_rate_samp_per_sec", 0) > 0]
    upd_rates   = [r["upd_per_sec"] for r in steady if r.get("upd_per_sec", 0) > 0]
    result = {
        "run_name": run_name,
        "use_rust": use_rust,
        "n_rows": len(steady),
        "mean_actor_samp_per_sec": statistics.mean(actor_rates) if actor_rates else 0.0,
        "mean_upd_per_sec": statistics.mean(upd_rates) if upd_rates else 0.0,
        "mean_queue_depth": statistics.mean(r.get("queue_depth", 0) for r in steady),
        "total_updates": rows[-1].get("updates", 0) if rows else 0,
    }
    print(f"[{run_name}] Done: {result}")
    return result


@app.local_entrypoint()
def main() -> None:
    print(f"Launching 2 × {_UPDATES}-update jobs on L4 + 32 vCPU ...")
    print(f"Config: {_CONFIG}")
    print()

    # Run sequentially — 32 vCPU each, not enough to run both in parallel.
    rust_r = run_variant.remote(use_rust=True,  run_name="rust_phase2")
    py_r   = run_variant.remote(use_rust=False, run_name="python_baseline")

    # Report
    print("\n" + "=" * 60)
    print("Phase 2 Benchmark Results (Modal L4, 32 vCPU, 32 actors)")
    print("=" * 60)
    header = f"{'Path':<16}  {'actor samp/s':>12}  {'upd/sec':>8}  {'queue':>6}"
    print(header)
    print("-" * len(header))
    for r in [rust_r, py_r]:
        tag = "rust (phase2)" if r["use_rust"] else "python"
        print(
            f"{tag:<16}  "
            f"{r['mean_actor_samp_per_sec']:>12.1f}  "
            f"{r['mean_upd_per_sec']:>8.2f}  "
            f"{r['mean_queue_depth']:>6.1f}"
        )
    if py_r["mean_actor_samp_per_sec"] > 0:
        speedup = rust_r["mean_actor_samp_per_sec"] / py_r["mean_actor_samp_per_sec"]
        print(f"\nActor throughput speedup: {speedup:.3f}×")
    print("=" * 60)
