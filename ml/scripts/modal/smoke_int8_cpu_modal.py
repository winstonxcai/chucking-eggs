"""Modal smoke: fp32 vs int8 actor — 100 learner updates each, GPU learner + 32 CPU actors.

Two back-to-back guanzero training jobs on an L4 GPU + 32 vCPU worker
(matches the production training shape). Learner runs on cuda; actors run
on cpu, no inference server. One pass with ``use_int8_actor=False``, one
with ``True``. Each caps at 100 learner updates.

Run:
    modal run ml/scripts/modal/smoke_int8_cpu_modal.py
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

app = modal.App("guanzero-smoke-int8-cpu")
vol = modal.Volume.from_name("pvguan-runs", create_if_missing=True)

_root = Path(__file__).resolve().parent.parent.parent.parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("build-essential", "curl", "ca-certificates")
    .run_commands(
        "curl --retry 5 --retry-delay 5 --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs "
        "| sh -s -- -y --default-toolchain stable --profile minimal",
        "ln -s /root/.cargo/bin/cargo /usr/local/bin/cargo",
        "ln -s /root/.cargo/bin/rustc /usr/local/bin/rustc",
    )
    .pip_install("torch", "numpy", "pyyaml", "maturin", "tqdm")
    .add_local_dir(
        str(_root / "ml" / "src"),
        remote_path="/root/ml/src",
        ignore=["**/target/**", "**/__pycache__/**", "**/*.pyc"],
    )
    .add_local_dir(str(_root / "ml" / "scripts"), remote_path="/root/ml/scripts")
)


# Smoke config — small budget so each run finishes in a few minutes on CPU.
# Mirrors m5_clean_baseline_l4 shape but with tiny buffer_min and update cap.
SMOKE_CONFIG: dict = {
    "n_actors": 32,
    "sync_interval_updates": 50,
    "sync_jitter_updates": 10,
    "actor_push_batch_size": 64,
    "sample_queue_maxsize": 64,
    "max_drain_batches_per_loop": 16,
    "publish_interval_updates": 100,
    "checkpoint_every_updates": 100_000,   # don't checkpoint during smoke
    "log_every_updates": 50,
    "updates_per_learner_step": 1,
    "total_updates_target": 500,

    "model_type": "shared_trick_heads",
    "shared_head_role_d_model": 128,
    "shared_head_history_hidden": 256,
    "shared_head_global_hidden": 128,
    "shared_head_action_hidden": 128,
    "shared_head_trunk_hidden": 1024,
    "shared_head_trunk_layers": 4,
    "dropout": 0.0,

    "use_inference_server": False,
    "device": "cuda",   # learner on GPU; actors always run play_episode on cpu (worker.py:469)

    "lr": 3.0e-5,
    "gamma": 1.0,
    "batch_size": 1024,
    "buffer_capacity": 100_000,
    "buffer_capacity_per_player": 25_000,
    "buffer_min_size": 2048,   # small so updates can start quickly

    "epsilon_start": 0.01,
    "epsilon_final": 0.01,
    "epsilon_decay_updates": 100,
    "epsilon_frozen": 0.00,
    "is_partner_visible": True,

    "latest_vs_latest_frac": 1.0,
    "latest_vs_hard_bot_frac": 0.0,
    "population_pool": [],
}


@app.function(
    image=image,
    gpu="L4",
    cpu=32,
    memory=24 * 1024,
    timeout=3600,
    volumes={"/runs": vol},
)
def smoke_remote() -> dict:
    """Run fp32 then int8 in one container so the Rust build is amortized."""
    import os
    import subprocess
    import time
    from pathlib import Path

    import yaml

    # Build the Rust extension once.
    lock = Path("/root/ml/src/guandan_rs/Cargo.lock")
    if lock.exists():
        lock.unlink()
    print("Building guandan_rs ...", flush=True)
    subprocess.run(
        ["maturin", "build", "--release", "--out", "/tmp/wheels"],
        cwd="/root/ml/src/guandan_rs",
        check=True,
    )
    wheels = list(Path("/tmp/wheels").glob("guandan_rs-*.whl"))
    subprocess.run(["pip", "install", "--no-deps", str(wheels[0])], check=True)

    base_env = os.environ.copy()
    base_env["PYTHONPATH"] = "/root/ml/src:" + base_env.get("PYTHONPATH", "")
    base_env["GUANZERO_STREAM_LOGS"] = "1"
    base_env["GUANZERO_ACTOR_PROFILE"] = "1"
    base_env["GUANZERO_LEARNER_PROFILE"] = "1"
    base_env["OMP_NUM_THREADS"] = "1"
    base_env["MKL_NUM_THREADS"] = "1"
    base_env["OPENBLAS_NUM_THREADS"] = "1"
    base_env["NUMEXPR_NUM_THREADS"] = "1"

    results: list[dict] = []
    for use_int8 in (False, True):
        run_tag = "int8" if use_int8 else "fp32"
        cfg = dict(SMOKE_CONFIG)
        cfg["use_int8_actor"] = use_int8

        run_dir = f"/runs/guanzero/smoke_int8_{run_tag}_{int(time.time())}"
        cfg_path = f"/tmp/smoke_cfg_{run_tag}.yaml"
        with open(cfg_path, "w") as f:
            yaml.safe_dump(cfg, f)

        env = base_env.copy()
        env["GUANZERO_WEIGHT_DIR"] = f"/tmp/guanzero_weights_{run_tag}"

        print(f"\n========== RUN: use_int8={use_int8} ==========", flush=True)
        print(f"run_dir={run_dir}", flush=True)
        t_start = time.perf_counter()
        subprocess.run(
            [
                "python", "-m", "guandan.guanzero.train",
                "--config", cfg_path,
                "--run-dir", run_dir,
            ],
            env=env,
            check=True,
        )
        wall_s = time.perf_counter() - t_start

        metrics_path = Path(run_dir) / "metrics_learner.jsonl"
        last_metric: dict = {}
        upd_per_sec_samples: list[float] = []
        if metrics_path.exists():
            with open(metrics_path) as f:
                for line in f:
                    try:
                        m = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    last_metric = m
                    if "upd_per_sec" in m:
                        upd_per_sec_samples.append(float(m["upd_per_sec"]))

        # Skip first sample (warmup spike) when computing mean.
        steady = upd_per_sec_samples[1:] if len(upd_per_sec_samples) > 1 else upd_per_sec_samples
        mean_upd_per_sec = sum(steady) / len(steady) if steady else None

        results.append({
            "use_int8": use_int8,
            "wall_s": wall_s,
            "last_metric": last_metric,
            "upd_per_sec_samples": upd_per_sec_samples,
            "mean_upd_per_sec": mean_upd_per_sec,
            "run_dir": run_dir,
        })

    return {"results": results}


@app.local_entrypoint()
def main():
    out = smoke_remote.remote()
    print("\n\n========================================")
    print("           SMOKE A/B RESULTS")
    print("========================================")
    for r in out["results"]:
        label = "int8" if r["use_int8"] else "fp32"
        m = r.get("mean_upd_per_sec")
        print(f"\n{label}:")
        print(f"  wall:                {r['wall_s']:.1f} s")
        print(f"  upd_per_sec samples: {r['upd_per_sec_samples']}")
        print(f"  mean upd_per_sec:    {m:.2f}" if m else "  mean upd_per_sec:    n/a")
        print(f"  last metric:         {r['last_metric']}")

    rs = {r["use_int8"]: r for r in out["results"]}
    if rs[False].get("mean_upd_per_sec") and rs[True].get("mean_upd_per_sec"):
        speedup = rs[True]["mean_upd_per_sec"] / rs[False]["mean_upd_per_sec"]
        print(f"\nspeedup (int8 / fp32 upd_per_sec): {speedup:.2f}x")
    if rs[False].get("wall_s") and rs[True].get("wall_s"):
        wall_speedup = rs[False]["wall_s"] / rs[True]["wall_s"]
        print(f"speedup (fp32_wall / int8_wall):   {wall_speedup:.2f}x")
