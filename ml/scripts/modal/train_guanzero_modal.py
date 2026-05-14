"""Modal launcher for GuanZero persistent actor-learner DMC.

Drives the distributed orchestrator (`guandan.guanzero.runtime.train`, invoked via
`python -m guandan.guanzero` through __main__.py) with
a config tuned for Modal GPU workers with 32 vCPUs. The default worker uses
an L4 GPU; override `config_path`, `n_actors`, and `updates` for other shapes.

Sanity smoke (~5 min, ~$0.25):
    modal run ml/scripts/modal/train_guanzero_modal.py \\
        --updates 2000 --n-actors 8 --run-name guanzero_l4_sanity

Benchmark (~10 min, ~$0.50) — primary perf-tuning target:
    modal run ml/scripts/modal/train_guanzero_modal.py \\
        --updates 4000 --run-name guanzero_l4_bench

Streamed stdout shows learner.log; metrics_learner.jsonl on the volume
carries upd_per_sec, queue_depth, gpu_mem_gb, drained_since_last_log.
"""

from __future__ import annotations

from pathlib import Path

import modal

app = modal.App("guanzero-train")
vol = modal.Volume.from_name("pvguan-runs", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
RUN_VOL = "/runs"

_root = Path(__file__).resolve().parent.parent.parent.parent  # repo root

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "gcc", "libssl-dev", "pkg-config", "build-essential")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable",
    )
    .pip_install("torch", "numpy", "pyyaml", "tqdm", "maturin==1.7.0")
    .add_local_dir(str(_root / "ml" / "src"),     remote_path="/root/ml/src", copy=True)
    .add_local_dir(str(_root / "ml" / "scripts"), remote_path="/root/ml/scripts", copy=True)
    .run_commands(
        ". $HOME/.cargo/env && cd /root/ml/src/guandan_rs"
        " && maturin build --release -i python3"
        " && pip install target/wheels/*.whl",
    )
)


@app.function(
    image=image,
    gpu="L4",
    cpu=32,
    memory=20 * 1024,   # peak observed 16.89 GB; 20 GB leaves ~3 GB headroom
    timeout=3600 * 6,  # 6-hour cap per CLAUDE.md
    volumes={RUN_VOL: vol, "/root/.cache/huggingface": hf_cache},
    secrets=[modal.Secret.from_name("huggingface-token")],
)
def train_remote(
    updates:     int,
    run_name:    str,
    seed:        int,
    config_path: str,
    n_actors:    int | None,
    device:      str,
    profile:     bool = False,
    resume:      str | None = None,
) -> str:
    import os
    import subprocess

    env = os.environ.copy()
    env["PYTHONPATH"] = "/root/ml/src:" + env.get("PYTHONPATH", "")
    # Stream learner/train logs to stdout so `modal run` shows live progress.
    env["GUANZERO_STREAM_LOGS"] = "1"
    # Keep weight publish/sync IO off the network-attached Modal volume.
    env["GUANZERO_WEIGHT_DIR"] = "/tmp/guanzero_weights"
    if profile:
        env["GUANZERO_SERVER_PROFILE"] = "1"
        env["GUANZERO_ACTOR_PROFILE"]  = "1"
        env["GUANZERO_LEARNER_PROFILE"] = "1"
    # Pin BLAS thread pools to 1 — actor processes already set torch.set_num_threads(1)
    # but numpy/MKL/OpenBLAS are separate and would otherwise contend across vCPUs.
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"

    run_dir = f"{RUN_VOL}/guanzero/{run_name}"

    cmd = [
        "python", "-m", "guandan.guanzero",
        "--config",  config_path,
        "--updates", str(updates),
        "--device",  device,
        "--seed",    str(seed),
        "--run-dir", run_dir,
    ]
    if n_actors is not None:
        cmd.extend(["--n-actors", str(n_actors)])
    if resume:
        cmd.extend(["--resume", resume])

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, env=env, check=True)
    vol.commit()
    return f"{run_dir}/checkpoints/final.pt"


@app.local_entrypoint()
def main(
    updates: int = 1000,
    run_name: str = "guanzero_l4_bench",
    seed: int = 0,
    config_path: str = "/root/ml/src/guandan/guanzero/configs/m0_l4_distributed.yaml",
    n_actors: int | None = None,
    device: str = "cuda",
    profile: bool = False,
    resume: str | None = None,
) -> None:
    out = train_remote.remote(
        updates=updates,
        run_name=run_name,
        seed=seed,
        config_path=config_path,
        n_actors=n_actors,
        device=device,
        profile=profile,
        resume=resume,
    )
    print(f"Final checkpoint: {out}")
