"""Modal launcher for Dart persistent actor-learner DMC.

Drives the distributed orchestrator (`guandan.dart.runtime.train`, invoked via
`python -m guandan.dart` through __main__.py) with
a config tuned for Modal GPU workers with 32 vCPUs. The default worker uses
an L4 GPU; actor count and all other hyperparameters are controlled by the YAML config.

Sanity smoke (~5 min, ~$0.25):
    modal run ml/scripts/modal/train_dart_modal.py \\
        --updates 2000 --run-name dart_l4_sanity

Benchmark (~10 min, ~$0.50) — primary perf-tuning target:
    modal run ml/scripts/modal/train_dart_modal.py \\
        --updates 4000 --run-name dart_l4_bench

Streamed stdout shows learner.log; metrics_learner.jsonl on the volume
carries upd_per_sec, queue_depth, gpu_mem_gb, drained_since_last_log.
"""

from __future__ import annotations

from pathlib import Path

import modal

app = modal.App("dart-train")
vol = modal.Volume.from_name("pvguan-runs", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
RUN_VOL = "/runs"

_root = Path(__file__).resolve().parent.parent.parent.parent  # repo root
_REMOTE_SRC = "/root/ml/src/"
_DEFAULT_CONFIG_PATH = "/root/ml/src/guandan/dart/configs/dart_l4.yaml"


def _local_config_path(config_path: str) -> str:
    """Map Modal container config paths to local paths for --dry-run validation."""
    if config_path.startswith(_REMOTE_SRC):
        return str(_root / "ml" / "src" / config_path[len(_REMOTE_SRC):])
    return config_path

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "gcc", "libssl-dev", "pkg-config", "build-essential")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable",
    )
    .pip_install("torch", "numpy", "pyyaml", "tqdm", "maturin==1.7.0")
    .add_local_dir(str(_root / "ml" / "src"), remote_path="/root/ml/src", copy=True)
    .run_commands(
        ". $HOME/.cargo/env && cd /root/ml/src/guandan_rs"
        " && maturin build --release -i python3"
        " && pip install target/wheels/*manylinux*.whl",
    )
)


@app.function(
    image=image,
    gpu="L4",
    cpu=32,
    memory=20 * 1024,   # peak observed ~22.5 GB with lanes=40; Modal soft limit
    timeout=3600 * 12,  # 12-hour cap
    volumes={RUN_VOL: vol, "/root/.cache/huggingface": hf_cache},
    secrets=[modal.Secret.from_name("huggingface-token")],
)
def train_remote(
    updates:     int,
    run_name:    str,
    seed:        int,
    config_path: str,
    device:      str,
    profile:     bool = False,
    resume:      str | None = None,
) -> str:
    import os
    import subprocess

    env = os.environ.copy()
    env["PYTHONPATH"] = "/root/ml/src:" + env.get("PYTHONPATH", "")
    # Stream learner/train logs to stdout so `modal run` shows live progress.
    env["DART_STREAM_LOGS"] = "1"
    # Keep weight publish/sync IO off the network-attached Modal volume.
    env["DART_WEIGHT_DIR"] = "/tmp/dart_weights"
    if profile:
        env["DART_SERVER_PROFILE"] = "1"
        env["DART_ACTOR_PROFILE"]  = "1"
        env["DART_LEARNER_PROFILE"] = "1"
    # Pin BLAS thread pools to 1 — actor processes already set torch.set_num_threads(1)
    # but numpy/MKL/OpenBLAS are separate and would otherwise contend across vCPUs.
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"

    run_dir = f"{RUN_VOL}/dart/{run_name}"

    # Auto-detect latest checkpoint in run dir so preemption restarts don't rewind.
    import glob
    checkpoints_dir = f"{run_dir}/checkpoints"
    existing = sorted(glob.glob(f"{checkpoints_dir}/update_*.pt"))
    if existing:
        latest = existing[-1]
        if latest != resume:
            print(f"Auto-resuming from latest checkpoint: {latest} (passed resume={resume})")
        resume = latest

    cmd = [
        "python", "-m", "guandan.dart",
        "--config",  config_path,
        "--updates", str(updates),
        "--device",  device,
        "--seed",    str(seed),
        "--run-dir", run_dir,
    ]
    if resume:
        cmd.extend(["--resume", resume])

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, env=env, check=True)
    vol.commit()
    return f"{run_dir}/checkpoints/final.pt"


@app.local_entrypoint()
def main(
    updates: int = 1000,
    run_name: str = "dart_l4_bench",
    seed: int = 0,
    config_path: str = _DEFAULT_CONFIG_PATH,
    device: str = "cuda",
    profile: bool = False,
    resume: str | None = None,
    dry_run: bool = False,
    wait: bool = False,
) -> None:
    if dry_run:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
        from guandan.dart.config import load_config_from_yaml
        local_path = _local_config_path(config_path)
        cfg = load_config_from_yaml(local_path)
        print(
            f"Config OK: path={local_path}, model_type={cfg.model_type}, "
            f"n_actors={cfg.n_actors}, updates={updates}"
        )
        return

    kwargs = dict(
        updates=updates,
        run_name=run_name,
        seed=seed,
        config_path=config_path,
        device=device,
        profile=profile,
        resume=resume,
    )
    if wait:
        final_ckpt = train_remote.remote(**kwargs)
        print(f"Completed. Final checkpoint: {final_ckpt}")
    else:
        fn = train_remote.spawn(**kwargs)
        print(f"Spawned function call id: {fn.object_id}")
