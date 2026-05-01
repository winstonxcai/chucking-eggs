"""Modal launcher for GuanZero (M0 baseline) DMC training.

Mirrors the structure of `distill_pvguan_modal.py`. Single-process actor +
learner; A10G GPU for the LSTM forward, 48 CPU for the env steps. The
checkpoint goes to `/runs/guanzero/<run_name>/checkpoints/final.pt` on
the shared `pvguan-runs` volume so existing eval scripts can pick it up.

Smoke test (~5 min):
    modal run ml/scripts/modal/train_guanzero_modal.py \\
        --episodes 500 --run-name guanzero_m0_smoke

Production run (calculate runtime first per CLAUDE.md 6h cap):
    modal run --detach ml/scripts/modal/train_guanzero_modal.py \\
        --episodes 30000 --run-name guanzero_m0_prod
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
    .pip_install("torch", "numpy", "pyyaml", "tqdm")
    .add_local_dir(str(_root / "ml" / "src"),     remote_path="/root/ml/src")
    .add_local_dir(str(_root / "ml" / "scripts"), remote_path="/root/ml/scripts")
)


@app.function(
    image=image,
    gpu="A10G",
    cpu=48,
    memory=32 * 1024,
    timeout=3600 * 6,  # 6-hour cap per CLAUDE.md
    volumes={RUN_VOL: vol, "/root/.cache/huggingface": hf_cache},
    secrets=[modal.Secret.from_name("huggingface-token")],
)
def train_remote(
    episodes:    int,
    run_name:    str,
    seed:        int,
    config_path: str | None,
    device:      str,
) -> str:
    import os
    import subprocess

    env = os.environ.copy()
    env["PYTHONPATH"] = "/root/ml/src:" + env.get("PYTHONPATH", "")

    run_dir = f"{RUN_VOL}/guanzero/{run_name}"

    cmd = [
        "python", "-m", "guandan.guanzero",
        "--episodes", str(episodes),
        "--seed",     str(seed),
        "--device",   device,
        "--run-dir",  run_dir,
    ]
    if config_path:
        cmd.extend(["--config", config_path])

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, env=env, check=True)
    vol.commit()
    return f"{run_dir}/checkpoints/final.pt"


@app.local_entrypoint()
def main(
    episodes: int = 30_000,
    run_name: str = "guanzero_m0",
    seed: int = 0,
    config_path: str | None = "/root/ml/src/guandan/guanzero/config/m0_baseline.yaml",
    device: str = "cuda",
) -> None:
    out = train_remote.remote(
        episodes=episodes,
        run_name=run_name,
        seed=seed,
        config_path=config_path,
        device=device,
    )
    print(f"Final checkpoint: {out}")
