"""Modal GPU launcher for self-play fine-tuning.

Usage:
    modal run --detach scripts/modal/selfplay_launch.py
    modal run --detach scripts/modal/selfplay_launch.py --episodes 80000 --resume selfplay_best.pt
"""
from __future__ import annotations

from pathlib import Path

import modal

app = modal.App("guandan-selfplay")
vol = modal.Volume.from_name("guandan-checkpoints", create_if_missing=True)
CHECKPOINT_DIR = "/checkpoints"

SELFPLAY_DEFAULTS = dict(
    resume=f"{CHECKPOINT_DIR}/selfplay_best.pt",
    episodes=80000,
    workers=0,
    n_envs=128,
    train_steps=4,
    batch_size=1024,
    lr=3e-5,
    buffer_size=250000,
    eval_interval=10000,
    eval_games=100,
    save_interval=20000,
    epsilon_start=0.15,
    epsilon_end=0.03,
    epsilon_decay_frac=0.80,
    checkpoint_dir=CHECKPOINT_DIR,
    run_name="selfplay_modal",
    no_baseline=True,
    team_spirit=0.0,
)

try:
    _root = Path(__file__).resolve().parents[2]
except IndexError:
    _root = Path(__file__).resolve().parent

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("curl", "build-essential")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
    )
    .pip_install("torch", "numpy", "tqdm", "maturin")
    .add_local_dir(str(_root / "guandan_rs"), remote_path="/root/guandan_rs", copy=True)
    .run_commands(
        "bash -c 'source $HOME/.cargo/env && cd /root/guandan_rs && maturin build --release --interpreter python3.10'",
        "pip install /root/guandan_rs/target/wheels/guandan_rs-*.whl",
    )
    .add_local_dir(str(_root / "src"), remote_path="/root/src")
    .add_local_dir(str(_root / "scripts"), remote_path="/root/scripts")
)


@app.function(
    image=image,
    gpu="A10G",
    cpu=16,
    timeout=3600 * 4,
    volumes={CHECKPOINT_DIR: vol},
)
def selfplay_remote(**kwargs) -> str:
    import argparse
    import os
    import sys

    sys.path.insert(0, "/root/src")
    sys.path.insert(0, "/root/scripts")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    from selfplay import main as selfplay_main

    params = {**SELFPLAY_DEFAULTS, **kwargs}
    ns = argparse.Namespace(**params)
    selfplay_main(ns)
    vol.commit()
    return f"Self-play complete. {params['episodes']} episodes."


@app.local_entrypoint()
def main(
    episodes: int = 80000,
    run_name: str = "selfplay_modal",
    resume: str = "",
    n_envs: int = 64,
    eval_interval: int = 10000,
    eval_games: int = 100,
    team_spirit: float = 0.0,
    train_steps: int = 4,
) -> None:
    """Launch self-play fine-tuning on Modal A10G + GPU-batched GameRunner."""
    kwargs: dict = {
        "episodes": episodes,
        "run_name": run_name,
        "n_envs": n_envs,
        "eval_interval": eval_interval,
        "eval_games": eval_games,
        "team_spirit": team_spirit,
        "train_steps": train_steps,
    }
    if resume:
        kwargs["resume"] = f"{CHECKPOINT_DIR}/{resume}"
    result = selfplay_remote.remote(**kwargs)
    print(result)
