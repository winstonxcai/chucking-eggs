"""Modal launcher for QMIX training (Phase A + B).

Usage:
    modal run --detach scripts/modal/qmix_launch.py
    modal run --detach scripts/modal/qmix_launch.py --phase AB --episodes 15000
"""

from __future__ import annotations

from pathlib import Path

import modal

app = modal.App("guandan-qmix")
vol = modal.Volume.from_name("guandan-checkpoints", create_if_missing=True)
CHECKPOINT_DIR = "/checkpoints"

QMIX_DEFAULTS = dict(
    resume=f"{CHECKPOINT_DIR}/selfplay_best.pt",
    phase="AB",
    episodes=50000,
    batch_size=512,
    train_steps=4,
    lr_mixer=1e-3,
    lr_q=1e-6,
    qmix_loss_weight=0.001,
    wqmix_alpha=0.1,
    eval_interval=10000,
    eval_games=500,
    epsilon_start=0.10,
    epsilon_end=0.01,
    epsilon_decay_frac=0.80,
    checkpoint_dir=CHECKPOINT_DIR,
    run_name="qmix_v1",
    mixer_resume="",
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
    .add_local_dir(str(_root / "src" / "guandan_rs"), remote_path="/root/guandan_rs", copy=True)
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
    cpu=4,
    timeout=3600 * 4,
    volumes={CHECKPOINT_DIR: vol},
)
def qmix_remote(**kwargs) -> str:
    import argparse
    import os
    import sys

    sys.path.insert(0, "/root/src")
    sys.path.insert(0, "/root/scripts")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    from qmix_train import main as qmix_main

    params = {**QMIX_DEFAULTS, **kwargs}
    ns = argparse.Namespace(**params)
    qmix_main(ns)
    vol.commit()
    return f"QMIX complete. Phase: {params['phase']}, {params['episodes']} episodes."


@app.local_entrypoint()
def main(
    phase: str = "AB",
    episodes: int = 50000,
    run_name: str = "qmix_v2",
    resume: str = "",
    mixer_resume: str = "",
    eval_games: int = 500,
) -> None:
    """Launch QMIX training on Modal A10G."""
    kwargs: dict = {
        "phase": phase,
        "episodes": episodes,
        "run_name": run_name,
        "eval_games": eval_games,
    }
    if resume:
        kwargs["resume"] = f"{CHECKPOINT_DIR}/{resume}"
    if mixer_resume:
        kwargs["mixer_resume"] = f"{CHECKPOINT_DIR}/{mixer_resume}"
    result = qmix_remote.remote(**kwargs)
    print(result)
