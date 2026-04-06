"""Modal GPU launcher for Guan Dan DMC training.

Usage:
    modal run scripts/modal/launch.py
    modal run scripts/modal/launch.py --episodes 50000
"""

from __future__ import annotations

from pathlib import Path

import modal

# Modal app
app = modal.App("guandan-train")

# Volume for persisting checkpoints
vol = modal.Volume.from_name("guandan-checkpoints", create_if_missing=True)
CHECKPOINT_DIR = "/checkpoints"

# Single source of truth for training hyperparameter defaults
TRAIN_DEFAULTS = dict(
    episodes=30000,
    batch_size=1024,
    lr=1e-4,
    buffer_size=250_000,
    eval_interval=1000,
    eval_games=100,
    save_interval=5000,
    patience=10,
    lstm_hidden=128,
    mlp_hidden=512,
    train_steps=4,
    pretrain_games=5000,
    n_workers=1,  # Modal sandbox doesn't support multiprocessing
    quick=False,
    run_name=None,
)

_root = Path(__file__).resolve().parent.parent.parent

# Build image: torch + numpy + Rust movegen, mount local src/
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("curl", "build-essential")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
    )
    .pip_install("torch", "numpy", "maturin", "tqdm")
    .add_local_dir(str(_root / "src" / "guandan_rs"), remote_path="/root/guandan_rs", copy=True)
    .run_commands(
        "bash -c 'source $HOME/.cargo/env && cd /root/guandan_rs && maturin build --release --interpreter python3.10'",
        "pip install /root/guandan_rs/target/wheels/guandan_rs-*.whl",
    )
    .add_local_dir(
        str(_root / "src"),
        remote_path="/root/src",
    )
)


@app.function(
    image=image,
    gpu="A10G",
    timeout=3600 * 6,  # 6 hour max
    volumes={CHECKPOINT_DIR: vol},
)
def train_remote(**kwargs) -> str:
    import argparse
    import os
    import sys

    sys.path.insert(0, "/root/src")

    from datetime import datetime

    from guandan.training.train import train

    os.chdir(CHECKPOINT_DIR)

    params = {**TRAIN_DEFAULTS, **kwargs}
    if "run_name" not in params or params["run_name"] is None:
        params["run_name"] = datetime.now().strftime("%Y%m%d_%H%M%S")
    train(argparse.Namespace(**params))
    vol.commit()

    return f"Training complete. {params['episodes']} episodes. Checkpoints saved to volume."


@app.local_entrypoint()
def main(
    episodes: int = 30000,
    batch_size: int = 1024,
    lr: float = 1e-4,
    buffer_size: int = 250_000,
    eval_interval: int = 500,
    eval_games: int = 200,
    save_interval: int = 5000,
    patience: int = 10,
    lstm_hidden: int = 128,
    mlp_hidden: int = 512,
    train_steps: int = 4,
):
    """Launch Guan Dan DMC training on Modal GPU."""
    # Modal CLI needs typed params for --flag parsing, but we forward as kwargs
    local_args = {k: v for k, v in locals().items()}
    result = train_remote.remote(**local_args)
    print(result)
