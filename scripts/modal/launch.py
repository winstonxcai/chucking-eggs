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

# Build image: torch + numpy, mount local src/
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("torch", "numpy")
    .add_local_dir(
        str(Path(__file__).resolve().parent.parent.parent / "src"),
        remote_path="/root/src",
    )
)


@app.function(
    image=image,
    gpu="any",
    timeout=3600 * 6,  # 6 hour max
    volumes={CHECKPOINT_DIR: vol},
)
def train_remote(
    episodes: int = 50000,
    batch_size: int = 256,
    lr: float = 1e-3,
    buffer_size: int = 100_000,
    eval_interval: int = 500,
    eval_games: int = 100,
    target_sync: int = 1000,
    save_interval: int = 5000,
) -> str:
    import argparse
    import os
    import sys

    sys.path.insert(0, "/root/src")

    from guandan.train import train

    os.chdir(CHECKPOINT_DIR)

    train(argparse.Namespace(
        episodes=episodes,
        batch_size=batch_size,
        lr=lr,
        buffer_size=buffer_size,
        eval_interval=eval_interval,
        eval_games=eval_games,
        target_sync=target_sync,
        save_interval=save_interval,
    ))
    vol.commit()

    return f"Training complete. {episodes} episodes. Checkpoints saved to volume."


@app.local_entrypoint()
def main(
    episodes: int = 50000,
    batch_size: int = 256,
    lr: float = 1e-3,
    buffer_size: int = 100_000,
    eval_interval: int = 500,
    eval_games: int = 100,
    target_sync: int = 1000,
    save_interval: int = 5000,
):
    """Launch Guan Dan DMC training on Modal GPU."""
    result = train_remote.remote(
        episodes=episodes,
        batch_size=batch_size,
        lr=lr,
        buffer_size=buffer_size,
        eval_interval=eval_interval,
        eval_games=eval_games,
        target_sync=target_sync,
        save_interval=save_interval,
    )
    print(result)
