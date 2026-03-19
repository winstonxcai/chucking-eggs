"""Modal GPU launcher for supervised distillation.

Usage:
    modal run --detach scripts/modal/distill_launch.py
    modal run --detach scripts/modal/distill_launch.py --stage 1 --games 100 --epochs 1 --run-name smoke_modal
    modal run --detach scripts/modal/distill_launch.py --run-name distill_modal --stage 2
"""
from __future__ import annotations

from pathlib import Path

import modal

app = modal.App("guandan-distill")
vol = modal.Volume.from_name("guandan-checkpoints", create_if_missing=True)
CHECKPOINT_DIR = "/checkpoints"

DISTILL_DEFAULTS = dict(
    stage=2,
    games=5000,
    epochs=3,
    batch_size=32,
    lr=1e-4,
    checkpoint_dir=CHECKPOINT_DIR,
    run_name="distill_modal",
    resume=None,
)

try:
    _root = Path(__file__).resolve().parents[2]  # local: scripts/modal/distill_launch.py
except IndexError:
    _root = Path(__file__).resolve().parent  # Modal: /root/distill_launch.py

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("torch", "numpy", "tqdm")
    .add_local_dir(str(_root / "src"), remote_path="/root/src")
    .add_local_dir(str(_root / "scripts"), remote_path="/root/scripts")
)


@app.function(
    image=image,
    gpu="A10G",
    timeout=3600 * 6,
    volumes={CHECKPOINT_DIR: vol},
)
def distill_remote(**kwargs) -> str:
    import argparse
    import os
    import sys

    sys.path.insert(0, "/root/src")
    sys.path.insert(0, "/root/scripts")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    from distill import main as distill_main

    params = {**DISTILL_DEFAULTS, **kwargs}
    ns = argparse.Namespace(**params)
    distill_main(ns)
    vol.commit()
    return f"Distillation complete. Stage {params['stage']}, {params['games']} games/stage."


@app.local_entrypoint()
def main(
    stage: int = 2,
    games: int = 5000,
    epochs: int = 3,
    run_name: str = "distill_modal",
    resume: str = "",
) -> None:
    """Launch supervised distillation on Modal A10G."""
    kwargs: dict = {"stage": stage, "games": games, "epochs": epochs, "run_name": run_name}
    if resume:
        kwargs["resume"] = resume
    result = distill_remote.remote(**kwargs)
    print(result)
