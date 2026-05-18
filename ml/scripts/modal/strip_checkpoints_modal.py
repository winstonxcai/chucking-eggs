"""Strip optimizer/replay state from checkpoints on the Modal volume.

Reads update_*.pt from checkpoints/, writes weights-only versions to
checkpoints_slim/. Download from checkpoints_slim/ for eval — 19 MB vs 1.5 GB.

Usage:
    modal run ml/scripts/modal/strip_checkpoints_modal.py --run-name dart_v5_baseline_400k
"""
from __future__ import annotations

from pathlib import Path

import modal

app   = modal.App("dart-strip")
vol   = modal.Volume.from_name("pvguan-runs", create_if_missing=False)
RUN_VOL = "/runs"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch")
)


@app.function(
    image=image,
    cpu=4,
    memory=8 * 1024,
    timeout=3600 * 2,
    volumes={RUN_VOL: vol},
)
def strip_remote(run_name: str, force: bool = False) -> list[str]:
    import glob, os
    import torch

    ckpt_dir  = Path(f"{RUN_VOL}/dart/{run_name}/checkpoints")
    slim_dir  = Path(f"{RUN_VOL}/dart/{run_name}/checkpoints_slim")
    slim_dir.mkdir(parents=True, exist_ok=True)

    stripped = []
    for src in sorted(ckpt_dir.glob("update_*.pt")):
        dst = slim_dir / src.name
        if dst.exists() and not force:
            continue
        size_mb = os.path.getsize(src) / 1e6
        ckpt = torch.load(src, map_location="cpu", weights_only=False)
        slim = {
            "checkpoint_format_version": ckpt.get("checkpoint_format_version"),
            "total_updates":             ckpt.get("total_updates"),
            "episode":                   ckpt.get("episode"),
            "q_net":                     ckpt["q_net"],
        }
        torch.save(slim, dst)
        slim_mb = os.path.getsize(dst) / 1e6
        print(f"{src.name}: {size_mb:.0f}MB -> {slim_mb:.1f}MB")
        stripped.append(src.name)

    vol.commit()
    print(f"Done. {len(stripped)} checkpoints stripped to {slim_dir}")
    return stripped


@app.local_entrypoint()
def main(run_name: str = "dart_v5_baseline_400k", force: bool = False) -> None:
    result = strip_remote.remote(run_name=run_name, force=force)
    print(f"Stripped {len(result)} checkpoints.")
