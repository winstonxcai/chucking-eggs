"""Modal launcher for pvguan supervised distillation.

Collects supervisor decisions in parallel (CPU-heavy) then trains the actor
head with cross-entropy (GPU-light). Lives on the same `pvguan-runs` volume
as the PPO training output. The resulting checkpoint is written to
`/runs/warmstart/pvguan_distilled_<supervisor>.pt` so it can be referenced
by `train_pvguan_modal.py --warmstart-path ...`.

Smoke test (~2 min):
    modal run ml/scripts/modal/distill_pvguan_modal.py \\
        --supervisor yaoji --n-decisions 20000 --epochs 2 --workers 8

Production run (~10-15 min on 48-CPU + A10G):
    modal run --detach ml/scripts/modal/distill_pvguan_modal.py \\
        --supervisor yaoji --n-decisions 500000
"""

from __future__ import annotations

from pathlib import Path

import modal

app = modal.App("pvguan-distill")
vol = modal.Volume.from_name("pvguan-runs", create_if_missing=True)
RUN_VOL = "/runs"

_root = Path(__file__).resolve().parent.parent.parent.parent  # repo root

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "tqdm", "matplotlib")
    .add_local_dir(str(_root / "ml" / "src"),     remote_path="/root/ml/src")
    .add_local_dir(str(_root / "ml" / "scripts"), remote_path="/root/ml/scripts")
)


@app.function(
    image=image,
    gpu="A10G",
    cpu=48,
    memory=32 * 1024,
    timeout=3600 * 2,           # 2-hour cap (production ~15 min)
    volumes={RUN_VOL: vol},
)
def distill_remote(
    supervisor:        str,
    supervisor_search: str,
    n_decisions:       int,
    epochs:            int,
    batch_size:        int,
    lr:                float,
    hidden:            int,
    workers:           int,
    seed:              int,
    label_smoothing:   float,
    output_name:       str,
) -> str:
    import os
    import subprocess

    env = os.environ.copy()
    env["PYTHONPATH"] = "/root/ml/src"

    output_path = f"{RUN_VOL}/warmstart/{output_name}"

    cmd = [
        "python", "/root/ml/scripts/train/distill_pvguan.py",
        "--supervisor",        supervisor,
        "--supervisor-search", supervisor_search,
        "--n-decisions",       str(n_decisions),
        "--epochs",            str(epochs),
        "--batch-size",        str(batch_size),
        "--lr",                str(lr),
        "--hidden",            str(hidden),
        "--workers",           str(workers),
        "--seed",              str(seed),
        "--label-smoothing",   str(label_smoothing),
        "--output",            output_path,
    ]

    print("=" * 64)
    print("Launching:", " ".join(cmd))
    print("=" * 64)
    subprocess.run(cmd, env=env, cwd="/root", check=True)
    vol.commit()
    return f"done: supervisor={supervisor} → {output_path}"


@app.local_entrypoint()
def main(
    supervisor:        str   = "yaoji",
    supervisor_search: str   = "off",
    n_decisions:       int   = 500_000,
    epochs:            int   = 8,
    batch_size:        int   = 512,
    lr:                float = 3e-4,
    hidden:            int   = 256,
    workers:           int   = 32,
    seed:              int   = 0,
    label_smoothing:   float = 0.1,
    output_name:       str   = "",
) -> None:
    name = output_name or f"pvguan_distilled_{supervisor}.pt"
    result = distill_remote.remote(
        supervisor=supervisor,
        supervisor_search=supervisor_search,
        n_decisions=n_decisions,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        hidden=hidden,
        workers=workers,
        seed=seed,
        label_smoothing=label_smoothing,
        output_name=name,
    )
    print(result)
