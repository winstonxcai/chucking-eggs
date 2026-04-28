"""Modal GPU launcher for pvguan PPO training.

Trains a PV-AC or PV-PTIE actor-critic with PPO from the Jidan-distilled
warmstart. Output (config, metrics, train.log, validation_eval.jsonl,
figures, checkpoints) lives on the `pvguan-runs` Modal volume.

Smoke test (~5 min wall):
    modal run ml/scripts/modal/train_pvguan_modal.py \\
        --critic pv --seed 0 --detach \\
        --total-decisions 50000 --iter-decisions 4096 \\
        --val-every 5 --val-games 20

Production run (~75-100 min on A10G):
    modal run ml/scripts/modal/train_pvguan_modal.py \\
        --critic pv --seed 0 --detach
"""

from __future__ import annotations

from pathlib import Path

import modal

app = modal.App("pvguan-train")
vol = modal.Volume.from_name("pvguan-runs", create_if_missing=True)
RUN_VOL = "/runs"

_root = Path(__file__).resolve().parent.parent.parent.parent  # repo root

# Pure-Python image: guandan_rs is optional; combos.py falls back automatically
# Warmstart checkpoint is read from the volume (not bundled into the image) to
# keep the image build fast and avoid the local-file upload path.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "tqdm", "matplotlib")
    .add_local_dir(str(_root / "ml" / "src"),     remote_path="/root/ml/src")
    .add_local_dir(str(_root / "ml" / "scripts"), remote_path="/root/ml/scripts")
)

WARMSTART_VOL_PATH = "/runs/warmstart/pvguan_distilled_jidan.pt"


@app.function(
    image=image,
    gpu="A10G",
    cpu=48,
    memory=32 * 1024,
    timeout=3600 * 6,           # 6-hour hard cap (48 CPUs → ~3h estimated)
    volumes={RUN_VOL: vol},
)
def train_remote(
    critic:               str,
    seed:                 int,
    total_decisions:      int,
    iter_decisions:       int,
    critic_warmup:        int,
    kl_init:              float,
    kl_decay_actor_iters: int,
    temperature:          float,
    hidden:               int,
    mini_batch:           int,
    val_every:            int,
    val_games:            int,
    val_opponents:        str,
    rollout_workers:      int,
    snapshot_interval:    int,
    resume:               str | None,
    opponent_mix:         str,
    selfplay_frac:        float,
    val_patience:         int,
    warmstart_path:       str | None,
    going_out_shape:      float,
    run_dir_override:     str | None,
) -> str:
    import os
    import subprocess

    env = os.environ.copy()
    env["PYTHONPATH"] = "/root/ml/src"

    run_dir = run_dir_override or f"{RUN_VOL}/pvguan_{critic}_seed{seed}"

    cmd = [
        "python", "/root/ml/scripts/train/train_pvguan.py",
        "--critic",                  critic,
        "--seed",                    str(seed),
        "--total-decisions",         str(total_decisions),
        "--iter-decisions",          str(iter_decisions),
        "--critic-warmup",           str(critic_warmup),
        "--kl-init",                 str(kl_init),
        "--kl-decay-actor-iters",    str(kl_decay_actor_iters),
        "--temperature",             str(temperature),
        "--hidden",                  str(hidden),
        "--mini-batch",              str(mini_batch),
        "--val-every",               str(val_every),
        "--val-games",               str(val_games),
        "--val-opponents",           val_opponents,
        "--rollout-workers",         str(rollout_workers),
        "--snapshot-interval",       str(snapshot_interval),
        "--run-dir",                 run_dir,
        "--selfplay-frac",           str(selfplay_frac),
        "--val-patience",            str(val_patience),
        "--going-out-shape",         str(going_out_shape),
    ]
    if opponent_mix:
        cmd.extend(["--opponent-mix", opponent_mix])
    if resume:
        cmd.extend(["--resume", resume])
    else:
        ws = warmstart_path or WARMSTART_VOL_PATH
        cmd.extend(["--warmstart", ws])

    print("=" * 64)
    print("Launching:", " ".join(cmd))
    print("=" * 64)
    subprocess.run(cmd, env=env, cwd="/root", check=True)
    vol.commit()
    return f"done: critic={critic} seed={seed} → {run_dir}"


@app.local_entrypoint()
def main(
    critic:               str   = "pv",
    seed:                 int   = 0,
    total_decisions:      int   = 6_000_000,
    iter_decisions:       int   = 8192,
    critic_warmup:        int   = 30,
    kl_init:              float = 0.05,
    kl_decay_actor_iters: int   = 100,
    temperature:          float = 1.0,
    hidden:               int   = 256,
    mini_batch:           int   = 1024,
    val_every:            int   = 50,
    val_games:            int   = 100,
    val_opponents:        str   = "jidan,yaoji",
    rollout_workers:      int   = 16,
    snapshot_interval:    int   = 1_000_000,
    resume:               str   = "",
    opponent_mix:         str   = "",
    selfplay_frac:        float = 1.0,
    val_patience:         int   = 0,
    warmstart_path:       str   = "",
    going_out_shape:      float = 0.0,
    run_dir_override:     str   = "",
) -> None:
    result = train_remote.remote(
        critic=critic, seed=seed,
        total_decisions=total_decisions,
        iter_decisions=iter_decisions,
        critic_warmup=critic_warmup,
        kl_init=kl_init,
        kl_decay_actor_iters=kl_decay_actor_iters,
        temperature=temperature,
        hidden=hidden,
        mini_batch=mini_batch,
        val_every=val_every,
        val_games=val_games,
        val_opponents=val_opponents,
        rollout_workers=rollout_workers,
        snapshot_interval=snapshot_interval,
        resume=resume if resume else None,
        opponent_mix=opponent_mix,
        selfplay_frac=selfplay_frac,
        val_patience=val_patience,
        warmstart_path=warmstart_path if warmstart_path else None,
        going_out_shape=going_out_shape,
        run_dir_override=run_dir_override if run_dir_override else None,
    )
    print(result)
