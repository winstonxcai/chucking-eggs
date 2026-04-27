"""Modal GPU launcher for pvguan PPO training.

Launch matrix (6 runs: 3 seeds × 2 ablations):
    for critic in pv ptie; do
      for seed in 0 1 2; do
        modal run ml/scripts/modal/train_pvguan_modal.py \\
          --critic $critic --seed $seed --detach
      done
    done

Each run trains to 6M decisions (~75 min on A10G). Use --detach for all 6.
"""

from __future__ import annotations

from pathlib import Path

import modal

app  = modal.App("pvguan-train")
vol  = modal.Volume.from_name("pvguan-checkpoints", create_if_missing=True)
CKPT_DIR = "/checkpoints"

_root = Path(__file__).resolve().parent.parent.parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "build-essential")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
    )
    .pip_install("torch", "numpy", "maturin", "tqdm")
    .add_local_dir(
        str(_root / "src" / "guandan_rs"),
        remote_path="/root/guandan_rs",
        copy=True,
    )
    .run_commands(
        "bash -c 'source $HOME/.cargo/env && "
        "cd /root/guandan_rs && maturin build --release --interpreter python3.11'",
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
    cpu=16,
    timeout=3600 * 6,   # 6-hour hard cap
    volumes={CKPT_DIR: vol},
)
def train_remote(
    critic:               str,
    seed:                 int,
    warmstart:            str,
    total_decisions:      int,
    iter_decisions:       int,
    temperature:          float,
    kl_init:              float,
    kl_decay_actor_iters: int,
    critic_warmup:        int,
    hidden:               int,
) -> str:
    import os, sys
    sys.path.insert(0, "/root/src")
    import argparse
    from guandan.scripts.train.train_pvguan import main  # noqa: F401

    # Build synthetic argv
    argv = [
        "train_pvguan",
        "--critic", critic,
        "--seed", str(seed),
        "--total-decisions", str(total_decisions),
        "--iter-decisions", str(iter_decisions),
        "--temperature", str(temperature),
        "--kl-init", str(kl_init),
        "--kl-decay-actor-iters", str(kl_decay_actor_iters),
        "--critic-warmup", str(critic_warmup),
        "--hidden", str(hidden),
        "--run-dir", f"{CKPT_DIR}/pvguan_{critic}_seed{seed}",
    ]
    if warmstart:
        argv += ["--warmstart", warmstart]

    sys.argv = argv
    main()
    vol.commit()
    return f"done: critic={critic} seed={seed} decisions={total_decisions}"


@app.local_entrypoint()
def main(
    critic:                str  = "pv",
    seed:                  int  = 0,
    warmstart:             str  = "",
    total_decisions:       int  = 6_000_000,
    iter_decisions:        int  = 4096,
    temperature:           float = 1.0,
    kl_init:               float = 0.05,
    kl_decay_actor_iters:  int  = 100,
    critic_warmup:         int  = 30,
    hidden:                int  = 256,
) -> None:
    result = train_remote.remote(
        critic=critic,
        seed=seed,
        warmstart=warmstart,
        total_decisions=total_decisions,
        iter_decisions=iter_decisions,
        temperature=temperature,
        kl_init=kl_init,
        kl_decay_actor_iters=kl_decay_actor_iters,
        critic_warmup=critic_warmup,
        hidden=hidden,
    )
    print(result)
