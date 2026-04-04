"""Modal GPU launcher for GuanZero training (arXiv:2402.13582).

Usage:
    # Benchmark run — measure speed, stops before full training:
    modal run ml/scripts/modal/guanzero.py --benchmark

    # Full training (run after benchmark approval):
    modal run ml/scripts/modal/guanzero.py --detach
    modal run ml/scripts/modal/guanzero.py --distill-games 8000 --selfplay-episodes 100000 --detach
"""

from __future__ import annotations

from pathlib import Path

import modal

app = modal.App("guanzero-train")

vol = modal.Volume.from_name("guandan-checkpoints", create_if_missing=True)
CHECKPOINT_DIR = "/checkpoints"

_root = Path(__file__).resolve().parent.parent.parent.parent  # chucking-eggs/

# Same image as dmc.py: Rust movegen + torch + numpy
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("curl", "build-essential")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
    )
    .pip_install("torch", "numpy", "maturin")
    .add_local_dir(str(_root / "src" / "guandan_rs"), remote_path="/root/guandan_rs", copy=True)
    .run_commands(
        "bash -c 'source $HOME/.cargo/env && cd /root/guandan_rs && maturin build --release --interpreter python3.10'",
        "pip install /root/guandan_rs/target/wheels/guandan_rs-*.whl",
    )
    .add_local_dir(str(_root / "src"), remote_path="/root/src")
)


@app.function(
    image=image,
    gpu="A10G",
    timeout=3600 * 8,
    volumes={CHECKPOINT_DIR: vol},
)
def guanzero_remote(
    distill_games: int = 8000,
    distill_epochs: int = 3,
    selfplay_episodes: int = 100_000,
    batch_size: int = 512,
    eval_interval: int = 10_000,
    eval_games: int = 200,
    benchmark: bool = False,
) -> str:
    import os
    import sys
    import time

    sys.path.insert(0, "/root/src")
    os.chdir(CHECKPOINT_DIR)

    import torch
    from guandan.cards import Rank
    from guandan.training.guanzero_network import GuanZeroNetwork
    from guandan.training.guanzero_selfplay import ReplayBuffer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = GuanZeroNetwork().to(device)
    level_rank = Rank.TWO
    save_dir = Path(CHECKPOINT_DIR)
    buffer = ReplayBuffer(capacity=200_000)

    # ── Buffer prefill from Jidan (MSE-compatible) ───────────────────────
    from guandan.agents.jidan_bot import JidanBot
    from guandan.training.guanzero_distill import prefill_buffer_from_jidan

    if not benchmark:
        print(f"[GuanZero] Pre-filling buffer with {distill_games} Jidan games...")
        total = prefill_buffer_from_jidan(JidanBot(), buffer, distill_games, level_rank)
        print(f"[GuanZero] Buffer prefilled: {total} transitions, buf_size={len(buffer)}")
        vol.commit()

    if benchmark:
        # Time 100 episodes and extrapolate
        from guandan.game import GuanDanEnv
        from guandan.training.guanzero_selfplay import play_selfplay_episode, train_dmc_step
        import torch.optim as optim

        optimizer = optim.Adam(net.parameters(), lr=3e-5)
        env = GuanDanEnv(level_rank=level_rank)
        N = 100
        t0 = time.perf_counter()
        for i in range(N):
            transitions = play_selfplay_episode(net, env, epsilon=0.1, device=device, level_rank=level_rank)
            for nh, hist, hl, a_enc, G in transitions:
                buffer.push(nh, hist, hl, a_enc, G)
            for _ in range(4):
                train_dmc_step(net, optimizer, buffer, batch_size, device)
        elapsed = time.perf_counter() - t0
        sec_per_ep = elapsed / N

        lines = [
            f"[Benchmark] {N} episodes in {elapsed:.1f}s ({sec_per_ep:.2f}s/ep)",
            f"  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}",
        ]
        for target in [15_000, 50_000, 100_000]:
            hrs = sec_per_ep * target / 3600
            lines.append(f"  {target:,} episodes → {hrs:.1f} hrs")
        vol.commit()
        return "\n".join(lines)

    # Full self-play training
    prod_ckpt = save_dir / "prod_03_29_11_51.pt"  # may not exist on Modal vol
    from guandan.training.guanzero_selfplay import train_selfplay

    train_selfplay(
        net=net,
        buffer=buffer,
        device=device,
        level_rank=level_rank,
        total_episodes=selfplay_episodes,
        batch_size=batch_size,
        eval_interval=eval_interval,
        eval_games=eval_games,
        save_dir=save_dir,
        prod_ckpt_path=prod_ckpt if prod_ckpt.exists() else None,
    )
    vol.commit()
    return f"[GuanZero] Training complete. {selfplay_episodes} episodes. Checkpoints in volume."


@app.local_entrypoint()
def main(
    distill_games: int = 8000,
    distill_epochs: int = 3,
    selfplay_episodes: int = 150_000,
    batch_size: int = 512,
    eval_interval: int = 10_000,
    eval_games: int = 200,
    benchmark: bool = False,
):
    """Launch GuanZero training on Modal A10G GPU.

    Distillation is skipped if /checkpoints/jidan_distill.pt already exists in the volume.
    Pass --benchmark to measure speed and extrapolate runtime before full training.
    """
    result = guanzero_remote.remote(
        distill_games=distill_games,
        distill_epochs=distill_epochs,
        selfplay_episodes=selfplay_episodes,
        batch_size=batch_size,
        eval_interval=eval_interval,
        eval_games=eval_games,
        benchmark=benchmark,
    )
    print(result)
