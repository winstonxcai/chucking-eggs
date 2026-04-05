"""Modal GPU launcher for GuanZero training — 4-network version (arXiv:2402.13582).

Usage:
    modal run ml/scripts/modal/guanzero.py --benchmark
    modal run --detach ml/scripts/modal/guanzero.py
"""

from __future__ import annotations

from pathlib import Path

import modal

app = modal.App("guanzero-train")

vol = modal.Volume.from_name("guandan-checkpoints", create_if_missing=True)
CHECKPOINT_DIR = "/checkpoints"

_root = Path(__file__).resolve().parent.parent.parent  # ml/

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
    timeout=3600 * 10,
    volumes={CHECKPOINT_DIR: vol},
)
def guanzero_remote(
    distill_games: int = 8000,
    selfplay_episodes: int = 150_000,
    batch_size: int = 512,
    eval_interval: int = 10_000,
    eval_games: int = 200,
    buffer_capacity: int = 50_000,
    benchmark: bool = False,
) -> str:
    import os
    import sys
    import time

    sys.path.insert(0, "/root/src")
    os.chdir(CHECKPOINT_DIR)

    import torch
    from guandan.cards import Rank
    from guandan.training.guanzero_selfplay import make_nets_buffers_optimizers

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    level_rank = Rank.TWO
    save_dir = Path(CHECKPOINT_DIR)

    # Create 4 nets, 4 buffers, 4 optimizers
    nets, buffers, optimizers = make_nets_buffers_optimizers(
        device, buffer_capacity=buffer_capacity, lr=3e-5
    )
    n_params = sum(p.numel() for net in nets for p in net.parameters())
    print(f"[GuanZero] 4 networks, {n_params:,} total params, device={device}")

    # ── Benchmark ─────────────────────────────────────────────────────────
    if benchmark:
        from guandan.game import GuanDanEnv
        from guandan.training.guanzero_selfplay import play_selfplay_episode, train_dmc_step

        env = GuanDanEnv(level_rank=level_rank)
        N = 100
        t0 = time.perf_counter()
        for i in range(N):
            transitions = play_selfplay_episode(nets, env, epsilon=0.1, device=device, level_rank=level_rank)
            for (seat, nh, hist, hl, a_enc, G) in transitions:
                buffers[seat].push(nh, hist, hl, a_enc, G)
            for seat in range(4):
                for _ in range(4):
                    train_dmc_step(nets[seat], optimizers[seat], buffers[seat], batch_size, device)
        elapsed = time.perf_counter() - t0
        sec_per_ep = elapsed / N
        lines = [
            f"[Benchmark] {N} episodes in {elapsed:.1f}s ({sec_per_ep:.2f}s/ep)",
            f"  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}",
        ]
        for target in [15_000, 50_000, 150_000]:
            hrs = sec_per_ep * target / 3600
            lines.append(f"  {target:,} episodes → {hrs:.1f} hrs")
        vol.commit()
        return "\n".join(lines)

    # ── Buffer prefill + pretrain from Jidan ──────────────────────────────
    from guandan.agents.jidan_bot import JidanBot
    from guandan.training.guanzero_distill import prefill_buffer_from_jidan

    print(f"[GuanZero] Pre-filling 4 seat buffers with {distill_games} Jidan games...")
    total = prefill_buffer_from_jidan(
        JidanBot(), buffers, distill_games, level_rank,
        nets=nets, optimizers=optimizers, device=device,
        batch_size=batch_size, train_steps_per_game=2,
    )
    print(f"[GuanZero] Prefill done: {total} transitions")
    vol.commit()

    # ── Self-play training ─────────────────────────────────────────────────
    prod_ckpt = save_dir / "prod_03_29_11_51.pt"
    from guandan.training.guanzero_selfplay import train_selfplay

    train_selfplay(
        nets=nets,
        buffers=buffers,
        optimizers=optimizers,
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
    return f"[GuanZero] Training complete. {selfplay_episodes} episodes."


@app.local_entrypoint()
def main(
    distill_games: int = 8000,
    selfplay_episodes: int = 150_000,
    batch_size: int = 512,
    eval_interval: int = 10_000,
    eval_games: int = 200,
    buffer_capacity: int = 50_000,
    benchmark: bool = False,
):
    """Launch GuanZero 4-network training on Modal A10G GPU."""
    result = guanzero_remote.remote(
        distill_games=distill_games,
        selfplay_episodes=selfplay_episodes,
        batch_size=batch_size,
        eval_interval=eval_interval,
        eval_games=eval_games,
        buffer_capacity=buffer_capacity,
        benchmark=benchmark,
    )
    print(result)
