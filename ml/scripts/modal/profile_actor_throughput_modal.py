"""Modal CPU profile for Dart actor rollout throughput."""

from __future__ import annotations

from pathlib import Path

import modal

app = modal.App("dart-actor-profile")

_root = Path(__file__).resolve().parent.parent.parent.parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "gcc", "libssl-dev", "pkg-config", "build-essential")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable",
    )
    .pip_install("torch", "numpy", "pyyaml", "tqdm", "maturin==1.7.0")
    .add_local_dir(str(_root / "ml" / "src"), remote_path="/root/ml/src", copy=True)
    .add_local_dir(str(_root / "ml" / "scripts"), remote_path="/root/ml/scripts", copy=True)
    .run_commands(
        ". $HOME/.cargo/env && cd /root/ml/src/guandan_rs"
        " && maturin build --release -i python3"
        " && pip install target/wheels/*manylinux*.whl",
    )
)


@app.function(
    image=image,
    cpu=32,
    memory=20 * 1024,
    timeout=3600,
)
def profile_remote(
    workers: int,
    episodes: int,
    lanes: int,
    epsilon: float,
    max_forced_pass_replay_frac: float,
    threads: int,
    seed_base: int,
) -> str:
    return _run_profile_subprocess(
        workers=workers,
        episodes=episodes,
        lanes=lanes,
        epsilon=epsilon,
        max_forced_pass_replay_frac=max_forced_pass_replay_frac,
        threads=threads,
        seed_base=seed_base,
    )


@app.function(
    image=image,
    gpu="A10G",
    cpu=32,
    memory=20 * 1024,
    timeout=3600,
)
def profile_remote_a10g(
    workers: int,
    episodes: int,
    lanes: int,
    epsilon: float,
    max_forced_pass_replay_frac: float,
    threads: int,
    seed_base: int,
) -> str:
    return _run_profile_subprocess(
        workers=workers,
        episodes=episodes,
        lanes=lanes,
        epsilon=epsilon,
        max_forced_pass_replay_frac=max_forced_pass_replay_frac,
        threads=threads,
        seed_base=seed_base,
    )


def _run_profile_subprocess(
    *,
    workers: int,
    episodes: int,
    lanes: int,
    epsilon: float,
    max_forced_pass_replay_frac: float,
    threads: int,
    seed_base: int,
) -> str:
    import os
    import subprocess

    env = os.environ.copy()
    env["PYTHONPATH"] = "/root/ml/src:" + env.get("PYTHONPATH", "")
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"

    cmd = [
        "python",
        "/root/ml/scripts/util/profile_actor_throughput.py",
        "--workers", str(workers),
        "--episodes", str(episodes),
        "--lanes", str(lanes),
        "--threads", str(threads),
        "--epsilon", str(epsilon),
        "--max-forced-pass-replay-frac", str(max_forced_pass_replay_frac),
        "--seed-base", str(seed_base),
    ]
    print("Running:", " ".join(cmd), flush=True)
    proc = subprocess.run(
        cmd,
        env=env,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(proc.stdout, end="", flush=True)
    return proc.stdout


@app.local_entrypoint()
def main(
    workers: int = 32,
    episodes: int = 64,
    lanes: int = 40,
    epsilon: float = 0.01,
    max_forced_pass_replay_frac: float = 0.0,
    threads: int = 1,
    seed_base: int = 10_000,
    gpu: str = "",
) -> None:
    fn = profile_remote_a10g if gpu.upper() == "A10G" else profile_remote
    fn.remote(
        workers=workers,
        episodes=episodes,
        lanes=lanes,
        epsilon=epsilon,
        max_forced_pass_replay_frac=max_forced_pass_replay_frac,
        threads=threads,
        seed_base=seed_base,
    )
