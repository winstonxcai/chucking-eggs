"""Modal CPU benchmark for Dart actor rollout throughput.

Runs ``ml/scripts/util/bench_actor_throughput.py`` on a 32-vCPU Modal worker.

Example:
    modal run ml/scripts/modal/bench_actor_throughput_modal.py \\
        --workers 32 --episodes 16 --lanes 1,8,16,24,32,40
"""

from __future__ import annotations

from pathlib import Path

import modal

app = modal.App("dart-actor-bench")

_root = Path(__file__).resolve().parent.parent.parent.parent  # repo root

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
        " && pip install target/wheels/*.whl",
    )
)


@app.function(
    image=image,
    cpu=32,
    memory=20 * 1024,
    timeout=3600,
)
def bench_remote(
    workers: int,
    episodes: int,
    lanes: list[int],
    epsilon: float,
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
        "/root/ml/scripts/util/bench_actor_throughput.py",
        "--workers", str(workers),
        "--episodes", str(episodes),
        "--threads", str(threads),
        "--epsilon", str(epsilon),
        "--seed-base", str(seed_base),
        "--lanes", *[str(x) for x in lanes],
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
    episodes: int = 16,
    lanes: str = "1,8,16,24,32,40",
    epsilon: float = 0.0,
    threads: int = 1,
    seed_base: int = 10_000,
) -> None:
    lane_values = [int(x.strip()) for x in lanes.split(",") if x.strip()]
    bench_remote.remote(
        workers=workers,
        episodes=episodes,
        lanes=lane_values,
        epsilon=epsilon,
        threads=threads,
        seed_base=seed_base,
    )
