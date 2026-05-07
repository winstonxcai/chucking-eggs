"""Modal launcher to run profile_actor_forward.py on x86 CPU + A10G CUDA.

Lets us compare M1 ARM CPU vs Modal x86 CPU for paper-spec actor forward,
since torch.compile / Inductor have different code paths per platform.

Usage:
    modal run --detach ml/scripts/modal/profile_actor_forward_modal.py \\
        --device cpu --k 5 --compile
    modal run --detach ml/scripts/modal/profile_actor_forward_modal.py \\
        --device cuda --k 5 --compile
"""

from __future__ import annotations

from pathlib import Path

import modal


app = modal.App("guanzero-profile-actor-forward")

_root = Path(__file__).resolve().parent.parent.parent.parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "pyyaml", "tqdm")
    .add_local_dir(str(_root / "ml" / "src"),     remote_path="/root/ml/src")
    .add_local_dir(str(_root / "ml" / "scripts"), remote_path="/root/ml/scripts")
)


@app.function(image=image, gpu="A10G", cpu=4, memory=8 * 1024, timeout=1800)
def profile_remote(
    device:       str,
    K:            int,
    K_dist:       str,
    warmup:       int,
    repeats:      int,
    compile_net:  bool,
    threads:      int,
    hidden_lstm:  int,
    hidden_mlp:   int,
    n_mlp_layers: int,
    row_limit:    int,
) -> str:
    import os
    import subprocess
    import sys

    env = os.environ.copy()
    env["PYTHONPATH"] = "/root/ml/src:" + env.get("PYTHONPATH", "")

    cmd = [
        "python", "/root/ml/scripts/util/profile_actor_forward.py",
        "--device",       device,
        "--threads",      str(threads),
        "--hidden-lstm",  str(hidden_lstm),
        "--hidden-mlp",   str(hidden_mlp),
        "--n-mlp-layers", str(n_mlp_layers),
        "--K",            str(K),
        "--K-dist",       K_dist,
        "--warmup",       str(warmup),
        "--repeats",      str(repeats),
        "--row-limit",    str(row_limit),
    ]
    if compile_net:
        cmd.append("--compile")

    print("running:", " ".join(cmd))
    out = subprocess.run(cmd, env=env, capture_output=True, text=True)
    print(out.stdout)
    if out.returncode != 0:
        print("STDERR:", out.stderr)
    return out.stdout


@app.local_entrypoint()
def main(
    device:       str  = "cpu",
    k:            int  = 5,
    k_dist:       str  = "fixed",
    warmup:       int  = 50,
    repeats:      int  = 500,
    compile:      bool = False,
    threads:      int  = 1,
    hidden_lstm:  int  = 256,
    hidden_mlp:   int  = 1024,
    n_mlp_layers: int  = 6,
    row_limit:    int  = 25,
) -> None:
    out = profile_remote.remote(
        device=device, K=k, K_dist=k_dist,
        warmup=warmup, repeats=repeats, compile_net=compile,
        threads=threads, hidden_lstm=hidden_lstm,
        hidden_mlp=hidden_mlp, n_mlp_layers=n_mlp_layers,
        row_limit=row_limit,
    )
    print("=== modal output (first 200 lines) ===")
    print("\n".join(out.splitlines()[:200]))
