"""Benchmark pvguan parallel rollout speedup on Modal.

Submits N short smoke runs in parallel — all on identical cpu=48 A10G
hardware — varying only `--rollout-workers`. Reports mean iter wall
time parsed from each run's metrics.jsonl.

Each smoke runs ~8 iters of 8192 dec/iter (~65k decisions) on A10G.
Validation disabled. Total cost: ~$1 across all variants.

Usage:
    modal run --detach ml/scripts/modal/bench_pvguan_rollout.py

Variants tested (workers): 1, 8, 16, 32, 48
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

app = modal.App("pvguan-bench-rollout")
vol = modal.Volume.from_name("pvguan-runs", create_if_missing=True)
RUN_VOL = "/runs"

_root = Path(__file__).resolve().parent.parent.parent.parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "tqdm", "matplotlib")
    .add_local_dir(str(_root / "ml" / "src"),     remote_path="/root/ml/src")
    .add_local_dir(str(_root / "ml" / "scripts"), remote_path="/root/ml/scripts")
)

WARMSTART_VOL_PATH = "/runs/warmstart/pvguan_distilled_jidan.pt"


@app.function(
    image=image, gpu="A10G", cpu=48, memory=32 * 1024, timeout=1800,
    volumes={RUN_VOL: vol},
)
def bench(n_workers: int, n_iters: int = 8) -> dict:
    """Run a smoke training and return per-iter timing stats."""
    import os
    import subprocess
    import time

    env = os.environ.copy()
    env["PYTHONPATH"] = "/root/ml/src"

    run_dir = f"{RUN_VOL}/bench_w{n_workers}"
    total_decisions = 8192 * n_iters
    cmd = [
        "python", "/root/ml/scripts/train/train_pvguan.py",
        "--critic", "ptie", "--seed", "0",
        "--total-decisions",   str(total_decisions),
        "--iter-decisions",    "8192",
        "--critic-warmup",     "0",
        "--kl-init",           "0.0",
        "--val-every",         "0",
        "--rollout-workers",   str(n_workers),
        "--snapshot-interval", str(10**12),  # disable mid-run snapshots
        "--run-dir",           run_dir,
        "--warmstart",         WARMSTART_VOL_PATH,
    ]

    print(f"[workers={n_workers}] launching: {' '.join(cmd)}")
    t0 = time.time()
    subprocess.run(cmd, env=env, cwd="/root", check=True)
    wall = time.time() - t0
    vol.commit()

    # Parse metrics.jsonl for per-iter timing
    metrics_path = Path(run_dir) / "metrics.jsonl"
    iter_walls:    list[float] = []
    rollout_walls: list[float] = []
    if metrics_path.exists():
        for line in metrics_path.read_text().splitlines():
            try:
                m = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "iter_wall_s" in m:
                iter_walls.append(float(m["iter_wall_s"]))
            if "rollout_wall_s" in m:
                rollout_walls.append(float(m["rollout_wall_s"]))

    # Skip iter 0 (includes pool spawn + first state_dict write)
    skip = 1 if len(iter_walls) > 1 else 0
    iw = iter_walls[skip:]
    rw = rollout_walls[skip:]
    return {
        "workers":           n_workers,
        "n_iters":           n_iters,
        "wall_s_total":      wall,
        "mean_iter_s":       sum(iw) / len(iw) if iw else None,
        "mean_rollout_s":    sum(rw) / len(rw) if rw else None,
        "decisions_per_s":   total_decisions / wall if wall > 0 else None,
        "iter_walls":        iter_walls,
        "rollout_walls":     rollout_walls,
    }


@app.local_entrypoint()
def main(n_iters: int = 8) -> None:
    """Submit all variants in parallel, collect results, print table."""
    worker_counts = [1, 8, 16, 32, 48]
    print(f"Launching {len(worker_counts)} benchmark variants in parallel "
          f"(n_iters={n_iters} each, cpu=48 each)…")

    futures = {w: bench.spawn(w, n_iters) for w in worker_counts}
    print("All variants spawned. Waiting for completion…\n")

    results: dict[int, dict] = {}
    for w, fc in futures.items():
        try:
            results[w] = fc.get()
            r = results[w]
            print(f"  ✓ workers={w:2d}: "
                  f"iter={r['mean_iter_s']:.2f}s  "
                  f"rollout={r['mean_rollout_s']:.2f}s  "
                  f"dec/s={r['decisions_per_s']:.0f}")
        except Exception as e:
            results[w] = {"error": str(e)}
            print(f"  ✗ workers={w:2d} failed: {e}")

    # Summary table
    print("\n" + "=" * 76)
    print(f"{'workers':>7}  {'iter_s':>8}  {'rollout_s':>10}  "
          f"{'dec/s':>8}  {'iter_speedup':>13}  {'rollout_speedup':>16}")
    print("-" * 76)
    base_iter = results.get(1, {}).get("mean_iter_s")
    base_roll = results.get(1, {}).get("mean_rollout_s")
    for w in worker_counts:
        r = results.get(w, {})
        if "error" in r:
            print(f"  {w:>5}  ERROR: {r['error']}")
            continue
        i_sp = (base_iter / r["mean_iter_s"])    if (base_iter and r.get("mean_iter_s")) else None
        r_sp = (base_roll / r["mean_rollout_s"]) if (base_roll and r.get("mean_rollout_s")) else None
        print(
            f"  {w:>5}  "
            f"{r.get('mean_iter_s', 0):>8.2f}  "
            f"{r.get('mean_rollout_s', 0):>10.2f}  "
            f"{r.get('decisions_per_s', 0):>8.0f}  "
            f"{(f'{i_sp:.2f}x' if i_sp else '—'):>13}  "
            f"{(f'{r_sp:.2f}x' if r_sp else '—'):>16}"
        )
    print("=" * 76)
