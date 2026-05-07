"""BLAS-thread sweep for the distributed run, with n_actors fixed.

With n_actors=1 there is no inter-actor CPU contention, so the question is:
how many threads should the learner's numpy / torch-CPU ops be allowed to
use? Each setting maps to OMP_NUM_THREADS / MKL_NUM_THREADS /
VECLIB_MAXIMUM_THREADS / OPENBLAS_NUM_THREADS, all set to the same value
before subprocess launch so torch picks them up at import time.

The actor process still calls ``torch.set_num_threads(1)`` internally, so
this sweep mostly affects the learner + numpy paths.

Usage:
    PYTHONPATH=ml/src .venv/bin/python ml/scripts/util/sweep_distributed_threads.py \\
        --threads-list 1,2,4,8,default --updates 300 --n-actors 1
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def make_temp_config(base_path: Path, overrides: dict, out_path: Path) -> None:
    cfg = yaml.safe_load(base_path.read_text()) or {}
    cfg.update(overrides)
    out_path.write_text(yaml.safe_dump(cfg, sort_keys=False))


def run_one(threads_label: str, args, sweep_root: Path) -> dict:
    run_dir = sweep_root / f"t_{threads_label}"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = run_dir / "config_sweep.yaml"
    overrides = {
        "n_actors": args.n_actors,
        "buffer_min_size": args.buffer_min_size,
        "log_every_updates": args.log_every,
        "publish_interval_updates": max(args.log_every, 50),
        "checkpoint_every_updates": args.updates + 1,
        "total_updates_target": args.updates,
        "device": args.device,
    }
    make_temp_config(Path(args.config), overrides, cfg_path)

    cmd = [
        sys.executable, "-m", "guandan.guanzero.train",
        "--config",  str(cfg_path),
        "--run-dir", str(run_dir),
        "--updates", str(args.updates),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "ml" / "src") + ":" + env.get("PYTHONPATH", "")
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    if threads_label == "default":
        # leave env vars as-is (whatever the OS / torch default)
        for v in THREAD_ENV_VARS:
            env.pop(v, None)
    else:
        for v in THREAD_ENV_VARS:
            env[v] = str(threads_label)

    print(f"\n--- threads={threads_label} ---")
    pairs = ", ".join(f"{v}={env.get(v, '-')}" for v in THREAD_ENV_VARS)
    print(f"    env: {{ {pairs} }}")
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, env=env, cwd=REPO_ROOT,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    wall = time.perf_counter() - t0
    if proc.returncode != 0:
        print(proc.stdout[-2000:])
        return {"threads": threads_label, "wall": wall, "ok": False}

    metrics_path = run_dir / "metrics_learner.jsonl"
    rows = []
    if metrics_path.exists():
        for line in metrics_path.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass

    out = {"threads": threads_label, "wall": wall, "ok": True, "n_rows": len(rows)}
    if len(rows) >= 2:
        first, last = rows[0], rows[-1]
        d_upd = last["updates"] - first["updates"]
        d_t   = last["elapsed_s"] - first["elapsed_s"]
        out["steady_upd_s"]   = d_upd / d_t if d_t > 0 else 0.0
        out["last_upd_s"]     = last.get("upd_per_sec", 0.0)
        out["last_q_depth"]   = last.get("queue_depth", -1)
        out["last_drained"]   = last.get("drained_since_last_log", -1)
        out["target_updates"] = last["updates"]
        out["warmup_s"]       = first["elapsed_s"]
    else:
        out["steady_upd_s"] = 0.0
        out["last_upd_s"]   = 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="ml/src/guandan/guanzero/config/m0_faithful_distributed.yaml")
    ap.add_argument("--threads-list", default="1,2,4,8,default")
    ap.add_argument("--n-actors", type=int, default=1)
    ap.add_argument("--updates", type=int, default=300)
    ap.add_argument("--buffer-min-size", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out-dir", default="ml/runs/sweep_threads")
    args = ap.parse_args()

    sweep_root = REPO_ROOT / args.out_dir
    sweep_root.mkdir(parents=True, exist_ok=True)

    labels = [s.strip() for s in args.threads_list.split(",")]
    results = []
    for t in labels:
        results.append(run_one(t, args, sweep_root))

    print("\n=== SUMMARY ===")
    print(f"{'threads':>8s} {'wall(s)':>8s} {'rows':>5s} {'warmup(s)':>10s} "
          f"{'steady upd/s':>13s} {'last upd/s':>11s} {'drained':>8s}")
    for r in results:
        if not r.get("ok"):
            print(f"{str(r['threads']):>8s} {r['wall']:>8.1f}  FAILED")
            continue
        print(f"{str(r['threads']):>8s} {r['wall']:>8.1f} {r['n_rows']:>5d} "
              f"{r.get('warmup_s', 0):>10.1f} "
              f"{r.get('steady_upd_s', 0):>13.2f} "
              f"{r.get('last_upd_s', 0):>11.2f} "
              f"{r.get('last_drained', -1):>8d}")

    summary_path = sweep_root / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {summary_path}")


if __name__ == "__main__":
    main()
