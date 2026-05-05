"""End-to-end distributed n_actors sweep on local M1.

Spawns the real ``train_distributed`` orchestrator (paper-spec net, MPS
learner, CPU actors per actor.py) at varying ``n_actors``, each for a fixed
update target, and reports steady-state upd/s read from the learner's
``metrics_learner.jsonl``.

Why not just use ``bench_actor_scaling.py``? That measures actor-only ep/s,
not the full pipeline. The end-to-end upd/s depends on actor → queue →
buffer → learner balance, weight publish/sync overhead, and per-process MPS
contention — all of which only the real orchestrator exposes.

Steady-state upd/s = (last_row.updates - first_row.updates) /
                     (last_row.elapsed - first_row.elapsed)
which excludes the buffer warmup window (only logged rows are post-warmup).

Usage:
    PYTHONPATH=ml/src .venv/bin/python ml/scripts/util/sweep_distributed_n_actors.py \\
        --n-list 1,2,4,6,8 --updates 300
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


def make_temp_config(base_path: Path, overrides: dict, out_path: Path) -> None:
    cfg = yaml.safe_load(base_path.read_text()) or {}
    cfg.update(overrides)
    out_path.write_text(yaml.safe_dump(cfg, sort_keys=False))


def run_one(N: int, args, sweep_root: Path) -> dict:
    run_dir = sweep_root / f"n{N}"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = run_dir / "config_sweep.yaml"
    overrides = {
        "n_actors": N,
        "buffer_min_size": args.buffer_min_size,
        "log_every_updates": args.log_every,
        "publish_interval_updates": max(args.log_every, 50),
        "checkpoint_every_updates": args.updates + 1,   # never checkpoint mid-sweep
        "total_updates_target": args.updates,
        "device": args.device,
    }
    make_temp_config(Path(args.config), overrides, cfg_path)

    cmd = [
        sys.executable, "-m", "guandan.guanzero.train_distributed",
        "--config",  str(cfg_path),
        "--run-dir", str(run_dir),
        "--updates", str(args.updates),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "ml" / "src") + ":" + env.get("PYTHONPATH", "")
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    print(f"\n--- N={N} ---  cmd: {' '.join(cmd)}")
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, env=env, cwd=REPO_ROOT,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True)
    wall = time.perf_counter() - t0
    if proc.returncode != 0:
        print(proc.stdout[-2000:])
        return {"N": N, "wall": wall, "ok": False}

    # Parse metrics — steady-state upd/s from row diff, not last-row average
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

    out = {"N": N, "wall": wall, "ok": True, "n_rows": len(rows)}
    if len(rows) >= 2:
        first, last = rows[0], rows[-1]
        d_upd = last["updates"] - first["updates"]
        d_t   = last["elapsed_s"] - first["elapsed_s"]
        out["steady_upd_s"]   = d_upd / d_t if d_t > 0 else 0.0
        out["last_upd_s"]     = last.get("upd_per_sec", 0.0)
        out["last_q_depth"]   = last.get("queue_depth", -1)
        out["last_drained"]   = last.get("drained_since_last_log", -1)
        out["target_updates"] = last["updates"]
        out["warmup_s"]       = first["elapsed_s"]   # rough warmup proxy
    else:
        out["steady_upd_s"]   = 0.0
        out["last_upd_s"]     = 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="ml/src/guandan/guanzero/config/m0_faithful_distributed.yaml")
    ap.add_argument("--n-list", default="1,2,4,6,8")
    ap.add_argument("--updates", type=int, default=300)
    ap.add_argument("--buffer-min-size", type=int, default=500,
                    help="Lower than prod (5000) so warmup is short and fair across N.")
    ap.add_argument("--log-every", type=int, default=50,
                    help="Smaller than prod (200) so we get >2 metric rows per run.")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out-dir", default="ml/runs/sweep_distributed")
    args = ap.parse_args()

    sweep_root = REPO_ROOT / args.out_dir
    sweep_root.mkdir(parents=True, exist_ok=True)

    Ns = [int(s) for s in args.n_list.split(",")]
    results = []
    for N in Ns:
        results.append(run_one(N, args, sweep_root))

    print("\n=== SUMMARY ===")
    print(f"{'N':>3s} {'wall(s)':>8s} {'rows':>5s} {'warmup(s)':>10s} "
          f"{'steady upd/s':>13s} {'last upd/s':>11s} {'q_depth':>8s} {'drained':>8s}")
    for r in results:
        if not r.get("ok"):
            print(f"{r['N']:>3d} {r['wall']:>8.1f}  FAILED")
            continue
        print(f"{r['N']:>3d} {r['wall']:>8.1f} {r['n_rows']:>5d} "
              f"{r.get('warmup_s', 0):>10.1f} "
              f"{r.get('steady_upd_s', 0):>13.2f} "
              f"{r.get('last_upd_s', 0):>11.2f} "
              f"{r.get('last_q_depth', -1):>8d} "
              f"{r.get('last_drained', -1):>8d}")

    summary_path = sweep_root / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {summary_path}")


if __name__ == "__main__":
    main()
