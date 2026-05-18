from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


RUN_DIR = Path("ml/runs/dart_v5_baseline_400k")
CHECKPOINT_DIR = RUN_DIR / "checkpoints"
EVAL_DIR = RUN_DIR / "eval"
OPPONENTS = [
    "random",
    "greedy",
    "heuristic",
    "strategic",
    "xingdream",
    "lalala",
    "liuzha",
    "hulalala",
    "yaoji",
    "jidan",
    "ez",
    "wjsd",
]
N_GAMES = 1000
WORKERS = 8
LANES = 32
DEVICE = "cpu"
START_METHOD = "fork"


def _completed_rows(out_path: Path) -> int:
    if not out_path.exists():
        return 0
    try:
        results = json.loads(out_path.read_text()).get("results", {})
    except (OSError, json.JSONDecodeError):
        return 0
    return sum(
        int(results.get(opponent, {}).get("n_games", 0) or 0) >= N_GAMES
        for opponent in OPPONENTS
    )


def _acquire_lock(out_path: Path) -> Path | None:
    lock_path = out_path.with_suffix(out_path.suffix + ".lock")
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return None
    with os.fdopen(fd, "w") as f:
        f.write(f"pid={os.getpid()} started={time.time()}\n")
    return lock_path


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    return env


def main() -> None:
    checkpoints = sorted(CHECKPOINT_DIR.glob("update_*.pt"))
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    print(
        f"sweep_start run_dir={RUN_DIR} checkpoints={len(checkpoints)} "
        f"opponents={len(OPPONENTS)} games={N_GAMES} workers={WORKERS} "
        f"lanes={LANES} device={DEVICE}",
        flush=True,
    )
    for index, checkpoint in enumerate(checkpoints, start=1):
        out_path = EVAL_DIR / f"{checkpoint.stem}.json"
        completed = _completed_rows(out_path)
        if completed == len(OPPONENTS):
            print(
                f"[{index}/{len(checkpoints)}] skip {checkpoint.name} complete",
                flush=True,
            )
            continue

        lock_path = _acquire_lock(out_path)
        if lock_path is None:
            print(
                f"[{index}/{len(checkpoints)}] skip {checkpoint.name} locked",
                flush=True,
            )
            continue

        started = time.monotonic()
        print(
            f"[{index}/{len(checkpoints)}] run {checkpoint.name} "
            f"complete_rows={completed}/{len(OPPONENTS)} out={out_path}",
            flush=True,
        )
        cmd = [
            sys.executable,
            "-m",
            "guandan.scripts.eval.eval_dart",
            "--checkpoint",
            str(checkpoint),
            "--opponent",
            *OPPONENTS,
            "--games",
            str(N_GAMES),
            "--workers",
            str(WORKERS),
            "--lanes",
            str(LANES),
            "--device",
            DEVICE,
            "--start-method",
            START_METHOD,
            "--out",
            str(out_path),
        ]
        try:
            subprocess.run(cmd, check=True, env=_subprocess_env())
            elapsed = time.monotonic() - started
            print(
                f"[{index}/{len(checkpoints)}] done {checkpoint.name} "
                f"elapsed_s={elapsed:.1f}",
                flush=True,
            )
        except subprocess.CalledProcessError as exc:
            elapsed = time.monotonic() - started
            print(
                f"[{index}/{len(checkpoints)}] failed {checkpoint.name} "
                f"elapsed_s={elapsed:.1f} returncode={exc.returncode}",
                flush=True,
            )
        finally:
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass
    print("sweep_complete", flush=True)


if __name__ == "__main__":
    main()
