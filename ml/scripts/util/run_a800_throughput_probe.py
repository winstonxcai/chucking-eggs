"""Run one warmup plus one measured A800 CPU/actor/lane throughput probe."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


GIB = 1024**3
RESOURCE_SAMPLE_INTERVAL_S = 5.0
_STOP_REQUESTED = threading.Event()
_ACTIVE_PROCESS: subprocess.Popen[str] | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _read_resources() -> dict[str, float]:
    row: dict[str, float] = {}
    try:
        values = {
            line.split(":", 1)[0]: int(line.split()[1]) * 1024
            for line in Path("/proc/meminfo").read_text().splitlines()
            if line.startswith(("MemTotal:", "MemAvailable:", "SwapTotal:", "SwapFree:"))
        }
        total = values["MemTotal"]
        row["host_ram_used_gib"] = (total - values["MemAvailable"]) / GIB
        row["host_mem_available_fraction"] = values["MemAvailable"] / total
        row["swap_used_gib"] = (values["SwapTotal"] - values["SwapFree"]) / GIB
    except (KeyError, OSError, ValueError):
        pass
    try:
        gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",", 1)[0]
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={gpu_id}",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
        gpu_util, memory_used, memory_total = [
            float(value.strip()) for value in result.stdout.splitlines()[0].split(",")
        ]
        row.update(
            gpu_util_pct=gpu_util,
            gpu_memory_used_mib=memory_used,
            gpu_memory_total_mib=memory_total,
        )
    except (FileNotFoundError, IndexError, OSError, subprocess.SubprocessError, ValueError):
        pass
    return row


def _sample_resources(path: Path, stop: threading.Event) -> None:
    with path.open("a", buffering=1) as handle:
        while not stop.is_set():
            row: dict[str, Any] = {"timestamp": _utc_now(), "epoch_s": time.time()}
            row.update(_read_resources())
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
            stop.wait(RESOURCE_SAMPLE_INTERVAL_S)


def _command(args: argparse.Namespace, run_dir: Path, updates: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "guandan.dart",
        "--config",
        str(args.config),
        "--n-actors",
        str(args.actors),
        "--actor-batch-lanes",
        str(args.lanes),
        "--updates",
        str(updates),
        "--seed",
        str(args.seed),
        "--run-dir",
        str(run_dir),
    ]


def _run_training(
    args: argparse.Namespace,
    *,
    run_dir: Path,
    scratch_root: Path,
    updates: int,
    sample_resources: bool,
) -> int:
    global _ACTIVE_PROCESS
    run_dir.mkdir(parents=True, exist_ok=True)
    weight_dir = scratch_root / run_dir.name / "weights"
    weight_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        DART_WEIGHT_DIR=str(weight_dir),
        DART_TQDM="0",
        DART_STREAM_LOGS="1",
        PYTHONUNBUFFERED="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
    )
    command = _command(args, run_dir, updates)
    _write_json(
        run_dir / "probe_command.json",
        {"command": command, "started_at": _utc_now(), "target_updates": updates},
    )
    stop = threading.Event()
    sampler: threading.Thread | None = None
    if sample_resources:
        sampler = threading.Thread(
            target=_sample_resources,
            args=(run_dir / "resources.jsonl", stop),
            daemon=True,
        )
        sampler.start()
    try:
        _ACTIVE_PROCESS = subprocess.Popen(command, env=env, text=True)
        return _ACTIVE_PROCESS.wait()
    finally:
        _ACTIVE_PROCESS = None
        stop.set()
        if sampler is not None:
            sampler.join(timeout=RESOURCE_SAMPLE_INTERVAL_S + 1)


def _last_update(run_dir: Path) -> int:
    metrics = run_dir / "metrics_learner.jsonl"
    if not metrics.exists():
        return 0
    for line in reversed(metrics.read_text().splitlines()):
        if line.strip():
            return int(json.loads(line).get("updates", 0))
    return 0


def _prune_final_checkpoint(run_dir: Path) -> None:
    checkpoint = run_dir / "checkpoints" / "final.pt"
    if checkpoint.is_file():
        checkpoint.unlink()


def run(args: argparse.Namespace) -> None:
    global _ACTIVE_PROCESS
    args.config = args.config.resolve()
    output_root = args.output_root.resolve()
    scratch_root = args.scratch_root.resolve()
    if not args.config.is_file():
        raise FileNotFoundError(args.config)
    output_root.mkdir(parents=True, exist_ok=True)
    scratch_root.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_root / "manifest.json",
        {
            "created_at": _utc_now(),
            "config": str(args.config),
            "actors": args.actors,
            "lanes": args.lanes,
            "warmup_updates": args.warmup_updates,
            "probe_updates": args.updates,
            "seed": args.seed,
            "stage": args.stage,
        },
    )

    if args.warmup_updates:
        warmup_dir = output_root / "warmup"
        warmup_code = _run_training(
            args,
            run_dir=warmup_dir,
            scratch_root=scratch_root,
            updates=args.warmup_updates,
            sample_resources=False,
        )
        if warmup_code != 0 or _last_update(warmup_dir) < args.warmup_updates:
            raise RuntimeError(f"warmup failed: exit={warmup_code}, dir={warmup_dir}")
        _prune_final_checkpoint(warmup_dir)

    probe_dir = output_root / "probe"
    probe_dir.mkdir(parents=True, exist_ok=True)
    status = {
        "name": args.name,
        "stage": args.stage,
        "actors": args.actors,
        "lanes": args.lanes,
        "seed": args.seed,
        "target_updates": args.updates,
        "started_at": _utc_now(),
        "completed": False,
    }
    _write_json(probe_dir / "sweep_status.json", status)
    code = _run_training(
        args,
        run_dir=probe_dir,
        scratch_root=scratch_root,
        updates=args.updates,
        sample_resources=True,
    )
    final_updates = _last_update(probe_dir)
    status.update(
        ended_at=_utc_now(),
        returncode=code,
        final_updates=final_updates,
        completed=(code == 0 and final_updates >= args.updates),
    )
    _write_json(probe_dir / "sweep_status.json", status)
    if status["completed"]:
        _prune_final_checkpoint(probe_dir)
        print(f"complete {args.name}: {final_updates} updates", flush=True)
        return
    raise RuntimeError(f"probe failed: exit={code}, updates={final_updates}/{args.updates}")


def _install_signal_handlers() -> None:
    def _forward(signum: int, _frame: Any) -> None:
        _STOP_REQUESTED.set()
        if _ACTIVE_PROCESS is not None and _ACTIVE_PROCESS.poll() is None:
            _ACTIVE_PROCESS.send_signal(signum)

    signal.signal(signal.SIGINT, _forward)
    signal.signal(signal.SIGTERM, _forward)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("ml/src/guandan/dart/configs/dart_a800_throughput_sweep.yaml"))
    parser.add_argument("--name", required=True)
    parser.add_argument("--stage", default="cpu_scaling")
    parser.add_argument("--actors", type=int, required=True)
    parser.add_argument("--lanes", type=int, required=True)
    parser.add_argument("--updates", type=int, default=10_000)
    parser.add_argument("--warmup-updates", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    _install_signal_handlers()
    run(parse_args())


if __name__ == "__main__":
    main()
