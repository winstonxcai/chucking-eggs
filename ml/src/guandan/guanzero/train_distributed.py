"""Faithful persistent actor-learner DMC orchestrator.

Spawns N CPU actor processes and 1 learner process communicating via a
bounded multiprocessing.Queue. Actors run episodes continuously; the
learner drains samples, updates global Q-nets, and publishes weights
atomically to disk. Actors sync periodically from disk.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .actor import actor_loop
from .learner import learner_loop
from .train import TrainConfig, _setup_logging


def _wait_for_weights(weight_dir: Path, timeout: float = 30.0) -> None:
    """Block until the learner publishes initial weights."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if (weight_dir / "latest.txt").exists():
            return
        time.sleep(0.2)
    raise TimeoutError(f"Learner did not publish weights within {timeout}s")


def _read_update_count(run_dir: Path) -> int:
    metrics = run_dir / "metrics_learner.jsonl"
    if not metrics.exists():
        return 0
    last = None
    with metrics.open() as f:
        for line in f:
            line = line.strip()
            if line:
                last = line
    if last is None:
        return 0
    try:
        return json.loads(last).get("updates", 0)
    except Exception:
        return 0


def train_distributed(cfg: TrainConfig) -> None:
    import multiprocessing as mp

    run_dir    = Path(cfg.resolved_run_dir)
    weight_dir = run_dir / "weights"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2))

    logger, log_path = _setup_logging(run_dir)
    logger.info("train_distributed: %d actors, target %d updates", cfg.n_actors, cfg.checkpoint_every_updates)

    ctx          = mp.get_context("spawn")
    sample_queue = ctx.Queue(maxsize=cfg.sample_queue_maxsize)
    stop_event   = ctx.Event()
    cfg_dict     = dataclasses.asdict(cfg)

    learner_proc = ctx.Process(
        target=learner_loop,
        args=(cfg_dict, sample_queue, stop_event, weight_dir, run_dir),
        daemon=True,
        name="learner",
    )
    learner_proc.start()

    tqdm.write(f"  Learner started (pid={learner_proc.pid})")
    tqdm.write(f"  Waiting for initial weights → {weight_dir}")
    _wait_for_weights(weight_dir, timeout=60)
    tqdm.write("  Initial weights ready — starting actors")

    actor_procs = []
    for actor_id in range(cfg.n_actors):
        p = ctx.Process(
            target=actor_loop,
            args=(actor_id, cfg_dict, sample_queue, stop_event, weight_dir),
            daemon=True,
            name=f"actor-{actor_id}",
        )
        p.start()
        actor_procs.append(p)

    sep = "=" * 68
    tqdm.write(sep)
    tqdm.write(f"  GuanZero distributed  |  {cfg.n_actors} actors + 1 learner")
    tqdm.write(f"  Log → {log_path}")
    tqdm.write(sep)

    target_updates = cfg.total_updates_target or cfg.checkpoint_every_updates
    bar = tqdm(total=target_updates, desc="learner updates", unit="upd", dynamic_ncols=True)
    last_count = 0
    try:
        while True:
            time.sleep(2)
            count = _read_update_count(run_dir)
            bar.update(min(count - last_count, target_updates - last_count))
            last_count = count
            if not learner_proc.is_alive():
                tqdm.write("  Learner exited — stopping")
                break
            if count >= target_updates:
                tqdm.write(f"  Reached {count} updates — stopping")
                break
    except KeyboardInterrupt:
        tqdm.write("\n  Interrupted — shutting down…")
    finally:
        stop_event.set()
        for p in actor_procs:
            p.join(timeout=10)
        learner_proc.join(timeout=15)
        bar.close()

    tqdm.write(sep)
    tqdm.write(f"  Done. Checkpoints → {run_dir / 'checkpoints'}")
    tqdm.write(sep)


# ─── CLI ─────────────────────────────────────────────────


def _parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="GuanZero faithful persistent actor-learner DMC")
    p.add_argument("--config", type=str, default=None,
                   help="Path to YAML config; CLI flags override.")
    p.add_argument("--n-actors", type=int)
    p.add_argument("--updates", type=int, help="Target learner updates (checkpoint_every_updates).")
    p.add_argument("--device", type=str)
    p.add_argument("--run-dir", type=str)
    p.add_argument("--quick", action="store_true",
                   help="Smoke: 2 actors, tiny network, 200 updates.")
    args = p.parse_args()

    cfg_dict: dict[str, Any] = {}
    if args.config:
        import yaml
        cfg_dict.update(yaml.safe_load(Path(args.config).read_text()) or {})
    if args.quick:
        cfg_dict.update({
            "n_actors": 2,
            "hidden_lstm": 64,
            "hidden_mlp": 128,
            "n_mlp_layers": 3,
            "buffer_min_size": 50,
            "total_updates_target": 200,
            "checkpoint_every_updates": 100,
            "log_every_updates": 20,
            "publish_interval_updates": 10,
            "actor_push_batch_size": 64,
        })
    # CLI flags override --quick / --config
    if args.n_actors is not None:
        cfg_dict["n_actors"] = args.n_actors
    if args.updates is not None:
        cfg_dict["total_updates_target"] = args.updates
    if args.device is not None:
        cfg_dict["device"] = args.device
    if args.run_dir is not None:
        cfg_dict["run_dir"] = args.run_dir

    valid = {f.name for f in dataclasses.fields(TrainConfig)}
    cfg_dict = {k: v for k, v in cfg_dict.items() if k in valid}
    return TrainConfig(**cfg_dict)


def main() -> None:
    cfg = _parse_args()
    train_distributed(cfg)


if __name__ == "__main__":
    main()
