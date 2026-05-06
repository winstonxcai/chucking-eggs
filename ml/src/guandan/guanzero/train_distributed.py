"""Faithful persistent actor-learner DMC orchestrator.

Architecture
────────────

  ┌────────────────────────────────────────────────────────────────┐
  │  Main process  (train_distributed)                             │
  │  Spawns learner + N actors; polls metrics_learner.jsonl (tqdm) │
  │  Sets stop_event → joins all procs on done / KeyboardInterrupt │
  └──────────┬─────────────────────────────────────┬──────────────┘
             │ spawn                               │ spawn ×N
             ▼                                     ▼
  ┌─────────────────────────┐     ┌──────────────────────────────┐
  │  Learner  (GPU/CPU)      │     │  Actor-i  (CPU, no grad)      │
  │                          │     │                               │
  │  Q-nets[0..3] compiled   │     │  Q-nets[0..3]  local copy     │
  │  ReplayBuffer  per-pos   │     │  loop:                        │
  │  loop:                   │     │    play_episode() → samples   │
  │    drain queue → buffer  │◄────│    accumulate actor_push_batch│
  │    MSE update ×4 pos     │     │    queue.put(stacked_msg)     │
  │    every P updates:      │     │    every K episodes:          │
  │      publish weights─────┼────►│      read latest.txt          │
  │      (atomic os.replace) │     │      load weights_{ver}.pt    │
  │    every C updates:      │     └──────────────────────────────┘
  │      save checkpoint     │
  │    every L updates:      │     weight_dir/  (disk or /tmp on Modal)
  │      append metrics.jsonl│      ├── latest.txt       ← atomic rename
  └─────────────────────────┘      └── weights_{ver}.pt  ← atomic rename

  Communication:
    Actors → Learner : mp.Queue (bounded, pre-stacked numpy arrays)
    Learner → Actors : filesystem poll  (latest.txt + weights_{ver}.pt)
    Main    → all    : mp.Event  (stop_event)
    Learner → Main   : metrics_learner.jsonl  (progress polling)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
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


def train_distributed(cfg: TrainConfig, resume_checkpoint: Path | None = None) -> None:
    import multiprocessing as mp

    run_dir    = Path(cfg.resolved_run_dir)
    # Weight dir defaults to run_dir/weights but can be redirected to a
    # container-local path (e.g. /tmp on Modal) to avoid network-volume IO
    # during steady-state weight publishing/syncing. Checkpoints stay in run_dir.
    weight_dir_env = os.environ.get("GUANZERO_WEIGHT_DIR")
    weight_dir = Path(weight_dir_env) if weight_dir_env else run_dir / "weights"
    run_dir.mkdir(parents=True, exist_ok=True)
    weight_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2))

    logger, log_path = _setup_logging(run_dir)
    logger.info("train_distributed: %d actors, target %d updates", cfg.n_actors, cfg.total_updates_target)
    if resume_checkpoint:
        logger.info("resuming from checkpoint: %s", resume_checkpoint)

    ctx          = mp.get_context("spawn")
    sample_queue = ctx.Queue(maxsize=cfg.sample_queue_maxsize)
    stop_event   = ctx.Event()
    cfg_dict     = dataclasses.asdict(cfg)

    learner_proc = ctx.Process(
        target=learner_loop,
        args=(cfg_dict, sample_queue, stop_event, weight_dir, run_dir),
        kwargs={"resume_checkpoint": resume_checkpoint},
        daemon=True,
        name="learner",
    )
    learner_proc.start()

    tqdm.write(f"  Learner started (pid={learner_proc.pid})")
    tqdm.write(f"  Waiting for initial weights → {weight_dir}")
    _wait_for_weights(weight_dir, timeout=60)
    tqdm.write("  Initial weights ready — starting actors")

    # ── Optional inference server ─────────────────────────
    inf_bufs = inf_meta = inf_server_proc = None
    inference_args = None
    if cfg.use_inference_server:
        from . import inference_server as _isrv

        inf_bufs, inf_meta = _isrv.allocate_shared_buffers(
            num_slots   = cfg.inference_n_slots,
            max_actions = cfg.inference_max_actions,
            n_actors    = cfg.n_actors,
            ctx         = ctx,
        )
        q_net_kwargs = {
            "hidden_lstm":            cfg.hidden_lstm,
            "hidden_mlp":             cfg.hidden_mlp,
            "n_mlp_layers":           cfg.n_mlp_layers,
            "dropout":                cfg.dropout,
            "use_oracle_others_hand": cfg.use_oracle_others_hand,
        }
        inf_server_proc = ctx.Process(
            target=_isrv.shared_server_loop_entry,
            args=(
                cfg_dict, q_net_kwargs, inf_meta,
                inf_bufs.free_slots, inf_bufs.request_queue, inf_bufs.events,
                stop_event,
            ),
            kwargs={
                "weight_dir":      weight_dir,
                "server_log_path": str(run_dir / "inference_server.log"),
            },
            daemon=True,
            name="inference_server",
        )
        inf_server_proc.start()
        tqdm.write(f"  Inference server started (pid={inf_server_proc.pid}) "
                   f"on device={cfg.inference_device}")

        inference_args = {
            "meta":            inf_meta,
            "free_slots":      inf_bufs.free_slots,
            "request_queue":   inf_bufs.request_queue,
            "events":          inf_bufs.events,
            "weights_version": None,    # Phase 5 will wire the shared-mem version
        }

    actor_procs = []
    for actor_id in range(cfg.n_actors):
        p = ctx.Process(
            target=actor_loop,
            args=(actor_id, cfg_dict, sample_queue, stop_event, weight_dir),
            kwargs={"inference_args": inference_args},
            daemon=True,
            name=f"actor-{actor_id}",
        )
        p.start()
        actor_procs.append(p)

    sep = "=" * 68
    tqdm.write(sep)
    if cfg.use_inference_server:
        tqdm.write(f"  GuanZero distributed  |  {cfg.n_actors} actors + 1 learner + 1 inference server")
    else:
        tqdm.write(f"  GuanZero distributed  |  {cfg.n_actors} actors + 1 learner")
    tqdm.write(f"  Log → {log_path}")
    tqdm.write(sep)

    target_updates = cfg.total_updates_target or cfg.checkpoint_every_updates
    resume_updates = _read_update_count(run_dir)  # 0 if fresh run, >0 if resumed
    bs = cfg.batch_size
    bar = tqdm(total=target_updates * bs, initial=resume_updates * bs,
               desc="learner", unit="samp", unit_scale=True, dynamic_ncols=True)
    last_count = resume_updates
    try:
        while True:
            time.sleep(2)
            count = _read_update_count(run_dir)
            delta = min(count - last_count, target_updates - last_count)
            bar.update(delta * bs)
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
        if inf_server_proc is not None:
            inf_server_proc.join(timeout=15)
        if inf_bufs is not None:
            from . import inference_server as _isrv
            _isrv.release_shared_buffers(inf_bufs, unlink=True)
        bar.close()

    tqdm.write(sep)
    tqdm.write(f"  Done. Checkpoints → {run_dir / 'checkpoints'}")
    tqdm.write(sep)


# ─── CLI ─────────────────────────────────────────────────


def _parse_args() -> tuple[TrainConfig, Path | None]:
    p = argparse.ArgumentParser(description="GuanZero faithful persistent actor-learner DMC")
    p.add_argument("--config", type=str, default=None,
                   help="Path to YAML config; CLI flags override.")
    p.add_argument("--n-actors", type=int)
    p.add_argument("--updates", type=int, help="Target learner updates.")
    p.add_argument("--device", type=str)
    p.add_argument("--run-dir", type=str)
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint .pt to warm-start from.")
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
            "total_updates_target": 500,
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
    resume = Path(args.resume) if args.resume else None
    return TrainConfig(**cfg_dict), resume


def main() -> None:
    cfg, resume = _parse_args()
    train_distributed(cfg, resume_checkpoint=resume)


if __name__ == "__main__":
    main()
