"""GuanZero persistent actor-learner DMC orchestrator.

Architecture
────────────

  ┌────────────────────────────────────────────────────────────────┐
  │  Main process  (train)                                         │
  │  Spawns learner + N actors; polls metrics_learner.jsonl (tqdm) │
  │  Sets stop_event → joins all procs on done / KeyboardInterrupt │
  └──────────┬─────────────────────────────────────┬──────────────┘
             │ spawn                               │ spawn ×N
             ▼                                     ▼
  ┌─────────────────────────┐     ┌──────────────────────────────┐
  │  Learner  (GPU/CPU)      │     │  Actor-i  (CPU, no grad)      │
  │                          │     │                               │
  │  Q-nets[0..3] compiled   │     │  Q-nets[0..3]  local copy     │
  │  ReplayBuffer  per-seat  │     │  loop:                        │
  │  loop:                   │     │    play_episode() → samples   │
  │    drain queue → buffer  │◄────│    accumulate actor_push_batch│
  │    MSE update ×4 seats   │     │    queue.put(stacked_msg)     │
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

  Environment overrides:
    GUANZERO_WEIGHT_DIR  — redirect weight publishing to a container-local
                           path (e.g. /tmp on Modal) to avoid network-volume
                           I/O during steady-state syncing. Checkpoints stay
                           in run_dir regardless.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from pathlib import Path

from tqdm import tqdm

from .worker import actor_loop
from .learner import learner_loop, load_latest_weights
from .config import TrainConfig, load_config_from_cli
from .logging_setup import setup_run_logging
from . import inference_server as _isrv


# ─── Process-lifecycle constants ─────────────────────────────

_INITIAL_WEIGHTS_TIMEOUT_S = 60.0
_WATCHER_POLL_INTERVAL_S   = 2.0
_ACTOR_JOIN_TIMEOUT_S      = 10.0
_LEARNER_JOIN_TIMEOUT_S    = 15.0
_SERVER_JOIN_TIMEOUT_S     = 15.0

_QUICK_OVERRIDES = {
    "n_actors": 2,
    "buffer_min_size": 50,
    "batch_size": 64,
    "total_updates_target": 500,
    "checkpoint_every_updates": 100,
    "log_every_updates": 20,
    "publish_interval_updates": 10,
    "actor_push_batch_size": 64,
}


# ─── Helpers ─────────────────────────────────────────────────


def _wait_for_weights(weight_dir: Path, timeout: float = _INITIAL_WEIGHTS_TIMEOUT_S) -> None:
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


def _shutdown_processes(
    procs: list,
    soft_timeout: float,
    label: str = "proc",
) -> None:
    """join → terminate → join for a list of processes."""
    for p in procs:
        if p is None:
            continue
        p.join(timeout=soft_timeout)
        if p.is_alive():
            tqdm.write(f"  {label} {p.name} did not exit; terminating…")
            p.terminate()
            p.join(timeout=3.0)
        if p.is_alive():
            tqdm.write(f"  {label} {p.name} still alive after terminate; killing…")
            p.kill()


# ─── Main entry point ────────────────────────────────────────


def train(cfg: TrainConfig, resume_checkpoint: Path | None = None) -> None:
    import multiprocessing as mp

    run_dir    = Path(cfg.resolved_run_dir)
    weight_dir_env = os.environ.get("GUANZERO_WEIGHT_DIR")
    weight_dir = Path(weight_dir_env) if weight_dir_env else run_dir / "weights"
    run_dir.mkdir(parents=True, exist_ok=True)
    weight_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2))

    logger, log_path = setup_run_logging(run_dir)
    logger.info("train: %d actors, target %d updates", cfg.n_actors, cfg.total_updates_target)
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
    _wait_for_weights(weight_dir, timeout=_INITIAL_WEIGHTS_TIMEOUT_S)
    initial_snapshot = load_latest_weights(weight_dir)
    if initial_snapshot is None:
        raise RuntimeError(f"Initial weights became unreadable in {weight_dir}")
    tqdm.write("  Initial weights ready — starting actors")

    # ── Optional inference server ─────────────────────────────
    inf_bufs = inf_meta = inf_server_proc = None
    inference_args = None
    if cfg.inference.enabled:
        inf_bufs, inf_meta = _isrv.allocate_shared_buffers(
            num_slots   = cfg.inference.n_slots,
            max_actions = cfg.inference.max_actions,
            n_actors    = cfg.n_actors,
            ctx         = ctx,
        )
        inf_server_proc = ctx.Process(
            target=_isrv.run_server,
            args=(
                cfg_dict, inf_meta,
                inf_bufs.free_slots, inf_bufs.request_queue, inf_bufs.events,
                stop_event,
            ),
            kwargs={
                "initial_state_dicts": initial_snapshot.state_dicts,
                "initial_version":     initial_snapshot.version,
                "weight_dir":      weight_dir,
                "server_log_path": str(run_dir / "inference_server.log"),
            },
            daemon=True,
            name="inference_server",
        )
        inf_server_proc.start()
        tqdm.write(f"  Inference server started (pid={inf_server_proc.pid}) "
                   f"on device={cfg.inference.device}")

        inference_args = {
            "meta":          inf_meta,
            "free_slots":    inf_bufs.free_slots,
            "request_queue": inf_bufs.request_queue,
            "events":        inf_bufs.events,
        }

    actor_procs = []
    for actor_id in range(cfg.n_actors):
        p = ctx.Process(
            target=actor_loop,
            args=(actor_id, cfg_dict, sample_queue, stop_event, weight_dir),
            kwargs={"inference_args": inference_args, "run_dir": run_dir},
            daemon=True,
            name=f"actor-{actor_id}",
        )
        p.start()
        actor_procs.append(p)

    sep = "=" * 68
    tqdm.write(sep)
    if cfg.inference.enabled:
        tqdm.write(f"  GuanZero  |  {cfg.n_actors} actors + 1 learner + 1 inference server")
    else:
        tqdm.write(f"  GuanZero  |  {cfg.n_actors} actors + 1 learner")
    tqdm.write(f"  Log → {log_path}")
    tqdm.write(sep)

    target_updates = cfg.total_updates_target or cfg.checkpoint_every_updates
    resume_updates = _read_update_count(run_dir)
    bs = cfg.batch_size
    bar = tqdm(total=target_updates * bs, initial=resume_updates * bs,
               desc="learner", unit="samp", unit_scale=True, dynamic_ncols=True)
    last_count = resume_updates
    abort_reason: str | None = None
    try:
        while True:
            time.sleep(_WATCHER_POLL_INTERVAL_S)
            count = _read_update_count(run_dir)
            delta = min(count - last_count, target_updates - last_count)
            bar.update(delta * bs)
            last_count = count
            if not learner_proc.is_alive():
                abort_reason = f"Learner exited (code={learner_proc.exitcode})"
                break
            # Actor-crash detection: any dead actor is a bug; abort the run
            for ap in actor_procs:
                if not ap.is_alive() and ap.exitcode not in (0, None):
                    abort_reason = f"{ap.name} died (code={ap.exitcode})"
                    break
            if abort_reason:
                break
            if count >= target_updates:
                abort_reason = None   # clean stop
                break
    except KeyboardInterrupt:
        tqdm.write("\n  Interrupted — shutting down…")
    finally:
        if abort_reason:
            tqdm.write(f"  Aborting: {abort_reason}")
        stop_event.set()
        _shutdown_processes(actor_procs, soft_timeout=_ACTOR_JOIN_TIMEOUT_S, label="actor")
        _shutdown_processes([learner_proc], soft_timeout=_LEARNER_JOIN_TIMEOUT_S, label="learner")
        if inf_server_proc is not None:
            _shutdown_processes([inf_server_proc], soft_timeout=_SERVER_JOIN_TIMEOUT_S, label="server")
        if inf_bufs is not None:
            _isrv.release_shared_buffers(inf_bufs, unlink=True)
        bar.close()

    if abort_reason:
        tqdm.write(sep)
        tqdm.write(f"  Failed. Check logs in {run_dir}")
        tqdm.write(sep)
        raise RuntimeError(abort_reason)
    tqdm.write(sep)
    tqdm.write(f"  Done. Checkpoints → {run_dir / 'checkpoints'}")
    tqdm.write(sep)


# ─── CLI ─────────────────────────────────────────────────────


def _parse_args() -> tuple[TrainConfig, Path | None]:
    p = argparse.ArgumentParser(description="GuanZero persistent actor-learner DMC")
    p.add_argument("--config", type=str, default=None,
                   help="Path to YAML config; CLI flags override.")
    p.add_argument("--n-actors", type=int)
    p.add_argument("--updates", type=int, help="Target learner updates.")
    p.add_argument("--device", type=str)
    p.add_argument("--seed", type=int)
    p.add_argument("--run-dir", type=str)
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint .pt to warm-start from.")
    p.add_argument("--quick", action="store_true",
                   help="Smoke: 2 actors, tiny network, 500 updates.")
    args = p.parse_args()

    resume = Path(args.resume) if args.resume else None
    cfg = load_config_from_cli(
        args.config,
        quick=args.quick,
        quick_overrides=_QUICK_OVERRIDES,
        n_actors=args.n_actors,
        total_updates_target=args.updates,
        device=args.device,
        seed=args.seed,
        run_dir=args.run_dir,
    )
    return cfg, resume


def main() -> None:
    cfg, resume = _parse_args()
    train(cfg, resume_checkpoint=resume)


if __name__ == "__main__":
    main()
