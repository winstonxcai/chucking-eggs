"""Dart persistent actor-learner DMC orchestrator.

Architecture
────────────

  ┌────────────────────────────────────────────────────────────────┐
  │  Main process  (train)                                         │
  │  Spawns learner + N actors; tqdm reads update_counter (mp)     │
  │  Sets stop_event → joins all procs on done / KeyboardInterrupt │
  └──────────┬─────────────────────────────────────┬──────────────┘
             │ spawn                               │ spawn ×N
             ▼                                     ▼
  ┌─────────────────────────┐     ┌──────────────────────────────┐
  │  Learner  (GPU/CPU)      │     │  Actor-i  (CPU, no grad)      │
  │                          │     │                               │
  │  Q-nets[0..3] compiled   │     │  Q-nets[0..3]  local copy     │
  │  ReplayBuffer  per-seat  │     │  loop:                        │
  │  loop:                   │     │    play_episodes_batched()     │
  │                          │     │      → per-lane samples        │
  │    drain queue → buffer  │◄────│    accumulate actor_push_batch│
  │    MSE update ×4 seats   │     │    queue.put(stacked_msg)     │
  │    update_counter += 1   │     │                               │
  │    every P updates:      │     │    every K episodes:          │
  │      publish weights─────┼────►│      read latest.txt          │
  │      (atomic os.replace) │     │      load weights_{ver}.pt    │
  │    every C updates:      │     └──────────────────────────────┘
  │      save checkpoint     │
  │    every L updates:      │     weight_dir/  (disk or /tmp on Modal)
  │      MetricsWriter.write │      ├── latest.txt       ← atomic rename
  └─────────────────────────┘      └── weights_{ver}.pt  ← atomic rename

  Communication:
    Actors → Learner : mp.Queue (bounded, pre-stacked numpy arrays)
    Learner → Actors : filesystem poll  (latest.txt + weights_{ver}.pt)
    Main    → all    : mp.Event  (stop_event; SIGINT + SIGTERM both set it)
    Learner → Main   : mp.Value  (update_counter; progress only)
    Learner → disk   : metrics_learner.jsonl  (durable, write-only at runtime)

  Environment overrides:
    DART_WEIGHT_DIR  — redirect weight publishing to a container-local
                           path (e.g. /tmp on Modal) to avoid network-volume
                           I/O during steady-state syncing. Checkpoints stay
                           in run_dir regardless.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path

from tqdm import tqdm

from .worker import actor_loop
from .learner import learner_loop
from .weights import load_latest_weights
from ..config import TrainConfig, load_config_from_cli
from ..utils.logging_setup import setup_run_logging
from ..utils.reproducibility import seed_everything
from ..utils.run_layout import RunLayout


# ─── Process-lifecycle constants ─────────────────────────────

_INITIAL_WEIGHTS_TIMEOUT_S = 60.0
_WATCHER_POLL_INTERVAL_S   = 2.0
_ACTOR_JOIN_TIMEOUT_S      = 10.0
_LEARNER_JOIN_TIMEOUT_S    = 15.0


# ─── Helpers ─────────────────────────────────────────────────


def _shutdown_processes(
    procs: list,
    soft_timeout: float,
    label: str = "proc",
) -> None:
    """join → terminate → join for a list of processes."""
    logger = logging.getLogger("dart")
    for p in procs:
        if p is None:
            continue
        p.join(timeout=soft_timeout)
        if p.is_alive():
            logger.warning("%s %s did not exit; terminating…", label, p.name)
            p.terminate()
            p.join(timeout=3.0)
        if p.is_alive():
            logger.warning("%s %s still alive after terminate; killing…", label, p.name)
            p.kill()


def _install_signal_handlers(stop_event) -> None:
    """Route SIGINT and SIGTERM to ``stop_event.set()`` so cloud schedulers
    (Modal, k8s) trigger the same clean shutdown path as a local Ctrl-C."""
    def _handler(signum, _frame):
        logging.getLogger("dart").warning(
            "received %s — initiating shutdown", signal.Signals(signum).name
        )
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError):
            # Not on the main thread (test harness, etc.) — skip.
            pass


def _tqdm_disabled() -> bool:
    """Return whether the live progress bar should be suppressed.

    Cloud log collectors such as Modal merge stdout/stderr from multiple
    processes. A redrawing tqdm bar then interleaves with learner/actor log
    lines even when each process writes cleanly. Default to disabling the bar
    outside an interactive terminal, with an env override for local debugging.
    """
    setting = os.environ.get("DART_TQDM", "").strip().lower()
    if setting in {"0", "false", "no", "off"}:
        return True
    if setting in {"1", "true", "yes", "on"}:
        return False
    return not sys.stderr.isatty()


# ─── Stage helpers ───────────────────────────────────────────


def _spawn_learner(
    ctx,
    cfg_dict: dict,
    sample_queue,
    stop_event,
    weight_dir: Path,
    layout: RunLayout,
    update_counter,
    weights_ready,
    resume_checkpoint: Path | None,
):
    proc = ctx.Process(
        target=learner_loop,
        args=(cfg_dict, sample_queue, stop_event, weight_dir, layout.run_dir, update_counter),
        kwargs={"weights_ready": weights_ready, "resume_checkpoint": resume_checkpoint},
        daemon=True,
        name="learner",
    )
    proc.start()
    return proc


def _spawn_actors(
    ctx,
    cfg: TrainConfig,
    cfg_dict: dict,
    sample_queue,
    stop_event,
    weight_dir: Path,
    layout: RunLayout,
) -> list:
    procs = []
    for actor_id in range(cfg.n_actors):
        p = ctx.Process(
            target=actor_loop,
            args=(actor_id, cfg_dict, sample_queue, stop_event, weight_dir),
            kwargs={"run_dir": layout.run_dir},
            daemon=True,
            name=f"actor-{actor_id}",
        )
        p.start()
        procs.append(p)
    return procs


def _run_progress_loop(
    update_counter,
    target_updates: int,
    batch_size: int,
    learner_proc,
    actor_procs: list,
    stop_event,
) -> str | None:
    """Drive the tqdm bar and watch for crashes. Returns abort_reason or None."""
    resume_updates = int(update_counter.value)
    bar = tqdm(
        total=target_updates * batch_size,
        initial=resume_updates * batch_size,
        desc="learner", unit="samp", unit_scale=True, dynamic_ncols=True,
        disable=_tqdm_disabled(),
    )
    last_count = resume_updates
    try:
        while not stop_event.is_set():
            time.sleep(_WATCHER_POLL_INTERVAL_S)
            count = int(update_counter.value)
            delta = min(count - last_count, target_updates - last_count)
            bar.update(delta * batch_size)
            last_count = count
            if not learner_proc.is_alive():
                return f"Learner exited (code={learner_proc.exitcode})"
            for ap in actor_procs:
                if not ap.is_alive() and ap.exitcode not in (0, None):
                    return f"{ap.name} died (code={ap.exitcode})"
            if count >= target_updates:
                return None
        return None  # signal-driven shutdown
    finally:
        bar.close()


def _shutdown_all(
    stop_event,
    actor_procs: list,
    learner_proc,
) -> None:
    stop_event.set()
    _shutdown_processes(actor_procs, soft_timeout=_ACTOR_JOIN_TIMEOUT_S, label="actor")
    _shutdown_processes([learner_proc], soft_timeout=_LEARNER_JOIN_TIMEOUT_S, label="learner")


# ─── Main entry point ────────────────────────────────────────


def train(cfg: TrainConfig, resume_checkpoint: Path | None = None) -> None:
    import multiprocessing as mp

    seed_everything(cfg.seed)

    layout         = RunLayout(Path(cfg.resolved_run_dir))
    weight_dir_env = os.environ.get("DART_WEIGHT_DIR")
    weight_dir     = Path(weight_dir_env) if weight_dir_env else layout.run_dir / "weights"
    layout.run_dir.mkdir(parents=True, exist_ok=True)
    weight_dir.mkdir(parents=True, exist_ok=True)
    layout.config_json.write_text(json.dumps(dataclasses.asdict(cfg), indent=2))

    logger, log_path = setup_run_logging(layout.run_dir, layout.train_log.name, stream_to_stdout=True)
    logger.info("train: %d actors, target %d updates", cfg.n_actors, cfg.total_updates_target)
    if resume_checkpoint:
        logger.info("resuming from checkpoint: %s", resume_checkpoint)

    ctx            = mp.get_context("spawn")
    sample_queue   = ctx.Queue(maxsize=cfg.sample_queue_maxsize)
    stop_event     = ctx.Event()
    weights_ready  = ctx.Event()
    update_counter = ctx.Value("q", 0)   # int64; learner advances per gradient step
    cfg_dict       = dataclasses.asdict(cfg)

    _install_signal_handlers(stop_event)

    learner_proc = _spawn_learner(
        ctx, cfg_dict, sample_queue, stop_event,
        weight_dir, layout, update_counter, weights_ready, resume_checkpoint,
    )
    logger.info("Learner started (pid=%d)", learner_proc.pid)
    logger.info("Waiting for initial weights → %s", weight_dir)
    if not weights_ready.wait(timeout=_INITIAL_WEIGHTS_TIMEOUT_S):
        raise TimeoutError(
            f"Learner did not publish weights within {_INITIAL_WEIGHTS_TIMEOUT_S}s"
        )
    if load_latest_weights(weight_dir) is None:
        raise RuntimeError(f"Initial weights became unreadable in {weight_dir}")
    logger.info("Initial weights ready — starting actors")

    actor_procs = _spawn_actors(
        ctx, cfg, cfg_dict, sample_queue, stop_event,
        weight_dir, layout,
    )

    logger.info("Dart | %d actors + 1 learner", cfg.n_actors)
    logger.info("log → %s", log_path)

    target_updates = cfg.total_updates_target or cfg.checkpoint_every_updates
    abort_reason: str | None = None
    try:
        abort_reason = _run_progress_loop(
            update_counter, target_updates, cfg.batch_size,
            learner_proc, actor_procs, stop_event,
        )
    finally:
        if abort_reason:
            logger.error("Aborting: %s", abort_reason)
        _shutdown_all(stop_event, actor_procs, learner_proc)

    if abort_reason:
        logger.error("training failed — check logs in %s", layout.run_dir)
        raise RuntimeError(abort_reason)
    logger.info("done — checkpoints → %s", layout.checkpoints_dir)


# ─── CLI ─────────────────────────────────────────────────────


def _parse_args() -> tuple[TrainConfig, Path | None]:
    p = argparse.ArgumentParser(description="Dart persistent actor-learner DMC")
    p.add_argument("--config", type=str, default=None,
                   help="Path to YAML config; CLI flags override.")
    p.add_argument("--n-actors", type=int)
    p.add_argument("--updates", type=int, help="Target learner updates.")
    p.add_argument("--device", type=str)
    p.add_argument("--seed", type=int)
    p.add_argument("--run-dir", type=str)
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint .pt to warm-start from.")
    args = p.parse_args()

    resume = Path(args.resume) if args.resume else None

    # When resuming without an explicit --run-dir, infer run_dir from the checkpoint
    # so that metrics_learner.jsonl, train.log, and actor logs all append in place.
    run_dir = args.run_dir
    if resume and not run_dir:
        run_dir = str(resume.resolve().parent.parent)

    cfg = load_config_from_cli(
        args.config,
        n_actors=args.n_actors,
        total_updates_target=args.updates,
        device=args.device,
        seed=args.seed,
        run_dir=run_dir,
    )
    return cfg, resume


def main() -> None:
    cfg, resume = _parse_args()
    train(cfg, resume_checkpoint=resume)


if __name__ == "__main__":
    main()
