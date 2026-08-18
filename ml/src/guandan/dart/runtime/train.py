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
import hashlib
import json
import logging
import os
import queue
import signal
import subprocess
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

from ..config import TrainConfig, load_config_from_cli
from ..utils.logging_setup import setup_run_logging
from ..utils.reproducibility import seed_everything
from ..utils.run_layout import RunLayout
from .learner import learner_loop
from .weights import load_latest_weights
from .worker import actor_loop

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


def _validate_torch_device_available(device: str, *, field: str) -> None:
    """Fail before spawning subprocesses when a config requests unavailable hardware."""
    try:
        torch_device = torch.device(device)
    except (RuntimeError, TypeError) as exc:
        raise ValueError(f"{field}={device!r} is not a valid Torch device") from exc

    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            f"{field}={device!r} requires CUDA, but this PyTorch install cannot use CUDA. "
            "Use --device cpu, dart_cpu_smoke.yaml, dart_mps.yaml on Apple Silicon, "
            "or run a CUDA config on Modal/GPU hardware."
        )
    if torch_device.type == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is None or not mps_backend.is_available():
            raise ValueError(
                f"{field}={device!r} requires Apple MPS, but MPS is not available. "
                "Use --device cpu or dart_cpu_smoke.yaml for a local smoke test."
            )


def _validate_config_devices(cfg: TrainConfig) -> None:
    _validate_torch_device_available(cfg.device, field="device")
    if cfg.eval.enabled:
        _validate_torch_device_available(cfg.eval.device, field="eval.device")


def _resolve_eval_opponents(cfg: TrainConfig) -> list[str]:
    from guandan.scripts.eval.eval_dart import DEFAULT_OPPONENTS

    opponents = cfg.eval.opponents
    if opponents == "all":
        return list(DEFAULT_OPPONENTS)
    requested = list(opponents)
    unknown = set(requested) - set(DEFAULT_OPPONENTS)
    if unknown:
        raise ValueError(
            f"eval.opponents has unknown or unsupported agent(s): {sorted(unknown)}. "
            f"Valid: {DEFAULT_OPPONENTS}"
        )
    return requested


def _run_identity(cfg: TrainConfig) -> tuple[str, str]:
    cfg_json = json.dumps(dataclasses.asdict(cfg), sort_keys=True, separators=(",", ":"))
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=Path(__file__).resolve().parents[5],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_sha = "unknown"
    digest = hashlib.sha256(f"{git_sha}:{cfg.seed}:{cfg_json}".encode()).hexdigest()[:12]
    return digest, git_sha


def _run_checkpoint_eval(
    cfg: TrainConfig,
    layout: RunLayout,
    request: dict,
    logger: logging.Logger,
) -> None:
    updates = int(request["updates"])
    checkpoint = Path(request["checkpoint"])
    eval_dir = layout.run_dir / "eval"
    out_path = eval_dir / f"{checkpoint.stem}.json"
    log_path = eval_dir / f"{checkpoint.stem}.log"
    opponents = _resolve_eval_opponents(cfg)
    workers = cfg.eval.workers or cfg.n_actors
    lanes = cfg.eval.lanes or cfg.actor_batch_lanes

    cmd = [
        sys.executable,
        "-m",
        "guandan.scripts.eval.eval_dart",
        "--checkpoint",
        str(checkpoint),
        "--opponent",
        *opponents,
        "--games",
        str(cfg.eval.n_eval_games_per_opponent),
        "--workers",
        str(workers),
        "--lanes",
        str(lanes),
        "--device",
        cfg.eval.device,
        "--seed",
        str(cfg.seed),
        "--out",
        str(out_path),
    ]
    eval_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "checkpoint eval start update=%d opponents=%d games/opponent=%d workers=%d lanes=%d → %s",
        updates,
        len(opponents),
        cfg.eval.n_eval_games_per_opponent,
        workers,
        lanes,
        out_path,
    )
    env = os.environ.copy()
    env["DART_TQDM"] = "0"
    env["PYTHONUNBUFFERED"] = "1"
    with log_path.open("w") as log_file:
        log_file.write("$ " + " ".join(cmd) + "\n\n")
        log_file.flush()
        result = subprocess.run(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            check=False,
            timeout=cfg.eval.max_wait_s,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"checkpoint eval failed for update {updates} "
            f"(exit={result.returncode}); see {log_path}"
        )
    logger.info("checkpoint eval complete update=%d → %s", updates, out_path)


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
    pause_event=None,
    eval_request_queue=None,
    eval_done_event=None,
    queue_full_counter=None,
    training_start_event=None,
    training_start_time=None,
):
    proc = ctx.Process(
        target=learner_loop,
        args=(cfg_dict, sample_queue, stop_event, weight_dir, layout.run_dir, update_counter),
        kwargs={
            "weights_ready": weights_ready,
            "resume_checkpoint": resume_checkpoint,
            "pause_event": pause_event,
            "eval_request_queue": eval_request_queue,
            "eval_done_event": eval_done_event,
            "queue_full_counter": queue_full_counter,
            "training_start_event": training_start_event,
            "training_start_time": training_start_time,
        },
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
    pause_event=None,
    actor_rng_states: dict[int, dict] | None = None,
    queue_full_counter=None,
) -> list:
    procs = []
    for actor_id in range(cfg.n_actors):
        p = ctx.Process(
            target=actor_loop,
            args=(actor_id, cfg_dict, sample_queue, stop_event, weight_dir),
            kwargs={
                "run_dir": layout.run_dir,
                "pause_event": pause_event,
                "resume_actor_rng_state": (
                    actor_rng_states.get(actor_id) if actor_rng_states else None
                ),
                "queue_full_counter": queue_full_counter,
            },
            daemon=True,
            name=f"actor-{actor_id}",
        )
        p.start()
        procs.append(p)
    return procs


def _load_actor_rng_states(
    resume_checkpoint: Path | None,
    logger: logging.Logger,
) -> dict[int, dict] | None:
    if resume_checkpoint is None:
        return None
    import torch

    ckpt = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
    raw = ckpt.get("actor_rng_states")
    if not raw:
        logger.warning(
            "resume checkpoint has no actor_rng_states; actor RNG streams will "
            "restart from seed-derived states"
        )
        return None
    return {int(actor_id): state for actor_id, state in raw.items()}


def _run_progress_loop(
    cfg: TrainConfig,
    layout: RunLayout,
    update_counter,
    target_updates: int | None,
    batch_size: int,
    learner_proc,
    actor_procs: list,
    stop_event,
    pause_event=None,
    eval_request_queue=None,
    eval_done_event=None,
) -> str | None:
    """Drive the tqdm bar and watch for crashes. Returns abort_reason or None."""
    resume_updates = int(update_counter.value)
    bar = tqdm(
        total=(target_updates * batch_size) if target_updates is not None else None,
        initial=resume_updates * batch_size,
        desc="learner", unit="samp", unit_scale=True, dynamic_ncols=True,
        disable=_tqdm_disabled(),
    )
    last_count = resume_updates
    logger = logging.getLogger("dart")
    try:
        while not stop_event.is_set():
            time.sleep(_WATCHER_POLL_INTERVAL_S)
            count = int(update_counter.value)
            delta = count - last_count
            if target_updates is not None:
                delta = min(delta, target_updates - last_count)
            bar.update(delta * batch_size)
            last_count = count
            if eval_request_queue is not None and eval_done_event is not None:
                while True:
                    try:
                        request = eval_request_queue.get_nowait()
                    except queue.Empty:
                        break
                    try:
                        _run_checkpoint_eval(cfg, layout, request, logger)
                    except Exception as exc:
                        logger.exception("%s", exc)
                        stop_event.set()
                        if pause_event is not None:
                            pause_event.clear()
                        eval_done_event.set()
                        return str(exc)
                    if pause_event is not None:
                        pause_event.clear()
                    eval_done_event.set()
            if not learner_proc.is_alive():
                if learner_proc.exitcode == 0:
                    return None
                return f"Learner exited (code={learner_proc.exitcode})"
            for ap in actor_procs:
                if not ap.is_alive() and ap.exitcode not in (0, None):
                    return f"{ap.name} died (code={ap.exitcode})"
            if target_updates is not None and count >= target_updates:
                if cfg.eval.enabled and learner_proc.is_alive():
                    continue
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

    _validate_config_devices(cfg)
    seed_everything(cfg.seed)

    layout         = RunLayout(Path(cfg.resolved_run_dir))
    weight_dir_env = os.environ.get("DART_WEIGHT_DIR")
    weight_dir     = Path(weight_dir_env) if weight_dir_env else layout.run_dir / "weights"
    layout.run_dir.mkdir(parents=True, exist_ok=True)
    weight_dir.mkdir(parents=True, exist_ok=True)
    layout.config_json.write_text(json.dumps(dataclasses.asdict(cfg), indent=2))

    logger, log_path = setup_run_logging(layout.run_dir, layout.train_log.name, stream_to_stdout=True)
    run_id, git_sha = _run_identity(cfg)
    logger.info("run_id=%s git_sha=%s seed=%d", run_id, git_sha, cfg.seed)
    logger.info("train: %d actors, target %d updates", cfg.n_actors, cfg.total_updates_target)
    if resume_checkpoint:
        logger.info("resuming from checkpoint: %s", resume_checkpoint)

    default_start_method = os.environ.get("DART_MP_START_METHOD", "spawn")
    learner_start_method = os.environ.get(
        "DART_LEARNER_MP_START_METHOD", default_start_method
    )
    actor_start_method = os.environ.get(
        "DART_ACTOR_MP_START_METHOD", default_start_method
    )
    available_methods = mp.get_all_start_methods()
    invalid_methods = [
        method
        for method in (learner_start_method, actor_start_method)
        if method not in available_methods
    ]
    if invalid_methods:
        available = ", ".join(mp.get_all_start_methods())
        raise ValueError(
            f"unsupported multiprocessing start method(s): {invalid_methods}; "
            f"available methods: {available}"
        )
    learner_ctx    = mp.get_context(learner_start_method)
    actor_ctx      = mp.get_context(actor_start_method)
    logger.info(
        "multiprocessing start methods: learner=%s actors=%s",
        learner_start_method,
        actor_start_method,
    )
    sample_queue   = learner_ctx.Queue(maxsize=cfg.sample_queue_maxsize)
    stop_event     = learner_ctx.Event()
    pause_event    = learner_ctx.Event()
    weights_ready  = learner_ctx.Event()
    update_counter = learner_ctx.Value("q", 0)   # int64; learner advances per gradient step
    training_start_event = learner_ctx.Event()
    training_start_time = learner_ctx.Value("d", 0.0)
    queue_full_counter = learner_ctx.Value("q", 0)
    eval_request_queue = learner_ctx.Queue() if cfg.eval.enabled else None
    eval_done_event = learner_ctx.Event() if cfg.eval.enabled else None
    cfg_dict       = dataclasses.asdict(cfg)
    actor_rng_states = _load_actor_rng_states(resume_checkpoint, logger)

    _install_signal_handlers(stop_event)

    learner_proc = _spawn_learner(
        learner_ctx, cfg_dict, sample_queue, stop_event,
        weight_dir, layout, update_counter, weights_ready, resume_checkpoint,
        pause_event=pause_event,
        eval_request_queue=eval_request_queue,
        eval_done_event=eval_done_event,
        queue_full_counter=queue_full_counter,
        training_start_event=training_start_event,
        training_start_time=training_start_time,
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
        actor_ctx, cfg, cfg_dict, sample_queue, stop_event,
        weight_dir, layout, pause_event=pause_event,
        actor_rng_states=actor_rng_states,
        queue_full_counter=queue_full_counter,
    )
    training_start_time.value = time.monotonic()
    training_start_event.set()

    logger.info("Dart | %d actors + 1 learner", cfg.n_actors)
    logger.info("log → %s", log_path)

    target_updates = cfg.total_updates_target or None
    abort_reason: str | None = None
    try:
        abort_reason = _run_progress_loop(
            cfg, layout, update_counter, target_updates,
            cfg.learner_samples_per_update,
            learner_proc, actor_procs, stop_event,
            pause_event=pause_event,
            eval_request_queue=eval_request_queue,
            eval_done_event=eval_done_event,
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
    p.add_argument(
        "--actor-batch-lanes",
        type=int,
        help="Override the number of batched self-play lanes per actor.",
    )
    p.add_argument("--updates", type=int, help="Target learner updates.")
    p.add_argument(
        "--checkpoint-every-updates",
        type=int,
        help="Override the periodic checkpoint interval for this run.",
    )
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
        actor_batch_lanes=args.actor_batch_lanes,
        total_updates_target=args.updates,
        checkpoint_every_updates=args.checkpoint_every_updates,
        device=args.device,
        seed=args.seed,
        run_dir=run_dir,
    )
    return cfg, resume


def main() -> None:
    cfg, resume = _parse_args()
    try:
        train(cfg, resume_checkpoint=resume)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
