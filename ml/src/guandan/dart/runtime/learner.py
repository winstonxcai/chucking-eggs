"""Unified learner process entry-point for persistent actor-learner DMC.

Architecture
------------
``learner_loop`` is the single process target for all model variants.  It
dispatches to a concrete ``LearnerProtocol`` adapter (``_SeatAdapter`` or
``_DartAdapter``) that owns both the gradient-step logic and the replay
buffer.  Adding a new variant = implement one adapter class + wire it into
``_make_adapter``.

Module map
----------
``learners.guanzero`` — ``SeatLearner``       (per-position Q-nets, paper §4.2)
``learners.dart``     — ``DartLearner``       (role-aware Dart Q-net)
``learners.loss_buckets`` — phase/trick-role/action metric stratification helpers
``weights``           — atomic disk publish + read helpers used by actors too
"""

from __future__ import annotations

import dataclasses
import multiprocessing as mp
import time
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import torch

from ..data.buffer import ReplayBuffer, RoleAwareReplayBuffer
from ..model.checkpoint import (
    CheckpointSaveType,
    save_checkpoint_base,
    save_checkpoint_dart,
)
from ..utils.logging_setup import setup_run_logging
from ..utils.metrics import jsonl_writer
from ..utils.reproducibility import (
    capture_torch_rng_state,
    restore_torch_rng_state,
    seed_everything,
)
from ..utils.run_layout import RunLayout
from .learners.dart import DartLearner
from .learners.guanzero import SeatLearner
from .learners.loss_buckets import PHASE_KEY_PREFIXES
from .weights import (
    publish_weights,
    publish_weights_dart,
)


# ─── Shared result type ───────────────────────────────────────────────────────


@dataclasses.dataclass
class StepResult:
    """Per-step metrics returned by a learner adapter's ``step()`` method."""
    loss:  dict[str, Any]   # scalar loss fields (str-keyed floats or None)
    phase: dict[str, Any]   # phase-stratified metrics; empty dict for seat variant


# ─── Protocol ────────────────────────────────────────────────────────────────


@runtime_checkable
class LearnerProtocol(Protocol):
    """Interface every learner adapter must satisfy to plug into ``learner_loop``.

    Each adapter owns both the concrete learner (gradient math) and its
    replay buffer (sample storage + drain logic).  ``learner_loop`` only
    calls protocol methods — it has no knowledge of the underlying variant.
    """

    def drain_message(self, msg: dict) -> int:
        """Push one queue message into the internal buffer. Return #samples."""
        ...

    def is_ready(self) -> bool:
        """True when the buffer has enough data for a gradient step."""
        ...

    def step(self) -> StepResult | None:
        """Take one gradient step. Returns None if the buffer is still cold."""
        ...

    def publish(self, weight_dir: Path, version: int, updates: int) -> None:
        """Atomically write current weights to disk."""
        ...

    def checkpoint(
        self, path: Path, updates: int, *, save_type: CheckpointSaveType = "full"
    ) -> None:
        """Write a checkpoint to ``path``."""
        ...

    def resume(self, path: Path) -> int:
        """Load checkpoint state; return the total_updates stored in it."""
        ...

    def buffer_size(self) -> int:
        """Total samples across all buckets (for logging)."""
        ...

    def format_log_row(self, result: StepResult, timing: dict) -> dict:
        """Build the full jsonl metrics row from a StepResult + timing stats."""
        ...

    def log_progress(self, logger: Any, result: StepResult, timing: dict) -> None:
        """Write the human-readable logger.info summary line."""
        ...

    def on_idle(self, drained: int) -> None:
        """Called when total_updates hasn't advanced this iteration."""
        ...

    def on_profiler_report(self, logger: Any, interval_dt: float, interval_upd: int) -> None:
        """Log profiler breakdown if enabled. No-op for variants without one."""
        ...


# ─── Concrete adapters ────────────────────────────────────────────────────────


class _SeatAdapter:
    """LearnerProtocol adapter for the per-seat paper-repro variant."""

    def __init__(self, cfg) -> None:
        from ..model.q_network import init_guanzero_nets

        self._cfg = cfg
        device = cfg.device

        # CUDA tuning — global for this process
        if device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

        q_nets = init_guanzero_nets(cfg.qnet)
        compile_mode = cfg.compile_mode or "default"
        self._startup_messages: list[str] = []
        if compile_mode == "reduce-overhead" and device == "cuda":
            # Per-stream cudagraph_trees serialise shared CUDA memory pool.
            compile_mode = "default"
            self._startup_messages.append(
                "compile_mode downgraded reduce-overhead→default for multi-stream positions"
            )
        if device == "cuda":
            for p, net in q_nets.items():
                q_nets[p] = torch.compile(net, mode=compile_mode)
        else:
            self._startup_messages.append(
                f"torch.compile disabled for learner device={device}"
            )

        self._learner = SeatLearner(
            q_nets=q_nets,
            lr=cfg.lr,
            device=device,
            use_bf16=cfg.use_bf16_learner,
            max_grad_norm=cfg.max_grad_norm,
        )
        self._buffer = ReplayBuffer(
            capacity_per_player=cfg.buffer_capacity_per_player,
            seed=cfg.seed + 101,
        )

    # ── LearnerProtocol ──

    def drain_message(self, msg: dict) -> int:
        self._buffer.push_stacked(msg["stacked"], msg["players"], msg["returns"])
        return len(msg["players"])

    def is_ready(self) -> bool:
        cfg = self._cfg
        return all(self._buffer.size(p) >= cfg.buffer_min_size for p in range(4))

    def step(self) -> StepResult | None:
        cfg = self._cfg
        for net in self._learner.q_nets.values():
            net.train()
        losses = self._learner.update(self._buffer, cfg.batch_size)
        if not losses:
            return None
        return StepResult(loss={str(p): v for p, v in losses.items()}, phase={})

    def publish(self, weight_dir: Path, version: int, updates: int) -> None:
        publish_weights(self._learner.q_nets, weight_dir, version, updates)

    def checkpoint(
        self, path: Path, updates: int, *, save_type: CheckpointSaveType = "full"
    ) -> None:
        save_checkpoint_base(
            path,
            self._learner.q_nets,
            self._cfg,
            updates,
            learner_state=self._learner.state_dict(),
            replay_state=self._buffer.state_dict(),
            rng_state=capture_torch_rng_state(),
            save_type=save_type,
        )

    def resume(self, path: Path) -> int:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        for p in range(4):
            net = self._learner.q_nets[p]
            getattr(net, "_orig_mod", net).load_state_dict(ckpt["q_nets"][p])
        self._learner.load_state_dict(ckpt.get("learner_state"))
        if ckpt.get("replay_state") is not None:
            self._buffer.load_state_dict(ckpt["replay_state"])
        restore_torch_rng_state(ckpt.get("rng_state"))
        return int(ckpt.get("total_updates", ckpt.get("episode", 0)))

    def buffer_size(self) -> int:
        return self._buffer.total_size()

    def format_log_row(self, result: StepResult, timing: dict) -> dict:
        cfg = self._cfg
        upd = timing["updates"]
        interval_dt = timing["interval_dt"]
        interval_upd = timing["interval_upd"]
        upd_per_sec = interval_upd / interval_dt if interval_dt > 0 else 0.0
        session_upd = upd - timing["session_start_updates"]
        elapsed = timing["elapsed_s"]
        return {
            "updates":                 upd,
            "version":                 timing["version"],
            "buffer_total":            self._buffer.total_size(),
            "buffer_per_player":       {p: self._buffer.size(p) for p in range(4)},
            "loss":                    {k: round(v, 6) for k, v in result.loss.items()},
            "elapsed_s":               round(elapsed, 1),
            "upd_per_sec":             round(upd_per_sec, 3),
            "samples_per_sec":         round(upd_per_sec * cfg.batch_size, 1),
            "cum_upd_per_sec":         round(session_upd / elapsed if elapsed > 0 else 0.0, 3),
            "queue_depth":             timing["queue_depth"],
            "gpu_mem_gb":              timing["gpu_mem_gb"],
            "drained_since_last_log":  timing["drained_since_log"],
            "cumulative_drained":      timing["cumulative_drained"],
            "fresh_samples_total":     timing["fresh_samples_total"],
            "actor_rate_samp_per_sec": round(timing["ema_actor_rate"], 1),
            "replay_interval":         timing["interval_replay"],
            "replay_cumulative":       timing["cum_replay"],
            "throttle_sleeps":         timing["n_throttle_sleeps"],
            "throttle_sleep_s":        round(timing["throttle_sleep_total_s"], 2),
        }

    def log_progress(self, logger: Any, result: StepResult, timing: dict) -> None:
        cfg = self._cfg
        interval_dt = timing["interval_dt"]
        interval_upd = timing["interval_upd"]
        upd_per_sec = interval_upd / interval_dt if interval_dt > 0 else 0.0
        samp_per_sec = upd_per_sec * cfg.batch_size
        ir = timing["interval_replay"]
        cr = timing["cum_replay"]
        target = cfg.total_updates_target or cfg.checkpoint_every_updates
        remaining = max(0, target - timing["updates"])
        eta_s = remaining / upd_per_sec if upd_per_sec > 0 else 0.0
        eta_h, eta_m = divmod(int(eta_s), 3600)[0], divmod(int(eta_s), 60)[0] % 60
        gpu = timing["gpu_mem_gb"]
        logger.info(
            "updates=%d ver=%d buf=%d loss=%s  %.0f samp/s (%.2f upd/s)"
            "  replay=%.1fx(int)/%.1fx(cum)  q=%d gpu=%sGB drained=%d  ETA %dh%02dm",
            timing["updates"], timing["version"], self._buffer.total_size(),
            " ".join(f"p{p}={v:.4f}" for p, v in sorted(result.loss.items())),
            samp_per_sec, upd_per_sec,
            ir if ir is not None else -1.0,
            cr if cr is not None else -1.0,
            timing["queue_depth"],
            f"{gpu:.2f}" if gpu is not None else "n/a",
            timing["drained_since_log"],
            eta_h, eta_m,
        )

    def on_idle(self, drained: int) -> None:
        pass  # seat loop doesn't need to yield on idle

    def on_profiler_report(self, logger: Any, interval_dt: float, interval_upd: int) -> None:
        if self._learner.prof.enabled and interval_upd > 0:
            report = self._learner.prof.report(
                wall_s=interval_dt,
                n_events=interval_upd,
                event_label="updates",
            )
            if report:
                for line in report.splitlines():
                    logger.info(line)


class _DartAdapter:
    """LearnerProtocol adapter for the role-aware Dart model."""

    def __init__(self, cfg) -> None:
        from ..config import dart_qnet_config
        from ..model.q_network import DartQNet

        self._cfg = cfg
        q_net = DartQNet(dart_qnet_config(cfg))

        if cfg.device == "cuda":
            compile_mode = cfg.compile_mode or "default"
            q_net = torch.compile(q_net, mode=compile_mode)

        self._learner = DartLearner(
            q_net=q_net,
            lr=cfg.lr,
            device=cfg.device,
            use_bf16=cfg.use_bf16_learner,
            max_grad_norm=cfg.max_grad_norm,
        )
        capacity = cfg.buffer_capacity or 4 * cfg.buffer_capacity_per_player
        self._buffer = RoleAwareReplayBuffer(
            capacity=capacity,
            seed=cfg.seed + 101,
        )
        self._compile_warn = None

    # ── LearnerProtocol ──

    def drain_message(self, msg: dict) -> int:
        tag_keys = [name for name, _ in RoleAwareReplayBuffer._TAG_FIELDS]
        tags_in_msg = {k: msg[k] for k in tag_keys if k in msg}
        self._buffer.push_stacked(
            msg["stacked"],
            msg["returns"],
            msg.get("buckets"),
            tags=tags_in_msg or None,
        )
        return len(msg["returns"])

    def is_ready(self) -> bool:
        cfg = self._cfg
        return min(self._buffer.size_by_seat().values()) >= max(
            cfg.buffer_min_size, cfg.batch_size // 4
        )

    def step(self) -> StepResult | None:
        cfg = self._cfg
        metrics = self._learner.update(
            buffer=self._buffer,
            batch_size=cfg.batch_size,
            replay_mix=cfg.replay_mix or None,
            max_forced_k1_replay_frac=cfg.max_forced_k1_replay_frac,
        )
        if metrics is None:
            return None
        phase_payload = {
            k: v for k, v in metrics.items()
            if any(k.startswith(p) for p in PHASE_KEY_PREFIXES)
        }
        scalar_loss = {
            k: (round(v, 6) if isinstance(v, (int, float)) and v is not None else v)
            for k, v in metrics.items()
            if k not in phase_payload
        }
        return StepResult(loss=scalar_loss, phase=phase_payload)

    def publish(self, weight_dir: Path, version: int, updates: int) -> None:
        publish_weights_dart(self._learner.q_net, weight_dir, version, updates)

    def checkpoint(
        self, path: Path, updates: int, *, save_type: CheckpointSaveType = "full"
    ) -> None:
        save_checkpoint_dart(
            path,
            self._learner.q_net,
            self._cfg,
            updates,
            learner_state=self._learner.state_dict(),
            replay_state=self._buffer.state_dict(),
            rng_state=capture_torch_rng_state(),
            save_type=save_type,
        )

    def resume(self, path: Path) -> int:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        net = self._learner.q_net
        getattr(net, "_orig_mod", net).load_state_dict(ckpt["q_net"])
        self._learner.load_state_dict(ckpt.get("learner_state"))
        if ckpt.get("replay_state") is not None:
            self._buffer.load_state_dict(ckpt["replay_state"])
        restore_torch_rng_state(ckpt.get("rng_state"))
        return int(ckpt.get("total_updates", ckpt.get("episode", 0)))

    def buffer_size(self) -> int:
        return self._buffer.size()

    def format_log_row(self, result: StepResult, timing: dict) -> dict:
        interval_dt = timing["interval_dt"]
        interval_upd = timing["interval_upd"]
        upd_per_sec = interval_upd / interval_dt if interval_dt > 0 else 0.0
        return {
            "updates":                 timing["updates"],
            "version":                 timing["version"],
            "buffer_total":            self._buffer.size(),
            "buffer_per_player":       self._buffer.size_by_seat(),
            "loss":                    result.loss,
            "phase":                   result.phase,
            "elapsed_s":               round(timing["elapsed_s"], 1),
            "upd_per_sec":             round(upd_per_sec, 3),
            "samples_per_sec":         round(upd_per_sec * self._cfg.batch_size, 1),
            "queue_depth":             timing["queue_depth"],
            "drained_since_last_log":  timing["drained_since_log"],
            "cumulative_drained":      timing["cumulative_drained"],
            "fresh_samples_total":     timing["fresh_samples_total"],
            "actor_rate_samp_per_sec": round(timing["ema_actor_rate"], 1),
            "replay_interval":         timing["interval_replay"],
            "replay_cumulative":       timing["cum_replay"],
            "throttle_sleeps":         timing["n_throttle_sleeps"],
            "throttle_sleep_s":        round(timing["throttle_sleep_total_s"], 2),
        }

    def log_progress(self, logger: Any, result: StepResult, timing: dict) -> None:
        interval_dt = timing["interval_dt"]
        interval_upd = timing["interval_upd"]
        upd_per_sec = interval_upd / interval_dt if interval_dt > 0 else 0.0
        logger.info(
            "updates=%d ver=%d buf=%d loss=%.4f %.2f upd/s drained=%d",
            timing["updates"],
            timing["version"],
            self._buffer.size(),
            result.loss.get("loss", 0.0) or 0.0,
            upd_per_sec,
            timing["drained_since_log"],
        )

    def on_idle(self, drained: int) -> None:
        if drained == 0:
            time.sleep(0.001)

    def on_profiler_report(self, logger: Any, interval_dt: float, interval_upd: int) -> None:
        pass  # DartLearner has no per-phase profiler


# ─── Factory ──────────────────────────────────────────────────────────────────


def _make_adapter(cfg) -> LearnerProtocol:
    from ..config import MODEL_TYPE_DART

    if cfg.model_type == MODEL_TYPE_DART:
        return _DartAdapter(cfg)
    return _SeatAdapter(cfg)


# ─── Unified learner process entry-point ──────────────────────────────────────


def learner_loop(
    cfg_dict:          dict,
    sample_queue:      "mp.Queue[bytes]",
    stop_event:        "mp.Event",
    weight_dir:        Path,
    run_dir:           Path,
    update_counter:    "mp.Value | None" = None,
    weights_ready:     "mp.Event | None" = None,
    resume_checkpoint: Path | None = None,
) -> None:
    """Central learner process for faithful persistent actor-learner DMC.

    Dispatches to the appropriate ``LearnerProtocol`` adapter based on
    ``cfg.model_type``, then runs a single unified loop that:

      1. Drains actor samples from the queue into the adapter's replay buffer.
      2. Throttles the gradient rate when replay ratio exceeds the configured cap.
      3. Takes one gradient step when the buffer is warm.
      4. Periodically publishes weights, checkpoints, and logs metrics.
    """
    # Lazy import — this function runs in a spawned child process.
    from ..config import TrainConfig

    cfg = TrainConfig.from_flat_dict(cfg_dict)
    seed_everything(cfg.seed + 1)
    layout     = RunLayout(Path(run_dir))
    weight_dir = Path(weight_dir)

    logger, _ = setup_run_logging(
        layout.run_dir, layout.learner_log.name, name="dart.learner"
    )
    metrics_writer = jsonl_writer(layout.metrics_jsonl)

    adapter = _make_adapter(cfg)
    for msg in getattr(adapter, "_startup_messages", []):
        logger.info("%s", msg)

    total_updates = 0
    if resume_checkpoint is not None:
        total_updates = adapter.resume(Path(resume_checkpoint), )
        version = total_updates // cfg.publish_interval_updates
        logger.info("resumed from %s  (total_updates=%d)", resume_checkpoint, total_updates)
    else:
        version = 0

    if update_counter is not None:
        update_counter.value = total_updates

    adapter.publish(weight_dir, version, total_updates)
    logger.info("initial weights published (version %d)", version)
    if weights_ready is not None:
        weights_ready.set()

    t0 = time.time()
    session_start_updates = total_updates
    last_result: StepResult | None = None
    last_ticked = -1
    drained_since_log = 0
    cumulative_drained = 0
    fresh_samples_total = 0
    last_log_t = t0
    last_log_updates = total_updates
    last_log_fresh_samples = 0
    ema_actor_rate = 0.0
    n_throttle_sleeps = 0
    throttle_sleep_total_s = 0.0

    while not stop_event.is_set():
        # 1. Drain sample queue into replay buffer
        drained = 0
        while drained < cfg.max_drain_batches_per_loop:
            try:
                msg = sample_queue.get_nowait()
                fresh_samples_total += adapter.drain_message(msg)
                drained += 1
            except Exception:
                break
        drained_since_log += drained
        cumulative_drained += drained

        # 2. Replay-ratio throttle
        if (
            cfg.target_replay_ratio > 0
            and total_updates > session_start_updates
            and fresh_samples_total > 0
        ):
            session_uses = (total_updates - session_start_updates) * cfg.batch_size
            cum_replay = session_uses / max(fresh_samples_total, 1)
            if cum_replay > cfg.max_replay_ratio:
                extra_fresh = (session_uses / cfg.target_replay_ratio) - fresh_samples_total
                if extra_fresh > 0 and ema_actor_rate > 0:
                    sleep_s = min(extra_fresh / ema_actor_rate, cfg.max_throttle_sleep_s)
                    if sleep_s > 0.001:
                        time.sleep(sleep_s)
                        n_throttle_sleeps += 1
                        throttle_sleep_total_s += sleep_s
                continue

        # 3. Gradient step
        if adapter.is_ready():
            result = adapter.step()
            if result is not None:
                last_result = result
                total_updates += 1
                if update_counter is not None:
                    update_counter.value = total_updates

        # 4–6 only fire when total_updates advanced to a new tick
        if total_updates == last_ticked or total_updates == 0:
            adapter.on_idle(drained)
            continue
        last_ticked = total_updates

        # 4. Publish weights
        if total_updates % cfg.publish_interval_updates == 0:
            version += 1
            adapter.publish(weight_dir, version, total_updates)
            logger.debug("weights published version=%d", version)

        # 5. Checkpoint
        if total_updates % cfg.checkpoint_every_updates == 0:
            ckpt = layout.update_checkpoint(total_updates)
            adapter.checkpoint(ckpt, total_updates, save_type=cfg.checkpoint_save_type)
            logger.info("checkpoint → %s (%s)", ckpt, cfg.checkpoint_save_type)

        # 6. Metrics log
        if total_updates % cfg.log_every_updates == 0:
            now = time.time()
            elapsed = now - t0
            interval_dt = max(1e-6, now - last_log_t)
            interval_upd = total_updates - last_log_updates
            interval_unique = fresh_samples_total - last_log_fresh_samples

            interval_replay: float | None = (
                round((interval_upd * cfg.batch_size) / interval_unique, 2)
                if interval_unique > 0 else None
            )
            session_updates = total_updates - session_start_updates
            cum_replay: float | None = (
                round((session_updates * cfg.batch_size) / fresh_samples_total, 2)
                if fresh_samples_total > 0 else None
            )

            interval_actor_rate = interval_unique / interval_dt if interval_dt > 0 else 0.0
            if ema_actor_rate <= 0:
                ema_actor_rate = interval_actor_rate
            else:
                ema_actor_rate = 0.5 * ema_actor_rate + 0.5 * interval_actor_rate

            try:
                queue_depth = sample_queue.qsize()
            except (NotImplementedError, OSError):
                queue_depth = -1
            gpu_mem_gb = (
                round(torch.cuda.memory_allocated() / 1e9, 3)
                if cfg.device == "cuda" and torch.cuda.is_available() else None
            )

            timing = {
                "updates":               total_updates,
                "version":               version,
                "elapsed_s":             elapsed,
                "interval_dt":           interval_dt,
                "interval_upd":          interval_upd,
                "queue_depth":           queue_depth,
                "gpu_mem_gb":            gpu_mem_gb,
                "drained_since_log":     drained_since_log,
                "cumulative_drained":    cumulative_drained,
                "fresh_samples_total":   fresh_samples_total,
                "ema_actor_rate":        ema_actor_rate,
                "interval_replay":       interval_replay,
                "cum_replay":            cum_replay,
                "n_throttle_sleeps":     n_throttle_sleeps,
                "throttle_sleep_total_s": throttle_sleep_total_s,
                "session_start_updates": session_start_updates,
            }

            if last_result is not None:
                row = adapter.format_log_row(last_result, timing)
                metrics_writer.write(row)
                adapter.log_progress(logger, last_result, timing)
                adapter.on_profiler_report(logger, interval_dt, interval_upd)

            drained_since_log = 0
            last_log_t = now
            last_log_updates = total_updates
            last_log_fresh_samples = fresh_samples_total

    # Final checkpoint on clean shutdown
    if total_updates > 0:
        adapter.checkpoint(layout.final_checkpoint, total_updates, save_type="full")
    logger.info("learner stopped after %d updates", total_updates)
    metrics_writer.close()


__all__ = [
    # Protocol surface
    "StepResult",
    "LearnerProtocol",
    # Concrete classes (re-exported for convenience)
    "SeatLearner",
    "DartLearner",
    # Entry point
    "learner_loop",
]
