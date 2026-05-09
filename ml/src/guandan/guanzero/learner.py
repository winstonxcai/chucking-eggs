"""Per-position MSE update against MC returns.

Four optimizers — one per Q-network — keep the four position-specific
networks fully independent (paper §4.2).

Also contains the learner *process* entry-point (``learner_loop``) for
the faithful persistent actor-learner DMC setup, plus atomic weight
publishing helpers used by both the learner and the actors.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Mapping

import torch
import torch.nn.functional as F

from .buffer import ReplayBuffer, RoleAwareReplayBuffer
from .checkpoint import (
    WeightSnapshot,
    migrate_state_dict,
    save_checkpoint_base,
    save_checkpoint_shared,
    unwrap_compiled,
)
from .profiler import PhaseProfiler
from .q_network import GuanZeroQNet, SharedHeadQNet, init_seat_nets


# ─── Learner class (single-process) ──────────────────────


class Learner:
    def __init__(
        self,
        q_nets: Mapping[int, GuanZeroQNet],
        lr: float = 1e-4,
        device: torch.device | str = "cpu",
        use_bf16: bool = False,
        max_grad_norm: float = 10.0,
    ) -> None:
        self.device = torch.device(device)
        self.q_nets = {p: q_nets[p].to(self.device) for p in range(4)}
        self.optims = {
            p: torch.optim.Adam(self.q_nets[p].parameters(), lr=lr, foreach=True)
            for p in range(4)
        }
        self.use_bf16 = use_bf16 and self.device.type == "cuda"
        self.max_grad_norm = max_grad_norm
        # One stream per position so CUDA can schedule all 4 forward+backward
        # passes concurrently. Not used on MPS (no multi-stream support).
        self.streams: dict[int, torch.cuda.Stream] | None = (
            {p: torch.cuda.Stream(device=self.device) for p in range(4)}
            if self.device.type == "cuda" else None
        )
        # Per-phase profiling. Enabled with GUANZERO_LEARNER_PROFILE=1.
        # CUDA sync on each phase boundary makes wall time accurate but
        # destroys async stream overlap — only use for diagnosis.
        self.prof = PhaseProfiler(
            enabled=os.environ.get("GUANZERO_LEARNER_PROFILE") == "1",
            device=self.device if self.device.type == "cuda" else None,
        )

    def update(
        self,
        buffer: ReplayBuffer,
        batch_size: int,
    ) -> dict[int, float]:
        """One gradient step per position, all 4 in parallel on CUDA.

        Collates all positions first so H2D copies can overlap, then launches
        forward+backward on separate streams. Loss tensors are read after a
        single synchronize() to avoid per-position CPU stalls.
        """
        # --- sample + H2D ---
        with self.prof.time("sample+h2d", sync=True):
            batches: dict[int, tuple] = {}
            for p in range(4):
                res = buffer.sample_batch_for_player(p, batch_size, device=self.device)
                if res is not None:
                    batches[p] = res

        if not batches:
            return {}

        # --- forward ---
        q_preds: dict[int, torch.Tensor] = {}
        targets_d: dict[int, torch.Tensor] = {}
        with self.prof.time("forward", sync=True):
            for p, (batch, targets) in batches.items():
                ctx = (torch.cuda.stream(self.streams[p])
                       if self.streams else contextlib.nullcontext())
                with ctx:
                    self.q_nets[p].train()
                    if self.use_bf16:
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            q_preds[p] = self.q_nets[p](batch)
                    else:
                        q_preds[p] = self.q_nets[p](batch)
                targets_d[p] = targets

        # --- loss + backward + step ---
        loss_tensors: dict[int, torch.Tensor] = {}
        with self.prof.time("backward+step", sync=True):
            for p, q_pred in q_preds.items():
                ctx = (torch.cuda.stream(self.streams[p])
                       if self.streams else contextlib.nullcontext())
                with ctx:
                    if self.use_bf16:
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            loss = F.mse_loss(q_pred, targets_d[p])
                    else:
                        loss = F.mse_loss(q_pred, targets_d[p])
                    loss_tensors[p] = loss
                    if torch.isfinite(loss):
                        self.optims[p].zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            self.q_nets[p].parameters(), self.max_grad_norm
                        )
                        self.optims[p].step()

        # --- final stream sync + read losses ---
        with self.prof.time("final_sync"):
            if self.streams:
                torch.cuda.synchronize(self.device)
        with self.prof.time("loss_item"):
            out = {p: float(lt.item()) for p, lt in loss_tensors.items()}
        return out


def _grad_norm_of(module: torch.nn.Module) -> torch.Tensor:
    total = torch.zeros((), device=next(module.parameters()).device)
    for p in module.parameters():
        if p.grad is not None:
            total = total + p.grad.detach().norm().pow(2)
    return total.sqrt()


class SharedHeadLearner:
    """Learner for a single SharedHeadQNet with one shared optimizer."""

    def __init__(
        self,
        q_net: SharedHeadQNet,
        lr: float = 1e-4,
        device: torch.device | str = "cpu",
        use_bf16: bool = False,
        max_grad_norm: float = 10.0,
    ) -> None:
        self.device = torch.device(device)
        self.q_net = q_net.to(self.device)
        self.opt = torch.optim.Adam(self.q_net.parameters(), lr=lr, foreach=True)
        self.use_bf16 = use_bf16 and self.device.type == "cuda"
        self.max_grad_norm = max_grad_norm

    def update(
        self,
        buffer: RoleAwareReplayBuffer,
        batch_size: int,
    ) -> dict[str, float] | None:
        """One balanced gradient step, or None if any seat is cold."""
        sizes = buffer.size_by_seat()
        min_per_seat = batch_size // 4
        if min(sizes.values()) < min_per_seat:
            return None

        batch, targets = buffer.sample_batch_balanced(batch_size, self.device)
        seat_ids = batch["seat_id"].long()
        self.q_net.train()
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=self.use_bf16,
        ):
            preds = self.q_net(batch)
            loss = F.mse_loss(preds, targets)

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm_total = torch.nn.utils.clip_grad_norm_(
            self.q_net.parameters(),
            self.max_grad_norm,
        )
        self.opt.step()

        with torch.no_grad():
            metrics: dict[str, float] = {
                "loss": float(loss.item()),
                "grad_norm": float(grad_norm_total.item()),
                "grad_norm_trunk": float(_grad_norm_of(self.q_net.trunk).item()),
            }
            for k in range(4):
                mask = seat_ids == k
                metrics[f"sample_count_seat_{k}"] = float(mask.sum().item())
                if mask.any():
                    seat_preds = preds[mask]
                    seat_targets = targets[mask]
                    metrics[f"loss_seat_{k}"] = float(F.mse_loss(seat_preds, seat_targets).item())
                    metrics[f"q_mean_seat_{k}"] = float(seat_preds.mean().item())
                    metrics[f"q_std_seat_{k}"] = float(seat_preds.std(unbiased=False).item())
                metrics[f"grad_norm_head_{k}"] = float(_grad_norm_of(self.q_net.heads[k]).item())
        return metrics


# ─── Metrics row ─────────────────────────────────────────


@dataclasses.dataclass
class LearnerMetricsRow:
    """One row written to ``metrics_learner.jsonl`` per log interval."""
    updates:                 int
    version:                 int
    buffer_total:            int
    buffer_per_player:       dict[int, int]
    loss:                    dict[str, float]
    elapsed_s:               float
    upd_per_sec:             float
    samples_per_sec:         float
    cum_upd_per_sec:         float
    queue_depth:             int
    gpu_mem_gb:              float | None
    drained_since_last_log:  int
    cumulative_drained:      int
    fresh_samples_total:     int
    actor_rate_samp_per_sec: float
    replay_interval:         float | None
    replay_cumulative:       float | None
    throttle_sleeps:         int
    throttle_sleep_s:        float


# ─── Atomic weight publishing ─────────────────────────────


def publish_weights(q_nets: dict, weight_dir: Path, version: int, updates: int = 0) -> None:
    """Atomically write global Q-net weights to disk.

    Actors poll ``weight_dir/latest.txt`` for the current version number,
    then load ``weight_dir/weights_{version}.pt``. Both writes are made
    atomic via ``os.replace`` (POSIX rename — no partial-read window).
    """
    weight_dir.mkdir(parents=True, exist_ok=True)
    tmp   = weight_dir / f"weights_{version}.tmp"
    final = weight_dir / f"weights_{version}.pt"

    torch.save(
        {
            "version": version,
            "updates": updates,
            "state_dicts": {
                p: {k: v.detach().cpu()
                    for k, v in unwrap_compiled(q_nets[p]).state_dict().items()}
                for p in range(4)
            },
        },
        tmp,
    )
    os.replace(tmp, final)                     # atomic on POSIX + macOS

    ver_tmp = weight_dir / "latest.tmp"
    ver_tmp.write_text(f"{version} {updates}")
    os.replace(ver_tmp, weight_dir / "latest.txt")  # atomic

    # Clean up all prior weight files — only the current version is needed.
    for stale in weight_dir.glob("weights_*.pt"):
        if stale != final:
            stale.unlink(missing_ok=True)
    for stale in weight_dir.glob("weights_*.tmp"):
        stale.unlink(missing_ok=True)


def publish_weights_shared(q_net: SharedHeadQNet, weight_dir: Path, version: int, updates: int = 0) -> None:
    """Atomically write one shared-head Q-net snapshot to disk."""
    weight_dir.mkdir(parents=True, exist_ok=True)
    tmp = weight_dir / f"weights_{version}.tmp"
    final = weight_dir / f"weights_{version}.pt"
    torch.save(
        {
            "version": version,
            "updates": updates,
            "state_dicts": {
                "shared": {
                    k: v.detach().cpu()
                    for k, v in unwrap_compiled(q_net).state_dict().items()
                }
            },
        },
        tmp,
    )
    os.replace(tmp, final)

    ver_tmp = weight_dir / "latest.tmp"
    ver_tmp.write_text(f"{version} {updates}")
    os.replace(ver_tmp, weight_dir / "latest.txt")

    for stale in weight_dir.glob("weights_*.pt"):
        if stale != final:
            stale.unlink(missing_ok=True)
    for stale in weight_dir.glob("weights_*.tmp"):
        stale.unlink(missing_ok=True)


def read_latest_metadata(weight_dir: Path) -> tuple[int, int] | None:
    """Cheaply read (version, updates) from ``latest.txt`` without torch.load.

    Returns ``None`` if no published weights yet, or the file is unreadable.
    Format: ``"<version> <updates>"`` (post-2026-05). Falls back to bare int
    for backward compatibility with old runs that wrote just the version.
    """
    ver_path = weight_dir / "latest.txt"
    if not ver_path.exists():
        return None
    try:
        parts = ver_path.read_text().strip().split()
        version = int(parts[0])
        updates = int(parts[1]) if len(parts) > 1 else 0
        return version, updates
    except (ValueError, IndexError):
        return None


def load_latest_weights(weight_dir: Path) -> WeightSnapshot | None:
    """Read the latest published version and state dicts.

    Returns ``None`` if weights have not been published yet or the latest
    snapshot is temporarily unreadable during an atomic replacement.
    """
    meta = read_latest_metadata(weight_dir)
    if meta is None:
        return None
    version, _ = meta
    try:
        payload = torch.load(
            weight_dir / f"weights_{version}.pt",
            map_location="cpu",
            weights_only=True,
        )
        return WeightSnapshot(
            version=int(payload["version"]),
            state_dicts=payload["state_dicts"],
            updates=int(payload.get("updates", 0)),
        )
    except Exception:
        return None


# ─── Learner process entry-point ──────────────────────────


def learner_loop(
    cfg_dict:          dict,
    sample_queue,               # multiprocessing.Queue
    stop_event,                 # multiprocessing.Event
    weight_dir:        Path,
    run_dir:           Path,
    resume_checkpoint: Path | None = None,
) -> None:
    """Central learner process for faithful persistent actor-learner DMC.

    Owns the global Q-nets and replay buffer. Continuously:
      1. Drains actor samples from the queue into the replay buffer.
      2. Updates global Q-nets via MSE (one step per loop iteration).
      3. Periodically publishes updated weights to disk.
      4. Periodically checkpoints and logs metrics.
    """
    # Lazy import here — this function runs in a spawned child process where
    # the full guanzero package is re-imported from scratch.
    from .config import TrainConfig

    cfg = TrainConfig.from_flat_dict(cfg_dict)
    if cfg.model_type == "shared_heads":
        _learner_loop_shared(
            cfg=cfg,
            sample_queue=sample_queue,
            stop_event=stop_event,
            weight_dir=weight_dir,
            run_dir=run_dir,
            resume_checkpoint=resume_checkpoint,
        )
        return

    run_dir    = Path(run_dir)
    weight_dir = Path(weight_dir)

    # Dedicated log file for learner process
    log_path = run_dir / "learner.log"
    logger = logging.getLogger("guanzero.learner")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)-5s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, mode="a")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    if os.environ.get("GUANZERO_STREAM_LOGS") == "1":
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        sh.setLevel(logging.INFO)
        logger.addHandler(sh)

    # Free TF32 + cuDNN tuning on CUDA paths (Adam math, anything outside BF16 autocast).
    if cfg.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    q_nets  = init_seat_nets(cfg.qnet)
    compile_mode = cfg.compile_mode or "default"
    if compile_mode == "reduce-overhead" and cfg.device == "cuda":
        # Empirically: per-stream cudagraph_trees did NOT compose
        # with multi-stream parallelism — graphs share a CUDA memory pool that
        # serializes them. "default" + streams gave the best throughput.
        compile_mode = "default"
        logger.info("compile_mode downgraded reduce-overhead→default for multi-stream positions")
    if cfg.device == "cuda":
        for p, net in q_nets.items():
            q_nets[p] = torch.compile(net, mode=compile_mode)
    else:
        logger.info("torch.compile disabled for learner device=%s", cfg.device)
    learner = Learner(q_nets=q_nets, lr=cfg.lr, device=cfg.device,
                      use_bf16=cfg.use_bf16_learner,
                      max_grad_norm=cfg.max_grad_norm)
    buffer  = ReplayBuffer(capacity_per_player=cfg.buffer_capacity_per_player)

    version       = 0
    total_updates = 0
    last_losses: dict[int, float] = {}
    metrics_path  = run_dir / "metrics_learner.jsonl"
    last_ticked   = -1   # tracks the last total_updates value that triggered periodic actions

    # Optionally resume from a prior checkpoint
    if resume_checkpoint is not None:
        ckpt = torch.load(Path(resume_checkpoint), map_location="cpu", weights_only=True)
        for p in range(4):
            unwrap_compiled(q_nets[p]).load_state_dict(migrate_state_dict(ckpt["q_nets"][p]))
        total_updates = int(ckpt.get("episode", 0))
        version       = total_updates // cfg.publish_interval_updates
        logger.info("resumed from %s  (total_updates=%d)", resume_checkpoint, total_updates)

    # Publish initial weights so actors can start immediately
    publish_weights(q_nets, weight_dir, version, updates=total_updates)
    logger.info("initial weights published (version %d)", version)

    t0 = time.time()
    session_start_updates = total_updates  # for accurate upd/s on resumed runs
    drained_since_log = 0
    cumulative_drained = 0
    fresh_samples_total = 0          # exact sample count from drained messages
    last_log_t = t0
    last_log_updates = total_updates
    last_log_fresh_samples = 0
    ema_actor_rate = 0.0             # samples/sec, EMA across log intervals
    n_throttle_sleeps = 0
    throttle_sleep_total_s = 0.0
    while not stop_event.is_set():
        # 1. Drain sample queue into replay buffer (pre-stacked actor messages)
        drained = 0
        while drained < cfg.max_drain_batches_per_loop:
            try:
                msg = sample_queue.get_nowait()
                buffer.push_stacked(msg["stacked"], msg["players"], msg["returns"])
                # Exact sample count for replay-ratio bookkeeping.
                # players is a length-N int8 array; len() == # samples in this push.
                fresh_samples_total += len(msg["players"])
                drained += 1
            except Exception:
                break
        drained_since_log += drained
        cumulative_drained += drained

        # 2a. Replay-ratio controller: throttle the learner if it's running
        # ahead of actor production. Disabled by default (target_replay_ratio=0).
        if (
            cfg.target_replay_ratio > 0
            and total_updates > session_start_updates
            and fresh_samples_total > 0
        ):
            session_uses = (total_updates - session_start_updates) * cfg.batch_size
            cum_replay = session_uses / max(fresh_samples_total, 1)
            if cum_replay > cfg.max_replay_ratio:
                # Sleep just long enough for actors to produce enough fresh data
                # to bring cum_replay down to target_replay_ratio. Use the EMA
                # actor rate (computed at log time) to size the sleep.
                target_uses = cfg.target_replay_ratio * fresh_samples_total
                # We want to wait until session_uses == target_uses, but
                # target_uses depends on future fresh samples. Approximate:
                # extra_fresh_needed = (session_uses / target_replay_ratio) - fresh_samples_total
                extra_fresh_needed = (session_uses / cfg.target_replay_ratio) - fresh_samples_total
                if extra_fresh_needed > 0 and ema_actor_rate > 0:
                    sleep_s = extra_fresh_needed / ema_actor_rate
                    sleep_s = min(sleep_s, cfg.max_throttle_sleep_s)
                    if sleep_s > 0.001:
                        time.sleep(sleep_s)
                        n_throttle_sleeps += 1
                        throttle_sleep_total_s += sleep_s
                # Re-check: skip the gradient update so next loop drains more
                continue

        # 2. Gradient update when every position's buffer is warm
        if all(buffer.size(p) >= cfg.buffer_min_size for p in range(4)):
            for net in q_nets.values():
                net.train()
            new_losses = learner.update(
                buffer=buffer,
                batch_size=cfg.batch_size,
            )
            if new_losses:
                last_losses = new_losses
                total_updates += 1

        # 3–5 only fire when total_updates actually advanced to a new tick
        if total_updates == last_ticked or total_updates == 0:
            continue
        last_ticked = total_updates

        # 3. Publish updated weights periodically
        if total_updates % cfg.publish_interval_updates == 0:
            version += 1
            publish_weights(q_nets, weight_dir, version, updates=total_updates)
            logger.debug("weights published version=%d", version)

        # 4. Checkpoint
        if total_updates % cfg.checkpoint_every_updates == 0:
            ckpt = run_dir / "checkpoints" / f"update_{total_updates:08d}.pt"
            save_checkpoint_base(ckpt, q_nets, cfg, total_updates)
            logger.info("checkpoint → %s", ckpt)

        # 5. Metrics log
        if total_updates % cfg.log_every_updates == 0:
            now = time.time()
            elapsed = now - t0
            # Interval rate (since last log) — converges to true steady-state quickly,
            # unlike cumulative rate which is dragged down by warmup.
            interval_dt = max(1e-6, now - last_log_t)
            interval_upd = total_updates - last_log_updates
            interval_upd_per_sec = interval_upd / interval_dt
            interval_samp_per_sec = interval_upd_per_sec * cfg.batch_size
            # Cumulative for ETA only (steady ETA estimate)
            session_updates = total_updates - session_start_updates
            cum_upd_per_sec = session_updates / elapsed if elapsed > 0 else 0.0
            target = cfg.total_updates_target or cfg.checkpoint_every_updates
            remaining = max(0, target - total_updates)
            eta_s = remaining / interval_upd_per_sec if interval_upd_per_sec > 0 else 0.0
            eta_h, eta_m = divmod(int(eta_s), 3600)[0], divmod(int(eta_s), 60)[0] % 60
            try:
                queue_depth = sample_queue.qsize()
            except (NotImplementedError, OSError):
                queue_depth = -1
            gpu_mem_gb = (
                round(torch.cuda.memory_allocated() / 1e9, 3)
                if cfg.device == "cuda" and torch.cuda.is_available() else None
            )
            # Replay ratio: use EXACT sample counts from drained messages
            # (was approximated by drained * actor_push_batch_size, which can
            # lag the actual production by up to one push batch per actor).
            interval_unique = fresh_samples_total - last_log_fresh_samples
            interval_replay = (
                (interval_upd * cfg.batch_size) / interval_unique
                if interval_unique > 0 else float("inf")
            )
            cum_unique = fresh_samples_total
            cum_replay = (
                (session_updates * cfg.batch_size) / cum_unique
                if cum_unique > 0 else float("inf")
            )

            # EMA actor rate (samples/sec). Used by the throttle to size sleeps.
            # 0.5 weight on this interval keeps the EMA responsive to startup
            # transients without being too noisy.
            interval_actor_rate = interval_unique / interval_dt if interval_dt > 0 else 0.0
            if ema_actor_rate <= 0:
                ema_actor_rate = interval_actor_rate
            else:
                ema_actor_rate = 0.5 * ema_actor_rate + 0.5 * interval_actor_rate
            row = LearnerMetricsRow(
                updates                 = total_updates,
                version                 = version,
                buffer_total            = buffer.total_size(),
                buffer_per_player       = {p: buffer.size(p) for p in range(4)},
                loss                    = {str(p): round(v, 6) for p, v in last_losses.items()},
                elapsed_s               = round(elapsed, 1),
                upd_per_sec             = round(interval_upd_per_sec, 3),
                samples_per_sec         = round(interval_samp_per_sec, 1),
                cum_upd_per_sec         = round(cum_upd_per_sec, 3),
                queue_depth             = queue_depth,
                gpu_mem_gb              = gpu_mem_gb,
                drained_since_last_log  = drained_since_log,
                cumulative_drained      = cumulative_drained,
                fresh_samples_total     = fresh_samples_total,
                actor_rate_samp_per_sec = round(ema_actor_rate, 1),
                replay_interval         = round(interval_replay, 2) if interval_replay != float("inf") else None,
                replay_cumulative       = round(cum_replay, 2) if cum_replay != float("inf") else None,
                throttle_sleeps         = n_throttle_sleeps,
                throttle_sleep_s        = round(throttle_sleep_total_s, 2),
            )
            with metrics_path.open("a") as f:
                f.write(json.dumps(dataclasses.asdict(row)) + "\n")
            logger.info(
                "updates=%d ver=%d buf=%d loss=%s  %.0f samp/s (%.2f upd/s)  replay=%.1fx(int)/%.1fx(cum)  q=%d gpu=%sGB drained=%d  ETA %dh%02dm",
                total_updates, version, buffer.total_size(),
                " ".join(f"p{p}={v:.4f}" for p, v in sorted(last_losses.items())),
                interval_samp_per_sec, interval_upd_per_sec,
                interval_replay if interval_replay != float("inf") else -1.0,
                cum_replay if cum_replay != float("inf") else -1.0,
                queue_depth,
                f"{gpu_mem_gb:.2f}" if gpu_mem_gb is not None else "n/a",
                drained_since_log,
                eta_h, eta_m,
            )
            if learner.prof.enabled and interval_upd > 0:
                report = learner.prof.report(
                    wall_s=interval_dt,
                    n_events=interval_upd,
                    event_label="updates",
                )
                if report:
                    for line in report.splitlines():
                        logger.info(line)
            drained_since_log = 0
            last_log_t = now
            last_log_updates = total_updates
            last_log_fresh_samples = fresh_samples_total

    # Final checkpoint on clean shutdown
    if total_updates > 0:
        save_checkpoint_base(
            run_dir / "checkpoints" / "final.pt",
            q_nets, cfg, total_updates,
        )
    logger.info("learner stopped after %d updates", total_updates)


def _learner_loop_shared(
    cfg,
    sample_queue,
    stop_event,
    weight_dir: Path,
    run_dir: Path,
    resume_checkpoint: Path | None = None,
) -> None:
    """Central learner process for role-aware shared-head training."""
    from .config import shared_head_qnet_config

    if cfg.inference.enabled:
        raise ValueError(
            "Inference server not supported for model_type='shared_heads'. "
            "Set use_inference_server: false in config."
        )

    run_dir = Path(run_dir)
    weight_dir = Path(weight_dir)
    log_path = run_dir / "learner.log"
    logger = logging.getLogger("guanzero.learner")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)-5s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, mode="a")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    if os.environ.get("GUANZERO_STREAM_LOGS") == "1":
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        sh.setLevel(logging.INFO)
        logger.addHandler(sh)

    q_net = SharedHeadQNet(shared_head_qnet_config(cfg))
    if cfg.device == "cuda":
        compile_mode = cfg.compile_mode or "default"
        q_net = torch.compile(q_net, mode=compile_mode)
    learner = SharedHeadLearner(
        q_net=q_net,
        lr=cfg.lr,
        device=cfg.device,
        use_bf16=cfg.use_bf16_learner,
        max_grad_norm=cfg.max_grad_norm,
    )
    capacity = cfg.buffer_capacity or 4 * cfg.buffer_capacity_per_player
    buffer = RoleAwareReplayBuffer(capacity=capacity)

    version = 0
    total_updates = 0
    last_metrics: dict[str, float] = {}
    metrics_path = run_dir / "metrics_learner.jsonl"
    last_ticked = -1

    if resume_checkpoint is not None:
        ckpt = torch.load(Path(resume_checkpoint), map_location="cpu", weights_only=True)
        unwrap_compiled(q_net).load_state_dict(ckpt["q_net"])
        total_updates = int(ckpt.get("episode", 0))
        version = total_updates // cfg.publish_interval_updates
        logger.info("resumed from %s  (total_updates=%d)", resume_checkpoint, total_updates)

    publish_weights_shared(q_net, weight_dir, version, updates=total_updates)
    logger.info("initial shared weights published (version %d)", version)

    t0 = time.time()
    session_start_updates = total_updates
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
        drained = 0
        while drained < cfg.max_drain_batches_per_loop:
            try:
                msg = sample_queue.get_nowait()
                buffer.push_stacked(msg["stacked"], msg["returns"])
                fresh_samples_total += len(msg["returns"])
                drained += 1
            except Exception:
                break
        drained_since_log += drained
        cumulative_drained += drained

        if (
            cfg.target_replay_ratio > 0
            and total_updates > session_start_updates
            and fresh_samples_total > 0
        ):
            session_uses = (total_updates - session_start_updates) * cfg.batch_size
            cum_replay = session_uses / max(fresh_samples_total, 1)
            if cum_replay > cfg.max_replay_ratio:
                extra_fresh_needed = (session_uses / cfg.target_replay_ratio) - fresh_samples_total
                if extra_fresh_needed > 0 and ema_actor_rate > 0:
                    sleep_s = min(extra_fresh_needed / ema_actor_rate, cfg.max_throttle_sleep_s)
                    if sleep_s > 0.001:
                        time.sleep(sleep_s)
                        n_throttle_sleeps += 1
                        throttle_sleep_total_s += sleep_s
                continue

        if min(buffer.size_by_seat().values()) >= max(cfg.buffer_min_size, cfg.batch_size // 4):
            metrics = learner.update(buffer=buffer, batch_size=cfg.batch_size)
            if metrics is not None:
                last_metrics = metrics
                total_updates += 1

        if total_updates == last_ticked or total_updates == 0:
            if drained == 0:
                time.sleep(0.001)
            continue
        last_ticked = total_updates

        if total_updates % cfg.publish_interval_updates == 0:
            version += 1
            publish_weights_shared(q_net, weight_dir, version, updates=total_updates)

        if total_updates % cfg.checkpoint_every_updates == 0:
            ckpt = run_dir / "checkpoints" / f"update_{total_updates:08d}.pt"
            save_checkpoint_shared(ckpt, q_net, cfg, total_updates)
            logger.info("checkpoint → %s", ckpt)

        if total_updates % cfg.log_every_updates == 0:
            now = time.time()
            interval_dt = max(1e-6, now - last_log_t)
            interval_upd = total_updates - last_log_updates
            interval_unique = fresh_samples_total - last_log_fresh_samples
            interval_replay = (
                (interval_upd * cfg.batch_size) / interval_unique
                if interval_unique > 0 else float("inf")
            )
            session_updates = total_updates - session_start_updates
            cum_replay = (
                (session_updates * cfg.batch_size) / fresh_samples_total
                if fresh_samples_total > 0 else float("inf")
            )
            interval_actor_rate = interval_unique / interval_dt if interval_dt > 0 else 0.0
            if ema_actor_rate <= 0:
                ema_actor_rate = interval_actor_rate
            else:
                ema_actor_rate = 0.5 * ema_actor_rate + 0.5 * interval_actor_rate
            row = {
                "updates": total_updates,
                "version": version,
                "buffer_total": buffer.size(),
                "buffer_per_player": buffer.size_by_seat(),
                "loss": {k: round(v, 6) for k, v in last_metrics.items()},
                "elapsed_s": round(now - t0, 1),
                "upd_per_sec": round(interval_upd / interval_dt, 3),
                "samples_per_sec": round((interval_upd / interval_dt) * cfg.batch_size, 1),
                "queue_depth": getattr(sample_queue, "qsize", lambda: -1)(),
                "drained_since_last_log": drained_since_log,
                "cumulative_drained": cumulative_drained,
                "fresh_samples_total": fresh_samples_total,
                "actor_rate_samp_per_sec": round(ema_actor_rate, 1),
                "replay_interval": round(interval_replay, 2) if interval_replay != float("inf") else None,
                "replay_cumulative": round(cum_replay, 2) if cum_replay != float("inf") else None,
                "throttle_sleeps": n_throttle_sleeps,
                "throttle_sleep_s": round(throttle_sleep_total_s, 2),
            }
            with metrics_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            logger.info(
                "updates=%d ver=%d buf=%d loss=%.4f %.2f upd/s drained=%d",
                total_updates,
                version,
                buffer.size(),
                last_metrics.get("loss", 0.0),
                interval_upd / interval_dt,
                drained_since_log,
            )
            drained_since_log = 0
            last_log_t = now
            last_log_updates = total_updates
            last_log_fresh_samples = fresh_samples_total

    if total_updates > 0:
        save_checkpoint_shared(
            run_dir / "checkpoints" / "final.pt",
            q_net,
            cfg,
            total_updates,
        )
    logger.info("shared learner stopped after %d updates", total_updates)


__all__ = [
    "Learner",
    "SharedHeadLearner",
    "LearnerMetricsRow",
    "publish_weights",
    "publish_weights_shared",
    "load_latest_weights",
    "learner_loop",
]
