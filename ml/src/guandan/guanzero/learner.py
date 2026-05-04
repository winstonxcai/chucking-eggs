"""Per-position MSE update against MC returns.

Four optimizers — one per Q-network — keep the four position-specific
networks fully independent (paper §4.2).

Also contains the learner *process* entry-point (``learner_loop``) for
the faithful persistent actor-learner DMC setup, plus atomic weight
publishing helpers used by both the learner and the actors.
"""

from __future__ import annotations

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

from .buffer import ReplayBuffer, collate
from .q_network import GuanZeroQNet, init_position_nets


# ─── Learner class (single-process) ──────────────────────


class Learner:
    def __init__(
        self,
        q_nets: Mapping[int, GuanZeroQNet],
        lr: float = 1e-4,
        device: torch.device | str = "cpu",
        use_bf16: bool = False,
    ) -> None:
        self.device = torch.device(device)
        self.q_nets = {p: q_nets[p].to(self.device) for p in range(4)}
        self.optims = {
            p: torch.optim.Adam(self.q_nets[p].parameters(), lr=lr, foreach=True)
            for p in range(4)
        }
        self.use_bf16 = use_bf16 and self.device.type == "cuda"
        # One dedicated CUDA stream per position — lets the 4 independent
        # forward+backward passes overlap on the GPU rather than running sequentially.
        # Only created on CUDA; CPU/MPS fall back to sequential.
        self.streams: list[torch.cuda.Stream] | None = (
            [torch.cuda.Stream() for _ in range(4)]
            if self.device.type == "cuda" else None
        )

    def update(
        self,
        buffer: ReplayBuffer,
        batch_size: int,
        min_buffer_size: int,
    ) -> dict[int, float]:
        """One gradient step per position, run in parallel on CUDA.

        On CUDA: all 4 forward+backward passes launch concurrently on separate
        streams, then a single synchronize() gates the optimizer steps. On CPU/MPS
        the positions are updated sequentially as before.
        """
        # Sample all ready positions up front (cheap CPU work before any GPU dispatch).
        ready: dict[int, tuple[dict, torch.Tensor]] = {}
        for p in range(4):
            if buffer.size(p) < min_buffer_size:
                continue
            samples = buffer.sample_for_player(p, batch_size)
            if samples:
                ready[p] = collate(samples, device=self.device)

        if not ready:
            return {}

        losses: dict[int, float] = {}

        if self.streams is not None:
            # ── Parallel CUDA streams ────────────────────────────────────────
            # The 4 Q-nets are fully independent (separate params, separate grads)
            # so concurrent backward passes are safe.
            pending: dict[int, torch.Tensor] = {}
            for p, (batch, targets) in ready.items():
                with torch.cuda.stream(self.streams[p]):
                    self.optims[p].zero_grad(set_to_none=True)
                    if self.use_bf16:
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            q_pred = self.q_nets[p](batch)
                            loss = F.mse_loss(q_pred, targets)
                    else:
                        q_pred = self.q_nets[p](batch)
                        loss = F.mse_loss(q_pred, targets)
                    loss.backward()
                    pending[p] = loss

            # All fwd+bwd done; sync before optimizer dispatches.
            torch.cuda.synchronize()
            for p, loss in pending.items():
                self.optims[p].step()
                losses[p] = float(loss.item())

        else:
            # ── Sequential fallback (CPU / MPS) ─────────────────────────────
            for p, (batch, targets) in ready.items():
                if self.use_bf16:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        q_pred = self.q_nets[p](batch)
                        loss = F.mse_loss(q_pred, targets)
                else:
                    q_pred = self.q_nets[p](batch)
                    loss = F.mse_loss(q_pred, targets)
                self.optims[p].zero_grad(set_to_none=True)
                loss.backward()
                self.optims[p].step()
                losses[p] = float(loss.item())

        return losses


# ─── Atomic weight publishing ─────────────────────────────


def publish_weights(q_nets: dict, weight_dir: Path, version: int) -> None:
    """Atomically write global Q-net weights to disk.

    Actors poll ``weight_dir/latest.txt`` for the current version number,
    then load ``weight_dir/weights_{version}.pt``. Both writes are made
    atomic via ``os.replace`` (POSIX rename — no partial-read window).
    """
    weight_dir.mkdir(parents=True, exist_ok=True)
    tmp   = weight_dir / f"weights_{version}.tmp"
    final = weight_dir / f"weights_{version}.pt"

    def _unwrap(net):
        # torch.compile wraps in _orig_mod; actors load into plain nets
        return getattr(net, "_orig_mod", net)

    torch.save(
        {
            "version": version,
            "state_dicts": {
                p: {k: v.detach().cpu() for k, v in _unwrap(q_nets[p]).state_dict().items()}
                for p in range(4)
            },
        },
        tmp,
    )
    os.replace(tmp, final)                     # atomic on POSIX + macOS

    ver_tmp = weight_dir / "latest.tmp"
    ver_tmp.write_text(str(version))
    os.replace(ver_tmp, weight_dir / "latest.txt")  # atomic

    # Clean up all prior weight files — only the current version is needed.
    for stale in weight_dir.glob("weights_*.pt"):
        if stale != final:
            stale.unlink(missing_ok=True)
    for stale in weight_dir.glob("weights_*.tmp"):
        stale.unlink(missing_ok=True)


def load_latest_weights(weight_dir: Path) -> tuple[int, dict] | tuple[None, None]:
    """Read the latest published version + state_dicts.

    Returns ``(None, None)`` if weights have not been published yet.
    """
    ver_path = weight_dir / "latest.txt"
    if not ver_path.exists():
        return None, None
    try:
        version = int(ver_path.read_text().strip())
        payload = torch.load(
            weight_dir / f"weights_{version}.pt",
            map_location="cpu",
            weights_only=False,
        )
        return payload["version"], payload["state_dicts"]
    except Exception:
        return None, None


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
    from .train import TrainConfig, _save_checkpoint

    cfg = TrainConfig(**{k: v for k, v in cfg_dict.items()
                         if k in {f.name for f in dataclasses.fields(TrainConfig)}})

    run_dir    = Path(run_dir)
    weight_dir = Path(weight_dir)

    # Dedicated log file for learner process
    log_path = run_dir / "learner.log"
    logger = logging.getLogger("guanzero.learner")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)-5s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, mode="w")
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

    q_nets  = init_position_nets(
        hidden_lstm=cfg.hidden_lstm,
        hidden_mlp=cfg.hidden_mlp,
        n_mlp_layers=cfg.n_mlp_layers,
        dropout=cfg.dropout,
        use_oracle_others_hand=cfg.use_oracle_others_hand,
    )
    compile_mode = getattr(cfg, "compile_mode", "default") or "default"
    for p, net in q_nets.items():
        q_nets[p] = torch.compile(net, mode=compile_mode)
    learner = Learner(q_nets=q_nets, lr=cfg.lr, device=cfg.device,
                      use_bf16=getattr(cfg, "use_bf16_learner", False))
    buffer  = ReplayBuffer(capacity_per_player=cfg.buffer_capacity_per_player)

    version       = 0
    total_updates = 0
    last_losses: dict[int, float] = {}
    metrics_path  = run_dir / "metrics_learner.jsonl"
    last_ticked   = -1   # tracks the last total_updates value that triggered periodic actions

    # Optionally resume from a prior checkpoint
    if resume_checkpoint is not None:
        ckpt = torch.load(Path(resume_checkpoint), map_location="cpu")
        for p in range(4):
            q_nets[p].load_state_dict(ckpt["q_nets"][p])
        total_updates = int(ckpt.get("episode", 0))
        version       = total_updates // cfg.publish_interval_updates
        logger.info("resumed from %s  (total_updates=%d)", resume_checkpoint, total_updates)

    # Publish initial weights so actors can start immediately
    publish_weights(q_nets, weight_dir, version)
    logger.info("initial weights published (version %d)", version)

    t0 = time.time()
    session_start_updates = total_updates  # for accurate upd/s on resumed runs
    drained_since_log = 0
    while not stop_event.is_set():
        # 1. Drain sample queue into replay buffer (pre-stacked actor messages)
        drained = 0
        while drained < cfg.max_drain_batches_per_loop:
            try:
                msg = sample_queue.get_nowait()
                buffer.push_stacked(msg["stacked"], msg["players"], msg["returns"])
                drained += 1
            except Exception:
                break
        drained_since_log += drained

        # 2. Gradient update when every position's buffer is warm.
        #    cfg.updates_per_learner_step controls how many gradient steps fire
        #    per Python loop iteration — amortizes drain/loop overhead across
        #    multiple GPU-side updates.
        if all(buffer.size(p) >= cfg.buffer_min_size for p in range(4)):
            for net in q_nets.values():
                net.train()
            n_steps = max(1, getattr(cfg, "updates_per_learner_step", 1))
            for _ in range(n_steps):
                new_losses = learner.update(
                    buffer=buffer,
                    batch_size=cfg.batch_size,
                    min_buffer_size=cfg.buffer_min_size,
                )
                if new_losses:
                    last_losses = new_losses
                    total_updates += 1
                else:
                    break

        # 3–5 only fire when total_updates actually advanced to a new tick
        if total_updates == last_ticked or total_updates == 0:
            continue
        last_ticked = total_updates

        # 3. Publish updated weights periodically
        if total_updates % cfg.publish_interval_updates == 0:
            version += 1
            publish_weights(q_nets, weight_dir, version)
            logger.debug("weights published version=%d", version)

        # 4. Checkpoint
        if total_updates % cfg.checkpoint_every_updates == 0:
            ckpt = run_dir / "checkpoints" / f"update_{total_updates:08d}.pt"
            _save_checkpoint(ckpt, q_nets, cfg, total_updates)
            logger.info("checkpoint → %s", ckpt)

        # 5. Metrics log
        if total_updates % cfg.log_every_updates == 0:
            elapsed = time.time() - t0
            session_updates = total_updates - session_start_updates
            upd_per_sec = session_updates / elapsed if elapsed > 0 else 0.0
            target = cfg.total_updates_target or cfg.checkpoint_every_updates
            remaining = max(0, target - total_updates)
            eta_s = remaining / upd_per_sec if upd_per_sec > 0 else 0.0
            eta_h, eta_m = divmod(int(eta_s), 3600)[0], divmod(int(eta_s), 60)[0] % 60
            try:
                queue_depth = sample_queue.qsize()
            except (NotImplementedError, OSError):
                queue_depth = -1
            gpu_mem_gb = (
                round(torch.cuda.memory_allocated() / 1e9, 3)
                if cfg.device == "cuda" and torch.cuda.is_available() else None
            )
            row = {
                "updates": total_updates,
                "version": version,
                "buffer_total": buffer.total_size(),
                "buffer_per_player": {p: buffer.size(p) for p in range(4)},
                "loss": {str(p): round(v, 6) for p, v in last_losses.items()},
                "elapsed_s": round(elapsed, 1),
                "upd_per_sec": round(upd_per_sec, 3),
                "queue_depth": queue_depth,
                "gpu_mem_gb": gpu_mem_gb,
                "drained_since_last_log": drained_since_log,
            }
            with metrics_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            logger.info(
                "updates=%d ver=%d buf=%d loss=%s  %.2f upd/s  q=%d gpu=%sGB drained=%d  ETA %dh%02dm",
                total_updates, version, buffer.total_size(),
                " ".join(f"p{p}={v:.4f}" for p, v in sorted(last_losses.items())),
                upd_per_sec, queue_depth,
                f"{gpu_mem_gb:.2f}" if gpu_mem_gb is not None else "n/a",
                drained_since_log,
                eta_h, eta_m,
            )
            drained_since_log = 0

    # Final checkpoint on clean shutdown
    if total_updates > 0:
        _save_checkpoint(
            run_dir / "checkpoints" / "final.pt",
            q_nets, cfg, total_updates,
        )
    logger.info("learner stopped after %d updates", total_updates)


__all__ = [
    "Learner",
    "publish_weights",
    "load_latest_weights",
    "learner_loop",
]
