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
import os
import time
from pathlib import Path
from typing import Mapping

import torch
import torch.nn.functional as F

from .data.buffer import ReplayBuffer, RoleAwareReplayBuffer
from .checkpoint import (
    WeightSnapshot,
    migrate_state_dict,
    save_checkpoint_base,
    save_checkpoint_shared,
    unwrap_compiled,
)
from .utils.logging_setup import setup_run_logging
from .utils.metrics import jsonl_writer
from .utils.profiler import PhaseProfiler
from .q_network import GuanZeroQNet, SharedHeadQNet, SharedTrickHeadQNet, init_seat_nets
from .utils.run_layout import RunLayout
from .data.sample_tags import ACTION_CLASS_LOOKUP, OPP_GRID_TOP


PHASE_KEY_PREFIXES = (
    "phase_role_", "phase_pair_", "source_phase_", "opp_phase_",
    "action_phase_", "epsilon_", "is_pass_", "is_bomb_",
    "k_bucket_", "q_gap_", "team_", "reward_",
)


def _emit_grid(
    metrics: dict[str, float | None],
    key_prefix: str,
    axis_a: torch.Tensor,
    axis_b: torch.Tensor,
    n_a: int,
    n_b: int,
    preds: torch.Tensor,
    targets: torch.Tensor,
    min_n: int = 32,
) -> None:
    total = max(axis_a.shape[0], 1)
    for a in range(n_a):
        for b in range(n_b):
            mask = (axis_a == a) & (axis_b == b)
            n = int(mask.sum().item())
            cell = a * n_b + b
            metrics[f"{key_prefix}_{cell}_n"] = n
            metrics[f"{key_prefix}_{cell}_frac"] = n / total
            if n < min_n:
                metrics[f"{key_prefix}_{cell}_loss"] = None
            else:
                metrics[f"{key_prefix}_{cell}_loss"] = float(
                    F.mse_loss(preds[mask], targets[mask]).item()
                )


def _emit_marginal(
    metrics: dict[str, float | None],
    key_prefix: str,
    axis: torch.Tensor,
    n_levels: int,
    preds: torch.Tensor,
    targets: torch.Tensor,
    min_n: int = 32,
) -> None:
    total = max(axis.shape[0], 1)
    for a in range(n_levels):
        mask = axis == a
        n = int(mask.sum().item())
        metrics[f"{key_prefix}_{a}_n"] = n
        metrics[f"{key_prefix}_{a}_frac"] = n / total
        if n < min_n:
            metrics[f"{key_prefix}_{a}_loss"] = None
        else:
            metrics[f"{key_prefix}_{a}_loss"] = float(
                F.mse_loss(preds[mask], targets[mask]).item()
            )


def _k_bucket_tensor(k: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(k, dtype=torch.long)
    out[(k >= 2) & (k <= 5)] = 1
    out[(k >= 6) & (k <= 20)] = 2
    out[k > 20] = 3
    return out


def _q_gap_bucket_tensor(q: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(q, dtype=torch.long)
    nan_mask = torch.isnan(q)
    out[nan_mask] = 0
    out[~nan_mask & (q < 0.05)] = 1
    out[~nan_mask & (q >= 0.05) & (q < 0.20)] = 2
    out[~nan_mask & (q >= 0.20)] = 3
    return out


def _reward_bucket_tensor(r: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(r, dtype=torch.long)
    out[r <= -2] = 0
    out[(r > -2) & (r < 0)] = 1
    out[(r >= 0) & (r <= 1)] = 2
    out[r > 1] = 3
    return out


def _opp_grid_index_tensor(opp_id: torch.Tensor) -> torch.Tensor:
    """Map raw opponent_id to index-in-OPP_GRID_TOP; non-listed → -1 (excluded)."""
    out = torch.full_like(opp_id, -1, dtype=torch.long)
    for i, v in enumerate(OPP_GRID_TOP):
        out[opp_id == v] = i
    return out


def _emit_phase_aggregations(
    metrics: dict,
    tags: dict[str, torch.Tensor],
    preds: torch.Tensor,
    targets: torch.Tensor,
) -> None:
    phase_self = tags["phase_self"].long()
    trick_role = tags["trick_role"].long()
    phase_partner = tags["phase_partner"].long()
    episode_mode = tags["episode_mode"].long()
    opponent_id = tags["opponent_id"].long()
    action_type = tags["action_type"].long()
    is_pass = tags["is_pass"].long()
    is_bomb = tags["is_bomb"].long()
    chosen_by_epsilon = tags["chosen_by_epsilon"].long()
    latest_team = tags["latest_team"].long()
    num_legal_actions = tags["num_legal_actions"].long()
    q_gap = tags["q_gap"].float()
    terminal_reward = tags["terminal_reward"].float()

    _emit_grid(metrics, "phase_role", phase_self, trick_role, 3, 3, preds, targets)
    _emit_grid(metrics, "phase_pair", phase_self, phase_partner, 3, 4, preds, targets)
    _emit_grid(metrics, "source_phase", episode_mode, phase_self, 3, 3, preds, targets)

    opp_idx = _opp_grid_index_tensor(opponent_id)
    keep = opp_idx >= 0
    if keep.any():
        _emit_grid(
            metrics, "opp_phase",
            opp_idx[keep], phase_self[keep],
            len(OPP_GRID_TOP), 3,
            preds[keep], targets[keep],
        )
    else:
        # No samples from listed opponents — emit empty cells so schema stays
        # stable across runs.
        for a in range(len(OPP_GRID_TOP)):
            for b in range(3):
                cell = a * 3 + b
                metrics[f"opp_phase_{cell}_n"] = 0
                metrics[f"opp_phase_{cell}_frac"] = 0.0
                metrics[f"opp_phase_{cell}_loss"] = None

    lookup = torch.as_tensor(ACTION_CLASS_LOOKUP, dtype=torch.long, device=action_type.device)
    action_class = lookup[action_type.clamp(min=0, max=lookup.numel() - 1)]
    _emit_grid(metrics, "action_phase", action_class, phase_self, 6, 3, preds, targets)

    _emit_marginal(metrics, "epsilon", chosen_by_epsilon, 2, preds, targets)
    _emit_marginal(metrics, "is_pass", is_pass, 2, preds, targets)
    _emit_marginal(metrics, "is_bomb", is_bomb, 2, preds, targets)
    _emit_marginal(metrics, "k_bucket", _k_bucket_tensor(num_legal_actions), 4, preds, targets)
    _emit_marginal(metrics, "q_gap", _q_gap_bucket_tensor(q_gap), 4, preds, targets)
    _emit_marginal(metrics, "team", latest_team, 2, preds, targets)
    _emit_marginal(metrics, "reward", _reward_bucket_tensor(terminal_reward), 4, preds, targets)


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
    """Learner for a single shared-head Q-net with one shared optimizer.

    Supports both ``SharedHeadQNet`` (absolute-seat heads, ``seat_id`` field)
    and ``SharedTrickHeadQNet`` (trick-relative heads, ``trick_head_id``
    field). The head field is auto-detected from the buffer at update time
    and drives both stratification and per-head metric naming.
    """

    def __init__(
        self,
        q_net: SharedHeadQNet | SharedTrickHeadQNet,
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
        replay_mix: dict | None = None,
        max_forced_k1_replay_frac: float = 1.0,
    ) -> dict | None:
        """One balanced gradient step, or None if any head bucket is cold."""
        sizes = buffer.size_by_seat()
        min_per_seat = batch_size // 4
        if min(sizes.values()) < min_per_seat:
            return None

        if max_forced_k1_replay_frac < 1.0:
            batch, targets, tags = buffer.sample_batch_balanced_k1_capped(
                batch_size, max_forced_k1_replay_frac, self.device, return_tags=True,
            )
        elif replay_mix:
            batch, targets, tags = buffer.sample_batch_stratified(
                batch_size, replay_mix, self.device, return_tags=True,
            )
        else:
            batch, targets, tags = buffer.sample_batch_balanced(
                batch_size, self.device, return_tags=True,
            )
        head_field = buffer.head_field
        head_ids = batch[head_field].long()
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

        suffix = "seat" if head_field == "seat_id" else "trick_head"
        with torch.no_grad():
            metrics: dict = {
                "loss": float(loss.item()),
                "grad_norm": float(grad_norm_total.item()),
                "grad_norm_trunk": float(_grad_norm_of(self.q_net.trunk).item()),
            }
            for k in range(4):
                mask = head_ids == k
                metrics[f"sample_count_{suffix}_{k}"] = float(mask.sum().item())
                if mask.any():
                    seat_preds = preds[mask]
                    seat_targets = targets[mask]
                    metrics[f"loss_{suffix}_{k}"] = float(F.mse_loss(seat_preds, seat_targets).item())
                    metrics[f"q_mean_{suffix}_{k}"] = float(seat_preds.mean().item())
                    metrics[f"q_std_{suffix}_{k}"] = float(seat_preds.std(unbiased=False).item())
                metrics[f"grad_norm_head_{k}"] = float(_grad_norm_of(self.q_net.heads[k]).item())
            _emit_phase_aggregations(metrics, tags, preds.detach(), targets.detach())
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
    update_counter=None,        # multiprocessing.Value('i'), advanced per gradient step
    weights_ready=None,         # multiprocessing.Event, set after first publish
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
    if cfg.model_type in ("shared_heads", "shared_trick_heads"):
        _learner_loop_shared(
            cfg=cfg,
            sample_queue=sample_queue,
            stop_event=stop_event,
            weight_dir=weight_dir,
            run_dir=run_dir,
            update_counter=update_counter,
            weights_ready=weights_ready,
            resume_checkpoint=resume_checkpoint,
        )
        return

    layout     = RunLayout(Path(run_dir))
    weight_dir = Path(weight_dir)

    logger, _      = setup_run_logging(layout.run_dir, layout.learner_log.name, name="guanzero.learner")
    metrics_writer = jsonl_writer(layout.metrics_jsonl)

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
    last_ticked   = -1   # tracks the last total_updates value that triggered periodic actions

    # Optionally resume from a prior checkpoint
    if resume_checkpoint is not None:
        ckpt = torch.load(Path(resume_checkpoint), map_location="cpu", weights_only=True)
        for p in range(4):
            unwrap_compiled(q_nets[p]).load_state_dict(migrate_state_dict(ckpt["q_nets"][p]))
        total_updates = int(ckpt.get("episode", 0))
        version       = total_updates // cfg.publish_interval_updates
        logger.info("resumed from %s  (total_updates=%d)", resume_checkpoint, total_updates)

    if update_counter is not None:
        update_counter.value = total_updates

    # Publish initial weights so actors can start immediately
    publish_weights(q_nets, weight_dir, version, updates=total_updates)
    logger.info("initial weights published (version %d)", version)
    if weights_ready is not None:
        weights_ready.set()

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
                if update_counter is not None:
                    update_counter.value = total_updates

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
            ckpt = layout.update_checkpoint(total_updates)
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
            metrics_writer.write(row)
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
        save_checkpoint_base(layout.final_checkpoint, q_nets, cfg, total_updates)
    logger.info("learner stopped after %d updates", total_updates)
    metrics_writer.close()


def _learner_loop_shared(
    cfg,
    sample_queue,
    stop_event,
    weight_dir: Path,
    run_dir: Path,
    update_counter=None,
    weights_ready=None,
    resume_checkpoint: Path | None = None,
) -> None:
    """Central learner process for role-aware shared-head training.

    Handles both ``shared_heads`` (absolute-seat heads, legacy) and
    ``shared_trick_heads`` (trick-relative heads, new) variants. The Q-net
    class, encoder schema, and replay buffer mode are dispatched on
    ``cfg.model_type``.
    """
    from .config import shared_head_qnet_config, shared_trick_head_qnet_config

    trick_path = cfg.model_type == "shared_trick_heads"

    if cfg.inference.enabled:
        raise ValueError(
            f"Inference server not supported for model_type={cfg.model_type!r}. "
            "Set use_inference_server: false in config."
        )

    layout     = RunLayout(Path(run_dir))
    weight_dir = Path(weight_dir)
    logger, _      = setup_run_logging(layout.run_dir, layout.learner_log.name, name="guanzero.learner")
    metrics_writer = jsonl_writer(layout.metrics_jsonl)

    if trick_path:
        q_net = SharedTrickHeadQNet(shared_trick_head_qnet_config(cfg))
    else:
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
    buffer = RoleAwareReplayBuffer(
        capacity=capacity,
        head_scheme="trick_relative" if trick_path else "absolute_seat",
    )

    version = 0
    total_updates = 0
    last_metrics: dict[str, float] = {}
    last_ticked = -1

    if resume_checkpoint is not None:
        ckpt = torch.load(Path(resume_checkpoint), map_location="cpu", weights_only=True)
        unwrap_compiled(q_net).load_state_dict(ckpt["q_net"])
        total_updates = int(ckpt.get("episode", 0))
        version = total_updates // cfg.publish_interval_updates
        logger.info("resumed from %s  (total_updates=%d)", resume_checkpoint, total_updates)

    if update_counter is not None:
        update_counter.value = total_updates

    publish_weights_shared(q_net, weight_dir, version, updates=total_updates)
    logger.info("initial shared weights published (version %d)", version)
    if weights_ready is not None:
        weights_ready.set()

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
                tag_keys = [name for name, _ in RoleAwareReplayBuffer._TAG_FIELDS]
                tags_in_msg = {k: msg[k] for k in tag_keys if k in msg}
                buffer.push_stacked(
                    msg["stacked"],
                    msg["returns"],
                    msg.get("buckets"),
                    tags=tags_in_msg or None,
                )
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
            metrics = learner.update(
                buffer=buffer,
                batch_size=cfg.batch_size,
                replay_mix=cfg.replay_mix or None,
                max_forced_k1_replay_frac=cfg.max_forced_k1_replay_frac,
            )
            if metrics is not None:
                last_metrics = metrics
                total_updates += 1
                if update_counter is not None:
                    update_counter.value = total_updates

        if total_updates == last_ticked or total_updates == 0:
            if drained == 0:
                time.sleep(0.001)
            continue
        last_ticked = total_updates

        if total_updates % cfg.publish_interval_updates == 0:
            version += 1
            publish_weights_shared(q_net, weight_dir, version, updates=total_updates)

        if total_updates % cfg.checkpoint_every_updates == 0:
            ckpt = layout.update_checkpoint(total_updates)
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
            phase_payload = {
                k: v for k, v in last_metrics.items()
                if any(k.startswith(p) for p in PHASE_KEY_PREFIXES)
            }
            scalar_loss = {
                k: (round(v, 6) if isinstance(v, (int, float)) and v is not None else v)
                for k, v in last_metrics.items()
                if k not in phase_payload
            }
            row = {
                "updates": total_updates,
                "version": version,
                "buffer_total": buffer.size(),
                "buffer_per_player": buffer.size_by_seat(),
                "loss": scalar_loss,
                "phase": phase_payload,
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
            metrics_writer.write(row)
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
        save_checkpoint_shared(layout.final_checkpoint, q_net, cfg, total_updates)
    logger.info("shared learner stopped after %d updates", total_updates)
    metrics_writer.close()


__all__ = [
    "Learner",
    "SharedHeadLearner",
    "LearnerMetricsRow",
    "publish_weights",
    "publish_weights_shared",
    "load_latest_weights",
    "learner_loop",
]
