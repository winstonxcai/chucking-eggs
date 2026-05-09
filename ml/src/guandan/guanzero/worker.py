"""Persistent actor subprocess for distributed actor-learner DMC.

``actor_loop`` is the target function for spawned actor processes.
``maybe_sync_weights_base`` is the non-blocking weight-sync helper.
"""

from __future__ import annotations

import dataclasses
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from .actor import play_episode
from .encoder import ENCODE_CHANNEL_KEYS, StateActionEncoder
from .encoding.role_encoder import ROLE_ENCODE_CHANNEL_KEYS, RoleAwareStateActionEncoder
from .profiler import PhaseProfiler


def maybe_sync_weights_base(
    q_nets: dict,
    weight_dir: Path,
    local_version: int,
    local_updates: int = 0,
    max_lag_updates: int = 0,
) -> tuple[int, int, int]:
    """Conditionally load newer weights; return (version, local_updates, global_updates).

    Cheap path: reads only ``latest.txt`` to learn the latest published metadata.
    Expensive ``torch.load`` is skipped when:
      - the published version is not newer than ``local_version``, or
      - ``max_lag_updates > 0`` and we are fewer than that many updates behind.

    ``local_updates`` is the update count at which the *currently loaded* policy
    was published; ``global_updates`` is the latest published count (used by the
    actor for ε-schedule). They diverge when the lag gate defers a sync.
    """
    from .learner import read_latest_metadata, load_latest_weights

    meta = read_latest_metadata(weight_dir)
    if meta is None:
        return local_version, local_updates, 0
    latest_version, latest_updates = meta

    # Already at this version → no work, but report fresh global_updates.
    if latest_version <= local_version:
        return local_version, local_updates, latest_updates

    # Lag gate: defer load if not stale enough (skip on first sync where local_version=-1).
    if max_lag_updates > 0 and local_version >= 0 \
            and (latest_updates - local_updates) < max_lag_updates:
        return local_version, local_updates, latest_updates

    snapshot = load_latest_weights(weight_dir)
    if snapshot is None or snapshot.version <= local_version:
        return local_version, local_updates, latest_updates
    for p in range(4):
        q_nets[p].load_state_dict(snapshot.state_dicts[p])
        q_nets[p].eval()
    return snapshot.version, snapshot.updates, snapshot.updates


maybe_sync_weights = maybe_sync_weights_base


def maybe_sync_weights_shared(
    q_net,
    weight_dir: Path,
    local_version: int,
    local_updates: int = 0,
    max_lag_updates: int = 0,
) -> tuple[int, int, int]:
    """Conditionally load newer shared-head weights; return (version, local_updates, global_updates)."""
    from .learner import read_latest_metadata, load_latest_weights

    meta = read_latest_metadata(weight_dir)
    if meta is None:
        return local_version, local_updates, 0
    latest_version, latest_updates = meta

    if latest_version <= local_version:
        return local_version, local_updates, latest_updates

    if max_lag_updates > 0 and local_version >= 0 \
            and (latest_updates - local_updates) < max_lag_updates:
        return local_version, local_updates, latest_updates

    snapshot = load_latest_weights(weight_dir)
    if snapshot is None or snapshot.version <= local_version:
        return local_version, local_updates, latest_updates
    q_net.load_state_dict(snapshot.state_dicts["shared"])
    q_net.eval()
    return snapshot.version, snapshot.updates, snapshot.updates


def actor_loop(
    actor_id:     int,
    cfg_dict:     dict,
    sample_queue,           # multiprocessing.Queue
    stop_event,             # multiprocessing.Event
    weight_dir:   Path,
    inference_args=None,    # Optional[dict] — shared-memory inference handles
    run_dir:      Path | None = None,
) -> None:
    """Persistent actor process for distributed actor-learner DMC.

    Runs self-play continuously, serializes samples to the shared queue, and
    periodically syncs local Q-net copies from the learner's published weights.
    Top-level module function — must be picklable for the 'spawn' start method.

    If ``inference_args`` is provided, action selection routes through a shared
    GPU inference server; the actor skips local q-net initialization entirely.
    """
    # Lazy imports — spawned child re-imports the full package from scratch.
    from .config import TrainConfig
    from .schedules import epsilon_linear
    from .q_network import SharedHeadQNet
    from .q_network import init_seat_nets
    from .config import shared_head_qnet_config

    torch.set_num_threads(1)

    cfg = TrainConfig.from_flat_dict(cfg_dict)
    if cfg.model_type == "shared_heads" and cfg.inference.enabled:
        raise ValueError("Inference server not supported for model_type='shared_heads'.")

    shared_path = cfg.model_type == "shared_heads"
    if shared_path:
        encoder = RoleAwareStateActionEncoder(use_oracle_others_hand=cfg.qnet.use_oracle_others_hand)
    else:
        encoder = StateActionEncoder(use_oracle_others_hand=cfg.qnet.use_oracle_others_hand)

    inference_client = _build_inference_client(actor_id, inference_args, cfg) if inference_args else None
    if shared_path:
        q_nets = SharedHeadQNet(shared_head_qnet_config(cfg))
        q_nets.eval()
    elif inference_client is None:
        q_nets = init_seat_nets(cfg.qnet)
        for net in q_nets.values():
            net.eval()
    else:
        q_nets = None

    weight_dir    = Path(weight_dir)
    run_dir       = Path(run_dir) if run_dir is not None else None
    local_version  = -1
    local_updates  = 0   # update count of the policy actor currently holds
    global_updates = 0   # latest published update count (for ε-schedule)
    episode_count  = 0
    # Pre-stacked accumulator: keep raw encoded dicts and stack at push-time.
    # One pickle of 9 contiguous arrays is ~6× faster to unpickle than 512 dicts
    # of 9 small arrays each (measured: 3.81 ms → 0.63 ms per push).
    buf_dicts:   list[dict]  = []
    buf_players: list[int]   = []
    buf_returns: list[float] = []
    rng = random.Random(cfg.seed + actor_id * 10_000)

    profile_enabled = os.environ.get("GUANZERO_ACTOR_PROFILE") == "1"
    prof = PhaseProfiler(enabled=profile_enabled)
    prof_t_start = time.perf_counter()
    snapshot_every_episodes = max(1, cfg.log_every_updates * 10)

    def _push_buffered() -> None:
        nonlocal buf_dicts, buf_players, buf_returns
        if len(buf_dicts) < cfg.actor_push_batch_size:
            return
        with prof.time("buffer_stack"):
            keys = ROLE_ENCODE_CHANNEL_KEYS if shared_path else ENCODE_CHANNEL_KEYS
            stacked = {k: np.stack([d[k] for d in buf_dicts], axis=0) for k in keys}
            msg = {
                "actor_id": actor_id,
                "version":  local_version,
                "stacked":  stacked,
                "returns":  np.asarray(buf_returns, dtype=np.float32),
            }
            if not shared_path:
                msg["players"] = np.asarray(buf_players, dtype=np.int8)
        with prof.time("queue_put"):
            try:
                sample_queue.put(msg, timeout=5)
            except Exception:
                pass   # queue full or closed — drop and continue
        buf_dicts.clear()
        buf_players.clear()
        buf_returns.clear()

    n_inference_timeouts = 0
    while not stop_event.is_set():
        # Periodic weight sync — only on the local-CPU path; the inference
        # server handles its own weight refresh.
        if q_nets is not None and episode_count % cfg.sync_interval_episodes == 0:
            with prof.time("weight_sync"):
                if shared_path:
                    local_version, local_updates, global_updates = maybe_sync_weights_shared(
                        q_nets, weight_dir, local_version, local_updates, cfg.max_version_lag_updates,
                    )
                else:
                    local_version, local_updates, global_updates = maybe_sync_weights_base(
                        q_nets, weight_dir, local_version, local_updates, cfg.max_version_lag_updates,
                    )

        eps = epsilon_linear(global_updates, cfg.epsilon)

        seed = rng.randint(0, 10_000_000)
        try:
            samples = play_episode(
                q_nets=q_nets,
                encoder=encoder,
                epsilon=eps,
                seed=seed,
                device="cpu",
                gamma=cfg.gamma,
                inference_client=inference_client,
                profiler=prof,
            )
        except Exception as e:
            from .inference_server import InferenceTimeoutError
            if isinstance(e, InferenceTimeoutError):
                n_inference_timeouts += 1
                if n_inference_timeouts <= 3 or n_inference_timeouts % 10 == 0:
                    print(f"[actor-{actor_id}] inference timeout #{n_inference_timeouts}: {e}",
                          flush=True)
                # Brief sleep before retrying so we don't busy-loop if the server
                # is wedged. stop_event check below caps it.
                if stop_event.wait(timeout=0.5):
                    break
                continue
            raise
        episode_count += 1

        for s in samples:
            buf_dicts.append(s.encoded)
            buf_players.append(s.player)
            buf_returns.append(s.mc_return)

        _push_buffered()

        # Periodic snapshot to stdout — first actor only, to avoid N-actor spam.
        if (profile_enabled and actor_id == 0
                and episode_count > 0 and episode_count % snapshot_every_episodes == 0):
            wall_s = time.perf_counter() - prof_t_start
            n_dec = prof._counts.get("num_decisions", 0)
            snap = prof.report(wall_s=wall_s, n_events=episode_count, event_label="episodes")
            snap += prof.report_k_buckets(n_decisions=n_dec)
            print(f"[actor-{actor_id}] profile snapshot @ episode {episode_count}:{snap}",
                  flush=True)

    if profile_enabled and run_dir is not None:
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
            wall_s = time.perf_counter() - prof_t_start
            n_dec = prof._counts.get("num_decisions", 0)
            report = prof.report(wall_s=wall_s, n_events=episode_count, event_label="episodes")
            report += prof.report_k_buckets(n_decisions=n_dec)
            out = run_dir / f"actor_{actor_id}_profile.txt"
            out.write_text(report + "\n")
        except Exception as e:
            print(f"[actor-{actor_id}] failed to write profile: {e}", flush=True)


def _build_inference_client(actor_id: int, inference_args: dict, cfg) -> "object":
    from .inference_server import build_client
    return build_client(
        actor_id       = actor_id,
        inference_args = inference_args,
        timeout_s      = cfg.inference.timeout_s,
        max_actions    = cfg.inference.max_actions,
    )


__all__ = [
    "actor_loop",
    "maybe_sync_weights_base",
    "maybe_sync_weights",
    "maybe_sync_weights_shared",
]
