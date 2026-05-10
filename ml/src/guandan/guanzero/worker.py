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
    sync_interval_updates: int = 0,
) -> tuple[int, int, int]:
    """Conditionally load newer weights; return (version, local_updates, global_updates).

    Cheap path: reads only ``latest.txt`` to learn the latest published metadata.
    Expensive ``torch.load`` happens only when the actor is at least
    ``sync_interval_updates`` learner updates behind the latest publish.
    First sync (``local_version=-1``) always loads regardless of the threshold.

    ``local_updates`` is the update count at which the *currently loaded* policy
    was published; ``global_updates`` is the latest published count (used by the
    actor for ε-schedule). They diverge whenever the gate defers a load.
    """
    from .learner import read_latest_metadata, load_latest_weights

    meta = read_latest_metadata(weight_dir)
    if meta is None:
        return local_version, local_updates, 0
    latest_version, latest_updates = meta

    # Already at this version → no work, but report fresh global_updates.
    if latest_version <= local_version:
        return local_version, local_updates, latest_updates

    # Update-lag gate: defer load until far enough behind (always sync on first call).
    if local_version >= 0 and sync_interval_updates > 0 \
            and (latest_updates - local_updates) < sync_interval_updates:
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
    sync_interval_updates: int = 0,
) -> tuple[int, int, int]:
    """Conditionally load newer shared-head weights; return (version, local_updates, global_updates)."""
    from .learner import read_latest_metadata, load_latest_weights

    meta = read_latest_metadata(weight_dir)
    if meta is None:
        return local_version, local_updates, 0
    latest_version, latest_updates = meta

    if latest_version <= local_version:
        return local_version, local_updates, latest_updates

    if local_version >= 0 and sync_interval_updates > 0 \
            and (latest_updates - local_updates) < sync_interval_updates:
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

    # ── Checkpoint-population pool (shared-head + non-empty pool only) ──
    # Each actor preloads every frozen checkpoint once. Per vs-frozen episode
    # we sample one uniformly and use it for the opponent seats.
    frozen_nets: list = []
    population_active = (
        shared_path
        and bool(cfg.population_pool)
        and cfg.latest_vs_latest_frac < 1.0
    )
    if population_active:
        from .checkpoint import load_frozen_shared_qnet
        qnet_cfg_for_pool = shared_head_qnet_config(cfg)
        for ckpt_path in cfg.population_pool:
            frozen_nets.append(load_frozen_shared_qnet(ckpt_path, qnet_cfg_for_pool, device="cpu"))
        print(
            f"[actor-{actor_id}] preloaded {len(frozen_nets)} frozen opponents",
            flush=True,
        )

    # Per-actor counters for end-of-run summary (printed by every actor).
    mode_counts = {"self_play": 0, "vs_frozen": 0}
    frozen_pick_counts = [0] * len(frozen_nets)
    team_counts = {"latest_even": 0, "latest_odd": 0}

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

    def _sample_sync_threshold() -> int:
        """Per-actor sync threshold = sync_interval_updates ± sync_jitter_updates (uniform)."""
        if cfg.sync_jitter_updates <= 0:
            return cfg.sync_interval_updates
        return cfg.sync_interval_updates + rng.randint(
            -cfg.sync_jitter_updates, cfg.sync_jitter_updates,
        )

    sync_threshold = _sample_sync_threshold()

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
        # Weight sync — cheap metadata read every iteration (~10 bytes from latest.txt);
        # torch.load only fires when actor is sync_threshold+ updates behind the learner.
        # Each actor's threshold is jittered ± sync_jitter_updates so the 32 actors
        # don't all reload on the same publish cycle (smoother self-play diversity).
        if q_nets is not None:
            prev_version = local_version
            with prof.time("weight_sync"):
                if shared_path:
                    local_version, local_updates, global_updates = maybe_sync_weights_shared(
                        q_nets, weight_dir, local_version, local_updates, sync_threshold,
                    )
                else:
                    local_version, local_updates, global_updates = maybe_sync_weights_base(
                        q_nets, weight_dir, local_version, local_updates, sync_threshold,
                    )
            if local_version > prev_version:
                # Just synced — draw a fresh threshold so the next reload is independently jittered.
                sync_threshold = _sample_sync_threshold()

        eps = epsilon_linear(global_updates, cfg.epsilon)

        # ── Per-episode dispatch: self-play vs vs-frozen ──
        # frozen_seats are the OPPONENT seats (frozen net + epsilon_frozen);
        # latest controls the complementary pair and is the only contributor
        # of training samples.
        use_frozen = bool(frozen_nets) and rng.random() >= cfg.latest_vs_latest_frac
        frozen_seats: frozenset[int] = frozenset()
        q_net_frozen_this_ep = None
        if use_frozen:
            pick = rng.randrange(len(frozen_nets))
            q_net_frozen_this_ep = frozen_nets[pick]
            frozen_pick_counts[pick] += 1
            if rng.random() < 0.5:
                frozen_seats = frozenset({1, 3})  # latest controls {0, 2}
                team_counts["latest_even"] += 1
            else:
                frozen_seats = frozenset({0, 2})  # latest controls {1, 3}
                team_counts["latest_odd"] += 1
            mode_counts["vs_frozen"] += 1
        else:
            mode_counts["self_play"] += 1

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
                q_nets_frozen=q_net_frozen_this_ep,
                frozen_seats=frozen_seats,
                epsilon_frozen=cfg.epsilon.frozen,
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

        if frozen_seats:
            # Drop frozen-team samples — they're opponents, not teachers.
            samples = [s for s in samples if s.player not in frozen_seats]

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

    # Per-actor population counters — printed by every actor so off-balance
    # frozen-pool sampling or bad team assignment shows up immediately in logs.
    print(f"[actor-{actor_id}] episode modes: {mode_counts}", flush=True)
    if frozen_nets:
        pool_str = ", ".join(
            f"{Path(p).stem}={c}"
            for p, c in zip(cfg.population_pool, frozen_pick_counts)
        )
        print(f"[actor-{actor_id}] frozen picks: {pool_str}", flush=True)
        print(f"[actor-{actor_id}] team assignment: {team_counts}", flush=True)


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
