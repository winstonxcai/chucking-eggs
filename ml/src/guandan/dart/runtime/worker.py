"""Persistent actor subprocess for distributed actor-learner DMC."""

from __future__ import annotations

import errno
import multiprocessing as mp
import os
import queue
import random
import time
from pathlib import Path

import torch

from ..utils.logging_setup import setup_run_logging
from ..utils.profiler import PhaseProfiler
from ..utils.reproducibility import (
    capture_actor_rng_state,
    restore_actor_rng_state,
    seed_everything,
)
from .actor import LaneConfig, play_episodes_batched
from .actor.opponents import OpponentPools
from .actor.runtime import (
    build_actor_runtime,
    jitter_sync_threshold,
    sync_actor_weights,
)
from .actor.samples import ActorSampleAccumulator, QueueBatchMeta, SampleBucketPolicy


def _build_lanes(
    pools: OpponentPools,
    rng: random.Random,
    *,
    n_lanes: int,
    eps: float,
) -> list[LaneConfig]:
    return [
        LaneConfig(
            seed=rng.randint(0, 10_000_000),
            seats=seats,
            tags=tags,
        )
        for seats, tags in (
            pools.sample_lane(rng, eps)
            for _ in range(n_lanes)
        )
    ]


def _is_closed_queue_error(exc: BaseException) -> bool:
    if isinstance(exc, (BrokenPipeError, EOFError)):
        return True
    if isinstance(exc, ValueError):
        return "closed" in str(exc).lower()
    if isinstance(exc, OSError):
        return exc.errno in {errno.EBADF, errno.EPIPE} or "closed" in str(exc).lower()
    return False


def _put_sample_message(
    *,
    actor_id: int,
    sample_queue,
    msg: dict,
    stop_event,
    logger,
    policy: str,
    timeout_s: float,
    log_every: int,
    queue_full_counter: mp.Value | None = None,
) -> bool:
    """Put one actor batch, optionally treating a full queue as backpressure."""
    attempts = 0
    while not stop_event.is_set():
        try:
            sample_queue.put(msg, timeout=timeout_s)
            return True
        except queue.Full as exc:
            attempts += 1
            if queue_full_counter is not None:
                with queue_full_counter.get_lock():
                    queue_full_counter.value += 1

            if policy == "stop":
                logger.error(
                    "actor-%d: sample queue full after %.1fs; stopping to avoid "
                    "silent sample loss: %s",
                    actor_id, timeout_s, exc,
                )
                stop_event.set()
                raise

            if attempts == 1 or attempts % log_every == 0:
                logger.warning(
                    "actor-%d: sample queue full after %.1fs; waiting for learner "
                    "backpressure (timeouts=%d)",
                    actor_id, timeout_s, attempts,
                )
    return False


def actor_loop(
    actor_id:     int,
    cfg_dict:     dict,
    sample_queue: mp.Queue[bytes],
    stop_event:   mp.Event,
    weight_dir:   Path,
    run_dir:      Path | None = None,
    pause_event:  mp.Event | None = None,
    resume_actor_rng_state: dict | None = None,
    queue_full_counter: mp.Value | None = None,
) -> None:
    """Run self-play continuously and push stacked sample batches to the learner."""
    from ..config import TrainConfig
    from ..utils.schedules import epsilon_linear

    torch.set_num_threads(1)

    cfg = TrainConfig.from_flat_dict(cfg_dict)
    actor_seed = cfg.seed + actor_id * 10_000
    seed_everything(actor_seed, include_cuda=False)
    log_run_dir = Path(run_dir) if run_dir is not None else Path("/tmp/dart_actor_logs")
    logger, _ = setup_run_logging(
        log_run_dir,
        log_filename=f"actor_{actor_id}.log",
        name=f"dart.actor{actor_id}",
        stream_to_stdout=True,
    )

    runtime = build_actor_runtime(cfg)
    pools = OpponentPools.from_config(cfg, logger)
    accumulator = ActorSampleAccumulator(
        channel_keys=runtime.channel_keys,
        include_players=runtime.include_players,
        bucket_policy=SampleBucketPolicy(
            coordination_card_threshold=cfg.coordination_bucket_card_threshold,
            coordination_final_fraction=cfg.coordination_bucket_final_fraction,
        ),
    )

    weight_dir = Path(weight_dir)
    run_dir = Path(run_dir) if run_dir is not None else None
    rng = random.Random(actor_seed)
    if resume_actor_rng_state is not None:
        restore_actor_rng_state(rng, resume_actor_rng_state)
        logger.info("actor-%d restored RNG state from checkpoint", actor_id)
    local_version = -1
    local_updates = 0
    global_updates = 0
    episode_count = 0
    dropped_batches = 0
    sync_threshold = jitter_sync_threshold(cfg, rng)

    profile_enabled = os.environ.get("DART_ACTOR_PROFILE", "").strip().lower() in {
        "1", "true", "yes", "on",
    }
    prof = PhaseProfiler(enabled=profile_enabled)
    prof_t_start = time.perf_counter()
    snapshot_every_episodes = max(1, cfg.log_every_updates * 10)

    def _push_buffered() -> None:
        nonlocal dropped_batches
        batch_size = cfg.actor_push_batch_size
        queue_policy = cfg.actor_queue_full_policy
        queue_timeout_s = float(cfg.actor_queue_put_timeout_s)
        queue_log_every = int(cfg.actor_queue_full_log_every)
        while len(accumulator) >= batch_size and not stop_event.is_set():
            with prof.time("buffer_stack"):
                msg = accumulator.pop_message(
                    batch_size,
                    QueueBatchMeta(
                        actor_id=actor_id,
                        version=local_version,
                        local_updates=local_updates,
                        global_updates=global_updates,
                        actor_rng_state=capture_actor_rng_state(rng),
                    ),
                )
                if msg is None:
                    return
            with prof.time("queue_put"):
                try:
                    _put_sample_message(
                        actor_id=actor_id,
                        sample_queue=sample_queue,
                        msg=msg,
                        stop_event=stop_event,
                        logger=logger,
                        policy=queue_policy,
                        timeout_s=queue_timeout_s,
                        log_every=queue_log_every,
                        queue_full_counter=queue_full_counter,
                    )
                except queue.Full:
                    # _put_sample_message owns the fail-fast logging/counter.
                    raise
                except (BrokenPipeError, EOFError, OSError, ValueError) as exc:
                    if stop_event.is_set():
                        return
                    if not _is_closed_queue_error(exc):
                        raise
                    dropped_batches += 1
                    if dropped_batches <= 3 or dropped_batches % 20 == 0:
                        logger.warning(
                            "actor-%d: sample queue drop #%d: %s",
                            actor_id, dropped_batches, exc,
                        )

    was_paused = False
    force_next_sync = False
    while not stop_event.is_set():
        if pause_event is not None and pause_event.is_set():
            if not was_paused:
                logger.info("actor-%d paused for checkpoint eval", actor_id)
                was_paused = True
            while pause_event.is_set() and not stop_event.is_set():
                time.sleep(0.2)
            if was_paused and not stop_event.is_set():
                logger.info("actor-%d resumed after checkpoint eval", actor_id)
                was_paused = False
                force_next_sync = True
            continue

        prev_version = local_version
        with prof.time("weight_sync"):
            local_version, local_updates, global_updates = sync_actor_weights(
                runtime,
                weight_dir,
                local_version=local_version,
                local_updates=local_updates,
                sync_threshold=0 if force_next_sync else sync_threshold,
            )
            force_next_sync = False
        if local_version > prev_version:
            sync_threshold = jitter_sync_threshold(cfg, rng)
            runtime.refresh_actor_nets()

        eps = epsilon_linear(global_updates, cfg.epsilon)
        n_lanes = (
            1
            if os.environ.get("DART_NO_BATCHED_ACTOR")
            else int(getattr(cfg, "actor_batch_lanes", 1))
        )
        lanes = _build_lanes(pools, rng, n_lanes=n_lanes, eps=eps)
        samples_per_lane = play_episodes_batched(
            q_nets=runtime.q_nets_actor,
            encoder=runtime.encoder,
            lanes=lanes,
            device="cpu",
            gamma=cfg.gamma,
            profiler=prof,
            rng=rng,
            stop_event=stop_event,
            record_forced_k1_samples=cfg.max_forced_pass_replay_frac > 0.0,
        )
        if stop_event.is_set():
            break

        for lane, samples in zip(lanes, samples_per_lane, strict=False):
            accumulator.append_lane(lane, samples)
        episode_count += len(lanes)
        _push_buffered()

        if (profile_enabled and actor_id == 0
                and episode_count > 0 and episode_count % snapshot_every_episodes == 0):
            wall_s = time.perf_counter() - prof_t_start
            n_dec = prof._counts.get("num_decisions", 0)
            snap = prof.report(wall_s=wall_s, n_events=episode_count, event_label="episodes")
            snap += prof.report_k_buckets(n_decisions=n_dec)
            logger.info("profile snapshot @ episode %d:%s", episode_count, snap)

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
            logger.warning("failed to write profile: %s", e)

    if dropped_batches:
        logger.warning("actor-%d: dropped %d sample batches (queue full/closed)", actor_id, dropped_batches)
    pools.log_summary(logger)
    try:
        sample_queue.cancel_join_thread()
        sample_queue.close()
    except Exception:
        pass


__all__ = ["actor_loop"]
