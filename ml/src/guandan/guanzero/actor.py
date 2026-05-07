"""Self-play actor: rolls a single episode and returns MC samples.

Single-process for M0 — distributed actors (actor_loop / maybe_sync_weights)
implement the faithful persistent actor-learner DMC paper architecture.
"""

from __future__ import annotations

import dataclasses
import os
import random
import time
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from ..cards import ComboType
from ..combos import Combo
from ..game import GuanDanEnv
from .buffer import collate_encoded
from .encoder import StateActionEncoder
from .legal_utils import dedup_strategic
from .profiler import PhaseProfiler, _k_bucket, _K_BUCKETS
from .q_network import GuanZeroQNet
from .returns import TrainSample, compute_mc_returns

# Channel keys in the encoded sample dict — order matches encoder.encode_all().
_KEYS = (
    "own_hand", "others_hand", "recent_action_each_player",
    "played_cards_others", "remaining_counts_others",
    "level", "history", "behavior", "candidate_action",
)


_PASS = Combo(ComboType.PASS, 0, [])


def _select_legal(env: GuanDanEnv, player: int) -> list[Combo]:
    legal = dedup_strategic(env.legal_moves(player))
    if not env.is_leading() and not any(m.type == ComboType.PASS for m in legal):
        legal.append(_PASS)
    return legal


@torch.no_grad()
def _argmax_q(
    net: GuanZeroQNet,
    encoded_list: list[dict],
    device: torch.device,
) -> int:
    batch = collate_encoded(encoded_list, device=device)
    q = net(batch)
    return int(q.argmax().item())


def play_episode(
    q_nets: Mapping[int, GuanZeroQNet] | None,
    encoder: StateActionEncoder,
    epsilon: float,
    seed: int | None = None,
    device: torch.device | str = "cpu",
    gamma: float = 1.0,
    inference_client=None,
    profiler: PhaseProfiler | None = None,
) -> list[TrainSample]:
    """Roll one self-play episode, return per-step MC training samples.

    If ``inference_client`` is provided, action selection routes through the
    shared GPU inference server and ``q_nets`` may be None. Otherwise the
    legacy local-CPU path runs (q_nets must be provided).
    """
    device = torch.device(device)
    env = GuanDanEnv()
    env.reset(seed=seed)

    trajectory: list[dict] = []
    prof = profiler if profiler is not None else PhaseProfiler(enabled=False)

    while not env.done:
        p = env.current_player
        with prof.time("legal_actions"):
            legal = _select_legal(env, p)
        K = len(legal)
        bucket = _k_bucket(K)
        prof.add_count("num_decisions", 1)
        prof.add_count("num_legal_actions", K)
        # Per-K-bucket decision counts (covers shortcut + epsilon + server +
        # forward paths so the bucket totals sum to num_decisions).
        prof.add_count(f"decisions_{bucket}", 1)

        with prof.time("encode_all"):
            encoded_list = encoder.encode_all(env, p, legal)

        if K == 1:
            idx = 0
            prof.add_count("shortcut_K1", 1)
        elif random.random() < epsilon:
            idx = random.randrange(K)
        elif inference_client is not None:
            with prof.time("inference_submit"):
                idx, _server_version = inference_client.submit(p, encoded_list)
        else:
            net = q_nets[p]
            # Inlined and instrumented version of _argmax_q so we can see
            # where time goes in the local-CPU forward path. q_net_forward is
            # bucketed by K so we can attribute time to the K distribution.
            with prof.time("q_collate"):
                batch = collate_encoded(encoded_list, device=device)
            with prof.time(f"q_net_forward_{bucket}"):
                with torch.no_grad():
                    q_vals = net(batch)
            with prof.time("q_argmax_item"):
                idx = int(q_vals.argmax().item())

        trajectory.append({"player": p, "encoded": encoded_list[idx]})
        with prof.time("env_step"):
            env.step(legal[idx])

    rewards = env.get_rewards()
    with prof.time("mc_returns"):
        return compute_mc_returns(trajectory, rewards, gamma=gamma)


# ─── Distributed actor helpers ───────────────────────────


def maybe_sync_weights(
    q_nets: dict,
    weight_dir: Path,
    local_version: int,
) -> int:
    """Non-blocking weight sync. Loads newer weights if available, returns new version."""
    from .learner import load_latest_weights

    new_ver, state_dicts = load_latest_weights(weight_dir)
    if new_ver is None or new_ver <= local_version:
        return local_version
    for p in range(4):
        q_nets[p].load_state_dict(state_dicts[p])
        q_nets[p].eval()
    return new_ver


def actor_loop(
    actor_id:     int,
    cfg_dict:     dict,
    sample_queue,          # multiprocessing.Queue
    stop_event,            # multiprocessing.Event
    weight_dir:   Path,
    inference_args=None,   # Optional[dict] — see _build_inference_client below
    run_dir:      Path | None = None,
) -> None:
    """Persistent actor process for faithful actor-learner DMC.

    Runs self-play continuously, serializes samples to the shared queue, and
    periodically syncs local Q-net copies from the learner's published weights.
    Top-level module function — must be picklable for 'spawn' start method.

    If ``inference_args`` is provided (Phase 4+), action selection routes
    through a shared GPU inference server. The actor skips local q-net
    initialization and weight syncing entirely.
    """
    # Lazy import — runs in a spawned child; full package re-imported from scratch
    from .config import TrainConfig
    from .schedules import epsilon_linear
    from .q_network import init_position_nets

    torch.set_num_threads(1)

    cfg = TrainConfig(**{k: v for k, v in cfg_dict.items()
                         if k in {f.name for f in dataclasses.fields(TrainConfig)}})

    encoder  = StateActionEncoder(use_oracle_others_hand=cfg.use_oracle_others_hand)

    # Inference path: server-backed (Phase 4+) vs local CPU q-nets (legacy)
    inference_client = _build_inference_client(actor_id, inference_args, cfg) if inference_args else None
    if inference_client is None:
        q_nets = init_position_nets(
            hidden_lstm=cfg.hidden_lstm,
            hidden_mlp=cfg.hidden_mlp,
            n_mlp_layers=cfg.n_mlp_layers,
            dropout=cfg.dropout,
            use_oracle_others_hand=cfg.use_oracle_others_hand,
        )
        for net in q_nets.values():
            net.eval()
    else:
        q_nets = None

    weight_dir    = Path(weight_dir)
    run_dir       = Path(run_dir) if run_dir is not None else None
    local_version = -1
    episode_count = 0
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
        """Drain accumulated samples to the learner queue if we have a full batch."""
        nonlocal buf_dicts, buf_players, buf_returns
        if len(buf_dicts) < cfg.actor_push_batch_size:
            return
        with prof.time("buffer_stack"):
            stacked = {k: np.stack([d[k] for d in buf_dicts], axis=0) for k in _KEYS}
            msg = {
                "actor_id": actor_id,
                "version":  local_version,
                "stacked":  stacked,
                "players":  np.asarray(buf_players, dtype=np.int8),
                "returns":  np.asarray(buf_returns, dtype=np.float32),
            }
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
                local_version = maybe_sync_weights(q_nets, weight_dir, local_version)

        eps  = epsilon_linear(episode_count + actor_id, cfg)

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
            # InferenceTimeoutError or any other transient failure: skip this
            # episode and try again. Actor processes that crash here are gone
            # for the rest of the run — surviving the timeout keeps the buffer
            # producer pipeline alive.
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
            # Anything else: re-raise (don't hide real bugs)
            raise
        episode_count += 1

        for s in samples:
            buf_dicts.append(s.encoded)
            buf_players.append(s.player)
            buf_returns.append(s.mc_return)

        _push_buffered()

        # Periodic snapshot to stdout — first actor only, to avoid 32× spam.
        if (profile_enabled and actor_id == 0
                and episode_count > 0 and episode_count % snapshot_every_episodes == 0):
            wall_s = time.perf_counter() - prof_t_start
            n_dec = prof._counts.get("num_decisions", 0)
            snap = prof.report(wall_s=wall_s, n_events=episode_count, event_label="episodes")
            snap += prof.report_k_buckets(n_decisions=n_dec)
            print(f"[actor-{actor_id}] profile snapshot @ episode {episode_count}:{snap}",
                  flush=True)

    # ── Shutdown: write per-actor profile summary ────────────
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
    """Construct a SharedInferenceClient inside the actor subprocess.

    `inference_args` must include the SharedBufferMeta + the queue/event
    handles passed by the parent. The actor reattaches to the shared-memory
    blocks here (since spawn-context children don't inherit mappings).
    """
    from .inference_server import (
        SharedInferenceClient,
        attach_shared_buffers,
    )
    bufs = attach_shared_buffers(
        meta           = inference_args["meta"],
        free_slots     = inference_args["free_slots"],
        request_queue  = inference_args["request_queue"],
        events         = inference_args["events"],
        weights_version= inference_args.get("weights_version"),
    )
    return SharedInferenceClient(
        actor_id    = actor_id,
        bufs        = bufs,
        timeout_s   = cfg.inference_timeout_s,
        max_actions = cfg.inference_max_actions,
    )


__all__ = ["play_episode", "maybe_sync_weights", "actor_loop"]
