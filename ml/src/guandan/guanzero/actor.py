"""Self-play actor: rolls a single episode and returns MC samples.

Single-process for M0 — distributed actors (actor_loop / maybe_sync_weights)
implement the faithful persistent actor-learner DMC paper architecture.
"""

from __future__ import annotations

import dataclasses
import os
import random
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from ..cards import ComboType
from ..combos import Combo
from ..game import GuanDanEnv
from ..pvguan.legal_utils import dedup_strategic
from ..pvguan.rollout import _cap_legal
from .buffer import collate_encoded
from .encoder import StateActionEncoder
from .q_network import GuanZeroQNet
from .returns import TrainSample, compute_mc_returns

# Channel keys in the encoded sample dict — order matches encoder.encode_all().
_KEYS = (
    "own_hand", "others_hand", "recent_action_each_player",
    "played_cards_others", "remaining_counts_others",
    "level", "history", "behavior", "candidate_action",
)


_PASS = Combo(ComboType.PASS, 0, [])


class _ActorProfiler:
    """Per-actor phase timer. Enabled with GUANZERO_ACTOR_PROFILE=1.

    Mirrors the format of ml/scripts/util/profile_guanzero_rollout.py so output
    tables read consistently across the rollout-only and in-loop profilers.
    """

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.times: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)
        self.t_start = time.perf_counter()

    @contextmanager
    def time(self, name: str):
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.times[name] += time.perf_counter() - t0
            self.counts[name] += 1

    def add_count(self, name: str, value: int) -> None:
        if self.enabled:
            self.counts[name] += value

    def report(self, episodes: int) -> str:
        total_wall  = time.perf_counter() - self.t_start
        total_timed = sum(self.times.values())
        n_dec   = self.counts.get("num_decisions", 1) or 1
        n_legal = self.counts.get("num_legal_actions", 0)

        lines: list[str] = []
        lines.append("")
        lines.append(f"Actor profile  |  {episodes} episodes  |  {total_wall:.2f}s wall")
        lines.append(f"{'section':18s} {'sec':>9s} {'%':>7s} {'count':>10s} {'ms/call':>10s}")
        lines.append("-" * 60)
        for k, v in sorted(self.times.items(), key=lambda x: -x[1]):
            c = max(1, self.counts.get(k, 1))
            pct = 100 * v / total_timed if total_timed > 0 else 0
            lines.append(f"{k:18s} {v:9.3f} {pct:7.1f} {c:10d} {1000*v/c:10.3f}")
        lines.append("-" * 60)
        lines.append(f"{'TOTAL TIMED':18s} {total_timed:9.3f}  ({100*total_timed/total_wall:.1f}% of wall)")
        lines.append("")
        lines.append("Derived:")
        if total_wall > 0:
            lines.append(f"  episodes/sec            : {episodes / total_wall:.2f}")
            lines.append(f"  decisions/sec           : {n_dec / total_wall:.1f}")
        if episodes > 0:
            lines.append(f"  decisions/episode (avg) : {n_dec / episodes:.1f}")
        lines.append(f"  legal_actions/decision  : {n_legal / n_dec:.1f}")
        for ph in ("legal_actions", "encode_all", "q_forward", "q_collate",
                   "q_net_forward", "q_argmax_item", "inference_submit",
                   "env_step"):
            if ph in self.times:
                lines.append(f"  {ph:22s}: {1000*self.times[ph]/n_dec:.3f} ms/decision")
        return "\n".join(lines)


def _select_legal(env: GuanDanEnv, player: int, max_legal: int) -> list[Combo]:
    legal = env.legal_moves(player)
    legal = dedup_strategic(legal)
    if max_legal and len(legal) > max_legal:
        keep = _cap_legal(env, player, legal, max_legal)
        legal = [legal[i] for i in keep]
    # _cap_legal preserves PASS, but guarantee it here so callers never see
    # an empty list when responding (leading with empty hand is impossible).
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
    max_legal_actions: int = 128,
    seed: int | None = None,
    device: torch.device | str = "cpu",
    gamma: float = 1.0,
    inference_client=None,
    profiler: _ActorProfiler | None = None,
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
    prof = profiler if profiler is not None else _ActorProfiler(enabled=False)

    while not env.done:
        p = env.current_player
        with prof.time("legal_actions"):
            legal = _select_legal(env, p, max_legal_actions)
        prof.add_count("num_decisions", 1)
        prof.add_count("num_legal_actions", len(legal))

        with prof.time("encode_all"):
            encoded_list = encoder.encode_all(env, p, legal)

        if random.random() < epsilon:
            idx = random.randrange(len(legal))
        elif inference_client is not None:
            with prof.time("inference_submit"):
                idx, _server_version = inference_client.submit(p, encoded_list)
        else:
            net = q_nets[p]
            # Inlined and instrumented version of _argmax_q so we can see
            # where time goes in the local-CPU forward path: collate (np.stack
            # + H2D), the actual q-net forward, then argmax+.item() on CPU.
            with prof.time("q_collate"):
                batch = collate_encoded(encoded_list, device=device)
            with prof.time("q_net_forward"):
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
    from .train import TrainConfig, _epsilon
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
    prof = _ActorProfiler(enabled=profile_enabled)
    snapshot_every_episodes = max(1, cfg.log_every_updates * 10)

    n_inference_timeouts = 0
    while not stop_event.is_set():
        # Periodic weight sync — only on the local-CPU path; the inference
        # server handles its own weight refresh.
        if q_nets is not None and episode_count % cfg.sync_interval_episodes == 0:
            with prof.time("weight_sync"):
                local_version = maybe_sync_weights(q_nets, weight_dir, local_version)

        eps  = _epsilon(episode_count + actor_id, cfg)
        seed = rng.randint(0, 10_000_000)

        try:
            samples = play_episode(
                q_nets=q_nets,
                encoder=encoder,
                epsilon=eps,
                max_legal_actions=cfg.max_legal_actions,
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

        if len(buf_dicts) >= cfg.actor_push_batch_size:
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

        # Periodic snapshot to stdout — first actor only, to avoid 32× spam.
        if (profile_enabled and actor_id == 0
                and episode_count > 0 and episode_count % snapshot_every_episodes == 0):
            print(f"[actor-{actor_id}] profile snapshot @ episode {episode_count}:\n"
                  f"{prof.report(episode_count)}", flush=True)

    # ── Shutdown: write per-actor profile summary ────────────
    if profile_enabled and run_dir is not None:
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
            out = run_dir / f"actor_{actor_id}_profile.txt"
            out.write_text(prof.report(episode_count) + "\n")
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
