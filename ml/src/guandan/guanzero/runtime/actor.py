"""Single-episode self-play rollout.

Public API: play_episode, play_episode_rust, make_scorer, select_legal, argmax_q.
"""

from __future__ import annotations

import random
from typing import Callable, Mapping

import numpy as np
import torch

import math

from ...cards import ComboType
from ...game import GuanDanEnv
from ..data.buffer import collate_base_encoded, collate_role_encoded
from ..model.encoding.base_encoder import StateActionEncoder
from ..model.encoding.role_encoder import (
    ROLE_ENCODE_CHANNEL_SHAPES,
    ROLE_ENCODE_STATE_KEYS,
    RoleAwareStateActionEncoder,
)
from ..utils.legal_utils import select_legal
from ..utils.profiler import PhaseProfiler, k_bucket_label
from ..model.q_network import GuanZeroQNet, SharedHeadQNet, SharedTrickHeadQNet
from ..data.returns import EpisodeTags, TrainSample, compute_mc_returns
from ..data.sample_tags import phase_bucket, phase_with_out, trick_role


def argmax_q(
    net: GuanZeroQNet | SharedHeadQNet | SharedTrickHeadQNet,
    encoded_list: list[dict],
    device: torch.device,
    *,
    role_encoded: bool = False,
    profiler: PhaseProfiler | None = None,
    bucket: str = "",
) -> tuple[int, float]:
    """Greedy argmax over candidate actions; returns ``(idx, q_gap)``.

    ``q_gap = q_max - q_second`` (NaN if only one action). Pass
    ``role_encoded=True`` for shared-head networks (uses ``collate_role_encoded``).
    Optional ``profiler``/``bucket`` instrument the collate/forward/argmax phases.
    """
    collate = collate_role_encoded if role_encoded else collate_base_encoded
    prof = profiler if profiler is not None else PhaseProfiler(enabled=False)
    bucket_suffix = f"_{bucket}" if bucket else ""

    with prof.time("q_collate"):
        state_batch, action_batch, repeats = collate([encoded_list], device=device)
    with prof.time(f"q_net_forward{bucket_suffix}"):
        with torch.inference_mode():
            q_vals = net.forward_grouped(state_batch, action_batch, repeats)
    with prof.time("q_argmax_item"):
        q_flat = q_vals.view(-1)
        if q_flat.numel() >= 2:
            top2 = torch.topk(q_flat, 2, largest=True).values
            q_gap = float(top2[0].item() - top2[1].item())
        else:
            q_gap = float("nan")
        return int(q_flat.argmax().item()), q_gap


def play_episode(
    q_nets: Mapping[int, GuanZeroQNet] | SharedHeadQNet | SharedTrickHeadQNet | None,
    encoder: StateActionEncoder | RoleAwareStateActionEncoder,
    epsilon: float,
    *,
    seed: int | None = None,
    device: torch.device | str = "cpu",
    gamma: float = 1.0,
    inference_client=None,
    profiler: PhaseProfiler | None = None,
    q_nets_frozen: SharedHeadQNet | SharedTrickHeadQNet | None = None,
    frozen_seats: frozenset[int] = frozenset(),
    epsilon_frozen: float = 0.0,
    hard_bots=None,
    hard_bot_seats: frozenset[int] = frozenset(),
    tags: EpisodeTags = EpisodeTags(),
) -> list[TrainSample]:
    """Roll one self-play episode, return per-step MC training samples.

    If ``inference_client`` is provided, action selection routes through the
    shared GPU inference server and ``q_nets`` may be None. Otherwise the
    local-CPU path runs (q_nets must be provided).

    Optional checkpoint-population kwargs (shared-head path only):
    - ``q_nets_frozen``: a frozen ``SharedHeadQNet`` used for ``frozen_seats``
      decisions. Defaults to None (pure self-play).
    - ``frozen_seats``: which absolute seats use the frozen net. Empty = none.
    - ``epsilon_frozen``: ε for frozen-seat decisions (typically 0.0 so the
      frozen policy plays deterministically, not a noisy version of itself).

    Optional hard-bot kwargs:
    - ``hard_bots``: either a single ``Agent`` instance (same bot on every
      hard-bot seat) or a ``dict[int, Agent]`` mapping seat → bot (mixed-pair
      mode, different bots per opponent seat). Hard-bot seats call
      ``bot.act(env, p)`` and the step is NOT added to the training trajectory
      (bots are opponents, not teachers — only latest-team seats appear in the
      returned samples).

    Sample emission is unchanged for frozen-checkpoint seats — callers filter
    by ``s.player`` to exclude frozen-team rows. Hard-bot seats are filtered
    here (skip trajectory append) so the worker filter is a no-op for them.
    """
    device = torch.device(device)
    shared_path = isinstance(q_nets, (SharedHeadQNet, SharedTrickHeadQNet))
    if shared_path and inference_client is not None:
        raise ValueError(
            "Inference server is not supported for shared-head Q-networks."
        )
    if q_nets_frozen is not None and not shared_path:
        raise ValueError("q_nets_frozen is only supported on the shared-head path.")
    env = GuanDanEnv()
    env.reset(seed=seed)

    trajectory: list[dict] = []
    prof = profiler if profiler is not None else PhaseProfiler(enabled=False)

    while not env.done:
        p = env.current_player

        # Hard-bot seats: opponent acts via Agent.act and the step is NOT
        # recorded. The full game still plays out so terminal rewards remain
        # well-defined for the latest-team trajectory entries.
        if hard_bots is not None and p in hard_bot_seats:
            bot = hard_bots[p] if isinstance(hard_bots, dict) else hard_bots
            with prof.time("hard_bot_act"):
                action = bot.act(env, p)
            with prof.time("env_step"):
                env.step(action)
            continue

        # Per-seat routing: frozen seats use q_nets_frozen + epsilon_frozen,
        # latest seats use q_nets + epsilon. Both default to the latest path
        # when no frozen network is supplied.
        on_frozen = q_nets_frozen is not None and p in frozen_seats
        eps_p = epsilon_frozen if on_frozen else epsilon

        with prof.time("legal_actions"):
            legal = select_legal(env, p)
        K = len(legal)
        bucket = k_bucket_label(K)
        prof.add_count("num_decisions", 1)
        prof.add_count("num_legal_actions", K)
        # Per-K-bucket decision counts — totals sum to num_decisions.
        prof.add_count(f"decisions_{bucket}", 1)

        q_gap_for_this_step = float("nan")
        chosen_by_epsilon_flag = 0
        if K == 1:
            idx = 0
            prof.add_count("shortcut_K1", 1)
            with prof.time("encode_selected"):
                encoded = encoder.encode_one(env, p, legal[idx], legal)
        elif random.random() < eps_p:
            idx = random.randrange(K)
            chosen_by_epsilon_flag = 1
            prof.add_count("epsilon_random", 1)
            with prof.time("encode_selected"):
                encoded = encoder.encode_one(env, p, legal[idx], legal)
        elif inference_client is not None:
            with prof.time("encode_all"):
                encoded_list = encoder.encode_all(env, p, legal)
            with prof.time("inference_submit"):
                idx, _server_version = inference_client.submit(p, encoded_list)
            encoded = encoded_list[idx]
        else:
            with prof.time("encode_all"):
                encoded_list = encoder.encode_all(env, p, legal)
            if shared_path:
                acting_net = q_nets_frozen if on_frozen else q_nets
                idx, q_gap_for_this_step = argmax_q(
                    acting_net, encoded_list, device,
                    role_encoded=True, profiler=prof, bucket=bucket,
                )
            else:
                assert q_nets is not None
                idx, q_gap_for_this_step = argmax_q(
                    q_nets[p], encoded_list, device,
                    profiler=prof, bucket=bucket,
                )
            encoded = encoded_list[idx]

        partner = (p + 2) % 4
        action_type = int(legal[idx].type)
        trajectory.append({
            "player":             p,
            "encoded":            encoded,
            "phase_self":         phase_bucket(len(env.hands[p])),
            "trick_role":         trick_role(env, partner),
            "phase_partner":      phase_with_out(env, partner),
            "action_type":        action_type,
            "is_pass":            int(action_type == 0),
            "is_bomb":            int(action_type >= int(ComboType.BOMB_4)),
            "bomb_available":     int(any(m.type >= ComboType.BOMB_4 for m in legal)),
            "num_legal_actions":  K,
            "q_gap":              q_gap_for_this_step,
            "chosen_by_epsilon":  chosen_by_epsilon_flag,
        })
        with prof.time("env_step"):
            env.step(legal[idx])

    rewards = env.get_rewards()
    with prof.time("mc_returns"):
        return compute_mc_returns(
            trajectory,
            rewards,
            gamma=gamma,
            tags=tags,
        )


# ─── Rust rollout support (Phase 2: bytes-based scorer) ─────────────────────
#
# Rust encodes the full game state to f32 bytes. Python only needs to:
#   1. np.frombuffer the bytes into channel arrays   (zero-copy)
#   2. Build state_batch / action_batch tensors      (~10 tensor views)
#   3. Run NN forward                                (the bottleneck)
#   4. Pack the chosen action's encoding into a dict (stored in trajectory)
#
# This eliminates the Phase 1 overhead of serializing the full game state to a
# Python dict and reconstructing multihots from scratch on every step.

# Precompute state channel layout from role_encoder constants (once at import).
# Each entry: (channel_key, shape_tuple, byte_offset_in_f32s, num_f32s)
_STATE_CHANNELS: list[tuple[str, tuple[int, ...], int, int]] = []
_off = 0
for _key in ROLE_ENCODE_STATE_KEYS:
    _shape = ROLE_ENCODE_CHANNEL_SHAPES[_key]
    _numel = math.prod(_shape) if _shape else 1
    _STATE_CHANNELS.append((_key, _shape, _off, _numel))
    _off += _numel
del _off, _key, _shape, _numel

_ACTION_DIM = 108  # candidate_action only — matches encoder::ACTION_DIM in Rust


def _build_encoded_dict(
    state_bytes: bytes,
    action_bytes: bytes,
    idx: int,
    n_actions: int,
    head_id: int,
    head_field: str,
) -> dict[str, np.ndarray]:
    """Build the numpy-dict encoding for the idx-th action from byte buffers.

    The returned dict has the same layout as RoleAwareStateActionEncoder.encode_one
    so it is transparently accepted by the replay buffer and collate functions.
    Arrays are copied from the buffer so state_bytes can be freed immediately.
    """
    state_arr = np.frombuffer(state_bytes, dtype=np.float32)
    enc: dict[str, np.ndarray] = {}
    for key, shape, off, numel in _STATE_CHANNELS:
        enc[key] = state_arr[off : off + numel].reshape(shape).copy()
    action_arr = np.frombuffer(action_bytes, dtype=np.float32).reshape(
        n_actions, _ACTION_DIM
    )
    enc["candidate_action"] = action_arr[idx].copy()
    enc[head_field] = np.array(head_id, dtype=np.int64)
    return enc


def make_scorer(
    q_nets: Mapping[int, GuanZeroQNet] | SharedHeadQNet | SharedTrickHeadQNet,
    encoder: StateActionEncoder | RoleAwareStateActionEncoder,
    device: torch.device | str,
    epsilon: float,
) -> Callable[[bytes, bytes, int], tuple]:
    """Return a scorer closure for use with guandan_rs.play_episode_rust (Phase 2).

    The closure is called once per decision by the Rust loop with signature:
        scorer(state_bytes, action_bytes, head_id)
            -> (idx, encoded_dict, q_gap, chosen_by_epsilon)

    `state_bytes` and `action_bytes` are flat f32 buffers produced by the Rust
    encoder (layout matches ROLE_ENCODE_STATE_KEYS / ACTION_DIM). Only the shared-
    head (RoleAwareStateActionEncoder + absolute_seat) path is supported; the
    worker guards against other combinations before calling play_episode_rust.
    """
    device = torch.device(device)
    shared_path = isinstance(q_nets, (SharedHeadQNet, SharedTrickHeadQNet))
    head_field: str = getattr(encoder, "head_field", "seat_id")

    def scorer(state_bytes: bytes, action_bytes: bytes, head_id: int) -> tuple:
        n_actions = len(action_bytes) // (_ACTION_DIM * 4)

        if n_actions == 1 or random.random() < epsilon:
            idx = 0 if n_actions == 1 else random.randrange(n_actions)
            chosen = 0 if n_actions == 1 else 1
            encoded = _build_encoded_dict(
                state_bytes, action_bytes, idx, n_actions, head_id, head_field
            )
            return idx, encoded, float("nan"), chosen

        # np.frombuffer returns a read-only view; copy once so torch.from_numpy
        # gets a writable array (avoids PyTorch non-writable-tensor warning).
        state_arr = np.frombuffer(state_bytes, dtype=np.float32).copy()
        state_batch = {
            key: torch.from_numpy(state_arr[off : off + numel].reshape(1, *shape)).to(
                device, non_blocking=True
            )
            for key, shape, off, numel in _STATE_CHANNELS
        }

        action_arr = np.frombuffer(action_bytes, dtype=np.float32).copy().reshape(
            n_actions, _ACTION_DIM
        )
        action_batch = {
            "candidate_action": torch.from_numpy(action_arr).to(
                device, non_blocking=True
            ),
            head_field: torch.full(
                (n_actions,), head_id, dtype=torch.long, device=device
            ),
        }

        repeats = torch.tensor([n_actions], dtype=torch.long, device=device)
        with torch.inference_mode():
            if shared_path:
                q_vals = q_nets.forward_grouped(state_batch, action_batch, repeats)
            else:
                q_vals = q_nets[head_id].forward_grouped(
                    state_batch, action_batch, repeats
                )

        q_flat = q_vals.view(-1)
        idx = int(q_flat.argmax().item())
        q_gap = float("nan")
        if q_flat.numel() >= 2:
            top2 = torch.topk(q_flat, 2, largest=True).values
            q_gap = float(top2[0].item() - top2[1].item())

        encoded = _build_encoded_dict(
            state_bytes, action_bytes, idx, n_actions, head_id, head_field
        )
        return idx, encoded, q_gap, 0

    return scorer


def play_episode_rust(
    q_nets: Mapping[int, GuanZeroQNet] | SharedHeadQNet | SharedTrickHeadQNet,
    encoder: StateActionEncoder | RoleAwareStateActionEncoder,
    epsilon: float,
    *,
    seed: int | None = None,
    device: torch.device | str = "cpu",
    gamma: float = 1.0,
    tags: EpisodeTags = EpisodeTags(),
) -> list[TrainSample]:
    """Run one self-play episode via the Rust loop, return MC training samples.

    Only supports RoleAwareStateActionEncoder with head_scheme="absolute_seat".
    The worker guards this constraint; calling with other encoder types will
    produce silently wrong encodings.
    """
    import guandan_rs

    scorer = make_scorer(q_nets, encoder, device, epsilon)
    trajectory, rewards = guandan_rs.play_episode_rust(scorer, level_rank=2, seed=seed)
    return compute_mc_returns(
        trajectory,
        dict(enumerate(rewards)),
        gamma=gamma,
        tags=tags,
    )


__all__ = [
    "play_episode",
    "play_episode_rust",
    "make_scorer",
    "select_legal",
    "argmax_q",
]
