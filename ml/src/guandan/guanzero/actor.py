"""Single-episode self-play rollout.

Public API: play_episode, select_legal, argmax_q, argmax_q_role.
"""

from __future__ import annotations

import random
from typing import Mapping

import torch

from ..cards import ComboType
from ..combos import Combo
from ..game import GuanDanEnv
from .buffer import collate_base_encoded, collate_role_encoded
from .encoder import StateActionEncoder
from .encoding.role_encoder import RoleAwareStateActionEncoder
from .legal_utils import dedup_strategic
from .profiler import PhaseProfiler, _k_bucket
from .q_network import GuanZeroQNet, SharedHeadQNet
from .returns import TrainSample, compute_mc_returns


_PASS = Combo(ComboType.PASS, 0, [])


def select_legal(env: GuanDanEnv, player: int) -> list[Combo]:
    """Return the deduplicated legal moves for ``player``.

    Appends an explicit PASS move if the player must respond but PASS is
    absent from the raw legal-move set (can happen with strict-response rules).
    """
    legal = dedup_strategic(env.legal_moves(player))
    if not env.is_leading() and not any(m.type == ComboType.PASS for m in legal):
        legal.append(_PASS)
    return legal


@torch.no_grad()
def argmax_q(
    net: GuanZeroQNet,
    encoded_list: list[dict],
    device: torch.device,
) -> int:
    """Score ``encoded_list`` with ``net`` and return the greedy argmax index."""
    state_batch, action_batch, repeats = collate_base_encoded([encoded_list], device=device)
    q = net.forward_grouped(state_batch, action_batch, repeats)
    return int(q.argmax().item())


@torch.no_grad()
def argmax_q_role(
    net: SharedHeadQNet,
    encoded_list: list[dict],
    device: torch.device,
) -> int:
    """Score role-encoded candidates and return the greedy argmax index."""
    state_batch, action_batch, repeats = collate_role_encoded([encoded_list], device=device)
    q = net.forward_grouped(state_batch, action_batch, repeats)
    return int(q.argmax().item())


def _argmax_with_timing(
    net: GuanZeroQNet,
    encoded_list: list[dict],
    device: torch.device,
    prof: PhaseProfiler,
    bucket: str,
) -> int:
    """Instrumented greedy action selection used inside play_episode."""
    with prof.time("q_collate"):
        state_batch, action_batch, repeats = collate_base_encoded(
            [encoded_list],
            device=device,
        )
    with prof.time(f"q_net_forward_{bucket}"):
        with torch.no_grad():
            q_vals = net.forward_grouped(state_batch, action_batch, repeats)
    with prof.time("q_argmax_item"):
        return int(q_vals.argmax().item())


def _argmax_role_with_timing(
    net: SharedHeadQNet,
    encoded_list: list[dict],
    device: torch.device,
    prof: PhaseProfiler,
    bucket: str,
) -> int:
    """Instrumented greedy action selection for the shared-head path."""
    with prof.time("q_collate"):
        state_batch, action_batch, repeats = collate_role_encoded(
            [encoded_list],
            device=device,
        )
    with prof.time(f"q_net_forward_{bucket}"):
        with torch.no_grad():
            q_vals = net.forward_grouped(state_batch, action_batch, repeats)
    with prof.time("q_argmax_item"):
        return int(q_vals.argmax().item())


def play_episode(
    q_nets: Mapping[int, GuanZeroQNet] | SharedHeadQNet | None,
    encoder: StateActionEncoder | RoleAwareStateActionEncoder,
    epsilon: float,
    seed: int | None = None,
    device: torch.device | str = "cpu",
    gamma: float = 1.0,
    inference_client=None,
    profiler: PhaseProfiler | None = None,
    *,
    q_nets_frozen: SharedHeadQNet | None = None,
    frozen_seats: frozenset[int] = frozenset(),
    epsilon_frozen: float = 0.0,
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

    Sample emission is unchanged — callers filter by ``s.player`` if they want
    to exclude frozen-team rows from the buffer.
    """
    device = torch.device(device)
    shared_path = isinstance(q_nets, SharedHeadQNet)
    if shared_path and inference_client is not None:
        raise ValueError("Inference server is not supported for SharedHeadQNet.")
    if q_nets_frozen is not None and not shared_path:
        raise ValueError("q_nets_frozen is only supported on the shared-head path.")
    env = GuanDanEnv()
    env.reset(seed=seed)

    trajectory: list[dict] = []
    prof = profiler if profiler is not None else PhaseProfiler(enabled=False)

    while not env.done:
        p = env.current_player
        # Per-seat routing: frozen seats use q_nets_frozen + epsilon_frozen,
        # latest seats use q_nets + epsilon. Both default to the latest path
        # when no frozen network is supplied.
        on_frozen = q_nets_frozen is not None and p in frozen_seats
        eps_p = epsilon_frozen if on_frozen else epsilon

        with prof.time("legal_actions"):
            legal = select_legal(env, p)
        K = len(legal)
        bucket = _k_bucket(K)
        prof.add_count("num_decisions", 1)
        prof.add_count("num_legal_actions", K)
        # Per-K-bucket decision counts — totals sum to num_decisions.
        prof.add_count(f"decisions_{bucket}", 1)

        if K == 1:
            idx = 0
            prof.add_count("shortcut_K1", 1)
            with prof.time("encode_selected"):
                encoded = encoder.encode_one(env, p, legal[idx], legal)
        elif random.random() < eps_p:
            idx = random.randrange(K)
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
                idx = _argmax_role_with_timing(acting_net, encoded_list, device, prof, bucket)
            else:
                assert q_nets is not None
                idx = _argmax_with_timing(q_nets[p], encoded_list, device, prof, bucket)
            encoded = encoded_list[idx]

        trajectory.append({"player": p, "encoded": encoded})
        with prof.time("env_step"):
            env.step(legal[idx])

    rewards = env.get_rewards()
    with prof.time("mc_returns"):
        return compute_mc_returns(trajectory, rewards, gamma=gamma)


__all__ = ["play_episode", "select_legal", "argmax_q", "argmax_q_role"]
