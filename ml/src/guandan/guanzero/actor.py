"""Single-episode self-play rollout.

Public API: play_episode, select_legal, argmax_q.
"""

from __future__ import annotations

import random
from typing import Mapping

import torch

from ..cards import ComboType
from ..combos import Combo
from ..game import GuanDanEnv
from .buffer import collate_encoded
from .encoder import StateActionEncoder
from .legal_utils import dedup_strategic
from .profiler import PhaseProfiler, _k_bucket
from .q_network import GuanZeroQNet
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
    batch = collate_encoded(encoded_list, device=device)
    q = net(batch)
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
        batch = collate_encoded(encoded_list, device=device)
    with prof.time(f"q_net_forward_{bucket}"):
        with torch.no_grad():
            q_vals = net(batch)
    with prof.time("q_argmax_item"):
        return int(q_vals.argmax().item())


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
    local-CPU path runs (q_nets must be provided).
    """
    device = torch.device(device)
    env = GuanDanEnv()
    env.reset(seed=seed)

    trajectory: list[dict] = []
    prof = profiler if profiler is not None else PhaseProfiler(enabled=False)

    while not env.done:
        p = env.current_player
        with prof.time("legal_actions"):
            legal = select_legal(env, p)
        K = len(legal)
        bucket = _k_bucket(K)
        prof.add_count("num_decisions", 1)
        prof.add_count("num_legal_actions", K)
        # Per-K-bucket decision counts — totals sum to num_decisions.
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
            idx = _argmax_with_timing(q_nets[p], encoded_list, device, prof, bucket)

        trajectory.append({"player": p, "encoded": encoded_list[idx]})
        with prof.time("env_step"):
            env.step(legal[idx])

    rewards = env.get_rewards()
    with prof.time("mc_returns"):
        return compute_mc_returns(trajectory, rewards, gamma=gamma)


__all__ = ["play_episode", "select_legal", "argmax_q"]
