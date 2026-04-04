"""Q-guided simulation primitives for determinized search.

All functions pick q_lead vs q_follow per-step based on env.current_trick.
Supports hybrid rollout: Q-network for our team, bot for opponents.
"""

from __future__ import annotations

import numpy as np
import torch

from ..agents.base import Agent
from ..game import GuanDanEnv
from ..training.encoding import encode_action, encode_history, encode_state
from ..training.q_network import QNetworkLSTM


def _q_forward(q_net, state_enc, action_encs, history, hist_len, device):
    """Shared forward pass logic. Returns Q-values tensor."""
    B = len(action_encs)
    s = torch.tensor(state_enc, device=device).unsqueeze(0).expand(B, -1)
    a = torch.tensor(np.array(action_encs), device=device)
    h = torch.tensor(history, device=device).unsqueeze(0).expand(B, -1, -1)
    hl = torch.tensor([hist_len], dtype=torch.long, device=device).expand(B)
    return q_net(s, a, h, hl)


def q_argmax(
    env: GuanDanEnv,
    player: int,
    q_lead: QNetworkLSTM,
    q_follow: QNetworkLSTM,
    device: torch.device,
    level_rank: int,
) -> "Combo":
    """Pick highest-Q legal move. Routes to q_lead/q_follow per-step."""
    legal = env.legal_moves(player)
    if len(legal) == 1:
        return legal[0]

    q_net = q_lead if env.current_trick is None else q_follow
    state_enc = encode_state(env, player)
    action_encs = [encode_action(m, env.hands[player], level_rank) for m in legal]
    history, hist_len = encode_history(env, player, level_rank)

    with torch.no_grad():
        q_vals = _q_forward(q_net, state_enc, action_encs, history, hist_len, device)
        idx = q_vals.argmax().item()

    return legal[idx]


def rollout_terminal(
    env: GuanDanEnv,
    search_player: int,
    rollout_bot: Agent,
) -> float:
    """Fast rollout using a bot policy (no NN). Returns terminal reward."""
    if env.done:
        return env.get_rewards()[search_player]

    while not env.done:
        cp = env.current_player
        move = rollout_bot.act(env, cp)
        env.step(move)

    return env.get_rewards()[search_player]


def evaluate_position(
    env: GuanDanEnv,
    search_player: int,
    q_lead: QNetworkLSTM,
    q_follow: QNetworkLSTM,
    device: torch.device,
    level_rank: int,
    opp_bot: Agent | None = None,
) -> float:
    """Evaluate position via rollout to terminal reward.

    If opp_bot is provided, uses it for ALL players (fast PIMC rollout).
    Otherwise uses q_argmax for all players (slow but higher quality).
    """
    if env.done:
        return env.get_rewards()[search_player]

    our_team = {search_player, (search_player + 2) % 4}

    # Hybrid rollout: Q-argmax for our team, opp_bot for opponents
    while not env.done:
        cp = env.current_player
        if opp_bot is not None and cp not in our_team:
            move = opp_bot.act(env, cp)
        else:
            move = q_argmax(env, cp, q_lead, q_follow, device, level_rank)
        env.step(move)

    return env.get_rewards()[search_player]


def simulate(
    env: GuanDanEnv,
    depth: int,
    q_lead: QNetworkLSTM,
    q_follow: QNetworkLSTM,
    device: torch.device,
    level_rank: int,
) -> None:
    """Simulate `depth` decisions in-place using q_argmax for all players.

    Stops early if env.done.
    """
    for _ in range(depth):
        if env.done:
            return
        player = env.current_player
        move = q_argmax(env, player, q_lead, q_follow, device, level_rank)
        env.step(move)
