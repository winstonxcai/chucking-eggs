"""Q-guided determinized search (IS-MCTS without the tree).

Sample possible opponent hands, simulate moves ahead with Q-network guidance,
pick the move that scores best on average across sampled worlds.
"""

from __future__ import annotations

import copy

import numpy as np
import torch

from ..agents.base import Agent
from ..game import GuanDanEnv
from ..training.encoding import encode_action, encode_history, encode_state
from ..training.q_network import QNetworkLSTM
from .determinize import sample_consistent_deal
from .simulate import evaluate_position


def q_search(
    env: GuanDanEnv,
    player: int,
    q_lead: QNetworkLSTM,
    q_follow: QNetworkLSTM,
    device: torch.device,
    level_rank: int,
    n_worlds: int = 20,
    depth: int = 3,
    prune_top_k: int = 5,
    opp_bot: Agent | None = None,
) -> tuple:
    """Main search: determinize → simulate → evaluate → pick best.

    Returns (best_combo, info_dict).
    """
    legal = env.legal_moves(player)
    if len(legal) == 1:
        return legal[0], {"n_candidates": 1, "pruned": False}

    # Step 1: compute raw Q-values for pruning
    is_leading = env.current_trick is None
    q_net = q_lead if is_leading else q_follow

    state_enc = encode_state(env, player)
    hand = env.hands[player]
    action_encs = np.array(
        [encode_action(m, hand, level_rank) for m in legal]
    )
    history, hist_len = encode_history(env, player, level_rank)

    B = len(legal)
    with torch.no_grad():
        s = torch.tensor(state_enc, device=device).unsqueeze(0).expand(B, -1)
        a = torch.tensor(action_encs, device=device)
        h = torch.tensor(history, device=device).unsqueeze(0).expand(B, -1, -1)
        hl = torch.tensor([hist_len], dtype=torch.long, device=device).expand(B)
        raw_q = q_net(s, a, h, hl).cpu().numpy()

    # Step 2: prune to top-K candidates
    if len(legal) <= prune_top_k:
        candidates = list(range(len(legal)))
        pruned = False
    else:
        candidates = np.argsort(raw_q)[-prune_top_k:].tolist()
        pruned = True

    # Step 3: search over determinized worlds
    scores = np.zeros(len(candidates), dtype=np.float64)

    for w in range(n_worlds):
        world_base = sample_consistent_deal(env, player)

        for ci, move_idx in enumerate(candidates):
            world = copy.deepcopy(world_base)
            world.step(legal[move_idx])
            # Full rollout to terminal reward
            scores[ci] += evaluate_position(
                world, player, q_lead, q_follow, device, level_rank,
                opp_bot=opp_bot,
            )

    scores /= n_worlds
    best_ci = int(np.argmax(scores))
    best_idx = candidates[best_ci]

    info = {
        "raw_q": raw_q.tolist(),
        "search_scores": {candidates[i]: scores[i] for i in range(len(candidates))},
        "n_candidates": len(candidates),
        "pruned": pruned,
        "n_worlds": n_worlds,
        "depth": depth,
    }

    return legal[best_idx], info
