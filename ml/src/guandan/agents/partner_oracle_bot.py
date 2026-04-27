"""PartnerOracleBot — Direction C: Jidan-distilled policy + (optional) PIMC search.

The policy net is trained on team {0,2} only. At inference for seats {1,3}
we swap (0↔1, 2↔3) before encoding so the network sees its training-time view.

Two modes:
  use_search=False : pure policy net argmax (Stage 1 eval)
  use_search=True  : policy proposes top-K, PartnerPIMCBot evaluates (Stage 2)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..cards import Rank
from ..combos import Combo
from ..game import GuanDanEnv
from ..azguan import (
    ACTION_DIM,
    QValueNet,
    STATE_DIM_TEAM_WITH_FLAGS,
    encode_action,
    encode_state_team_with_flags,
    get_device,
)
from .base import Agent
from .belief import BeliefModel
from .partner_pimc_bot import _N_WORKERS, PartnerPIMCBot, _clone_env


def _reflect_env(env: GuanDanEnv) -> GuanDanEnv:
    """Swap seats (0↔1, 2↔3) so seat 1 becomes 0, seat 3 becomes 2.

    Used at inference when our team is {1,3} but the encoder expects team
    {0,2}. The Combo objects (cards) are unchanged — only seat indices move.
    """
    new = _clone_env(env)
    new.hands = [env.hands[1], env.hands[0], env.hands[3], env.hands[2]]
    new.played = [env.played[1], env.played[0], env.played[3], env.played[2]]
    new.is_out = [env.is_out[1], env.is_out[0], env.is_out[3], env.is_out[2]]
    new.current_player = env.current_player ^ 1
    if env.trick_winner is not None:
        new.trick_winner = env.trick_winner ^ 1
    new.finish_order = [s ^ 1 for s in env.finish_order]
    return new


class PartnerOracleBot(Agent):
    """Jidan-distilled policy net, optionally wrapped with PartnerPIMC search.

    Args:
        checkpoint_path: path to jidan_policy.pt (saved by distill_jidan.py).
        level_rank: current game level (passed to internal sub-agents).
        use_search: if True, run PartnerPIMCBot on top-K candidates.
        top_k: number of candidates to pass to search (default 3).
        n_det: PIMC determinizations (only if use_search).
        use_belief: if True and use_search, pass a BeliefModel into PartnerPIMC
                    so determinizations respect pass-derived hard constraints.
        device: torch device override (default auto-detect).
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        level_rank: int = Rank.TWO,
        use_search: bool = True,
        top_k: int = 3,
        n_det: int = 30,
        use_belief: bool = False,
        n_workers: int = _N_WORKERS,
        use_value_leaf: bool = False,
        device: torch.device | None = None,
    ):
        self.level_rank = level_rank
        self.use_search = use_search
        self.top_k = top_k
        self.device = device or get_device()

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        cfg = ckpt["config"]
        self.net = QValueNet(
            d_state=cfg["d_state"],
            d_action=cfg["d_action"],
            hidden=cfg["hidden"],
        ).to(self.device)
        # strict=False: gen-0 checkpoints (jidan_policy.pt) lack v_net weights;
        # those stay zero-init until AZ training fills them in.
        self.net.load_state_dict(ckpt["state_dict"], strict=False)
        self.net.eval()

        if use_search:
            self._search = PartnerPIMCBot(
                level_rank=level_rank,
                n_det=n_det,
                n_cands=top_k,
                depth_limit=0,
                pre_filter=self._policy_top_k,
                rollout_policy="jidan",
                belief=BeliefModel() if use_belief else None,
                n_workers=n_workers,
                value_net=self.net if use_value_leaf else None,
            )
        else:
            self._search = None

    @torch.no_grad()
    def _score_actions(
        self, env: GuanDanEnv, player: int, legal: list[Combo]
    ) -> np.ndarray:
        """Q-values per legal action. State is per-action (includes 9-dim
        team-coordination behavior flags computed for each candidate)."""
        if player in (0, 2):
            enc_env = env
            enc_player = player
        else:
            enc_env = _reflect_env(env)
            enc_player = player ^ 1

        states = np.stack([
            encode_state_team_with_flags(enc_env, enc_player, a, legal)
            for a in legal
        ]).astype(np.float32)
        actions = np.stack([
            encode_action(a, enc_env.hands[enc_player], enc_env.level_rank)
            for a in legal
        ]).astype(np.float32)

        st = torch.from_numpy(states).to(self.device)
        at = torch.from_numpy(actions).to(self.device)
        q = self.net(st, at)  # [K]
        return q.cpu().numpy()

    def _policy_top_k(
        self, env: GuanDanEnv, player: int, candidates: list[Combo]
    ) -> list[Combo]:
        """Pre-filter callable plugged into PartnerPIMCBot."""
        if len(candidates) <= self.top_k:
            return candidates
        q = self._score_actions(env, player, candidates)
        top_idx = np.argsort(-q)[: self.top_k]
        return [candidates[i] for i in top_idx]

    def act(self, env: GuanDanEnv, player: int) -> Combo:
        legal = env.legal_moves(player)
        if len(legal) == 1:
            return legal[0]

        if self._search is not None:
            return self._search.act(env, player)

        # Stage 1: pure policy argmax
        q = self._score_actions(env, player, legal)
        return legal[int(np.argmax(q))]
