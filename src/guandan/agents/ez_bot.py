"""EzBot — faithful port of HYIT's 3rd-place competition entry.

Source: 1st NJUPT Guan Dan AI Competition (2020)
Team: ez
Author: Wu Jun (吴俊), Huaiyin Institute of Technology (淮阴工学院)
Prize: 3rd Place

Architecture: card combination analysis (Myfunc1014) with composite action
selection — straight/bomb/endgame special cases, team awareness via
myclient % 2 == greaterPos % 2. Originally used separate per-position
client files reading player index from data*.txt; player index is now
injected via Action.myclient.
"""

from __future__ import annotations

from ..cards import ComboType, Rank
from ._vendor.adapter import (
    cards_to_strings,
    combo_to_action_list,
    find_pass,
    rank_to_string,
)
from ._vendor.ez.action import Action
from .base import Agent


class EzBot(Agent):
    """HYIT 3rd Prize — combo analysis with team-aware action selection."""

    def __init__(self, level_rank: int = Rank.TWO):
        self.level_rank = level_rank
        self._action = Action()

    def act(self, env, player: int):
        legal = env.legal_moves(player)
        if len(legal) == 1:
            return legal[0]

        rank_str = rank_to_string(self.level_rank)
        hand_strings = cards_to_strings(env.hands[player])

        action_list = [["PASS", "PASS", []]]
        combo_map = [None]
        for combo in legal:
            if combo.type == ComboType.PASS:
                continue
            action_list.append(combo_to_action_list(combo, self.level_rank))
            combo_map.append(combo)

        greater_pos = -1 if env.current_trick is None else (
            env.trick_winner if env.trick_winner is not None else player
        )
        greater_action = (
            combo_to_action_list(env.current_trick, self.level_rank)
            if env.current_trick is not None
            else ["PASS", "PASS", []]
        )

        msg = {
            "curRank": rank_str,
            "stage": "play",
            "greaterPos": greater_pos,
            "greaterAction": greater_action,
            "curAction": greater_action,
            "handCards": hand_strings,
            "actionList": action_list,
            "indexRange": len(action_list) - 1,
            "publicInfo": [{"rest": len(env.hands[p])} for p in range(4)],
        }

        try:
            self._action.myclient = player
            idx = self._action.parse(msg)
            if idx is None or idx < 0 or idx >= len(action_list):
                idx = 0
        except Exception:
            idx = 0

        if idx == 0:
            return find_pass(legal)
        return combo_map[idx] if idx < len(combo_map) else find_pass(legal)
