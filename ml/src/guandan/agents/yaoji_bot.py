"""YaojiBot — faithful port of NUAA's 3rd-place competition entry.

Source: 1st NJUPT Guan Dan AI Competition (2020)
Team: 幺鸡小分队
Author: Qian Zenghui (钱增辉), NUAA (南京航空航天大学)
Prize: 3rd Place

Architecture: Score = Gain * (1 + Possibility) / Value scoring formula applied
to all legal actions, with partner-awareness (mate_pos). Passes server msg dict
to solve() and returns the highest-scoring action index.
"""

from __future__ import annotations

from ..cards import ComboType, Rank
from ._vendor.adapter import (
    cards_to_strings,
    combo_to_action_list,
    find_pass,
    rank_to_string,
)
from ._vendor.yaoji.mysolve import solve
from .base import Agent


class YaojiBot(Agent):
    """NUAA 3rd Prize — gain/possibility/value scoring with partner awareness."""

    label = "Yaoji"
    description = "NUAA 3rd Prize · 2020 NJUPT entry."
    source = "NUAA"
    award = "3rd Prize"
    sample_tag = 2
    coord_target = True

    def __init__(self, level_rank: int = Rank.TWO):
        self.level_rank = level_rank

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
        mate_pos = (player + 2) % 4

        msg = {
            "curRank": rank_str,
            "greaterPos": greater_pos,
            "handCards": hand_strings,
            "actionList": action_list,
            "publicInfo": [{"rest": len(env.hands[p])} for p in range(4)],
        }

        try:
            idx = solve(msg, mate_pos)
            if idx is None or idx < 0 or idx >= len(action_list):
                idx = 0
        except Exception:
            idx = 0

        if idx == 0:
            return find_pass(legal)
        return combo_map[idx] if idx < len(combo_map) else find_pass(legal)
