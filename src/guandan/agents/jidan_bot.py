"""JidanBot — faithful port of NUAA's 2nd-place competition entry.

Source: 1st NJUPT Guan Dan AI Competition (2020)
Team: 鸡蛋灌饼
Author: Sun Hailong (孙海龙), NUAA (南京航空航天大学)
Prize: 2nd Place

Architecture: Reyn_AI 2.0 — weighted card value system with comprehensive
suit-by-suit hand analysis. Uses get_VAL() to score each legal action and
returns the highest-value move. Separate logic for leading vs following.
"""

from __future__ import annotations

from ..cards import ComboType, Rank
from ._vendor.adapter import (
    cards_to_strings,
    combo_to_action_list,
    find_pass,
    rank_to_string,
)
from ._vendor.jidan.message_Reyn_CUR import check_message
from .base import Agent


class JidanBot(Agent):
    """NUAA 2nd Prize — weighted card value scoring (Reyn_AI 2.0)."""

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
            "handCards": hand_strings,
            "actionList": action_list,
            "publicInfo": [{"rest": len(env.hands[p])} for p in range(4)],
        }

        try:
            idx = check_message(msg, player)
            if idx is None or idx < 0 or idx >= len(action_list):
                idx = 0
        except Exception:
            idx = 0

        if idx == 0:
            return find_pass(legal)
        return combo_map[idx] if idx < len(combo_map) else find_pass(legal)
