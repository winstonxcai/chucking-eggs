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

from ..cards import Rank
from ._vendor.adapter import (
    cards_to_strings,
    find_pass,
    rank_to_string,
)
from ._vendor.jidan.message_Reyn_CUR import check_message
from .base import Agent
from .competition_bridge import (
    build_action_list,
    current_comp_context,
    env_to_comp_pos,
    public_info_for_comp_positions,
)


class JidanBot(Agent):
    """NUAA 2nd Prize — weighted card value scoring (Reyn_AI 2.0)."""

    label = "Jidan"
    description = "NUAA 2nd Prize · 2020 NJUPT entry."
    source = "NUAA"
    color = "#8c564b"
    award = "2nd Prize"
    sample_tag = 3
    coord_target = True

    def __init__(self, level_rank: int = Rank.TWO):
        self.level_rank = level_rank

    def act(self, env, player: int):
        legal = env.legal_moves(player)
        if len(legal) == 1:
            return legal[0]

        rank_str = rank_to_string(self.level_rank)
        hand_strings = cards_to_strings(env.hands[player])
        action_list, combo_map = build_action_list(legal, self.level_rank)
        _, _, greater_pos, greater_action = current_comp_context(env, self.level_rank)
        comp_player = env_to_comp_pos(player)

        msg = {
            "curRank": rank_str,
            "stage": "play",
            "greaterPos": greater_pos,
            "greaterAction": greater_action,
            "handCards": hand_strings,
            "actionList": action_list,
            "publicInfo": public_info_for_comp_positions(env),
        }

        try:
            idx = check_message(msg, comp_player)
            if idx is None or idx < 0 or idx >= len(action_list):
                idx = 0
        except Exception:
            idx = 0

        if idx == 0:
            return find_pass(legal)
        return combo_map[idx] if idx < len(combo_map) else find_pass(legal)
