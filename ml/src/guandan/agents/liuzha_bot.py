"""LiuzhaBot — faithful port of Southeast University's 2nd-place competition entry.

Source: 1st NJUPT Guan Dan AI Competition (2020)
Team: 小子六炸同花顺
Author: Yan Hui (严辉), Southeast University (东南大学)
Prize: 2nd Place

Architecture: heuristic with per-combo-type handlers (Single, Pair, Trips,
ThreePair, ThreeWithTwo, TwoTrips, Straight, Bomb), next-player card-count
optimization, and endgame one_hand special-casing. Same active()/passive()
interface as Lalala (also SEU).
"""

from __future__ import annotations

from ..cards import Rank
from ._vendor.adapter import (
    cards_to_strings,
    find_pass,
    rank_to_string,
)
from ._vendor.liuzha.action import Action
from .base import Agent
from .competition_bridge import (
    CompetitionPlayState,
    build_action_list,
    build_play_message,
    env_to_comp_pos,
)


class LiuzhaBot(Agent):
    """SEU 2nd Prize — heuristic with per-type handlers and next-player optimization."""

    label = "Liuzha"
    description = "SEU 2nd Prize · 2020 NJUPT entry."
    source = "SEU"
    color = "#7f7f7f"
    award = "2nd Prize"
    sample_tag = 10

    def __init__(self, level_rank: int = Rank.TWO):
        self.level_rank = level_rank
        self._actions: dict[int, Action] = {}
        self._states: dict[int, CompetitionPlayState] = {}

    def act(self, env, player: int):
        legal = env.legal_moves(player)
        if len(legal) == 1:
            return legal[0]

        comp_player = env_to_comp_pos(player)
        state = self._states.setdefault(player, CompetitionPlayState(comp_player))
        state.sync_from_env(env, self.level_rank)
        action = self._actions.setdefault(player, Action(f"liuzha_bot_{player}"))

        rank_str = rank_to_string(self.level_rank)
        hand_strings = cards_to_strings(env.hands[player])
        action_list, combo_map = build_action_list(legal, self.level_rank)

        msg = build_play_message(
            env,
            player,
            action_list=action_list,
            hand_strings=hand_strings,
            rank_str=rank_str,
            level_rank=self.level_rank,
        )
        try:
            idx = action.rule_parse(
                msg,
                comp_player,
                state.remain_cards,
                state.history,
                state.remain_cards_classbynum,
                state.pass_num,
                state.my_pass_num,
                state.tribute_result,
            )
        except Exception:
            idx = 0

        if idx is None or idx < 0 or idx >= len(action_list):
            idx = 0

        if idx == 0:
            return find_pass(legal)
        return combo_map[idx] if idx < len(combo_map) else find_pass(legal)
