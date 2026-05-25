"""HulalalaBot — faithful port of Southeast University's 3rd-place competition entry.

Source: 1st NJUPT Guan Dan AI Competition (2020)
Team: hulalala
Author: Yang Xinyan (杨欣妍), Southeast University (东南大学)
Prize: 3rd Place

Architecture: same per-type heuristic style as Lalala/Liuzha (SEU family),
with module-level active() and passive() functions. Same active()/passive()
interface pattern.
"""

from __future__ import annotations

from ..cards import Rank
from ._vendor.adapter import (
    cards_to_strings,
    find_pass,
    rank_to_string,
)
from ._vendor.hulalala.active import active
from ._vendor.hulalala.passive import passive
from .base import Agent
from .competition_bridge import (
    CompetitionPlayState,
    build_action_list,
    current_comp_context,
    env_to_comp_pos,
    public_info_for_comp_positions,
)


class HulalalaBot(Agent):
    """SEU 3rd Prize — per-type heuristic with active()/passive() dispatch."""

    label = "Hulalala"
    description = "SEU 3rd Prize · 2020 NJUPT entry."
    source = "SEU"
    color = "#bcbd22"
    award = "3rd Prize"
    sample_tag = 11

    def __init__(self, level_rank: int = Rank.TWO):
        self.level_rank = level_rank
        self._states: dict[int, CompetitionPlayState] = {}

    def act(self, env, player: int):
        legal = env.legal_moves(player)
        if len(legal) == 1:
            return legal[0]

        comp_player = env_to_comp_pos(player)
        state = self._states.setdefault(player, CompetitionPlayState(comp_player))
        state.sync_from_env(env, self.level_rank)

        rank_str = rank_to_string(self.level_rank)
        hand_strings = cards_to_strings(env.hands[player])
        numofplayers = [
            int(row["rest"]) for row in public_info_for_comp_positions(env)
        ]
        action_list, combo_map = build_action_list(legal, self.level_rank)

        if env.current_trick is None:
            idx = self._lead(
                action_list,
                hand_strings,
                rank_str,
                numofplayers,
                comp_player,
                state.remain_cards,
            )
        else:
            idx = self._follow(
                env,
                comp_player,
                action_list,
                hand_strings,
                rank_str,
                state.remain_cards,
                numofplayers,
                state,
            )

        if idx is None or idx < 0 or idx >= len(action_list):
            idx = 0

        if idx == 0:
            return find_pass(legal)
        return combo_map[idx] if idx < len(combo_map) else find_pass(legal)

    def _lead(self, action_list, hand_strings, rank_str, numofplayers, player, remaincards):
        try:
            idx = active(action_list, hand_strings, rank_str, numofplayers, player, remaincards)
            return idx if idx is not None else 0
        except Exception:
            return 0

    def _follow(
        self,
        env,
        comp_player,
        action_list,
        hand_strings,
        rank_str,
        remaincards,
        numofplayers,
        state,
    ):
        _, cur_action, greater_pos, greater_action = current_comp_context(
            env,
            self.level_rank,
        )
        try:
            idx = passive(
                action_list, hand_strings, rank_str,
                cur_action, greater_action,
                comp_player, greater_pos,
                remaincards, numofplayers,
                state.pass_num, state.my_pass_num,
                state.remain_cards_classbynum,
            )
            result = idx if idx is not None else 0
            return result
        except Exception:
            return 1
