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

from ..cards import Rank
from ._vendor.adapter import (
    cards_to_strings,
    find_pass,
    rank_to_string,
)
from ._vendor.ez.action import Action
from .base import Agent
from .competition_bridge import (
    build_action_list,
    build_play_message,
    env_to_comp_pos,
)


class EzBot(Agent):
    """HYIT 3rd Prize — combo analysis with team-aware action selection."""

    label = "Ez"
    description = "HYIT 3rd Prize · 2020 NJUPT entry."
    source = "HYIT"
    color = "#17becf"
    award = "3rd Prize"
    sample_tag = 12
    coord_target = True

    def __init__(self, level_rank: int = Rank.TWO):
        self.level_rank = level_rank
        self._action = Action()
        self._last_history_len = -1

    def act(self, env, player: int):
        legal = env.legal_moves(player)
        if len(legal) == 1:
            return legal[0]

        history_len = len(env.move_history)
        if history_len < self._last_history_len or self._last_history_len == -1:
            self._action = Action()
        self._last_history_len = history_len

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
            self._action.myclient = env_to_comp_pos(player)
            idx = self._action.parse(msg)
            if idx is None or idx < 0 or idx >= len(action_list):
                idx = 0
        except Exception:
            idx = 0

        if idx == 0:
            return find_pass(legal)
        return combo_map[idx] if idx < len(combo_map) else find_pass(legal)
