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

from ..cards import ComboType, Rank
from ._vendor.adapter import (
    cards_to_strings,
    combo_to_action_list,
    find_pass,
    rank_to_string,
)
from ._vendor.liuzha.action import Action
from .base import Agent


_RANK_IDX = {
    "A": 0, "2": 1, "3": 2, "4": 3, "5": 4, "6": 5, "7": 6,
    "8": 7, "9": 8, "T": 9, "J": 10, "Q": 11, "K": 12,
}


def _build_remaincards(hand_strings: list[str]) -> dict:
    """Build 2-deck card count dict, subtracting known hand cards."""
    remaincards = {
        "S": [2] * 13 + [2],
        "H": [2] * 13 + [2],
        "C": [2] * 13 + [0],
        "D": [2] * 13 + [0],
    }
    for card in hand_strings:
        suit = card[0]
        rank = card[1]
        if rank in ("B", "R"):
            remaincards[suit][13] = max(0, remaincards[suit][13] - 1)
        elif rank in _RANK_IDX:
            idx = _RANK_IDX[rank]
            remaincards[suit][idx] = max(0, remaincards[suit][idx] - 1)
    return remaincards


class LiuzhaBot(Agent):
    """SEU 2nd Prize — heuristic with per-type handlers and next-player optimization."""

    def __init__(self, level_rank: int = Rank.TWO):
        self.level_rank = level_rank
        self._action = Action("liuzha_bot")
        self._last_history_len = -1
        self._remaining = {0: 27, 1: 27, 2: 27, 3: 27}
        self._pass_num = 0
        self._my_pass_num = 0

    def act(self, env, player: int):
        legal = env.legal_moves(player)
        if len(legal) == 1:
            return legal[0]

        history_len = len(env.move_history)
        if history_len < self._last_history_len or self._last_history_len == -1:
            self._remaining = {0: 27, 1: 27, 2: 27, 3: 27}
            self._pass_num = 0
            self._my_pass_num = 0
        self._last_history_len = history_len

        for p in range(4):
            self._remaining[p] = len(env.hands[p])

        rank_str = rank_to_string(self.level_rank)
        hand_strings = cards_to_strings(env.hands[player])
        remaincards = _build_remaincards(hand_strings)

        action_list = [["PASS", "PASS", []]]
        combo_map = [None]
        for combo in legal:
            if combo.type == ComboType.PASS:
                continue
            action_list.append(combo_to_action_list(combo, self.level_rank))
            combo_map.append(combo)

        if env.current_trick is None:
            idx = self._lead(action_list, hand_strings, rank_str, player, remaincards)
        else:
            idx = self._follow(env, player, action_list, hand_strings, rank_str, remaincards)

        if idx is None or idx < 0 or idx >= len(action_list):
            idx = 0

        if idx == 0:
            return find_pass(legal)
        return combo_map[idx] if idx < len(combo_map) else find_pass(legal)

    def _lead(self, action_list, hand_strings, rank_str, player, remaincards):
        try:
            idx = self._action.active(
                action_list, hand_strings, rank_str,
                self._remaining, player, remaincards,
            )
            return idx if idx is not None else 0
        except Exception:
            return 0

    def _follow(self, env, player, action_list, hand_strings, rank_str, remaincards):
        trick_type = combo_to_action_list(env.current_trick, self.level_rank)
        greater_pos = env.trick_winner if env.trick_winner is not None else player
        try:
            idx = self._action.passive(
                action_list, hand_strings, rank_str,
                trick_type, trick_type,
                player, greater_pos,
                remaincards, self._remaining,
                self._pass_num, self._my_pass_num,
                None,
            )
            return idx if idx is not None else 0
        except Exception:
            return 0
