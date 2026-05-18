"""LalalaBot — faithful port of Southeast University's 1st-place competition entry.

Source: 1st NJUPT Guan Dan AI Competition (2020)
Team: lalala
Author: Li Jing (李菁), Southeast University (东南大学)
Prize: 1st Place

Architecture: elaborate heuristic with next-player optimization, sophisticated
bomb selection (avoids breaking straights), endgame special-casing (one_hand),
and threshold-based leading strategy with rankone/ranktwo/rankthree/rankfour
sub-decisions.

The original code returns an index into the server's actionList. This wrapper
builds the same actionList format from our legal_moves() and converts back.
"""

from __future__ import annotations

from ..cards import ComboType, Rank
from ._vendor.adapter import (
    cards_to_strings,
    combo_to_action_list,
    find_pass,
    rank_to_string,
)
from ._vendor.lalala.action import Action
from .base import Agent


# Rank index for building remaincards structure (2-deck card count dict)
_RANK_IDX = {
    "A": 0, "2": 1, "3": 2, "4": 3, "5": 4, "6": 5, "7": 6,
    "8": 7, "9": 8, "T": 9, "J": 10, "Q": 11, "K": 12,
}


def _build_remaincards(hand_strings: list[str]) -> dict:
    """Build 2-deck card count dict, subtracting known hand cards.

    Structure: {'S': [count_A, ..., count_K, count_joker], ...}
    Index 13 = joker (Black joker under 'S', Red joker under 'H').
    """
    # 2 decks: 2 copies of each rank per suit
    remaincards = {
        "S": [2] * 13 + [2],  # index 13 = Black Joker
        "H": [2] * 13 + [2],  # index 13 = Red Joker
        "C": [2] * 13 + [0],  # no joker
        "D": [2] * 13 + [0],  # no joker
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


class LalalaBot(Agent):
    """SEU 1st Prize — heuristic with next-player optimization."""

    label = "Lalala"
    description = "SEU 1st Prize · Li Jing (2020)."
    source = "SEU"
    award = "1st Prize · Li Jing"
    sample_tag = 9

    def __init__(self, level_rank: int = Rank.TWO):
        self.level_rank = level_rank
        self._action = Action("lalala_bot")
        self._last_history_len = -1
        # Per-game state (mirrors competition client)
        self._remaining = {0: 27, 1: 27, 2: 27, 3: 27}
        self._pass_num = 0        # cumulative passes observed this game
        self._my_pass_num = 0

    def act(self, env, player: int):
        legal = env.legal_moves(player)
        if len(legal) == 1:
            return legal[0]

        # Detect new game → reset state
        history_len = len(env.move_history)
        if history_len < self._last_history_len or self._last_history_len == -1:
            self._remaining = {0: 27, 1: 27, 2: 27, 3: 27}
            self._pass_num = 0
            self._my_pass_num = 0
        self._last_history_len = history_len

        # Reset my_pass_num at the start of each new trick
        if env.current_trick is None:
            self._my_pass_num = 0

        # Sync remaining counts from env
        for p in range(4):
            self._remaining[p] = len(env.hands[p])

        rank_str = rank_to_string(self.level_rank)
        hand_strings = cards_to_strings(env.hands[player])
        remaincards = _build_remaincards(hand_strings)

        # Build actionList in competition format: [['PASS','PASS',[]], ['Single','5',['S5']], ...]
        action_list = [["PASS", "PASS", []]]
        combo_map = [None]  # index 0 = PASS
        for combo in legal:
            if combo.type == ComboType.PASS:
                continue
            action_list.append(combo_to_action_list(combo, self.level_rank))
            combo_map.append(combo)

        if env.current_trick is None:
            idx = self._lead(action_list, hand_strings, rank_str, player, remaincards)
        else:
            idx = self._follow(
                env, player, action_list, hand_strings, rank_str, remaincards,
            )

        # Clamp index
        if idx is None or idx < 0 or idx >= len(action_list):
            idx = 0

        if idx == 0:
            return find_pass(legal)
        return combo_map[idx] if idx < len(combo_map) else find_pass(legal)

    def _lead(self, action_list, hand_strings, rank_str, player, remaincards):
        """Leading play using Action.active()."""
        try:
            idx = self._action.active(
                action_list, hand_strings, rank_str,
                self._remaining, player, remaincards,
            )
            return idx if idx is not None else 0
        except Exception:
            return 0

    def _follow(self, env, player, action_list, hand_strings, rank_str, remaincards):
        """Following play using Action.passive() — the competition code path."""
        trick_type = combo_to_action_list(env.current_trick, self.level_rank)
        greater_pos = env.trick_winner if env.trick_winner is not None else player

        try:
            idx = self._action.passive(
                action_list, hand_strings, rank_str,
                trick_type, trick_type,   # curAction = greaterAction = current trick
                player, greater_pos,
                remaincards, self._remaining,
                self._pass_num,           # cumulative passes this game
                self._my_pass_num,
                None,  # remain_cards_classbynum — not used inside passive()
            )
            result = idx if idx is not None else 0
            if result == 0:
                self._pass_num += 1
                self._my_pass_num += 1
            return result
        except Exception:
            self._pass_num += 1
            self._my_pass_num += 1
            return 0
