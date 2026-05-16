"""NoAIBot — faithful port of Fudan University's 2nd-place competition entry.

Source: 1st NJUPT Guan Dan AI Competition (2020)
Team: 不会AI怎么办 ("What if you can't do AI")
Author: Chen Yuguan (陈羽观), Fudan University
Prize: 2nd Place

Architecture: value-based hand decomposition with Revised Value (RV) system.
The bot recursively evaluates all possible hand decompositions to score each
play by (ActionValue + ContextBonus) + RestHandValue.
"""

from __future__ import annotations

from ..cards import ComboType, Rank
from ._vendor.adapter import (
    card_to_string,
    cards_to_strings,
    combo_to_action_dict,
    combo_to_action_list,
    find_combo_by_cards,
    find_pass,
    rank_to_string,
)
from ._vendor.noai.play_card import PlayCard
from ._vendor.noai.strategy import Strategy
from .base import Agent


class NoAIBot(Agent):
    """Fudan 2nd Prize — value-based heuristic with RV system."""

    label = "NoAI"
    description = "Fudan 2nd Prize · Chen Yuguan."
    source = "Fudan"
    award = "2nd Prize · Chen Yuguan"

    def __init__(self, level_rank: int = Rank.TWO):
        self.level_rank = level_rank
        self._strategy = Strategy()
        self._play_card = PlayCard()
        self._last_history_len = -1
        self._synced_up_to = 0

    def act(self, env, player: int):
        legal = env.legal_moves(player)
        if len(legal) == 1:
            return legal[0]

        cur_rank_str = rank_to_string(self.level_rank)

        # Detect new game → reset strategy
        history_len = len(env.move_history)
        if history_len < self._last_history_len or self._last_history_len == -1:
            hand_strings = cards_to_strings(env.hands[player])
            self._strategy.SetBeginning(player, hand_strings)
            self._strategy.UpdateCurRank(cur_rank_str)
            self._synced_up_to = 0
        self._last_history_len = history_len

        # Sync strategy state from move_history (new moves since last call)
        self._sync_state(env, player)

        # Convert hand and legal moves to competition format
        hand_strings = cards_to_strings(env.hands[player])
        full_action_list = []
        for combo in legal:
            if combo.type != ComboType.PASS:
                full_action_list.append(combo_to_action_list(combo, self.level_rank))

        # Decide
        if env.current_trick is None:
            result = self._play_card.FreePlay(
                hand_strings, cur_rank_str, self._strategy, full_action_list,
            )
        else:
            former = combo_to_action_dict(env.current_trick, self.level_rank)
            result = self._play_card.RestrictedPlay(
                hand_strings, former, cur_rank_str, self._strategy, full_action_list,
            )

        # Map result back to Combo
        if not result or result.get("type") == "PASS":
            return find_pass(legal)
        return find_combo_by_cards(result["action"], legal, self.level_rank)

    def _sync_state(self, env, player: int):
        """Replay new moves from env.move_history into strategy state."""
        cur_rank_str = rank_to_string(self.level_rank)
        history = env.move_history

        # Track who currently has the greatest play (trick winner)
        for i in range(self._synced_up_to, len(history)):
            move_player, combo = history[i]

            # Convert to competition format
            if combo.type == ComboType.PASS:
                cur_action = ["PASS", "PASS", []]
            else:
                cur_action = combo_to_action_list(combo, self.level_rank)

            # Determine greaterPos/greaterAction
            if env.trick_winner is not None:
                greater_pos = env.trick_winner
            else:
                greater_pos = -1

            # For greaterAction, we need the trick winner's combo
            if env.current_trick is not None:
                greater_action = combo_to_action_list(env.current_trick, self.level_rank)
            elif greater_pos == -1:
                greater_action = None
            else:
                greater_action = cur_action

            if greater_action is not None:
                self._strategy.UpdatePlay(
                    move_player, cur_action, greater_pos, greater_action,
                )

        self._synced_up_to = len(history)
