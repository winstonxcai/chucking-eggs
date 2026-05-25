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
from .competition_bridge import env_to_comp_pos


class NoAIBot(Agent):
    """Fudan 2nd Prize — value-based heuristic with RV system."""

    label = "NoAI"
    description = "Fudan 2nd Prize · Chen Yuguan."
    source = "Fudan"
    color = "#393b79"
    award = "2nd Prize · Chen Yuguan"
    sample_tag = 8

    def __init__(self, level_rank: int = Rank.TWO):
        self.level_rank = level_rank
        self._play_card = PlayCard()
        self._strategies: dict[int, Strategy] = {}
        self._synced_up_to: dict[int, int] = {}

    def act(self, env, player: int):
        legal = env.legal_moves(player)
        if len(legal) == 1:
            return legal[0]

        cur_rank_str = rank_to_string(self.level_rank)
        comp_player = env_to_comp_pos(player)
        hand_strings = cards_to_strings(env.hands[player])

        history_len = len(env.move_history)
        strategy = self._strategies.get(player)
        if (
            strategy is None
            or history_len < self._synced_up_to.get(player, 0)
        ):
            strategy = Strategy()
            strategy.SetBeginning(comp_player, hand_strings)
            strategy.UpdateCurRank(cur_rank_str)
            self._strategies[player] = strategy
            self._synced_up_to[player] = 0

        self._sync_state(env, player, strategy)

        # Convert hand and legal moves to competition format
        full_action_list = []
        for combo in legal:
            if combo.type != ComboType.PASS:
                full_action_list.append(combo_to_action_list(combo, self.level_rank))

        # Decide
        if env.current_trick is None:
            result = self._play_card.FreePlay(
                hand_strings, cur_rank_str, strategy, full_action_list,
            )
        else:
            former = combo_to_action_dict(env.current_trick, self.level_rank)
            result = self._play_card.RestrictedPlay(
                hand_strings, former, cur_rank_str, strategy, full_action_list,
            )

        # Map result back to Combo
        if not result or result.get("type") == "PASS":
            return find_pass(legal)
        return find_combo_by_cards(result["action"], legal, self.level_rank)

    def _sync_state(self, env, player: int, strategy: Strategy):
        """Replay new moves from env.move_history into strategy state."""
        history = env.move_history
        synced_up_to = self._synced_up_to.get(player, 0)

        greater_pos = getattr(strategy, "greaterPos", -1)
        greater_action = getattr(strategy, "greaterAction", None)
        if greater_action is None:
            greater_action = ["PASS", "PASS", []]

        for i in range(synced_up_to, len(history)):
            move_player, combo = history[i]
            comp_move_player = env_to_comp_pos(move_player)

            if combo.type == ComboType.PASS:
                cur_action = ["PASS", "PASS", []]
            else:
                cur_action = combo_to_action_list(combo, self.level_rank)
                greater_pos = comp_move_player
                greater_action = cur_action

            strategy.UpdatePlay(
                comp_move_player,
                cur_action,
                greater_pos,
                greater_action,
            )

        self._synced_up_to[player] = len(history)
