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
tracks the same public state the competition client maintained, calls
Action.rule_parse(), and converts the selected actionList index back.
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

_RANK_IDX = {
    "A": 0, "2": 1, "3": 2, "4": 3, "5": 4, "6": 5, "7": 6,
    "8": 7, "9": 8, "T": 9, "J": 10, "Q": 11, "K": 12,
    "R": 13, "B": 13,
}


def _initial_history() -> dict[str, dict[str, object]]:
    return {str(pos): {"send": [], "remain": 27} for pos in range(4)}


def _initial_remain_cards() -> dict[str, list[int]]:
    return {
        "S": [2] * 13 + [2],
        "H": [2] * 13 + [2],
        "C": [2] * 13 + [0],
        "D": [2] * 13 + [0],
    }


def _env_to_comp_pos(pos: int) -> int:
    """Map engine seats to vendor seats where next player is +1 mod 4."""
    return (-pos) % 4


def _comp_to_env_pos(pos: int) -> int:
    return (-pos) % 4


class _CompetitionState:
    """Minimal port of lalala/state.py for play-stage decisions.

    The official client kept one State object per websocket seat. Eval reuses a
    single Agent object for both teammates, so this wrapper keeps one tracker per
    seat and replays env.move_history into each tracker before acting.
    """

    def __init__(self, my_pos: int):
        self.my_pos = my_pos
        self.reset()

    def reset(self) -> None:
        self.history = _initial_history()
        self.remain_cards = _initial_remain_cards()
        self.remain_cards_classbynum = [8] * 13 + [2, 2]
        self.tribute_result = None
        self.pass_num = 0
        self.my_pass_num = 0
        self._synced_history_len = 0

    def sync_from_env(self, env, level_rank: int) -> None:
        if len(env.move_history) < self._synced_history_len:
            self.reset()

        for cur_pos, combo in env.move_history[self._synced_history_len:]:
            self._notify_play(
                _env_to_comp_pos(cur_pos),
                combo_to_action_list(combo, level_rank),
            )

        self._synced_history_len = len(env.move_history)

    def _notify_play(self, cur_pos: int, cur_action: list) -> None:
        if cur_action[0] != "PASS":
            row = self.history[str(cur_pos)]
            sent = row["send"]
            assert isinstance(sent, list)
            for card in cur_action[2]:
                sent.append(card)
                row["remain"] = int(row["remain"]) - 1
                self.remain_cards[card[0]][_RANK_IDX[card[1]]] -= 1

        if cur_pos in (self.my_pos, (self.my_pos + 2) % 4):
            if cur_action[0] == "PASS":
                self.pass_num += 1
            else:
                self.pass_num = 0

        if cur_pos == self.my_pos:
            if cur_action[0] == "PASS":
                self.my_pass_num += 1
            else:
                self.my_pass_num = 0


class LalalaBot(Agent):
    """SEU 1st Prize — heuristic with next-player optimization."""

    label = "Lalala"
    description = "SEU 1st Prize · Li Jing (2020)."
    source = "SEU"
    color = "#e377c2"
    award = "1st Prize · Li Jing"
    sample_tag = 9

    def __init__(self, level_rank: int = Rank.TWO):
        self.level_rank = level_rank
        self._actions: dict[int, Action] = {}
        self._states: dict[int, _CompetitionState] = {}

    def act(self, env, player: int):
        legal = env.legal_moves(player)
        if len(legal) == 1:
            return legal[0]

        comp_player = _env_to_comp_pos(player)
        state = self._states.setdefault(player, _CompetitionState(comp_player))
        state.sync_from_env(env, self.level_rank)
        action = self._actions.setdefault(player, Action(f"lalala_bot_{player}"))

        rank_str = rank_to_string(self.level_rank)
        hand_strings = cards_to_strings(env.hands[player])

        # Build actionList in competition format: [['PASS','PASS',[]], ['Single','5',['S5']], ...]
        action_list = [["PASS", "PASS", []]]
        combo_map = [None]  # index 0 = PASS
        for combo in legal:
            if combo.type == ComboType.PASS:
                continue
            action_list.append(combo_to_action_list(combo, self.level_rank))
            combo_map.append(combo)

        msg = self._build_play_message(env, player, action_list, hand_strings, rank_str)
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

        # Clamp index
        if idx is None or idx < 0 or idx >= len(action_list):
            idx = 0

        if idx == 0:
            return find_pass(legal)
        return combo_map[idx] if idx < len(combo_map) else find_pass(legal)

    def _build_play_message(self, env, player: int, action_list: list, hand_strings: list[str], rank_str: str) -> dict:
        comp_player = _env_to_comp_pos(player)
        if env.current_trick is None:
            cur_pos = -1
            greater_pos = -1
            cur_action = ["PASS", "PASS", []]
            greater_action = ["PASS", "PASS", []]
        else:
            env_cur_pos, cur_combo = env.move_history[-1]
            cur_pos = _env_to_comp_pos(env_cur_pos)
            cur_action = combo_to_action_list(cur_combo, self.level_rank)
            greater_pos = (
                _env_to_comp_pos(env.trick_winner)
                if env.trick_winner is not None
                else comp_player
            )
            greater_action = combo_to_action_list(env.current_trick, self.level_rank)

        return {
            "type": "act",
            "stage": "play",
            "myPos": comp_player,
            "curPos": cur_pos,
            "curAction": cur_action,
            "greaterPos": greater_pos,
            "greaterAction": greater_action,
            "handCards": hand_strings,
            "curRank": rank_str,
            "selfRank": rank_str,
            "oppoRank": rank_str,
            "actionList": action_list,
            "publicInfo": [
                {"rest": len(env.hands[_comp_to_env_pos(pos)]), "playArea": None}
                for pos in range(4)
            ],
        }
