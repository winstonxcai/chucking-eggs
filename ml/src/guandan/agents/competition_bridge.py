"""Helpers for adapting NJUPT competition bots to this engine.

The competition clients index seats so the next player is ``+1 mod 4``.  This
engine advances counterclockwise with ``-1 mod 4``.  Keep that convention swap
centralized so position-aware vendor heuristics see the same table geometry
they saw under the original websocket server.
"""

from __future__ import annotations

from ..cards import ComboType
from ._vendor.adapter import combo_to_action_list

_RANK_IDX = {
    "A": 0, "2": 1, "3": 2, "4": 3, "5": 4, "6": 5, "7": 6,
    "8": 7, "9": 8, "T": 9, "J": 10, "Q": 11, "K": 12,
    "R": 13, "B": 13,
}


def env_to_comp_pos(pos: int) -> int:
    return (-pos) % 4


def comp_to_env_pos(pos: int) -> int:
    return (-pos) % 4


def public_info_for_comp_positions(env) -> list[dict[str, object]]:
    return [
        {"rest": len(env.hands[comp_to_env_pos(pos)]), "playArea": None}
        for pos in range(4)
    ]


def build_action_list(legal: list, level_rank: int, *, transform=None) -> tuple[list, list]:
    action_list = [["PASS", "PASS", []]]
    combo_map = [None]
    for combo in legal:
        if combo.type == ComboType.PASS:
            continue
        entry = combo_to_action_list(combo, level_rank)
        action_list.append(transform(entry) if transform is not None else entry)
        combo_map.append(combo)
    return action_list, combo_map


def current_comp_context(env, level_rank: int) -> tuple[int, list, int, list]:
    if env.current_trick is None:
        pass_action = ["PASS", "PASS", []]
        return -1, pass_action, -1, pass_action

    env_cur_pos, cur_combo = env.move_history[-1]
    cur_pos = env_to_comp_pos(env_cur_pos)
    cur_action = combo_to_action_list(cur_combo, level_rank)
    greater_pos = env_to_comp_pos(env.trick_winner) if env.trick_winner is not None else -1
    greater_action = combo_to_action_list(env.current_trick, level_rank)
    return cur_pos, cur_action, greater_pos, greater_action


def build_play_message(
    env,
    player: int,
    *,
    action_list: list,
    hand_strings: list[str],
    rank_str: str,
    level_rank: int,
) -> dict[str, object]:
    cur_pos, cur_action, greater_pos, greater_action = current_comp_context(
        env,
        level_rank,
    )
    return {
        "type": "act",
        "stage": "play",
        "myPos": env_to_comp_pos(player),
        "curPos": cur_pos,
        "curAction": cur_action,
        "greaterPos": greater_pos,
        "greaterAction": greater_action,
        "handCards": hand_strings,
        "curRank": rank_str,
        "selfRank": rank_str,
        "oppoRank": rank_str,
        "actionList": action_list,
        "indexRange": len(action_list) - 1,
        "publicInfo": public_info_for_comp_positions(env),
    }


def initial_history() -> dict[str, dict[str, object]]:
    return {str(pos): {"send": [], "remain": 27} for pos in range(4)}


def initial_remain_cards() -> dict[str, list[int]]:
    return {
        "S": [2] * 13 + [2],
        "H": [2] * 13 + [2],
        "C": [2] * 13 + [0],
        "D": [2] * 13 + [0],
    }


class CompetitionPlayState:
    """Play-stage subset of the original competition ``State`` class."""

    def __init__(self, my_pos: int):
        self.my_pos = my_pos
        self.reset()

    def reset(self) -> None:
        self.history = initial_history()
        self.remain_cards = initial_remain_cards()
        self.remain_cards_classbynum = [8] * 13 + [2, 2]
        self.tribute_result = None
        self.pass_num = 0
        self.my_pass_num = 0
        self._synced_history_len = 0

    def sync_from_env(self, env, level_rank: int) -> None:
        if len(env.move_history) < self._synced_history_len:
            self.reset()

        for cur_pos, combo in env.move_history[self._synced_history_len:]:
            self.notify_play(
                env_to_comp_pos(cur_pos),
                combo_to_action_list(combo, level_rank),
            )
        self._synced_history_len = len(env.move_history)

    def notify_play(self, cur_pos: int, cur_action: list) -> None:
        if cur_action[0] != "PASS":
            row = self.history[str(cur_pos)]
            sent = row["send"]
            assert isinstance(sent, list)
            for card in cur_action[2]:
                sent.append(card)
                row["remain"] = int(row["remain"]) - 1
                self.remain_cards[card[0]][_RANK_IDX[card[1]]] -= 1

        if cur_pos in (self.my_pos, (self.my_pos + 2) % 4):
            self.pass_num = self.pass_num + 1 if cur_action[0] == "PASS" else 0

        if cur_pos == self.my_pos:
            self.my_pass_num = self.my_pass_num + 1 if cur_action[0] == "PASS" else 0
