"""WjsdBot — faithful port of Shenyang Aerospace's 3rd-place competition entry.

Source: 1st NJUPT Guan Dan AI Competition (2020)
Team: 我就是掼蛋
Author: Liang Kai (梁凯), Shenyang Aerospace University (沈阳航空航天大学)
Prize: 3rd Place

Architecture: numpy 4×16 card matrix with per-combo-type analysis methods
(tonghuashun, zhadan, shunzi, gangban, sandui, trips, duizi, danpai). Uses
team-position awareness (cha = abs(greaterPos - myPos)). Originally a
WebSocket client; self.send()+return replaced with return act_index.
"""

from __future__ import annotations

from ..cards import Rank
from ._vendor.adapter import (
    cards_to_strings,
    find_pass,
    rank_to_string,
)
from ._vendor.wjsd.client import WjsdClient
from .base import Agent
from .competition_bridge import build_action_list, build_play_message

# Vendor scans its 4x16 shoupaijuzhen matrix with row order S,H,C,D
# (see hua_dict in client.py). Match its convention by ordering each
# action's card list (rank desc, then S<H<C<D) so positional checks like
# `actionlist[i][2][0][0] == hua_dict[Pair_j[j]]` resolve.
_SUIT_RANK = {"S": 0, "H": 1, "C": 2, "D": 3}


def _wjsd_card_key(card_str: str) -> tuple:
    # B = Black Joker, R = Red Joker — fixed priorities (B before R)
    if len(card_str) >= 2 and card_str[1] in ("B", "R"):
        return (100 if card_str[1] == "R" else 99, _SUIT_RANK.get(card_str[0], 4))
    rank_chars = {"T": 10, "J": 11, "Q": 12, "K": 13, "A": 14}
    rank = rank_chars.get(card_str[1:], None)
    if rank is None:
        try:
            rank = int(card_str[1:])
        except ValueError:
            rank = 0
    return (rank, _SUIT_RANK.get(card_str[0], 4))


def _sort_action_cards(entry: list) -> list:
    if not isinstance(entry[2], list) or not entry[2]:
        return entry
    type_str = entry[0]
    cards = entry[2]
    # Group cards by rank, then arrange groups so that the vendor's positional
    # checks line up with what its handsort matrix produces.
    by_rank: dict[int, list[str]] = {}
    order: list[int] = []
    for c in cards:
        r = _wjsd_card_key(c)[0]
        if r not in by_rank:
            by_rank[r] = []
            order.append(r)
        by_rank[r].append(c)
    for r in by_rank:
        by_rank[r].sort(key=_wjsd_card_key)

    if type_str == "ThreeWithTwo":
        # triple first, pair last — by group size descending, ties by rank asc
        order.sort(key=lambda r: (-len(by_rank[r]), r))
    else:
        # everything else: by rank ascending
        order.sort()

    entry[2] = [c for r in order for c in by_rank[r]]
    return entry


class WjsdBot(Agent):
    """Shenyang 3rd Prize — numpy card matrix analysis with team-aware play."""

    label = "Wjsd"
    description = "SAU 3rd Prize · 2020 NJUPT entry."
    source = "SAU"
    color = "#aec7e8"
    award = "3rd Prize"
    sample_tag = 13

    def __init__(self, level_rank: int = Rank.TWO):
        self.level_rank = level_rank
        self._client = WjsdClient()
        self._last_history_len = -1

    def act(self, env, player: int):
        legal = env.legal_moves(player)
        if len(legal) == 1:
            return legal[0]

        history_len = len(env.move_history)
        if history_len < self._last_history_len or self._last_history_len == -1:
            self._client = WjsdClient()
        self._last_history_len = history_len

        rank_str = rank_to_string(self.level_rank)
        hand_strings = cards_to_strings(env.hands[player])
        action_list, combo_map = build_action_list(
            legal,
            self.level_rank,
            transform=_sort_action_cards,
        )
        msg = build_play_message(
            env,
            player,
            action_list=action_list,
            hand_strings=hand_strings,
            rank_str=rank_str,
            level_rank=self.level_rank,
        )

        try:
            idx = self._client.received_message(msg)
            if idx is None or idx < 0 or idx >= len(action_list):
                idx = 0
        except Exception:
            idx = 0

        if idx == 0:
            return find_pass(legal)
        return combo_map[idx] if idx < len(combo_map) else find_pass(legal)
