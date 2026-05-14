from __future__ import annotations

from guandan.cards import Card, ComboType, Rank, Suit
from guandan.combos import Combo
from guandan.guanzero.legal_utils import dedup_strategic, strategic_key


def _combo(ctype: ComboType, key: int, cards: list[Card], length: int = 0) -> Combo:
    return Combo(ctype, key, cards, length=length)


def test_dedup_strategic_collapses_suit_variants_for_ordinary_moves():
    first = _combo(
        ComboType.PAIR,
        Rank.ACE,
        [Card(Rank.ACE, Suit.SPADE, 0), Card(Rank.ACE, Suit.HEART, 0)],
    )
    second = _combo(
        ComboType.PAIR,
        Rank.ACE,
        [Card(Rank.ACE, Suit.DIAMOND, 0), Card(Rank.ACE, Suit.CLUB, 0)],
    )

    assert strategic_key(first) == strategic_key(second)
    assert dedup_strategic([first, second]) == [first]


def test_dedup_strategic_preserves_pass_and_suit_sensitive_bombs():
    pass_move = _combo(ComboType.PASS, 0, [])
    spade_sf = _combo(
        ComboType.STRAIGHT_FLUSH,
        Rank.SIX,
        [Card(rank, Suit.SPADE, 0) for rank in range(Rank.TWO, Rank.SEVEN)],
        length=5,
    )
    heart_sf = _combo(
        ComboType.STRAIGHT_FLUSH,
        Rank.SIX,
        [Card(rank, Suit.HEART, 0) for rank in range(Rank.TWO, Rank.SEVEN)],
        length=5,
    )
    joker_bomb = _combo(
        ComboType.BOMB_JOKER,
        99,
        [
            Card(Rank.BLACK_JOKER, 0, 0),
            Card(Rank.BLACK_JOKER, 0, 1),
            Card(Rank.RED_JOKER, 1, 0),
            Card(Rank.RED_JOKER, 1, 1),
        ],
    )

    out = dedup_strategic([pass_move, spade_sf, heart_sf, joker_bomb])
    assert out == [pass_move, spade_sf, heart_sf, joker_bomb]
    assert strategic_key(spade_sf) != strategic_key(heart_sf)
