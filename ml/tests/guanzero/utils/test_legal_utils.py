from __future__ import annotations

from guandan.cards import Card, ComboType, Rank, Suit
from guandan.combos import Combo
from guandan.guanzero.utils.legal_utils import dedup_strategic


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

    out = dedup_strategic([first, second])
    assert len(out) == 1
    assert out[0].type == first.type and out[0].key == first.key


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
    assert len(out) == 4
    assert [c.type for c in out] == [pass_move.type, spade_sf.type, heart_sf.type, joker_bomb.type]
