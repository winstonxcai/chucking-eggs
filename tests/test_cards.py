"""Tests for cards.py — Card model, constants, utilities."""

from guandan.cards import (
    Card,
    ComboType,
    Rank,
    Suit,
    is_wild,
    level_order_key,
    make_deck,
)


def test_deck_size():
    deck = make_deck()
    assert len(deck) == 108


def test_deck_composition():
    deck = make_deck()
    # 2 copies of 52 normal cards + 2 copies of 2 jokers = 108
    normal = [c for c in deck if c.rank <= Rank.ACE]
    jokers = [c for c in deck if c.rank >= Rank.BLACK_JOKER]
    assert len(normal) == 104  # 13 ranks × 4 suits × 2 decks
    assert len(jokers) == 4  # 2 BJ + 2 RJ


def test_joker_suits():
    """BJ has suit 0, RJ has suit 1."""
    deck = make_deck()
    bjs = [c for c in deck if c.rank == Rank.BLACK_JOKER]
    rjs = [c for c in deck if c.rank == Rank.RED_JOKER]
    assert all(c.suit == 0 for c in bjs)
    assert all(c.suit == 1 for c in rjs)


def test_deck_unique_cards():
    """Each (rank, suit, deck) triple should be unique."""
    deck = make_deck()
    identities = [(c.rank, c.suit, c.deck) for c in deck]
    assert len(identities) == len(set(identities))


def test_rank_ordering():
    """Natural rank ordering: 2 < 3 < ... < A < BJ < RJ."""
    assert Rank.TWO < Rank.THREE < Rank.ACE < Rank.BLACK_JOKER < Rank.RED_JOKER


def test_level_order_key_at_level_2():
    """At level 2, the 2s rank above A (key=15) in level order."""
    assert level_order_key(Rank.TWO, Rank.TWO) == 15
    assert level_order_key(Rank.ACE, Rank.TWO) == 14
    assert level_order_key(Rank.KING, Rank.TWO) == 13
    assert level_order_key(Rank.TWO, Rank.TWO) > level_order_key(Rank.ACE, Rank.TWO)


def test_level_order_key_at_level_7():
    """At level 7, the 7s rank above A in level order."""
    assert level_order_key(7, 7) == 15
    assert level_order_key(7, 7) > level_order_key(Rank.ACE, 7)
    # Non-level cards keep their natural rank
    assert level_order_key(Rank.THREE, 7) == 3


def test_is_wild_at_level_2():
    wild1 = Card(Rank.TWO, Suit.HEART, 0)
    wild2 = Card(Rank.TWO, Suit.HEART, 1)
    not_wild = Card(Rank.TWO, Suit.SPADE, 0)
    assert is_wild(wild1, Rank.TWO)
    assert is_wild(wild2, Rank.TWO)
    assert not is_wild(not_wild, Rank.TWO)


def test_is_wild_jokers_not_wild():
    """Jokers are never wild."""
    bj = Card(Rank.BLACK_JOKER, 0, 0)
    rj = Card(Rank.RED_JOKER, 1, 0)
    assert not is_wild(bj, Rank.TWO)
    assert not is_wild(rj, Rank.TWO)


def test_combo_type_bomb_ordering():
    """Bomb enum values should be strictly ordered."""
    assert (
        ComboType.BOMB_4
        < ComboType.BOMB_5
        < ComboType.STRAIGHT_FLUSH
        < ComboType.BOMB_6
        < ComboType.BOMB_7
        < ComboType.BOMB_8
        < ComboType.BOMB_9
        < ComboType.BOMB_10
        < ComboType.BOMB_JOKER
    )
