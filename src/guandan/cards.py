from enum import IntEnum
from typing import NamedTuple


class Suit(IntEnum):
    SPADE = 0
    HEART = 1
    DIAMOND = 2
    CLUB = 3


class Rank(IntEnum):
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14
    # 15 is reserved for level_order_key
    BLACK_JOKER = 16
    RED_JOKER = 17


class Card(NamedTuple):
    rank: int
    suit: int   # 0-3 for normal cards, 0/1 for jokers
    deck: int   # 0 or 1 (which copy from the double deck)


class ComboType(IntEnum):
    PASS = 0
    # Ordinary (7 types)
    SINGLE = 1
    PAIR = 2
    TRIPLE = 3
    FULL_HOUSE = 4      # triple + pair (5 cards)
    STRAIGHT = 5        # exactly 5 consecutive, natural order, NOT all same suit
    TUBE = 6            # exactly 3 consecutive pairs (6 cards), natural order
    PLATE = 7           # exactly 2 consecutive triples (6 cards), natural order
    # Bombs (9 tiers, ordered low→high by enum value)
    BOMB_4 = 8          # quadruple
    BOMB_5 = 9          # quintuple
    STRAIGHT_FLUSH = 10 # 5 consecutive same suit (between 5-of-a-kind and 6-of-a-kind)
    BOMB_6 = 11         # sextuple
    BOMB_7 = 12         # septuple
    BOMB_8 = 13         # octuple
    BOMB_9 = 14         # nonuple (requires wilds)
    BOMB_10 = 15        # decuple (requires wilds)
    BOMB_JOKER = 16     # 2BJ + 2RJ — highest bomb


BOMB_TYPES = frozenset({
    ComboType.BOMB_4, ComboType.BOMB_5, ComboType.STRAIGHT_FLUSH,
    ComboType.BOMB_6, ComboType.BOMB_7, ComboType.BOMB_8,
    ComboType.BOMB_9, ComboType.BOMB_10, ComboType.BOMB_JOKER,
})

# Map N-of-a-kind size to bomb ComboType
BOMB_SIZE_TO_TYPE: dict[int, ComboType] = {
    4: ComboType.BOMB_4, 5: ComboType.BOMB_5,
    6: ComboType.BOMB_6, 7: ComboType.BOMB_7,
    8: ComboType.BOMB_8, 9: ComboType.BOMB_9,
    10: ComboType.BOMB_10,
}


def level_order_key(rank: int, level_rank: int) -> int:
    """Comparison key for LEVEL ORDER.

    Level cards rank above A(14), below BJ(16). Returns 15 for level cards.
    Used for: singles, pairs, triples, full houses, N-of-a-kind bombs.
    """
    if rank == level_rank:
        return 15  # above A(14), below BJ(16)
    return rank


def make_deck() -> list[Card]:
    """Create the 108-card double deck."""
    cards: list[Card] = []
    for deck_id in range(2):
        for rank in range(2, 15):  # 2 through A(=14)
            for suit in range(4):
                cards.append(Card(rank, suit, deck_id))
        cards.append(Card(Rank.BLACK_JOKER, 0, deck_id))
        cards.append(Card(Rank.RED_JOKER, 1, deck_id))
    return cards


def is_wild(card: Card, level_rank: int) -> bool:
    """Is this card a wild (♥ of level rank)?"""
    return card.rank == level_rank and card.suit == Suit.HEART
