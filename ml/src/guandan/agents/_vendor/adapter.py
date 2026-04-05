"""Adapter between our Card/Combo types and the competition string format.

Competition format:
  Cards: 'S5' (Spade 5), 'HR' (Heart Red Joker), 'SB' (Spade Black Joker)
  Actions: ['Single', '5', ['S5']]  or  {'type': 'Pair', 'rank': '5', 'action': ['S5', 'C5']}
"""

from __future__ import annotations

from ...cards import Card, ComboType, Rank, Suit

# ── Card conversion ──────────────────────────────────────

_SUIT_TO_STR = {Suit.SPADE: "S", Suit.HEART: "H", Suit.CLUB: "C", Suit.DIAMOND: "D"}
_STR_TO_SUIT = {v: k for k, v in _SUIT_TO_STR.items()}

_RANK_TO_STR = {
    2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
    10: "T", 11: "J", 12: "Q", 13: "K", 14: "A",
    Rank.BLACK_JOKER: "B", Rank.RED_JOKER: "R",
}
_STR_TO_RANK = {v: k for k, v in _RANK_TO_STR.items()}


def card_to_string(card: Card) -> str:
    """Card → competition string. Card(5, 0, 0) → 'S5'."""
    return _SUIT_TO_STR[card.suit] + _RANK_TO_STR[card.rank]


def cards_to_strings(hand, level_rank: int | None = None) -> list[str]:
    """set[Card] or iterable of Card → sorted list of competition strings."""
    return sorted(card_to_string(c) for c in hand)


def string_to_card_key(s: str) -> tuple[int, int]:
    """'S5' → (rank=5, suit=0). For matching purposes (ignores deck)."""
    return (_STR_TO_RANK[s[1:]], _STR_TO_SUIT[s[0]])


def rank_to_string(rank: int) -> str:
    """Rank int → competition rank string."""
    return _RANK_TO_STR[rank]


def string_to_rank(s: str) -> int:
    """Competition rank string → Rank int."""
    return _STR_TO_RANK[s]


# ── Combo type conversion ────────────────────────────────

_COMBO_TYPE_TO_STR = {
    ComboType.PASS: "PASS",
    ComboType.SINGLE: "Single",
    ComboType.PAIR: "Pair",
    ComboType.TRIPLE: "Trips",
    ComboType.FULL_HOUSE: "ThreeWithTwo",
    ComboType.STRAIGHT: "Straight",
    ComboType.TUBE: "ThreePair",       # 3 consecutive pairs
    ComboType.PLATE: "TwoTrips",       # 2 consecutive triples
    ComboType.BOMB_4: "Bomb",
    ComboType.BOMB_5: "Bomb",
    ComboType.BOMB_6: "Bomb",
    ComboType.BOMB_7: "Bomb",
    ComboType.BOMB_8: "Bomb",
    ComboType.BOMB_9: "Bomb",
    ComboType.BOMB_10: "Bomb",
    ComboType.STRAIGHT_FLUSH: "StraightFlush",
    ComboType.BOMB_JOKER: "Bomb",
}

_STR_TO_COMBO_TYPES = {
    "PASS": {ComboType.PASS},
    "Single": {ComboType.SINGLE},
    "Pair": {ComboType.PAIR},
    "Trips": {ComboType.TRIPLE},
    "ThreeWithTwo": {ComboType.FULL_HOUSE},
    "Straight": {ComboType.STRAIGHT},
    "ThreePair": {ComboType.TUBE},
    "TwoTrips": {ComboType.PLATE},
    "Bomb": {ComboType.BOMB_4, ComboType.BOMB_5, ComboType.BOMB_6,
             ComboType.BOMB_7, ComboType.BOMB_8, ComboType.BOMB_9,
             ComboType.BOMB_10, ComboType.BOMB_JOKER},
    "StraightFlush": {ComboType.STRAIGHT_FLUSH},
}


def combo_type_to_string(combo) -> str:
    return _COMBO_TYPE_TO_STR.get(combo.type, "PASS")


# ── Combo ↔ action dict conversion ──────────────────────

def combo_to_action_dict(combo, level_rank: int) -> dict:
    """Combo → competition action dict {type, rank, action}."""
    type_str = combo_type_to_string(combo)
    rank_str = _RANK_TO_STR.get(combo.key, "2")
    card_strs = [card_to_string(c) for c in combo.cards]
    return {"type": type_str, "rank": rank_str, "action": card_strs}


def combo_to_action_list(combo, level_rank: int) -> list:
    """Combo → competition action list [type, rank, [cards]]."""
    type_str = combo_type_to_string(combo)
    rank_str = _RANK_TO_STR.get(combo.key, "2")
    card_strs = [card_to_string(c) for c in combo.cards]
    return [type_str, rank_str, card_strs]


def find_combo_by_cards(card_strings: list[str], legal_moves: list, level_rank: int):
    """Find the Combo in legal_moves whose cards match the given strings.

    Matches by multiset of (rank, suit) pairs since deck index may differ.
    """
    target = sorted((string_to_card_key(s)) for s in card_strings)

    for combo in legal_moves:
        if combo.type == ComboType.PASS:
            continue
        combo_keys = sorted((c.rank, c.suit) for c in combo.cards)
        if combo_keys == target:
            return combo

    # Fallback: match by type and rank only (less strict)
    for combo in legal_moves:
        if combo.type == ComboType.PASS:
            continue
        combo_strs = sorted(card_to_string(c) for c in combo.cards)
        if sorted(card_strings) == combo_strs:
            return combo

    # Last resort: return first non-pass move
    for combo in legal_moves:
        if combo.type != ComboType.PASS:
            return combo
    return legal_moves[0]


def find_pass(legal_moves: list):
    """Find PASS in legal moves."""
    for combo in legal_moves:
        if combo.type == ComboType.PASS:
            return combo
    return legal_moves[0]
