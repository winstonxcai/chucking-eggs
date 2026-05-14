"""Tests for combos.py — Combo class, beats(), movegen."""

import random

from guandan.cards import (
    BOMB_TYPES,
    Card,
    ComboType,
    Rank,
    Suit,
    is_wild,
    level_order_key,
    make_deck,
)
from guandan.combos import Combo, generate_all_leads, generate_responses


LEVEL_RANK = Rank.TWO  # MVP level


def _hand_from_cards(cards: list[Card]) -> set[Card]:
    return set(cards)


# ─── beats() tests ──────────────────────────────────────


def test_bomb_beats_nonbomb():
    b4 = Combo(ComboType.BOMB_4, Rank.FIVE, [])
    single = Combo(ComboType.SINGLE, Rank.ACE, [])
    assert b4.beats(single, LEVEL_RANK)
    assert not single.beats(b4, LEVEL_RANK)


def test_bomb_hierarchy():
    """4 < 5 < SF < 6 < 7 < 8 < 9 < 10 < JokerBomb."""
    b4 = Combo(ComboType.BOMB_4, Rank.FIVE, [])
    b5 = Combo(ComboType.BOMB_5, Rank.THREE, [])
    sf = Combo(ComboType.STRAIGHT_FLUSH, Rank.FIVE, [])
    b6 = Combo(ComboType.BOMB_6, Rank.THREE, [])
    bj = Combo(ComboType.BOMB_JOKER, 99, [])

    assert b5.beats(b4, LEVEL_RANK)
    assert sf.beats(b5, LEVEL_RANK)
    assert b6.beats(sf, LEVEL_RANK)
    assert bj.beats(b6, LEVEL_RANK)
    assert not b4.beats(b5, LEVEL_RANK)


def test_same_tier_bomb_level_order():
    """Within same N-of-a-kind tier, level cards rank highest."""
    b4_level = Combo(ComboType.BOMB_4, Rank.TWO, [])  # level card at level 2
    b4_ace = Combo(ComboType.BOMB_4, Rank.ACE, [])
    assert b4_level.beats(b4_ace, LEVEL_RANK)  # level card > ace in level order


def test_straight_flush_natural_order():
    """Straight flushes ranked by top card in natural order (no level promotion)."""
    sf_low = Combo(ComboType.STRAIGHT_FLUSH, 5, [])  # A-2-3-4-5
    sf_high = Combo(ComboType.STRAIGHT_FLUSH, 14, [])  # 10-J-Q-K-A
    assert sf_high.beats(sf_low, LEVEL_RANK)
    assert not sf_low.beats(sf_high, LEVEL_RANK)


def test_ordinary_must_match_type():
    """Non-bomb combos of different types can't beat each other."""
    single = Combo(ComboType.SINGLE, Rank.ACE, [])
    pair = Combo(ComboType.PAIR, Rank.THREE, [])
    assert not single.beats(pair, LEVEL_RANK)
    assert not pair.beats(single, LEVEL_RANK)


def test_single_level_order():
    """Singles compared in level order."""
    s_level = Combo(ComboType.SINGLE, Rank.TWO, [])  # level card at level 2
    s_ace = Combo(ComboType.SINGLE, Rank.ACE, [])
    assert s_level.beats(s_ace, LEVEL_RANK)


def test_straight_natural_order():
    """Straights compared by top card in natural order."""
    s_low = Combo(ComboType.STRAIGHT, 5, [])  # A-2-3-4-5
    s_high = Combo(ComboType.STRAIGHT, 14, [])  # 10-J-Q-K-A
    assert s_high.beats(s_low, LEVEL_RANK)


def test_beats_free_lead():
    """Any combo beats None (free lead)."""
    single = Combo(ComboType.SINGLE, Rank.THREE, [])
    assert single.beats(None, LEVEL_RANK)


# ─── Movegen tests ──────────────────────────────────────


def test_singles_generated():
    """Each distinct card should generate a single."""
    cards = [
        Card(Rank.THREE, Suit.SPADE, 0),
        Card(Rank.FIVE, Suit.HEART, 0),
        Card(Rank.ACE, Suit.DIAMOND, 0),
    ]
    hand = _hand_from_cards(cards)
    combos = generate_all_leads(hand, LEVEL_RANK)
    singles = [c for c in combos if c.type == ComboType.SINGLE]
    # Should have at least 3 singles (one per card)
    assert len(singles) >= 3


def test_natural_pair():
    """Two cards of same rank form a pair."""
    cards = [
        Card(Rank.FIVE, Suit.SPADE, 0),
        Card(Rank.FIVE, Suit.HEART, 0),
    ]
    hand = _hand_from_cards(cards)
    combos = generate_all_leads(hand, LEVEL_RANK)
    pairs = [c for c in combos if c.type == ComboType.PAIR]
    assert any(p.key == Rank.FIVE for p in pairs)


def test_bj_rj_not_pair():
    """BJ + RJ is NOT a valid pair."""
    cards = [
        Card(Rank.BLACK_JOKER, 0, 0),
        Card(Rank.RED_JOKER, 1, 0),
    ]
    hand = _hand_from_cards(cards)
    combos = generate_all_leads(hand, LEVEL_RANK)
    pairs = [c for c in combos if c.type == ComboType.PAIR]
    # No pair should contain both BJ and RJ
    for p in pairs:
        ranks = {c.rank for c in p.cards}
        assert not (Rank.BLACK_JOKER in ranks and Rank.RED_JOKER in ranks)


def test_bj_bj_is_pair():
    """BJ + BJ from double deck IS a valid pair."""
    cards = [
        Card(Rank.BLACK_JOKER, 0, 0),
        Card(Rank.BLACK_JOKER, 0, 1),
    ]
    hand = _hand_from_cards(cards)
    combos = generate_all_leads(hand, LEVEL_RANK)
    pairs = [c for c in combos if c.type == ComboType.PAIR]
    assert any(p.key == Rank.BLACK_JOKER for p in pairs)


def test_wild_pair():
    """Wild + natural card forms a pair."""
    cards = [
        Card(Rank.TWO, Suit.HEART, 0),  # wild at level 2
        Card(Rank.FIVE, Suit.SPADE, 0),
    ]
    hand = _hand_from_cards(cards)
    combos = generate_all_leads(hand, LEVEL_RANK)
    pairs = [c for c in combos if c.type == ComboType.PAIR]
    assert any(p.key == Rank.FIVE for p in pairs)


def test_wild_pair_of_wilds():
    """Two wilds together = pair of level cards."""
    cards = [
        Card(Rank.TWO, Suit.HEART, 0),
        Card(Rank.TWO, Suit.HEART, 1),
    ]
    hand = _hand_from_cards(cards)
    combos = generate_all_leads(hand, LEVEL_RANK)
    pairs = [c for c in combos if c.type == ComboType.PAIR]
    assert any(p.key == LEVEL_RANK and p.wild_count == 2 for p in pairs)


def test_quad_bomb():
    """4 cards of same rank = BOMB_4."""
    cards = [
        Card(Rank.SEVEN, Suit.SPADE, 0),
        Card(Rank.SEVEN, Suit.HEART, 0),
        Card(Rank.SEVEN, Suit.DIAMOND, 0),
        Card(Rank.SEVEN, Suit.CLUB, 0),
    ]
    hand = _hand_from_cards(cards)
    combos = generate_all_leads(hand, LEVEL_RANK)
    bombs = [c for c in combos if c.type == ComboType.BOMB_4]
    assert any(b.key == Rank.SEVEN for b in bombs)


def test_four_joker_bomb():
    """2 BJ + 2 RJ = four-joker bomb."""
    cards = [
        Card(Rank.BLACK_JOKER, 0, 0),
        Card(Rank.BLACK_JOKER, 0, 1),
        Card(Rank.RED_JOKER, 1, 0),
        Card(Rank.RED_JOKER, 1, 1),
    ]
    hand = _hand_from_cards(cards)
    combos = generate_all_leads(hand, LEVEL_RANK)
    joker_bombs = [c for c in combos if c.type == ComboType.BOMB_JOKER]
    assert len(joker_bombs) == 1


def test_ace_low_straight():
    """A-2-3-4-5 is a valid straight with key=5."""
    cards = [
        Card(Rank.ACE, Suit.SPADE, 0),
        Card(Rank.TWO, Suit.DIAMOND, 0),
        Card(Rank.THREE, Suit.CLUB, 0),
        Card(Rank.FOUR, Suit.HEART, 0),
        Card(Rank.FIVE, Suit.SPADE, 0),
    ]
    hand = _hand_from_cards(cards)
    combos = generate_all_leads(hand, LEVEL_RANK)
    straights = [c for c in combos if c.type == ComboType.STRAIGHT]
    assert any(s.key == 5 for s in straights), f"Expected ace-low straight, got: {straights}"


def test_ace_high_straight():
    """10-J-Q-K-A is a valid straight with key=14."""
    cards = [
        Card(Rank.TEN, Suit.SPADE, 0),
        Card(Rank.JACK, Suit.HEART, 0),
        Card(Rank.QUEEN, Suit.DIAMOND, 0),
        Card(Rank.KING, Suit.CLUB, 0),
        Card(Rank.ACE, Suit.SPADE, 0),
    ]
    hand = _hand_from_cards(cards)
    combos = generate_all_leads(hand, LEVEL_RANK)
    straights = [c for c in combos if c.type == ComboType.STRAIGHT]
    assert any(s.key == 14 for s in straights)


def test_no_wrapping_straight():
    """K-A-2-3-4 is NOT a valid straight."""
    cards = [
        Card(Rank.KING, Suit.SPADE, 0),
        Card(Rank.ACE, Suit.HEART, 0),
        Card(Rank.TWO, Suit.DIAMOND, 0),
        Card(Rank.THREE, Suit.CLUB, 0),
        Card(Rank.FOUR, Suit.SPADE, 0),
    ]
    hand = _hand_from_cards(cards)
    combos = generate_all_leads(hand, LEVEL_RANK)
    straights = [c for c in combos if c.type == ComboType.STRAIGHT]
    # Should not have a straight spanning K-A-2-3-4
    for s in straights:
        card_ranks = sorted(c.rank for c in s.cards if not is_wild(c, LEVEL_RANK))
        # No straight should contain both K(13) and 2/3/4
        assert not (Rank.KING in card_ranks and Rank.TWO in card_ranks), \
            f"Wrapping straight found: {card_ranks}"


def test_straight_vs_straight_flush():
    """5 consecutive same-suit = straight flush BOMB, not ordinary straight."""
    cards = [
        Card(Rank.FIVE, Suit.SPADE, 0),
        Card(Rank.SIX, Suit.SPADE, 0),
        Card(Rank.SEVEN, Suit.SPADE, 0),
        Card(Rank.EIGHT, Suit.SPADE, 0),
        Card(Rank.NINE, Suit.SPADE, 0),
    ]
    hand = _hand_from_cards(cards)
    combos = generate_all_leads(hand, LEVEL_RANK)
    straights = [c for c in combos if c.type == ComboType.STRAIGHT]
    sfs = [c for c in combos if c.type == ComboType.STRAIGHT_FLUSH]
    # Should be a straight flush, NOT an ordinary straight
    assert len(sfs) >= 1, "Expected straight flush"
    # The same cards should NOT also appear as an ordinary straight
    assert len(straights) == 0, "All-same-suit should NOT be ordinary straight"


def test_tube_basic():
    """3 consecutive pairs = tube."""
    cards = [
        Card(Rank.FIVE, Suit.SPADE, 0),
        Card(Rank.FIVE, Suit.HEART, 0),
        Card(Rank.SIX, Suit.SPADE, 0),
        Card(Rank.SIX, Suit.HEART, 0),
        Card(Rank.SEVEN, Suit.SPADE, 0),
        Card(Rank.SEVEN, Suit.HEART, 0),
    ]
    hand = _hand_from_cards(cards)
    combos = generate_all_leads(hand, LEVEL_RANK)
    tubes = [c for c in combos if c.type == ComboType.TUBE]
    assert any(t.key == 7 for t in tubes), f"Expected tube with key=7, got: {tubes}"


def test_plate_basic():
    """2 consecutive triples = plate."""
    cards = [
        Card(Rank.FIVE, Suit.SPADE, 0),
        Card(Rank.FIVE, Suit.HEART, 0),
        Card(Rank.FIVE, Suit.DIAMOND, 0),
        Card(Rank.SIX, Suit.SPADE, 0),
        Card(Rank.SIX, Suit.HEART, 0),
        Card(Rank.SIX, Suit.DIAMOND, 0),
    ]
    hand = _hand_from_cards(cards)
    combos = generate_all_leads(hand, LEVEL_RANK)
    plates = [c for c in combos if c.type == ComboType.PLATE]
    assert any(p.key == 6 for p in plates), f"Expected plate with key=6, got: {plates}"


def test_full_house_basic():
    """Triple + pair = full house."""
    cards = [
        Card(Rank.FIVE, Suit.SPADE, 0),
        Card(Rank.FIVE, Suit.HEART, 0),
        Card(Rank.FIVE, Suit.DIAMOND, 0),
        Card(Rank.NINE, Suit.SPADE, 0),
        Card(Rank.NINE, Suit.HEART, 0),
    ]
    hand = _hand_from_cards(cards)
    combos = generate_all_leads(hand, LEVEL_RANK)
    fhs = [c for c in combos if c.type == ComboType.FULL_HOUSE]
    assert any(f.key == Rank.FIVE for f in fhs)


def test_response_includes_pass():
    """Response always includes PASS."""
    hand = _hand_from_cards([Card(Rank.THREE, Suit.SPADE, 0)])
    trick = Combo(ComboType.SINGLE, Rank.ACE, [])
    responses = generate_responses(hand, LEVEL_RANK, trick)
    assert any(r.type == ComboType.PASS for r in responses)


def test_wild_in_bomb():
    """3 naturals + 1 wild = quad bomb."""
    cards = [
        Card(Rank.NINE, Suit.SPADE, 0),
        Card(Rank.NINE, Suit.DIAMOND, 0),
        Card(Rank.NINE, Suit.CLUB, 0),
        Card(Rank.TWO, Suit.HEART, 0),  # wild at level 2
    ]
    hand = _hand_from_cards(cards)
    combos = generate_all_leads(hand, LEVEL_RANK)
    bombs = [c for c in combos if c.type == ComboType.BOMB_4]
    assert any(b.key == Rank.NINE and b.wild_count == 1 for b in bombs)


def test_level_card_natural_position_in_straight():
    """At level 2, a straight containing 2 uses 2 in its natural position.

    E.g., A-2-3-4-5 at level 2: the 2 sits between A-low and 3.
    """
    cards = [
        Card(Rank.ACE, Suit.SPADE, 0),
        Card(Rank.TWO, Suit.DIAMOND, 0),  # non-wild 2 (not heart)
        Card(Rank.THREE, Suit.CLUB, 0),
        Card(Rank.FOUR, Suit.HEART, 0),  # This is NOT wild (4♥ at level 2)
        Card(Rank.FIVE, Suit.SPADE, 0),
    ]
    hand = _hand_from_cards(cards)
    combos = generate_all_leads(hand, LEVEL_RANK)
    straights = [c for c in combos if c.type == ComboType.STRAIGHT]
    assert any(s.key == 5 for s in straights)


def test_combo_types_from_random_hands():
    """Across many random 27-card hands, we should see most combo types."""
    seen_types: set[int] = set()
    deck = make_deck()
    for _ in range(1000):
        random.shuffle(deck)
        hand = set(deck[:27])
        combos = generate_all_leads(hand, LEVEL_RANK)
        for c in combos:
            seen_types.add(c.type)

    expected_common = {
        ComboType.SINGLE,
        ComboType.PAIR,
        ComboType.TRIPLE,
        ComboType.FULL_HOUSE,
        ComboType.STRAIGHT,
        ComboType.BOMB_4,
    }
    missing = expected_common - seen_types
    assert not missing, f"Common combo types never generated: {missing}"


# ─── generate_responses() tests ─────────────────────────────────────


def test_responses_to_single_include_higher_singles_and_bombs():
    """Responding to a single: all higher-rank singles must be included, plus bombs."""
    hand = _hand_from_cards([
        Card(Rank.FIVE, Suit.SPADE, 0),
        Card(Rank.KING, Suit.HEART, 0),
        Card(Rank.THREE, Suit.SPADE, 0),
        Card(Rank.THREE, Suit.HEART, 0),
        Card(Rank.THREE, Suit.CLUB, 0),
        Card(Rank.THREE, Suit.DIAMOND, 0),
    ])
    trick = Combo(ComboType.SINGLE, Rank.SEVEN, [])
    responses = generate_responses(hand, LEVEL_RANK, trick)
    types = {r.type for r in responses}

    # KING beats SEVEN; THREE (lower) does not → only KING single expected
    singles = [r for r in responses if r.type == ComboType.SINGLE]
    assert all(r.key > Rank.SEVEN or r.key == LEVEL_RANK for r in singles if r.key is not None), (
        "Response singles must all beat the trick or be the level rank"
    )
    # Bomb (4× THREE) always allowed as a response
    assert ComboType.BOMB_4 in types, "generate_responses must always include bombs"
    # PASS is always present
    assert ComboType.PASS in types, "generate_responses must always include PASS"


def test_responses_to_bomb_only_higher_bombs():
    """Responding to a bomb: only strictly larger bombs (or joker bomb) are valid."""
    hand = _hand_from_cards([
        Card(Rank.FIVE, Suit.SPADE, 0),
        Card(Rank.FIVE, Suit.HEART, 0),
        Card(Rank.FIVE, Suit.CLUB, 0),
        Card(Rank.FIVE, Suit.DIAMOND, 0),
        Card(Rank.THREE, Suit.SPADE, 0),
        Card(Rank.THREE, Suit.HEART, 0),
        Card(Rank.THREE, Suit.CLUB, 0),
        Card(Rank.THREE, Suit.DIAMOND, 0),
    ])
    trick = Combo(ComboType.BOMB_4, Rank.FIVE, [])
    responses = generate_responses(hand, LEVEL_RANK, trick)

    non_pass = [r for r in responses if r.type != ComboType.PASS]
    for r in non_pass:
        assert r.type in BOMB_TYPES, f"Response to a bomb must be a bomb, got {r.type}"
        assert r.beats(trick), f"Response bomb {r} must beat the trick bomb {trick}"


def test_responses_always_include_pass():
    """PASS is a valid response in all following positions."""
    hand = _hand_from_cards([Card(Rank.THREE, Suit.SPADE, 0)])
    for trick_type, trick_rank in [
        (ComboType.SINGLE, Rank.ACE),
        (ComboType.PAIR, Rank.KING),
        (ComboType.BOMB_4, Rank.QUEEN),
    ]:
        trick = Combo(trick_type, trick_rank, [])
        responses = generate_responses(hand, LEVEL_RANK, trick)
        assert any(r.type == ComboType.PASS for r in responses), (
            f"PASS must always be a valid response (trick={trick_type})"
        )
