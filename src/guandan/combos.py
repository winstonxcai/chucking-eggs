"""Combo classification, generation, comparison, and move generation.

This is the most complex file: 7 ordinary combo types + 9 bomb tiers,
wild card substitution, ace-low sequences, straight vs straight-flush
routing, and deduplication.
"""

from __future__ import annotations

from itertools import combinations

from .cards import (
    BOMB_SIZE_TO_TYPE,
    BOMB_TYPES,
    Card,
    ComboType,
    Rank,
    Suit,
    is_wild,
    level_order_key,
)

# Try to import Rust backend for fast movegen
try:
    from guandan_rs import (
        generate_all_leads as _rs_leads,
        generate_responses as _rs_responses,
    )
    _USE_RUST = True
except ImportError:
    _USE_RUST = False


class Combo:
    __slots__ = ("type", "key", "cards", "length", "wild_count")

    def __init__(
        self,
        combo_type: ComboType,
        key_rank: int,
        cards: list[Card],
        length: int = 0,
        wild_count: int = 0,
    ):
        self.type = combo_type
        self.key = key_rank          # primary rank for comparison
        self.cards = tuple(cards)    # actual cards played
        self.length = length         # for straights (always 5 for now)
        self.wild_count = wild_count

    def beats(self, other: Combo | None, level_rank: int) -> bool:
        """Can this combo beat *other*? (other=None means free lead.)"""
        if other is None:
            return True

        my_bomb = self.type in BOMB_TYPES
        their_bomb = other.type in BOMB_TYPES

        # Any bomb beats any non-bomb
        if my_bomb and not their_bomb:
            return True
        if not my_bomb and their_bomb:
            return False

        # Both bombs
        if my_bomb and their_bomb:
            if self.type != other.type:
                return self.type > other.type
            # Same bomb tier
            if self.type == ComboType.STRAIGHT_FLUSH:
                return self.key > other.key  # natural order
            if self.type == ComboType.BOMB_JOKER:
                return False  # only one four-joker bomb exists
            # N-of-a-kind: level order
            return level_order_key(self.key, level_rank) > level_order_key(
                other.key, level_rank
            )

        # Neither is bomb: must match type
        if self.type != other.type:
            return False

        # Straights, tubes, plates: natural order
        if self.type in (ComboType.STRAIGHT, ComboType.TUBE, ComboType.PLATE):
            return self.key > other.key

        # Singles, pairs, triples, full houses: level order
        return level_order_key(self.key, level_rank) > level_order_key(
            other.key, level_rank
        )

    def __repr__(self) -> str:
        return f"Combo({self.type.name}, key={self.key}, cards={len(self.cards)}, wilds={self.wild_count})"


# ─── PUBLIC API ─────────────────────────────────────────


def _rs_tuple_to_combo(t: tuple) -> Combo:
    """Convert Rust (combo_type, key, cards, length, wild_count) tuple to Combo."""
    combo_type, key, cards_raw, length, wild_count = t
    cards = [Card(r, s, d) for r, s, d in cards_raw]
    return Combo(ComboType(combo_type), key, cards, length, wild_count)


def generate_all_leads(hand: set[Card], level_rank: int) -> list[Combo]:
    """Generate every legal combo from hand (free lead)."""
    if _USE_RUST:
        hand_tuples = [(c.rank, c.suit, c.deck) for c in hand]
        return [_rs_tuple_to_combo(t) for t in _rs_leads(hand_tuples, level_rank)]
    return _py_generate_all_leads(hand, level_rank)


def generate_responses(
    hand: set[Card], level_rank: int, trick: Combo
) -> list[Combo]:
    """Generate all combos that beat the current trick, plus PASS."""
    if _USE_RUST:
        hand_tuples = [(c.rank, c.suit, c.deck) for c in hand]
        trick_tuple = (
            int(trick.type), trick.key,
            [(c.rank, c.suit, c.deck) for c in trick.cards],
            trick.length, trick.wild_count,
        )
        return [_rs_tuple_to_combo(t) for t in _rs_responses(hand_tuples, level_rank, trick_tuple)]
    return _py_generate_responses(hand, level_rank, trick)


def _py_generate_all_leads(hand: set[Card], level_rank: int) -> list[Combo]:
    """Pure Python implementation of generate_all_leads."""
    combos: list[Combo] = []

    wilds = sorted([c for c in hand if is_wild(c, level_rank)], key=lambda c: (c.suit, c.deck))
    naturals = [c for c in hand if not is_wild(c, level_rank)]

    by_rank: dict[int, list[Card]] = {}
    for c in naturals:
        by_rank.setdefault(c.rank, []).append(c)
    for cards in by_rank.values():
        cards.sort(key=lambda c: (c.suit, c.deck))

    # Singles
    for rank, cards in by_rank.items():
        for c in _unique_cards(cards):
            combos.append(_make(ComboType.SINGLE, rank, [c]))
    for w in wilds:
        combos.append(_make(ComboType.SINGLE, level_rank, [w]))

    # Pairs
    _add_natural_pairs(combos, by_rank)
    _add_wild_pairs(combos, by_rank, wilds, level_rank)
    if len(wilds) >= 2:
        combos.append(_make(ComboType.PAIR, level_rank, wilds[:2], wild_count=2))

    # Triples
    _add_natural_triples(combos, by_rank)
    _add_wild_triples(combos, by_rank, wilds, level_rank)

    # Full houses
    _add_full_houses(combos, by_rank, wilds, level_rank)

    # Straights (exactly 5 consecutive, natural order, NOT all same suit)
    _add_straights(combos, by_rank, wilds, level_rank)

    # Tubes (exactly 3 consecutive pairs, natural order)
    _add_tubes(combos, by_rank, wilds, level_rank)

    # Plates (exactly 2 consecutive triples, natural order)
    _add_plates(combos, by_rank, wilds, level_rank)

    # N-of-a-kind bombs (4 through 10)
    _add_nofakind_bombs(combos, by_rank, wilds, level_rank)

    # Straight flushes (5 consecutive same suit, natural order)
    _add_straight_flushes(combos, hand, wilds, level_rank)

    # Four-joker bomb
    jokers = [c for c in hand if c.rank in (Rank.BLACK_JOKER, Rank.RED_JOKER)]
    if len(jokers) == 4:
        combos.append(_make(ComboType.BOMB_JOKER, 99, jokers))

    return _deduplicate(combos)


def _py_generate_responses(
    hand: set[Card], level_rank: int, trick: Combo
) -> list[Combo]:
    """Pure Python implementation of generate_responses."""
    leads = _py_generate_all_leads(hand, level_rank)
    responses = [c for c in leads if c.beats(trick, level_rank)]
    responses.append(Combo(ComboType.PASS, 0, []))
    return responses


# ─── HELPERS ────────────────────────────────────────────


def _make(
    ctype: ComboType,
    key: int,
    cards: list[Card],
    length: int = 0,
    wild_count: int = 0,
) -> Combo:
    return Combo(ctype, key, cards, length, wild_count)


def _unique_cards(cards: list[Card]) -> list[Card]:
    """Deduplicate identical cards (same rank+suit but different deck copy)."""
    seen: set[tuple[int, int]] = set()
    result: list[Card] = []
    for c in cards:
        k = (c.rank, c.suit)
        if k not in seen:
            seen.add(k)
            result.append(c)
    return result


# ─── PAIRS ──────────────────────────────────────────────


def _add_natural_pairs(combos: list[Combo], by_rank: dict[int, list[Card]]) -> None:
    """Natural pairs: 2+ cards of same rank. BJ+BJ ok, RJ+RJ ok, BJ+RJ NOT ok."""
    for rank, cards in by_rank.items():
        if rank == Rank.BLACK_JOKER:
            if len(cards) >= 2:
                combos.append(_make(ComboType.PAIR, rank, cards[:2]))
            continue
        if rank == Rank.RED_JOKER:
            if len(cards) >= 2:
                combos.append(_make(ComboType.PAIR, rank, cards[:2]))
            continue
        if len(cards) >= 2:
            # Could have up to 8 cards of same rank from double deck,
            # but for pairs we just need any 2
            combos.append(_make(ComboType.PAIR, rank, cards[:2]))


def _add_wild_pairs(
    combos: list[Combo],
    by_rank: dict[int, list[Card]],
    wilds: list[Card],
    level_rank: int,
) -> None:
    """Wild + any non-joker natural = pair of that rank."""
    if not wilds:
        return
    for rank, cards in by_rank.items():
        if rank >= Rank.BLACK_JOKER:
            continue  # wilds can't substitute jokers
        if cards:
            combos.append(
                _make(ComboType.PAIR, rank, [cards[0], wilds[0]], wild_count=1)
            )


# ─── TRIPLES ────────────────────────────────────────────


def _add_natural_triples(
    combos: list[Combo], by_rank: dict[int, list[Card]]
) -> None:
    for rank, cards in by_rank.items():
        if rank >= Rank.BLACK_JOKER:
            continue  # no joker triples possible
        if len(cards) >= 3:
            combos.append(_make(ComboType.TRIPLE, rank, cards[:3]))


def _add_wild_triples(
    combos: list[Combo],
    by_rank: dict[int, list[Card]],
    wilds: list[Card],
    level_rank: int,
) -> None:
    """Wild(s) + naturals to form triples."""
    n_wild = len(wilds)
    for rank, cards in by_rank.items():
        if rank >= Rank.BLACK_JOKER:
            continue
        n = len(cards)
        if n >= 2 and n_wild >= 1:
            combos.append(
                _make(ComboType.TRIPLE, rank, cards[:2] + wilds[:1], wild_count=1)
            )
        if n >= 1 and n_wild >= 2:
            combos.append(
                _make(ComboType.TRIPLE, rank, cards[:1] + wilds[:2], wild_count=2)
            )


# ─── FULL HOUSES ────────────────────────────────────────


def _collect_triples_and_pairs(
    by_rank: dict[int, list[Card]], wilds: list[Card], level_rank: int
) -> tuple[list[tuple[int, list[Card], int]], list[tuple[int, list[Card], int]]]:
    """Collect all possible triples and pairs (with wild augmentation).

    Returns (triples, pairs) where each entry is (rank, cards_used, wilds_used).
    """
    n_wild = len(wilds)
    triples: list[tuple[int, list[Card], int]] = []
    pairs: list[tuple[int, list[Card], int]] = []

    for rank, cards in by_rank.items():
        if rank >= Rank.BLACK_JOKER:
            continue
        n = len(cards)
        # Triples
        if n >= 3:
            triples.append((rank, cards[:3], 0))
        if n >= 2 and n_wild >= 1:
            triples.append((rank, cards[:2] + wilds[:1], 1))
        if n >= 1 and n_wild >= 2:
            triples.append((rank, cards[:1] + wilds[:2], 2))
        # Pairs
        if n >= 2:
            pairs.append((rank, cards[:2], 0))
        if n >= 1 and n_wild >= 1:
            pairs.append((rank, [cards[0], wilds[0]], 1))

    # Pair of wilds = pair of level cards
    if n_wild >= 2:
        pairs.append((level_rank, wilds[:2], 2))

    return triples, pairs


def _add_full_houses(
    combos: list[Combo],
    by_rank: dict[int, list[Card]],
    wilds: list[Card],
    level_rank: int,
) -> None:
    """Triple + Pair from different ranks. Total wilds used ≤ len(wilds)."""
    n_wild = len(wilds)
    triples, pairs = _collect_triples_and_pairs(by_rank, wilds, level_rank)

    for t_rank, t_cards, t_wilds in triples:
        for p_rank, p_cards, p_wilds in pairs:
            # Triple and pair must use different ranks
            if t_rank == p_rank:
                continue
            # Total wilds used must not exceed available
            if t_wilds + p_wilds > n_wild:
                continue
            # Cards must not overlap (check actual card objects)
            t_set = set(id(c) for c in t_cards)
            p_set = set(id(c) for c in p_cards)
            if t_set & p_set:
                continue
            combos.append(
                _make(
                    ComboType.FULL_HOUSE,
                    t_rank,
                    list(t_cards) + list(p_cards),
                    wild_count=t_wilds + p_wilds,
                )
            )


# ─── SEQUENCES (straights, tubes, plates) ──────────────


def _natural_rank_for_seq(rank_val: int) -> int:
    """Map sequence rank values to actual card ranks.

    In sequences, we use natural order positions 1..14 where:
    - 1 = Ace-low
    - 2..13 = normal ranks
    - 14 = Ace-high
    Both 1 and 14 map to the Ace card (Rank.ACE = 14).
    """
    if rank_val == 1:
        return Rank.ACE  # Ace-low
    return rank_val


def _get_natural_cards_at_rank(
    by_rank: dict[int, list[Card]], rank_val: int
) -> list[Card]:
    """Get non-joker natural cards at a given sequence position."""
    card_rank = _natural_rank_for_seq(rank_val)
    return [c for c in by_rank.get(card_rank, []) if c.rank < Rank.BLACK_JOKER]


def _add_straights(
    combos: list[Combo],
    by_rank: dict[int, list[Card]],
    wilds: list[Card],
    level_rank: int,
) -> None:
    """Exactly 5 consecutive cards in natural order, NOT all same suit.

    Ace-low: A-2-3-4-5 (key=5). Ace-high: 10-J-Q-K-A (key=14).
    No wrapping. Jokers never in straights. Level cards in NATURAL position.
    """
    n_wild = len(wilds)
    # Valid start positions: 1(A-low) through 10
    for start in range(1, 11):
        ranks_needed = list(range(start, start + 5))
        # Collect available natural cards at each position
        available: list[list[Card]] = []
        gaps = 0
        for r in ranks_needed:
            cards_at_r = _get_natural_cards_at_rank(by_rank, r)
            if not cards_at_r:
                gaps += 1
            available.append(cards_at_r)

        if gaps > n_wild:
            continue

        # Generate card selections using wilds for gaps
        _enumerate_straight_selections(
            combos, available, ranks_needed, wilds, n_wild, level_rank
        )


def _enumerate_straight_selections(
    combos: list[Combo],
    available: list[list[Card]],
    ranks_needed: list[int],
    wilds: list[Card],
    n_wild: int,
    level_rank: int,
) -> None:
    """Enumerate card selections for a 5-card straight window."""
    top_rank = ranks_needed[-1]

    def _select(
        idx: int,
        chosen: list[Card],
        wilds_used: int,
        suits_seen: set[int],
    ) -> None:
        if idx == 5:
            # Must NOT be all same suit (that would be a straight flush)
            if len(suits_seen) > 1:
                combos.append(
                    _make(
                        ComboType.STRAIGHT,
                        top_rank,
                        list(chosen),
                        length=5,
                        wild_count=wilds_used,
                    )
                )
            return

        cards_at = available[idx]
        if cards_at:
            # Use a natural card (pick first unique by suit for variety)
            seen_suits: set[int] = set()
            for c in cards_at:
                if c.suit not in seen_suits:
                    seen_suits.add(c.suit)
                    _select(
                        idx + 1,
                        chosen + [c],
                        wilds_used,
                        suits_seen | {c.suit},
                    )
        if wilds_used < n_wild and (not cards_at or True):
            # Use a wild to fill this position (wild declared as needed card)
            # Wild doesn't have a fixed suit for straight purposes,
            # so we treat it as having a unique "wild suit" that won't
            # make it all-same-suit. But actually, the wild is declared
            # as a specific card — in practice for straights we just
            # need at least 2 different suits among the natural cards.
            # Simpler: if using wild, it doesn't constrain suit.
            if cards_at:
                # Can also choose to use a wild even when natural cards exist
                # (might matter for saving naturals), but for movegen we
                # generate all combos — using natural is always preferred
                pass
            if not cards_at:
                # Must use wild for this gap
                _select(
                    idx + 1,
                    chosen + [wilds[wilds_used]],
                    wilds_used + 1,
                    suits_seen,  # wild doesn't add a suit
                )

    _select(0, [], 0, set())


def _add_tubes(
    combos: list[Combo],
    by_rank: dict[int, list[Card]],
    wilds: list[Card],
    level_rank: int,
) -> None:
    """Exactly 3 consecutive pairs in natural order (6 cards).

    No joker pairs. Level cards in natural position.
    Ace-low: AA-22-33 (key=3). Ace-high: QQ-KK-AA (key=14).
    """
    n_wild = len(wilds)
    # Valid start positions: 1(A-low) through 12
    for start in range(1, 13):
        ranks_needed = list(range(start, start + 3))
        # Need 2 cards at each of 3 consecutive ranks
        total_gaps = 0
        available: list[list[Card]] = []
        for r in ranks_needed:
            cards_at_r = _get_natural_cards_at_rank(by_rank, r)
            have = len(cards_at_r)
            need = 2
            gap = max(0, need - have)
            total_gaps += gap
            available.append(cards_at_r)

        if total_gaps > n_wild:
            continue

        # Build the tube
        top_rank = ranks_needed[-1]
        selected: list[Card] = []
        wilds_used = 0
        valid = True
        for i, r in enumerate(ranks_needed):
            cards_at = available[i]
            have = len(cards_at)
            if have >= 2:
                selected.extend(cards_at[:2])
            elif have == 1:
                selected.append(cards_at[0])
                if wilds_used < n_wild:
                    selected.append(wilds[wilds_used])
                    wilds_used += 1
                else:
                    valid = False
                    break
            else:
                # Need 2 wilds
                if wilds_used + 2 <= n_wild:
                    selected.append(wilds[wilds_used])
                    selected.append(wilds[wilds_used + 1])
                    wilds_used += 2
                else:
                    valid = False
                    break

        if valid and len(selected) == 6:
            combos.append(
                _make(ComboType.TUBE, top_rank, selected, wild_count=wilds_used)
            )


def _add_plates(
    combos: list[Combo],
    by_rank: dict[int, list[Card]],
    wilds: list[Card],
    level_rank: int,
) -> None:
    """Exactly 2 consecutive triples in natural order (6 cards).

    Level cards in natural position.
    Ace-low: AAA-222 (key=2). Ace-high: KKK-AAA (key=14).
    """
    n_wild = len(wilds)
    # Valid start positions: 1(A-low) through 13
    for start in range(1, 14):
        ranks_needed = list(range(start, start + 2))
        total_gaps = 0
        available: list[list[Card]] = []
        for r in ranks_needed:
            cards_at_r = _get_natural_cards_at_rank(by_rank, r)
            have = len(cards_at_r)
            gap = max(0, 3 - have)
            total_gaps += gap
            available.append(cards_at_r)

        if total_gaps > n_wild:
            continue

        top_rank = ranks_needed[-1]
        selected: list[Card] = []
        wilds_used = 0
        valid = True
        for i, r in enumerate(ranks_needed):
            cards_at = available[i]
            have = len(cards_at)
            take_natural = min(have, 3)
            selected.extend(cards_at[:take_natural])
            need_wilds = 3 - take_natural
            if wilds_used + need_wilds <= n_wild:
                for j in range(need_wilds):
                    selected.append(wilds[wilds_used])
                    wilds_used += 1
            else:
                valid = False
                break

        if valid and len(selected) == 6:
            combos.append(
                _make(ComboType.PLATE, top_rank, selected, wild_count=wilds_used)
            )


# ─── BOMBS ──────────────────────────────────────────────


def _add_nofakind_bombs(
    combos: list[Combo],
    by_rank: dict[int, list[Card]],
    wilds: list[Card],
    level_rank: int,
) -> None:
    """N-of-a-kind bombs (4 through 10). Wilds extend natural groups."""
    n_wild = len(wilds)
    for rank, cards in by_rank.items():
        if rank >= Rank.BLACK_JOKER:
            continue
        n = len(cards)
        for size in range(4, min(n + n_wild, 10) + 1):
            needed_wilds = max(0, size - n)
            if needed_wilds > n_wild:
                continue
            bomb_type = BOMB_SIZE_TO_TYPE.get(size)
            if bomb_type:
                used_cards = cards[: min(n, size)] + wilds[:needed_wilds]
                combos.append(
                    _make(bomb_type, rank, used_cards, wild_count=needed_wilds)
                )


def _add_straight_flushes(
    combos: list[Combo],
    hand: set[Card],
    wilds: list[Card],
    level_rank: int,
) -> None:
    """5 consecutive same-suit cards in natural order.

    Ace high or low. Wilds can fill gaps (declared as needed suit).
    Jokers cannot be used. Level cards in natural position.
    """
    n_wild = len(wilds)

    # Organize non-joker, non-wild cards by suit then rank
    by_suit_rank: dict[int, dict[int, list[Card]]] = {s: {} for s in range(4)}
    for c in hand:
        if c.rank >= Rank.BLACK_JOKER:
            continue
        if is_wild(c, level_rank):
            continue
        by_suit_rank[c.suit].setdefault(c.rank, []).append(c)

    for suit in range(4):
        suit_cards = by_suit_rank[suit]
        for start in range(1, 11):  # start positions for 5-card window
            ranks_needed = list(range(start, start + 5))
            top_rank = ranks_needed[-1]

            selected: list[Card] = []
            wilds_used = 0
            valid = True

            for r in ranks_needed:
                card_rank = _natural_rank_for_seq(r)
                cards_at = suit_cards.get(card_rank, [])
                if cards_at:
                    selected.append(cards_at[0])
                elif wilds_used < n_wild:
                    selected.append(wilds[wilds_used])
                    wilds_used += 1
                else:
                    valid = False
                    break

            if valid and len(selected) == 5:
                combos.append(
                    _make(
                        ComboType.STRAIGHT_FLUSH,
                        top_rank,
                        selected,
                        wild_count=wilds_used,
                    )
                )


# ─── DEDUPLICATION ──────────────────────────────────────


def _deduplicate(combos: list[Combo]) -> list[Combo]:
    """Remove duplicate combos (same type + key + card ranks/suits, ignoring deck copy)."""
    seen: set[tuple[int, int, int, tuple[tuple[int, int], ...]]] = set()
    result: list[Combo] = []
    for c in combos:
        card_ids = tuple(sorted((card.rank, card.suit) for card in c.cards))
        key = (c.type, c.key, c.length, card_ids)
        if key not in seen:
            seen.add(key)
            result.append(c)
    return result
