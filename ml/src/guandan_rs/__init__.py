"""Python facade for the optional native Guan Dan move generator.

The project can run from a clean Python install without compiling Rust. When
``_guandan_rs`` is installed, these names are provided by the native extension;
otherwise a pure-Python implementation keeps the public API available.
"""

from __future__ import annotations

try:
    from _guandan_rs import (  # type: ignore[import-not-found]
        dedup_strategic,
        generate_all_leads,
        generate_responses,
        mc_rollout,
        mc_rollout_batch,
    )
    HAS_NATIVE = True
except ImportError:
    HAS_NATIVE = False
    from guandan.cards import (
        BOMB_SIZE_TO_TYPE,
        BOMB_TYPES,
        Card,
        ComboType,
        Rank,
        Suit,
        is_wild,
        level_order_key,
    )

    PyCard = tuple[int, int, int]
    PyCombo = tuple[int, int, list[PyCard], int, int]

    def _card_from_tuple(t: PyCard) -> Card:
        return Card(int(t[0]), int(t[1]), int(t[2]))

    def _combo(
        ctype: ComboType,
        key: int,
        cards: list[Card],
        length: int = 0,
        wild_count: int = 0,
    ) -> PyCombo:
        return (
            int(ctype),
            int(key),
            [(int(c.rank), int(c.suit), int(c.deck)) for c in cards],
            int(length),
            int(wild_count),
        )

    def _combo_cards(c: PyCombo) -> list[Card]:
        return [_card_from_tuple(t) for t in c[2]]

    def _beats(this: PyCombo, other: PyCombo | None, level_rank: int) -> bool:
        if other is None:
            return True

        my_type = ComboType(this[0])
        their_type = ComboType(other[0])
        my_bomb = my_type in BOMB_TYPES
        their_bomb = their_type in BOMB_TYPES

        if my_bomb and not their_bomb:
            return True
        if not my_bomb and their_bomb:
            return False

        if my_bomb and their_bomb:
            if my_type != their_type:
                return my_type > their_type
            if my_type == ComboType.STRAIGHT_FLUSH:
                return this[1] > other[1]
            if my_type == ComboType.BOMB_JOKER:
                return False
            return level_order_key(this[1], level_rank) > level_order_key(
                other[1], level_rank
            )

        if my_type != their_type:
            return False

        if my_type in (ComboType.STRAIGHT, ComboType.TUBE, ComboType.PLATE):
            return this[1] > other[1]

        return level_order_key(this[1], level_rank) > level_order_key(
            other[1], level_rank
        )

    def generate_all_leads(hand: list[PyCard], level_rank: int) -> list[PyCombo]:
        cards = {_card_from_tuple(t) for t in hand}
        combos: list[PyCombo] = []

        wilds = sorted(
            [c for c in cards if is_wild(c, level_rank)],
            key=lambda c: (c.suit, c.deck),
        )
        naturals = [c for c in cards if not is_wild(c, level_rank)]

        by_rank: dict[int, list[Card]] = {}
        for card in naturals:
            by_rank.setdefault(int(card.rank), []).append(card)
        for rank_cards in by_rank.values():
            rank_cards.sort(key=lambda c: (c.suit, c.deck))

        for rank, rank_cards in by_rank.items():
            for card in _unique_cards(rank_cards):
                combos.append(_combo(ComboType.SINGLE, rank, [card]))
        for wild in wilds:
            combos.append(_combo(ComboType.SINGLE, level_rank, [wild]))

        _add_natural_pairs(combos, by_rank)
        _add_wild_pairs(combos, by_rank, wilds, level_rank)
        if len(wilds) >= 2:
            combos.append(_combo(ComboType.PAIR, level_rank, wilds[:2], wild_count=2))

        _add_natural_triples(combos, by_rank)
        _add_wild_triples(combos, by_rank, wilds)
        _add_full_houses(combos, by_rank, wilds, level_rank)
        _add_straights(combos, by_rank, wilds)
        _add_tubes(combos, by_rank, wilds)
        _add_plates(combos, by_rank, wilds)
        _add_nofakind_bombs(combos, by_rank, wilds)
        _add_straight_flushes(combos, cards, wilds, level_rank)

        jokers = [
            c for c in cards if c.rank in (Rank.BLACK_JOKER, Rank.RED_JOKER)
        ]
        if len(jokers) == 4:
            combos.append(_combo(ComboType.BOMB_JOKER, 99, jokers))

        return _deduplicate(combos)

    def generate_responses(
        hand: list[PyCard], level_rank: int, trick: PyCombo
    ) -> list[PyCombo]:
        responses = [
            combo
            for combo in generate_all_leads(hand, level_rank)
            if _beats(combo, trick, level_rank)
        ]
        responses.append(_combo(ComboType.PASS, 0, []))
        return responses

    def dedup_strategic(legal: list[PyCombo]) -> list[PyCombo]:
        seen: set[tuple[int, ...]] = set()
        out: list[PyCombo] = []
        for combo in legal:
            key = _strategic_key(combo)
            if key not in seen:
                seen.add(key)
                out.append(combo)
        return out

    def mc_rollout(*args, **kwargs) -> float:
        raise NotImplementedError("mc_rollout requires the native _guandan_rs extension")

    def mc_rollout_batch(*args, **kwargs) -> list[float]:
        raise NotImplementedError(
            "mc_rollout_batch requires the native _guandan_rs extension"
        )

    def _unique_cards(cards: list[Card]) -> list[Card]:
        seen: set[tuple[int, int]] = set()
        result: list[Card] = []
        for card in cards:
            key = (int(card.rank), int(card.suit))
            if key not in seen:
                seen.add(key)
                result.append(card)
        return result

    def _add_natural_pairs(
        combos: list[PyCombo], by_rank: dict[int, list[Card]]
    ) -> None:
        for rank, cards in by_rank.items():
            if len(cards) >= 2:
                combos.append(_combo(ComboType.PAIR, rank, cards[:2]))

    def _add_wild_pairs(
        combos: list[PyCombo],
        by_rank: dict[int, list[Card]],
        wilds: list[Card],
        level_rank: int,
    ) -> None:
        if not wilds:
            return
        for rank, cards in by_rank.items():
            if rank >= Rank.BLACK_JOKER:
                continue
            if cards:
                combos.append(
                    _combo(ComboType.PAIR, rank, [cards[0], wilds[0]], wild_count=1)
                )

    def _add_natural_triples(
        combos: list[PyCombo], by_rank: dict[int, list[Card]]
    ) -> None:
        for rank, cards in by_rank.items():
            if rank < Rank.BLACK_JOKER and len(cards) >= 3:
                combos.append(_combo(ComboType.TRIPLE, rank, cards[:3]))

    def _add_wild_triples(
        combos: list[PyCombo], by_rank: dict[int, list[Card]], wilds: list[Card]
    ) -> None:
        n_wild = len(wilds)
        for rank, cards in by_rank.items():
            if rank >= Rank.BLACK_JOKER:
                continue
            n_cards = len(cards)
            if n_cards >= 2 and n_wild >= 1:
                combos.append(
                    _combo(
                        ComboType.TRIPLE,
                        rank,
                        cards[:2] + wilds[:1],
                        wild_count=1,
                    )
                )
            if n_cards >= 1 and n_wild >= 2:
                combos.append(
                    _combo(
                        ComboType.TRIPLE,
                        rank,
                        cards[:1] + wilds[:2],
                        wild_count=2,
                    )
                )

    def _collect_triples_and_pairs(
        by_rank: dict[int, list[Card]], wilds: list[Card], level_rank: int
    ) -> tuple[list[tuple[int, list[Card], int]], list[tuple[int, list[Card], int]]]:
        n_wild = len(wilds)
        triples: list[tuple[int, list[Card], int]] = []
        pairs: list[tuple[int, list[Card], int]] = []

        for rank, cards in by_rank.items():
            if rank >= Rank.BLACK_JOKER:
                continue
            n_cards = len(cards)
            if n_cards >= 3:
                triples.append((rank, cards[:3], 0))
            if n_cards >= 2 and n_wild >= 1:
                triples.append((rank, cards[:2] + wilds[:1], 1))
            if n_cards >= 1 and n_wild >= 2:
                triples.append((rank, cards[:1] + wilds[:2], 2))
            if n_cards >= 2:
                pairs.append((rank, cards[:2], 0))
            if n_cards >= 1 and n_wild >= 1:
                pairs.append((rank, [cards[0], wilds[0]], 1))

        if n_wild >= 2:
            pairs.append((level_rank, wilds[:2], 2))

        return triples, pairs

    def _add_full_houses(
        combos: list[PyCombo],
        by_rank: dict[int, list[Card]],
        wilds: list[Card],
        level_rank: int,
    ) -> None:
        n_wild = len(wilds)
        triples, pairs = _collect_triples_and_pairs(by_rank, wilds, level_rank)

        for triple_rank, triple_cards, triple_wilds in triples:
            for pair_rank, pair_cards, pair_wilds in pairs:
                if triple_rank == pair_rank:
                    continue
                if triple_wilds + pair_wilds > n_wild:
                    continue
                if set(triple_cards) & set(pair_cards):
                    continue
                combos.append(
                    _combo(
                        ComboType.FULL_HOUSE,
                        triple_rank,
                        list(triple_cards) + list(pair_cards),
                        wild_count=triple_wilds + pair_wilds,
                    )
                )

    def _natural_rank_for_seq(rank_val: int) -> int:
        if rank_val == 1:
            return int(Rank.ACE)
        return rank_val

    def _get_natural_cards_at_rank(
        by_rank: dict[int, list[Card]], rank_val: int
    ) -> list[Card]:
        card_rank = _natural_rank_for_seq(rank_val)
        return [c for c in by_rank.get(card_rank, []) if c.rank < Rank.BLACK_JOKER]

    def _add_straights(
        combos: list[PyCombo], by_rank: dict[int, list[Card]], wilds: list[Card]
    ) -> None:
        n_wild = len(wilds)
        for start in range(1, 11):
            ranks_needed = list(range(start, start + 5))
            available: list[list[Card]] = []
            gaps = 0
            for rank in ranks_needed:
                cards_at_rank = _get_natural_cards_at_rank(by_rank, rank)
                if not cards_at_rank:
                    gaps += 1
                available.append(cards_at_rank)

            if gaps <= n_wild:
                _enumerate_straight_selections(
                    combos, available, ranks_needed, wilds, n_wild
                )

    def _enumerate_straight_selections(
        combos: list[PyCombo],
        available: list[list[Card]],
        ranks_needed: list[int],
        wilds: list[Card],
        n_wild: int,
    ) -> None:
        top_rank = ranks_needed[-1]

        def select(
            idx: int,
            chosen: list[Card],
            wilds_used: int,
            suits_seen: set[int],
        ) -> None:
            if idx == 5:
                if len(suits_seen) > 1:
                    combos.append(
                        _combo(
                            ComboType.STRAIGHT,
                            top_rank,
                            list(chosen),
                            length=5,
                            wild_count=wilds_used,
                        )
                    )
                return

            cards_at_rank = available[idx]
            if cards_at_rank:
                seen_suits: set[int] = set()
                for card in cards_at_rank:
                    if card.suit not in seen_suits:
                        seen_suits.add(int(card.suit))
                        select(
                            idx + 1,
                            chosen + [card],
                            wilds_used,
                            suits_seen | {int(card.suit)},
                        )
            elif wilds_used < n_wild:
                select(
                    idx + 1,
                    chosen + [wilds[wilds_used]],
                    wilds_used + 1,
                    suits_seen,
                )

        select(0, [], 0, set())

    def _add_tubes(
        combos: list[PyCombo], by_rank: dict[int, list[Card]], wilds: list[Card]
    ) -> None:
        n_wild = len(wilds)
        for start in range(1, 13):
            ranks_needed = list(range(start, start + 3))
            available: list[list[Card]] = []
            total_gaps = 0
            for rank in ranks_needed:
                cards_at_rank = _get_natural_cards_at_rank(by_rank, rank)
                total_gaps += max(0, 2 - len(cards_at_rank))
                available.append(cards_at_rank)

            if total_gaps > n_wild:
                continue

            selected: list[Card] = []
            wilds_used = 0
            valid = True
            for cards_at_rank in available:
                if len(cards_at_rank) >= 2:
                    selected.extend(cards_at_rank[:2])
                elif len(cards_at_rank) == 1:
                    selected.append(cards_at_rank[0])
                    if wilds_used < n_wild:
                        selected.append(wilds[wilds_used])
                        wilds_used += 1
                    else:
                        valid = False
                        break
                elif wilds_used + 2 <= n_wild:
                    selected.extend([wilds[wilds_used], wilds[wilds_used + 1]])
                    wilds_used += 2
                else:
                    valid = False
                    break

            if valid and len(selected) == 6:
                combos.append(
                    _combo(
                        ComboType.TUBE,
                        ranks_needed[-1],
                        selected,
                        wild_count=wilds_used,
                    )
                )

    def _add_plates(
        combos: list[PyCombo], by_rank: dict[int, list[Card]], wilds: list[Card]
    ) -> None:
        n_wild = len(wilds)
        for start in range(1, 14):
            ranks_needed = list(range(start, start + 2))
            available: list[list[Card]] = []
            total_gaps = 0
            for rank in ranks_needed:
                cards_at_rank = _get_natural_cards_at_rank(by_rank, rank)
                total_gaps += max(0, 3 - len(cards_at_rank))
                available.append(cards_at_rank)

            if total_gaps > n_wild:
                continue

            selected: list[Card] = []
            wilds_used = 0
            valid = True
            for cards_at_rank in available:
                take_natural = min(len(cards_at_rank), 3)
                selected.extend(cards_at_rank[:take_natural])
                need_wilds = 3 - take_natural
                if wilds_used + need_wilds <= n_wild:
                    for _ in range(need_wilds):
                        selected.append(wilds[wilds_used])
                        wilds_used += 1
                else:
                    valid = False
                    break

            if valid and len(selected) == 6:
                combos.append(
                    _combo(
                        ComboType.PLATE,
                        ranks_needed[-1],
                        selected,
                        wild_count=wilds_used,
                    )
                )

    def _add_nofakind_bombs(
        combos: list[PyCombo], by_rank: dict[int, list[Card]], wilds: list[Card]
    ) -> None:
        n_wild = len(wilds)
        for rank, cards in by_rank.items():
            if rank >= Rank.BLACK_JOKER:
                continue
            n_cards = len(cards)
            for size in range(4, min(n_cards + n_wild, 10) + 1):
                needed_wilds = max(0, size - n_cards)
                if needed_wilds > n_wild:
                    continue
                bomb_type = BOMB_SIZE_TO_TYPE.get(size)
                if bomb_type:
                    used_cards = cards[: min(n_cards, size)] + wilds[:needed_wilds]
                    combos.append(
                        _combo(
                            bomb_type,
                            rank,
                            used_cards,
                            wild_count=needed_wilds,
                        )
                    )

    def _add_straight_flushes(
        combos: list[PyCombo],
        hand: set[Card],
        wilds: list[Card],
        level_rank: int,
    ) -> None:
        n_wild = len(wilds)
        by_suit_rank: dict[int, dict[int, list[Card]]] = {s: {} for s in range(4)}
        for card in hand:
            if card.rank >= Rank.BLACK_JOKER or is_wild(card, level_rank):
                continue
            by_suit_rank[int(card.suit)].setdefault(int(card.rank), []).append(card)

        for suit in range(4):
            suit_cards = by_suit_rank[suit]
            for start in range(1, 11):
                ranks_needed = list(range(start, start + 5))
                selected: list[Card] = []
                wilds_used = 0
                valid = True

                for rank in ranks_needed:
                    card_rank = _natural_rank_for_seq(rank)
                    cards_at_rank = suit_cards.get(card_rank, [])
                    if cards_at_rank:
                        selected.append(cards_at_rank[0])
                    elif wilds_used < n_wild:
                        selected.append(wilds[wilds_used])
                        wilds_used += 1
                    else:
                        valid = False
                        break

                if valid and len(selected) == 5:
                    combos.append(
                        _combo(
                            ComboType.STRAIGHT_FLUSH,
                            ranks_needed[-1],
                            selected,
                            length=0,
                            wild_count=wilds_used,
                        )
                    )

    def _deduplicate(combos: list[PyCombo]) -> list[PyCombo]:
        seen: set[tuple[int, int, int, tuple[tuple[int, int], ...]]] = set()
        result: list[PyCombo] = []
        for combo in combos:
            card_ids = tuple(sorted((rank, suit) for rank, suit, _deck in combo[2]))
            key = (combo[0], combo[1], combo[3], card_ids)
            if key not in seen:
                seen.add(key)
                result.append(combo)
        return result

    def _is_suit_agnostic(combo_type: int) -> bool:
        return 1 <= combo_type <= 9 or 11 <= combo_type <= 15

    def _strategic_key(combo: PyCombo) -> tuple[int, ...]:
        combo_type, key, cards, length, wild_count = combo
        out = [combo_type, key, length, wild_count]
        if _is_suit_agnostic(combo_type):
            out.extend(sorted(rank for rank, _suit, _deck in cards))
        else:
            for rank, suit in sorted((rank, suit) for rank, suit, _deck in cards):
                out.extend([rank, suit])
        return tuple(out)

try:
    from _guandan_rs import select_legal as select_legal  # type: ignore[import-not-found]
except ImportError:

    def select_legal(hand, level_rank, trick):
        legal = (
            generate_all_leads(hand, level_rank)
            if trick is None
            else generate_responses(hand, level_rank, trick)
        )
        return dedup_strategic(legal)

__all__ = [
    "dedup_strategic",
    "generate_all_leads",
    "generate_responses",
    "select_legal",
    "mc_rollout",
    "mc_rollout_batch",
    "HAS_NATIVE",
]
