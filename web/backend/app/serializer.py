"""Serialize GuanDanEnv state to JSON for the frontend."""

from __future__ import annotations

from itertools import combinations as _combinations

from guandan.cards import BOMB_TYPES, Card, ComboType, Rank, Suit, is_wild, level_order_key
from guandan.combos import Combo
from guandan.game import GuanDanEnv

SUIT_SYMBOLS = {Suit.SPADE: "\u2660", Suit.HEART: "\u2665", Suit.DIAMOND: "\u2666", Suit.CLUB: "\u2663"}
RANK_NAMES = {
    2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8",
    9: "9", 10: "10", 11: "J", 12: "Q", 13: "K", 14: "A",
    Rank.BLACK_JOKER: "BJ", Rank.RED_JOKER: "RJ",
}
COMBO_TYPE_NAMES = {
    ComboType.PASS: "Pass",
    ComboType.SINGLE: "Single",
    ComboType.PAIR: "Pair",
    ComboType.TRIPLE: "Triple",
    ComboType.FULL_HOUSE: "Full House",
    ComboType.STRAIGHT: "Straight",
    ComboType.TUBE: "Tube",
    ComboType.PLATE: "Plate",
    ComboType.BOMB_4: "Bomb",
    ComboType.BOMB_5: "Bomb",
    ComboType.STRAIGHT_FLUSH: "Straight Flush",
    ComboType.BOMB_6: "Bomb",
    ComboType.BOMB_7: "Bomb",
    ComboType.BOMB_8: "Bomb",
    ComboType.BOMB_9: "Bomb",
    ComboType.BOMB_10: "Bomb",
    ComboType.BOMB_JOKER: "Rocket",
}
COMBO_TYPE_CHINESE = {
    ComboType.PASS: "\u8fc7",
    ComboType.SINGLE: "\u5355\u724c",
    ComboType.PAIR: "\u5bf9\u5b50",
    ComboType.TRIPLE: "\u4e09\u6761",
    ComboType.FULL_HOUSE: "\u4e09\u5e26\u4e8c",
    ComboType.STRAIGHT: "\u987a\u5b50",
    ComboType.TUBE: "\u8fde\u5bf9",
    ComboType.PLATE: "\u94a2\u677f",
    ComboType.BOMB_4: "\u70b8\u5f39",
    ComboType.BOMB_5: "\u70b8\u5f39",
    ComboType.STRAIGHT_FLUSH: "\u540c\u82b1\u987a",
    ComboType.BOMB_6: "\u70b8\u5f39",
    ComboType.BOMB_7: "\u70b8\u5f39",
    ComboType.BOMB_8: "\u70b8\u5f39",
    ComboType.BOMB_9: "\u70b8\u5f39",
    ComboType.BOMB_10: "\u70b8\u5f39",
    ComboType.BOMB_JOKER: "\u5929\u738b\u70b8",
}


def card_to_dto(card: Card) -> dict:
    is_joker = card.rank in (Rank.BLACK_JOKER, Rank.RED_JOKER)
    rank_display = RANK_NAMES.get(card.rank, "?")
    suit_symbol = "" if is_joker else SUIT_SYMBOLS.get(card.suit, "?")
    return {
        "rank": card.rank,
        "suit": card.suit,
        "deck": card.deck,
        "id": f"{card.rank}-{card.suit}-{card.deck}",
        "display": rank_display if is_joker else f"{rank_display}{suit_symbol}",
        "rank_display": rank_display,
        "suit_symbol": suit_symbol,
    }


def combo_to_dto(combo: Combo, level_rank: int) -> dict:
    if combo.type == ComboType.PASS:
        return {"type": "PASS", "type_id": 0, "type_name": "Pass",
                "type_chinese": "\u8fc7", "cards": [], "is_pass": True}
    cards = [card_to_dto(c) for c in combo.cards]
    display_parts = [c["display"] for c in cards]
    return {
        "type": combo.type.name,
        "type_id": int(combo.type),
        "type_name": COMBO_TYPE_NAMES.get(combo.type, "Unknown"),
        "type_chinese": COMBO_TYPE_CHINESE.get(combo.type, ""),
        "cards": cards,
        "display": " ".join(display_parts),
        "key": combo.key,
        "wild_count": combo.wild_count,
        "is_pass": False,
    }


def combo_sort_key(combo: Combo, level_rank: int) -> tuple:
    """Sort key for ordering combos smallest → biggest."""
    is_bomb = combo.type in BOMB_TYPES
    if is_bomb:
        bomb_tier = int(combo.type)
        if combo.type == ComboType.STRAIGHT_FLUSH:
            rank_key = combo.key  # natural order
        elif combo.type == ComboType.BOMB_JOKER:
            rank_key = 99
        else:
            rank_key = level_order_key(combo.key, level_rank)
        return (1, bomb_tier, rank_key)
    else:
        if combo.type in (ComboType.STRAIGHT, ComboType.TUBE, ComboType.PLATE):
            key_val = combo.key  # natural order
        else:
            key_val = level_order_key(combo.key, level_rank)
        return (0, int(combo.type), key_val)


def sort_hand(cards: set[Card], level_rank: int) -> list[Card]:
    """Sort hand by level order, then suit, then deck."""
    def sort_key(c: Card):
        if c.rank in (Rank.BLACK_JOKER, Rank.RED_JOKER):
            return (100 + c.rank, c.suit, c.deck)
        if is_wild(c, level_rank):
            return (99, c.suit, c.deck)
        return (level_order_key(c.rank, level_rank), c.suit, c.deck)
    return sorted(cards, key=sort_key)


def _expand_natural_pairs(raw_moves: list[Combo], hand: set, level_rank: int) -> list[Combo]:
    """Ensure all unique-suit-pair combos are present in legal_moves.

    The backend pair generator takes cards[:2] per rank. In a double deck, if
    deck-0 of one suit is grouped on the frontend, only deck-1 of a different
    suit may form a valid pair — but that combo won't be generated unless we
    enumerate all suit combinations here.
    """
    # Build {rank: {suit: representative_card}} for non-wild naturals
    by_rank_suit: dict[int, dict[int, Card]] = {}
    for c in hand:
        if not is_wild(c, level_rank) and c.rank < Rank.BLACK_JOKER:
            by_rank_suit.setdefault(c.rank, {}).setdefault(c.suit, c)

    # Track existing natural pairs so we don't duplicate them
    existing: set[tuple] = set()
    for combo in raw_moves:
        if combo.type == ComboType.PAIR and combo.wild_count == 0:
            existing.add(tuple(sorted((c.rank, c.suit) for c in combo.cards)))

    new_combos: list[Combo] = []
    for rank, suit_map in by_rank_suit.items():
        suit_cards = sorted(suit_map.values(), key=lambda c: c.suit)
        for c1, c2 in _combinations(suit_cards, 2):
            key = tuple(sorted([(c1.rank, c1.suit), (c2.rank, c2.suit)]))
            if key not in existing:
                existing.add(key)
                new_combos.append(Combo(
                    ComboType.PAIR, rank, [c1, c2],
                    length=2, wild_count=0,
                ))

    return raw_moves + new_combos


def serialize_game_state(
    env: GuanDanEnv,
    game_id: str,
    human_seat: int,
    player_infos: list[dict],
    trick_plays: list[tuple[int, Combo]] | None = None,
    groups: list[dict] | None = None,
) -> dict:
    """Serialize full game state from the human player's perspective."""
    sorted_hand = sort_hand(env.hands[human_seat], env.level_rank)
    hand_dtos = [card_to_dto(c) for c in sorted_hand]

    # Mark wild cards
    for dto, card in zip(hand_dtos, sorted_hand):
        dto["is_wild"] = is_wild(card, env.level_rank)

    # Player infos with card counts
    players = []
    for i, info in enumerate(player_infos):
        card_count = len(env.hands[i])
        players.append({
            "seat": i,
            "name": info["name"],
            "avatar": info.get("avatar"),
            "elo": info.get("elo", 1200),
            "card_count": card_count,
            "is_out": env.is_out[i],
            "is_teammate": (i % 2) == (human_seat % 2),
            "is_human": i == human_seat,
        })

    # Per-seat trick actions (geometric layout)
    trick_actions: dict[str, dict | None] = {"0": None, "1": None, "2": None, "3": None}
    if trick_plays:
        for seat, combo in trick_plays:
            if combo.type == ComboType.PASS:
                trick_actions[str(seat)] = {"type": "pass"}
            else:
                trick_actions[str(seat)] = {
                    "type": "play",
                    "combo": combo_to_dto(combo, env.level_rank),
                }

    # Legal moves sorted smallest → biggest (only when it's the human's turn)
    legal_moves = []
    is_my_turn = env.current_player == human_seat and not env.done
    if is_my_turn:
        raw_moves = env.legal_moves(human_seat)
        if env.current_trick is None:
            raw_moves = _expand_natural_pairs(raw_moves, env.hands[human_seat], env.level_rank)
        non_pass = [c for c in raw_moves if c.type != ComboType.PASS]
        non_pass.sort(key=lambda c: combo_sort_key(c, env.level_rank))
        for combo in non_pass:
            legal_moves.append(combo_to_dto(combo, env.level_rank))
        # Add pass at the end if it was in the original list
        if any(c.type == ComboType.PASS for c in raw_moves):
            legal_moves.append(combo_to_dto(
                next(c for c in raw_moves if c.type == ComboType.PASS),
                env.level_rank,
            ))

    return {
        "game_id": game_id,
        "my_seat": human_seat,
        "current_player": env.current_player,
        "is_my_turn": is_my_turn,
        "my_hand": hand_dtos,
        "players": players,
        "trick_actions": trick_actions,
        "is_leading": env.is_leading(),
        "legal_moves": legal_moves,
        "finish_order": env.finish_order,
        "done": env.done,
        "level_rank": env.level_rank,
        "rewards": env.get_rewards() if env.done else None,
        "groups": groups or [],
    }
