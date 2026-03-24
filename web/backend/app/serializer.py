"""Serialize GuanDanEnv state to JSON for the frontend."""

from __future__ import annotations

from guandan.cards import Card, ComboType, Rank, Suit, is_wild, level_order_key
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
    return {
        "rank": card.rank,
        "suit": card.suit,
        "deck": card.deck,
        "id": f"{card.rank}-{card.suit}-{card.deck}",
        "display": f"{RANK_NAMES.get(card.rank, '?')}{SUIT_SYMBOLS.get(card.suit, '?')}",
        "rank_display": RANK_NAMES.get(card.rank, "?"),
        "suit_symbol": SUIT_SYMBOLS.get(card.suit, "?"),
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


def sort_hand(cards: set[Card], level_rank: int) -> list[Card]:
    """Sort hand by level order, then suit, then deck."""
    def sort_key(c: Card):
        if c.rank in (Rank.BLACK_JOKER, Rank.RED_JOKER):
            return (100 + c.rank, c.suit, c.deck)
        if is_wild(c, level_rank):
            return (99, c.suit, c.deck)
        return (level_order_key(c.rank, level_rank), c.suit, c.deck)
    return sorted(cards, key=sort_key)


def serialize_game_state(
    env: GuanDanEnv,
    game_id: str,
    human_seat: int,
    player_infos: list[dict],
) -> dict:
    """Serialize full game state from the human player's perspective."""
    sorted_hand = sort_hand(env.hands[human_seat], env.level_rank)
    hand_dtos = [card_to_dto(c) for c in sorted_hand]

    # Mark wild cards
    for dto, card in zip(hand_dtos, sorted_hand):
        dto["is_wild"] = is_wild(card, env.level_rank)

    # Player infos with card counts and last actions
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

    # Current trick plays
    current_trick = None
    if env.trick_winner is not None:
        # Gather recent plays for this trick from move_history
        trick_plays = []
        for seat, combo in reversed(env.move_history):
            trick_plays.insert(0, {
                "seat": seat,
                "player_name": player_infos[seat]["name"],
                "combo": combo_to_dto(combo, env.level_rank),
                "is_pass": combo.type == ComboType.PASS,
            })
            # Stop when we hit the trick leader (the first non-pass play
            # after the last trick reset)
            if combo.type != ComboType.PASS and seat == env.trick_winner and len(trick_plays) > 0:
                # Check if this is actually the start
                break
        current_trick = {"plays": trick_plays, "leader": env.trick_winner}

    # Legal moves (only when it's the human's turn)
    legal_moves = []
    is_my_turn = env.current_player == human_seat and not env.done
    if is_my_turn:
        for combo in env.legal_moves(human_seat):
            legal_moves.append(combo_to_dto(combo, env.level_rank))

    return {
        "game_id": game_id,
        "my_seat": human_seat,
        "current_player": env.current_player,
        "is_my_turn": is_my_turn,
        "my_hand": hand_dtos,
        "players": players,
        "current_trick": current_trick,
        "is_leading": env.is_leading(),
        "legal_moves": legal_moves,
        "finish_order": env.finish_order,
        "done": env.done,
        "level_rank": env.level_rank,
        "rewards": env.get_rewards() if env.done else None,
    }
