"""Hand structure graph for GNN-based Q-network.

Builds a graph from a player's hand where nodes are cards and edges encode
structural relationships (same rank, consecutive, same suit, wild bridges).
The GNN learns card combination patterns (bombs, straights, etc.) that flat
encodings miss.
"""

from __future__ import annotations

import numpy as np
import torch

from ..cards import Rank, is_wild, level_order_key

D_NODE = 23  # per-card feature dimension
D_EDGE = 8   # per-edge feature dimension

# Edge type indices (one-hot in first 5 dims of edge features)
EDGE_SAME_RANK = 0        # pair/triple/bomb potential
EDGE_CONSECUTIVE = 1      # straight potential
EDGE_SAME_SUIT_CONSEC = 2 # straight flush potential
EDGE_WILD_BRIDGE = 3      # wild substitution
EDGE_SAME_SUIT = 4        # same suit (weaker signal)


def encode_card_node(card, level_rank: int) -> np.ndarray:
    """Per-card features. 23 dims.

    [0:15]  rank one-hot (2..A=0..12, BJ=13, RJ=14)
    [15:19] suit one-hot (spade, heart, diamond, club)
    [19]    is_wild flag
    [20]    is_joker flag
    [21]    is_level_card flag
    [22]    normalized strength (level_order_key / 17)
    """
    feat = np.zeros(D_NODE, dtype=np.float32)

    # Rank one-hot
    if card.rank <= 14:
        feat[card.rank - 2] = 1.0
    elif card.rank == Rank.BLACK_JOKER:
        feat[13] = 1.0
    elif card.rank == Rank.RED_JOKER:
        feat[14] = 1.0

    # Suit one-hot (only for normal cards)
    if card.rank <= 14:
        feat[15 + card.suit] = 1.0

    # Flags
    feat[19] = float(is_wild(card, level_rank))
    feat[20] = float(card.rank >= Rank.BLACK_JOKER)
    feat[21] = float(card.rank == level_rank)
    feat[22] = level_order_key(card.rank, level_rank) / 17.0

    return feat


def build_hand_graph(hand_cards: list, level_rank: int):
    """Build graph from a list of Card objects.

    Args:
        hand_cards: list of Card objects (order matters for mask indexing)
        level_rank: current level rank for wild detection

    Returns:
        node_features:  [N, 23] float tensor
        edge_index:     [2, E]  long tensor (COO, bidirectional)
        edge_features:  [E, 8]  float tensor
    """
    N = len(hand_cards)
    if N == 0:
        return (torch.zeros(1, D_NODE),
                torch.zeros(2, 0, dtype=torch.long),
                torch.zeros(0, D_EDGE))

    # Node features
    node_feats = np.stack([encode_card_node(c, level_rank) for c in hand_cards])

    # Edges (bidirectional)
    src, dst, edge_feats = [], [], []
    for i in range(N):
        for j in range(i + 1, N):
            edges = _compute_edges(hand_cards[i], hand_cards[j], level_rank)
            for ef in edges:
                src.extend([i, j])
                dst.extend([j, i])
                edge_feats.extend([ef, ef])

    node_features = torch.tensor(node_feats, dtype=torch.float32)

    if not src:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_features = torch.zeros(0, D_EDGE)
    else:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_features = torch.tensor(np.stack(edge_feats), dtype=torch.float32)

    return node_features, edge_index, edge_features


def _compute_edges(card_a, card_b, level_rank: int) -> list[np.ndarray]:
    """Compute all edge types between two cards. Returns list of [8] arrays."""
    edges = []
    a_rank, b_rank = card_a.rank, card_b.rank
    a_suit, b_suit = card_a.suit, card_b.suit

    # Joker-joker: same-rank edge if identical rank (BJ-BJ or RJ-RJ)
    if a_rank >= Rank.BLACK_JOKER and b_rank >= Rank.BLACK_JOKER:
        if a_rank == b_rank:
            edges.append(_make_edge(EDGE_SAME_RANK, 0, False))
        return edges

    # Joker-normal: no edges
    if a_rank >= Rank.BLACK_JOKER or b_rank >= Rank.BLACK_JOKER:
        return edges

    rank_gap = abs(a_rank - b_rank)
    same_suit = (a_suit == b_suit)

    # Same rank → pair/triple/bomb potential
    if a_rank == b_rank:
        edges.append(_make_edge(EDGE_SAME_RANK, 0, same_suit))

    # Consecutive or within straight range (gap 1-4)
    effective_gap = rank_gap
    if effective_gap == 0:
        pass  # already handled above
    elif 1 <= effective_gap <= 4:
        edges.append(_make_edge(EDGE_CONSECUTIVE, effective_gap, same_suit))
        if same_suit:
            edges.append(_make_edge(EDGE_SAME_SUIT_CONSEC, effective_gap, same_suit))

    # Wild bridge: wild card can substitute in combos
    if is_wild(card_a, level_rank) or is_wild(card_b, level_rank):
        edges.append(_make_edge(EDGE_WILD_BRIDGE, min(effective_gap, 5), False))

    return edges


def _make_edge(edge_type: int, rank_gap: int, same_suit: bool) -> np.ndarray:
    """Edge feature vector. 8 dims.

    [0:5]  edge type one-hot
    [5]    normalized rank gap
    [6]    same suit flag
    [7]    closeness signal (1 / (1 + gap))
    """
    feat = np.zeros(D_EDGE, dtype=np.float32)
    feat[edge_type] = 1.0
    feat[5] = rank_gap / 12.0
    feat[6] = float(same_suit)
    feat[7] = 1.0 / (1.0 + rank_gap)
    return feat


def encode_action_subgraph(combo_cards, hand_cards_list: list):
    """Compute action and remaining masks for a combo played from a hand.

    Args:
        combo_cards: iterable of Card objects (the played combo)
        hand_cards_list: list of Card objects (same order as graph nodes)

    Returns:
        action_mask:    [N] float tensor — 1.0 for played cards
        remaining_mask: [N] float tensor — 1.0 for remaining cards
    """
    N = len(hand_cards_list)
    action_mask = torch.zeros(N)
    remaining_mask = torch.zeros(N)

    played_ids = {(c.rank, c.suit, c.deck) for c in combo_cards}
    used = set()

    for i, card in enumerate(hand_cards_list):
        card_id = (card.rank, card.suit, card.deck)
        if card_id in played_ids and card_id not in used:
            action_mask[i] = 1.0
            used.add(card_id)
        else:
            remaining_mask[i] = 1.0

    return action_mask, remaining_mask
