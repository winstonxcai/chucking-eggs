"""Tests for hand graph construction and GNN."""

import torch

from guandan.cards import Card, Rank
from guandan.training.hand_graph import (
    build_hand_graph,
    encode_action_subgraph,
    encode_card_node,
    D_NODE,
    D_EDGE,
)
from guandan.training.gnn import HandGNN


def test_encode_card_node_shape():
    c = Card(rank=5, suit=0, deck=0)
    feat = encode_card_node(c, level_rank=2)
    assert feat.shape == (D_NODE,)
    assert feat.dtype.name == "float32"


def test_encode_card_node_wild():
    # Heart of level rank is wild
    wild = Card(rank=2, suit=1, deck=0)  # 2 of hearts, level=2 → wild
    feat = encode_card_node(wild, level_rank=2)
    assert feat[19] == 1.0  # is_wild flag

    not_wild = Card(rank=2, suit=0, deck=0)  # 2 of spades → not wild
    feat2 = encode_card_node(not_wild, level_rank=2)
    assert feat2[19] == 0.0


def test_graph_construction_basic():
    # 7-card hand: 5♠ 6♠ 7♠ 7♥ 7♦ 8♠ 9♠
    hand = [
        Card(5, 0, 0), Card(6, 0, 0), Card(7, 0, 0),
        Card(7, 1, 0), Card(7, 2, 0),
        Card(8, 0, 0), Card(9, 0, 0),
    ]
    nf, ei, ef = build_hand_graph(hand, level_rank=2)

    assert nf.shape == (7, D_NODE)
    assert ei.shape[0] == 2
    assert ei.shape[1] > 0, "Should have edges"
    assert ef.shape == (ei.shape[1], D_EDGE)
    # Edges are bidirectional
    assert ei.shape[1] % 2 == 0


def test_graph_construction_empty():
    nf, ei, ef = build_hand_graph([], level_rank=2)
    assert nf.shape == (1, D_NODE)
    assert ei.shape == (2, 0)
    assert ef.shape == (0, D_EDGE)


def test_graph_has_same_rank_edges():
    # Three 7s should have same-rank edges
    hand = [Card(7, 0, 0), Card(7, 1, 0), Card(7, 2, 0)]
    nf, ei, ef = build_hand_graph(hand, level_rank=2)
    # 3 pairs × 2 directions = 6 same-rank edges (at minimum)
    assert ei.shape[1] >= 6


def test_graph_has_consecutive_edges():
    # 5, 6 are consecutive
    hand = [Card(5, 0, 0), Card(6, 0, 0)]
    nf, ei, ef = build_hand_graph(hand, level_rank=2)
    # Should have consecutive + same_suit_consec edges (both directions)
    assert ei.shape[1] >= 2


def test_action_subgraph_masks():
    hand = [Card(5, 0, 0), Card(6, 0, 0), Card(7, 0, 0), Card(8, 0, 0)]
    combo = [Card(5, 0, 0), Card(6, 0, 0)]  # play a pair-ish

    am, rm = encode_action_subgraph(combo, hand)
    assert am.shape == (4,)
    assert rm.shape == (4,)
    assert am.sum() == 2  # 2 cards played
    assert rm.sum() == 2  # 2 remaining
    assert (am + rm).sum() == 4  # all cards accounted for


def test_gnn_forward():
    gnn = HandGNN()
    N = 10
    nf = torch.randn(N, D_NODE)
    ei = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
    ef = torch.randn(4, D_EDGE)
    am = torch.zeros(N); am[:3] = 1.0
    rm = torch.zeros(N); rm[3:] = 1.0

    hand_emb, action_emb, remain_emb = gnn(nf, ei, ef, am, rm)

    assert hand_emb.shape == (128,)
    assert action_emb.shape == (128,)
    assert remain_emb.shape == (128,)


def test_gnn_no_edges():
    gnn = HandGNN()
    nf = torch.randn(5, D_NODE)
    ei = torch.zeros(2, 0, dtype=torch.long)
    ef = torch.zeros(0, D_EDGE)

    hand_emb, action_emb, remain_emb = gnn(nf, ei, ef)
    assert hand_emb.shape == (128,)


def test_amortized_pass():
    gnn = HandGNN()
    N = 15
    nf = torch.randn(N, D_NODE)
    ei = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    ef = torch.randn(2, D_EDGE)

    h, hand_emb = gnn.encode_nodes(nf, ei, ef)
    assert h.shape == (N, 64)
    assert hand_emb.shape == (128,)

    # Pool multiple actions
    for _ in range(5):
        am = torch.zeros(N); am[torch.randint(0, N, (3,))] = 1.0
        rm = 1.0 - am
        a_emb, r_emb = gnn.pool_action(h, am, rm)
        assert a_emb.shape == (128,)
        assert r_emb.shape == (128,)


def test_gnn_params():
    gnn = HandGNN()
    total = sum(p.numel() for p in gnn.parameters())
    assert 30_000 < total < 80_000, f"Expected ~50K params, got {total:,}"


def test_end_to_end_graph_to_gnn():
    """Full pipeline: cards → graph → GNN embeddings."""
    hand = [
        Card(3, 0, 0), Card(4, 0, 0), Card(5, 0, 0),
        Card(5, 1, 0), Card(5, 2, 0), Card(6, 0, 0),
        Card(7, 0, 0), Card(Rank.ACE, 0, 0), Card(Rank.ACE, 1, 0),
    ]
    nf, ei, ef = build_hand_graph(hand, level_rank=2)
    combo = [Card(5, 0, 0), Card(5, 1, 0), Card(5, 2, 0)]  # triple 5s
    am, rm = encode_action_subgraph(combo, hand)

    gnn = HandGNN()
    hand_emb, action_emb, remain_emb = gnn(nf, ei, ef, am, rm)

    assert hand_emb.shape == (128,)
    assert action_emb.shape == (128,)
    assert remain_emb.shape == (128,)

    # Gradients should flow
    loss = hand_emb.sum() + action_emb.sum() + remain_emb.sum()
    loss.backward()
    for p in gnn.parameters():
        if p.grad is not None:
            assert p.grad.abs().sum() > 0, "GNN should have nonzero gradients"
            break
