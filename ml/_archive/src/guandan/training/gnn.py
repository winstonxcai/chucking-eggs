"""GATv2 graph neural network for hand structure encoding.

Pure PyTorch implementation — no PyG dependency. Graphs are small enough
(≤27 nodes, ~150 edges) that scatter ops are fast without optimization.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hand_graph import D_EDGE, D_NODE


class GATv2Layer(nn.Module):
    """GATv2 attention layer with edge features.

    Attention: a^T LeakyReLU(W_src·x_i + W_dst·x_j + W_edge·e_ij)
    """

    def __init__(self, d_in: int, d_out: int, d_edge: int = D_EDGE,
                 n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_out // n_heads
        assert d_out % n_heads == 0

        self.W_src = nn.Linear(d_in, d_out, bias=False)
        self.W_dst = nn.Linear(d_in, d_out, bias=False)
        self.W_edge = nn.Linear(d_edge, d_out, bias=False)
        self.attn = nn.Parameter(torch.randn(n_heads, self.d_head))
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.W_src.weight)
        nn.init.xavier_uniform_(self.W_dst.weight)
        nn.init.xavier_uniform_(self.W_edge.weight)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:          [N, d_in]
            edge_index: [2, E]
            edge_attr:  [E, d_edge]
        Returns:        [N, d_out]
        """
        N = x.size(0)
        E = edge_index.size(1)

        if E == 0:
            return F.elu(self.out_proj(self.W_src(x)))

        src, dst = edge_index[0], edge_index[1]

        h_src = self.W_src(x)[src]       # [E, d_out]
        h_dst = self.W_dst(x)[dst]       # [E, d_out]
        h_edge = self.W_edge(edge_attr)  # [E, d_out]

        # GATv2: LeakyReLU BEFORE dot product with attention vector
        msg = F.leaky_relu(h_src + h_dst + h_edge, 0.2)
        msg = msg.view(E, self.n_heads, self.d_head)

        # Attention scores
        alpha = (msg * self.attn.unsqueeze(0)).sum(dim=-1)  # [E, n_heads]
        alpha = self._scatter_softmax(alpha, dst, N)
        alpha = self.dropout(alpha)

        # Weighted message aggregation
        msg_weighted = (msg * alpha.unsqueeze(-1)).view(E, -1)  # [E, d_out]
        out = torch.zeros(N, msg_weighted.size(1), device=x.device)
        out.scatter_add_(0, dst.unsqueeze(-1).expand_as(msg_weighted), msg_weighted)

        return F.elu(self.out_proj(out))

    def _scatter_softmax(self, scores: torch.Tensor, index: torch.Tensor,
                         N: int) -> torch.Tensor:
        """Softmax over groups defined by scatter index."""
        max_scores = torch.full((N, scores.size(1)), -1e9, device=scores.device)
        max_scores.scatter_reduce_(
            0, index.unsqueeze(-1).expand_as(scores),
            scores, reduce='amax', include_self=True)
        scores = scores - max_scores[index]
        exp_scores = scores.exp()
        sum_exp = torch.zeros(N, scores.size(1), device=scores.device)
        sum_exp.scatter_add_(
            0, index.unsqueeze(-1).expand_as(exp_scores), exp_scores)
        return exp_scores / (sum_exp[index] + 1e-8)


class HandGNN(nn.Module):
    """2-layer GATv2 over hand structure graph.

    Input: variable-size card graph (N nodes, E edges)
    Output: 3 fixed-size embeddings:
      hand_emb    [d_out] — attentive readout of full hand
      action_emb  [d_out] — pooled over played cards
      remain_emb  [d_out] — pooled over remaining cards
    """

    def __init__(self, d_node: int = D_NODE, d_edge: int = D_EDGE,
                 d_hidden: int = 64, d_out: int = 128, n_heads: int = 4):
        super().__init__()
        self.d_hidden = d_hidden
        self.d_out = d_out

        self.gat1 = GATv2Layer(d_node, d_hidden, d_edge, n_heads=n_heads)
        self.gat2 = GATv2Layer(d_hidden, d_hidden, d_edge, n_heads=n_heads)

        # Attentive readout for full hand
        self.readout_gate = nn.Sequential(
            nn.Linear(d_hidden, 1),
            nn.Sigmoid()
        )
        self.readout_proj = nn.Linear(d_hidden, d_out)

        # Action/remaining projections
        self.action_proj = nn.Linear(d_hidden, d_out)
        self.remain_proj = nn.Linear(d_hidden, d_out)

    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor,
                edge_features: torch.Tensor,
                action_mask: torch.Tensor | None = None,
                remaining_mask: torch.Tensor | None = None,
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward: GNN + readout + masked pooling.

        Returns: (hand_emb [d_out], action_emb [d_out], remain_emb [d_out])
        """
        h = self.gat1(node_features, edge_index, edge_features)
        h = self.gat2(h, edge_index, edge_features)

        # Attentive readout
        gate = self.readout_gate(h)
        hand_emb = self.readout_proj((h * gate).sum(dim=0))

        # Masked pooling
        if action_mask is not None and action_mask.sum() > 0:
            action_emb = self.action_proj(
                (h * action_mask.unsqueeze(-1)).sum(dim=0))
        else:
            action_emb = torch.zeros(self.d_out, device=h.device)

        if remaining_mask is not None and remaining_mask.sum() > 0:
            remain_emb = self.remain_proj(
                (h * remaining_mask.unsqueeze(-1)).sum(dim=0))
        else:
            remain_emb = torch.zeros(self.d_out, device=h.device)

        return hand_emb, action_emb, remain_emb

    def encode_nodes(self, node_features: torch.Tensor, edge_index: torch.Tensor,
                     edge_features: torch.Tensor,
                     ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run GNN only — return node embeddings for amortized use.

        Called ONCE per decision. Then pool_action() per legal move.

        Returns: (h [N, d_hidden], hand_emb [d_out])
        """
        h = self.gat1(node_features, edge_index, edge_features)
        h = self.gat2(h, edge_index, edge_features)

        gate = self.readout_gate(h)
        hand_emb = self.readout_proj((h * gate).sum(dim=0))

        return h, hand_emb

    def pool_action(self, h: torch.Tensor, action_mask: torch.Tensor,
                    remaining_mask: torch.Tensor,
                    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Masked pooling from pre-computed node embeddings.

        Called B times (once per legal action) — cheap.
        """
        if action_mask.sum() > 0:
            action_emb = self.action_proj(
                (h * action_mask.unsqueeze(-1)).sum(dim=0))
        else:
            action_emb = torch.zeros(self.d_out, device=h.device)

        if remaining_mask.sum() > 0:
            remain_emb = self.remain_proj(
                (h * remaining_mask.unsqueeze(-1)).sum(dim=0))
        else:
            remain_emb = torch.zeros(self.d_out, device=h.device)

        return action_emb, remain_emb
