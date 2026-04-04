"""RL Q-network action recommender for LLMBot.

Loads the trained RL checkpoint once (lazy singleton) and scores candidate
moves using the same Q-network used by RLAgentLSTM.

Higher Q-value = better move.
Returns empty dict on any failure — callers fall back to static scoring.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from ..combos import Combo
from ..training.encoding import encode_action, encode_history, encode_state
from ..training.q_network import QNetworkLSTM, get_device

if TYPE_CHECKING:
    from ..game import GuanDanEnv

# Checkpoint: prod_03_29_11_51.pt — Elo #1 (1786), no-GNN, lstm_hidden=256, hidden=1024
# Keys: 'lead' / 'follow'
_DEFAULT_CKPT = Path(__file__).resolve().parents[5] / "checkpoints" / "prod_03_29_11_51.pt"

_q_lead: QNetworkLSTM | None = None
_q_follow: QNetworkLSTM | None = None
_device = None


def _load_models(ckpt_path: Path) -> tuple[QNetworkLSTM, QNetworkLSTM, torch.device]:
    global _q_lead, _q_follow, _device
    if _q_lead is not None:
        return _q_lead, _q_follow, _device  # type: ignore[return-value]

    _device = get_device()
    kwargs = dict(lstm_hidden=256, hidden=1024, use_gnn=False)
    _q_lead = QNetworkLSTM(**kwargs).to(_device)
    _q_follow = QNetworkLSTM(**kwargs).to(_device)

    ckpt = torch.load(ckpt_path, map_location=_device, weights_only=True)
    _q_lead.load_state_dict(ckpt["lead"], strict=False)
    _q_follow.load_state_dict(ckpt["follow"], strict=False)
    _q_lead.eval()
    _q_follow.eval()
    return _q_lead, _q_follow, _device


def score_candidates(
    env: GuanDanEnv,
    player: int,
    candidates: list[Combo],
    level_rank: int,
    ckpt_path: Path = _DEFAULT_CKPT,
) -> dict[int, float]:
    """Return RL Q-values for each candidate. Keys are id(combo).

    Returns empty dict on failure — caller falls back to static _jidan_score.
    """
    if not candidates or not ckpt_path.exists():
        return {}
    try:
        q_lead, q_follow, device = _load_models(ckpt_path)
        net = q_lead if env.current_trick is None else q_follow

        hand = env.hands[player]
        state_enc = encode_state(env, player)
        history, hist_len = encode_history(env, player, level_rank)

        import numpy as np
        actions_arr = [encode_action(combo, hand, level_rank) for combo in candidates]
        k = len(candidates)

        s = torch.from_numpy(np.stack([state_enc] * k)).float().to(device)       # [K, 417]
        a = torch.from_numpy(np.stack(actions_arr)).float().to(device)           # [K, 160]
        h = torch.from_numpy(np.stack([history] * k)).float().to(device)         # [K, T, 83]
        hl = torch.tensor([hist_len] * k, dtype=torch.long, device=device)       # [K]

        with torch.no_grad():
            q_vals = net(s, a, h, hl).cpu().tolist()                   # [K]

        return {id(combo): q for combo, q in zip(candidates, q_vals)}
    except Exception:
        return {}
