from __future__ import annotations

import dataclasses
import tempfile
from pathlib import Path

import torch

from guandan.combos import Combo
from guandan.game import GuanDanEnv
from guandan.dart.agent import DartBot
from guandan.dart.config import MODEL_TYPE_GUANZERO, QNetConfig, TrainConfig
from guandan.dart.model.q_network import init_guanzero_nets


def _save_dummy_checkpoint(path: Path) -> None:
    qnet = QNetConfig(hidden_lstm=16, hidden_mlp=32, n_mlp_layers=2)
    cfg = TrainConfig(model_type=MODEL_TYPE_GUANZERO, qnet=qnet)
    q_nets = init_guanzero_nets(cfg.qnet)
    torch.save(
        {
            "episode": 0,
            "config": dataclasses.asdict(cfg),
            "q_nets": {p: q_nets[p].state_dict() for p in range(4)},
        },
        path,
    )


def test_load_and_act_returns_legal_combo():
    with tempfile.TemporaryDirectory() as td:
        ckpt = Path(td) / "ckpt.pt"
        _save_dummy_checkpoint(ckpt)
        bot = DartBot.load(ckpt)

        env = GuanDanEnv()
        env.reset(seed=42)
        action = bot.act(env, env.current_player)

        assert isinstance(action, Combo)
        legal = env.legal_moves(env.current_player)
        my_key = (action.type, action.key, tuple(sorted((c.rank, c.suit, c.deck) for c in action.cards)))
        legal_keys = {
            (m.type, m.key, tuple(sorted((c.rank, c.suit, c.deck) for c in m.cards)))
            for m in legal
        }
        assert my_key in legal_keys
