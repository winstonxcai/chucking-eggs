from __future__ import annotations

import numpy as np

from guandan.game import GuanDanEnv
from guandan.cards import CARD_ID_DIM, card_to_id, id_to_card
from guandan.guanzero.model.encoder import BEHAVIOR_DIM, HISTORY_LEN, LEVEL_DIM, RANK_BUCKETS
from guandan.guanzero.model.encoding.base_encoder import StateActionEncoder, static_dim


def _fresh_env(seed: int = 0) -> GuanDanEnv:
    env = GuanDanEnv()
    env.reset(seed=seed)
    return env


def test_card_id_roundtrip_is_bijective():
    seen = set()
    for cid in range(CARD_ID_DIM):
        c = id_to_card(cid)
        assert c is not None
        assert card_to_id(c) == cid
        seen.add(cid)
    assert len(seen) == CARD_ID_DIM


def test_channel_shapes_match_paper():
    env = _fresh_env()
    legal = env.legal_moves(env.current_player)
    out = StateActionEncoder().encode_all(env, env.current_player, legal)[0]

    expected = {
        "own_hand": (CARD_ID_DIM,),
        "others_hand": (CARD_ID_DIM,),
        "recent_action_each_player": (4, CARD_ID_DIM),
        "played_cards_others": (3, CARD_ID_DIM),
        "remaining_counts_others": (3, RANK_BUCKETS),
        "level": (LEVEL_DIM,),
        "history": (HISTORY_LEN, CARD_ID_DIM),
        "behavior": (BEHAVIOR_DIM,),
        "candidate_action": (CARD_ID_DIM,),
    }
    for k, shape in expected.items():
        assert out[k].shape == shape, f"{k}: {out[k].shape} != {shape}"
        assert out[k].dtype == np.float32


def test_static_dim_matches_actual_concat():
    env = _fresh_env()
    legal = env.legal_moves(env.current_player)
    out = StateActionEncoder().encode_all(env, env.current_player, legal)[0]
    flat = np.concatenate([
        out["own_hand"], out["others_hand"],
        out["recent_action_each_player"].reshape(-1),
        out["played_cards_others"].reshape(-1),
        out["remaining_counts_others"].reshape(-1),
        out["level"], out["behavior"], out["candidate_action"],
    ])
    assert flat.shape[0] == static_dim(is_partner_visible=True)


def test_partner_hidden_zeroes_others_hand():
    env = _fresh_env()
    p = env.current_player
    legal = env.legal_moves(p)
    enc = StateActionEncoder(is_partner_visible=False)
    out = enc.encode_all(env, p, legal)[0]
    assert np.all(out["others_hand"] == 0.0)
    assert out["own_hand"].sum() > 0


def test_partner_visible_fills_union_of_3_non_self_hands():
    env = _fresh_env()
    p = env.current_player
    legal = env.legal_moves(p)
    expected = np.zeros(108, dtype=np.uint8)
    for seat in range(4):
        if seat == p:
            continue
        for c in env.hands[seat]:
            expected[card_to_id(c)] = 1
    enc = StateActionEncoder(is_partner_visible=True)
    out = enc.encode_all(env, p, legal)[0]
    assert np.array_equal(out["others_hand"].astype(np.uint8), expected)


def test_encode_one_matches_encode_all_for_selected_action():
    env = _fresh_env(seed=3)
    p = env.current_player
    legal = env.legal_moves(p)
    idx = min(2, len(legal) - 1)
    enc = StateActionEncoder()

    one = enc.encode_one(env, p, legal[idx], legal)
    all_rows = enc.encode_all(env, p, legal)

    for key, value in one.items():
        assert np.array_equal(value, all_rows[idx][key]), key


def test_own_hand_count_matches_initial_deal():
    env = _fresh_env()
    p = env.current_player
    legal = env.legal_moves(p)
    out = StateActionEncoder().encode_all(env, p, legal)[0]
    assert int(out["own_hand"].sum()) == 27  # initial deal
