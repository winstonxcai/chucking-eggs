from __future__ import annotations

import numpy as np

from guandan.cards import ComboType
from guandan.combos import Combo
from guandan.game import GuanDanEnv
from guandan.guanzero.encoding.base_encoder import StateActionEncoder
from guandan.guanzero.encoding.role_encoder import (
    REL_PARTNER,
    REL_SELF,
    ROLE_ENCODE_CHANNEL_SHAPES,
    ROLE_ENCODE_TRICK_CHANNEL_SHAPES,
    RoleAwareStateActionEncoder,
    absolute_player,
    relative_role,
    trick_head_id,
)


def _fresh_env(seed: int = 0) -> GuanDanEnv:
    env = GuanDanEnv()
    env.reset(seed=seed)
    return env


def _non_pass_legal(env: GuanDanEnv) -> Combo:
    for combo in env.legal_moves(env.current_player):
        if combo.type != ComboType.PASS:
            return combo
    raise AssertionError("expected a non-pass legal move")


def test_relative_role_identity_and_partner():
    for actor in range(4):
        assert relative_role(actor, actor) == REL_SELF
        assert relative_role((actor + 2) % 4, actor) == REL_PARTNER


def test_relative_role_full_table():
    expected = {
        0: [0, 1, 2, 3],
        1: [3, 0, 1, 2],
        2: [2, 3, 0, 1],
        3: [1, 2, 3, 0],
    }
    for actor in range(4):
        assert [relative_role(abs_player, actor) for abs_player in range(4)] == expected[actor]


def test_absolute_player_is_inverse():
    for actor in range(4):
        for role in range(4):
            assert relative_role(absolute_player(actor, role), actor) == role


def test_encoder_output_keys_shapes_and_dtypes():
    env = _fresh_env()
    p = env.current_player
    legal = env.legal_moves(p)
    out = RoleAwareStateActionEncoder().encode_all(env, p, legal)[0]

    assert set(out) == set(ROLE_ENCODE_CHANNEL_SHAPES)
    for key, shape in ROLE_ENCODE_CHANNEL_SHAPES.items():
        assert out[key].shape == shape
        expected_dtype = np.int64 if key == "seat_id" else np.uint8
        assert out[key].dtype == expected_dtype


def test_encode_one_matches_encode_all_for_selected_action():
    env = _fresh_env(seed=3)
    p = env.current_player
    legal = env.legal_moves(p)
    idx = min(2, len(legal) - 1)
    enc = RoleAwareStateActionEncoder()

    one = enc.encode_one(env, p, legal[idx], legal)
    all_rows = enc.encode_all(env, p, legal)

    for key, value in one.items():
        assert np.array_equal(value, all_rows[idx][key]), key


def test_pass_history_preserves_actor_role_and_pass_bit():
    env = _fresh_env(seed=4)
    lead = env.current_player
    env.step(_non_pass_legal(env))
    passer = env.current_player
    env.step(Combo(ComboType.PASS, 0, []))

    p = env.current_player
    legal = env.legal_moves(p)
    out = RoleAwareStateActionEncoder().encode_all(env, p, legal)[0]
    row = -1

    assert out["history_actions"][row].sum() == 0
    assert out["history_roles"][row, relative_role(passer, p)] == 1
    assert out["history_is_pass"][row, 0] == 1
    assert out["history_roles"][-2, relative_role(lead, p)] == 1
    assert out["history_is_pass"][-2, 0] == 0
    assert out["history_actions"][-2].sum() > 0


def test_own_hand_matches_self_hand():
    env = _fresh_env(seed=5)
    p = env.current_player
    legal = env.legal_moves(p)
    out = RoleAwareStateActionEncoder(is_partner_visible=True).encode_all(env, p, legal)[0]
    assert int(out["own_hand"].sum()) == len(env.hands[p])


def test_partner_hand_filled_when_visible():
    env = _fresh_env(seed=5)
    p = env.current_player
    legal = env.legal_moves(p)
    partner_seat = (p + 2) % 4
    out = RoleAwareStateActionEncoder(is_partner_visible=True).encode_all(env, p, legal)[0]
    assert int(out["partner_hand"].sum()) == len(env.hands[partner_seat])


def test_partner_hand_zero_when_hidden():
    env = _fresh_env(seed=6)
    p = env.current_player
    legal = env.legal_moves(p)
    out = RoleAwareStateActionEncoder(is_partner_visible=False).encode_all(env, p, legal)[0]
    assert int(out["partner_hand"].sum()) == 0
    assert int(out["own_hand"].sum()) == len(env.hands[p])


def test_others_hand_is_union_of_two_opponents():
    env = _fresh_env(seed=5)
    p = env.current_player
    legal = env.legal_moves(p)
    next_opp = (p + 1) % 4
    prev_opp = (p + 3) % 4
    expected = np.zeros(108, dtype=np.uint8)
    for seat in (next_opp, prev_opp):
        for c in env.hands[seat]:
            from guandan.cards import card_to_id
            expected[card_to_id(c)] = 1
    out = RoleAwareStateActionEncoder(is_partner_visible=True).encode_all(env, p, legal)[0]
    assert np.array_equal(out["others_hand"], expected)


def test_player_blocks_carries_public_info_only():
    env = _fresh_env(seed=7)
    p = env.current_player
    legal = env.legal_moves(p)
    out = RoleAwareStateActionEncoder().encode_all(env, p, legal)[0]
    assert out["player_blocks"].shape == (4, 252)
    # No hand slots — only played, last_action, count, bomb-tier-histogram remain.


def test_player_blocks_bombs_slot_is_zero_at_start():
    env = _fresh_env(seed=7)
    p = env.current_player
    legal = env.legal_moves(p)
    out = RoleAwareStateActionEncoder().encode_all(env, p, legal)[0]
    # Bomb histogram is at [243:252] per role; no bombs played yet.
    assert int(out["player_blocks"][:, 243:252].sum()) == 0
    assert int(env.bombs_played.sum()) == 0


def test_m0_feature_parity_for_initial_state():
    env = _fresh_env(seed=8)
    p = env.current_player
    legal = env.legal_moves(p)
    role_out = RoleAwareStateActionEncoder().encode_all(env, p, legal)[0]
    base_out = StateActionEncoder().encode_all(env, p, legal)[0]

    assert np.array_equal(role_out["own_hand"], base_out["own_hand"].astype(np.uint8))
    assert np.array_equal(role_out["global_features"], base_out["level"].astype(np.uint8))
    assert np.array_equal(role_out["history_actions"], base_out["history"].astype(np.uint8))
    assert np.array_equal(role_out["candidate_action"], base_out["candidate_action"].astype(np.uint8))


# ─── Trick-relative head scheme ───────────────────────────────────────


def test_absolute_seat_scheme_is_default():
    """Default head_scheme preserves legacy seat_id output verbatim."""
    enc = RoleAwareStateActionEncoder()
    assert enc.head_scheme == "absolute_seat"
    assert enc.head_field == "seat_id"
    assert enc.channel_shapes["player_blocks"] == (4, 252)


def test_trick_scheme_advertises_trick_head_id():
    enc = RoleAwareStateActionEncoder(head_scheme="trick_relative")
    assert enc.head_field == "trick_head_id"
    assert enc.channel_shapes["player_blocks"] == (4, 256)
    assert "seat_id" not in enc.channel_shapes
    assert "trick_head_id" in enc.channel_shapes


def test_trick_head_id_when_leading_is_zero():
    """No current trick → leader → trick_head_id == 0."""
    env = _fresh_env(seed=0)
    p = env.current_player
    assert env.current_trick is None
    assert trick_head_id(env, p) == 0


def test_trick_head_id_after_first_play_is_one():
    """After leader plays, the next player to act is the first responder (head 1)."""
    env = _fresh_env(seed=4)
    env.step(_non_pass_legal(env))
    assert env.current_trick is not None
    p = env.current_player
    # First responder is leader - 1 (counterclockwise)
    leader = env.trick_winner
    assert (leader - p) % 4 == 1
    assert trick_head_id(env, p) == 1


def test_trick_head_id_full_response_cycle():
    """Three responders after the leader → head ids 1, 2, 3."""
    env = _fresh_env(seed=4)
    env.step(_non_pass_legal(env))
    seen_head_ids: list[int] = []
    leader = env.trick_winner
    for _ in range(3):
        p = env.current_player
        seen_head_ids.append(trick_head_id(env, p))
        # Pass so we don't change the trick winner — just walk through responders.
        env.step(Combo(ComboType.PASS, 0, []))
        if env.current_trick is None:
            break
        # After last_pass that ends the trick, current_trick is None; we want to
        # observe just the 3 in-trick responders, so break if we've exited.
    # The first three responders must be head ids 1, 2, 3 (cyclic order).
    assert seen_head_ids[:3] == [1, 2, 3], seen_head_ids
    # Sanity: the leader is unchanged across the pass cycle.
    # (After 3 passes the trick ends and leader becomes current_player again.)
    assert leader == env.current_player


def test_trick_head_id_after_pass_then_play():
    """Pass doesn't change trick_winner; head ids stay relative to original leader."""
    env = _fresh_env(seed=4)
    env.step(_non_pass_legal(env))
    leader = env.trick_winner
    env.step(Combo(ComboType.PASS, 0, []))
    # The next actor is still in the trick (passes don't end it unless 3 consecutive)
    p = env.current_player
    assert env.current_trick is not None
    assert trick_head_id(env, p) == (leader - p) % 4
    assert trick_head_id(env, p) == 2  # second responder


def test_trick_head_id_resets_after_trick_ends():
    """After _end_trick clears current_trick, the new actor is head 0 (leader)."""
    env = _fresh_env(seed=4)
    env.step(_non_pass_legal(env))
    # Three passes end the trick.
    env.step(Combo(ComboType.PASS, 0, []))
    env.step(Combo(ComboType.PASS, 0, []))
    env.step(Combo(ComboType.PASS, 0, []))
    assert env.current_trick is None
    p = env.current_player
    assert trick_head_id(env, p) == 0


def test_trick_encoder_outputs_trick_head_id_field():
    env = _fresh_env(seed=5)
    p = env.current_player
    legal = env.legal_moves(p)
    enc = RoleAwareStateActionEncoder(head_scheme="trick_relative")
    out = enc.encode_all(env, p, legal)[0]

    assert "seat_id" not in out
    assert "trick_head_id" in out
    assert out["trick_head_id"].dtype == np.int64
    # No trick yet → leader → 0
    assert int(out["trick_head_id"]) == 0


def test_trick_encoder_player_blocks_width_is_256():
    env = _fresh_env(seed=5)
    p = env.current_player
    legal = env.legal_moves(p)
    enc = RoleAwareStateActionEncoder(head_scheme="trick_relative")
    out = enc.encode_all(env, p, legal)[0]

    assert out["player_blocks"].shape == (4, 256)


def test_trick_encoder_trick_pos_onehot_in_player_blocks():
    """When trick_leader is on role r from actor's view, each role's [252:256]
    slot is a one-hot at position (leader_role - role) % 4."""
    env = _fresh_env(seed=4)
    env.step(_non_pass_legal(env))
    leader = env.trick_winner
    p = env.current_player

    enc = RoleAwareStateActionEncoder(head_scheme="trick_relative")
    out = enc.encode_all(env, p, legal_moves=env.legal_moves(p))[0]
    blocks = out["player_blocks"]

    leader_role = relative_role(leader, p)
    for role in range(4):
        trick_pos = (leader_role - role) % 4
        slot = blocks[role, 252:256]
        assert slot.sum() == 1, f"role {role} trick-pos slot should be one-hot, got {slot}"
        assert slot[trick_pos] == 1, (
            f"role {role}: expected trick_pos {trick_pos}, got {slot}"
        )


def test_trick_encoder_trick_pos_zero_when_no_trick():
    """When current_trick is None, the trick-pos slot per role is all-zero."""
    env = _fresh_env(seed=0)
    p = env.current_player
    legal = env.legal_moves(p)
    assert env.current_trick is None

    enc = RoleAwareStateActionEncoder(head_scheme="trick_relative")
    out = enc.encode_all(env, p, legal)[0]
    assert out["player_blocks"][:, 252:256].sum() == 0


def test_trick_encoder_other_features_match_absolute_scheme_at_start():
    """At game start, the state channels match between the two head schemes
    (only player_blocks shape and the head-id field differ)."""
    env = _fresh_env(seed=8)
    p = env.current_player
    legal = env.legal_moves(p)
    abs_out = RoleAwareStateActionEncoder().encode_all(env, p, legal)[0]
    trick_out = RoleAwareStateActionEncoder(head_scheme="trick_relative").encode_all(env, p, legal)[0]

    # Shared state channels must be identical.
    for k in ("own_hand", "partner_hand", "others_hand", "global_features",
              "behavior", "history_actions", "history_roles", "history_is_pass"):
        assert np.array_equal(abs_out[k], trick_out[k]), f"mismatch on {k}"

    # player_blocks: first 252 dims must match; trick block has extra 4 dims.
    assert np.array_equal(abs_out["player_blocks"], trick_out["player_blocks"][:, :252])
