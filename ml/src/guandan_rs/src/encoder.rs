//! Rust encoder for GuanZero — absolute_seat scheme (Phase 2).
//!
//! Produces flat `f32` byte buffers from game state so the Python scorer only
//! needs `np.frombuffer(buf, dtype=np.float32)` followed by a NN forward pass.
//!
//! The layout mirrors `ROLE_ENCODE_STATE_KEYS` in `role_encoder.py` exactly.
//! Any deviation here will silently corrupt training features, so the constants
//! and channel ordering are pinned and tested.

use crate::cards::{card_to_id, PASS, TRIPLE};
use crate::combos::Combo;

// ─── Public dimension constants ──────────────────────────────────────────────

/// Number of unique card identifiers (2 decks × 13 ranks × 4 suits + 4 jokers).
pub const CARD_ID_DIM: usize = 108;

/// One-hot dimension for the level rank (ranks 2..=A, index = rank − 2).
pub const LEVEL_DIM: usize = 13;

/// One-hot dimension for remaining hand-size (0..=26, capped at 26).
pub const RANK_BUCKETS: usize = 27;

/// Number of bomb-tier histogram bins (BOMB_4 through BOMB_JOKER).
pub const BEHAVIOR_DIM: usize = 9;

/// Number of recent moves kept in the history window.
pub const HISTORY_LEN: usize = 20;

/// Width of a single player block in the absolute_seat encoding.
///
/// Layout: played(108) | last_nonpass(108) | hand_size_onehot(27) | bombs(9) = 252
pub const PLAYER_BLOCK_W: usize = 252;

/// Total number of f32 scalars in one encoded state vector.
///
/// Breakdown:
/// - own_hand        (108)
/// - partner_hand    (108)
/// - others_hand     (108)
/// - player_blocks   (4 × 252 = 1008)
/// - global_features (13)
/// - behavior        (9)
/// - history_actions (20 × 108 = 2160)
/// - history_roles   (20 × 4  =   80)
/// - history_is_pass (20 × 1  =   20)
///
/// Total = 108 + 108 + 108 + 1008 + 13 + 9 + 2160 + 80 + 20 = 3614
pub const STATE_DIM: usize = 3614;

/// Number of f32 scalars in one encoded action (candidate card multi-hot).
pub const ACTION_DIM: usize = 108;

// ─── Input struct ────────────────────────────────────────────────────────────

/// All game-state data required by the encoder for one decision step.
///
/// Slices reference pre-computed arrays owned by the rollout loop to avoid
/// per-step allocation.  The lifetimes ensure these references remain valid for
/// the duration of the encode call.
pub struct EncoderInput<'a> {
    /// Multi-hot card presence per seat (index into the 108-card id space).
    pub hand_multihot: &'a [[u8; 108]; 4],
    /// Cards played so far per seat, as a 108-dim multi-hot.
    pub played_multihot: &'a [[u8; 108]; 4],
    /// Most recent non-pass action per seat, as a 108-dim multi-hot.
    pub last_nonpass_multihot: &'a [[u8; 108]; 4],
    /// Per-seat bomb-tier histogram (9 bins, BOMB_4..=BOMB_JOKER).
    pub bombs_played: &'a [[u32; 9]; 4],
    /// Current number of cards in each seat's hand.
    pub hand_sizes: [usize; 4],
    /// Whether each seat has already gone out this round.
    pub is_out: [bool; 4],
    /// The current leading trick, or `None` when a new trick is starting.
    pub current_trick: Option<&'a Combo>,
    /// Seat that played the current trick, or `None` when leading.
    pub trick_winner: Option<usize>,
    /// The active level rank (2 = Two … 14 = Ace).
    pub level_rank: u8,
    /// Chronological (seat, combo) pairs for the whole episode so far.
    pub move_history: &'a [(usize, Combo)],
}

// ─── Public API ──────────────────────────────────────────────────────────────

/// Encode the full game state for `player` into a flat `f32` byte buffer.
///
/// The buffer length is always `STATE_DIM * 4` bytes.  The caller converts it
/// with `np.frombuffer(buf, dtype=np.float32)`.
///
/// The layout matches `ROLE_ENCODE_STATE_KEYS` in `role_encoder.py`:
/// `own_hand | partner_hand | others_hand | player_blocks |
///  global_features | behavior | history_actions | history_roles | history_is_pass`
pub fn encode_state(input: &EncoderInput, player: usize, legal: &[Combo]) -> Vec<u8> {
    let mut floats: Vec<f32> = Vec::with_capacity(STATE_DIM);

    let partner = (player + 2) % 4;
    let next_opp = (player + 1) % 4;
    let prev_opp = (player + 3) % 4;

    // ── own_hand (108) ──────────────────────────────────────────────────────
    for i in 0..CARD_ID_DIM {
        floats.push(input.hand_multihot[player][i] as f32);
    }

    // ── partner_hand (108) ──────────────────────────────────────────────────
    for i in 0..CARD_ID_DIM {
        floats.push(input.hand_multihot[partner][i] as f32);
    }

    // ── others_hand (108) — bitwise-OR of next_opp and prev_opp multihots ──
    for i in 0..CARD_ID_DIM {
        let v = (input.hand_multihot[next_opp][i] | input.hand_multihot[prev_opp][i]) as f32;
        floats.push(v);
    }

    // ── player_blocks (4 × 252 = 1008) ─────────────────────────────────────
    // roles 0..3 from player's perspective: 0=self, 1=next, 2=partner, 3=prev
    for role in 0..4usize {
        let seat = (player + role) % 4;
        write_player_block(input, seat, &mut floats);
    }

    // ── global_features (13) — level rank one-hot ───────────────────────────
    {
        let mut level_onehot = [0.0f32; LEVEL_DIM];
        // rank 2 → index 0, rank 14 (Ace) → index 12
        let idx = (input.level_rank as usize).saturating_sub(2).min(LEVEL_DIM - 1);
        level_onehot[idx] = 1.0;
        floats.extend_from_slice(&level_onehot);
    }

    // ── behavior (9) ────────────────────────────────────────────────────────
    let behavior = compute_behavior_flags(input, player, legal);
    floats.extend_from_slice(&behavior);

    // ── history (history_actions | history_roles | history_is_pass) ─────────
    // history_actions: (20, 108)
    // history_roles:   (20, 4)
    // history_is_pass: (20, 1)
    let mut h_actions = [[0.0f32; CARD_ID_DIM]; HISTORY_LEN];
    let mut h_roles = [[0.0f32; 4]; HISTORY_LEN];
    let mut h_is_pass = [0.0f32; HISTORY_LEN];

    let hist = input.move_history;
    let n = hist.len();
    // Take at most the last HISTORY_LEN entries; zero-pad at the front so that
    // the most recent entry always appears in the last row (matching Python's
    // `offset = HISTORY_LEN - len(recent)` convention).
    let start = if n > HISTORY_LEN { n - HISTORY_LEN } else { 0 };
    let entries = &hist[start..];
    let row_offset = HISTORY_LEN - entries.len();
    for (i, &(actor, ref combo)) in entries.iter().enumerate() {
        let row = row_offset + i;
        let role = (actor + 4 - player) % 4;
        h_roles[row][role] = 1.0;
        if combo.combo_type == PASS {
            h_is_pass[row] = 1.0;
        } else {
            for card in &combo.cards {
                let id = card_to_id(card);
                if id < CARD_ID_DIM {
                    h_actions[row][id] = 1.0;
                }
            }
        }
    }

    // Write in the required order: actions first, then roles, then is_pass.
    for row in 0..HISTORY_LEN {
        floats.extend_from_slice(&h_actions[row]);
    }
    for row in 0..HISTORY_LEN {
        floats.extend_from_slice(&h_roles[row]);
    }
    for row in 0..HISTORY_LEN {
        floats.push(h_is_pass[row]);
    }

    debug_assert_eq!(floats.len(), STATE_DIM);

    // Reinterpret the f32 slice as raw bytes (little-endian on all supported
    // platforms — matches numpy's default on x86/arm64).
    floats_to_bytes(&floats)
}

/// Encode all candidate legal actions as a flat `f32` byte buffer.
///
/// Each action is a 108-dim card multi-hot.  Buffer length = `ACTION_DIM * 4`
/// per action, so the total is `legal.len() * ACTION_DIM * 4` bytes.
pub fn encode_actions(legal: &[Combo]) -> Vec<u8> {
    let mut floats: Vec<f32> = Vec::with_capacity(legal.len() * ACTION_DIM);
    for combo in legal {
        let mut action = [0.0f32; ACTION_DIM];
        if combo.combo_type != PASS {
            for card in &combo.cards {
                let id = card_to_id(card);
                if id < ACTION_DIM {
                    action[id] = 1.0;
                }
            }
        }
        floats.extend_from_slice(&action);
    }
    floats_to_bytes(&floats)
}

/// Return the head id for `player` under the absolute_seat scheme.
///
/// In the absolute_seat scheme the head id is simply the player's seat index.
pub fn compute_head_id(player: usize) -> i64 {
    player as i64
}

// ─── Internal helpers ────────────────────────────────────────────────────────

/// Append one player block (252 f32s) to `out`.
///
/// Layout: played(108) | last_nonpass(108) | hand_size_onehot(27) | bombs(9)
fn write_player_block(input: &EncoderInput, seat: usize, out: &mut Vec<f32>) {
    // played_multihot (108)
    for i in 0..CARD_ID_DIM {
        out.push(input.played_multihot[seat][i] as f32);
    }
    // last_nonpass_multihot (108)
    for i in 0..CARD_ID_DIM {
        out.push(input.last_nonpass_multihot[seat][i] as f32);
    }
    // hand_size one-hot (27) — index = min(hand_size, 26)
    let mut size_onehot = [0.0f32; RANK_BUCKETS];
    let idx = input.hand_sizes[seat].min(26);
    size_onehot[idx] = 1.0;
    out.extend_from_slice(&size_onehot);
    // bombs_played (9) — capped at 3.0
    for i in 0..BEHAVIOR_DIM {
        out.push((input.bombs_played[seat][i] as f32).min(3.0));
    }
}

/// Return `true` if `combo` is a "highest rank" play — a pure triple of the
/// level rank with no wilds.
///
/// Matches `is_highest_rank` in `encoder.py`.
fn is_highest_rank(combo: &Combo, level_rank: u8) -> bool {
    combo.combo_type == TRIPLE && combo.cards.iter().all(|c| c.rank == level_rank)
}

/// Compute the 9-dim behavior flag vector.
///
/// Mirrors `compute_state_behavior_flags` in `encoder.py` exactly.
fn compute_behavior_flags(input: &EncoderInput, player: usize, legal: &[Combo]) -> [f32; 9] {
    let mut flags = [0.0f32; 9];

    let is_leading = input.current_trick.is_none();
    let partner = (player + 2) % 4;
    let opp_left = (player + 1) % 4;
    let opp_right = (player + 3) % 4;

    let partner_hand_size = input.hand_sizes[partner];

    // Minimum hand size among active (not out) opponents.
    // Falls back to 27 if both opponents are already out.
    let min_opp: usize = {
        let mut m = 27usize;
        for &opp in &[opp_left, opp_right] {
            if !input.is_out[opp] {
                m = m.min(input.hand_sizes[opp]);
            }
        }
        m
    };

    // ── can_coop ────────────────────────────────────────────────────────────
    let can_coop = !is_leading
        && input.trick_winner == Some(partner)
        && legal.iter().any(|m| m.combo_type != PASS);

    if !can_coop {
        flags[0] = 1.0;
    } else {
        if legal.iter().any(|m| m.combo_type == PASS) {
            flags[1] = 1.0;
        }
        if legal.iter().any(|m| m.combo_type != PASS) {
            flags[2] = 1.0;
        }
    }

    // ── can_dwarf ───────────────────────────────────────────────────────────
    let can_dwarf = is_leading
        && legal
            .iter()
            .any(|m| m.combo_type != PASS && m.cards.len() > min_opp);

    if !can_dwarf {
        flags[3] = 1.0;
    } else {
        for m in legal {
            if m.combo_type != PASS && m.cards.len() > min_opp {
                flags[4] = 1.0;
            } else {
                flags[5] = 1.0;
            }
            if flags[4] != 0.0 && flags[5] != 0.0 {
                break;
            }
        }
    }

    // ── can_assist ──────────────────────────────────────────────────────────
    let can_assist = is_leading
        && legal.iter().any(|m| {
            m.combo_type != PASS
                && m.cards.len() < partner_hand_size
                && !is_highest_rank(m, input.level_rank)
        });

    if !can_assist {
        flags[6] = 1.0;
    } else {
        for m in legal {
            let is_assisting = m.combo_type != PASS
                && m.cards.len() < partner_hand_size
                && !is_highest_rank(m, input.level_rank);
            if is_assisting {
                flags[7] = 1.0;
            } else {
                flags[8] = 1.0;
            }
            if flags[7] != 0.0 && flags[8] != 0.0 {
                break;
            }
        }
    }

    flags
}

/// Reinterpret a `&[f32]` as a `Vec<u8>` (native byte order).
///
/// Uses `f32::to_ne_bytes` so the encoding matches numpy's host-endian default
/// on x86_64 and arm64 (both little-endian in practice).
fn floats_to_bytes(floats: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(floats.len() * 4);
    for &f in floats {
        bytes.extend_from_slice(&f.to_ne_bytes());
    }
    bytes
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cards::{Card, RANK_BLACK_JOKER, RANK_RED_JOKER};
    use crate::combos::Combo;

    // ── Helpers ──────────────────────────────────────────────────────────────

    fn zero_input() -> (
        [[u8; 108]; 4],
        [[u8; 108]; 4],
        [[u8; 108]; 4],
        [[u32; 9]; 4],
    ) {
        (
            [[0u8; 108]; 4],
            [[0u8; 108]; 4],
            [[0u8; 108]; 4],
            [[0u32; 9]; 4],
        )
    }

    fn minimal_input<'a>(
        hand_multihot: &'a [[u8; 108]; 4],
        played_multihot: &'a [[u8; 108]; 4],
        last_nonpass_multihot: &'a [[u8; 108]; 4],
        bombs_played: &'a [[u32; 9]; 4],
        history: &'a [(usize, Combo)],
    ) -> EncoderInput<'a> {
        EncoderInput {
            hand_multihot,
            played_multihot,
            last_nonpass_multihot,
            bombs_played,
            hand_sizes: [27, 27, 27, 27],
            is_out: [false; 4],
            current_trick: None,
            trick_winner: None,
            level_rank: 2,
            move_history: history,
        }
    }

    // ── card_to_id ───────────────────────────────────────────────────────────

    #[test]
    fn card_to_id_regular() {
        // rank=2 (2-2=0), suit=0, deck=0 → 0*8 + 0*2 + 0 = 0
        let c = Card::new(2, 0, 0);
        assert_eq!(card_to_id(&c), 0);
    }

    #[test]
    fn card_to_id_jokers() {
        let bj0 = Card::new(RANK_BLACK_JOKER, 0, 0);
        assert_eq!(card_to_id(&bj0), 104);

        let rj1 = Card::new(RANK_RED_JOKER, 1, 1);
        assert_eq!(card_to_id(&rj1), 107);
    }

    // ── encode_state ─────────────────────────────────────────────────────────

    #[test]
    fn state_dim_correct() {
        let (h, p, l, b) = zero_input();
        let hist: Vec<(usize, Combo)> = vec![];
        let input = minimal_input(&h, &p, &l, &b, &hist);
        let legal: Vec<Combo> = vec![Combo::pass_combo()];
        let buf = encode_state(&input, 0, &legal);
        assert_eq!(buf.len(), STATE_DIM * 4);
    }

    // ── encode_actions ───────────────────────────────────────────────────────

    #[test]
    fn action_dim_correct() {
        // One-card action (single 2♠ deck 0, id=0)
        let card = Card::new(2, 0, 0);
        let combo = Combo::new(crate::cards::SINGLE, 2, vec![card], 0, 0);
        let buf = encode_actions(&[combo]);
        assert_eq!(buf.len(), ACTION_DIM * 4);
    }

    // ── behavior flags ───────────────────────────────────────────────────────

    #[test]
    fn behavior_not_leading_noop() {
        // is_leading=true (current_trick=None), no legal moves that can
        // dwarf (cards.len() > min_opp=27 is impossible) or assist
        // (cards.len() < partner_hand_size=27 requires < 27 cards).
        // All opponents have 27 cards (min_opp=27).
        // Partner has 27 cards. A 1-card single has len=1 < 27, so
        // can_assist would fire unless we use only PASS.
        // Using only PASS: can_coop=false (is_leading), can_dwarf=false,
        // can_assist=false → flags[0]=1, flags[3]=1, flags[6]=1.
        let (h, p, l, b) = zero_input();
        let hist: Vec<(usize, Combo)> = vec![];
        let input = minimal_input(&h, &p, &l, &b, &hist);
        let legal: Vec<Combo> = vec![Combo::pass_combo()];
        let buf = encode_state(&input, 0, &legal);

        // Behavior starts at byte offset:
        // (108 + 108 + 108 + 1008 + 13) * 4 = 1345 * 4 = 5380
        let base = (108 + 108 + 108 + 1008 + 13) * 4;
        let flags: Vec<f32> = (0..9)
            .map(|i| f32::from_ne_bytes(buf[base + i * 4..base + i * 4 + 4].try_into().unwrap()))
            .collect();

        assert_eq!(flags[0], 1.0, "flags[0] (no coop) should be 1");
        assert_eq!(flags[3], 1.0, "flags[3] (no dwarf) should be 1");
        assert_eq!(flags[6], 1.0, "flags[6] (no assist) should be 1");
    }

    // ── compute_head_id ──────────────────────────────────────────────────────

    #[test]
    fn head_id_absolute_seat() {
        for p in 0..4 {
            assert_eq!(compute_head_id(p), p as i64);
        }
    }
}
