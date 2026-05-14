//! Rust-driven self-play episode with a Python NN scoring callback.
//!
//! `play_episode_rust` runs the full episode loop in Rust — game stepping,
//! legal-move generation, and state encoding — and calls back into Python only
//! for Q-network scoring.  The Python callback receives pre-encoded `f32` byte
//! buffers and needs only `np.frombuffer` + a single NN forward pass.
//!
//! Scorer signature (Python side):
//!   ```python
//!   scorer(state_bytes: bytes, action_bytes: bytes, head_id: int)
//!       -> (idx, encoded_dict, q_gap, chosen_by_epsilon)
//!   ```

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};

use crate::cards::{card_to_id, is_bomb_type, BOMB_4, PASS};
use crate::combos::{dedup_strategic, Combo};
use crate::encoder::{self, EncoderInput};
use crate::game::GameEnv;

// ─── Phase / role helpers ────────────────────────────────────────────────────
// Mirror the Python helpers in actor.py verbatim so trajectory metadata is
// identical whether the Python or Rust rollout path is used.

fn phase_bucket(hand_size: usize) -> u8 {
    if hand_size >= 20 {
        0
    } else if hand_size >= 10 {
        1
    } else {
        2
    }
}

fn phase_with_out(env: &GameEnv, seat: usize) -> u8 {
    if env.is_out[seat] {
        3
    } else {
        phase_bucket(env.hands[seat].len())
    }
}

/// 0 = leading a new trick, 1 = following (partner alive), 2 = following (partner out).
fn trick_role(env: &GameEnv, partner: usize) -> u8 {
    if env.current_trick.is_none() {
        0
    } else if env.is_out[partner] {
        2
    } else {
        1
    }
}

/// Map a bomb combo_type to a 0-based tier index (BOMB_4=0 … BOMB_JOKER=8).
fn bomb_tier(combo_type: u8) -> Option<usize> {
    if !is_bomb_type(combo_type) {
        return None;
    }
    Some((combo_type - BOMB_4) as usize)
}

// ─── Episode-level tracking state ───────────────────────────────────────────

/// State maintained by the rollout loop beyond what `GameEnv` stores.
///
/// Multihots (`hand_multihot`, `played_multihot`, `last_nonpass_multihot`) are
/// maintained incrementally — updated in O(cards_played) per step — so the
/// encoder never reconstructs them from scratch.
struct EpisodeState {
    env: GameEnv,
    /// Chronological (seat, combo) pairs — fed directly to the encoder history.
    move_history: Vec<(usize, Combo)>,
    /// Per-seat bomb-tier histogram (9 bins: BOMB_4..=BOMB_JOKER).
    bombs_played: [[u32; 9]; 4],
    /// Multi-hot card presence per seat; cleared as cards are played.
    hand_multihot: [[u8; 108]; 4],
    /// Multi-hot cumulative played cards per seat; set as cards are played.
    played_multihot: [[u8; 108]; 4],
    /// Most recent non-pass action per seat as a multi-hot; overwritten each time.
    last_nonpass_multihot: [[u8; 108]; 4],
}

impl EpisodeState {
    fn new(level_rank: u8, seed: Option<u64>) -> Self {
        let mut env = GameEnv::new(level_rank);
        if let Some(s) = seed {
            env.reset_seeded(s);
        }
        let mut hand_multihot = [[0u8; 108]; 4];
        for seat in 0..4 {
            for card in &env.hands[seat] {
                hand_multihot[seat][card_to_id(card)] = 1;
            }
        }
        Self {
            env,
            move_history: Vec::with_capacity(100),
            bombs_played: [[0u32; 9]; 4],
            hand_multihot,
            played_multihot: [[0u8; 108]; 4],
            last_nonpass_multihot: [[0u8; 108]; 4],
        }
    }

    /// Record a completed step: update all incremental arrays and move history.
    fn record_step(&mut self, player: usize, combo: &Combo) {
        if combo.combo_type != PASS {
            let mut last_action = [0u8; 108];
            for card in &combo.cards {
                let cid = card_to_id(card);
                self.hand_multihot[player][cid] = 0;
                self.played_multihot[player][cid] = 1;
                last_action[cid] = 1;
            }
            self.last_nonpass_multihot[player] = last_action;

            if let Some(tier) = bomb_tier(combo.combo_type) {
                self.bombs_played[player][tier] =
                    self.bombs_played[player][tier].saturating_add(1);
            }
        }
        self.move_history.push((player, combo.clone()));
    }

    /// Build an `EncoderInput` borrowing from this state.
    fn encoder_input(&self) -> EncoderInput<'_> {
        EncoderInput {
            hand_multihot: &self.hand_multihot,
            played_multihot: &self.played_multihot,
            last_nonpass_multihot: &self.last_nonpass_multihot,
            bombs_played: &self.bombs_played,
            hand_sizes: [
                self.env.hands[0].len(),
                self.env.hands[1].len(),
                self.env.hands[2].len(),
                self.env.hands[3].len(),
            ],
            is_out: self.env.is_out,
            current_trick: self.env.current_trick.as_ref(),
            trick_winner: self.env.trick_winner,
            level_rank: self.env.level_rank,
            move_history: &self.move_history,
        }
    }
}

// ─── Per-step record ─────────────────────────────────────────────────────────

/// Metadata for one decision step, computed from pre-step game state.
///
/// `encoded` is the Python numpy-array dict produced by the scorer callback and
/// stored as an opaque `PyObject` until the episode ends and `compute_mc_returns`
/// is called on the Python side.
struct StepRecord {
    player: usize,
    encoded: PyObject,
    phase_self: u8,
    trick_role: u8,
    phase_partner: u8,
    action_type: u8,
    is_pass: u8,
    is_bomb: u8,
    bomb_available: u8,
    num_legal_actions: usize,
    q_gap: f64,
    chosen_by_epsilon: u8,
}

impl StepRecord {
    /// Convert to a Python dict matching the `TrajectoryStep` TypedDict layout
    /// consumed by `compute_mc_returns`.
    fn to_pydict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        d.set_item("player", self.player)?;
        d.set_item("encoded", self.encoded.bind(py))?;
        d.set_item("phase_self", self.phase_self)?;
        d.set_item("trick_role", self.trick_role)?;
        d.set_item("phase_partner", self.phase_partner)?;
        d.set_item("action_type", self.action_type)?;
        d.set_item("is_pass", self.is_pass)?;
        d.set_item("is_bomb", self.is_bomb)?;
        d.set_item("bomb_available", self.bomb_available)?;
        d.set_item("num_legal_actions", self.num_legal_actions)?;
        d.set_item("q_gap", self.q_gap)?;
        d.set_item("chosen_by_epsilon", self.chosen_by_epsilon)?;
        Ok(d)
    }
}

// ─── Main rollout function ────────────────────────────────────────────────────

/// Run one self-play episode, returning a trajectory list and terminal rewards.
///
/// The game loop — stepping, legal-move generation, and state encoding — runs
/// entirely in Rust. For each decision the `scorer` callback is invoked once
/// with pre-encoded byte buffers; all other work runs without the GIL.
///
/// # Scorer signature (Python)
/// ```python
/// scorer(state_bytes: bytes, action_bytes: bytes, head_id: int)
///     -> (idx: int, encoded: dict, q_gap: float, chosen_by_epsilon: int)
/// ```
/// - `state_bytes`: flat `f32` buffer of length `STATE_DIM * 4` bytes
///   (one encoded state vector, layout matches `ROLE_ENCODE_STATE_KEYS`).
/// - `action_bytes`: flat `f32` buffer of length `n_actions * ACTION_DIM * 4`
///   bytes (one candidate-action multi-hot per legal action, concatenated).
/// - `head_id`: absolute seat index for head routing.
/// - Returns the chosen action index, its numpy-dict encoding (stored in the
///   trajectory), the Q-value gap (NaN for ε-random choices), and a 0/1 flag.
///
/// # Arguments
/// - `level_rank` — starting level card rank (default 2).
/// - `seed` — optional RNG seed for reproducible deals.
///
/// # Returns
/// `(trajectory, rewards)` where `trajectory` is a Python list of step dicts
/// accepted by `compute_mc_returns` and `rewards` is a `[f64; 4]` reward vector.
#[pyfunction]
#[pyo3(signature = (scorer, level_rank = 2, seed = None))]
pub fn play_episode_rust(
    py: Python<'_>,
    scorer: Py<PyAny>,
    level_rank: u8,
    seed: Option<u64>,
) -> PyResult<(PyObject, Vec<f64>)> {
    let mut state = EpisodeState::new(level_rank, seed);
    let mut trajectory: Vec<StepRecord> = Vec::with_capacity(60);

    while !state.env.done {
        let p = state.env.current_player;
        let partner = GameEnv::partner(p);
        // Collapse suit-variants before encoding/scoring — mirrors Python's
        // dedup_strategic in legal_utils.py. Without this, suit-equivalent
        // combos produce identical f32 encodings, causing redundant NN evals.
        let legal = dedup_strategic(state.env.legal_moves());
        let k = legal.len();

        // Pre-step metadata
        let phase_self_val = phase_bucket(state.env.hands[p].len());
        let phase_partner_val = phase_with_out(&state.env, partner);
        let trick_role_val = trick_role(&state.env, partner);
        let bomb_available_val = legal.iter().any(|m| is_bomb_type(m.combo_type)) as u8;

        // Encode state and actions in Rust — Python only runs the NN forward.
        let enc_input = state.encoder_input();
        let state_bytes = encoder::encode_state(&enc_input, p, &legal);
        let action_bytes = encoder::encode_actions(&legal);
        let head_id = encoder::compute_head_id(p);

        let result = scorer
            .call1(
                py,
                (
                    PyBytes::new_bound(py, &state_bytes),
                    PyBytes::new_bound(py, &action_bytes),
                    head_id,
                ),
            )
            .map_err(|e| {
                PyValueError::new_err(format!("scorer raised at player {p}: {e}"))
            })?;

        let (idx, encoded_obj, q_gap, chosen_by_epsilon): (usize, PyObject, f64, u8) =
            result.extract(py)?;

        if idx >= k {
            return Err(PyValueError::new_err(format!(
                "scorer returned idx={idx} but there are only {k} legal actions \
                 (player={p})"
            )));
        }

        let action = &legal[idx];
        trajectory.push(StepRecord {
            player: p,
            encoded: encoded_obj,
            phase_self: phase_self_val,
            trick_role: trick_role_val,
            phase_partner: phase_partner_val,
            action_type: action.combo_type,
            is_pass: (action.combo_type == PASS) as u8,
            is_bomb: is_bomb_type(action.combo_type) as u8,
            bomb_available: bomb_available_val,
            num_legal_actions: k,
            q_gap,
            chosen_by_epsilon,
        });

        state.record_step(p, action);
        state.env.step(action);
    }

    let rewards = state.env.get_rewards();

    let step_dicts = trajectory
        .iter()
        .map(|r| r.to_pydict(py))
        .collect::<PyResult<Vec<_>>>()?;
    let steps: PyObject = PyList::new_bound(py, step_dicts).into();

    Ok((steps, rewards.to_vec()))
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phase_bucket_boundaries() {
        assert_eq!(phase_bucket(27), 0);
        assert_eq!(phase_bucket(20), 0);
        assert_eq!(phase_bucket(19), 1);
        assert_eq!(phase_bucket(10), 1);
        assert_eq!(phase_bucket(9), 2);
        assert_eq!(phase_bucket(0), 2);
    }

    #[test]
    fn bomb_tier_mapping() {
        use crate::cards::{BOMB_4, BOMB_JOKER, PASS, SINGLE};
        assert_eq!(bomb_tier(BOMB_4), Some(0));
        assert_eq!(bomb_tier(BOMB_JOKER), Some(8));
        assert_eq!(bomb_tier(PASS), None);
        assert_eq!(bomb_tier(SINGLE), None);
    }

    #[test]
    fn episode_state_seeded_is_deterministic() {
        let s1 = EpisodeState::new(2, Some(42));
        let s2 = EpisodeState::new(2, Some(42));
        assert_eq!(s1.env.current_player, s2.env.current_player);
        for seat in 0..4 {
            assert_eq!(s1.env.hands[seat].len(), s2.env.hands[seat].len());
        }
    }

    #[test]
    fn hand_multihot_initializes_from_deal() {
        let state = EpisodeState::new(2, Some(7));
        // Each seat should have exactly 27 bits set in hand_multihot.
        for seat in 0..4 {
            let count: usize = state.hand_multihot[seat].iter().map(|&b| b as usize).sum();
            assert_eq!(count, 27, "seat {seat} should have 27 cards");
        }
        // All bits across all seats should be distinct (no sharing).
        let mut union = [0u8; 108];
        for seat in 0..4 {
            for i in 0..108 {
                union[i] += state.hand_multihot[seat][i];
            }
        }
        assert!(union.iter().all(|&b| b <= 1), "cards should not appear in two hands");
    }

    #[test]
    fn record_step_moves_card_between_multihots() {
        use crate::cards::{Card, SINGLE};
        let mut state = EpisodeState::new(2, Some(11));
        let p = state.env.current_player;
        // Pick any card from player p's hand.
        let card = *state.env.hands[p].iter().next().unwrap();
        let cid = card_to_id(&card);
        assert_eq!(state.hand_multihot[p][cid], 1);
        assert_eq!(state.played_multihot[p][cid], 0);

        let combo = Combo::new(SINGLE, card.rank, vec![card], 1, 0);
        state.record_step(p, &combo);

        assert_eq!(state.hand_multihot[p][cid], 0, "card should leave hand");
        assert_eq!(state.played_multihot[p][cid], 1, "card should enter played");
        assert_eq!(state.last_nonpass_multihot[p][cid], 1, "should be last action");
    }
}
