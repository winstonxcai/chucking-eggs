//! Rust-driven self-play episode with a Python NN scoring callback.
//!
//! `play_episode_rust` runs the full episode loop in Rust — game stepping and
//! legal-move generation — and calls back into Python only for Q-network
//! scoring.  The Python "curried hearth" callback is the only GIL-holding
//! operation per step; game stepping and move generation run without the GIL.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::cards::{Card, BOMB_4, PASS, is_bomb_type};
use crate::combos::Combo;
use crate::game::GameEnv;
use crate::{PyCombo, combo_to_py};

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

/// Map a bomb combo_type (BOMB_4..=BOMB_JOKER) to a 0-based tier index.
///
/// The 9 tiers correspond to the 9 bomb types in ascending strength order,
/// matching the `bombs_played` histogram layout the encoder reads.
/// Returns `None` for non-bomb combo types.
fn bomb_tier(combo_type: u8) -> Option<usize> {
    if !is_bomb_type(combo_type) {
        return None;
    }
    // BOMB_4=8 is tier 0; BOMB_JOKER=16 is tier 8.
    Some((combo_type - BOMB_4) as usize)
}

// ─── Episode-level tracking state ───────────────────────────────────────────

/// State maintained by the rollout loop beyond what `GameEnv` stores.
///
/// `GameEnv` tracks the live game state. `EpisodeState` adds the history and
/// aggregate statistics the Python encoder needs to produce training features.
struct EpisodeState {
    env: GameEnv,
    /// Chronological (player, combo) pairs for all moves — used to build the
    /// encoder's history window and last-action-per-role features.
    move_history: Vec<(usize, Combo)>,
    /// Cards each seat has played so far (excluding their current hand).
    played_cards: [Vec<Card>; 4],
    /// Per-seat bomb-tier histogram — 9 entries matching BOMB_4..=BOMB_JOKER.
    bombs_played: [[u32; 9]; 4],
}

impl EpisodeState {
    fn new(level_rank: u8, seed: Option<u64>) -> Self {
        let mut env = GameEnv::new(level_rank);
        if let Some(s) = seed {
            // reset was called by new(); re-reset with the fixed seed.
            env.reset_seeded(s);
        }
        Self {
            env,
            move_history: Vec::with_capacity(100),
            played_cards: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
            bombs_played: [[0u32; 9]; 4],
        }
    }

    /// Record a completed step: update move history, played cards, and bomb histogram.
    fn record_step(&mut self, player: usize, combo: &Combo) {
        if combo.combo_type != PASS {
            for &card in &combo.cards {
                self.played_cards[player].push(card);
            }
            if let Some(tier) = bomb_tier(combo.combo_type) {
                self.bombs_played[player][tier] =
                    self.bombs_played[player][tier].saturating_add(1);
            }
        }
        self.move_history.push((player, combo.clone()));
    }

    /// Serialize the current game state to a Python dict for the scorer callback.
    ///
    /// The dict is consumed by `SnapshotEnv` on the Python side, which converts
    /// raw card tuples into the cached multihot arrays the encoder expects.
    ///
    /// # Dict keys
    /// - `hands`: `[[PyCard; N]; 4]` — current hands as raw card tuples
    /// - `played_cards`: `[[PyCard; N]; 4]` — cumulative played cards per seat
    /// - `bombs_played`: `[[u32; 9]; 4]` — bomb-tier histogram per seat
    /// - `level_rank`, `current_player`, `consecutive_passes` — scalars
    /// - `finish_order`, `is_out` — lists
    /// - `current_trick`: `None | PyCombo`
    /// - `trick_winner`: `None | int`
    /// - `move_history`: `[(player, PyCombo)]`
    fn to_snapshot<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);

        let hands: Vec<Vec<(u8, u8, u8)>> = self
            .env
            .hands
            .iter()
            .map(|h| h.iter().map(|c| (c.rank, c.suit, c.deck)).collect())
            .collect();
        d.set_item("hands", hands)?;

        let played: Vec<Vec<(u8, u8, u8)>> = self
            .played_cards
            .iter()
            .map(|ps| ps.iter().map(|c| (c.rank, c.suit, c.deck)).collect())
            .collect();
        d.set_item("played_cards", played)?;

        let bombs: Vec<Vec<u32>> =
            self.bombs_played.iter().map(|b| b.to_vec()).collect();
        d.set_item("bombs_played", bombs)?;

        d.set_item("level_rank", self.env.level_rank)?;
        d.set_item("current_player", self.env.current_player)?;
        d.set_item("consecutive_passes", self.env.consecutive_passes)?;
        d.set_item("finish_order", self.env.finish_order.clone())?;
        d.set_item("is_out", self.env.is_out.to_vec())?;

        match &self.env.current_trick {
            None => d.set_item("current_trick", py.None())?,
            Some(trick) => d.set_item("current_trick", combo_to_py(trick))?,
        }
        match self.env.trick_winner {
            None => d.set_item("trick_winner", py.None())?,
            Some(w) => d.set_item("trick_winner", w)?,
        }

        let hist: Vec<(usize, PyCombo)> = self
            .move_history
            .iter()
            .map(|(p, c)| (*p, combo_to_py(c)))
            .collect();
        d.set_item("move_history", hist)?;

        Ok(d)
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
/// The game loop runs entirely in Rust.  For each decision the `scorer` callback
/// is invoked once; all other work (game stepping, legal-move generation) runs
/// without holding the GIL.
///
/// # Arguments
/// - `scorer` — Python callable with signature:
///   ```python
///   scorer(state_dict, legal_tuples, player) -> (idx, encoded, q_gap, chosen_by_epsilon)
///   ```
///   `state_dict` is the snapshot produced by `EpisodeState.to_snapshot`;
///   `legal_tuples` is the full (un-deduped) legal move list as `PyCombo` tuples;
///   `player` is the current seat (0-3).  The scorer returns the chosen action
///   index, the encoded numpy-array dict for that action, the Q-value gap (NaN
///   for ε-random decisions), and a 0/1 flag for ε-random selection.
/// - `level_rank` — starting level card rank (default 2 = Two, matching `GuanDanEnv`).
/// - `seed` — optional RNG seed for the deal and starting player.
///
/// # Returns
/// `(trajectory, rewards)` where `trajectory` is a Python list of step dicts
/// accepted by `compute_mc_returns` and `rewards` is a `[f64; 4]` terminal
/// reward vector.
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
        let legal = state.env.legal_moves();
        let k = legal.len();

        // Compute pre-step metadata before the action is chosen.
        let phase_self_val = phase_bucket(state.env.hands[p].len());
        let phase_partner_val = phase_with_out(&state.env, partner);
        let trick_role_val = trick_role(&state.env, partner);
        let bomb_available_val =
            legal.iter().any(|m| is_bomb_type(m.combo_type)) as u8;
        let legal_py: Vec<PyCombo> = legal.iter().map(combo_to_py).collect();

        // Build the game-state snapshot and invoke the Python scorer.
        let snapshot: PyObject = state.to_snapshot(py)?.into();
        let result = scorer
            .call1(py, (snapshot, legal_py, p as u8))
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
        // Same seed → same starting player and same hand sizes.
        assert_eq!(s1.env.current_player, s2.env.current_player);
        for seat in 0..4 {
            assert_eq!(s1.env.hands[seat].len(), s2.env.hands[seat].len());
        }
    }
}
