mod cards;
mod combos;
pub mod encoder;
mod game;
mod rollout;

use std::collections::HashSet;

use pyo3::prelude::*;
use pyo3::types::PyBytes;

use cards::Card;

// Combo as raw tuple: (combo_type, key, cards_vec, length, wild_count)
pub(crate) type PyCard = (u8, u8, u8);
pub(crate) type PyCombo = (u8, u8, Vec<PyCard>, u8, u8);

pub(crate) fn card_from_py(t: &PyCard) -> Card {
    Card::new(t.0, t.1, t.2)
}

pub(crate) fn combo_to_py(c: &combos::Combo) -> PyCombo {
    let cards: Vec<PyCard> = c.cards.iter().map(|card| (card.rank, card.suit, card.deck)).collect();
    (c.combo_type, c.key, cards, c.length, c.wild_count)
}

pub(crate) fn combo_from_py(t: &PyCombo) -> combos::Combo {
    let cards: Vec<Card> = t.2.iter().map(card_from_py).collect();
    combos::Combo::new(t.0, t.1, cards, t.3, t.4)
}

#[pyfunction]
fn generate_all_leads(hand: Vec<PyCard>, level_rank: u8) -> Vec<PyCombo> {
    let hand_set: HashSet<Card> = hand.iter().map(card_from_py).collect();
    let results = combos::generate_all_leads(&hand_set, level_rank);
    results.iter().map(combo_to_py).collect()
}

#[pyfunction]
fn generate_responses(hand: Vec<PyCard>, level_rank: u8, trick: PyCombo) -> Vec<PyCombo> {
    let hand_set: HashSet<Card> = hand.iter().map(card_from_py).collect();
    let trick_cards: Vec<Card> = trick.2.iter().map(card_from_py).collect();
    let trick_combo = combos::Combo::new(trick.0, trick.1, trick_cards, trick.3, trick.4);
    let results = combos::generate_responses(&hand_set, level_rank, &trick_combo);
    results.iter().map(combo_to_py).collect()
}

/// Collapse suit-variants of strategically-equivalent plays; first-seen wins.
/// Single source of truth — Python `legal_utils.dedup_strategic` calls this.
#[pyfunction]
fn dedup_strategic(legal: Vec<PyCombo>) -> Vec<PyCombo> {
    let combos_in: Vec<combos::Combo> = legal.iter().map(combo_from_py).collect();
    let combos_out = combos::dedup_strategic(combos_in);
    combos_out.iter().map(combo_to_py).collect()
}

/// Run MC rollouts for a specific move from a game state.
///
/// Args:
///   hands: 4 lists of (rank, suit, deck) cards — one per player
///   current_player: whose turn it is (0-3)
///   level_rank: the level card rank
///   trick: current trick as combo tuple, or None if leading
///   trick_winner: seat that played the current trick (0-3), or None
///   consecutive_passes: number of consecutive passes
///   finish_order: list of players who have gone out, in order
///   is_out: 4 bools — whether each player is out
///   move_combo: the move to evaluate (combo tuple)
///   player: the player making the move (0-3)
///   n_sims: number of rollout simulations
///
/// Returns: average reward for the player.
#[pyfunction]
#[pyo3(signature = (hands, current_player, level_rank, trick, trick_winner, consecutive_passes, finish_order, is_out, move_combo, player, n_sims))]
fn mc_rollout(
    hands: Vec<Vec<PyCard>>,
    current_player: usize,
    level_rank: u8,
    trick: Option<PyCombo>,
    trick_winner: Option<usize>,
    consecutive_passes: u32,
    finish_order: Vec<usize>,
    is_out: Vec<bool>,
    move_combo: PyCombo,
    player: usize,
    n_sims: usize,
) -> f64 {
    let mut env = game::GameEnv {
        level_rank,
        hands: [
            hands[0].iter().map(card_from_py).collect(),
            hands[1].iter().map(card_from_py).collect(),
            hands[2].iter().map(card_from_py).collect(),
            hands[3].iter().map(card_from_py).collect(),
        ],
        current_player,
        current_trick: trick.as_ref().map(|t| combo_from_py(t)),
        trick_winner,
        consecutive_passes,
        finish_order,
        is_out: [is_out[0], is_out[1], is_out[2], is_out[3]],
        done: false,
    };

    let move_c = combo_from_py(&move_combo);
    game::evaluate_move(&env, player, &move_c, n_sims)
}

/// Run MC rollouts for ALL candidate moves from a game state.
/// Returns a list of average rewards, one per candidate.
/// This avoids per-move Python→Rust overhead.
#[pyfunction]
#[pyo3(signature = (hands, current_player, level_rank, trick, trick_winner, consecutive_passes, finish_order, is_out, candidates, player, n_sims))]
fn mc_rollout_batch(
    hands: Vec<Vec<PyCard>>,
    current_player: usize,
    level_rank: u8,
    trick: Option<PyCombo>,
    trick_winner: Option<usize>,
    consecutive_passes: u32,
    finish_order: Vec<usize>,
    is_out: Vec<bool>,
    candidates: Vec<PyCombo>,
    player: usize,
    n_sims: usize,
) -> Vec<f64> {
    let env = game::GameEnv {
        level_rank,
        hands: [
            hands[0].iter().map(card_from_py).collect(),
            hands[1].iter().map(card_from_py).collect(),
            hands[2].iter().map(card_from_py).collect(),
            hands[3].iter().map(card_from_py).collect(),
        ],
        current_player,
        current_trick: trick.as_ref().map(|t| combo_from_py(t)),
        trick_winner,
        consecutive_passes,
        finish_order,
        is_out: [is_out[0], is_out[1], is_out[2], is_out[3]],
        done: false,
    };

    candidates.iter()
        .map(|c| {
            let move_c = combo_from_py(c);
            game::evaluate_move(&env, player, &move_c, n_sims)
        })
        .collect()
}

/// Test helper: encode a game state from Python-supplied arrays.
///
/// Returns the same flat `f32` byte buffer (`STATE_DIM * 4` bytes) that
/// `play_episode_rust` produces, allowing Python-side parity checks against
/// `RoleAwareStateActionEncoder`.
///
/// Args:
///   hand_multihot:        4 × 108 nested list of u8
///   played_multihot:      4 × 108 nested list of u8
///   last_nonpass_multihot:4 × 108 nested list of u8
///   bombs_played:         4 × 9  nested list of u32
///   hand_sizes:           4 ints
///   is_out:               4 bools
///   current_trick:        optional combo tuple (or None)
///   trick_winner:         optional seat index (or None)
///   level_rank:           u8
///   move_history:         list of (seat, combo_tuple)
///   player:               current actor seat
///   legal_moves:          list of combo tuples
#[pyfunction]
#[pyo3(signature = (
    hand_multihot, played_multihot, last_nonpass_multihot, bombs_played,
    hand_sizes, is_out, current_trick, trick_winner, level_rank,
    move_history, player, legal_moves
))]
fn encode_state_for_parity(
    py: Python<'_>,
    hand_multihot: Vec<Vec<u8>>,
    played_multihot: Vec<Vec<u8>>,
    last_nonpass_multihot: Vec<Vec<u8>>,
    bombs_played: Vec<Vec<u32>>,
    hand_sizes: Vec<usize>,
    is_out: Vec<bool>,
    current_trick: Option<PyCombo>,
    trick_winner: Option<usize>,
    level_rank: u8,
    move_history: Vec<(usize, PyCombo)>,
    player: usize,
    legal_moves: Vec<PyCombo>,
) -> PyResult<Py<PyBytes>> {
    // Convert flat Vec<Vec<_>> → fixed-size arrays
    let mut hm = [[0u8; 108]; 4];
    let mut pm = [[0u8; 108]; 4];
    let mut lm = [[0u8; 108]; 4];
    let mut bp = [[0u32; 9]; 4];
    for s in 0..4 {
        hm[s].copy_from_slice(&hand_multihot[s]);
        pm[s].copy_from_slice(&played_multihot[s]);
        lm[s].copy_from_slice(&last_nonpass_multihot[s]);
        bp[s].copy_from_slice(&bombs_played[s]);
    }
    let hs: [usize; 4] = hand_sizes.try_into().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err("hand_sizes must have exactly 4 elements")
    })?;
    let io: [bool; 4] = is_out.try_into().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err("is_out must have exactly 4 elements")
    })?;
    let trick = current_trick.as_ref().map(combo_from_py);
    let history: Vec<(usize, combos::Combo)> = move_history
        .iter()
        .map(|(seat, c)| (*seat, combo_from_py(c)))
        .collect();
    let legal: Vec<combos::Combo> = legal_moves.iter().map(combo_from_py).collect();
    let input = encoder::EncoderInput {
        hand_multihot: &hm,
        played_multihot: &pm,
        last_nonpass_multihot: &lm,
        bombs_played: &bp,
        hand_sizes: hs,
        is_out: io,
        current_trick: trick.as_ref(),
        trick_winner,
        level_rank,
        move_history: &history,
    };
    let bytes = encoder::encode_state(&input, player, &legal);
    Ok(PyBytes::new_bound(py, &bytes).into())
}

#[pymodule]
fn guandan_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(generate_all_leads, m)?)?;
    m.add_function(wrap_pyfunction!(generate_responses, m)?)?;
    m.add_function(wrap_pyfunction!(dedup_strategic, m)?)?;
    m.add_function(wrap_pyfunction!(mc_rollout, m)?)?;
    m.add_function(wrap_pyfunction!(mc_rollout_batch, m)?)?;
    m.add_function(wrap_pyfunction!(rollout::play_episode_rust, m)?)?;
    m.add_function(wrap_pyfunction!(encode_state_for_parity, m)?)?;
    Ok(())
}
