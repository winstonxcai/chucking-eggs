mod cards;
mod combos;
mod game;

use std::collections::HashSet;

use pyo3::prelude::*;

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

#[pymodule]
fn _guandan_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(generate_all_leads, m)?)?;
    m.add_function(wrap_pyfunction!(generate_responses, m)?)?;
    m.add_function(wrap_pyfunction!(dedup_strategic, m)?)?;
    m.add_function(wrap_pyfunction!(mc_rollout, m)?)?;
    m.add_function(wrap_pyfunction!(mc_rollout_batch, m)?)?;
    Ok(())
}
