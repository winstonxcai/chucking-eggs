mod cards;
mod combos;

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

/// Generate and strategic-deduplicate legal moves in one native call.
#[pyfunction]
#[pyo3(signature = (hand, level_rank, trick=None))]
fn select_legal(hand: Vec<PyCard>, level_rank: u8, trick: Option<PyCombo>) -> Vec<PyCombo> {
    let hand_set: HashSet<Card> = hand.iter().map(card_from_py).collect();
    let legal = match trick {
        Some(t) => {
            let trick_combo = combo_from_py(&t);
            combos::generate_responses(&hand_set, level_rank, &trick_combo)
        }
        None => combos::generate_all_leads(&hand_set, level_rank),
    };
    let deduped = combos::dedup_strategic(legal);
    deduped.iter().map(combo_to_py).collect()
}

#[pymodule]
fn _guandan_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(generate_all_leads, m)?)?;
    m.add_function(wrap_pyfunction!(generate_responses, m)?)?;
    m.add_function(wrap_pyfunction!(dedup_strategic, m)?)?;
    m.add_function(wrap_pyfunction!(select_legal, m)?)?;
    Ok(())
}
