mod cards;
mod combos;

use std::collections::HashSet;

use pyo3::prelude::*;

use cards::Card;

// Combo as raw tuple: (combo_type, key, cards_vec, length, wild_count)
type PyCard = (u8, u8, u8);
type PyCombo = (u8, u8, Vec<PyCard>, u8, u8);

fn card_from_py(t: &PyCard) -> Card {
    Card::new(t.0, t.1, t.2)
}

fn combo_to_py(c: &combos::Combo) -> PyCombo {
    let cards: Vec<PyCard> = c.cards.iter().map(|card| (card.rank, card.suit, card.deck)).collect();
    (c.combo_type, c.key, cards, c.length, c.wild_count)
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

#[pymodule]
fn guandan_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(generate_all_leads, m)?)?;
    m.add_function(wrap_pyfunction!(generate_responses, m)?)?;
    Ok(())
}
