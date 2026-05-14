use std::collections::{HashMap, HashSet};

use crate::cards::*;

#[derive(Clone, Debug)]
pub struct Combo {
    pub combo_type: u8,
    pub key: u8,
    pub cards: Vec<Card>,
    pub length: u8,
    pub wild_count: u8,
}

impl Combo {
    pub fn new(combo_type: u8, key: u8, cards: Vec<Card>, length: u8, wild_count: u8) -> Self {
        Combo {
            combo_type,
            key,
            cards,
            length,
            wild_count,
        }
    }

    pub fn pass_combo() -> Self {
        Combo::new(PASS, 0, Vec::new(), 0, 0)
    }

    pub fn beats(&self, other: &Combo, level_rank: u8) -> bool {
        let my_bomb = is_bomb_type(self.combo_type);
        let their_bomb = is_bomb_type(other.combo_type);

        if my_bomb && !their_bomb {
            return true;
        }
        if !my_bomb && their_bomb {
            return false;
        }

        if my_bomb && their_bomb {
            if self.combo_type != other.combo_type {
                return self.combo_type > other.combo_type;
            }
            // Same bomb tier
            if self.combo_type == STRAIGHT_FLUSH {
                return self.key > other.key; // natural order
            }
            if self.combo_type == BOMB_JOKER {
                return false; // only one exists
            }
            // N-of-a-kind: level order
            return level_order_key(self.key, level_rank)
                > level_order_key(other.key, level_rank);
        }

        // Neither is bomb: must match type
        if self.combo_type != other.combo_type {
            return false;
        }

        // Straights, tubes, plates: natural order
        if matches!(self.combo_type, STRAIGHT | TUBE | PLATE) {
            return self.key > other.key;
        }

        // Singles, pairs, triples, full houses: level order
        level_order_key(self.key, level_rank) > level_order_key(other.key, level_rank)
    }
}

// ─── Helper types ───────────────────────────────────────

type ByRank = HashMap<u8, Vec<Card>>;

fn build_by_rank(naturals: &[Card]) -> ByRank {
    let mut by_rank: ByRank = HashMap::new();
    for c in naturals {
        by_rank.entry(c.rank).or_default().push(*c);
    }
    // Sort each rank's cards by (suit, deck) for deterministic ordering
    for cards in by_rank.values_mut() {
        cards.sort_by_key(|c| (c.suit, c.deck));
    }
    by_rank
}

fn make(combo_type: u8, key: u8, cards: Vec<Card>, length: u8, wild_count: u8) -> Combo {
    Combo::new(combo_type, key, cards, length, wild_count)
}

/// Deduplicate identical cards (same rank+suit but different deck copy).
fn unique_cards(cards: &[Card]) -> Vec<Card> {
    let mut seen: HashSet<(u8, u8)> = HashSet::new();
    let mut result = Vec::new();
    for c in cards {
        let k = (c.rank, c.suit);
        if seen.insert(k) {
            result.push(*c);
        }
    }
    result
}

// ─── PUBLIC API ─────────────────────────────────────────

pub fn generate_all_leads(hand: &HashSet<Card>, level_rank: u8) -> Vec<Combo> {
    let mut combos: Vec<Combo> = Vec::new();

    let mut wilds: Vec<Card> = hand.iter().filter(|c| is_wild(c, level_rank)).copied().collect();
    wilds.sort_by_key(|c| (c.suit, c.deck));
    let naturals: Vec<Card> = hand.iter().filter(|c| !is_wild(c, level_rank)).copied().collect();
    let by_rank = build_by_rank(&naturals);

    // Singles
    for (&rank, cards) in &by_rank {
        for c in unique_cards(cards) {
            combos.push(make(SINGLE, rank, vec![c], 0, 0));
        }
    }
    for w in &wilds {
        combos.push(make(SINGLE, level_rank, vec![*w], 0, 0));
    }

    // Pairs
    add_natural_pairs(&mut combos, &by_rank);
    add_wild_pairs(&mut combos, &by_rank, &wilds, level_rank);
    if wilds.len() >= 2 {
        combos.push(make(PAIR, level_rank, wilds[..2].to_vec(), 0, 2));
    }

    // Triples
    add_natural_triples(&mut combos, &by_rank);
    add_wild_triples(&mut combos, &by_rank, &wilds);

    // Full houses
    add_full_houses(&mut combos, &by_rank, &wilds, level_rank);

    // Straights (5 consecutive, natural order, NOT all same suit)
    add_straights(&mut combos, &by_rank, &wilds, level_rank);

    // Tubes (3 consecutive pairs)
    add_tubes(&mut combos, &by_rank, &wilds);

    // Plates (2 consecutive triples)
    add_plates(&mut combos, &by_rank, &wilds);

    // N-of-a-kind bombs (4-10)
    add_nofakind_bombs(&mut combos, &by_rank, &wilds);

    // Straight flushes (5 consecutive same suit)
    add_straight_flushes(&mut combos, hand, &wilds, level_rank);

    // Four-joker bomb
    let jokers: Vec<Card> = hand
        .iter()
        .filter(|c| c.rank == RANK_BLACK_JOKER || c.rank == RANK_RED_JOKER)
        .copied()
        .collect();
    if jokers.len() == 4 {
        combos.push(make(BOMB_JOKER, 99, jokers, 0, 0));
    }

    deduplicate(combos)
}

pub fn generate_responses(hand: &HashSet<Card>, level_rank: u8, trick: &Combo) -> Vec<Combo> {
    let leads = generate_all_leads(hand, level_rank);
    let mut responses: Vec<Combo> = leads
        .into_iter()
        .filter(|c| c.beats(trick, level_rank))
        .collect();
    responses.push(Combo::pass_combo());
    responses
}

// ─── PAIRS ──────────────────────────────────────────────

fn add_natural_pairs(combos: &mut Vec<Combo>, by_rank: &ByRank) {
    for (&rank, cards) in by_rank {
        if rank == RANK_BLACK_JOKER {
            if cards.len() >= 2 {
                combos.push(make(PAIR, rank, cards[..2].to_vec(), 0, 0));
            }
            continue;
        }
        if rank == RANK_RED_JOKER {
            if cards.len() >= 2 {
                combos.push(make(PAIR, rank, cards[..2].to_vec(), 0, 0));
            }
            continue;
        }
        if cards.len() >= 2 {
            combos.push(make(PAIR, rank, cards[..2].to_vec(), 0, 0));
        }
    }
}

fn add_wild_pairs(combos: &mut Vec<Combo>, by_rank: &ByRank, wilds: &[Card], level_rank: u8) {
    if wilds.is_empty() {
        return;
    }
    for (&rank, cards) in by_rank {
        if rank >= RANK_BLACK_JOKER {
            continue;
        }
        if !cards.is_empty() {
            combos.push(make(PAIR, rank, vec![cards[0], wilds[0]], 0, 1));
        }
    }
}

// ─── TRIPLES ────────────────────────────────────────────

fn add_natural_triples(combos: &mut Vec<Combo>, by_rank: &ByRank) {
    for (&rank, cards) in by_rank {
        if rank >= RANK_BLACK_JOKER {
            continue;
        }
        if cards.len() >= 3 {
            combos.push(make(TRIPLE, rank, cards[..3].to_vec(), 0, 0));
        }
    }
}

fn add_wild_triples(combos: &mut Vec<Combo>, by_rank: &ByRank, wilds: &[Card]) {
    let n_wild = wilds.len();
    for (&rank, cards) in by_rank {
        if rank >= RANK_BLACK_JOKER {
            continue;
        }
        let n = cards.len();
        if n >= 2 && n_wild >= 1 {
            let mut c = cards[..2].to_vec();
            c.push(wilds[0]);
            combos.push(make(TRIPLE, rank, c, 0, 1));
        }
        if n >= 1 && n_wild >= 2 {
            let mut c = vec![cards[0]];
            c.extend_from_slice(&wilds[..2]);
            combos.push(make(TRIPLE, rank, c, 0, 2));
        }
    }
}

// ─── FULL HOUSES ────────────────────────────────────────

struct Component {
    rank: u8,
    cards: Vec<Card>,
    wilds_used: u8,
}

fn collect_triples_and_pairs(
    by_rank: &ByRank,
    wilds: &[Card],
    level_rank: u8,
) -> (Vec<Component>, Vec<Component>) {
    let n_wild = wilds.len();
    let mut triples: Vec<Component> = Vec::new();
    let mut pairs: Vec<Component> = Vec::new();

    for (&rank, cards) in by_rank {
        if rank >= RANK_BLACK_JOKER {
            continue;
        }
        let n = cards.len();

        // Triples
        if n >= 3 {
            triples.push(Component {
                rank,
                cards: cards[..3].to_vec(),
                wilds_used: 0,
            });
        }
        if n >= 2 && n_wild >= 1 {
            let mut c = cards[..2].to_vec();
            c.push(wilds[0]);
            triples.push(Component {
                rank,
                cards: c,
                wilds_used: 1,
            });
        }
        if n >= 1 && n_wild >= 2 {
            let mut c = vec![cards[0]];
            c.extend_from_slice(&wilds[..2]);
            triples.push(Component {
                rank,
                cards: c,
                wilds_used: 2,
            });
        }

        // Pairs
        if n >= 2 {
            pairs.push(Component {
                rank,
                cards: cards[..2].to_vec(),
                wilds_used: 0,
            });
        }
        if n >= 1 && n_wild >= 1 {
            pairs.push(Component {
                rank,
                cards: vec![cards[0], wilds[0]],
                wilds_used: 1,
            });
        }
    }

    // Pair of wilds
    if n_wild >= 2 {
        pairs.push(Component {
            rank: level_rank,
            cards: wilds[..2].to_vec(),
            wilds_used: 2,
        });
    }

    (triples, pairs)
}

fn add_full_houses(
    combos: &mut Vec<Combo>,
    by_rank: &ByRank,
    wilds: &[Card],
    level_rank: u8,
) {
    let n_wild = wilds.len();
    let (triples, pairs) = collect_triples_and_pairs(by_rank, wilds, level_rank);

    for t in &triples {
        for p in &pairs {
            if t.rank == p.rank {
                continue;
            }
            if (t.wilds_used + p.wilds_used) as usize > n_wild {
                continue;
            }
            // Check card overlap by identity (pointer-like: use (rank, suit, deck))
            let t_set: HashSet<(u8, u8, u8)> =
                t.cards.iter().map(|c| (c.rank, c.suit, c.deck)).collect();
            let p_set: HashSet<(u8, u8, u8)> =
                p.cards.iter().map(|c| (c.rank, c.suit, c.deck)).collect();
            if !t_set.is_disjoint(&p_set) {
                continue;
            }
            let mut all_cards = t.cards.clone();
            all_cards.extend_from_slice(&p.cards);
            combos.push(make(
                FULL_HOUSE,
                t.rank,
                all_cards,
                0,
                t.wilds_used + p.wilds_used,
            ));
        }
    }
}

// ─── SEQUENCES ──────────────────────────────────────────

/// Map sequence rank values to actual card ranks.
/// 1 = Ace-low, 2..13 = normal, 14 = Ace-high.
fn natural_rank_for_seq(rank_val: u8) -> u8 {
    if rank_val == 1 {
        RANK_ACE
    } else {
        rank_val
    }
}

fn get_natural_cards_at_rank(by_rank: &ByRank, rank_val: u8) -> Vec<Card> {
    let card_rank = natural_rank_for_seq(rank_val);
    by_rank
        .get(&card_rank)
        .map(|cards| {
            cards
                .iter()
                .filter(|c| c.rank < RANK_BLACK_JOKER)
                .copied()
                .collect()
        })
        .unwrap_or_default()
}

fn add_straights(
    combos: &mut Vec<Combo>,
    by_rank: &ByRank,
    wilds: &[Card],
    level_rank: u8,
) {
    let n_wild = wilds.len();

    for start in 1u8..=10 {
        let ranks_needed: Vec<u8> = (start..start + 5).collect();

        let mut available: Vec<Vec<Card>> = Vec::with_capacity(5);
        let mut gaps = 0usize;
        for &r in &ranks_needed {
            let cards_at = get_natural_cards_at_rank(by_rank, r);
            if cards_at.is_empty() {
                gaps += 1;
            }
            available.push(cards_at);
        }

        if gaps > n_wild {
            continue;
        }

        let top_rank = *ranks_needed.last().unwrap();
        enumerate_straight_selections(combos, &available, top_rank, wilds, n_wild);
    }
}

fn enumerate_straight_selections(
    combos: &mut Vec<Combo>,
    available: &[Vec<Card>],
    top_rank: u8,
    wilds: &[Card],
    n_wild: usize,
) {
    // Recursive selection matching Python's _enumerate_straight_selections
    fn select(
        combos: &mut Vec<Combo>,
        available: &[Vec<Card>],
        top_rank: u8,
        wilds: &[Card],
        n_wild: usize,
        idx: usize,
        chosen: &mut Vec<Card>,
        wilds_used: usize,
        suits_seen: &mut HashSet<u8>,
    ) {
        if idx == 5 {
            if suits_seen.len() > 1 {
                combos.push(make(
                    STRAIGHT,
                    top_rank,
                    chosen.clone(),
                    5,
                    wilds_used as u8,
                ));
            }
            return;
        }

        let cards_at = &available[idx];
        if !cards_at.is_empty() {
            let mut seen_suits: HashSet<u8> = HashSet::new();
            for c in cards_at {
                if seen_suits.insert(c.suit) {
                    chosen.push(*c);
                    let was_new = suits_seen.insert(c.suit);
                    select(
                        combos, available, top_rank, wilds, n_wild, idx + 1, chosen,
                        wilds_used, suits_seen,
                    );
                    if was_new {
                        suits_seen.remove(&c.suit);
                    }
                    chosen.pop();
                }
            }
        }
        // Use a wild for this position (only when no natural cards available)
        if cards_at.is_empty() && wilds_used < n_wild {
            chosen.push(wilds[wilds_used]);
            select(
                combos, available, top_rank, wilds, n_wild, idx + 1, chosen,
                wilds_used + 1, suits_seen,
            );
            chosen.pop();
        }
    }

    let mut chosen = Vec::with_capacity(5);
    let mut suits_seen = HashSet::new();
    select(
        combos, available, top_rank, wilds, n_wild, 0, &mut chosen, 0, &mut suits_seen,
    );
}

fn add_tubes(combos: &mut Vec<Combo>, by_rank: &ByRank, wilds: &[Card]) {
    let n_wild = wilds.len();

    for start in 1u8..=12 {
        let ranks_needed: Vec<u8> = (start..start + 3).collect();

        let mut total_gaps = 0usize;
        let mut available: Vec<Vec<Card>> = Vec::with_capacity(3);
        for &r in &ranks_needed {
            let cards_at = get_natural_cards_at_rank(by_rank, r);
            let have = cards_at.len();
            let gap = if have < 2 { 2 - have } else { 0 };
            total_gaps += gap;
            available.push(cards_at);
        }

        if total_gaps > n_wild {
            continue;
        }

        let top_rank = *ranks_needed.last().unwrap();
        let mut selected: Vec<Card> = Vec::with_capacity(6);
        let mut wilds_used = 0usize;
        let mut valid = true;

        for (i, _) in ranks_needed.iter().enumerate() {
            let cards_at = &available[i];
            let have = cards_at.len();
            if have >= 2 {
                selected.extend_from_slice(&cards_at[..2]);
            } else if have == 1 {
                selected.push(cards_at[0]);
                if wilds_used < n_wild {
                    selected.push(wilds[wilds_used]);
                    wilds_used += 1;
                } else {
                    valid = false;
                    break;
                }
            } else {
                if wilds_used + 2 <= n_wild {
                    selected.push(wilds[wilds_used]);
                    selected.push(wilds[wilds_used + 1]);
                    wilds_used += 2;
                } else {
                    valid = false;
                    break;
                }
            }
        }

        if valid && selected.len() == 6 {
            combos.push(make(TUBE, top_rank, selected, 0, wilds_used as u8));
        }
    }
}

fn add_plates(combos: &mut Vec<Combo>, by_rank: &ByRank, wilds: &[Card]) {
    let n_wild = wilds.len();

    for start in 1u8..=13 {
        let ranks_needed: Vec<u8> = (start..start + 2).collect();

        let mut total_gaps = 0usize;
        let mut available: Vec<Vec<Card>> = Vec::with_capacity(2);
        for &r in &ranks_needed {
            let cards_at = get_natural_cards_at_rank(by_rank, r);
            let have = cards_at.len();
            let gap = if have < 3 { 3 - have } else { 0 };
            total_gaps += gap;
            available.push(cards_at);
        }

        if total_gaps > n_wild {
            continue;
        }

        let top_rank = *ranks_needed.last().unwrap();
        let mut selected: Vec<Card> = Vec::with_capacity(6);
        let mut wilds_used = 0usize;
        let mut valid = true;

        for (i, _) in ranks_needed.iter().enumerate() {
            let cards_at = &available[i];
            let have = cards_at.len();
            let take_natural = have.min(3);
            selected.extend_from_slice(&cards_at[..take_natural]);
            let need_wilds = 3 - take_natural;
            if wilds_used + need_wilds <= n_wild {
                for _ in 0..need_wilds {
                    selected.push(wilds[wilds_used]);
                    wilds_used += 1;
                }
            } else {
                valid = false;
                break;
            }
        }

        if valid && selected.len() == 6 {
            combos.push(make(PLATE, top_rank, selected, 0, wilds_used as u8));
        }
    }
}

// ─── BOMBS ──────────────────────────────────────────────

fn add_nofakind_bombs(combos: &mut Vec<Combo>, by_rank: &ByRank, wilds: &[Card]) {
    let n_wild = wilds.len();

    for (&rank, cards) in by_rank {
        if rank >= RANK_BLACK_JOKER {
            continue;
        }
        let n = cards.len();
        let max_size = (n + n_wild).min(10);
        for size in 4..=max_size {
            let needed_wilds = if size > n { size - n } else { 0 };
            if needed_wilds > n_wild {
                continue;
            }
            if let Some(bomb_type) = bomb_size_to_type(size) {
                let take_natural = n.min(size);
                let mut used = cards[..take_natural].to_vec();
                used.extend_from_slice(&wilds[..needed_wilds]);
                combos.push(make(bomb_type, rank, used, 0, needed_wilds as u8));
            }
        }
    }
}

fn add_straight_flushes(
    combos: &mut Vec<Combo>,
    hand: &HashSet<Card>,
    wilds: &[Card],
    level_rank: u8,
) {
    let n_wild = wilds.len();

    // Organize non-joker, non-wild cards by suit then rank
    let mut by_suit_rank: HashMap<u8, HashMap<u8, Vec<Card>>> = HashMap::new();
    for suit in 0..4u8 {
        by_suit_rank.insert(suit, HashMap::new());
    }
    for c in hand {
        if c.rank >= RANK_BLACK_JOKER {
            continue;
        }
        if is_wild(c, level_rank) {
            continue;
        }
        by_suit_rank
            .entry(c.suit)
            .or_default()
            .entry(c.rank)
            .or_default()
            .push(*c);
    }

    for suit in 0..4u8 {
        let suit_cards = &by_suit_rank[&suit];

        for start in 1u8..=10 {
            let ranks_needed: Vec<u8> = (start..start + 5).collect();
            let top_rank = *ranks_needed.last().unwrap();

            let mut selected: Vec<Card> = Vec::with_capacity(5);
            let mut wilds_used = 0usize;
            let mut valid = true;

            for &r in &ranks_needed {
                let card_rank = natural_rank_for_seq(r);
                let cards_at = suit_cards.get(&card_rank);
                if let Some(cards) = cards_at {
                    if !cards.is_empty() {
                        selected.push(cards[0]);
                        continue;
                    }
                }
                if wilds_used < n_wild {
                    selected.push(wilds[wilds_used]);
                    wilds_used += 1;
                } else {
                    valid = false;
                    break;
                }
            }

            if valid && selected.len() == 5 {
                combos.push(make(
                    STRAIGHT_FLUSH,
                    top_rank,
                    selected,
                    0,
                    wilds_used as u8,
                ));
            }
        }
    }
}

// ─── Strategic deduplication ──────────────────────────────
//
// Collapses suit-variants of strategically-equivalent plays. The encoder is
// suit-agnostic for most combo types, so emitting all suit variants only adds
// redundant NN evaluations without changing the action distribution. Used by
// both the Rust rollout loop and the Python `select_legal` helper.

/// Combo types where suit composition does not affect strength under `beats()`.
/// Excluded: PASS=0, STRAIGHT_FLUSH=10, BOMB_JOKER=16 (identity is intrinsic).
pub fn is_suit_agnostic(combo_type: u8) -> bool {
    matches!(combo_type, 1..=9 | 11..=15)
}

/// Canonical key under which suit-variants of the same play collapse.
pub fn strategic_key(combo: &Combo) -> Vec<u8> {
    let mut key = vec![combo.combo_type, combo.key, combo.length, combo.wild_count];
    if is_suit_agnostic(combo.combo_type) {
        let mut ranks: Vec<u8> = combo.cards.iter().map(|c| c.rank).collect();
        ranks.sort_unstable();
        key.extend(ranks);
    } else {
        let mut card_ids: Vec<(u8, u8)> = combo.cards.iter().map(|c| (c.rank, c.suit)).collect();
        card_ids.sort_unstable();
        for (rank, suit) in card_ids {
            key.push(rank);
            key.push(suit);
        }
    }
    key
}

/// Collapse suit-variants of strategically-equivalent plays; first-seen wins.
pub fn dedup_strategic(legal: Vec<Combo>) -> Vec<Combo> {
    let mut seen: HashSet<Vec<u8>> = HashSet::new();
    let mut out = Vec::with_capacity(legal.len());
    for combo in legal {
        let key = strategic_key(&combo);
        if seen.insert(key) {
            out.push(combo);
        }
    }
    out
}

// ─── DEDUPLICATION ──────────────────────────────────────

fn deduplicate(combos: Vec<Combo>) -> Vec<Combo> {
    // Ignore deck field — two combos using different deck copies of the same
    // rank+suit cards are strategically identical.
    let mut seen: HashSet<(u8, u8, u8, Vec<(u8, u8)>)> = HashSet::new();
    let mut result = Vec::new();

    for c in combos {
        let mut card_ids: Vec<(u8, u8)> =
            c.cards.iter().map(|card| (card.rank, card.suit)).collect();
        card_ids.sort();
        let key = (c.combo_type, c.key, c.length, card_ids);
        if seen.insert(key) {
            result.push(c);
        }
    }

    result
}
