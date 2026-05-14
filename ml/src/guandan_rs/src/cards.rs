// Suit constants (matching Python IntEnum)
pub const SUIT_SPADE: u8 = 0;
pub const SUIT_HEART: u8 = 1;
pub const SUIT_DIAMOND: u8 = 2;
pub const SUIT_CLUB: u8 = 3;

// Rank constants (matching Python IntEnum)
pub const RANK_TWO: u8 = 2;
pub const RANK_THREE: u8 = 3;
pub const RANK_FOUR: u8 = 4;
pub const RANK_FIVE: u8 = 5;
pub const RANK_SIX: u8 = 6;
pub const RANK_SEVEN: u8 = 7;
pub const RANK_EIGHT: u8 = 8;
pub const RANK_NINE: u8 = 9;
pub const RANK_TEN: u8 = 10;
pub const RANK_JACK: u8 = 11;
pub const RANK_QUEEN: u8 = 12;
pub const RANK_KING: u8 = 13;
pub const RANK_ACE: u8 = 14;
// 15 is reserved for level_order_key
pub const RANK_BLACK_JOKER: u8 = 16;
pub const RANK_RED_JOKER: u8 = 17;

// ComboType constants (matching Python IntEnum)
pub const PASS: u8 = 0;
pub const SINGLE: u8 = 1;
pub const PAIR: u8 = 2;
pub const TRIPLE: u8 = 3;
pub const FULL_HOUSE: u8 = 4;
pub const STRAIGHT: u8 = 5;
pub const TUBE: u8 = 6;
pub const PLATE: u8 = 7;
pub const BOMB_4: u8 = 8;
pub const BOMB_5: u8 = 9;
pub const STRAIGHT_FLUSH: u8 = 10;
pub const BOMB_6: u8 = 11;
pub const BOMB_7: u8 = 12;
pub const BOMB_8: u8 = 13;
pub const BOMB_9: u8 = 14;
pub const BOMB_10: u8 = 15;
pub const BOMB_JOKER: u8 = 16;

#[derive(Clone, Copy, Hash, Eq, PartialEq, Debug)]
pub struct Card {
    pub rank: u8,
    pub suit: u8,
    pub deck: u8,
}

impl Card {
    pub fn new(rank: u8, suit: u8, deck: u8) -> Self {
        Card { rank, suit, deck }
    }
}

pub fn level_order_key(rank: u8, level_rank: u8) -> u8 {
    if rank == level_rank {
        15
    } else {
        rank
    }
}

pub fn is_wild(card: &Card, level_rank: u8) -> bool {
    card.rank == level_rank && card.suit == SUIT_HEART
}

pub fn is_bomb_type(combo_type: u8) -> bool {
    matches!(
        combo_type,
        BOMB_4 | BOMB_5 | STRAIGHT_FLUSH | BOMB_6 | BOMB_7 | BOMB_8 | BOMB_9 | BOMB_10
            | BOMB_JOKER
    )
}

pub fn bomb_size_to_type(size: usize) -> Option<u8> {
    match size {
        4 => Some(BOMB_4),
        5 => Some(BOMB_5),
        6 => Some(BOMB_6),
        7 => Some(BOMB_7),
        8 => Some(BOMB_8),
        9 => Some(BOMB_9),
        10 => Some(BOMB_10),
        _ => None,
    }
}

pub fn make_deck() -> Vec<Card> {
    let mut cards = Vec::with_capacity(108);
    for deck_id in 0..2u8 {
        for rank in 2..=14u8 {
            for suit in 0..4u8 {
                cards.push(Card::new(rank, suit, deck_id));
            }
        }
        cards.push(Card::new(RANK_BLACK_JOKER, 0, deck_id));
        cards.push(Card::new(RANK_RED_JOKER, 1, deck_id));
    }
    cards
}

/// Stable id in [0, 108) matching Python's `card_to_id`.
///
/// Layout:
/// - ranks 2..=A × 4 suits × 2 decks = 104 ids (0..=103)
///   - id = (rank − 2) × 8 + suit × 2 + deck
/// - BLACK_JOKER deck 0 → 104, deck 1 → 105
/// - RED_JOKER   deck 0 → 106, deck 1 → 107
pub fn card_to_id(card: &Card) -> usize {
    if card.rank == RANK_BLACK_JOKER {
        return 104 + card.deck as usize;
    }
    if card.rank == RANK_RED_JOKER {
        return 106 + card.deck as usize;
    }
    (card.rank as usize - 2) * 8 + card.suit as usize * 2 + card.deck as usize
}
