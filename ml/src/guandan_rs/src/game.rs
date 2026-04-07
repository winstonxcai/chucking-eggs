//! 4-player Guan Dan game environment.
//!
//! Counterclockwise play, strict auto-pass logic (out players count as passing),
//! partner-leads rule, team-based game end.
//! Mirrors the Python GuanDanEnv exactly.

use std::collections::HashSet;

use crate::cards::{self, Card, PASS};
use crate::combos::{self, Combo};

use rand::seq::SliceRandom;
use rand::Rng;

/// Full game state — cheaply cloneable for MC rollouts.
#[derive(Clone)]
pub struct GameEnv {
    pub level_rank: u8,
    pub hands: [HashSet<Card>; 4],
    pub current_player: usize,
    pub current_trick: Option<Combo>,
    pub trick_winner: Option<usize>,
    pub consecutive_passes: u32,
    pub finish_order: Vec<usize>,
    pub is_out: [bool; 4],
    pub done: bool,
}

impl GameEnv {
    pub fn new(level_rank: u8) -> Self {
        let mut env = GameEnv {
            level_rank,
            hands: [HashSet::new(), HashSet::new(), HashSet::new(), HashSet::new()],
            current_player: 0,
            current_trick: None,
            trick_winner: None,
            consecutive_passes: 0,
            finish_order: Vec::with_capacity(4),
            is_out: [false; 4],
            done: false,
        };
        env.reset();
        env
    }

    pub fn reset(&mut self) {
        let mut rng = rand::rng();
        let mut deck = cards::make_deck();
        deck.shuffle(&mut rng);

        for h in self.hands.iter_mut() {
            h.clear();
        }
        for (i, card) in deck.into_iter().enumerate() {
            self.hands[i / 27].insert(card);
        }

        self.current_player = rng.random_range(0..4);
        self.current_trick = None;
        self.trick_winner = None;
        self.consecutive_passes = 0;
        self.finish_order.clear();
        self.is_out = [false; 4];
        self.done = false;
    }

    // ─── Seat helpers ─────────────────────────────────

    #[inline]
    pub fn partner(player: usize) -> usize {
        (player + 2) % 4
    }

    #[inline]
    fn next_seat(player: usize) -> usize {
        (player + 3) % 4  // counterclockwise = subtract 1 mod 4
    }

    fn next_active_player(&self, from: usize) -> Option<usize> {
        let mut p = Self::next_seat(from);
        for _ in 0..4 {
            if !self.is_out[p] {
                return Some(p);
            }
            p = Self::next_seat(p);
        }
        None
    }

    // ─── Game end logic ───────────────────────────────

    fn winning_team_done(&self) -> bool {
        (self.is_out[0] && self.is_out[2]) || (self.is_out[1] && self.is_out[3])
    }

    fn finalize_game(&mut self) {
        for p in 0..4 {
            if !self.finish_order.contains(&p) {
                self.finish_order.push(p);
            }
        }
        self.done = true;
    }

    // ─── Trick resolution ─────────────────────────────

    fn end_trick(&mut self) {
        self.current_trick = None;
    }

    fn resolve_leader(&self) -> usize {
        let winner = self.trick_winner.unwrap();
        if !self.is_out[winner] {
            return winner;
        }
        let partner = Self::partner(winner);
        if !self.is_out[partner] {
            return partner;
        }
        self.next_active_player(winner).unwrap()
    }

    fn count_auto_passes_between(&self, from: usize, to: usize) -> u32 {
        let mut count = 0;
        let mut seat = Self::next_seat(from);
        while seat != to {
            if self.is_out[seat] {
                count += 1;
            }
            seat = Self::next_seat(seat);
        }
        count
    }

    // ─── Main step ────────────────────────────────────

    pub fn step(&mut self, combo: &Combo) {
        let player = self.current_player;

        if combo.combo_type == PASS {
            self.handle_pass(player);
        } else {
            self.handle_play(player, combo);
        }
    }

    fn handle_pass(&mut self, player: usize) {
        self.consecutive_passes += 1;

        let next_p = self.next_active_player(player);

        if let Some(np) = next_p {
            let auto = self.count_auto_passes_between(player, np);
            self.consecutive_passes += auto;
        }

        if self.consecutive_passes >= 3 || next_p.is_none() {
            self.end_trick();
            if next_p.is_none() {
                self.finalize_game();
            } else {
                self.current_player = self.resolve_leader();
            }
            return;
        }

        let np = next_p.unwrap();
        if Some(np) == self.trick_winner {
            self.end_trick();
            self.current_player = self.resolve_leader();
            return;
        }

        self.current_player = np;
    }

    fn handle_play(&mut self, player: usize, combo: &Combo) {
        // Remove played cards from hand
        for card in &combo.cards {
            self.hands[player].remove(card);
        }

        self.current_trick = Some(combo.clone());
        self.trick_winner = Some(player);
        self.consecutive_passes = 0;

        // Check if player goes out
        if self.hands[player].is_empty() {
            self.finish_order.push(player);
            self.is_out[player] = true;

            if self.winning_team_done() {
                self.finalize_game();
                return;
            }
        }

        let next_p = self.next_active_player(player);
        if next_p.is_none() {
            self.finalize_game();
            return;
        }
        let np = next_p.unwrap();

        let auto = self.count_auto_passes_between(player, np);
        self.consecutive_passes = auto;

        if self.consecutive_passes >= 3 {
            self.end_trick();
            self.current_player = self.resolve_leader();
            return;
        }

        self.current_player = np;
    }

    // ─── Query methods ────────────────────────────────

    pub fn legal_moves(&self) -> Vec<Combo> {
        let hand = &self.hands[self.current_player];
        match &self.current_trick {
            None => combos::generate_all_leads(hand, self.level_rank),
            Some(trick) => combos::generate_responses(hand, self.level_rank, trick),
        }
    }

    pub fn get_rewards(&self) -> [f64; 4] {
        assert!(self.done);
        let pos0 = self.finish_order.iter().position(|&p| p == 0).unwrap();
        let pos2 = self.finish_order.iter().position(|&p| p == 2).unwrap();
        let (a, b) = if pos0 < pos2 { (pos0, pos2) } else { (pos2, pos0) };

        let r: f64 = match (a, b) {
            (0, 1) => 3.0,
            (0, 2) => 2.0,
            (0, 3) => 1.0,
            (1, 2) => -1.0,
            (1, 3) => -2.0,
            (2, 3) => -3.0,
            _ => 0.0,
        };
        [r, -r, r, -r]
    }
}

// ─── Greedy rollout bot ───────────────────────────────────────────────────────
// Simple but fast: pick the weakest legal move that beats the trick.
// For leading: play the weakest non-bomb combo.
// This is simpler than StrategicBot but runs entirely in Rust.

fn greedy_act(env: &GameEnv) -> Combo {
    let moves = env.legal_moves();
    if moves.len() == 1 {
        return moves[0].clone();
    }

    let is_leading = env.current_trick.is_none();

    if is_leading {
        // Lead: prefer weakest non-bomb, non-pass
        let mut best: Option<&Combo> = None;
        for m in &moves {
            if m.combo_type == PASS {
                continue;
            }
            if cards::is_bomb_type(m.combo_type) {
                continue;
            }
            if best.is_none() || m.key < best.unwrap().key {
                best = Some(m);
            }
        }
        if let Some(b) = best {
            return b.clone();
        }
        // Only bombs or pass — play weakest bomb
        for m in &moves {
            if m.combo_type != PASS {
                return m.clone();
            }
        }
        moves[0].clone()
    } else {
        // Follow: play weakest non-bomb beat, else pass
        let mut best_beat: Option<&Combo> = None;
        for m in &moves {
            if m.combo_type == PASS || cards::is_bomb_type(m.combo_type) {
                continue;
            }
            if best_beat.is_none() || m.key < best_beat.unwrap().key {
                best_beat = Some(m);
            }
        }
        if let Some(b) = best_beat {
            return b.clone();
        }
        // Pass if available
        for m in &moves {
            if m.combo_type == PASS {
                return m.clone();
            }
        }
        moves[0].clone()
    }
}

/// Run a single rollout from a cloned game state: play `move_combo` for
/// `player`, then all players use greedy bot until game ends.
/// Returns the reward for `player`.
pub fn rollout_one(env: &GameEnv, player: usize, move_combo: &Combo) -> f64 {
    let mut sim = env.clone();
    sim.step(move_combo);

    while !sim.done {
        let action = greedy_act(&sim);
        sim.step(&action);
    }

    sim.get_rewards()[player]
}

/// Run `n_sims` rollouts for `move_combo` and return the average reward.
pub fn evaluate_move(env: &GameEnv, player: usize, move_combo: &Combo, n_sims: usize) -> f64 {
    let total: f64 = (0..n_sims).map(|_| rollout_one(env, player, move_combo)).collect::<Vec<f64>>().iter().sum();
    total / n_sims as f64
}
