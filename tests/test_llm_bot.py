"""Tests for LLMBot — all LLM calls mocked."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from guandan.agents.llm_prompts import (
    _jidan_score,
    compute_cooperative_flags,
    filter_by_intent,
    format_candidates,
    format_game_state,
    get_system_prompt,
    parse_intent,
    parse_move_index,
)
from guandan.cards import ComboType, Rank, Suit
from guandan.combos import Combo
from guandan.game import GuanDanEnv


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_combo(ctype, key, *cards_tuples):
    """Helper: build a Combo from (rank, suit, deck) tuples."""
    from guandan.cards import Card
    cards = [Card(*t) for t in cards_tuples]
    return Combo(ctype, key, cards)


def _pass_combo():
    return Combo(ComboType.PASS, 0, [])


@pytest.fixture
def env():
    e = GuanDanEnv(level_rank=Rank.TWO)
    e.reset(seed=42)
    return e


# ── compute_cooperative_flags ─────────────────────────────────────────────────

class TestCooperativeFlags:
    def test_can_cooperate_when_partner_winning(self, env):
        """If partner is winning the current trick and PASS is legal, can_cooperate=True."""
        # Set up: trick exists, player 0's turn, partner (P2) is winning
        env.current_trick = _make_combo(ComboType.SINGLE, 5, (5, 0, 0))
        env.trick_winner = 2  # partner of P0
        env.current_player = 0

        legal = [_pass_combo(), _make_combo(ComboType.SINGLE, 7, (7, 0, 0))]
        flags = compute_cooperative_flags(env, player=0, legal=legal)

        assert flags["can_cooperate"] is True
        assert flags["partner"] == 2

    def test_no_cooperate_when_opponent_winning(self, env):
        env.current_trick = _make_combo(ComboType.SINGLE, 8, (8, 0, 0))
        env.trick_winner = 1  # opponent
        env.current_player = 0

        legal = [_pass_combo(), _make_combo(ComboType.SINGLE, 9, (9, 0, 0))]
        flags = compute_cooperative_flags(env, player=0, legal=legal)

        assert flags["can_cooperate"] is False

    def test_can_dwarf_when_combo_gte_opp_remaining(self, env):
        """If any legal combo has >= cards than min opponent hand, can_dwarf=True."""
        from guandan.cards import Card
        player = 0
        partner = 2
        left_opp = 1
        right_opp = 3

        # Manually set hand sizes (indirect via len(env.hands))
        # Give right opp just 3 cards
        env.hands[right_opp] = {Card(3, 0, 0), Card(4, 0, 0), Card(5, 0, 0)}
        env.hands[left_opp] = {Card(6, 0, 0)}  # 1 card

        # Legal move with 3 cards (triple) — qualifies for dwarf vs right_opp
        triple = _make_combo(ComboType.TRIPLE, 7, (7, 0, 0), (7, 1, 0), (7, 2, 0))
        legal = [_pass_combo(), triple]

        flags = compute_cooperative_flags(env, player=0, legal=legal)
        # min_opp_remaining = min(1, 3) = 1; triple has 3 cards >= 1 → can_dwarf
        assert flags["can_dwarf"] is True

    def test_can_assist_when_combo_lt_partner_remaining(self, env):
        from guandan.cards import Card
        env.hands[2] = set(list(env.hands[2])[:12])  # partner has 12 cards

        single = _make_combo(ComboType.SINGLE, 3, (3, 0, 0))  # 1 card < 12
        legal = [_pass_combo(), single]

        flags = compute_cooperative_flags(env, player=0, legal=legal)
        assert flags["can_assist"] is True
        assert flags["partner_remaining"] == 12

    def test_no_flags_when_free_lead(self, env):
        """On free lead (current_trick=None), can_cooperate must be False."""
        env.current_trick = None
        env.trick_winner = None
        env.current_player = 0

        single = _make_combo(ComboType.SINGLE, 3, (3, 0, 0))
        legal = [single]

        flags = compute_cooperative_flags(env, player=0, legal=legal)
        assert flags["can_cooperate"] is False


# ── filter_by_intent ──────────────────────────────────────────────────────────

class TestFilterByIntent:
    def test_cooperate_returns_only_pass(self):
        pass_c = _pass_combo()
        single = _make_combo(ComboType.SINGLE, 5, (5, 0, 0))
        legal = [pass_c, single]
        flags = {"min_opp_remaining": 10, "partner_remaining": 15}

        result = filter_by_intent(legal, "cooperate", flags, Rank.TWO)
        assert len(result) == 1
        assert result[0].type == ComboType.PASS

    def test_normal_limits_to_top_k(self):
        pass_c = _pass_combo()
        moves = [
            _make_combo(ComboType.SINGLE, r, (r, 0, 0))
            for r in range(3, 14)
        ]
        legal = [pass_c] + moves  # 11 non-pass + PASS = 12 total
        flags = {"min_opp_remaining": 20, "partner_remaining": 20}

        result = filter_by_intent(legal, "normal", flags, Rank.TWO, top_k=5)
        # 5 non-pass + PASS = 6
        assert len(result) <= 6
        assert any(c.type == ComboType.PASS for c in result)

    def test_dwarf_filters_by_card_count(self):
        from guandan.cards import Card
        pass_c = _pass_combo()
        single = _make_combo(ComboType.SINGLE, 3, (3, 0, 0))  # 1 card
        triple = _make_combo(ComboType.TRIPLE, 7, (7, 0, 0), (7, 1, 0), (7, 2, 0))  # 3 cards
        legal = [pass_c, single, triple]
        flags = {"min_opp_remaining": 3, "partner_remaining": 20}

        result = filter_by_intent(legal, "dwarf", flags, Rank.TWO)
        non_pass = [c for c in result if c.type != ComboType.PASS]
        assert all(len(c.cards) >= 3 for c in non_pass)

    def test_assist_filters_by_partner_remaining(self):
        pass_c = _pass_combo()
        single = _make_combo(ComboType.SINGLE, 3, (3, 0, 0))   # 1 card
        triple = _make_combo(ComboType.TRIPLE, 7, (7, 0, 0), (7, 1, 0), (7, 2, 0))  # 3 cards
        legal = [pass_c, single, triple]
        flags = {"min_opp_remaining": 20, "partner_remaining": 2}

        # Only combos with < 2 cards qualify (1 card = single)
        result = filter_by_intent(legal, "assist", flags, Rank.TWO)
        non_pass = [c for c in result if c.type != ComboType.PASS]
        assert all(len(c.cards) < 2 for c in non_pass)

    def test_pass_always_included(self):
        pass_c = _pass_combo()
        single = _make_combo(ComboType.SINGLE, 5, (5, 0, 0))
        legal = [pass_c, single]
        flags = {"min_opp_remaining": 20, "partner_remaining": 20}

        result = filter_by_intent(legal, "normal", flags, Rank.TWO)
        assert any(c.type == ComboType.PASS for c in result)


# ── Parsers ───────────────────────────────────────────────────────────────────

class TestParseIntent:
    def test_exact_words(self):
        assert parse_intent("cooperate") == "cooperate"
        assert parse_intent("DWARF") == "dwarf"
        assert parse_intent("Assist") == "assist"
        assert parse_intent("normal") == "normal"

    def test_embedded_in_text(self):
        assert parse_intent("I think I should cooperate here") == "cooperate"

    def test_fallback_on_garbage(self):
        assert parse_intent("xyz blah blah") == "normal"
        assert parse_intent("") == "normal"


class TestParseMoveIndex:
    def test_valid_index(self):
        assert parse_move_index("2", 5) == 2
        assert parse_move_index("0", 3) == 0

    def test_last_line_extracted(self):
        text = "I think the best move is to shed weak cards.\n3"
        assert parse_move_index(text, 8) == 3

    def test_out_of_range_returns_none(self):
        assert parse_move_index("10", 5) is None

    def test_no_number_returns_none(self):
        assert parse_move_index("pass the trick", 5) is None

    def test_embedded_in_last_line(self):
        text = "Move [2] is best"
        assert parse_move_index(text, 5) == 2


# ── LLMBot integration (mocked) ───────────────────────────────────────────────

class TestLLMBot:
    def _make_bot(self):
        """Instantiate LLMBot with litellm mocked."""
        with patch.dict("sys.modules", {"litellm": MagicMock()}):
            from guandan.agents.llm_bot import LLMBot
            bot = LLMBot(level_rank=Rank.TWO, top_k=8)
            return bot

    def test_single_legal_move_skips_llm(self, env):
        """If only one legal move, return it without any LLM call."""
        with patch("guandan.agents.llm_bot.litellm") as mock_litellm:
            from guandan.agents.llm_bot import LLMBot
            bot = LLMBot(level_rank=Rank.TWO)

            single_legal = [_pass_combo()]
            with patch.object(env, "legal_moves", return_value=single_legal):
                result = bot.act(env, 0)

            mock_litellm.completion.assert_not_called()
            assert result == single_legal[0]

    def test_act_normal_intent_returns_combo(self, env):
        """Normal flow: no flags → stage 3 only → returns candidate."""
        from guandan.cards import Card
        with patch("guandan.agents.llm_bot.litellm") as mock_litellm:
            from guandan.agents.llm_bot import LLMBot

            # Mock LLM response for stage 3
            mock_resp = MagicMock()
            mock_resp.choices[0].message.content = "1"
            mock_resp.usage.total_tokens = 50
            mock_litellm.completion.return_value = mock_resp

            bot = LLMBot(level_rank=Rank.TWO)

            env.current_trick = None  # free lead
            env.trick_winner = None
            # Empty partner hand so can_assist=False (no combo < 0 cards)
            env.hands[2] = set()

            result = bot.act(env, 0)

            assert isinstance(result, Combo)
            # Only stage 3 called (no flags → no stage 1)
            assert mock_litellm.completion.call_count == 1

    def test_act_cooperate_skips_move_selection(self, env):
        """cooperate intent → only stage 1 called, PASS returned, no stage 3."""
        with patch("guandan.agents.llm_bot.litellm") as mock_litellm:
            from guandan.agents.llm_bot import LLMBot

            # Stage 1 returns "cooperate"
            intent_resp = MagicMock()
            intent_resp.choices[0].message.content = "cooperate"
            intent_resp.usage.total_tokens = 5
            mock_litellm.completion.return_value = intent_resp

            bot = LLMBot(level_rank=Rank.TWO)

            # Set up: partner winning trick, PASS is legal
            env.current_trick = _make_combo(ComboType.SINGLE, 9, (9, 0, 0))
            env.trick_winner = 2  # partner of P0
            env.current_player = 0

            from guandan.cards import Card
            single = _make_combo(ComboType.SINGLE, 5, (5, 0, 0))
            legal = [_pass_combo(), single]

            with patch.object(env, "legal_moves", return_value=legal):
                result = bot.act(env, 0)

            # Only 1 call (stage 1); stage 3 skipped because cooperate → PASS
            assert mock_litellm.completion.call_count == 1
            assert result.type == ComboType.PASS

    def test_fallback_on_api_error(self, env):
        """On repeated API failure, return JidanBot's top candidate."""
        with patch("guandan.agents.llm_bot.litellm") as mock_litellm:
            from guandan.agents.llm_bot import LLMBot

            mock_litellm.completion.side_effect = Exception("API error")

            bot = LLMBot(level_rank=Rank.TWO, max_retries=1)
            env.current_trick = None
            env.trick_winner = None

            result = bot.act(env, 0)

            # Should return a Combo (fallback to candidates[0])
            assert isinstance(result, Combo)
            assert bot._fallback_count == 1

    def test_fallback_on_bad_parse(self, env):
        """On unparseable response, fall back to candidates[0]."""
        with patch("guandan.agents.llm_bot.litellm") as mock_litellm:
            from guandan.agents.llm_bot import LLMBot

            bad_resp = MagicMock()
            bad_resp.choices[0].message.content = "I cannot decide."
            bad_resp.usage.total_tokens = 10
            mock_litellm.completion.return_value = bad_resp

            bot = LLMBot(level_rank=Rank.TWO, max_retries=1)
            env.current_trick = None
            env.trick_winner = None

            result = bot.act(env, 0)
            assert isinstance(result, Combo)


# ── System prompt sanity ──────────────────────────────────────────────────────

def test_system_prompt_contains_key_concepts():
    prompt = get_system_prompt(Rank.TWO)
    assert "wild" in prompt.lower()
    assert "partner" in prompt.lower()
    assert "bomb" in prompt.lower()
    assert "cooperate" in prompt.lower()


def test_format_candidates_numbered(env):
    moves = [
        _make_combo(ComboType.SINGLE, 3, (3, 0, 0)),
        _make_combo(ComboType.PAIR, 5, (5, 0, 0), (5, 1, 0)),
        _pass_combo(),
    ]
    result = format_candidates(moves, Rank.TWO)
    assert "[0]" in result
    assert "[1]" in result
    assert "[2]" in result
