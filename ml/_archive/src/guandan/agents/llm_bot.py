"""LLMBot — 3-stage + ToM LLM agent for Guan Dan using GPT-5.4 Nano via LiteLLM.

Pipeline per move:
  1. compute_cooperative_flags() — GuanZero-style intent options
  2. Stage 1 (conditional): LLM classifies cooperative intent
  3. filter_by_intent() — JidanBot-scored candidate filtering
  4. Stage 2.5 (conditional): ToM belief inference about opponent/partner hands
  5. Stage 3: LLM selects best move from candidates (with ToM context injected)
  6. Fallback: JidanBot's top pick on parse/API failure

Production-fair: only uses info a real player would know (own hand + counts + history).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from ..cards import Rank
from ..cards import ComboType
from ..combos import Combo
from .base import Agent
from .rl_recommender import score_candidates as rl_score_candidates
from .llm_prompts import (
    compute_cooperative_flags,
    filter_by_intent,
    format_candidates,
    format_game_state,
    format_tom_beliefs,
    get_system_prompt,
    parse_intent,
    parse_move_index,
)

if TYPE_CHECKING:
    from ..game import GuanDanEnv


def _load_dotenv() -> None:
    """Walk up from this file to find .env and load it into os.environ."""
    here = Path(__file__).resolve()
    for parent in [here.parent] + list(here.parents):
        env_file = parent / ".env"
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, _, v = line.partition("=")
                        os.environ.setdefault(k.strip(), v.strip())
            break


_load_dotenv()

try:
    import litellm
    litellm.suppress_debug_info = True
    _LITELLM_AVAILABLE = True
except ImportError:
    _LITELLM_AVAILABLE = False

_TOM_INSTRUCTION = (
    "Based on the player profiles above, infer what card types each player likely holds.\n"
    "Consider: ranks/combos they've played, cards remaining, bomb likelihood.\n"
    "Give 1-2 sentences per player. End with: DANGER: P<seat> (most threatening) or DANGER: NONE.\n"
    "Stay under 100 words total."
)


class LLMBot(Agent):
    """GPT-5.4 Nano agent with GuanZero cooperative flags + Theory of Mind reasoning."""

    DEFAULT_MODEL = "gpt-5.4-nano"

    def __init__(
        self,
        level_rank: int = Rank.TWO,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.3,
        max_retries: int = 2,
        top_k: int = 8,
        enable_tom: bool = True,
        tom_level: int = 1,         # 0=off, 1=1st-order, 2=2nd-order
        tom_min_opp: int = 15,      # activate when min opp cards ≤ this
        tom_min_history: int = 12,  # OR when move_history length ≥ this
    ):
        if not _LITELLM_AVAILABLE:
            raise ImportError(
                "litellm is required for LLMBot. Install with: pip install litellm"
            )

        self.level_rank = level_rank
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.top_k = top_k
        self.enable_tom = enable_tom
        self.tom_level = tom_level
        self.tom_min_opp = tom_min_opp
        self.tom_min_history = tom_min_history
        self._system_prompt = get_system_prompt(level_rank)

        # Diagnostics
        self._total_calls: int = 0
        self._total_tokens: int = 0
        self._fallback_count: int = 0
        self._tom_calls: int = 0
        self._intent_counts: dict[str, int] = {
            "cooperate": 0, "dwarf": 0, "assist": 0, "normal": 0
        }

    def _call_llm(self, user_prompt: str, max_tokens: int = 80) -> str | None:
        """Single LLM call. Returns response text or None on failure."""
        try:
            resp = litellm.completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=max_tokens,
            )
            self._total_calls += 1
            if resp.usage:
                self._total_tokens += resp.usage.total_tokens
            return resp.choices[0].message.content
        except Exception:
            return None

    def _should_run_tom(self, env: GuanDanEnv, flags: dict, intent: str) -> bool:
        if not self.enable_tom or self.tom_level < 1:
            return False
        if intent == "cooperate":
            return False  # stage 3 is skipped anyway
        return (
            flags["min_opp_remaining"] <= self.tom_min_opp
            or len(env.move_history) >= self.tom_min_history
        )

    def act(self, env: GuanDanEnv, player: int) -> Combo:
        legal = env.legal_moves(player)

        # No choice to make
        if len(legal) == 1:
            return legal[0]

        flags = compute_cooperative_flags(env, player, legal)
        state_prompt = format_game_state(env, player, flags, self.level_rank)

        # RL Q-network scoring — contextual ranking (falls back to static on error)
        rl_scores = rl_score_candidates(env, player, legal, self.level_rank)

        # ── Stage 1: Intent classification ────────────────────────────────────
        intent = "normal"
        any_flag = flags["can_cooperate"] or flags["can_dwarf"] or flags["can_assist"]
        if any_flag:
            intent_prompt = (
                state_prompt
                + "\n\nChoose your cooperative intent for this turn.\n"
                "Output ONLY one word: cooperate / dwarf / assist / normal"
            )
            raw = self._call_llm(intent_prompt, max_tokens=10)
            if raw:
                intent = parse_intent(raw)
        self._intent_counts[intent] += 1

        # ── Stage 2: Filter by intent (RL-scored) ─────────────────────────────
        candidates = filter_by_intent(
            legal, intent, flags, self.level_rank, self.top_k,
            scores=rl_scores,
        )

        if len(candidates) == 1:
            return candidates[0]  # cooperate → PASS, or only one option

        # ── Stage 2.5: ToM belief inference ───────────────────────────────────
        tom_context = ""
        if self._should_run_tom(env, flags, intent):
            profiles = format_tom_beliefs(env, player, flags, self.level_rank)
            tom_raw = self._call_llm(
                profiles + "\n\n" + _TOM_INSTRUCTION,
                max_tokens=150,
            )
            if tom_raw:
                tom_context = f"\nPLAYER BELIEFS:\n{tom_raw.strip()}\n"
                self._tom_calls += 1

            # 2nd-order ToM: what do opponents think you hold?
            if self.tom_level >= 2 and tom_context:
                tom2_raw = self._call_llm(
                    profiles
                    + f"\n\nWhat do opponents P{flags['left_opp']} and "
                    f"P{flags['right_opp']} likely think you hold, "
                    "based on your moves so far? 1-2 sentences.",
                    max_tokens=80,
                )
                if tom2_raw:
                    tom_context += f"\nOPPONENT VIEW OF YOU:\n{tom2_raw.strip()}\n"
                    self._tom_calls += 1

        # ── Stage 3: Move selection ────────────────────────────────────────────
        move_prompt = state_prompt + tom_context + format_candidates(candidates, self.level_rank)
        for _ in range(self.max_retries + 1):
            raw = self._call_llm(move_prompt, max_tokens=60)
            if raw:
                idx = parse_move_index(raw, len(candidates))
                if idx is not None:
                    return candidates[idx]

        # Fallback: JidanBot's best pick (highest score = last non-pass candidate)
        self._fallback_count += 1
        best = next(
            (c for c in reversed(candidates) if c.type != ComboType.PASS),
            candidates[-1],
        )
        return best

    @property
    def stats(self) -> dict:
        """Return diagnostic stats for eval reporting."""
        n = max(1, self._total_calls)
        return {
            "total_calls": self._total_calls,
            "total_tokens": self._total_tokens,
            "fallback_count": self._fallback_count,
            "fallback_rate": self._fallback_count / n,
            "intent_counts": dict(self._intent_counts),
            "tom_calls": self._tom_calls,
            "tom_call_rate": self._tom_calls / n,
        }
