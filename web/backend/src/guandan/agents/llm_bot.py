"""LLMBot — 3-stage LLM agent for Guan Dan using GPT-5.4 Nano via LiteLLM.

Pipeline per move:
  1. compute_cooperative_flags() — GuanZero-style intent options
  2. Stage 1 (conditional): LLM classifies cooperative intent
  3. filter_by_intent() — JidanBot-scored candidate filtering
  4. Stage 3: LLM selects best move from candidates
  5. Fallback: JidanBot's top pick on parse/API failure

Production-fair: only uses info a real player would know (own hand + counts + history).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from ..cards import Rank
from ..combos import Combo
from .base import Agent
from .llm_prompts import (
    compute_cooperative_flags,
    filter_by_intent,
    format_candidates,
    format_game_state,
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


class LLMBot(Agent):
    """GPT-5.4 Nano agent with GuanZero-style cooperative intent reasoning."""

    DEFAULT_MODEL = "gpt-5.4-nano"

    def __init__(
        self,
        level_rank: int = Rank.TWO,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.3,
        max_retries: int = 2,
        top_k: int = 8,
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
        self._system_prompt = get_system_prompt(level_rank)

        # Diagnostics
        self._total_calls: int = 0
        self._total_tokens: int = 0
        self._fallback_count: int = 0
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

    def act(self, env: GuanDanEnv, player: int) -> Combo:
        legal = env.legal_moves(player)

        # No choice to make
        if len(legal) == 1:
            return legal[0]

        flags = compute_cooperative_flags(env, player, legal)
        state_prompt = format_game_state(env, player, flags, self.level_rank)

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

        # ── Stage 2: Filter by intent ──────────────────────────────────────────
        candidates = filter_by_intent(legal, intent, flags, self.level_rank, self.top_k)

        if len(candidates) == 1:
            return candidates[0]  # cooperate → PASS, or only one option

        # ── Stage 3: Move selection ────────────────────────────────────────────
        move_prompt = state_prompt + format_candidates(candidates, self.level_rank)
        for _ in range(self.max_retries + 1):
            raw = self._call_llm(move_prompt, max_tokens=80)
            if raw:
                idx = parse_move_index(raw, len(candidates))
                if idx is not None:
                    return candidates[idx]

        # Fallback: JidanBot's top pick
        self._fallback_count += 1
        return candidates[0]

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
        }
