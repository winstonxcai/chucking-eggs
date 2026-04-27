"""Self-play rollout collector for pvguan PPO.

Complete-hand collection: each iter runs full hands until decisions_collected
reaches the target. May overshoot (each hand adds ~135 decisions).

Per-player tracks: decisions stored in the acting player's track only.
GAE bootstraps from the same player's next decision — see buffer.py.

Deterministic deal schedule: (deal_seed, level_seed) derived from
(global_run_seed, global_hand_index) via hash, so matched PV-AC/PV-PTIE
seeds draw from the same deterministic deal stream.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

from ..agents.partner_oracle_bot import _reflect_env
from ..cards import Rank
from ..combos import Combo
from ..game import GuanDanEnv
from .actor_critic import ActorCriticNet
from .buffer import Decision, PlayerTrack, RolloutBuffer
from .encoders import (
    encode_action,
    encode_actor_pair_features,
    encode_critic_state,
)


def _hash64(seed: int, idx: int, tag: bytes = b"") -> int:
    """Deterministic 64-bit hash of (seed, idx, tag). Used for deal/level seeds."""
    data = struct.pack(">qq", seed, idx) + tag
    digest = hashlib.sha256(data).digest()
    return struct.unpack(">Q", digest[:8])[0]


@dataclass
class HandScheduler:
    """Issues deterministic (global_hand_index, deal_seed, level_seed) tuples.

    Independent of worker timing — coordinator calls next_hand() sequentially.
    """
    global_run_seed: int
    _counter: int = 0

    def next_hand(self) -> tuple[int, int, int]:
        idx = self._counter
        self._counter += 1
        deal_seed  = int(_hash64(self.global_run_seed, idx)              % (2**31))
        level_seed = int(_hash64(self.global_run_seed, idx, b"level")    % 13)
        return (idx, deal_seed, level_seed)

    @staticmethod
    def level_from_seed(level_seed: int) -> int:
        """Map level_seed ∈ [0,12] → rank ∈ {2..A=14}."""
        return Rank.TWO + level_seed   # 2..14


@dataclass
class RolloutConfig:
    target_decisions: int = 4096
    critic_mode: Literal["pv", "ptie"] = "pv"
    temperature: float = 1.0


def _canonical(player: int) -> tuple[GuanDanEnv | None, int]:
    """Return (None, player) if player ∈ {0,2} (no reflection needed).
    Reflection is applied by the caller; this is just the mapping function.
    """
    if player in (0, 2):
        return (False, player)
    # player 1 → canonical 0, player 3 → canonical 2
    return (True, player ^ 1)


def collect_rollout(
    net: ActorCriticNet,
    scheduler: HandScheduler,
    cfg: RolloutConfig,
    device: torch.device,
) -> RolloutBuffer:
    """Run complete hands until decisions >= cfg.target_decisions.

    Returns a RolloutBuffer with finalized per-player tracks.
    """
    net.eval()
    buf = RolloutBuffer()
    decisions_collected = 0

    while decisions_collected < cfg.target_decisions:
        _, deal_seed, level_seed = scheduler.next_hand()
        level_rank = HandScheduler.level_from_seed(level_seed)

        env = GuanDanEnv(level_rank=level_rank)
        env.reset(seed=deal_seed)

        # Per-player open tracks for this hand
        tracks: dict[int, PlayerTrack] = {p: PlayerTrack(player=p) for p in range(4)}

        while not env.done:
            player = env.current_player
            needs_reflect, canonical_player = _canonical(player)

            if needs_reflect:
                enc_env = _reflect_env(env)
            else:
                enc_env = env

            legal = enc_env.legal_moves(canonical_player)

            if len(legal) == 1 and legal[0].type.name == "PASS":
                # Forced pass — no decision to record
                env.step(legal[0])
                continue

            hand = list(enc_env.hands[canonical_player])

            # Encode candidates
            state_actor_list = []
            action_list = []
            for move in legal:
                sf = encode_actor_pair_features(enc_env, canonical_player, move, legal)
                af = encode_action(move, hand, enc_env.level_rank)
                state_actor_list.append(sf)
                action_list.append(af)

            sa = torch.tensor(
                np.array(state_actor_list, dtype=np.float32), device=device
            )
            ac = torch.tensor(
                np.array(action_list, dtype=np.float32), device=device
            )
            K = len(legal)
            mask = torch.ones(K, dtype=torch.bool, device=device)

            with torch.no_grad():
                logits = net.policy_logits(sa, ac, mask)
                log_probs_all = torch.nn.functional.log_softmax(logits, dim=0)
                dist = torch.distributions.Categorical(logits=logits)
                sampled_idx = dist.sample().item()
                log_prob = log_probs_all[sampled_idx].item()

                # Critic state (action-independent)
                state_critic = encode_critic_state(enc_env, canonical_player, cfg.critic_mode)
                sc_tensor = torch.tensor(
                    state_critic[None], dtype=torch.float32, device=device
                )
                v_old = net.value(sc_tensor).item()

            decision = Decision(
                state_actor=np.array(state_actor_list, dtype=np.float32),
                actions=np.array(action_list, dtype=np.float32),
                legal_mask=np.ones(K, dtype=bool),
                sampled_idx=int(sampled_idx),
                log_prob=float(log_prob),
                state_critic=state_critic,
                v_old=float(v_old),
            )
            tracks[player].add(decision)
            decisions_collected += 1

            chosen = legal[int(sampled_idx)]
            env.step(chosen)

        # Hand done — assign terminal rewards and finalize tracks
        rewards = env.get_rewards()
        for p in range(4):
            if tracks[p].decisions:
                tracks[p].finalize(rewards[p] / 3.0)
                buf.add_track(tracks[p])

    net.train()
    return buf
