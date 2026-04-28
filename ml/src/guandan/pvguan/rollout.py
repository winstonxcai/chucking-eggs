"""Self-play / mixed-opponent rollout collector for pvguan PPO.

Complete-hand collection: each iter runs full hands until decisions_collected
reaches the target. May overshoot (each hand adds ~135 decisions in self-play,
~half that in mixed-opponent hands since only 2 of 4 seats record).

Per-player tracks: decisions stored in the acting player's track only.
GAE bootstraps from the same player's next decision — see buffer.py.

Deterministic deal schedule: (deal_seed, level_seed) derived from
(global_run_seed, global_hand_index) via hash, so matched PV-AC/PV-PTIE
seeds draw from the same deterministic deal stream. Mode (self-play vs
mixed-opponent) is also derived from deal_seed so it's reproducible.

Mixed-opponent mode: per hand, with prob `cfg.selfplay_frac` we play pure
self-play (all 4 seats = net). Otherwise pick a random bot from
`cfg.opponent_mix` and seat it at {1,3}; net plays {0,2}. Only seat-0/2
tracks enter the PPO buffer in mixed-opponent hands. This lets the policy
see real jidan/yaoji/strategic playstyles during training and avoids the
self-play distribution drift that caused the iter-200→400 WR collapse.

Parallel rollout: collect_rollout_parallel() spawns N workers (spawn context),
each loading network weights from a temp file and playing a fixed list of
hand specs sequentially on CPU. Workers return per-worker RolloutBuffers
which are merged via RolloutBuffer.from_buffers().
"""

from __future__ import annotations

import hashlib
import math
import multiprocessing as mp
import random
import struct
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch

from ..agents.partner_oracle_bot import _reflect_env
from ..agents.partner_pimc_bot import ROLLOUT_FACTORIES
from ..cards import Rank
from ..combos import Combo
from ..game import GuanDanEnv
from .actor_critic import ActorCriticNet
from .buffer import Decision, PlayerTrack, RolloutBuffer
from .encoders import (
    ACTOR_DIM,
    ACTION_DIM,
    CRITIC_DIM,
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
    max_legal_per_decision: int = 64   # cap legal moves for memory; bombs always kept
    # Mixed-opponent mode (see module docstring)
    opponent_mix:    tuple[str, ...] = ()
    selfplay_frac:   float = 1.0
    # Reactive curriculum: per-opponent sampling weights (unnormalized).
    # Updated by trainer after each validation eval via softmax(-wr/temp).
    # None = uniform (default).
    opp_weights: dict[str, float] | None = None
    # Going-out reward shaping (potential-based, preserves optimal policy).
    # 0.0 = off (legacy terminal-only). When > 0: player gets +shape at the
    # step they go out, and -shape adjustment to their terminal reward.
    going_out_shape: float = 0.0


def _cap_legal(
    legal: list,
    cap: int,
) -> list[int]:
    """Return indices into `legal` to keep, length ≤ cap.

    Always preserves PASS and any BOMB/STRAIGHT_FLUSH (strategic moves).
    Fills the remainder with the first cheapest-ranked moves (deterministic).
    """
    if len(legal) <= cap:
        return list(range(len(legal)))

    bomb_types = {
        "BOMB_4", "BOMB_5", "BOMB_6", "BOMB_7", "BOMB_8", "BOMB_9", "BOMB_10",
        "STRAIGHT_FLUSH", "BOMB_JOKER",
    }
    must_keep = [
        i for i, m in enumerate(legal)
        if m.type.name == "PASS" or m.type.name in bomb_types
    ]
    other = [i for i in range(len(legal)) if i not in set(must_keep)]
    # Deterministic order: keep first (cheapest) — engine returns moves in
    # ascending rank order within each combo type
    remaining = cap - len(must_keep)
    if remaining > 0:
        other_sorted = sorted(other, key=lambda i: (
            sum(c.rank for c in legal[i].cards), i
        ))
        keep = sorted(must_keep + other_sorted[:remaining])
    else:
        keep = sorted(must_keep)[:cap]
    return keep


def _canonical(player: int) -> tuple[GuanDanEnv | None, int]:
    """Return (None, player) if player ∈ {0,2} (no reflection needed).
    Reflection is applied by the caller; this is just the mapping function.
    """
    if player in (0, 2):
        return (False, player)
    # player 1 → canonical 0, player 3 → canonical 2
    return (True, player ^ 1)


def _play_one_hand(
    net: ActorCriticNet,
    deal_seed: int,
    level_seed: int,
    cfg: RolloutConfig,
    device: torch.device,
    buf: RolloutBuffer,
    opp_bots: dict[str, "object"] | None = None,
) -> int:
    """Play one hand, append decisions/tracks/stats to buf.
    Returns the number of net-controlled decisions collected.

    If `opp_bots` is set and (rng.random() >= cfg.selfplay_frac), seats {1,3}
    are controlled by a randomly chosen opponent bot; otherwise pure self-play.
    The mode RNG is keyed on deal_seed so the same hand always picks the same
    mode and opponent, preserving determinism across resumes.
    """
    level_rank = HandScheduler.level_from_seed(level_seed)
    env = GuanDanEnv(level_rank=level_rank)
    env.reset(seed=deal_seed)

    # Per-hand RNG keyed on deal_seed → reproducible mode & opponent choice
    rng = random.Random(deal_seed)

    if opp_bots and rng.random() >= cfg.selfplay_frac:
        names = list(opp_bots.keys())
        if cfg.opp_weights:
            weights = [cfg.opp_weights.get(n, 1.0) for n in names]
            opp_name = rng.choices(names, weights=weights, k=1)[0]
        else:
            opp_name = rng.choice(names)
        opp_bot  = opp_bots[opp_name]
        our_team: set[int] = {0, 2}
    else:
        opp_bot  = None
        our_team = {0, 1, 2, 3}

    tracks: dict[int, PlayerTrack] = {p: PlayerTrack(player=p) for p in range(4)}
    hand_decisions = 0

    while not env.done:
        player = env.current_player

        # Opponent seats: bot acts, no decision recorded
        if opp_bot is not None and player not in our_team:
            env.step(opp_bot.act(env, player))
            continue

        needs_reflect, canonical_player = _canonical(player)
        enc_env = _reflect_env(env) if needs_reflect else env

        legal = enc_env.legal_moves(canonical_player)

        if len(legal) == 1 and legal[0].type.name == "PASS":
            env.step(legal[0])
            continue

        if len(legal) > cfg.max_legal_per_decision:
            keep_idx = _cap_legal(legal, cfg.max_legal_per_decision)
            legal = [legal[i] for i in keep_idx]

        buf.stats.K_values.append(len(legal))

        hand = list(enc_env.hands[canonical_player])

        state_actor_list = []
        action_list = []
        for move in legal:
            sf = encode_actor_pair_features(enc_env, canonical_player, move, legal)
            af = encode_action(move, hand, enc_env.level_rank)
            state_actor_list.append(sf)
            action_list.append(af)

        sa = torch.tensor(np.array(state_actor_list, dtype=np.float32), device=device)
        ac = torch.tensor(np.array(action_list, dtype=np.float32), device=device)
        K = len(legal)
        mask = torch.ones(K, dtype=torch.bool, device=device)

        with torch.no_grad():
            logits = net.policy_logits(sa, ac, mask)
            log_probs_all = torch.nn.functional.log_softmax(logits, dim=0)
            dist = torch.distributions.Categorical(logits=logits)
            sampled_idx = dist.sample().item()
            log_prob = log_probs_all[sampled_idx].item()

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
        hand_decisions += 1

        chosen = legal[int(sampled_idx)]

        ct = chosen.type.name
        if ct == "PASS":
            buf.stats.pass_count += 1
        else:
            buf.stats.non_pass_count += 1
            if ct.startswith("BOMB") or ct == "STRAIGHT_FLUSH":
                buf.stats.bomb_count += 1
            if env.current_trick is None:
                buf.stats.lead_count += 1

        prev_out = env.is_out[player]
        env.step(chosen)

        # Going-out shaping: if this player just emptied their hand and they're
        # net-controlled, give +shape at this decision step. Terminal reward is
        # adjusted by -shape later so total per-track reward is preserved.
        if (cfg.going_out_shape > 0.0
                and player in our_team
                and env.is_out[player]
                and not prev_out):
            tracks[player].intermediate_rewards[len(tracks[player].decisions) - 1] = (
                cfg.going_out_shape
            )

    rewards = env.get_rewards()
    for p in our_team:
        if tracks[p].decisions:
            terminal = rewards[p] / 3.0
            if cfg.going_out_shape > 0.0 and tracks[p].intermediate_rewards:
                terminal -= cfg.going_out_shape
            tracks[p].finalize(terminal)
            buf.add_track(tracks[p])
    buf.stats.hand_lengths.append(hand_decisions)
    return hand_decisions


def _build_opp_bots(cfg: RolloutConfig) -> dict[str, object] | None:
    """Instantiate opponent bots from cfg.opponent_mix. Returns None if empty
    or selfplay_frac >= 1 (no mixed-opponent hands will be played)."""
    if not cfg.opponent_mix or cfg.selfplay_frac >= 1.0:
        return None
    return {
        name: ROLLOUT_FACTORIES[name](Rank.TWO)
        for name in cfg.opponent_mix
    }


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
    opp_bots = _build_opp_bots(cfg)
    decisions_collected = 0
    while decisions_collected < cfg.target_decisions:
        _, deal_seed, level_seed = scheduler.next_hand()
        decisions_collected += _play_one_hand(
            net, deal_seed, level_seed, cfg, device, buf, opp_bots
        )
    net.train()
    return buf


def _collect_for_hands(
    net: ActorCriticNet,
    hand_specs: list[tuple[int, int, int]],
    cfg: RolloutConfig,
    device: torch.device,
) -> RolloutBuffer:
    """Play exactly the supplied list of hands. Used by parallel workers."""
    net.eval()
    buf = RolloutBuffer()
    opp_bots = _build_opp_bots(cfg)
    for _idx, deal_seed, level_seed in hand_specs:
        _play_one_hand(net, deal_seed, level_seed, cfg, device, buf, opp_bots)
    return buf


def _rollout_worker(args: tuple) -> RolloutBuffer:
    """Module-level worker — picklable for spawn context.

    Loads network weights from disk, plays a fixed list of hands sequentially
    on CPU, returns its RolloutBuffer. Workers stay off the GPU to avoid
    contention with the main process's PPO update.
    """
    state_path, hand_specs, cfg, hidden = args
    # PyTorch's intra-op threads — keep small per worker so 48 workers don't
    # collectively spawn 48 × N threads and thrash the host
    torch.set_num_threads(1)
    net = ActorCriticNet(
        d_state_actor=ACTOR_DIM,
        d_action=ACTION_DIM,
        d_state_critic=CRITIC_DIM,
        hidden=hidden,
    )
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    net.load_state_dict(state)
    net.eval()
    return _collect_for_hands(net, hand_specs, cfg, torch.device("cpu"))


# mp.Queue cannot be pickled across pool.map — they must be inherited at
# worker-spawn via Pool's initializer. These are populated by
# init_gpu_worker_queues() at pool creation time and read by _rollout_worker_gpu.
_GPU_REQUEST_Q: "mp.Queue | None" = None
_GPU_REPLY_QS: "list[mp.Queue] | None" = None


def init_gpu_worker_queues(request_q: "mp.Queue", reply_qs: "list[mp.Queue]") -> None:
    """Pool initializer — stores GPU server queues in module globals so
    _rollout_worker_gpu can pick the right reply_q by worker_id."""
    global _GPU_REQUEST_Q, _GPU_REPLY_QS
    _GPU_REQUEST_Q = request_q
    _GPU_REPLY_QS = reply_qs


def _rollout_worker_gpu(args: tuple) -> RolloutBuffer:
    """GPU-server worker — uses a NetProxy that forwards calls via mp.Queue
    to the main process's GPU inference server. No model loaded in worker.
    Queues are read from module globals populated by init_gpu_worker_queues.
    """
    from .gpu_server import NetProxy
    worker_id, hand_specs, cfg = args
    if _GPU_REQUEST_Q is None or _GPU_REPLY_QS is None:
        raise RuntimeError("GPU worker queues not initialized — Pool needs "
                           "initializer=init_gpu_worker_queues")
    torch.set_num_threads(1)
    proxy = NetProxy(
        request_q=_GPU_REQUEST_Q,
        reply_q=_GPU_REPLY_QS[worker_id],
        worker_id=worker_id,
    )
    return _collect_for_hands(proxy, hand_specs, cfg, torch.device("cpu"))


def collect_rollout_parallel(
    net: ActorCriticNet,
    scheduler: HandScheduler,
    cfg: RolloutConfig,
    n_workers: int,
    pool: "mp.pool.Pool",
    state_path: str,
    hidden: int,
    mean_hand_length: float = 135.0,
    gpu_server: "object | None" = None,
) -> RolloutBuffer:
    """Multiprocess rollout. Pool and state_path are reused across iters.

    Two modes:
      - gpu_server is None (default): workers load weights from state_path
        and run NN forward on CPU.
      - gpu_server is a GPUInferenceServer: workers use a NetProxy that
        forwards scoring calls via mp.Queue to the server thread; weights
        are held in main process GPU memory and updated via update_weights().
    """
    # 1. Allocate hands per worker. Slight per-worker overshoot is fine —
    #    decision count is approximate (mean ~135 dec/hand).
    target_hands = max(
        math.ceil(cfg.target_decisions / mean_hand_length), n_workers
    )
    hands_per_worker = math.ceil(target_hands / n_workers)

    # 2. Pull contiguous, deterministic hand specs from scheduler (main only)
    worker_specs: list[list[tuple[int, int, int]]] = []
    for _ in range(n_workers):
        specs = [scheduler.next_hand() for _ in range(hands_per_worker)]
        worker_specs.append(specs)

    if gpu_server is not None:
        # Workers obtain queue handles via the pool's initializer (see
        # init_gpu_worker_queues); only pass worker_id + specs + cfg here.
        args = [
            (wid, specs, cfg)
            for wid, specs in enumerate(worker_specs)
        ]
        bufs = pool.map(_rollout_worker_gpu, args, chunksize=1)
    else:
        # Sync weights to disk for workers to load
        torch.save(
            {k: v.detach().cpu() for k, v in net.state_dict().items()},
            state_path,
        )
        args = [(state_path, specs, cfg, hidden) for specs in worker_specs]
        bufs = pool.map(_rollout_worker, args)

    return RolloutBuffer.from_buffers(bufs)
