"""Persistent actor subprocess for distributed actor-learner DMC.

``actor_loop`` is the target function for spawned actor processes.
``maybe_sync_weights_base`` is the non-blocking weight-sync helper.
"""

from __future__ import annotations

import dataclasses
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from .actor import play_episode
from .encoding.base_encoder import ENCODE_CHANNEL_KEYS, StateActionEncoder
from .encoding.role_encoder import ROLE_ENCODE_CHANNEL_KEYS, RoleAwareStateActionEncoder
from .utils.profiler import PhaseProfiler
from .sample_tags import (
    EPISODE_MODE_SELF_PLAY,
    EPISODE_MODE_VS_CHECKPOINT,
    EPISODE_MODE_VS_HARD_BOT,
    OPPONENT_BY_NAME,
    OPPONENT_CHECKPOINT_BASE,
    OPPONENT_NONE,
)


def maybe_sync_weights_base(
    q_nets: dict,
    weight_dir: Path,
    local_version: int,
    local_updates: int = 0,
    sync_interval_updates: int = 0,
) -> tuple[int, int, int]:
    """Conditionally load newer weights; return (version, local_updates, global_updates).

    Cheap path: reads only ``latest.txt`` to learn the latest published metadata.
    Expensive ``torch.load`` happens only when the actor is at least
    ``sync_interval_updates`` learner updates behind the latest publish.
    First sync (``local_version=-1``) always loads regardless of the threshold.

    ``local_updates`` is the update count at which the *currently loaded* policy
    was published; ``global_updates`` is the latest published count (used by the
    actor for ε-schedule). They diverge whenever the gate defers a load.
    """
    from .learner import read_latest_metadata, load_latest_weights

    meta = read_latest_metadata(weight_dir)
    if meta is None:
        return local_version, local_updates, 0
    latest_version, latest_updates = meta

    # Already at this version → no work, but report fresh global_updates.
    if latest_version <= local_version:
        return local_version, local_updates, latest_updates

    # Update-lag gate: defer load until far enough behind (always sync on first call).
    if local_version >= 0 and sync_interval_updates > 0 \
            and (latest_updates - local_updates) < sync_interval_updates:
        return local_version, local_updates, latest_updates

    snapshot = load_latest_weights(weight_dir)
    if snapshot is None or snapshot.version <= local_version:
        return local_version, local_updates, latest_updates
    for p in range(4):
        q_nets[p].load_state_dict(snapshot.state_dicts[p])
        q_nets[p].eval()
    return snapshot.version, snapshot.updates, snapshot.updates


maybe_sync_weights = maybe_sync_weights_base


def maybe_sync_weights_shared(
    q_net,
    weight_dir: Path,
    local_version: int,
    local_updates: int = 0,
    sync_interval_updates: int = 0,
) -> tuple[int, int, int]:
    """Conditionally load newer shared-head weights; return (version, local_updates, global_updates)."""
    from .learner import read_latest_metadata, load_latest_weights

    meta = read_latest_metadata(weight_dir)
    if meta is None:
        return local_version, local_updates, 0
    latest_version, latest_updates = meta

    if latest_version <= local_version:
        return local_version, local_updates, latest_updates

    if local_version >= 0 and sync_interval_updates > 0 \
            and (latest_updates - local_updates) < sync_interval_updates:
        return local_version, local_updates, latest_updates

    snapshot = load_latest_weights(weight_dir)
    if snapshot is None or snapshot.version <= local_version:
        return local_version, local_updates, latest_updates
    q_net.load_state_dict(snapshot.state_dicts["shared"])
    q_net.eval()
    return snapshot.version, snapshot.updates, snapshot.updates


def actor_loop(
    actor_id:     int,
    cfg_dict:     dict,
    sample_queue,           # multiprocessing.Queue
    stop_event,             # multiprocessing.Event
    weight_dir:   Path,
    inference_args=None,    # Optional[dict] — shared-memory inference handles
    run_dir:      Path | None = None,
) -> None:
    """Persistent actor process for distributed actor-learner DMC.

    Runs self-play continuously, serializes samples to the shared queue, and
    periodically syncs local Q-net copies from the learner's published weights.
    Top-level module function — must be picklable for the 'spawn' start method.

    If ``inference_args`` is provided, action selection routes through a shared
    GPU inference server; the actor skips local q-net initialization entirely.
    """
    # Lazy imports — spawned child re-imports the full package from scratch.
    from .config import TrainConfig
    from .utils.schedules import epsilon_linear
    from .q_network import SharedHeadQNet
    from .q_network import init_seat_nets
    from .config import shared_head_qnet_config

    torch.set_num_threads(1)

    cfg = TrainConfig.from_flat_dict(cfg_dict)
    if cfg.model_type in ("shared_heads", "shared_trick_heads") and cfg.inference.enabled:
        raise ValueError(
            f"Inference server not supported for model_type={cfg.model_type!r}."
        )

    shared_path = cfg.model_type in ("shared_heads", "shared_trick_heads")
    trick_path = cfg.model_type == "shared_trick_heads"
    if shared_path:
        encoder = RoleAwareStateActionEncoder(
            is_partner_visible=cfg.qnet.is_partner_visible,
            head_scheme="trick_relative" if trick_path else "absolute_seat",
        )
    else:
        encoder = StateActionEncoder(is_partner_visible=cfg.qnet.is_partner_visible)

    inference_client = _build_inference_client(actor_id, inference_args, cfg) if inference_args else None
    if shared_path:
        if trick_path:
            from .q_network import SharedTrickHeadQNet
            from .config import shared_trick_head_qnet_config
            q_nets = SharedTrickHeadQNet(shared_trick_head_qnet_config(cfg))
        else:
            q_nets = SharedHeadQNet(shared_head_qnet_config(cfg))
        q_nets.eval()
    elif inference_client is None:
        q_nets = init_seat_nets(cfg.qnet)
        for net in q_nets.values():
            net.eval()
    else:
        q_nets = None

    # int8 dynamic quantization of actor Q-nets (CPU only). The fp32 q_nets are
    # kept as the load_state_dict target; q_nets_inference holds the quantized
    # copy that play_episode actually calls forward on. Re-quantized after each
    # weight sync.
    use_int8 = bool(getattr(cfg, "use_int8_actor", False)) and q_nets is not None
    if use_int8:
        import torch.ao.quantization as _qao

        def _quantize_for_inference(nets):
            if shared_path:
                return _qao.quantize_dynamic(nets, {torch.nn.Linear}, dtype=torch.qint8)
            return {p: _qao.quantize_dynamic(nets[p], {torch.nn.Linear}, dtype=torch.qint8) for p in range(4)}

        q_nets_inference = _quantize_for_inference(q_nets)
    else:
        q_nets_inference = q_nets

    # ── Checkpoint-population pool (shared-head + non-empty pool only) ──
    # Each actor preloads every frozen checkpoint once. Per vs-frozen episode
    # we sample one uniformly and use it for the opponent seats.
    frozen_nets: list = []
    population_active = (
        shared_path
        and bool(cfg.population_pool)
        and cfg.latest_vs_latest_frac < 1.0
    )
    if population_active:
        if trick_path:
            from .checkpoint import load_frozen_trick_qnet
            qnet_cfg_for_pool = shared_trick_head_qnet_config(cfg)
            for ckpt_path in cfg.population_pool:
                frozen_nets.append(load_frozen_trick_qnet(ckpt_path, qnet_cfg_for_pool, device="cpu"))
        else:
            from .checkpoint import load_frozen_shared_qnet
            qnet_cfg_for_pool = shared_head_qnet_config(cfg)
            for ckpt_path in cfg.population_pool:
                frozen_nets.append(load_frozen_shared_qnet(ckpt_path, qnet_cfg_for_pool, device="cpu"))
        print(
            f"[actor-{actor_id}] preloaded {len(frozen_nets)} frozen opponents",
            flush=True,
        )

    # ── Hard-bot opponent pool (shared-head + non-empty pool only) ──
    # Each actor instantiates one copy of every hard bot at startup and
    # samples one per vs-hard-bot episode. Bots are stateless across episodes
    # (level_rank fixed) so a single instance per actor is enough.
    hard_bots_pool: list = []   # list of (name, Agent) pairs
    hard_bot_active = (
        shared_path
        and bool(cfg.hard_bot_pool)
        and cfg.latest_vs_hard_bot_frac > 0.0
    )
    hard_bot_weights: list[float] | None = None
    hard_bots_by_name: dict = {}
    pair_names: list[str] = []
    pair_weights: list[float] = []
    if hard_bot_active:
        from ..agents import make_agent
        for bot_name in cfg.hard_bot_pool:
            hard_bots_pool.append((bot_name, make_agent(bot_name)))
        hard_bots_by_name = {n: agent for n, agent in hard_bots_pool}
        if cfg.hard_bot_pair_sampling:
            # Pair sampling: keys are "<a>_<b>" (alphabetized, validated in
            # TrainConfig). Two opponent seats get distinct bot instances
            # (or the same instance for homogeneous pairs).
            pair_names = list(cfg.hard_bot_pair_sampling.keys())
            pair_weights = list(cfg.hard_bot_pair_sampling.values())
        elif cfg.hard_bot_sampling:
            # Validation in TrainConfig already normalized weights to sum to 1
            # and verified all keys are in hard_bot_pool. Bots not listed get 0.
            hard_bot_weights = [
                cfg.hard_bot_sampling.get(n, 0.0) for n, _ in hard_bots_pool
            ]
        if cfg.hard_bot_pair_sampling:
            sampling_str = f"pair_sampling={cfg.hard_bot_pair_sampling}"
        elif cfg.hard_bot_sampling:
            sampling_str = f"weighted={cfg.hard_bot_sampling}"
        else:
            sampling_str = "uniform"
        print(
            f"[actor-{actor_id}] loaded {len(hard_bots_pool)} hard-bot opponents: "
            f"{[n for n, _ in hard_bots_pool]} ({sampling_str})",
            flush=True,
        )

    # Per-actor counters for end-of-run summary (printed by every actor).
    mode_counts = {"self_play": 0, "vs_frozen": 0, "vs_hard_bot": 0}
    frozen_pick_counts = [0] * len(frozen_nets)
    hard_bot_pick_counts = [0] * len(hard_bots_pool)
    pair_pick_counts: dict[str, int] = {k: 0 for k in pair_names}
    team_counts = {"latest_even": 0, "latest_odd": 0}

    weight_dir    = Path(weight_dir)
    run_dir       = Path(run_dir) if run_dir is not None else None
    local_version  = -1
    local_updates  = 0   # update count of the policy actor currently holds
    global_updates = 0   # latest published update count (for ε-schedule)
    episode_count  = 0
    # Pre-stacked accumulator: keep raw encoded dicts and stack at push-time.
    # One pickle of 9 contiguous arrays is ~6× faster to unpickle than 512 dicts
    # of 9 small arrays each (measured: 3.81 ms → 0.63 ms per push).
    buf_dicts:   list[dict]  = []
    buf_players: list[int]   = []
    buf_returns: list[float] = []
    buf_buckets: list[int]   = []
    buf_phase_self:        list[int]   = []
    buf_trick_role:        list[int]   = []
    buf_phase_partner:     list[int]   = []
    buf_action_type:       list[int]   = []
    buf_is_pass:           list[int]   = []
    buf_is_bomb:           list[int]   = []
    buf_bomb_available:    list[int]   = []
    buf_num_legal:         list[int]   = []
    buf_q_gap:             list[float] = []
    buf_chosen_by_epsilon: list[int]   = []
    buf_episode_mode:      list[int]   = []
    buf_opponent_id:       list[int]   = []
    buf_latest_team:       list[int]   = []
    buf_terminal_reward:   list[float] = []
    rng = random.Random(cfg.seed + actor_id * 10_000)

    def _sample_sync_threshold() -> int:
        """Per-actor sync threshold = sync_interval_updates ± sync_jitter_updates (uniform)."""
        if cfg.sync_jitter_updates <= 0:
            return cfg.sync_interval_updates
        return cfg.sync_interval_updates + rng.randint(
            -cfg.sync_jitter_updates, cfg.sync_jitter_updates,
        )

    sync_threshold = _sample_sync_threshold()

    profile_enabled = os.environ.get("GUANZERO_ACTOR_PROFILE") == "1"
    prof = PhaseProfiler(enabled=profile_enabled)
    prof_t_start = time.perf_counter()
    snapshot_every_episodes = max(1, cfg.log_every_updates * 10)

    def _push_buffered() -> None:
        if len(buf_dicts) < cfg.actor_push_batch_size:
            return
        with prof.time("buffer_stack"):
            if shared_path:
                keys = encoder.channel_keys
            else:
                keys = ENCODE_CHANNEL_KEYS
            stacked = {k: np.stack([d[k] for d in buf_dicts], axis=0) for k in keys}
            msg = {
                "actor_id":          actor_id,
                "version":           local_version,
                "stacked":           stacked,
                "returns":           np.asarray(buf_returns, dtype=np.float32),
                "buckets":           np.asarray(buf_buckets, dtype=np.int8),
                "phase_self":        np.asarray(buf_phase_self,        dtype=np.int8),
                "trick_role":        np.asarray(buf_trick_role,        dtype=np.int8),
                "phase_partner":     np.asarray(buf_phase_partner,     dtype=np.int8),
                "action_type":       np.asarray(buf_action_type,       dtype=np.int8),
                "is_pass":           np.asarray(buf_is_pass,           dtype=np.int8),
                "is_bomb":           np.asarray(buf_is_bomb,           dtype=np.int8),
                "bomb_available":    np.asarray(buf_bomb_available,    dtype=np.int8),
                "num_legal_actions": np.asarray(buf_num_legal,         dtype=np.int16),
                "q_gap":             np.asarray(buf_q_gap,             dtype=np.float32),
                "chosen_by_epsilon": np.asarray(buf_chosen_by_epsilon, dtype=np.int8),
                "episode_mode":      np.asarray(buf_episode_mode,      dtype=np.int8),
                "opponent_id":       np.asarray(buf_opponent_id,       dtype=np.int8),
                "latest_team":       np.asarray(buf_latest_team,       dtype=np.int8),
                "terminal_reward":   np.asarray(buf_terminal_reward,   dtype=np.float32),
            }
            if not shared_path:
                msg["players"] = np.asarray(buf_players, dtype=np.int8)
        with prof.time("queue_put"):
            try:
                sample_queue.put(msg, timeout=5)
            except Exception:
                pass   # queue full or closed — drop and continue
        buf_dicts.clear()
        buf_players.clear()
        buf_returns.clear()
        buf_buckets.clear()
        buf_phase_self.clear()
        buf_trick_role.clear()
        buf_phase_partner.clear()
        buf_action_type.clear()
        buf_is_pass.clear()
        buf_is_bomb.clear()
        buf_bomb_available.clear()
        buf_num_legal.clear()
        buf_q_gap.clear()
        buf_chosen_by_epsilon.clear()
        buf_episode_mode.clear()
        buf_opponent_id.clear()
        buf_latest_team.clear()
        buf_terminal_reward.clear()

    n_inference_timeouts = 0
    while not stop_event.is_set():
        # Weight sync — cheap metadata read every iteration (~10 bytes from latest.txt);
        # torch.load only fires when actor is sync_threshold+ updates behind the learner.
        # Each actor's threshold is jittered ± sync_jitter_updates so the 32 actors
        # don't all reload on the same publish cycle (smoother self-play diversity).
        if q_nets is not None:
            prev_version = local_version
            with prof.time("weight_sync"):
                if shared_path:
                    local_version, local_updates, global_updates = maybe_sync_weights_shared(
                        q_nets, weight_dir, local_version, local_updates, sync_threshold,
                    )
                else:
                    local_version, local_updates, global_updates = maybe_sync_weights_base(
                        q_nets, weight_dir, local_version, local_updates, sync_threshold,
                    )
            if local_version > prev_version:
                # Just synced — draw a fresh threshold so the next reload is independently jittered.
                sync_threshold = _sample_sync_threshold()
                if use_int8:
                    q_nets_inference = _quantize_for_inference(q_nets)

        eps = epsilon_linear(global_updates, cfg.epsilon)

        # ── Per-episode dispatch ──
        # population_pool and hard_bot_pool are mutually exclusive (validated
        # in TrainConfig.__post_init__); each branch handles its own RNG draw.
        # In every case `*_seats` are the OPPONENT seats — latest controls the
        # complement and is the only contributor of training samples.
        frozen_seats: frozenset[int] = frozenset()
        hard_bot_seats: frozenset[int] = frozenset()
        q_net_frozen_this_ep = None
        active_hard_bot = None  # Agent | dict[int, Agent] | None
        active_bot_name: str | None = None
        episode_mode = EPISODE_MODE_SELF_PLAY
        opponent_id = OPPONENT_NONE
        latest_team = 0

        if hard_bots_pool:
            u = rng.random()
            if u < cfg.latest_vs_latest_frac:
                mode_counts["self_play"] += 1
            elif u < cfg.latest_vs_latest_frac + cfg.latest_vs_hard_bot_frac:
                # Choose opponent-team seats (latest controls the complement).
                if rng.random() < cfg.latest_odd_probability:
                    seat_a, seat_b = 0, 2  # opp on even → latest on odd
                    team_counts["latest_odd"] += 1
                    latest_team = 1
                else:
                    seat_a, seat_b = 1, 3  # opp on odd → latest on even
                    team_counts["latest_even"] += 1
                    latest_team = 0
                hard_bot_seats = frozenset({seat_a, seat_b})

                if pair_names:
                    pair_key = rng.choices(pair_names, weights=pair_weights, k=1)[0]
                    pair_pick_counts[pair_key] += 1
                    name_a, name_b = pair_key.split("_")
                    bot_a = hard_bots_by_name[name_a]
                    bot_b = hard_bots_by_name[name_b]
                    if rng.random() < 0.5:
                        active_hard_bot = {seat_a: bot_a, seat_b: bot_b}
                    else:
                        active_hard_bot = {seat_a: bot_b, seat_b: bot_a}
                    active_bot_name = "yaoji" if "yaoji" in pair_key else name_a
                    opponent_id = OPPONENT_BY_NAME.get(active_bot_name, OPPONENT_NONE)
                else:
                    if hard_bot_weights is not None:
                        pick = rng.choices(
                            range(len(hard_bots_pool)), weights=hard_bot_weights, k=1
                        )[0]
                    else:
                        pick = rng.randrange(len(hard_bots_pool))
                    active_bot_name, active_hard_bot = hard_bots_pool[pick]
                    hard_bot_pick_counts[pick] += 1
                    opponent_id = OPPONENT_BY_NAME.get(active_bot_name, OPPONENT_NONE)
                episode_mode = EPISODE_MODE_VS_HARD_BOT
                mode_counts["vs_hard_bot"] += 1
            else:
                mode_counts["self_play"] += 1
        elif frozen_nets:
            use_frozen = rng.random() >= cfg.latest_vs_latest_frac
            if use_frozen:
                pick = rng.randrange(len(frozen_nets))
                q_net_frozen_this_ep = frozen_nets[pick]
                frozen_pick_counts[pick] += 1
                if rng.random() < cfg.latest_odd_probability:
                    frozen_seats = frozenset({0, 2})   # opp on even → latest on odd
                    team_counts["latest_odd"] += 1
                    latest_team = 1
                else:
                    frozen_seats = frozenset({1, 3})   # opp on odd → latest on even
                    team_counts["latest_even"] += 1
                    latest_team = 0
                episode_mode = EPISODE_MODE_VS_CHECKPOINT
                opponent_id = OPPONENT_CHECKPOINT_BASE + pick
                mode_counts["vs_frozen"] += 1
            else:
                mode_counts["self_play"] += 1
        else:
            mode_counts["self_play"] += 1

        seed = rng.randint(0, 10_000_000)
        try:
            samples = play_episode(
                q_nets=q_nets_inference,
                encoder=encoder,
                epsilon=eps,
                seed=seed,
                device="cpu",
                gamma=cfg.gamma,
                inference_client=inference_client,
                profiler=prof,
                q_nets_frozen=q_net_frozen_this_ep,
                frozen_seats=frozen_seats,
                epsilon_frozen=cfg.epsilon.frozen,
                hard_bots=active_hard_bot,
                hard_bot_seats=hard_bot_seats,
                episode_mode=episode_mode,
                opponent_id=opponent_id,
                latest_team=latest_team,
            )
        except Exception as e:
            from .inference_server import InferenceTimeoutError
            if isinstance(e, InferenceTimeoutError):
                n_inference_timeouts += 1
                if n_inference_timeouts <= 3 or n_inference_timeouts % 10 == 0:
                    print(f"[actor-{actor_id}] inference timeout #{n_inference_timeouts}: {e}",
                          flush=True)
                # Brief sleep before retrying so we don't busy-loop if the server
                # is wedged. stop_event check below caps it.
                if stop_event.wait(timeout=0.5):
                    break
                continue
            raise
        episode_count += 1

        if frozen_seats:
            # Drop frozen-team samples — they're opponents, not teachers.
            samples = [s for s in samples if s.player not in frozen_seats]
        # Hard-bot seats never enter the trajectory (skipped in play_episode),
        # so no filter is needed here for the hard-bot branch.

        # Episode-level info for bucket computation
        is_hard_bot_ep = active_hard_bot is not None and len(samples) > 0
        if is_hard_bot_ep:
            latest_won = samples[0].mc_return > 0
            target_bot = active_bot_name in ("yaoji", "jidan")
            is_coord_episode = (not latest_won) and target_bot
        else:
            is_coord_episode = False

        total = len(samples)
        for i, s in enumerate(samples):
            buf_dicts.append(s.encoded)
            buf_players.append(s.player)
            buf_returns.append(s.mc_return)
            buf_phase_self.append(s.phase_self)
            buf_trick_role.append(s.trick_role)
            buf_phase_partner.append(s.phase_partner)
            buf_action_type.append(s.action_type)
            buf_is_pass.append(s.is_pass)
            buf_is_bomb.append(s.is_bomb)
            buf_bomb_available.append(s.bomb_available)
            buf_num_legal.append(s.num_legal_actions)
            buf_q_gap.append(s.q_gap)
            buf_chosen_by_epsilon.append(s.chosen_by_epsilon)
            buf_episode_mode.append(s.episode_mode)
            buf_opponent_id.append(s.opponent_id)
            buf_latest_team.append(s.latest_team)
            buf_terminal_reward.append(s.terminal_reward)

            if not is_hard_bot_ep:
                sample_bucket = 0  # general_self_play
            elif not is_coord_episode:
                sample_bucket = 1  # hard_bot_general
            else:
                # yaoji/jidan loss episode — check per-sample coord state.
                # player_blocks layout (243 dims per role):
                #   [0:108] played, [108:216] last_action, [216:243] count one-hot.
                # role 0 = self, role 1 = next_opp, role 2 = partner, role 3 = prev_opp.
                blocks = s.encoded["player_blocks"]
                self_cards = int(s.encoded["own_hand"].sum())
                partner_cards = int(np.argmax(blocks[2, 216:243]))
                next_opp_cards = int(np.argmax(blocks[1, 216:243]))
                prev_opp_cards = int(np.argmax(blocks[3, 216:243]))
                min_opp_cards = min(next_opp_cards, prev_opp_cards)
                is_final_third = i >= total * 2 // 3
                is_coord = (
                    self_cards <= 5
                    or partner_cards <= 5
                    or min_opp_cards <= 5
                    or is_final_third
                )
                sample_bucket = 2 if is_coord else 1

            buf_buckets.append(sample_bucket)

        _push_buffered()

        # Periodic snapshot to stdout — first actor only, to avoid N-actor spam.
        if (profile_enabled and actor_id == 0
                and episode_count > 0 and episode_count % snapshot_every_episodes == 0):
            wall_s = time.perf_counter() - prof_t_start
            n_dec = prof._counts.get("num_decisions", 0)
            snap = prof.report(wall_s=wall_s, n_events=episode_count, event_label="episodes")
            snap += prof.report_k_buckets(n_decisions=n_dec)
            print(f"[actor-{actor_id}] profile snapshot @ episode {episode_count}:{snap}",
                  flush=True)

    if profile_enabled and run_dir is not None:
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
            wall_s = time.perf_counter() - prof_t_start
            n_dec = prof._counts.get("num_decisions", 0)
            report = prof.report(wall_s=wall_s, n_events=episode_count, event_label="episodes")
            report += prof.report_k_buckets(n_decisions=n_dec)
            out = run_dir / f"actor_{actor_id}_profile.txt"
            out.write_text(report + "\n")
        except Exception as e:
            print(f"[actor-{actor_id}] failed to write profile: {e}", flush=True)

    # Per-actor opponent-pool counters — printed by every actor so off-balance
    # pool sampling or skewed team assignment shows up immediately in logs.
    print(f"[actor-{actor_id}] episode modes: {mode_counts}", flush=True)
    if frozen_nets:
        pool_str = ", ".join(
            f"{Path(p).stem}={c}"
            for p, c in zip(cfg.population_pool, frozen_pick_counts)
        )
        print(f"[actor-{actor_id}] frozen picks: {pool_str}", flush=True)
        print(f"[actor-{actor_id}] team assignment: {team_counts}", flush=True)
    if hard_bots_pool:
        if pair_names:
            pool_str = ", ".join(f"{k}={c}" for k, c in pair_pick_counts.items())
            print(f"[actor-{actor_id}] hard-bot pair picks: {pool_str}", flush=True)
        else:
            pool_str = ", ".join(
                f"{n}={c}" for (n, _), c in zip(hard_bots_pool, hard_bot_pick_counts)
            )
            print(f"[actor-{actor_id}] hard-bot picks: {pool_str}", flush=True)
        print(f"[actor-{actor_id}] team assignment: {team_counts}", flush=True)


def _build_inference_client(actor_id: int, inference_args: dict, cfg) -> "object":
    from .inference_server import build_client
    return build_client(
        actor_id       = actor_id,
        inference_args = inference_args,
        timeout_s      = cfg.inference.timeout_s,
        max_actions    = cfg.inference.max_actions,
    )


__all__ = [
    "actor_loop",
    "maybe_sync_weights_base",
    "maybe_sync_weights",
    "maybe_sync_weights_shared",
]
