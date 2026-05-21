"""Shared game-playing utilities for eval scripts.

Provides ``play_n_games`` — a single function that runs N games with agent A
on seats {0, 2} and agent B on seats {1, 3} and returns win/loss counts. Also
provides a Dart-specific lane runner used by ``eval_dart.py`` to batch Q-net
forwards across multiple active environments.
"""

from __future__ import annotations

from typing import Any


def play_n_games(
    agent_a: Any,
    agent_b: Any,
    n_games: int,
    seeds: list[int] | None = None,
    label: str = "",
    progress: int = 0,
) -> dict[str, Any]:
    """Play ``n_games`` with agent_a on seats {0, 2} and agent_b on {1, 3}.

    If ``seeds`` is provided it must have ``n_games`` elements; each game is
    reset with the corresponding seed so decks are reproducible.

    Returns a dict with keys: ``wins``, ``losses``, ``n_games``, ``winrate``,
    ``avg_reward``.
    """
    from guandan.game import GuanDanEnv

    env = GuanDanEnv()
    wins_a = 0
    total_r = 0.0

    for i in range(n_games):
        kw = {"seed": seeds[i]} if seeds is not None else {}
        env.reset(**kw)
        while not env.done:
            p = env.current_player
            move = agent_a.act(env, p) if p in (0, 2) else agent_b.act(env, p)
            env.step(move)
        rewards = env.get_rewards()
        team_r = rewards[0] + rewards[2]
        if team_r > 0:
            wins_a += 1
        total_r += team_r
        if progress and label and (i + 1) % progress == 0:
            print(f"  {label}: {i+1}/{n_games}  WR={wins_a/(i+1):.1%}", flush=True)

    return {
        "wins": wins_a,
        "losses": n_games - wins_a,
        "n_games": n_games,
        "winrate": wins_a / n_games,
        "avg_reward": total_r / n_games,
    }


def play_n_games_dart_lanes(
    dart_bot: Any,
    opponent_name: str,
    n_games: int,
    *,
    seeds: list[int] | None = None,
    dart_seats: tuple[int, int] = (0, 2),
    lanes: int = 1,
) -> dict[str, Any]:
    """Play ``n_games`` with Dart on ``dart_seats`` using batched env lanes.

    This is the eval analogue of the actor rollout lane loop: hard-bot actions
    step immediately, while Dart decisions from currently-active lanes are
    grouped by acting Q-net and evaluated in one batched forward.
    """
    if lanes < 1:
        raise ValueError(f"lanes must be >= 1, got {lanes}")
    if seeds is not None and len(seeds) != n_games:
        raise ValueError(f"expected {n_games} seeds, got {len(seeds)}")
    if tuple(sorted(dart_seats)) not in ((0, 2), (1, 3)):
        raise ValueError(f"dart_seats must be (0, 2) or (1, 3), got {dart_seats}")

    from guandan.agents import make_agent
    from guandan.dart.model.q_network import DartQNet
    from guandan.dart.runtime.actor import argmax_q_batched, select_legal
    from guandan.game import GuanDanEnv

    q_nets = dart_bot.q_nets
    encoder = dart_bot.encoder
    device = dart_bot.device
    dart_seat_set = set(dart_seats)
    seed_values: list[int | None] = list(seeds) if seeds is not None else [None] * n_games

    wins = 0
    total_r = 0.0
    completed = 0

    while completed < n_games:
        batch_seeds = seed_values[completed : completed + min(lanes, n_games - completed)]
        envs = [GuanDanEnv(seed=seed) for seed in batch_seeds]
        # One opponent instance per lane avoids state bleed between interleaved
        # games for vendor bots that cache per-game history.
        opponents = [make_agent(opponent_name) for _ in batch_seeds]
        active = [True] * len(envs)

        while any(active):
            pending: list[dict[str, Any]] = []
            for lane_idx, env in enumerate(envs):
                if not active[lane_idx]:
                    continue
                if env.done:
                    active[lane_idx] = False
                    continue

                player = env.current_player
                if player not in dart_seat_set:
                    env.step(opponents[lane_idx].act(env, player))
                    if env.done:
                        active[lane_idx] = False
                    continue

                legal = select_legal(env, player)
                if len(legal) == 1:
                    env.step(legal[0])
                    if env.done:
                        active[lane_idx] = False
                    continue

                pending.append({
                    "lane": lane_idx,
                    "player": player,
                    "legal": legal,
                    "encoded_list": encoder.encode_all(env, player, legal),
                })

            if not pending:
                continue

            groups_by_net: dict[int, dict[str, Any]] = {}
            for pending_idx, item in enumerate(pending):
                if isinstance(q_nets, DartQNet):
                    net = q_nets
                else:
                    net = q_nets[item["player"]]
                group = groups_by_net.setdefault(id(net), {"net": net, "indices": []})
                group["indices"].append(pending_idx)

            choices_by_idx: dict[int, int] = {}
            for group in groups_by_net.values():
                indices = group["indices"]
                net = group["net"]
                encoded_groups = [pending[i]["encoded_list"] for i in indices]
                choices = argmax_q_batched(
                    net,
                    encoded_groups,
                    device,
                    role_encoded=isinstance(net, DartQNet),
                )
                for pending_idx, (choice_idx, _q_gap) in zip(indices, choices, strict=False):
                    choices_by_idx[pending_idx] = choice_idx

            for pending_idx, item in enumerate(pending):
                lane_idx = item["lane"]
                env = envs[lane_idx]
                choice_idx = choices_by_idx[pending_idx]
                env.step(item["legal"][choice_idx])
                if env.done:
                    active[lane_idx] = False

        for env in envs:
            rewards = env.get_rewards()
            team_r = sum(rewards[p] for p in dart_seat_set)
            if team_r > 0:
                wins += 1
            total_r += team_r

        completed += len(batch_seeds)

    return {
        "wins": wins,
        "losses": n_games - wins,
        "n_games": n_games,
        "winrate": wins / n_games if n_games else 0.0,
        "avg_reward": total_r / n_games if n_games else 0.0,
    }
