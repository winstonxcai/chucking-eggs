"""
Concurrent games load test for the Guan Dan backend.

Spins up N concurrent games (solo/duo/quad), plays each to completion
using a simple pass-first strategy, and reports latency/throughput metrics.

Usage:
    pip install aiohttp
    python web/backend/tests/load_test.py --url http://localhost:8000 --mode solo -c 10 -n 20
    python web/backend/tests/load_test.py --url https://chucking-eggs.fly.dev --mode duo -c 5 -n 10
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import aiohttp

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_URL = "http://localhost:8000"
DEFAULT_CONCURRENT = 10
DEFAULT_TOTAL = 20
DEFAULT_DIFFICULTY = "easy"
DEFAULT_MODE = "solo"

MSG_TIMEOUT = 60  # seconds to wait for any single WS message
GAME_TIMEOUT = 180  # seconds max for an entire game


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class SeatMetrics:
    """Metrics collected by a single WS seat client."""
    move_rtts_ms: list[float] = field(default_factory=list)
    ws_connect_ms: float = 0.0
    moves_played: int = 0
    messages_received: int = 0


@dataclass
class GameMetrics:
    game_id: str = ""
    mode: str = ""
    room_create_ms: float = 0.0
    join_ms: list[float] = field(default_factory=list)  # for duo/quad joins
    seat_metrics: list[SeatMetrics] = field(default_factory=list)
    total_duration_s: float = 0.0
    completed: bool = False
    error: str | None = None

    @property
    def ws_connect_ms(self) -> float:
        vals = [s.ws_connect_ms for s in self.seat_metrics if s.ws_connect_ms > 0]
        return statistics.mean(vals) if vals else 0.0

    @property
    def move_rtts_ms(self) -> list[float]:
        return [rtt for s in self.seat_metrics for rtt in s.move_rtts_ms]

    @property
    def total_moves(self) -> int:
        return sum(s.moves_played for s in self.seat_metrics)

    @property
    def total_messages(self) -> int:
        return sum(s.messages_received for s in self.seat_metrics)


@dataclass
class LoadTestResults:
    game_metrics: list[GameMetrics] = field(default_factory=list)
    wall_time_s: float = 0.0
    total_games: int = 0
    concurrent: int = 0
    mode: str = ""
    url: str = ""
    difficulty: str = ""


# ---------------------------------------------------------------------------
# Play logic
# ---------------------------------------------------------------------------

def pick_move(game_state: dict) -> dict | None:
    """Decide what to send given a game_state message. Returns None if not our turn."""
    if game_state.get("done"):
        return None
    if not game_state.get("is_my_turn"):
        return None

    legal = game_state.get("legal_moves", [])
    if not legal:
        return None

    # PASS is appended last by the serializer when available
    if legal[-1].get("is_pass"):
        return {"type": "pass"}

    # Play the smallest legal combo
    cards = legal[0].get("cards", [])
    card_ids = [c["id"] for c in cards]
    return {"type": "play_cards", "card_ids": card_ids}


# ---------------------------------------------------------------------------
# Per-seat WS client
# ---------------------------------------------------------------------------

async def seat_client(
    session: aiohttp.ClientSession,
    ws_url: str,
    seat_metrics: SeatMetrics,
    done_event: asyncio.Event,
    label: str = "",
) -> None:
    """Connect one WS seat, play when it's our turn, exit on game_over."""
    t_ws = time.perf_counter()
    first_state = True
    pending_rtt_t: float | None = None

    async with session.ws_connect(ws_url) as ws:
        async for msg in ws:
            if done_event.is_set():
                break
            if msg.type == aiohttp.WSMsgType.TEXT:
                seat_metrics.messages_received += 1
                data = json.loads(msg.data)
                msg_type = data.get("type")

                if msg_type == "game_state":
                    if first_state:
                        seat_metrics.ws_connect_ms = (time.perf_counter() - t_ws) * 1000
                        first_state = False

                    # Record RTT from previous move
                    if pending_rtt_t is not None:
                        seat_metrics.move_rtts_ms.append(
                            (time.perf_counter() - pending_rtt_t) * 1000
                        )
                        pending_rtt_t = None

                    if data.get("done"):
                        done_event.set()
                        break

                    move = pick_move(data)
                    if move:
                        pending_rtt_t = time.perf_counter()
                        await ws.send_json(move)
                        seat_metrics.moves_played += 1

                elif msg_type == "game_over":
                    done_event.set()
                    break

            elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR):
                if not done_event.is_set():
                    raise ConnectionError(f"WS closed unexpectedly: {msg.data}")
                break


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

async def create_room(
    session: aiohttp.ClientSession, base_url: str, mode: str, difficulty: str,
) -> tuple[dict, float]:
    """POST /api/room/create, returns (response_json, latency_ms)."""
    t0 = time.perf_counter()
    async with session.post(
        f"{base_url}/api/room/create",
        json={"mode": mode, "difficulty": difficulty},
    ) as resp:
        resp.raise_for_status()
        data = await resp.json()
    return data, (time.perf_counter() - t0) * 1000


async def join_room(
    session: aiohttp.ClientSession, base_url: str, room_code: str,
) -> tuple[dict, float]:
    """POST /api/room/join/{code}, returns (response_json, latency_ms)."""
    t0 = time.perf_counter()
    async with session.post(f"{base_url}/api/room/join/{room_code}") as resp:
        resp.raise_for_status()
        data = await resp.json()
    return data, (time.perf_counter() - t0) * 1000


def ws_url(base_url: str, game_id: str, token: str, seat: int) -> str:
    scheme = base_url.replace("http://", "ws://").replace("https://", "wss://")
    return f"{scheme}/ws/game/{game_id}?token={token}&seat={seat}"


# ---------------------------------------------------------------------------
# Per-mode game runners
# ---------------------------------------------------------------------------

async def run_solo_game(
    session: aiohttp.ClientSession, base_url: str, difficulty: str, game_num: int,
) -> GameMetrics:
    metrics = GameMetrics(mode="solo")
    t_start = time.perf_counter()
    try:
        room, latency = await create_room(session, base_url, "solo", difficulty)
        metrics.game_id = room["game_id"]
        metrics.room_create_ms = latency

        sm = SeatMetrics()
        metrics.seat_metrics.append(sm)
        done = asyncio.Event()

        url = ws_url(base_url, room["game_id"], room["reconnect_token"], 0)
        await asyncio.wait_for(seat_client(session, url, sm, done), timeout=GAME_TIMEOUT)
        metrics.completed = True
    except Exception as e:
        metrics.error = str(e)
    metrics.total_duration_s = time.perf_counter() - t_start
    return metrics


async def run_duo_game(
    session: aiohttp.ClientSession, base_url: str, difficulty: str, game_num: int,
) -> GameMetrics:
    metrics = GameMetrics(mode="duo")
    t_start = time.perf_counter()
    try:
        room, latency = await create_room(session, base_url, "duo", difficulty)
        metrics.game_id = room["game_id"]
        metrics.room_create_ms = latency

        join_data, join_lat = await join_room(session, base_url, room["room_code"])
        metrics.join_ms.append(join_lat)

        done = asyncio.Event()
        seats = [
            (0, room["reconnect_token"]),
            (join_data["seat"], join_data["reconnect_token"]),
        ]
        seat_metrics_list = []
        tasks = []
        for seat, token in seats:
            sm = SeatMetrics()
            seat_metrics_list.append(sm)
            url = ws_url(base_url, room["game_id"], token, seat)
            tasks.append(asyncio.create_task(seat_client(session, url, sm, done, f"seat{seat}")))

        metrics.seat_metrics = seat_metrics_list
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=GAME_TIMEOUT)
        metrics.completed = True
    except Exception as e:
        metrics.error = str(e)
    metrics.total_duration_s = time.perf_counter() - t_start
    return metrics


async def run_quad_game(
    session: aiohttp.ClientSession, base_url: str, difficulty: str, game_num: int,
) -> GameMetrics:
    metrics = GameMetrics(mode="quad")
    t_start = time.perf_counter()
    try:
        room, latency = await create_room(session, base_url, "quad", difficulty)
        metrics.game_id = room["game_id"]
        metrics.room_create_ms = latency

        seats = [(0, room["reconnect_token"])]
        for _ in range(3):
            join_data, join_lat = await join_room(session, base_url, room["room_code"])
            metrics.join_ms.append(join_lat)
            seats.append((join_data["seat"], join_data["reconnect_token"]))

        done = asyncio.Event()
        seat_metrics_list = []
        tasks = []
        for seat, token in seats:
            sm = SeatMetrics()
            seat_metrics_list.append(sm)
            url = ws_url(base_url, room["game_id"], token, seat)
            tasks.append(asyncio.create_task(seat_client(session, url, sm, done, f"seat{seat}")))

        metrics.seat_metrics = seat_metrics_list
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=GAME_TIMEOUT)
        metrics.completed = True
    except Exception as e:
        metrics.error = str(e)
    metrics.total_duration_s = time.perf_counter() - t_start
    return metrics


RUNNERS = {
    "solo": run_solo_game,
    "duo": run_duo_game,
    "quad": run_quad_game,
}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

async def run_load_test(
    url: str, concurrent: int, total: int, difficulty: str, mode: str,
) -> LoadTestResults:
    results = LoadTestResults(
        total_games=total, concurrent=concurrent, mode=mode, url=url, difficulty=difficulty,
    )
    sem = asyncio.Semaphore(concurrent)
    completed_count = 0
    runner = RUNNERS[mode]

    async def bounded(i: int) -> GameMetrics:
        nonlocal completed_count
        async with sem:
            m = await runner(session, url, difficulty, i)
        completed_count += 1
        status = "OK" if m.completed else f"FAIL: {m.error}"
        print(f"  [{completed_count}/{total}] {m.game_id[:8]}... {status} ({m.total_duration_s:.1f}s, {m.total_moves} moves)")
        return m

    print(f"\n=== Starting load test: {mode} x{total} (concurrency={concurrent}) ===")
    print(f"    Target: {url}")
    print(f"    Difficulty: {difficulty}\n")

    t_wall = time.perf_counter()
    async with aiohttp.ClientSession() as session:
        tasks = [asyncio.create_task(bounded(i)) for i in range(total)]
        game_metrics = await asyncio.gather(*tasks)

    results.wall_time_s = time.perf_counter() - t_wall
    results.game_metrics = list(game_metrics)
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _percentiles(values: list[float], qs: list[float] = [0.5, 0.95]) -> dict[str, float]:
    if not values:
        return {}
    result = {}
    sorted_v = sorted(values)
    for q in qs:
        idx = int(q * (len(sorted_v) - 1))
        key = f"p{int(q * 100)}"
        result[key] = sorted_v[idx]
    result["mean"] = statistics.mean(values)
    return result


def print_report(r: LoadTestResults) -> None:
    completed = [g for g in r.game_metrics if g.completed]
    failed = [g for g in r.game_metrics if not g.completed]
    throughput = len(completed) / r.wall_time_s * 60 if r.wall_time_s > 0 else 0

    print(f"\n{'=' * 50}")
    print(f"  LOAD TEST RESULTS")
    print(f"{'=' * 50}")
    print(f"  Target:      {r.url}")
    print(f"  Mode:        {r.mode}")
    print(f"  Difficulty:  {r.difficulty}")
    print(f"  Concurrency: {r.concurrent}")
    print(f"  Total Games: {r.total_games}")
    print(f"  Wall Time:   {r.wall_time_s:.1f}s")
    print(f"  Throughput:  {throughput:.1f} games/min")

    # Room creation
    create_vals = [g.room_create_ms for g in r.game_metrics]
    if create_vals:
        s = _percentiles(create_vals)
        print(f"\n  --- Room Creation (HTTP) ---")
        print(f"    Mean: {s['mean']:.0f}ms  p50: {s['p50']:.0f}ms  p95: {s['p95']:.0f}ms")

    # Join latency (duo/quad)
    join_vals = [j for g in r.game_metrics for j in g.join_ms]
    if join_vals:
        s = _percentiles(join_vals)
        print(f"\n  --- Room Join (HTTP) ---")
        print(f"    Mean: {s['mean']:.0f}ms  p50: {s['p50']:.0f}ms  p95: {s['p95']:.0f}ms  (N={len(join_vals)})")

    # WS connect
    ws_vals = [sm.ws_connect_ms for g in r.game_metrics for sm in g.seat_metrics if sm.ws_connect_ms > 0]
    if ws_vals:
        s = _percentiles(ws_vals)
        print(f"\n  --- WS Connect to First State ---")
        print(f"    Mean: {s['mean']:.0f}ms  p50: {s['p50']:.0f}ms  p95: {s['p95']:.0f}ms  (N={len(ws_vals)})")

    # Move RTT
    rtt_vals = [rtt for g in r.game_metrics for rtt in g.move_rtts_ms]
    if rtt_vals:
        s = _percentiles(rtt_vals)
        print(f"\n  --- Move Round-Trip Time ---")
        print(f"    Mean: {s['mean']:.0f}ms  p50: {s['p50']:.0f}ms  p95: {s['p95']:.0f}ms  (N={len(rtt_vals)})")

    # Games
    durations = [g.total_duration_s for g in completed]
    if durations:
        s = _percentiles(durations)
        print(f"\n  --- Games ---")
        print(f"    Completed:  {len(completed)}/{r.total_games} ({len(completed)/r.total_games*100:.0f}%)")
        print(f"    Duration:   mean={s['mean']:.1f}s  p50={s['p50']:.1f}s  p95={s['p95']:.1f}s")
        total_moves = sum(g.total_moves for g in completed)
        print(f"    Avg Moves:  {total_moves / len(completed):.1f}/game")
    else:
        print(f"\n  --- Games ---")
        print(f"    Completed: 0/{r.total_games}")

    # Errors
    if failed:
        print(f"\n  --- Errors ({len(failed)}) ---")
        error_counts: dict[str, int] = {}
        for g in failed:
            key = g.error or "unknown"
            # Truncate long errors
            if len(key) > 120:
                key = key[:120] + "..."
            error_counts[key] = error_counts.get(key, 0) + 1
        for err, count in sorted(error_counts.items(), key=lambda x: -x[1]):
            print(f"    [{count}x] {err}")
    else:
        print(f"\n  --- Errors ---")
        print(f"    (none)")

    print(f"{'=' * 50}\n")


def save_results(r: LoadTestResults) -> str:
    """Save results to runs/load_tests/<timestamp>.json. Returns the file path."""
    out_dir = Path("runs/load_tests")
    out_dir.mkdir(parents=True, exist_ok=True)

    completed = [g for g in r.game_metrics if g.completed]
    throughput = len(completed) / r.wall_time_s * 60 if r.wall_time_s > 0 else 0

    doc = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "url": r.url,
            "mode": r.mode,
            "difficulty": r.difficulty,
            "concurrent": r.concurrent,
            "total_games": r.total_games,
        },
        "summary": {
            "wall_time_s": round(r.wall_time_s, 2),
            "throughput_games_per_min": round(throughput, 1),
            "completed": len(completed),
            "failed": len(r.game_metrics) - len(completed),
            "room_create_ms": _percentiles([g.room_create_ms for g in r.game_metrics]),
            "ws_connect_ms": _percentiles(
                [sm.ws_connect_ms for g in r.game_metrics for sm in g.seat_metrics if sm.ws_connect_ms > 0]
            ),
            "move_rtt_ms": {
                **_percentiles([rtt for g in r.game_metrics for rtt in g.move_rtts_ms]),
                "samples": sum(len(g.move_rtts_ms) for g in r.game_metrics),
            },
            "game_duration_s": _percentiles([g.total_duration_s for g in completed]),
        },
        "games": [
            {
                "game_id": g.game_id,
                "completed": g.completed,
                "duration_s": round(g.total_duration_s, 2),
                "moves": g.total_moves,
                "messages": g.total_messages,
                "room_create_ms": round(g.room_create_ms, 1),
                "ws_connect_ms": round(g.ws_connect_ms, 1),
                "error": g.error,
            }
            for g in r.game_metrics
        ],
        "errors": [g.error for g in r.game_metrics if g.error],
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"{r.mode}_{ts}.json"
    path.write_text(json.dumps(doc, indent=2))
    return str(path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Load test for Guan Dan backend")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"Base URL (default: {DEFAULT_URL})")
    parser.add_argument("-c", "--concurrent", type=int, default=DEFAULT_CONCURRENT, help="Max concurrent games")
    parser.add_argument("-n", "--total-games", type=int, default=DEFAULT_TOTAL, help="Total games to run")
    parser.add_argument("-d", "--difficulty", default=DEFAULT_DIFFICULTY, help="Bot difficulty")
    parser.add_argument("--mode", choices=["solo", "duo", "quad"], default=DEFAULT_MODE, help="Game mode")
    args = parser.parse_args()

    results = asyncio.run(run_load_test(args.url, args.concurrent, args.total_games, args.difficulty, args.mode))
    print_report(results)
    path = save_results(results)
    print(f"  Results saved to: {path}\n")


if __name__ == "__main__":
    main()
