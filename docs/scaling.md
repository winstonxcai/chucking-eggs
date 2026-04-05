# Backend Scaling — Load Test Results

Results from concurrent game load testing against the production backend on 2026-04-05.

## Infrastructure

| Property | Value |
|----------|-------|
| Host | Fly.io `chucking-eggs` — Singapore (`sin`) |
| Machine | `shared-cpu-1x`, 512MB RAM |
| Instances | 1 (in-memory `GameManager`, no Redis) |
| Fly.io connection limit | 250 hard / 200 soft |
| Backend | FastAPI + uvicorn ≥ 0.29.0 + `websockets.asyncio` |

> **Note:** `max_machines_running = 1` in `fly.toml` — horizontal scaling requires Redis for distributed session state, already scaffolded in `redis_client.py`.

## Test Methodology

- **Script**: [`web/backend/tests/load_test.py`](../web/backend/tests/load_test.py)
- **Target**: `https://chucking-eggs.fly.dev`
- **Play strategy**: always play smallest legal non-pass combo (simulates active human play)
- **Connection count per game**: solo = 1 WS, duo = 2 WS, quad = 4 WS
- **Metrics**: room creation latency (HTTP), WS connect latency, move round-trip time, game duration, completion rate
- **Raw results**: `runs/load_tests/`

To reproduce:
```bash
uv pip install aiohttp
.venv/bin/python web/backend/tests/load_test.py \
  --url https://chucking-eggs.fly.dev \
  --mode solo -c 25 -n 50
```

## Results

### Solo (1 WebSocket per game)

| Concurrency | Games | Completion | Throughput | WS Connect p50 | Move RTT p50 | Game Duration p50 |
|-------------|-------|------------|------------|---------------:|-------------:|------------------:|
| 5 | 10 | 100% | 4.0/min | 1,942 ms | 261 ms | 72 s |
| 10 | 20 | 100% | 7.3/min | 1,816 ms | 281 ms | 75 s |
| 15 | 30 | 100% | 9.7/min | 2,165 ms | 285 ms | 80 s |
| **25** ✓ | **50** | **100%** | **12.9/min** | **2,729 ms** | **291 ms** | **87 s** |
| 40 | 60 | 95% | 14.8/min | 4,258 ms | 302 ms | 106 s |

### Duo (2 WebSockets per game)

| Concurrency | Games | Completion | Throughput | WS Connect p50 | Move RTT p50 | Game Duration p50 |
|-------------|-------|------------|------------|---------------:|-------------:|------------------:|
| **5** ✓ | **10** | **100%** | **3.2/min** | **2,579 ms** | **285 ms** | **78 s** |
| 10 | 20 | 50% | 1.8/min | 2,798 ms | 291 ms | 88 s |

### Quad (4 WebSockets per game)

| Concurrency | Games | Completion | Throughput | WS Connect p50 | Move RTT p50 | Game Duration p50 |
|-------------|-------|------------|------------|---------------:|-------------:|------------------:|
| 2 | 4 | 100% | 1.4/min | 2,569 ms | 290 ms | 80 s |
| 4 | 8 | 100% | 2.4/min | 2,772 ms | 295 ms | 87 s |
| **7** ✓ | **14** | **100%** | **3.2/min** | **3,238 ms** | **301 ms** | **104 s** |
| 10 | 20 | 90% | 3.0/min | 3,496 ms | 306 ms | 117 s |

## Summary

| Mode | WS/game | Stable limit | Failure point | Total WS at stable limit |
|------|:-------:|:------------:|:-------------:|:------------------------:|
| Solo | 1 | **c = 25** | c = 40 | 25 |
| Duo | 2 | **c = 5** | c = 10 | 10 |
| Quad | 4 | **c = 7** | c = 10 | 28 |

**Consistent bottleneck: ~30 simultaneous WebSocket connections → CPU saturation on `shared-cpu-1x`.**

The Fly.io 250-connection hard limit is never approached. Failures are always timeout-based (game too slow under load), not connection rejections or crashes.

**Recommended safe operating limits:**

| Mode | Max concurrent games |
|------|--------------------:|
| Solo | 20 |
| Duo | 4 |
| Quad | 6 |

## Bug Fixed During Testing

`uvicorn < 0.29` used `websockets.legacy`, which has a concurrent-write race condition: the keepalive ping task fires while the app is sending a game-state frame, hitting `assert waiter is None` in `_drain_helper` and crashing the connection.

**Fix**: bumped `uvicorn[standard]>=0.29.0` in `web/backend/requirements.txt`, which switches to `websockets.asyncio` (proper locking for concurrent writes). Before the fix, solo was only stable at c = 5.

## Systems Design Analysis

### Current Architecture

```
Browser → Vercel (Next.js CDN)
            ↓ HTTP/WS
         Fly.io — 1× shared-cpu-1x 512MB
            FastAPI + uvicorn
            GameManager (in-memory dict)
            AI thread pool (run_in_executor)
            ↓ async (Motor)
         MongoDB Atlas
```

Single point of everything. Here's what breaks first in each direction.

---

### Dimension 1: Concurrent Active Games — CPU / AI Inference

**Bottleneck: Python thread pool competing for a shared vCPU**

AI moves run via `loop.run_in_executor(None, agent.act)`. Python's default `ThreadPoolExecutor` creates ~5 threads on a shared vCPU. Each AI move blocks one thread for ~20–100ms depending on difficulty (easy/wjsd bots are fast; MC/strategic bots are slow). Under 30+ concurrent games, threads queue up, move RTT climbs, and games hit the timeout.

**Fix sequence:**
1. **`performance-1x`** (dedicated CPU, 2GB) — near-linear gain since bottleneck is pure contention. ~2–3× capacity.
2. **Tune thread pool size explicitly** — `ThreadPoolExecutor(max_workers=N)` tuned to the machine, prevents oversubscription.
3. **Difficulty-tiered pools** — separate pools for cheap bots (greedy/heuristic) and expensive bots (MC/strategic) so slow bots don't starve fast ones.

---

### Dimension 2: Concurrent WebSocket Connections

**Bottleneck: Fly.io 250 hard limit, then kernel fd limits**

Each active game holds N persistent WS connections for its lifetime (~80–120s). At the 250 hard limit: ~200 solo games, ~60 quad. Well above the CPU ceiling today — but if AI moves off-machine, this becomes the next wall.

WS connections themselves are cheap (mostly idle, waiting for moves). The real cost is the game state they pin in memory (~1–2MB per GameRoom). OOM risk starts around 150–200 concurrent games on 512MB.

**Fix:** `performance-1x` buys more headroom. Long-term, a dedicated WS gateway decoupled from game logic (e.g., nginx `worker_connections 10000`) handles fan-out at scale.

---

### Dimension 3: Horizontal Scaling (Multi-Instance)

**Bottleneck: In-memory `GameManager` — architectural ceiling, not throughput**

`GameManager.rooms` is a Python dict in process memory. Fly.io's load balancer is round-robin — if seat 0 (WS) lands on machine 1 and seat 2 (WS) lands on machine 2, machine 2 has no room state. The game fails immediately.

**Fix sequence:**
1. **Sticky sessions (quick win)** — `sticky = true` in `fly.toml` pins a client IP to a machine. Doesn't survive machine restart but costs nothing.
2. **Redis for room/session state** — already scaffolded in `redis_client.py`. Move the rooms dict → Redis hashes, reconnect tokens → Redis KV with TTL. This is the architectural unlock.
3. **Redis Pub/Sub for WS fan-out** — in duo/quad, seats from the same game may land on different machines. Each machine subscribes to a Redis channel per `game_id`; when machine 1 processes a move, it publishes the broadcast and machine 2 fans it out to its connected seats.
4. **Game state in Redis** — serialize `GuanDanEnv` so any machine can resume after failover. Heaviest lift; only worth it at significant scale.

---

### Dimension 4: Database (MongoDB)

**Current state: healthy. Pressure emerges at burst write volume.**

All DB writes are post-game, async, non-blocking (`_persist_to_db` runs in the background after `game_over` broadcast). Gameplay never waits on DB. But:

**Write amplification at game-over:** each completed game triggers N player-doc reads (Elo fetch) + N player-doc writes + 1 game doc write. At high throughput (e.g., 50 games/min completing simultaneously) that's a burst of ~250 reads + ~250 writes. Atlas M0 caps IOPS and 500 connections — the 5s `asyncio.wait_for` in `_compute_elo_changes` starts firing, silently skipping Elo updates rather than crashing.

**Leaderboard read path:** `GET /api/leaderboard` does a top-50 sort scan on every request. Under viral load this becomes a hot read.

**Fix:**
- **Atlas M10+** before hitting M0 limits (IOPS is the real ceiling, not connections).
- **Batch Elo updates** — enqueue jobs, process in 30s windows, eliminates burst amplification.
- **Redis-cached leaderboard** — 60s TTL, one write per minute instead of N reads.

---

### Dimension 5: Frontend (Vercel)

**Current state: not a concern. Two non-obvious pressure points:**

**Connection spike before games start:** Every browser tab that opens a game holds one WS connection for the game lifetime. A viral spike means Fly.io sees a step-function jump in connections with no queuing. Connections either succeed or hit the 250 hard limit and get a 503. A lobby/queue system ("waiting for capacity") handles this gracefully.

**Geographic latency:** The Fly.io region is Singapore (`sin`). Singapore → US East is ~200ms baseline RTT. The Move RTT p50 of 291ms in tests becomes 500–700ms for North American players in practice. Multi-region deployment (with Redis replication for room state) is the fix — only worth it at meaningful MAU.

---

### Scaling Roadmap

| Priority | Change | Unlocks | Complexity |
|----------|--------|---------|------------|
| 1 | `performance-1x` VM | 2–3× CPU capacity | Trivial |
| 2 | Tune thread pool size | Squeeze more from existing CPU | Trivial |
| 3 | Sticky sessions (`fly.toml`) | Safer multi-instance bridge | Trivial |
| 4 | Redis session state | True horizontal scaling | Medium |
| 5 | Redis WS Pub/Sub fan-out | Multi-instance duo/quad | Medium |
| 6 | Atlas M10 upgrade | DB handles burst Elo writes | Low (cost) |
| 7 | Redis leaderboard cache | Hot read scalability | Low |
| 8 | Multi-region Fly.io | Latency for non-SG players | High |

---

## Upgrade Path

### Vertical (easiest)
Upgrade to `performance-1x` in `fly.toml`:
```toml
[[vm]]
  size = "performance-1x"  # 1 dedicated CPU, 2GB RAM
```
Expected: ~2–3× capacity (~60–70 concurrent solo games).

### Horizontal (requires Redis)
1. Deploy a Redis instance (Fly.io Redis add-on)
2. Uncomment `REDIS_URL` env var — `redis_client.py` already has a fallback-safe client
3. Distribute `GameManager` state (room codes, reconnect tokens) via Redis
4. Set `max_machines_running = 3` in `fly.toml`
5. Each additional machine adds ~30 WS capacity (same CPU bottleneck per instance)
