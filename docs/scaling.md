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
