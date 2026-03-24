# Web App Scaling Phases

## Phase 1: "Friends Can Play" — Deploy & Stabilize

**Goal:** Get a public URL you can text to friends. 1-5 CCU, ~10 games.
**Cost:** Free tier or ~$5/month (Fly.io)
**Effort:** 2-3 days

### Infrastructure
- Containerize both services (Docker multi-stage for backend w/ Rust `guandan_rs` build)
- Deploy to **Fly.io** via `fly deploy` — built-in TLS, WebSocket proxying, no Caddy/nginx needed
- Backend and frontend as separate Fly apps (or single app with internal routing)
- `docker-compose.yml` kept for local dev only

### Code Changes
| File | Change |
|------|--------|
| `web/backend/app/main.py` | CORS from env var: `os.getenv("ALLOWED_ORIGINS")` instead of hardcoded localhost |
| `web/backend/app/game_room.py` | Add reconnection: store `reconnect_token` per room, keep room alive 60s after disconnect instead of instant removal |
| `web/backend/app/main.py` | Deferred room cleanup (replace immediate `remove_room` in `finally` with 60s grace period) |
| `web/backend/app/game_manager.py` | Add `last_activity` tracking + asyncio background task to expire idle rooms (>5 min) |
| `web/frontend/hooks/useGameSocket.ts` | Reconnection logic: store game_id/token in sessionStorage, exponential backoff (1s/2s/4s), "Reconnecting..." UI state |
| **New:** `web/backend/Dockerfile` | Multi-stage: Rust builds guandan_rs wheel, Python installs it + guandan package |
| **New:** `web/backend/fly.toml` | Fly.io config: app name, region, VM size, health check, WebSocket service |
| **New:** `web/frontend/Dockerfile` | Next.js standalone output mode |
| **New:** `docker-compose.yml` | Local dev only: backend + frontend services |

### Hardest Part
The backend Dockerfile — maturin + Rust toolchain in Docker, building the `guandan_rs` FFI wheel. Fly.io builds on x86, so the Rust compile happens natively in their builder (no cross-compilation issue).

---

## Phase 2: "Multiplayer" — Human vs Human with AI Fill

**Goal:** Two friends play as partners vs 2 AI, or 4 humans play together. 10-30 CCU.
**Cost:** ~$15/month (Fly scale-up + Upstash Redis)
**Effort:** 4-6 days
**Depends on:** Phase 1

### Infrastructure
- Add Redis (Upstash free tier or Fly Redis) for game state persistence + room discovery
- Scale Fly VM to 2 shared CPUs / 1GB if needed

### Code Changes
| File | Change |
|------|--------|
| `web/backend/app/game_room.py` | **Major refactor:** Replace `HUMAN_SEAT = 0` / `self.ws: WebSocket` with `human_seats: set[int]` / `connections: dict[int, WebSocket]`. `send()` becomes `broadcast()` + `send_to(seat)`. `run_ai_turns` only runs for AI-occupied seats. |
| `web/backend/app/main.py` | New endpoints: `POST /api/room/create` (mode: solo/duo/quad), `POST /api/room/join/{id}`, `GET /api/room/{id}/status`. WebSocket gets `seat` query param. |
| `web/backend/app/game_manager.py` | `join_room(game_id, seat)`, `list_rooms()`, Redis-backed persistence |
| `web/backend/app/serializer.py` | Already parameterized on `human_seat` — just call once per connected human |
| `web/frontend/app/page.tsx` | Expand from difficulty picker to lobby: "Play Solo" vs "Play with Friends" (create/join room via shareable code) |
| **New:** `web/frontend/app/lobby/page.tsx` | Waiting room: connected players, seat assignments, "Start" button |
| **New:** `web/frontend/hooks/useLobby.ts` | Polling hook for room status |
| `web/frontend/lib/types.ts` | Add `RoomInfo`, `LobbyState`, `SeatAssignment` |

### Hardest Part
The `game_room.py` refactor — the current design deeply assumes single human at seat 0 throughout the WebSocket lifecycle, message routing, and AI turn loop. The serialization layer is already multi-player ready (`human_seat` param), but the connection management is not.

---

## Phase 3: "Public Beta" — Accounts, Stats, Scale

**Goal:** Open registration, track stats, basic anti-abuse. 50-200 CCU.
**Cost:** ~$15-25/month (+ free-tier DB + Vercel)
**Effort:** 5-7 days
**Depends on:** Phase 2

### Infrastructure
- Database for games, players, stats (see tradeoff analysis below)
- Auth: GitHub OAuth or magic-link email (Resend free tier)
- Move frontend to Vercel (free CDN, independent scaling)
- Sentry free tier for error tracking

### Database: MongoDB vs PostgreSQL

**MongoDB (Lichess approach):**
| Pro | Con |
|-----|-----|
| Document-per-game is natural (4 players, 30-200 moves, result — one fetch, no JOINs) | Aggregation pipelines less ergonomic than SQL for leaderboards |
| Schema flexibility for fast iteration | Fewer managed free tiers with nice dashboards |
| Lichess precedent at 300M+ games | |
| Motor async driver integrates with FastAPI | |

**PostgreSQL:**
| Pro | Con |
|-----|-----|
| SQL aggregations for leaderboards | Requires JOINs or JSONB for move history |
| ACID transactions | Schema migrations needed |
| Neon/Supabase free tiers | ORM overhead or raw SQL |
| Relational queries ("games where X and Y both played") | |

### Elo Rating System

Team-aware Elo for 2v2 partnership game.

- **Initial rating:** 1200
- **Team rating:** Average of both partners' Elo
- **K-factor by finish margin:** maps from existing `get_rewards()` (+3 to -3)
  - 双上 (1st+2nd): K×1.5 | Normal win (1st+3rd): K×1.0 | Narrow (1st+4th): K×0.7
- **K-factor by experience:** 40 (first 30 games) → 24 (30-100) → 16 (100+)
- **Bot Elo anchors:** Easy=600, Medium=1000, Hard=1350, Expert=1700 (fixed)
- **Solo mode:** team = `(your_elo + bot_partner_elo) / 2`

### Code Changes
| File | Change |
|------|--------|
| **New:** `web/backend/app/auth.py` | JWT + OAuth/magic-link |
| **New:** `web/backend/app/db.py` | Database client + indexes |
| **New:** `web/backend/app/elo.py` | Elo computation |
| `web/backend/app/main.py` | Auth middleware, `/api/auth/*`, `/api/stats/*`, `/api/history` |
| `web/backend/app/game_room.py` | Persist GameRecord + update Elo at game end |
| `web/backend/app/game_manager.py` | Rate limiting: max 3 active games/user |
| **New:** `web/frontend/app/profile/page.tsx` | Elo, win rate, history sparkline |
| **New:** `web/frontend/app/history/page.tsx` | Game history list |

### Hardest Part
Auth + WebSocket: JWT must be validated during WebSocket upgrade handshake. Handle token expiry mid-game with long-lived tokens (7 days).

---

## Phase 4: "Production-Ready" — GPU AI, Matchmaking, Resilience

**Goal:** RL agent available online, auto-matchmaking, graceful degradation. 200-1000 CCU.
**Cost:** ~$30-80/month (Modal inference on-demand + LB)
**Effort:** 7-10 days
**Depends on:** Phase 3

### Infrastructure
- Modal `@web_endpoint` for RL/LSTM inference on T4 GPU (scale-to-zero)
- Horizontal backend: 2-3 instances behind load balancer, sticky sessions by game_id
- Redis pub/sub for cross-instance room events
- Matchmaking queue in Redis + background worker

### Code Changes
| File | Change |
|------|--------|
| **New:** `scripts/modal/serve_rl.py` | Modal web endpoint for RL inference |
| **New:** `web/backend/app/ai_remote.py` | HTTP client for Modal. 2s timeout + strategic fallback. |
| `web/backend/app/ai_service.py` | Route expert to remote RL inference |
| **New:** `web/backend/app/matchmaker.py` | Asyncio matchmaking worker |
| `web/backend/app/game_room.py` | `serialize()`/`deserialize()` for Redis. Spectator mode. |
| `web/backend/app/main.py` | `/api/matchmake` endpoints |
| **New:** `web/frontend/app/play/page.tsx` | "Find Match" UI |

### Hardest Part
1. **Modal RL serving** — encode on main server (state=417, action=160 per legal move), serialize tensors to Modal
2. **Horizontal WebSocket** — reconnection to different instance requires reconstructing GameRoom from Redis

---

## Summary

| Phase | Target | New Infra | Effort | Monthly Cost |
|-------|--------|-----------|--------|-------------|
| 1 — Deploy | 1-5 CCU | Fly.io + Docker | 2-3 days | ~$0-5 |
| 2 — Multiplayer | 10-30 CCU | + Redis | 4-6 days | ~$15 |
| 3 — Public Beta | 50-200 CCU | + DB + Auth + Vercel | 5-7 days | ~$15-25 |
| 4 — Production | 200-1000 CCU | + Modal inference + LB + Matchmaking | 7-10 days | ~$30-80 |

Each phase is independently shippable. Stop after Phase 1 (demo), Phase 2 (fun with friends), or Phase 3 (real product). Phase 4 only if you get traction.
