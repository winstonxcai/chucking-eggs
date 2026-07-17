# Guan Dan Web Backend

FastAPI service for the multiplayer Guan Dan web app. It owns room lifecycle,
WebSocket game state, AI move dispatch, Elo updates, and MongoDB persistence.
Room state is process-local; run one backend instance unless/until room state is
moved to Redis or another shared store.

## Local Development

From the repo root:

```bash
cp .env.example .env
uv sync --group dev
uv run uvicorn app.main:app --reload --app-dir web/backend
```

The backend listens on `http://localhost:8000`. The Next.js frontend expects
this URL during local development.

## Configuration

- `MONGODB_URL`: MongoDB connection string. Required for persistent accounts
  and leaderboard data.
- `APP_ENV`: set to `production` in deployed environments. Production MongoDB
  initialization failures are fatal.
- `AUTH_SECRET`: secret used to sign player identity tokens. Required when
  `APP_ENV=production`; set it through your host's secret manager, not git.
- `USE_MOCK_DB`: set to exactly `true` or `false`; use `true` for local/test
  runs without MongoDB.
- `REDIS_URL`: optional Redis connection string. The app falls back to an
  in-memory implementation for local development and tests.
- `DATA_DIR`: optional path for local JSONL game records. Defaults to the
  ignored `data/web_backend` directory from a repo checkout.

Never commit a populated `.env`; only `.env.example` belongs in git.

## Tests

```bash
uv run pytest -q web/backend/tests
```

CI runs these backend tests from the repo root.

## HTTP and WebSocket contract

The local service listens on `http://localhost:8000`; the frontend uses
`http://localhost:3000`, and games use
`ws://localhost:8000/ws/game/{game_id}`. The API is intentionally small:

- `GET /api/health` reports service health.
- `GET /api/bots` lists selectable bot metadata.
- `GET /api/profile/{username}` and `GET /api/leaderboard` expose profile and
  leaderboard data.
- `POST /api/auth/claim` creates or returns an identity and signed player token.
- `POST /api/game/create` is the legacy solo-game endpoint.
- `POST /api/room/create` creates a solo, duo, or quad room;
  `POST /api/room/join/{room_code}` joins one; and
  `GET /api/room/{game_id}/status` reads its state.
- `POST /api/room/{game_id}/set_difficulty` changes a pre-start room;
  `POST /api/room/{game_id}/leave` leaves it;
  `POST /api/room/{game_id}/forfeit` forfeits an active game; and
  `POST /api/room/{game_id}/rematch` creates a rematch.

Player-bound requests use `X-Player-Token`. Local tests may use
`X-Player-Id` outside production. Production requires `AUTH_SECRET` and does
not accept unsigned local identity headers.

### WebSocket messages

Connect with `GET /ws/game/{game_id}?seat=<seat>&token=<reconnect_token>`;
`player_token` may be added for Elo and profile tracking. Clients send
`play_cards`, `pass`, `abort`, `create_group`, and `delete_group` messages. The
server emits `game_state`, `ai_thinking`, `move_played`, `auto_played`,
`player_disconnected`, `game_over`, `game_forfeited`, and `error` messages.

The room response assigns the absolute seat and reconnect token. `game_state`
is viewer-relative and includes the hand, legal moves, players, trick state,
groups, mode, and an optional turn deadline. The server closes invalid-seat
connections with code `4003` and unknown games with code `4004`.
