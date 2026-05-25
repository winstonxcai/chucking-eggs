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
