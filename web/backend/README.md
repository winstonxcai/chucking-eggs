# Guan Dan Web Backend

FastAPI service for the multiplayer Guan Dan web app. It owns room lifecycle,
WebSocket game state, AI move dispatch, Elo updates, and MongoDB persistence.

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
- `REDIS_URL`: optional Redis connection string. The app falls back to an
  in-memory implementation for local development and tests.
- `DATA_DIR`: optional path for local game data fixtures.

Never commit a populated `.env`; only `.env.example` belongs in git.

## Tests

```bash
uv run pytest -q web/backend/tests
```

CI runs these backend tests from the repo root.
