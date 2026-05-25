# Testing

This project has four main test categories:

1. ML tests (engine + DART training stack)
2. Backend tests
3. Frontend tests
4. Frontend integration tests with the backend

CI runs all default suites in these categories. Slow and external-service tests are opt-in.

## ML

Run from the repository root:

```bash
uv run pytest -q ml/tests
```

ML lint and type-check:

```bash
uv run ruff check ml/src/guandan
uv run mypy
```

### What It Covers

- `ml/tests/test_cards.py`, `ml/tests/test_combos.py`, `ml/tests/test_game.py`
  Engine-level tests for card representation, combo enumeration/legality, and full-game turn flow. These guard the invariants the Rust extension and pure-Python fallback both have to satisfy.

- `ml/tests/test_agents.py`, `ml/tests/test_heuristic.py`
  Rule-bot smoke coverage: every registered agent returns legal moves over a short rollout, and the heuristic bot's specific rules behave as intended.

- `ml/tests/dart/test_agent.py`, `ml/tests/dart/test_config.py`
  `DartBot.load()` checkpoint round-trip and config schema migration (including legacy keys like `max_forced_k1_replay_frac` → `max_forced_pass_replay_frac`).

- `ml/tests/dart/model/`
  Model-layer tests: base encoder shape contracts, role-aware encoder channel layout, `DartQNet` forward/output shapes per trick-position head, and checkpoint save/load.

- `ml/tests/dart/data/`
  Replay buffer push/sample behavior, MC-return computation, and sample tagging used by `replay_mix` bucket sampling.

- `ml/tests/dart/runtime/`
  The most important DART suite. Covers the actor loop, the worker process, the learner loop, opponent-pool sampling, learner throttling under replay-ratio pressure, eval-lane handoff, and an end-to-end multi-process integration test. `test_train_smoke.py` runs a short real training loop and is the closest thing to a CI regression gate for the full pipeline.

- `ml/tests/dart/utils/`
  Schedule and legal-move utility tests.

### Native Extension

Most ML tests work against the pure-Python `guandan_rs` fallback. To exercise the native Rust move generator, rebuild it before running tests:

```bash
uv run maturin develop --release --manifest-path ml/src/guandan_rs/Cargo.toml
uv run pytest -q ml/tests
```

If `uv sync` removed the editable `_guandan_rs` install, rerun the maturin step.

### Focused Subsets

When iterating on a specific area, scope the run:

```bash
uv run pytest -q ml/tests/dart/runtime/test_train_smoke.py
uv run pytest -q ml/tests/dart/model
```

## Backend

Run from the repository root:

```bash
uv run pytest -q web/backend/tests
```

Backend lint:

```bash
uv run ruff check web/backend/app web/backend/tests
```

### What It Covers

- `web/backend/tests/test_auth.py`
  Tests signed player identity, token validation, username claiming, and auth edge cases.

- `web/backend/tests/test_multiplayer.py`
  Tests room creation, joining, seat assignment, multiplayer start behavior, and basic room protocol behavior.

- `web/backend/tests/test_room_lifecycle.py`
  Tests room cleanup, lobby lifecycle, disconnect handling, forfeit paths, and one-active-game behavior.

- `web/backend/tests/test_solo_bots.py`
  Tests solo-game creation and advertised bot compatibility at the backend layer.

- `web/backend/tests/test_trick_history.py`
  Tests trick history recording and game-over review data. These are lower-level game-room tests that bypass the browser but verify important game-state invariants.

- `web/backend/tests/test_websocket_contract.py`
  Tests WebSocket/API contract behavior: invalid player tokens, mismatched identity, wrong reconnect token, invalid move payloads, duplicate card ids, illegal combos, signed forfeit, Elo updates, and history persistence.

- `web/backend/tests/test_mongo_smoke.py`
  Opt-in smoke test for a real MongoDB-backed deployment path.

### Mongo Smoke Test

The Mongo smoke test is skipped by default because it requires an external MongoDB instance. Run it explicitly when validating production-like persistence:

```bash
RUN_MONGO_SMOKE=true MONGODB_URL='mongodb://localhost:27017' uv run pytest -q web/backend/tests/test_mongo_smoke.py
```

The test uses a temporary test database prefix and verifies basic player/game persistence, idempotent game save behavior, and profile/stat reads.

## Frontend

Run from `web/frontend`:

```bash
npm run lint
npm run test:unit
npm run build
```

### What It Covers

- `npm run lint`
  Runs ESLint against the frontend codebase.

- `npm run test:unit`
  Runs Vitest unit tests. Current unit coverage includes pure card/client logic such as `web/frontend/lib/cards.test.ts`.

- `npm run build`
  Runs a production Next.js build and TypeScript check. This catches route/build-time regressions that browser tests may not surface.

These tests should stay fast and should not require a running backend.

## Frontend Integration With Backend

These are Playwright browser tests that start an isolated backend and frontend server with explicit ports and test environment variables. They should not reuse a developer's already-running app server.

Run from `web/frontend`:

```bash
npm run e2e
npm run e2e:integration
```

### Standard E2E

```bash
npm run e2e
```

This starts:

- Backend: `http://127.0.0.1:8001`
- Frontend: `http://127.0.0.1:3001`

It runs the normal Playwright suite in `web/frontend/e2e` and covers:

- First-run auth modal and username claim flows
- Solo game loading across bot difficulties
- Core game UI behavior and trick-zone rendering
- Invalid move recovery without client/server desync
- Multiplayer duo and quad room creation/joining
- Deterministic quad random-move smoke coverage with four real browser clients
- Reconnect and refresh behavior
- Forfeit UI and broadcast behavior
- Active-game banner and lifecycle behavior
- Profile page rendering and chart/feed states
- Layout stability checks for key game controls

### Zero-Timeout Integration E2E

```bash
npm run e2e:integration
```

This starts:

- Backend: `http://127.0.0.1:8010`
- Frontend: `http://127.0.0.1:3010`

It uses `playwright.integration.config.ts` and explicit fast-test environment:

```bash
HUMAN_TURN_TIMEOUT_S=0
DISCONNECT_TAKEOVER_S=10
LOBBY_TIMEOUT_S=5
CLEANUP_INTERVAL_S=0.5
ACTION_PAUSE_S=0
AI_THINK_PAUSE_S=0
```

This suite covers backend-dependent timing and lifecycle paths that should not run against a normal local server:

- AFK auto-play
- Game completion under zero-timeout automation
- Auto-forfeit after disconnect timeout
- Game review modal and review navigation after game over
- Lobby expiry with a connected browser
- Fast AFK-driven duo board advancement

### Integration Contract Tests

`web/frontend/e2e/integration-contract.spec.ts` contains browser-level contract tests for backend/client identity and lifecycle behavior:

- Wrong reconnect token shows an authorization error and no board access
- Disconnect and reconnect before takeover preserves seat identity
- Duplicate tab for the same seat leaves the newest connection alive
- Active signed player is blocked from joining a second room
- Profile history reflects a signed forfeit game
- Rematch creates a new multiplayer room and both players can land in it
- Lobby expiry sends the browser home without a stale active-game banner
- Opt-in slow full quad game completion

### Slow Full-Game Quad Test

The full four-human-client game completion test is intentionally skipped by default. It is useful for nightly or pre-release runs, not every PR:

```bash
RUN_SLOW_E2E=true npm run e2e:integration
```

This runs the `@slow-full-game` Playwright test with zero-timeout AFK automation and a longer timeout.

## CI Expectations

CI currently runs:

```bash
uv run ruff check ml/src/guandan web/backend/app examples
uv run mypy
uv run pytest -q ml/tests
uv run pytest -q web/backend/tests

cd web/frontend
npm run lint
npm run test:unit
npm run build
npm run e2e
npm run e2e:integration
```

The default release gate is:

- Backend tests pass
- Frontend lint/unit/build pass
- Standard Playwright E2E passes
- Zero-timeout Playwright integration E2E passes
- Mongo smoke and slow full-game tests are run manually before major releases when needed
