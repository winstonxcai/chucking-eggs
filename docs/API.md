# Web Backend API

The backend is a FastAPI app under `web/backend/app`. It serves room creation,
profile/leaderboard data, and the game WebSocket used by the Next.js frontend.

Default local URLs:

- Backend: `http://localhost:8000`
- Frontend: `http://localhost:3000`
- WebSocket: `ws://localhost:8000/ws/game/{game_id}`

## Environment

- `ALLOWED_ORIGINS`: comma-separated CORS origins. Defaults to local frontend origins.
- `APP_ENV`: set to `production` in deployed environments.
- `AUTH_SECRET`: secret used to sign player identity tokens. Required when
  `APP_ENV=production`; set it through your host's secret manager.
- `DATA_DIR`: local JSONL game-record directory. Defaults to ignored `data/web_backend`.
- `DISCONNECT_TAKEOVER_S`: seconds before disconnected human seats forfeit or are taken over.
- `HUMAN_TURN_TIMEOUT_S`: AFK turn timer in seconds.
- `USE_MOCK_DB`: set to exactly `true` or `false`; use `true` for local/test
  runs without MongoDB. Production MongoDB initialization failures are fatal.
- MongoDB/Redis settings are read by `web/backend/app/db.py` and `redis_client.py`.

## HTTP Endpoints

### Bots

`GET /api/bots`

Returns the bot metadata shown by the frontend difficulty picker.

### Auth

`POST /api/auth/claim`

Body:

```json
{"username": "winston", "email": "optional@example.com", "is_test": false}
```

Claims or returns a player identity plus a signed `player_token`. Rate limited
to 5/minute per remote address. Send the token as `X-Player-Token` on
player-bound HTTP requests and as `player_token` on the game WebSocket.

### Profile And Leaderboard

`GET /api/profile/{username}`

Returns player metadata, game history, Elo history, and peak Elo.

`GET /api/leaderboard`

Returns human leaderboard entries and static bot leaderboard entries.

### Solo Compatibility

`POST /api/game/create`

Body:

```json
{"difficulty": "greedy"}
```

Creates a solo room and returns `game_id` plus seat-0 reconnect token. Prefer
`/api/room/create` for new clients.
Unknown difficulties return HTTP 400.

### Rooms

`POST /api/room/create`

Headers:

- `X-Player-Token`: optional signed player token used for in-game Elo/profile tracking.
- `X-Player-Id`: accepted only outside production for local smoke tests.

Body:

```json
{"mode": "solo", "difficulty": "greedy", "seed": 123}
```

`mode` is `solo`, `duo`, or `quad`. The creator always receives seat 0.
Unknown modes or difficulties return HTTP 400.

`POST /api/room/{game_id}/set_difficulty`

Body:

```json
{"difficulty": "strategic"}
```

Allowed only before the room starts.

`POST /api/room/join/{room_code}`

Headers:

- `X-Player-Token`: optional signed player token.
- `X-Player-Id`: accepted only outside production for local smoke tests.

Joins the next open human seat in a duo/quad room.

`GET /api/room/{game_id}/status`

Returns room mode, seat occupancy, start state, and lobby expiry.

`POST /api/room/{game_id}/leave`

Body:

```json
{"seat": 2}
```

Allowed only before the game starts.

`POST /api/room/{game_id}/forfeit`

Headers:

- `X-Player-Token`: signed player token for the forfeiting player.

Body:

```json
{"player_id": "player-object-id"}
```

Forfeits an active game for the matching human player. `player_id` is retained
for local compatibility; production authorization uses `X-Player-Token`.

`POST /api/room/{game_id}/rematch`

Creates a rematch after a finished game.

### Health

`GET /api/health`

Returns `{"status": "ok"}`.

## WebSocket Protocol

Connect:

```text
GET /ws/game/{game_id}?seat=0&token=<reconnect_token>&player_token=<player_token>
```

`seat` is the absolute seat assigned by the room create/join response. `token`
is optional on first connect and required for reconnect. `player_token` is
optional but needed for Elo/profile tracking. Unsigned `player_id` is accepted
only outside production for local smoke tests.

### Client Messages

Play cards:

```json
{"type": "play_cards", "card_ids": ["14-0-0", "14-1-0"]}
```

Pass:

```json
{"type": "pass"}
```

Abort before first move, converting that seat to AI control without Elo impact:

```json
{"type": "abort"}
```

Create a local hand group:

```json
{
  "type": "create_group",
  "card_ids": ["5-0-0", "5-1-0"],
  "combo_type": "pair",
  "combo_name": "Pair"
}
```

Delete a local hand group:

```json
{"type": "delete_group", "group_id": "grp-0-123456789"}
```

### Server Messages

`game_state`: full viewer-relative game state. Includes hand, legal moves,
players, trick state, groups, mode, and optionally `turn_deadline_ms`.

`ai_thinking`: bot turn indicator.

```json
{"type": "ai_thinking", "seat": 1}
```

`move_played`: emitted after a human or AI move.

```json
{"type": "move_played", "seat": 0, "combo": {}, "next_player": 3, "done": false}
```

`auto_played`: the AFK timer selected a move for the recipient.

`player_disconnected`: a seat disconnected and may be taken over or forfeited.

`game_over`: final finish order, rewards, player names, Elo changes, and trick history.

`game_forfeited`: forfeit result and Elo changes.

`error`: validation or protocol error.

Close codes:

- `4003`: requested seat is not valid for this room.
- `4004`: game id was not found.
