# Room Lifecycle & Joining Rules

This document is the canonical reference for how game rooms are created, joined, managed, and destroyed. It covers all invariants enforced by the backend and should be updated whenever the rules change.

---

## Room States

A room passes through three states:

```
LOBBY ──(all humans connect via WS)──► ACTIVE ──(game ends)──► FINISHED
  │                                                                │
  └──(host leaves / lobby_timeout / cleanup)──► DESTROYED ◄───────┘
```

| State | `room.started` | `room.env.done` | Description |
|-------|---------------|-----------------|-------------|
| LOBBY | `False` | `False` | Waiting for all human seats to WebSocket-connect |
| ACTIVE | `True` | `False` | Game in progress |
| FINISHED | `True` | `True` | Game over; room kept briefly for rematch |
| DESTROYED | — | — | Removed from `GameManager.rooms` |

> **Solo rooms skip LOBBY** — `room.started = True` immediately on creation.

---

## Room Modes

| Mode | Human Seats | Room Code | AI Seats |
|------|-------------|-----------|----------|
| `solo` | {0} | None | 1, 2, 3 |
| `duo` | {0, 2} | 6-char | 1, 3 |
| `quad` | {0, 1, 2, 3} | 6-char | none |

Seat 0 is always the creator. Partners in duo mode share seats 0 (creator) and 2.

---

## Player Identity

Players are identified by a UUID (`player_id`) stored in MongoDB, sent via the `X-Player-ID` HTTP header. Anonymous users (no header) are supported but receive no ownership enforcement.

Two identity dicts exist per room:

| Field | Type | Populated at | Purpose |
|-------|------|-------------|---------|
| `room.seat_player_ids` | `dict[int, str \| None]` | HTTP create/join | Ownership enforcement, one-room-per-player checks |
| `room.player_ids` | `dict[int, str \| None]` | WebSocket connect | Display name, Elo tracking, forfeit lookup |

**Critical distinction:** `player_ids[seat]` is `None` until the WebSocket handshake. A player who claimed a seat via HTTP but hasn't connected yet only appears in `seat_player_ids`.

---

## Room Creation Rules

**Endpoint:** `POST /api/room/create`

1. Creator always receives seat 0.
2. A room code (6-char uppercase alphanumeric, cryptographically secure) is generated for duo/quad; solo rooms have no code.
3. If the request includes a valid `X-Player-ID`:
   - **If the player is in any ACTIVE room** → HTTP 409 `already_in_game`. The player must finish or forfeit their current game first.
   - **If the player is in any LOBBY room** → that room is auto-closed (broadcast `room_closed` with reason `creator_left`) before the new room is created.
   - Only one LOBBY room per player is allowed at a time.
4. If no `X-Player-ID` is provided → no ownership check; room is created unconditionally.

---

## Room Joining Rules

**Endpoint:** `POST /api/room/join/{room_code}`

Joining is the most complex part of the lifecycle. Rules applied in order:

### A. Room eligibility checks (always, before identity checks)
1. Room code must exist and map to a known room.
2. Room must not be in ACTIVE or FINISHED state (`room.started == False`).
3. At least one human seat must be unassigned (`room.human_seats - room.assigned_seats` is non-empty).
   - If all human seats are taken → HTTP 404 `Room not found or already full`.

### B. Joiner identity checks (only if `X-Player-ID` provided)
4. **Self-join guard:** If the joiner's `player_id` matches `room.seat_player_ids[0]` (i.e., they're the creator), skip ownership conflict checks for this room. The join will fail naturally at step A.3 since seat 0 is already in `assigned_seats`.
5. **For all OTHER rooms** (not the target room):
   - If the joiner is in any ACTIVE room → HTTP 409 `already_in_game`.
   - If the joiner is in any LOBBY room → that room is auto-closed before completing the join.

### C. Seat assignment
6. Assign the lowest-numbered available human seat.
7. Add to `room.assigned_seats` (HTTP-level claim).
8. Record `room.seat_player_ids[seat] = joiner_player_id`.

### Error codes from join

| Condition | HTTP status | Detail |
|-----------|-------------|--------|
| Room code not found | 404 | `Room not found or already full` |
| Room already started | 404 | `Room not found or already full` |
| All seats taken | 404 | `Room not found or already full` |
| Joiner in an active game | 409 | `already_in_game` |

---

## Lobby-to-Game Transition

The LOBBY→ACTIVE transition happens at **WebSocket connect time**, not HTTP join time.

Trigger: all seats in `room.human_seats` have an entry in `room.connections`.

```python
if not room.started and all(s in room.connections for s in room.human_seats):
    room.started = True
    await room.broadcast_game_state()
    if current_player is AI:
        await room.run_ai_turns()
```

This means a player who claims a seat via HTTP but never opens a WebSocket will block the game from starting indefinitely — mitigated by the lobby timeout (see below).

---

## Leaving the Lobby

**Endpoint:** `POST /api/room/{game_id}/leave`

- Only valid in LOBBY state (`room.started == False`).
- If the host (seat 0) leaves → room is dissolved for all players. Broadcast: `{"type": "room_closed", "reason": "Host left the room"}`.
- If a non-host leaves → seat is freed (removed from `assigned_seats`). Room stays open.
- If all seats become empty → room is dissolved.

---

## Lobby Timeout

Unstarted duo/quad rooms auto-close after `LOBBY_TIMEOUT = 300` seconds (5 minutes).

- Checked in the cleanup loop every 30 seconds (`CLEANUP_INTERVAL`).
- Measured from `room.created_at` (not `room.last_activity` — the host's polling would otherwise reset it).
- Broadcast: `{"type": "room_closed", "reason": "lobby_timeout"}`.
- The frontend shows a countdown (`"Room closes in M:SS"`) derived from `lobby_expires_at` in the status response.
- Solo rooms are not subject to lobby timeout.

---

## Disconnect & Reconnect

### During LOBBY
- Players can disconnect and reconnect using their `reconnect_token`.
- If ALL human seats disconnect, `room.disconnected_at` is set.
- After 60 seconds (`GRACE_PERIOD`) with all seats disconnected, the room is removed by the cleanup loop (no broadcast — nobody is listening).

### During ACTIVE game
- `disconnect_seat(game_id, seat, ws)` marks the seat as disconnected.
- If the disconnected WebSocket is not the current active WebSocket for that seat (e.g., stale tab), the call is a no-op.
- After `DISCONNECT_TAKEOVER_S` (default 60s), `schedule_disconnect_takeover` fires:
  - Solo: AI takes over silently.
  - Duo/quad: Player forfeits.
- Reconnect via `reconnect_seat(game_id, seat, token)` cancels the pending takeover.

---

## Cleanup Mechanisms

The background `_cleanup_loop` runs every 30 seconds and checks (in priority order):

| Condition | Action |
|-----------|--------|
| Unstarted room, `created_at` > LOBBY_TIMEOUT (300s) | Broadcast `room_closed` (lobby_timeout), destroy |
| `disconnected_at` non-None and > GRACE_PERIOD (60s) | Destroy silently |
| `last_activity` > IDLE_TIMEOUT (300s) | Destroy silently |

`last_activity` is updated on every player message (move, heartbeat). Since the lobby page polls every 2 seconds, `last_activity` is continuously refreshed while the host is present — which is why `created_at` (not `last_activity`) is used for the lobby timeout.

Rooms can also be destroyed immediately by:
- `remove_room(game_id)` — called on game complete, rematch, or dissolve.
- `leave_room(game_id, seat)` — called when host leaves pre-game.

---

## One-Room-Per-Player Invariant

After the lobby timeout and one-room enforcement are implemented:

- A player with a known `player_id` can be in at most **one room at a time** (any state).
- Creating or joining a new room while in LOBBY → old LOBBY room is auto-closed.
- Creating or joining a new room while in ACTIVE → blocked (HTTP 409).
- If `player_id` is absent from the request header → no enforcement (anonymous fallback).

This invariant is enforced at HTTP create/join time, inside `_room_lock` to prevent race conditions.

---

## Room Code Properties

- 6 characters, uppercase A–Z and 0–9 (`string.ascii_uppercase + string.digits`).
- Cryptographically secure (`secrets.choice`).
- Guaranteed unique among currently-active rooms at generation time.
- Recycled after room destruction (no permanent registry).
- Case-insensitive on input (uppercased in `join_room`).

---

## Reconnect Tokens

- One per human seat, generated at room creation (`secrets.token_urlsafe(16)`).
- Stored in `room.reconnect_tokens[seat]`.
- Returned to the client at create/join time and stored client-side (sessionStorage).
- Used to authenticate WebSocket reconnections (`/ws/game/{game_id}?token=...&seat=...`).
- Never rotated during the room's lifetime.

---

## Sequence Diagrams

### Duo room: happy path

```
Creator                  Backend                 Joiner
  |                         |                      |
  |-- POST /api/room/create→|                      |
  |←─ {game_id, room_code,  |                      |
  |    seat=0, token}       |                      |
  |                         |                      |
  | [lobby page polls /status every 2s]            |
  |                         |                      |
  |                         |←─ POST /api/room/join/{code}
  |                         |──→ {game_id, seat=2, token}
  |                         |                      |
  |── WS /ws/game/{game_id}?seat=0&token=... ──→   |
  |                         |                      |── WS /ws/game/{game_id}?seat=2&token=... ──→
  |                         |                      |
  |    (all human seats connected → room.started=True)
  |←─ game_state broadcast ─|─────────────────────→|
```

### Lobby timeout

```
Creator                  Backend
  |                         |
  |-- POST /api/room/create→|   created_at = T
  |                         |
  | [polls /status, nobody joins]
  |                         |
  |                         | T + 300s: cleanup loop fires
  |←─ WS: room_closed       |   reason: "lobby_timeout"
  |   (lobby_timeout)       |
  |── router.push('/')      |
```

### Player creates second room while first is open

```
Player                   Backend
  |                         |
  |-- POST /room/create ───→|   room_A created, seat_player_ids[0] = player_id
  |                         |
  |-- POST /room/create ───→|   _rooms_for_player(player_id) → [room_A]
  |                         |   room_A.started == False → schedule close
  |                         |   room_A removed from rooms dict
  |                         |   room_B created
  |←─ {room_B details}      |
  |                         |──→ WS room_closed (creator_left) to room_A lobby page
```

---

## Constants Reference

| Constant | Value | Location |
|----------|-------|----------|
| `GRACE_PERIOD` | 60s | `game_manager.py` |
| `IDLE_TIMEOUT` | 300s | `game_manager.py` |
| `CLEANUP_INTERVAL` | 30s | `game_manager.py` |
| `LOBBY_TIMEOUT` | 300s | `game_manager.py` (planned) |
| `DISCONNECT_TAKEOVER_S` | 60s | `game_room.py` |
