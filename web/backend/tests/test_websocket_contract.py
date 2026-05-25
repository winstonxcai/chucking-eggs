"""WebSocket contract tests for auth, reconnect tokens, and move errors."""

from __future__ import annotations

from collections.abc import Callable

import app.main as main_module
import pytest
from fastapi.testclient import TestClient
from guandan.cards import ComboType
from starlette.websockets import WebSocketDisconnect

app = main_module.app


def _claim(client: TestClient, username: str) -> dict:
    response = client.post("/api/auth/claim", json={"username": username, "is_test": True})
    assert response.status_code == 200, response.text
    return response.json()


def _create_room(client: TestClient, mode: str = "solo", seed: int = 42) -> dict:
    response = client.post(
        "/api/room/create",
        json={"mode": mode, "difficulty": "greedy", "seed": seed},
    )
    assert response.status_code == 200, response.text
    return response.json()


def _create_signed_room(client: TestClient, player: dict, mode: str = "duo", seed: int = 42) -> dict:
    response = client.post(
        "/api/room/create",
        json={"mode": mode, "difficulty": "greedy", "seed": seed},
        headers={"X-Player-Token": player["player_token"]},
    )
    assert response.status_code == 200, response.text
    return response.json()


def _join_signed_room(client: TestClient, room_code: str, player: dict) -> dict:
    response = client.post(
        f"/api/room/join/{room_code}",
        headers={"X-Player-Token": player["player_token"]},
    )
    assert response.status_code == 200, response.text
    return response.json()


def _recv_until(
    ws,
    msg_type: str,
    predicate: Callable[[dict], bool] | None = None,
    *,
    limit: int = 20,
) -> dict:
    for _ in range(limit):
        message = ws.receive_json()
        if message.get("type") == msg_type and (predicate is None or predicate(message)):
            return message
    raise AssertionError(f"Did not receive {msg_type!r} within {limit} messages")


def _card_id(card) -> str:
    return f"{card.rank}-{card.suit}-{card.deck}"


def _create_room_where_human_leads(client: TestClient) -> dict:
    for seed in range(100):
        room = _create_room(client, seed=seed)
        game_room = main_module.game_manager.get_room(room["game_id"])
        assert game_room is not None
        if game_room.env.current_player == 0 and game_room.env.is_leading():
            return room
    raise AssertionError("Could not find a deterministic seed where human leads")


def test_websocket_rejects_invalid_player_token() -> None:
    with TestClient(app) as client:
        room = _create_room(client)
        with client.websocket_connect(
            f"/ws/game/{room['game_id']}?seat=0&token={room['reconnect_token']}&player_token=bad-token"
        ) as ws:
            with pytest.raises(WebSocketDisconnect) as exc:
                ws.receive_json()
        assert exc.value.code == 4003


def test_websocket_rejects_missing_reconnect_token() -> None:
    with TestClient(app) as client:
        room = _create_room(client)

        with client.websocket_connect(f"/ws/game/{room['game_id']}?seat=0") as ws:
            with pytest.raises(WebSocketDisconnect) as exc:
                ws.receive_json()

        assert exc.value.code == 4003


def test_websocket_rejects_blank_reconnect_token() -> None:
    with TestClient(app) as client:
        room = _create_room(client)

        with client.websocket_connect(f"/ws/game/{room['game_id']}?seat=0&token=") as ws:
            with pytest.raises(WebSocketDisconnect) as exc:
                ws.receive_json()

        assert exc.value.code == 4003


def test_websocket_rejects_mismatched_player_id_and_token_without_room_corruption() -> None:
    with TestClient(app) as client:
        player_a = _claim(client, "tok-a")
        player_b = _claim(client, "tok-b")
        room = _create_signed_room(client, player_a)

        with client.websocket_connect(
            f"/ws/game/{room['game_id']}?seat=0&token={room['reconnect_token']}"
            f"&player_id={player_a['player_id']}&player_token={player_b['player_token']}"
        ) as ws:
            with pytest.raises(WebSocketDisconnect) as exc:
                ws.receive_json()

        assert exc.value.code == 4003
        game_room = main_module.game_manager.get_room(room["game_id"])
        assert game_room is not None
        assert game_room.player_ids[0] is None
        assert not game_room.env.done


def test_websocket_rejects_reconnect_token_for_wrong_seat() -> None:
    with TestClient(app) as client:
        creator = _create_room(client, mode="duo")
        joiner_response = client.post(f"/api/room/join/{creator['room_code']}")
        assert joiner_response.status_code == 200, joiner_response.text
        joiner = joiner_response.json()

        with client.websocket_connect(
            f"/ws/game/{creator['game_id']}?seat={joiner['seat']}&token={creator['reconnect_token']}"
        ) as ws:
            with pytest.raises(WebSocketDisconnect) as exc:
                ws.receive_json()
        assert exc.value.code == 4003


def test_websocket_rejects_invalid_move_protocol_without_closing_or_desyncing() -> None:
    with TestClient(app) as client:
        room = _create_room_where_human_leads(client)
        with client.websocket_connect(
            f"/ws/game/{room['game_id']}?seat=0&token={room['reconnect_token']}"
        ) as ws:
            state = _recv_until(ws, "game_state", lambda msg: msg["is_my_turn"] is True)
            assert state["is_my_turn"] is True
            assert state["is_leading"] is True

            first_card_id = state["my_hand"][0]["id"]
            different_rank_pair = None
            for left in state["my_hand"]:
                for right in state["my_hand"]:
                    if left["id"] != right["id"] and left["rank"] != right["rank"]:
                        different_rank_pair = [left["id"], right["id"]]
                        break
                if different_rank_pair:
                    break
            assert different_rank_pair is not None

            ws.send_json({"type": "pass"})
            error = _recv_until(ws, "error")
            assert error["message"] == "Cannot pass when leading"

            ws.send_json({"type": "play_cards", "card_ids": [first_card_id, first_card_id]})
            error = _recv_until(ws, "error")
            assert error["message"] == "Invalid combo"

            ws.send_json({"type": "play_cards", "card_ids": ["14-0-99"]})
            error = _recv_until(ws, "error")
            assert error["message"] == "Invalid combo"

            ws.send_json({"type": "play_cards", "card_ids": different_rank_pair})
            error = _recv_until(ws, "error")
            assert error["message"] == "Invalid combo"

            ws.send_json({"type": "play_cards", "card_ids": ["not-a-card"]})
            error = _recv_until(ws, "error")
            assert error["message"] == "Invalid combo"

            ws.send_text("not-json")
            error = _recv_until(ws, "error")
            assert error["message"] == "Invalid message"

            game_room = main_module.game_manager.get_room(room["game_id"])
            assert game_room is not None
            legal_play = next(c for c in game_room.env.legal_moves(0) if c.type != ComboType.PASS)
            ws.send_json({"type": "play_cards", "card_ids": [_card_id(c) for c in legal_play.cards]})
            played = _recv_until(ws, "move_played")
            assert played["seat"] == 0


def test_forfeit_requires_matching_signed_identity_and_persists_history() -> None:
    with TestClient(app) as client:
        player_a = _claim(client, "forfeit-a")
        player_b = _claim(client, "forfeit-b")
        room = _create_signed_room(client, player_a, mode="duo")
        guest = _join_signed_room(client, room["room_code"], player_b)

        with (
            client.websocket_connect(
                f"/ws/game/{room['game_id']}?seat={room['seat']}&token={room['reconnect_token']}"
                f"&player_id={player_a['player_id']}&player_token={player_a['player_token']}"
            ) as ws_a,
            client.websocket_connect(
                f"/ws/game/{guest['game_id']}?seat={guest['seat']}&token={guest['reconnect_token']}"
                f"&player_id={player_b['player_id']}&player_token={player_b['player_token']}"
            ) as ws_b,
        ):
            _recv_until(ws_a, "game_state")
            _recv_until(
                ws_b,
                "game_state",
                lambda msg: any(p["name"] == player_a["username"] for p in msg["players"]),
                limit=50,
            )

            mismatch = client.post(
                f"/api/room/{room['game_id']}/forfeit",
                json={"player_id": player_a["player_id"]},
                headers={"X-Player-Token": player_b["player_token"]},
            )
            assert mismatch.status_code == 401

            game_room = main_module.game_manager.get_room(room["game_id"])
            assert game_room is not None
            assert not game_room.env.done

            ok = client.post(
                f"/api/room/{room['game_id']}/forfeit",
                json={"player_id": player_a["player_id"]},
                headers={"X-Player-Token": player_a["player_token"]},
            )
            assert ok.status_code == 200, ok.text
            forfeited = _recv_until(ws_b, "game_forfeited")
            assert forfeited["forfeiter_name"] == player_a["username"]

        profile = client.get(f"/api/profile/{player_a['username']}")
        assert profile.status_code == 200, profile.text
        data = profile.json()
        assert data["player"]["games_played"] == 1
        assert data["player"]["elo"] < 1200
        assert data["games"][0]["result_type"] == "forfeit"
        forfeiter = next(p for p in data["games"][0]["players"] if p["player_id"] == player_a["player_id"])
        assert forfeiter["team_result"] == "forfeit"
