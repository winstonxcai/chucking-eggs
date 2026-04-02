"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { GameState, ServerMessage, GameOverMsg } from "@/lib/types";
import { STORAGE_KEYS } from "@/lib/storage-keys";

const WS_BASE = (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000").replace(/^http/, "ws");

const MAX_RETRIES = 5;
const BACKOFF_BASE = 1000; // 1s, 2s, 4s, 8s, 8s

export type ConnectionStatus = "connecting" | "connected" | "reconnecting" | "disconnected";

export function useGameSocket(gameId: string | null, reconnectToken: string | null = null, seat: number = 0) {
  const wsRef = useRef<WebSocket | null>(null);
  const [gameState, setGameState] = useState<GameState | null>(null);
  const [aiThinking, setAiThinking] = useState<number | null>(null);
  const [gameOver, setGameOver] = useState<GameOverMsg | null>(null);
  const [connected, setConnected] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>("connecting");

  const retriesRef = useRef(0);
  const intentionalCloseRef = useRef(false);
  const gameIdRef = useRef(gameId);
  const tokenRef = useRef(reconnectToken);
  const seatRef = useRef(seat);

  // Keep refs in sync
  gameIdRef.current = gameId;
  tokenRef.current = reconnectToken;
  seatRef.current = seat;

  const connect = useCallback(() => {
    const gid = gameIdRef.current;
    const token = tokenRef.current;
    const seatNum = seatRef.current;
    if (!gid) return;

    const params = new URLSearchParams({ seat: String(seatNum) });
    if (token) params.set("token", token);
    const playerId = (() => { try { return localStorage.getItem(STORAGE_KEYS.PLAYER_ID); } catch { return null; } })();
    if (playerId) params.set("player_id", playerId);
    const ws = new WebSocket(`${WS_BASE}/ws/game/${gid}?${params.toString()}`);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      setConnectionStatus("connected");
      retriesRef.current = 0;
    };

    ws.onclose = (event) => {
      setConnected(false);
      wsRef.current = null;

      // Don't reconnect if intentionally closed or game is over
      if (intentionalCloseRef.current) {
        setConnectionStatus("disconnected");
        return;
      }

      // Permanent failures — don't retry, but keep session so a page refresh can reconnect
      if (event.code >= 4000) {
        console.error(`WebSocket closed: ${event.code} ${event.reason}`);
        setConnectionStatus("disconnected");
        return;
      }

      // Attempt reconnection with exponential backoff
      if (retriesRef.current < MAX_RETRIES) {
        setConnectionStatus("reconnecting");
        const delay = Math.min(BACKOFF_BASE * 2 ** retriesRef.current, 8000);
        retriesRef.current++;
        setTimeout(() => {
          if (gameIdRef.current) connect();
        }, delay);
      } else {
        setConnectionStatus("disconnected");
      }
    };

    ws.onerror = () => {
      // onclose will fire after this
    };

    ws.onmessage = (event) => {
      let msg: ServerMessage;
      try {
        msg = JSON.parse(event.data);
      } catch {
        console.error("WS: unparseable message", event.data);
        return;
      }

      switch (msg.type) {
        case "game_state":
          setGameState(msg);
          setAiThinking(null);
          break;
        case "move_played":
          break;
        case "ai_thinking":
          setAiThinking(msg.seat);
          break;
        case "game_over":
          setGameOver(msg);
          setAiThinking(null);
          // Clear session — game is done, no need to reconnect
          sessionStorage.removeItem(STORAGE_KEYS.GAME_ID);
          sessionStorage.removeItem(STORAGE_KEYS.RECONNECT_TOKEN);
          sessionStorage.removeItem(STORAGE_KEYS.SEAT);
          break;
        case "error":
          console.error("Game error:", msg.message);
          break;
      }
    };
  }, []);

  useEffect(() => {
    if (!gameId) return;

    setGameOver(null);
    setGameState(null);
    setAiThinking(null);
    retriesRef.current = 0;
    intentionalCloseRef.current = false;
    setConnectionStatus("connecting");

    connect();

    return () => {
      intentionalCloseRef.current = true;
      wsRef.current?.close();
      wsRef.current = null;
    };
  }, [gameId, connect]);

  const wsSend = useCallback((data: object) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data));
    }
  }, []);

  const playCards = useCallback((cardIds: string[]) => {
    wsSend({ type: "play_cards", card_ids: cardIds });
  }, [wsSend]);

  const pass = useCallback(() => {
    wsSend({ type: "pass" });
  }, [wsSend]);

  const createGroup = useCallback((cardIds: string[], comboType: string, comboName: string) => {
    wsSend({ type: "create_group", card_ids: cardIds, combo_type: comboType, combo_name: comboName });
  }, [wsSend]);

  const deleteGroup = useCallback((groupId: string) => {
    wsSend({ type: "delete_group", group_id: groupId });
  }, [wsSend]);

  return { gameState, aiThinking, gameOver, connected, connectionStatus, playCards, pass, createGroup, deleteGroup };
}
