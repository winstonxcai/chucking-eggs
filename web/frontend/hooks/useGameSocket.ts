"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { GameState, ServerMessage, GameOverMsg } from "@/lib/types";

const WS_BASE = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000";

const MAX_RETRIES = 5;
const BACKOFF_BASE = 1000; // 1s, 2s, 4s, 8s, 8s

export type ConnectionStatus = "connecting" | "connected" | "reconnecting" | "disconnected";

export function useGameSocket(gameId: string | null, reconnectToken: string | null = null) {
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

  // Keep refs in sync
  gameIdRef.current = gameId;
  tokenRef.current = reconnectToken;

  const connect = useCallback(() => {
    const gid = gameIdRef.current;
    const token = tokenRef.current;
    if (!gid) return;

    const params = token ? `?token=${encodeURIComponent(token)}` : "";
    const ws = new WebSocket(`${WS_BASE}/ws/game/${gid}${params}`);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      setConnectionStatus("connected");
      retriesRef.current = 0;
    };

    ws.onclose = () => {
      setConnected(false);
      wsRef.current = null;

      // Don't reconnect if intentionally closed or game is over
      if (intentionalCloseRef.current) {
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
      const msg: ServerMessage = JSON.parse(event.data);

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
          sessionStorage.removeItem("gd_game_id");
          sessionStorage.removeItem("gd_reconnect_token");
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

  const playCards = useCallback((cardIds: string[]) => {
    wsRef.current?.send(JSON.stringify({ type: "play_cards", card_ids: cardIds }));
  }, []);

  const pass = useCallback(() => {
    wsRef.current?.send(JSON.stringify({ type: "pass" }));
  }, []);

  const createGroup = useCallback((cardIds: string[], comboType: string, comboName: string) => {
    wsRef.current?.send(JSON.stringify({
      type: "create_group", card_ids: cardIds, combo_type: comboType, combo_name: comboName,
    }));
  }, []);

  const deleteGroup = useCallback((groupId: string) => {
    wsRef.current?.send(JSON.stringify({ type: "delete_group", group_id: groupId }));
  }, []);

  return { gameState, aiThinking, gameOver, connected, connectionStatus, playCards, pass, createGroup, deleteGroup };
}
