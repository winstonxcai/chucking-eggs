"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { GameState, ServerMessage, ComboDTO, GameOverMsg } from "@/lib/types";

const WS_BASE = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000";

export function useGameSocket(gameId: string | null) {
  const wsRef = useRef<WebSocket | null>(null);
  const [gameState, setGameState] = useState<GameState | null>(null);
  const [aiThinking, setAiThinking] = useState<number | null>(null);
  const [gameOver, setGameOver] = useState<GameOverMsg | null>(null);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    if (!gameId) return;

    setGameOver(null);
    setGameState(null);
    setAiThinking(null);

    const ws = new WebSocket(`${WS_BASE}/ws/game/${gameId}`);
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);
    ws.onerror = () => setConnected(false);

    ws.onmessage = (event) => {
      const msg: ServerMessage = JSON.parse(event.data);

      switch (msg.type) {
        case "game_state":
          setGameState(msg);
          setAiThinking(null);
          break;
        case "move_played":
          // We'll get a full game_state after AI turns complete
          break;
        case "ai_thinking":
          setAiThinking(msg.seat);
          break;
        case "game_over":
          setGameOver(msg);
          setAiThinking(null);
          break;
        case "error":
          console.error("Game error:", msg.message);
          break;
      }
    };

    return () => {
      ws.close();
      wsRef.current = null;
    };
  }, [gameId]);

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

  return { gameState, aiThinking, gameOver, connected, playCards, pass, createGroup, deleteGroup };
}
