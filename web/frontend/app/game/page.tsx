"use client";

import { Suspense, useCallback, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import { useGameSocket } from "@/hooks/useGameSocket";
import GameBoard from "@/components/game/GameBoard";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

function GameContent() {
  const searchParams = useSearchParams();
  const [gameId, setGameId] = useState<string | null>(null);
  const [reconnectToken, setReconnectToken] = useState<string | null>(null);
  const difficulty = searchParams.get("difficulty") || "medium";

  useEffect(() => {
    // Try to restore from sessionStorage first
    const savedId = sessionStorage.getItem("gd_game_id");
    const savedToken = sessionStorage.getItem("gd_reconnect_token");
    if (savedId && savedToken) {
      setGameId(savedId);
      setReconnectToken(savedToken);
      return;
    }

    async function createGame() {
      const res = await fetch(`${API_BASE}/api/game/create`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ difficulty }),
      });
      const data = await res.json();
      setGameId(data.game_id);
      setReconnectToken(data.reconnect_token);
      sessionStorage.setItem("gd_game_id", data.game_id);
      sessionStorage.setItem("gd_reconnect_token", data.reconnect_token);
    }
    createGame();
  }, [difficulty]);

  const { gameState, aiThinking, gameOver, connected, connectionStatus, playCards, pass, createGroup, deleteGroup } =
    useGameSocket(gameId, reconnectToken);

  const handlePlayAgain = useCallback(() => {
    // Clear stored session
    sessionStorage.removeItem("gd_game_id");
    sessionStorage.removeItem("gd_reconnect_token");
    setGameId(null);
    setReconnectToken(null);
    fetch(`${API_BASE}/api/game/create`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ difficulty }),
    })
      .then((res) => res.json())
      .then((data) => {
        setGameId(data.game_id);
        setReconnectToken(data.reconnect_token);
        sessionStorage.setItem("gd_game_id", data.game_id);
        sessionStorage.setItem("gd_reconnect_token", data.reconnect_token);
      });
  }, [difficulty]);

  if (!gameId || !gameState) {
    return (
      <div className="h-screen flex items-center justify-center bg-background">
        <div className="flex flex-col items-center gap-3">
          <div className="w-8 h-8 border-2 border-accent border-t-transparent rounded-full animate-spin" />
          <span className="text-sm text-text-secondary">
            {!gameId ? "Creating game..." : "Connecting..."}
          </span>
        </div>
      </div>
    );
  }

  return (
    <GameBoard
      gameState={gameState}
      aiThinking={aiThinking}
      gameOver={gameOver}
      connectionStatus={connectionStatus}
      onPlayCards={playCards}
      onPass={pass}
      onPlayAgain={handlePlayAgain}
      onCreateGroup={createGroup}
      onDeleteGroup={deleteGroup}
    />
  );
}

export default function GamePage() {
  return (
    <Suspense
      fallback={
        <div className="h-screen flex items-center justify-center bg-background">
          <div className="w-8 h-8 border-2 border-accent border-t-transparent rounded-full animate-spin" />
        </div>
      }
    >
      <GameContent />
    </Suspense>
  );
}
