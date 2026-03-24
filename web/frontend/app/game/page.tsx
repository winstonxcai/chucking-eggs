"use client";

import { Suspense, useCallback, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import { useGameSocket } from "@/hooks/useGameSocket";
import GameBoard from "@/components/game/GameBoard";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

function GameContent() {
  const searchParams = useSearchParams();
  const [gameId, setGameId] = useState<string | null>(null);
  const difficulty = searchParams.get("difficulty") || "medium";

  useEffect(() => {
    async function createGame() {
      const res = await fetch(`${API_BASE}/api/game/create`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ difficulty }),
      });
      const data = await res.json();
      setGameId(data.game_id);
    }
    createGame();
  }, [difficulty]);

  const { gameState, aiThinking, gameOver, connected, playCards, pass, createGroup, deleteGroup } =
    useGameSocket(gameId);

  const handlePlayAgain = useCallback(() => {
    setGameId(null);
    fetch(`${API_BASE}/api/game/create`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ difficulty }),
    })
      .then((res) => res.json())
      .then((data) => setGameId(data.game_id));
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
