"use client";

import { Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { useGameSocket } from "@/hooks/useGameSocket";
import { usePlayer } from "@/hooks/usePlayer";
import GameBoard from "@/components/game/GameBoard";
import { STORAGE_KEYS } from "@/lib/storage-keys";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

function GameContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [gameId, setGameId] = useState<string | null>(null);
  const [reconnectToken, setReconnectToken] = useState<string | null>(null);
  const [seat, setSeat] = useState<number>(0);
  const [createError, setCreateError] = useState<string | null>(null);

  const difficulty = searchParams.get("difficulty") || "medium";

  useEffect(() => {
    // URL params take priority (from lobby navigation)
    const urlGameId = searchParams.get("game_id");
    const urlToken = searchParams.get("token");
    const urlSeat = searchParams.get("seat");

    if (urlGameId && urlToken) {
      const seatNum = urlSeat ? parseInt(urlSeat, 10) : 0;
      setGameId(urlGameId);
      setReconnectToken(urlToken);
      setSeat(seatNum);
      sessionStorage.setItem(STORAGE_KEYS.GAME_ID, urlGameId);
      sessionStorage.setItem(STORAGE_KEYS.RECONNECT_TOKEN, urlToken);
      sessionStorage.setItem(STORAGE_KEYS.SEAT, String(seatNum));
      return;
    }

    // Try to restore from sessionStorage
    const savedId = sessionStorage.getItem(STORAGE_KEYS.GAME_ID);
    const savedToken = sessionStorage.getItem(STORAGE_KEYS.RECONNECT_TOKEN);
    const savedSeat = sessionStorage.getItem(STORAGE_KEYS.SEAT);
    if (savedId && savedToken) {
      setGameId(savedId);
      setReconnectToken(savedToken);
      setSeat(savedSeat ? parseInt(savedSeat, 10) : 0);
      return;
    }

    // Create a new solo game
    async function createGame() {
      try {
        const res = await fetch(`${API_BASE}/api/game/create`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ difficulty }),
        });
        if (!res.ok) throw new Error(`Server error ${res.status}`);
        const data = await res.json();
        setGameId(data.game_id);
        setReconnectToken(data.reconnect_token);
        setSeat(0);
        sessionStorage.setItem(STORAGE_KEYS.GAME_ID, data.game_id);
        sessionStorage.setItem(STORAGE_KEYS.RECONNECT_TOKEN, data.reconnect_token);
        sessionStorage.setItem(STORAGE_KEYS.SEAT, "0");
      } catch (err) {
        console.error("Failed to create game:", err);
        setCreateError("Failed to connect to server. Please try again.");
      }
    }
    createGame();
  }, [difficulty, searchParams]);

  const { gameState, aiThinking, gameOver, connected, connectionStatus, closeReason, playCards, pass, createGroup, deleteGroup, latestError, rematch } =
    useGameSocket(gameId, reconnectToken, seat);

  const { updateElo } = usePlayer();
  useEffect(() => {
    if (!gameOver?.elo_changes) return;
    const change = gameOver.elo_changes[String(seat)];
    if (change?.after != null) updateElo(change.after);
  }, [gameOver, seat, updateElo]);

  // When another player initiates rematch, navigate to join the new room
  const rematchHandled = useRef(false);
  useEffect(() => {
    if (!rematch || rematchHandled.current) return;
    rematchHandled.current = true;
    if (rematch.room_code) {
      router.push(`/join?code=${rematch.room_code}`);
    }
  }, [rematch, router]);

  const isMultiplayer = gameState ? gameState.players.filter((p) => p.is_human).length > 1 : false;

  // Solo games: clear session on unmount so navigating away starts fresh
  const isMultiplayerRef = useRef(false);
  isMultiplayerRef.current = isMultiplayer;
  useEffect(() => {
    return () => {
      if (!isMultiplayerRef.current) {
        sessionStorage.removeItem(STORAGE_KEYS.GAME_ID);
        sessionStorage.removeItem(STORAGE_KEYS.RECONNECT_TOKEN);
        sessionStorage.removeItem(STORAGE_KEYS.SEAT);
      }
    };
  }, []);

  const handleRematch = useCallback(async () => {
    if (!gameId) return;
    try {
      await fetch(`${API_BASE}/api/room/${gameId}/rematch`, { method: "POST" });
      // The broadcast handler above will navigate us to the new room
    } catch {
      // Fallback: go home
      router.push("/");
    }
  }, [gameId, router]);

  const handlePlayAgain = useCallback(() => {
    sessionStorage.removeItem(STORAGE_KEYS.GAME_ID);
    sessionStorage.removeItem(STORAGE_KEYS.RECONNECT_TOKEN);
    sessionStorage.removeItem(STORAGE_KEYS.SEAT);
    setGameId(null);
    setReconnectToken(null);
    setSeat(0);
    fetch(`${API_BASE}/api/game/create`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ difficulty }),
    })
      .then((res) => res.json())
      .then((data) => {
        setGameId(data.game_id);
        setReconnectToken(data.reconnect_token);
        setSeat(0);
        sessionStorage.setItem(STORAGE_KEYS.GAME_ID, data.game_id);
        sessionStorage.setItem(STORAGE_KEYS.RECONNECT_TOKEN, data.reconnect_token);
        sessionStorage.setItem(STORAGE_KEYS.SEAT, "0");
      });
  }, [difficulty]);

  if (createError) {
    return (
      <div className="h-full flex items-center justify-center bg-background">
        <div className="flex flex-col items-center gap-4 text-center max-w-sm px-4">
          <p className="text-text-secondary">{createError}</p>
          <a href="/" className="text-sm text-accent underline">Back to home</a>
        </div>
      </div>
    );
  }

  if (closeReason) {
    return (
      <div className="h-full flex items-center justify-center bg-background">
        <div className="flex flex-col items-center gap-4 text-center max-w-sm px-4">
          <p className="text-text-secondary">{closeReason}</p>
          <a href="/" className="text-sm text-accent underline">Back to home</a>
        </div>
      </div>
    );
  }

  if (!gameId || !gameState) {
    return (
      <div className="h-full flex items-center justify-center bg-background">
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
      onRematch={handleRematch}
      isMultiplayer={isMultiplayer}
      onCreateGroup={createGroup}
      onDeleteGroup={deleteGroup}
      latestError={latestError}
    />
  );
}

export default function GamePage() {
  return (
    <Suspense
      fallback={
        <div className="h-full flex items-center justify-center bg-background">
          <div className="w-8 h-8 border-2 border-accent border-t-transparent rounded-full animate-spin" />
        </div>
      }
    >
      <GameContent />
    </Suspense>
  );
}
