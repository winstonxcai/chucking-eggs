"use client";

import { Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { useGameSocket } from "@/hooks/useGameSocket";
import { usePlayer } from "@/hooks/usePlayer";
import { useActiveGame } from "@/hooks/useActiveGame";
import GameBoard from "@/components/game/GameBoard";
import StatusScreen from "@/components/layout/StatusScreen";
import { STORAGE_KEYS } from "@/lib/storage-keys";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

function GameContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [gameId, setGameId] = useState<string | null>(null);
  const [reconnectToken, setReconnectToken] = useState<string | null>(null);
  const [seat, setSeat] = useState<number>(0);
  const [createError, setCreateError] = useState<string | null>(null);
  const [playingAgain, setPlayingAgain] = useState(false);

  const difficulty = searchParams.get("difficulty") || "easy";

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

  const { gameState, aiThinking, gameOver, connected, connectionStatus, closeReason, playCards, pass, createGroup, deleteGroup, latestError, autoPlayed, rematch, forfeit, sendAbort, hasPlayedFirstMove } =
    useGameSocket(gameId, reconnectToken, seat);

  const { updateElo } = usePlayer();
  const { saveGame, clearGame } = useActiveGame();

  const isMultiplayer = gameState ? (gameState.mode ?? "solo") !== "solo" : false;

  // Save multiplayer game to localStorage for active-game banner
  useEffect(() => {
    if (gameId && reconnectToken && isMultiplayer) {
      saveGame(gameId, reconnectToken, seat);
    }
  }, [gameId, reconnectToken, seat, isMultiplayer, saveGame]);

  // Clear active game on game over or close
  useEffect(() => {
    if (gameOver || closeReason) {
      clearGame();
    }
  }, [gameOver, closeReason, clearGame]);

  // Reset playingAgain once the socket clears gameOver (new game transition complete)
  useEffect(() => {
    if (!gameOver) setPlayingAgain(false);
  }, [gameOver]);

  useEffect(() => {
    if (!gameOver?.elo_changes) return;
    const change = gameOver.elo_changes["0"];
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

  const handlePlayAgain = useCallback(async () => {
    setPlayingAgain(true);
    try {
      const res = await fetch(`${API_BASE}/api/game/create`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ difficulty }),
      });
      if (!res.ok) throw new Error(`Server error ${res.status}`);
      const data = await res.json();
      // Atomic swap: old gameId → new, no null intermediate state
      setGameId(data.game_id);
      setReconnectToken(data.reconnect_token);
      setSeat(0);
      sessionStorage.setItem(STORAGE_KEYS.GAME_ID, data.game_id);
      sessionStorage.setItem(STORAGE_KEYS.RECONNECT_TOKEN, data.reconnect_token);
      sessionStorage.setItem(STORAGE_KEYS.SEAT, "0");
    } catch {
      router.push("/");
    }
  }, [difficulty, router]);

  // Handle forfeit broadcast from another player
  useEffect(() => {
    if (!forfeit) return;
    clearGame();
    const t = setTimeout(() => router.push("/"), 3000);
    return () => clearTimeout(t);
  }, [forfeit, clearGame, router]);

  const handleForfeit = useCallback(async () => {
    if (!gameId) return;
    const playerId = (() => { try { return localStorage.getItem("ce_player_id"); } catch { return null; } })();
    if (!playerId) return;
    try {
      await fetch(`${API_BASE}/api/room/${gameId}/forfeit`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ player_id: playerId }),
      });
    } catch { /* best effort */ }
    clearGame();
    sessionStorage.removeItem(STORAGE_KEYS.GAME_ID);
    sessionStorage.removeItem(STORAGE_KEYS.RECONNECT_TOKEN);
    sessionStorage.removeItem(STORAGE_KEYS.SEAT);
    router.push("/");
  }, [gameId, clearGame, router]);

  const handleAbort = useCallback(() => {
    sendAbort();
    clearGame();
    sessionStorage.removeItem(STORAGE_KEYS.GAME_ID);
    sessionStorage.removeItem(STORAGE_KEYS.RECONNECT_TOKEN);
    sessionStorage.removeItem(STORAGE_KEYS.SEAT);
    router.push("/");
  }, [sendAbort, clearGame, router]);

  if (createError) {
    return (
      <StatusScreen
        variant="error"
        title="Connection failed"
        message="Could not reach the game server. Check your internet connection and try again."
        action={{ label: "Try again", onClick: () => window.location.reload() }}
        secondaryAction={{ label: "Home", href: "/" }}
      />
    );
  }

  if (forfeit) {
    return (
      <StatusScreen
        variant="warning"
        title="Game ended"
        message={`${forfeit.forfeiter_name} forfeited the game. Redirecting you home...`}
      />
    );
  }

  if (closeReason) {
    return (
      <StatusScreen
        variant="error"
        title="Game disconnected"
        message={closeReason}
        action={{ label: "Back to home", href: "/" }}
      />
    );
  }

  if (!gameId || !gameState || playingAgain) {
    return (
      <StatusScreen
        variant="loading"
        title={!gameId || playingAgain ? "Creating game..." : "Connecting..."}
        message="Setting up your table"
      />
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
      autoPlayed={autoPlayed}
      onForfeit={isMultiplayer ? handleForfeit : undefined}
      onAbort={isMultiplayer && !hasPlayedFirstMove ? handleAbort : undefined}
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
