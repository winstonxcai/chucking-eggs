"use client";

import { useCallback, useEffect, useState } from "react";

const LS_GAME_ID = "gd_game_id";
const LS_TOKEN = "gd_reconnect_token";
const LS_SEAT = "gd_seat";

export function useActiveGame() {
  const [gameId, setGameId] = useState<string | null>(null);

  useEffect(() => {
    setGameId(localStorage.getItem(LS_GAME_ID));
  }, []);

  const saveGame = useCallback((id: string, token: string, seat: number) => {
    localStorage.setItem(LS_GAME_ID, id);
    localStorage.setItem(LS_TOKEN, token);
    localStorage.setItem(LS_SEAT, String(seat));
    setGameId(id);
  }, []);

  const clearGame = useCallback(() => {
    localStorage.removeItem(LS_GAME_ID);
    localStorage.removeItem(LS_TOKEN);
    localStorage.removeItem(LS_SEAT);
    setGameId(null);
  }, []);

  const getResumeParams = useCallback(() => {
    const id = localStorage.getItem(LS_GAME_ID);
    const token = localStorage.getItem(LS_TOKEN);
    const seat = localStorage.getItem(LS_SEAT);
    if (!id || !token) return null;
    return { gameId: id, token, seat: seat ? parseInt(seat, 10) : 0 };
  }, []);

  return { isInGame: !!gameId, gameId, saveGame, clearGame, getResumeParams };
}
