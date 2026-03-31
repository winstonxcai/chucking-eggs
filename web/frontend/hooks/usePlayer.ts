"use client";

import { useCallback, useEffect, useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const LS_PLAYER_ID = "ce_player_id";
const LS_USERNAME = "ce_username";
const LS_ELO = "ce_elo";

export interface Player {
  playerId: string;
  username: string;
  elo: number;
}

export function usePlayer() {
  const [player, setPlayer] = useState<Player | null>(null);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    const id = localStorage.getItem(LS_PLAYER_ID);
    const name = localStorage.getItem(LS_USERNAME);
    const elo = localStorage.getItem(LS_ELO);
    if (id && name) {
      setPlayer({ playerId: id, username: name, elo: elo ? parseInt(elo, 10) : 1200 });
    }
    setLoaded(true);
  }, []);

  const claim = useCallback(async (username: string, email?: string): Promise<void> => {
    const res = await fetch(`${API_BASE}/api/auth/claim`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, email: email || null }),
    });
    if (!res.ok) {
      const data = await res.json();
      throw new Error(data.detail || "Failed to claim username");
    }
    const data = await res.json();
    localStorage.setItem(LS_PLAYER_ID, data.player_id);
    localStorage.setItem(LS_USERNAME, data.username);
    localStorage.setItem(LS_ELO, String(data.elo));
    setPlayer({ playerId: data.player_id, username: data.username, elo: data.elo });
  }, []);

  const updateElo = useCallback((elo: number) => {
    localStorage.setItem(LS_ELO, String(elo));
    setPlayer((p) => (p ? { ...p, elo } : p));
  }, []);

  return { player, loaded, claim, updateElo };
}
