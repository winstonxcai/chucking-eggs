"use client";

import { useCallback, useEffect, useState } from "react";
import { STORAGE_KEYS } from "@/lib/storage-keys";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const LS_PLAYER_ID = STORAGE_KEYS.PLAYER_ID;
const LS_PLAYER_TOKEN = STORAGE_KEYS.PLAYER_TOKEN;
const LS_USERNAME = STORAGE_KEYS.USERNAME;
const LS_ELO = STORAGE_KEYS.ELO;

export interface Player {
  playerId: string;
  playerToken: string | null;
  username: string;
  elo: number;
}

// Safari Private Mode throws SecurityError on localStorage access
function lsGet(key: string): string | null {
  try { return localStorage.getItem(key); } catch { return null; }
}
function lsSet(key: string, val: string): void {
  try { localStorage.setItem(key, val); } catch { /* ignore */ }
}
function storedValue(value: string | null): string | null {
  if (!value || value === "undefined" || value === "null") return null;
  return value;
}
function storedElo(value: string | null): number {
  const parsed = value ? parseInt(value, 10) : NaN;
  return Number.isFinite(parsed) ? parsed : 1200;
}

export function usePlayer() {
  const [player, setPlayer] = useState<Player | null>(null);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    const id = storedValue(lsGet(LS_PLAYER_ID));
    const token = storedValue(lsGet(LS_PLAYER_TOKEN));
    const name = storedValue(lsGet(LS_USERNAME));
    const elo = lsGet(LS_ELO);
    if (id && name) {
      setPlayer({ playerId: id, playerToken: token, username: name, elo: storedElo(elo) });
    }
    setLoaded(true);

    const onEloUpdate = (e: Event) => {
      const newElo = (e as CustomEvent<number>).detail;
      setPlayer((p) => (p ? { ...p, elo: newElo } : p));
    };
    window.addEventListener("ce-elo-update", onEloUpdate);
    return () => window.removeEventListener("ce-elo-update", onEloUpdate);
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
    lsSet(LS_PLAYER_ID, data.player_id);
    if (data.player_token) lsSet(LS_PLAYER_TOKEN, data.player_token);
    lsSet(LS_USERNAME, data.username);
    lsSet(LS_ELO, String(data.elo));
    setPlayer({ playerId: data.player_id, playerToken: data.player_token ?? null, username: data.username, elo: data.elo });
  }, []);

  const updateElo = useCallback((elo: number) => {
    lsSet(LS_ELO, String(elo));
    setPlayer((p) => (p ? { ...p, elo } : p));
    window.dispatchEvent(new CustomEvent("ce-elo-update", { detail: elo }));
  }, []);

  return { player, loaded, claim, updateElo };
}
