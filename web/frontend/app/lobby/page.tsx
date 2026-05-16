"use client";

import { Suspense, useCallback, useEffect, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import type { RoomStatus, RoomMode } from "@/lib/types";
import { OUR_BOTS, type BotInfo } from "@/lib/bots";
import { STORAGE_KEYS } from "@/lib/storage-keys";
import StatusScreen from "@/components/layout/StatusScreen";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const SEAT_LABELS = ["South (You)", "West", "North", "East"] as const;

const MODE_LABEL: Record<RoomMode, string> = {
  solo: "Solo",
  duo: "2-Player",
  quad: "4-Player",
};


function CopyButton({ text, label = "Copy" }: { text: string; label?: string }) {
  const [copied, setCopied] = useState(false);
  const copy = useCallback(async () => {
    await navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }, [text]);
  return (
    <button
      onClick={copy}
      className="text-xs text-accent hover:underline transition-colors"
    >
      {copied ? "Copied!" : label}
    </button>
  );
}

function LobbyContent() {
  const router = useRouter();
  const searchParams = useSearchParams();

  const gameId = searchParams.get("game_id");
  const seat = parseInt(searchParams.get("seat") || "0", 10);
  const token = searchParams.get("token");

  const [status, setStatus] = useState<RoomStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [difficulty, setDifficulty] = useState("greedy");
  const [botInfo, setBotInfo] = useState<Record<string, BotInfo>>({});

  useEffect(() => {
    fetch(`${API_BASE}/api/bots`).then(r => r.json()).then(setBotInfo);
  }, []);

  const competitionBots = Object.keys(botInfo).filter(
    k => !(OUR_BOTS as readonly string[]).includes(k)
  );
  const [countdown, setCountdown] = useState<number | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const countdownRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Persist session to survive page refresh
  useEffect(() => {
    if (gameId && token) {
      try {
        sessionStorage.setItem(STORAGE_KEYS.GAME_ID, gameId);
        sessionStorage.setItem(STORAGE_KEYS.RECONNECT_TOKEN, token);
        sessionStorage.setItem(STORAGE_KEYS.SEAT, String(seat));
      } catch { /* Safari private mode */ }
    }
  }, [gameId, seat, token]);

  const fetchStatus = useCallback(async () => {
    if (!gameId) return;
    try {
      const res = await fetch(`${API_BASE}/api/room/${gameId}/status`);
      if (!res.ok) {
        if (pollRef.current) clearInterval(pollRef.current);
        setError("This room has been closed or is no longer available.");
        return;
      }
      const data: RoomStatus = await res.json();
      setStatus(data);

      // Navigate to game once all humans are connected and game has started
      if (data.started) {
        if (pollRef.current) clearInterval(pollRef.current);
        router.push(`/game?game_id=${gameId}&seat=${seat}&token=${encodeURIComponent(token || "")}`);
      }
    } catch {
      setError("Could not reach the server. Check your connection and try again.");
    }
  }, [gameId, seat, token, router]);

  useEffect(() => {
    if (!gameId || !token) {
      setError("Invalid lobby link. The game ID or session token is missing.");
      return;
    }
    fetchStatus();
    pollRef.current = setInterval(fetchStatus, 2000);
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [gameId, token, fetchStatus]);

  // Start countdown when lobby_expires_at is available
  useEffect(() => {
    if (!status?.lobby_expires_at) {
      if (countdownRef.current) clearInterval(countdownRef.current);
      setCountdown(null);
      return;
    }
    const tick = () => {
      const remaining = Math.ceil(status.lobby_expires_at! - Date.now() / 1000);
      if (remaining <= 0) {
        if (countdownRef.current) clearInterval(countdownRef.current);
        setCountdown(0);
        sessionStorage.removeItem(STORAGE_KEYS.GAME_ID);
        sessionStorage.removeItem(STORAGE_KEYS.RECONNECT_TOKEN);
        sessionStorage.removeItem(STORAGE_KEYS.SEAT);
        router.push("/");
      } else {
        setCountdown(remaining);
      }
    };
    tick();
    countdownRef.current = setInterval(tick, 1000);
    return () => {
      if (countdownRef.current) clearInterval(countdownRef.current);
    };
  }, [status?.lobby_expires_at, router]);

  const changeDifficulty = useCallback(async (diff: string) => {
    setDifficulty(diff);
    try {
      await fetch(`${API_BASE}/api/room/${gameId}/set_difficulty`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ difficulty: diff }),
      });
    } catch { /* non-critical */ }
  }, [gameId]);

  if (error) {
    return (
      <StatusScreen
        variant="error"
        title="Room unavailable"
        message={error}
        action={{ label: "Back to home", href: "/" }}
        secondaryAction={{ label: "Try again", onClick: () => window.location.reload() }}
      />
    );
  }

  const humanSeats = status?.seats.filter((s) => s.is_human) ?? [];
  const connectedCount = humanSeats.filter((s) => s.connected).length;
  const totalHumans = humanSeats.length;
  const isHost = seat === 0;
  const isDuo = status?.mode === "duo";

  return (
    <div className="h-full bg-background flex flex-col items-center justify-center px-4">
      <div className="max-w-sm w-full flex flex-col gap-6">
        {/* Header */}
        <div className="flex flex-col items-center gap-1">
          <h1 className="text-2xl font-bold text-foreground">Waiting for players</h1>
          <span className="text-sm text-text-secondary">
            {status ? `${MODE_LABEL[status.mode]} · ${connectedCount}/${totalHumans} connected` : "Connecting…"}
          </span>
        </div>

        {/* Duo difficulty picker (host only) */}
        {isDuo && isHost && (
          <div className="bg-surface border border-border rounded-xl p-4 flex flex-col gap-2">
            <span className="text-xs font-semibold text-text-secondary tracking-widest uppercase">Bot Difficulty</span>
            <select
              value={difficulty}
              onChange={(e) => changeDifficulty(e.target.value)}
              className="w-full px-3 py-2 bg-background border border-border rounded-lg text-sm text-foreground focus:outline-none focus:border-accent transition-colors"
            >
              <optgroup label="Our Bots">
                {OUR_BOTS.map((diff) => {
                  const info = botInfo[diff];
                  if (!info) return null;
                  return (
                    <option key={diff} value={diff}>
                      {info.label} — {info.description}
                    </option>
                  );
                })}
              </optgroup>
              <optgroup label="Competition Bots">
                {competitionBots.map((diff) => {
                  const info = botInfo[diff];
                  if (!info) return null;
                  return (
                    <option key={diff} value={diff}>
                      {info.label} — {info.description}
                    </option>
                  );
                })}
              </optgroup>
            </select>
          </div>
        )}

        {/* Room code */}
        <div className="bg-surface border border-border rounded-xl p-5 flex flex-col items-center gap-3">
          <span className="text-xs font-semibold text-text-secondary tracking-widest uppercase">Room Code</span>
          {status?.room_code ? (
            <>
              <span data-testid="room-code" className="text-4xl font-bold tracking-widest text-foreground font-mono">
                {status.room_code}
              </span>
              <div className="flex items-center gap-3 text-xs text-text-secondary">
                <CopyButton text={status.room_code} label="Copy Code" />
                <span className="text-border">|</span>
                <CopyButton text={`${window.location.origin}/join?code=${status.room_code}`} label="Copy Link" />
                {typeof navigator !== "undefined" && navigator.share && (
                  <>
                    <span className="text-border">|</span>
                    <button
                      onClick={() =>
                        navigator.share({
                          title: "Join my Guan Dan game",
                          url: `${window.location.origin}/join?code=${status.room_code}`,
                        }).catch(() => {})
                      }
                      className="text-accent hover:underline transition-colors"
                    >
                      Share
                    </button>
                  </>
                )}
              </div>
            </>
          ) : (
            <div className="h-10 w-40 bg-border/40 rounded-lg animate-pulse" />
          )}
        </div>

        {/* Lobby expiry countdown */}
        {countdown !== null && (
          <div className="flex items-center justify-center gap-1.5 text-xs text-text-secondary">
            <span>Room closes in</span>
            <span className={`font-mono font-semibold ${countdown <= 60 ? "text-amber-400" : "text-text-secondary"}`}>
              {Math.floor(countdown / 60)}:{String(countdown % 60).padStart(2, "0")}
            </span>
          </div>
        )}

        {/* Seat list */}
        <div className="bg-surface border border-border rounded-xl overflow-hidden">
          {status ? status.seats.map((s) => (
            <div
              key={s.seat}
              className="flex items-center justify-between px-4 py-3 border-b border-border last:border-0"
            >
              <div className="flex items-center gap-3">
                <div
                  className={`w-2 h-2 rounded-full ${
                    s.connected ? "bg-green-500" : s.is_human ? "bg-amber-400" : "bg-border"
                  }`}
                />
                <div className="flex flex-col">
                  <span className="text-sm font-medium text-foreground">
                    {s.seat === seat ? "You" : s.name}
                  </span>
                  <span className="text-xs text-text-secondary">
                    {SEAT_LABELS[s.seat]}
                  </span>
                </div>
              </div>
              <span className="text-xs text-text-secondary">
                {s.is_human
                  ? s.connected
                    ? "Connected"
                    : "Waiting..."
                  : "AI"}
              </span>
            </div>
          )) : [0, 1, 2, 3].map((i) => (
            <div key={i} className="flex items-center justify-between px-4 py-3 border-b border-border last:border-0">
              <div className="flex items-center gap-3">
                <div className="w-2 h-2 rounded-full bg-border animate-pulse" />
                <div className="flex flex-col gap-1">
                  <div className="h-3.5 w-20 bg-border/40 rounded animate-pulse" />
                  <div className="h-3 w-14 bg-border/30 rounded animate-pulse" />
                </div>
              </div>
              <div className="h-3 w-14 bg-border/30 rounded animate-pulse" />
            </div>
          ))}
        </div>

        {/* Status indicator */}
        <div className="flex items-center justify-center gap-2">
          <div className="w-1.5 h-1.5 rounded-full bg-accent animate-pulse" />
          <span className="text-xs text-text-secondary">
            Waiting for all players to join…
          </span>
        </div>

        <button
          className="text-xs text-text-secondary hover:text-foreground transition-colors"
          onClick={async () => {
            if (gameId) {
              try { await fetch(`${API_BASE}/api/room/${gameId}/leave`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ seat }) }); } catch {}
            }
            sessionStorage.removeItem(STORAGE_KEYS.GAME_ID);
            sessionStorage.removeItem(STORAGE_KEYS.RECONNECT_TOKEN);
            sessionStorage.removeItem(STORAGE_KEYS.SEAT);
            router.push("/");
          }}
        >
          Cancel and go home
        </button>
      </div>
    </div>
  );
}

export default function LobbyPage() {
  return (
    <Suspense
      fallback={
        <div className="h-full flex items-center justify-center bg-background">
          <div className="w-8 h-8 border-2 border-accent border-t-transparent rounded-full animate-spin" />
        </div>
      }
    >
      <LobbyContent />
    </Suspense>
  );
}
