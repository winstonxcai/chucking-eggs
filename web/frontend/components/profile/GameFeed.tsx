"use client";

import { useState } from "react";

interface GamePlayer {
  player_id: string | null;
  display_name: string;
  is_bot: boolean;
  seat: number;
  finish_pos: number;
  team_result: string;
  elo_before: number | null;
  elo_after: number | null;
}

interface GameDoc {
  _id: string;
  mode: string;
  difficulty: string | null;
  played_at: string;
  players: GamePlayer[];
}

interface GameFeedProps {
  games: GameDoc[];
  username: string;
  viewerUsername?: string | null;
}

function formatDate(iso: string) {
  const d = new Date(iso);
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

const PAGE_SIZE = 10;

export default function GameFeed({ games, username, viewerUsername }: GameFeedProps) {
  const [visible, setVisible] = useState(PAGE_SIZE);

  if (games.length === 0) {
    return <p className="text-sm text-text-secondary">No games yet.</p>;
  }

  return (
    <div className="flex flex-col divide-y divide-border">
      {games.slice(0, visible).map((game) => {
        // Find the seat of this user
        const mySeat = game.players.find(
          (p) => !p.is_bot && p.display_name === username
        );
        if (!mySeat) return null;

        const partnerSeat = game.players.find(
          (p) => p.seat % 2 === mySeat.seat % 2 && p.seat !== mySeat.seat
        );
        const opps = game.players.filter(
          (p) => p.seat % 2 !== mySeat.seat % 2
        );

        const won = mySeat.team_result === "win";
        const eloDelta =
          mySeat.elo_after != null && mySeat.elo_before != null
            ? mySeat.elo_after - mySeat.elo_before
            : null;

        const myName = viewerUsername === username ? "you" : username;
        const partnerName = partnerSeat?.display_name ?? "bot";
        const opp1Name = opps[0]?.display_name ?? "?";
        const opp2Name = opps[1]?.display_name ?? "?";

        return (
          <div key={game._id} className="py-3 flex items-center justify-between gap-3">
            <div className="flex items-center gap-2 min-w-0">
              <span className={`text-xs font-bold w-3 ${won ? "text-team-green" : "text-team-red"}`}>
                {won ? "W" : "L"}
              </span>
              <span className="text-sm text-foreground truncate">
                {myName} + {partnerName}
                <span className="text-text-secondary mx-1.5">vs</span>
                {opp1Name} + {opp2Name}
              </span>
            </div>
            <div className="flex items-center gap-3 shrink-0">
              {eloDelta != null && (
                <span
                  className={`text-xs font-semibold tabular-nums ${
                    eloDelta >= 0 ? "text-team-green" : "text-team-red"
                  }`}
                >
                  {eloDelta >= 0 ? "+" : ""}
                  {eloDelta}
                </span>
              )}
              <span className="text-xs text-text-secondary">{formatDate(game.played_at)}</span>
            </div>
          </div>
        );
      })}
      {visible < games.length && (
        <button
          onClick={() => setVisible((v) => v + PAGE_SIZE)}
          className="pt-3 text-xs text-text-secondary hover:text-foreground transition-colors"
        >
          Show more
        </button>
      )}
    </div>
  );
}
