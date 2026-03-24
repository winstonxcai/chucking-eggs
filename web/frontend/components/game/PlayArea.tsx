"use client";

import type { CurrentTrick, PlayerDTO } from "@/lib/types";
import CardComponent from "./CardComponent";

interface PlayAreaProps {
  trick: CurrentTrick | null;
  isLeading: boolean;
  isMyTurn: boolean;
  players: PlayerDTO[];
}

function playerColor(seat: number, players: PlayerDTO[]): string {
  const p = players.find((pl) => pl.seat === seat);
  if (!p) return "#8C8478";
  if (p.is_human) return "#D97757";
  if (p.is_teammate) return "#3D8C6F";
  return "#C75D4A";
}

export default function PlayArea({ trick, isLeading, isMyTurn, players }: PlayAreaProps) {
  return (
    <div className="flex flex-col gap-3 px-7 py-5 bg-surface rounded-2xl border border-border shadow-[0_2px_8px_rgba(0,0,0,0.04)] min-w-[320px] min-h-[180px]">
      <span className="text-[13px] font-semibold text-text-secondary tracking-wider uppercase">
        Current Trick
      </span>

      {trick ? (
        <div className="flex flex-col gap-2">
          {trick.plays.map((play, i) => (
            <div key={i} className="flex items-center gap-3">
              <span
                className="text-[13px] font-medium w-14 shrink-0"
                style={{ color: playerColor(play.seat, players) }}
              >
                {play.player_name}
              </span>
              {play.is_pass ? (
                <span className="text-[13px] text-text-secondary">Pass</span>
              ) : (
                <>
                  <div className="flex gap-1">
                    {play.combo.cards.map((card) => (
                      <CardComponent key={card.id} card={card} size="sm" />
                    ))}
                  </div>
                  <span className="text-xs text-text-secondary">
                    {play.combo.type_name}
                  </span>
                </>
              )}
            </div>
          ))}
        </div>
      ) : (
        <div className="flex-1 flex items-center justify-center">
          <span className="text-sm text-text-secondary">
            {isLeading ? "Lead a combo" : "Waiting..."}
          </span>
        </div>
      )}

      {isMyTurn && (
        <div className="flex items-center gap-1.5 pt-1">
          <div className="w-1.5 h-1.5 rounded-full bg-accent" />
          <span className="text-[13px] font-medium text-accent">
            {isLeading ? "Your turn to lead" : "Your turn to play"}
          </span>
        </div>
      )}
    </div>
  );
}
