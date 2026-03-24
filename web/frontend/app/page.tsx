"use client";

import { useRouter } from "next/navigation";
import { DIFFICULTY_INFO } from "@/lib/bots";

const difficulties = ["easy", "medium", "hard", "expert"] as const;

export default function Home() {
  const router = useRouter();

  return (
    <div className="min-h-screen bg-background flex flex-col items-center justify-center px-4">
      <div className="max-w-md w-full flex flex-col items-center gap-8">
        {/* Title */}
        <div className="flex flex-col items-center gap-2">
          <h1 className="text-4xl font-bold text-foreground tracking-tight">
            掼蛋
          </h1>
          <h2 className="text-xl font-medium text-text-secondary">Guan Dan</h2>
          <p className="text-sm text-text-secondary text-center mt-2">
            Play the classic Chinese card game against AI opponents
          </p>
        </div>

        {/* Difficulty picker */}
        <div className="grid grid-cols-2 gap-3 w-full">
          {difficulties.map((diff) => {
            const info = DIFFICULTY_INFO[diff];
            return (
              <button
                key={diff}
                className="flex flex-col items-center gap-2 p-5 bg-surface border border-border rounded-xl hover:border-accent hover:shadow-md transition-all cursor-pointer group"
                onClick={() => router.push(`/game?difficulty=${diff}`)}
              >
                <span className="text-3xl">{info.emoji}</span>
                <span className="text-sm font-semibold text-foreground group-hover:text-accent transition-colors">
                  {info.label}
                </span>
                <span className="text-xs text-text-secondary text-center">
                  {info.description}
                </span>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}
