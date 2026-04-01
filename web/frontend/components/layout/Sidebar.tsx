"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import type { Player } from "@/hooks/usePlayer";
import { useActiveGame } from "@/hooks/useActiveGame";

interface SidebarProps {
  player: Player | null;
}

const navItems = [
  { label: "Play", href: "/" },
  { label: "Leaderboard", href: "/leaderboard" },
  { label: "Attributions", href: "/attributions" },
  { label: "Settings", href: "/settings" },
];

export default function Sidebar({ player }: SidebarProps) {
  const pathname = usePathname();
  const { isInGame } = useActiveGame();

  return (
    <aside
      className="w-[200px] shrink-0 flex flex-col border-r border-border bg-background"
      style={{ minHeight: "100dvh" }}
    >
      {/* Brand */}
      <div className="px-5 pt-6 pb-4">
        <span className="text-base font-bold text-foreground tracking-tight">Chucking Eggs</span>
      </div>

      {/* Nav */}
      <nav className="flex flex-col gap-0.5 px-3 flex-1">
        {navItems.map(({ label, href }) => {
          const resolvedHref = label === "Play" && isInGame ? "/game" : href;
          const resolvedLabel = label === "Play" && isInGame ? "Resume" : label;
          const isActive = pathname === resolvedHref || (label === "Play" && pathname === "/game");
          return (
            <Link
              key={label}
              href={resolvedHref}
              className={`px-3 py-2 rounded-lg text-sm font-medium transition-colors border-l-2 ${
                isActive
                  ? "border-accent text-accent bg-accent/5"
                  : "border-transparent text-foreground hover:bg-border/50 hover:text-foreground"
              }`}
            >
              {resolvedLabel}
            </Link>
          );
        })}
      </nav>

      {/* Player footer */}
      {player && (
        <div className="px-4 py-4 border-t border-border">
          <Link
            href={`/profile/${player.username}`}
            className="flex flex-col gap-0.5 hover:opacity-70 transition-opacity"
          >
            <span className="text-sm font-semibold text-foreground truncate">{player.username}</span>
            <span className="text-xs text-text-secondary">Elo {player.elo}</span>
          </Link>
        </div>
      )}
    </aside>
  );
}
