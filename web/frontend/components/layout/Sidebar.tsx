"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState, useEffect } from "react";
import type { Player } from "@/hooks/usePlayer";
import { useActiveGame } from "@/hooks/useActiveGame";
import { STORAGE_KEYS } from "@/lib/storage-keys";
import { Gamepad2, Trophy, BookOpen, Settings, HelpCircle, ChevronLeft, ChevronRight } from "lucide-react";

interface SidebarProps {
  player: Player | null;
}

const navItems = [
  { label: "Play", href: "/", icon: Gamepad2 },
  { label: "Leaderboard", href: "/leaderboard", icon: Trophy },
  { label: "Rules", href: "/rules", icon: HelpCircle },
  { label: "Attributions", href: "/attributions", icon: BookOpen },
  { label: "Settings", href: "/settings", icon: Settings },
];

export default function Sidebar({ player }: SidebarProps) {
  const pathname = usePathname();
  const { isInGame } = useActiveGame();
  const [collapsed, setCollapsed] = useState(false);
  const [currentElo, setCurrentElo] = useState<number | null>(null);

  useEffect(() => {
    const stored = localStorage.getItem("sidebar-collapsed");
    if (stored !== null) setCollapsed(stored === "true");
    // Seed ELO from localStorage in case prop arrives late
    const storedElo = localStorage.getItem(STORAGE_KEYS.ELO);
    if (storedElo) setCurrentElo(parseInt(storedElo, 10));
  }, []);

  // Keep in sync with prop (covers login/logout transitions)
  useEffect(() => {
    if (player?.elo != null) setCurrentElo(player.elo);
  }, [player?.elo]);

  // Update immediately when a game ends, without waiting for prop re-render
  useEffect(() => {
    const handler = (e: Event) => setCurrentElo((e as CustomEvent<number>).detail);
    window.addEventListener("ce-elo-update", handler);
    return () => window.removeEventListener("ce-elo-update", handler);
  }, []);

  function toggleCollapsed() {
    const next = !collapsed;
    setCollapsed(next);
    localStorage.setItem("sidebar-collapsed", String(next));
  }

  const expanded = !collapsed;

  return (
    <aside
      className={`shrink-0 flex flex-col border-r border-border bg-background transition-[width] duration-200 ease-out ${
        expanded ? "w-12 lg:w-[200px]" : "w-12"
      }`}
      style={{ height: "100%" }}
    >
      {/* Brand header */}
      <div className={`flex overflow-hidden ${expanded ? "items-center justify-center lg:justify-start gap-2 px-2 pt-5 pb-3 lg:px-3 lg:pt-8 lg:pb-5" : "flex-col items-center pt-3 pb-2 gap-2.5"}`}>
        {/* Expand button: above egg when collapsed, desktop only */}
        {!expanded && (
          <button
            onClick={toggleCollapsed}
            className="hidden lg:flex items-center justify-center p-0.5 rounded-md text-text-secondary hover:bg-accent/8 hover:text-accent transition-all"
            title="Expand sidebar"
          >
            <ChevronRight size={14} strokeWidth={2} />
          </button>
        )}
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" className="w-6 h-6 shrink-0">
          <path d="M16 3 C22 3 27 10 27 18 C27 24.6 22.1 29 16 29 C9.9 29 5 24.6 5 18 C5 10 10 3 16 3Z" fill="#D97757"/>
          <ellipse cx="12" cy="11" rx="2.2" ry="1.3" fill="rgba(255,255,255,0.22)" transform="rotate(-30 12 11)"/>
        </svg>
        {expanded && (
          <span className="hidden lg:inline text-sm font-bold text-foreground tracking-tight whitespace-nowrap flex-1">
            Chucking Eggs
          </span>
        )}
        {/* Collapse button: right-aligned in header when expanded, desktop only */}
        {expanded && (
          <button
            onClick={toggleCollapsed}
            className="hidden lg:flex items-center justify-center p-0.5 rounded-md text-text-secondary hover:bg-accent/8 hover:text-accent transition-all ml-auto"
            title="Collapse sidebar"
          >
            <ChevronLeft size={14} strokeWidth={2} />
          </button>
        )}
      </div>

      {/* Nav */}
      <nav className="flex flex-col gap-0.5 px-1 lg:px-2 flex-1">
        {navItems.map(({ label, href, icon: Icon }) => {
          const resolvedHref = label === "Play" && isInGame ? "/game" : href;
          const resolvedLabel = label === "Play" && isInGame ? "Resume" : label;
          const isActive =
            pathname === resolvedHref || (label === "Play" && pathname === "/game");
          const className = `flex items-center justify-center gap-2.5 px-2 py-3 lg:py-2 rounded-lg text-sm font-medium transition-all duration-150 ease-out border-l-2 ${
            expanded ? "lg:justify-start lg:px-3" : ""
          } ${
            isActive
              ? "border-accent text-accent bg-accent/5"
              : "border-transparent text-foreground hover:bg-accent/8 hover:text-accent"
          }`;
          return (
            <Link key={label} href={resolvedHref} className={className}>
              <Icon size={15} strokeWidth={isActive ? 2.5 : 2} />
              {expanded && <span className="hidden lg:inline">{resolvedLabel}</span>}
            </Link>
          );
        })}

      </nav>

      {/* Player footer — avatar + username/ELO when expanded */}
      {player && (
        <div className="px-2 py-3 pb-[max(0.75rem,env(safe-area-inset-bottom))] border-t border-border">
          <Link
            href={`/profile/${player.username}`}
            className={`flex items-center gap-2 rounded-lg transition-opacity ${
              pathname.startsWith("/profile") ? "opacity-100" : "hover:opacity-70"
            } ${expanded ? "lg:px-1" : "justify-center"}`}
          >
            <div className={`w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold shrink-0 ${
              pathname.startsWith("/profile") ? "bg-accent text-white" : "bg-accent/10 text-accent"
            }`}>
              {player.username[0].toUpperCase()}
            </div>
            {expanded && (
              <div className="hidden lg:flex flex-col min-w-0">
                <span className="text-xs font-semibold text-foreground truncate">{player.username}</span>
                <span className="text-[11px] text-text-secondary">{currentElo ?? player.elo} ELO</span>
              </div>
            )}
          </Link>
        </div>
      )}
    </aside>
  );
}
