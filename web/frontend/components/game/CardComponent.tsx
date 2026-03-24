"use client";

import type { CardDTO } from "@/lib/types";

interface CardProps {
  card: CardDTO;
  selected?: boolean;
  onClick?: () => void;
  size?: "sm" | "md";
}

const suitColor: Record<number, string> = {
  0: "#1A1612", // spade
  1: "#C75D4A", // heart
  2: "#C75D4A", // diamond
  3: "#1A1612", // club
};

export default function CardComponent({ card, selected, onClick, size = "md" }: CardProps) {
  const color = suitColor[card.suit] ?? "#1A1612";
  const isWild = card.is_wild;
  const w = size === "sm" ? "w-9 h-[52px]" : "w-12 h-[68px]";
  const textSize = size === "sm" ? "text-xs" : "text-base";
  const suitSize = size === "sm" ? "text-[11px]" : "text-sm";

  let borderClass = "border-[1.5px] border-border";
  let shadowClass = "shadow-[0_1px_3px_rgba(0,0,0,0.08)]";
  let bgClass = "bg-surface";

  if (selected) {
    borderClass = "border-2 border-accent";
    shadowClass = "shadow-[0_2px_8px_rgba(217,119,87,0.2)]";
  } else if (isWild) {
    borderClass = "border-2 border-wild-gold";
    shadowClass = "shadow-[0_0_8px_rgba(232,168,73,0.25)]";
    bgClass = "bg-[#FFFBF5]";
  }

  return (
    <div
      className={`${w} ${bgClass} ${borderClass} rounded-md flex flex-col items-center justify-center ${shadowClass} cursor-pointer hover:translate-y-[-2px] transition-transform`}
      onClick={onClick}
    >
      <span className={`font-bold ${textSize} leading-none`} style={{ color }}>
        {card.rank_display}
      </span>
      <span className={`${suitSize} leading-none mt-0.5`} style={{ color }}>
        {card.suit_symbol}
      </span>
    </div>
  );
}
