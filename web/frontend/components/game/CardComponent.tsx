"use client";

import type { CardDTO } from "@/lib/types";

interface CardProps {
  card: CardDTO;
  selected?: boolean;
  onClick?: () => void;
  size?: "xs" | "sm" | "md";
}

const suitColor: Record<number, string> = {
  0: "#1A1612", // spade
  1: "#C75D4A", // heart
  2: "#C75D4A", // diamond
  3: "#1A1612", // club
};

export default function CardComponent({ card, selected, onClick, size = "md" }: CardProps) {
  const isJoker = card.rank >= 16;
  const color = isJoker
    ? card.rank === 17 ? "#C75D4A" : "#1A1612"  // RJ red, BJ dark
    : suitColor[card.suit] ?? "#1A1612";
  const isWild = card.is_wild;
  const w = size === "xs" ? "w-7 h-[40px]" : size === "sm" ? "w-9 h-[52px] lg:w-10 lg:h-[60px]" : "w-12 h-[68px] lg:w-14 lg:h-[80px]";
  const textSize = size === "xs" ? "text-[10px]" : size === "sm" ? "text-xs lg:text-sm" : "text-base lg:text-lg";
  const suitSize = size === "xs" ? "text-[9px]" : size === "sm" ? "text-[11px] lg:text-xs" : "text-sm lg:text-base";

  let borderClass = "border-[1.5px] border-border";
  let shadowClass = "shadow-[0_1px_3px_rgba(0,0,0,0.08)]";
  let bgClass = "bg-surface";

  if (selected) {
    borderClass = "border-2 border-accent";
    shadowClass = "shadow-[0_2px_8px_rgba(217,119,87,0.2)]";
  } else if (isWild) {
    bgClass = "bg-[#FFF8F9]";
  }

  return (
    <div
      data-card-id={card.id}
      className={`${w} ${bgClass} ${borderClass} rounded-md flex flex-col items-center justify-center ${shadowClass} cursor-pointer hover:translate-y-[-2px] transition-transform`}
      onClick={onClick}
    >
      <span className={`font-bold ${textSize} leading-none`} style={{ color }}>
        {card.rank_display}
      </span>
      {!isJoker && (
        <span className={`${suitSize} leading-none mt-0.5`} style={{ color }}>
          {card.suit_symbol}
        </span>
      )}
    </div>
  );
}
