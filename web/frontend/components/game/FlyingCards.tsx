"use client";

import { useEffect, useState } from "react";
import type { CardDTO } from "@/lib/types";
import CardComponent from "./CardComponent";

interface FlyingCardsProps {
  cards: CardDTO[];
  fromRects: DOMRect[];
  toRect: DOMRect;
}

export default function FlyingCards({ cards, fromRects, toRect }: FlyingCardsProps) {
  const [arrived, setArrived] = useState(false);

  useEffect(() => {
    // Double rAF to ensure initial position is painted before triggering transition
    requestAnimationFrame(() => {
      requestAnimationFrame(() => setArrived(true));
    });
  }, []);

  // Target: center of toRect, cards side by side (sm = 36px wide, 2px gap)
  const totalWidth = cards.length * 36 + (cards.length - 1) * 2;
  const targetBaseX = toRect.left + toRect.width / 2 - totalWidth / 2;
  const targetY = toRect.top + toRect.height / 2 - 26;

  return (
    <div className="fixed inset-0 z-50 pointer-events-none">
      {cards.map((card, i) => {
        const from = fromRects[i];
        const targetX = targetBaseX + i * 38;

        const style: React.CSSProperties = arrived
          ? {
              left: targetX,
              top: targetY,
              width: 36,
              height: 52,
              transition: "all 300ms cubic-bezier(0.4, 0, 0.2, 1)",
            }
          : {
              left: from.left,
              top: from.top,
              width: from.width,
              height: from.height,
            };

        return (
          <div key={card.id} className="absolute" style={style}>
            <CardComponent card={card} size={arrived ? "sm" : "md"} />
          </div>
        );
      })}
    </div>
  );
}
