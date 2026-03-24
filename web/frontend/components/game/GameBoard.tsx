"use client";

import { useCallback, useMemo, useState } from "react";
import type { CardDTO, CardGroup, ComboDTO, GameOverMsg, GameState, TrickAction } from "@/lib/types";
import { findMatchingCombo, validateCombo } from "@/lib/cards";
import PlayerHand from "./PlayerHand";
import OpponentPanel from "./OpponentPanel";
import GameControls from "./GameControls";
import ComboBrowser from "./ComboBrowser";
import HandToolbar from "./HandToolbar";
import GameOverModal from "./GameOverModal";
import CardComponent from "./CardComponent";

interface GameBoardProps {
  gameState: GameState;
  aiThinking: number | null;
  gameOver: GameOverMsg | null;
  onPlayCards: (cardIds: string[]) => void;
  onPass: () => void;
  onPlayAgain: () => void;
}

/** Render a single trick action (cards or "Pass") */
function TrickActionDisplay({ action }: { action: TrickAction | null }) {
  if (!action) return null;
  if (action.type === "pass") {
    return <span className="text-xs text-text-secondary italic">Pass</span>;
  }
  if (action.combo) {
    return (
      <div className="flex gap-0.5">
        {action.combo.cards.map((card) => (
          <CardComponent key={card.id} card={card} size="sm" />
        ))}
      </div>
    );
  }
  return null;
}

export default function GameBoard({
  gameState,
  aiThinking,
  gameOver,
  onPlayCards,
  onPass,
  onPlayAgain,
}: GameBoardProps) {
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [groups, setGroups] = useState<CardGroup[]>([]);
  const [dragIdx, setDragIdx] = useState<number | null>(null);

  // --- Card selection ---
  const toggleCard = useCallback((id: string) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  const matchingCombo = useMemo(
    () => findMatchingCombo(selectedIds, gameState.legal_moves),
    [selectedIds, gameState.legal_moves]
  );

  const handlePlay = useCallback(() => {
    if (!matchingCombo) return;
    const ids = matchingCombo.cards.map((c) => c.id);
    onPlayCards(ids);
    setSelectedIds(new Set());
    // Dissolve groups that used played cards
    setGroups((prev) =>
      prev.filter((g) => !g.cardIds.some((cid) => ids.includes(cid)))
    );
  }, [matchingCombo, onPlayCards]);

  const handlePass = useCallback(() => {
    onPass();
    setSelectedIds(new Set());
  }, [onPass]);

  const handleSelectCombo = useCallback((combo: ComboDTO) => {
    setSelectedIds(new Set(combo.cards.map((c) => c.id)));
  }, []);

  // --- Grouping ---
  const groupedCardIds = useMemo(
    () => new Set(groups.flatMap((g) => g.cardIds)),
    [groups]
  );

  const handleGroup = useCallback(() => {
    if (selectedIds.size === 0) return;
    const selectedCards = gameState.my_hand.filter((c) => selectedIds.has(c.id));
    const result = validateCombo(selectedCards);
    if (!result) return;

    const newGroup: CardGroup = {
      id: `grp-${Date.now()}`,
      cardIds: selectedCards.map((c) => c.id),
      comboType: result.type,
      comboName: result.name,
    };
    setGroups((prev) => {
      // Remove any existing groups that overlap
      const cleaned = prev.filter(
        (g) => !g.cardIds.some((cid) => newGroup.cardIds.includes(cid))
      );
      return [...cleaned, newGroup];
    });
    setSelectedIds(new Set());
  }, [selectedIds, gameState.my_hand]);

  const handleUngroup = useCallback(() => {
    if (selectedIds.size === 0) return;
    setGroups((prev) =>
      prev.filter((g) => !g.cardIds.some((cid) => selectedIds.has(cid)))
    );
    setSelectedIds(new Set());
  }, [selectedIds]);

  const handleGroupClick = useCallback((group: CardGroup) => {
    setSelectedIds(new Set(group.cardIds));
  }, []);

  // --- Drag and drop for groups ---
  const handleDragStart = useCallback((idx: number) => {
    setDragIdx(idx);
  }, []);

  const handleDrop = useCallback(
    (targetIdx: number) => {
      if (dragIdx === null || dragIdx === targetIdx) return;
      setGroups((prev) => {
        const next = [...prev];
        const [moved] = next.splice(dragIdx, 1);
        next.splice(targetIdx, 0, moved);
        return next;
      });
      setDragIdx(null);
    },
    [dragIdx]
  );

  // --- Straight flush finder ---
  const handleFlushFind = useCallback(
    (suit: number) => {
      const suitCards = gameState.my_hand
        .filter((c) => c.suit === suit && c.rank >= 2 && c.rank <= 14)
        .sort((a, b) => a.rank - b.rank);

      if (suitCards.length < 5) return;

      // Deduplicate by rank (double deck)
      const uniqueByRank: CardDTO[] = [];
      let lastRank = -1;
      for (const c of suitCards) {
        if (c.rank !== lastRank) {
          uniqueByRank.push(c);
          lastRank = c.rank;
        }
      }

      // Find longest consecutive run
      let bestRun: CardDTO[] = [];
      let currentRun: CardDTO[] = [uniqueByRank[0]];

      for (let i = 1; i < uniqueByRank.length; i++) {
        if (uniqueByRank[i].rank === uniqueByRank[i - 1].rank + 1) {
          currentRun.push(uniqueByRank[i]);
        } else {
          if (currentRun.length > bestRun.length) bestRun = [...currentRun];
          currentRun = [uniqueByRank[i]];
        }
      }
      if (currentRun.length > bestRun.length) bestRun = [...currentRun];

      if (bestRun.length >= 5) {
        setSelectedIds(new Set(bestRun.slice(0, 5).map((c) => c.id)));
      }
    },
    [gameState.my_hand]
  );

  // --- Layout data ---
  const partner = gameState.players.find((p) => p.seat === 2);
  const leftOpp = gameState.players.find((p) => p.seat === 1);
  const rightOpp = gameState.players.find((p) => p.seat === 3);

  const ta = gameState.trick_actions;

  return (
    <div className="flex h-screen bg-background">
      {/* Main board area */}
      <div className="flex-1 flex flex-col p-6 gap-0">
        {/* Partner (top center) */}
        <div className="flex justify-center pb-2">
          {partner && (
            <OpponentPanel player={partner} thinking={aiThinking === 2} />
          )}
        </div>

        {/* Partner's trick action (below partner) */}
        <div className="flex justify-center pb-2 min-h-[56px]">
          <TrickActionDisplay action={ta?.["2"] ?? null} />
        </div>

        {/* Middle row: left action | TABLE | right action */}
        <div className="flex-1 flex items-center justify-center gap-4">
          {/* Left opponent + their action */}
          <div className="flex items-center gap-3">
            {leftOpp && (
              <OpponentPanel player={leftOpp} thinking={aiThinking === 1} />
            )}
            <div className="min-w-[80px] flex justify-center">
              <TrickActionDisplay action={ta?.["1"] ?? null} />
            </div>
          </div>

          {/* Table surface */}
          <div className="w-[300px] h-[180px] border border-border rounded-2xl flex items-center justify-center">
            {gameState.is_leading && !ta?.["0"] && !ta?.["1"] && !ta?.["2"] && !ta?.["3"] ? (
              <span className="text-sm text-text-secondary">New trick</span>
            ) : null}
          </div>

          {/* Right opponent + their action */}
          <div className="flex items-center gap-3">
            <div className="min-w-[80px] flex justify-center">
              <TrickActionDisplay action={ta?.["3"] ?? null} />
            </div>
            {rightOpp && (
              <OpponentPanel player={rightOpp} thinking={aiThinking === 3} />
            )}
          </div>
        </div>

        {/* Your trick action (above controls) */}
        <div className="flex justify-center pt-2 min-h-[56px]">
          <TrickActionDisplay action={ta?.["0"] ?? null} />
        </div>

        {/* Your turn indicator + Controls */}
        <div className="py-2">
          {gameState.is_my_turn && (
            <div className="flex items-center justify-center gap-1.5 pb-2">
              <div className="w-1.5 h-1.5 rounded-full bg-accent" />
              <span className="text-[13px] font-medium text-accent">
                {gameState.is_leading ? "Your turn to lead" : "Your turn to play"}
              </span>
            </div>
          )}
          <GameControls
            matchingCombo={matchingCombo}
            isLeading={gameState.is_leading}
            isMyTurn={gameState.is_my_turn}
            onPlay={handlePlay}
            onPass={handlePass}
          />
        </div>

        {/* Player hand with groups */}
        <div className="py-1">
          <PlayerHand
            cards={gameState.my_hand}
            selectedIds={selectedIds}
            onToggleCard={toggleCard}
            groups={groups}
            groupedCardIds={groupedCardIds}
            onGroupClick={handleGroupClick}
            onDragStart={handleDragStart}
            onDrop={handleDrop}
          />
        </div>

        {/* Hand toolbar */}
        <div className="py-2">
          <HandToolbar
            onFlushFind={handleFlushFind}
            onGroup={handleGroup}
            onUngroup={handleUngroup}
            canGroup={selectedIds.size > 0}
            canUngroup={groups.some((g) =>
              g.cardIds.some((cid) => selectedIds.has(cid))
            )}
          />
        </div>
      </div>

      {/* Sidebar: combo browser */}
      {gameState.is_my_turn && gameState.legal_moves.length > 0 && (
        <div className="w-[280px] bg-surface border-l border-border p-5 overflow-y-auto flex flex-col gap-4">
          <span className="text-[13px] font-semibold text-text-secondary tracking-wider uppercase">
            Legal Combos
          </span>
          <ComboBrowser
            legalMoves={gameState.legal_moves}
            onSelectCombo={handleSelectCombo}
          />
          <div className="mt-auto pt-4">
            <div className="p-3 bg-background rounded-lg">
              <span className="text-xs text-text-secondary">
                Click a combo to auto-select the cards in your hand
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Game over modal */}
      {gameOver && (
        <GameOverModal
          data={gameOver}
          humanSeat={gameState.my_seat}
          onPlayAgain={onPlayAgain}
        />
      )}
    </div>
  );
}
