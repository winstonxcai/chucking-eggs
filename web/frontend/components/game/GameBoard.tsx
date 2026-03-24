"use client";

import { useCallback, useMemo, useState } from "react";
import type { CardDTO, CardGroup, ComboDTO, GameOverMsg, GameState, TrickAction } from "@/lib/types";
import { findMatchingCombo, findStraightFlushes, validateCombo } from "@/lib/cards";
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
  const sfBySuit = useMemo(() => {
    const ungrouped = gameState.my_hand.filter((c) => !groupedCardIds.has(c.id));
    const result: Record<number, { label: string; cards: CardDTO[] }[]> = {};
    [0, 1, 2, 3].forEach((suit) => {
      result[suit] = findStraightFlushes(ungrouped, suit);
    });
    return result;
  }, [gameState.my_hand, groupedCardIds]);

  const handleFlushSelect = useCallback((cards: CardDTO[]) => {
    setSelectedIds(new Set(cards.map((c) => c.id)));
  }, []);

  // --- Layout data ---
  const partner = gameState.players.find((p) => p.seat === 2);
  const leftOpp = gameState.players.find((p) => p.seat === 1);
  const rightOpp = gameState.players.find((p) => p.seat === 3);

  const ta = gameState.trick_actions;

  return (
    <div className="flex h-screen bg-background">
      {/* Main board area */}
      <div className="flex-1 flex flex-col p-6 gap-0">
        {/* Play area: all players + table in a centered grid with equal gaps */}
        <div className="flex-1 flex items-center justify-center">
          <div className="grid grid-cols-[auto_480px_auto] grid-rows-[auto_260px] gap-32 items-center justify-items-center">
            {/* Partner (top center, spans column 2) */}
            <div className="col-start-2 row-start-1">
              {partner && (
                <OpponentPanel player={partner} thinking={aiThinking === 2} />
              )}
            </div>

            {/* Left opponent */}
            <div className="col-start-1 row-start-2">
              {leftOpp && (
                <OpponentPanel player={leftOpp} thinking={aiThinking === 1} />
              )}
            </div>

            {/* Table surface — all trick actions inside */}
            <div className="col-start-2 row-start-2 relative w-[480px] h-[260px] border border-border rounded-2xl">
              {/* Partner (top edge) */}
              <div className="absolute top-3 left-0 right-0 flex justify-center">
                <TrickActionDisplay action={ta?.["2"] ?? null} />
              </div>
              {/* Left opp (left edge) */}
              <div className="absolute left-3 top-0 bottom-0 flex items-center">
                <TrickActionDisplay action={ta?.["1"] ?? null} />
              </div>
              {/* Right opp (right edge) */}
              <div className="absolute right-3 top-0 bottom-0 flex items-center">
                <TrickActionDisplay action={ta?.["3"] ?? null} />
              </div>
              {/* You (bottom edge) */}
              <div className="absolute bottom-3 left-0 right-0 flex justify-center">
                <TrickActionDisplay action={ta?.["0"] ?? null} />
              </div>
              {/* New trick label */}
              {gameState.is_leading && !ta?.["0"] && !ta?.["1"] && !ta?.["2"] && !ta?.["3"] && (
                <div className="absolute inset-0 flex items-center justify-center">
                  <span className="text-sm text-text-secondary">New trick</span>
                </div>
              )}
            </div>

            {/* Right opponent */}
            <div className="col-start-3 row-start-2">
              {rightOpp && (
                <OpponentPanel player={rightOpp} thinking={aiThinking === 3} />
              )}
            </div>
          </div>
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
            onFlushSelect={handleFlushSelect}
            sfBySuit={sfBySuit}
            onGroup={handleGroup}
            onUngroup={handleUngroup}
            canGroup={selectedIds.size > 0}
            canUngroup={groups.some((g) =>
              g.cardIds.some((cid) => selectedIds.has(cid))
            )}
          />
        </div>
      </div>

      {/* Sidebar: always visible, two sections */}
      <div className="w-[280px] bg-surface border-l border-border p-5 overflow-y-auto flex flex-col gap-6">
        {/* Groups section */}
        <div className="flex flex-col gap-2">
          <span className="text-[13px] font-semibold text-text-secondary tracking-wider uppercase">
            Groups
          </span>
          {groups.length === 0 ? (
            <span className="text-sm text-text-secondary">No groups yet</span>
          ) : (
            <div className="flex flex-col gap-1.5">
              {groups.map((group, idx) => (
                <button
                  key={group.id}
                  className="flex items-center justify-between px-2.5 py-1.5 rounded-md bg-background border border-border hover:border-accent hover:text-accent text-left transition-colors"
                  onClick={() => handleGroupClick(group)}
                >
                  <span className="text-[13px] font-semibold text-foreground">
                    {group.comboName}
                  </span>
                  <span className="text-xs text-text-secondary">{group.cardIds.length}c</span>
                </button>
              ))}
            </div>
          )}
        </div>

        <div className="border-t border-border" />

        {/* Legal combos section */}
        <div className="flex flex-col gap-2">
          <span className="text-[13px] font-semibold text-text-secondary tracking-wider uppercase">
            Legal Combos
          </span>
          {gameState.is_my_turn ? (
            <ComboBrowser
              legalMoves={gameState.legal_moves.filter(
                (combo) => !combo.cards.some((c) => groupedCardIds.has(c.id))
              )}
              onSelectCombo={handleSelectCombo}
            />
          ) : (
            <span className="text-sm text-text-secondary">Not your turn</span>
          )}
        </div>
      </div>

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
