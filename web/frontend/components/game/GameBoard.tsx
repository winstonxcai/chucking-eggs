"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ChevronDown, ChevronRight, Crown, Flag, Hand, HelpCircle, LogOut, MoreVertical } from "lucide-react";
import * as DropdownMenu from "@radix-ui/react-dropdown-menu";
import type { CardDTO, CardGroup, ComboDTO, GameOverMsg, GameState, TrickAction, TrickPlay } from "@/lib/types";
import { findMatchingCombo, validateCombo } from "@/lib/cards";
import PlayerHand from "./PlayerHand";
import OpponentPanel from "./OpponentPanel";
import GameControls from "./GameControls";
import ComboBrowser from "./ComboBrowser";
import HandToolbar from "./HandToolbar";
import GameOverModal from "./GameOverModal";
import CardComponent from "./CardComponent";
import FlyingCards from "./FlyingCards";
import type { ConnectionStatus } from "@/hooks/useGameSocket";

interface GameBoardProps {
  gameState: GameState;
  aiThinking: number | null;
  gameOver: GameOverMsg | null;
  connectionStatus?: ConnectionStatus;
  onPlayCards: (cardIds: string[]) => void;
  onPass: () => void;
  onPlayAgain: () => void;
  onRematch?: () => void;
  isMultiplayer?: boolean;
  onCreateGroup: (cardIds: string[], comboType: string, comboName: string) => void;
  onDeleteGroup: (groupId: string) => void;
  latestError?: { message: string; key: number } | null;
  autoPlayed?: { leading: boolean; key: number } | null;
  onForfeit?: () => void;
  onAbort?: () => void;
}

/** Render a single trick action (cards or "Pass") */
function TrickActionDisplay({ action, size = "sm" }: { action: TrickAction | null; size?: "xs" | "sm" | "md" }) {
  if (!action) return null;
  if (action.type === "pass") {
    return (
      <span className="flex items-center gap-1 px-2 py-0.5 rounded-full border border-border lg:border-white/20 text-xs text-text-secondary lg:text-white/55">
        <Hand className="w-3 h-3 shrink-0" />
        Pass
      </span>
    );
  }
  if (action.combo) {
    return (
      <div className="flex gap-0.5">
        {action.combo.cards.map((card) => (
          <CardComponent key={card.id} card={card} size={size} />
        ))}
      </div>
    );
  }
  return null;
}

/** Derive per-seat display action from ordered plays list. Always uses the
 * latest action per seat so passes correctly overwrite earlier plays. */
function lastPlayBySeat(plays: TrickPlay[]): Record<string, TrickAction> {
  const result: Record<string, TrickAction> = {};
  for (const p of plays) {
    result[p.seat] = { type: p.type, combo: p.combo };
  }
  return result;
}

export default function GameBoard({
  gameState,
  aiThinking,
  gameOver,
  connectionStatus,
  onPlayCards,
  onPass,
  onPlayAgain,
  onRematch,
  isMultiplayer,
  onCreateGroup,
  onDeleteGroup,
  latestError,
  autoPlayed,
  onForfeit,
  onAbort,
}: GameBoardProps) {
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [forfeitConfirm, setForfeitConfirm] = useState(false);
  const [abortConfirm, setAbortConfirm] = useState(false);
  const [flyingCards, setFlyingCards] = useState<{
    cards: CardDTO[];
    fromRects: DOMRect[];
    toRect: DOMRect;
  } | null>(null);
  const [toasts, setToasts] = useState<{ id: number; text: string }[]>([]);
  const [gameOverDismissed, setGameOverDismissed] = useState(false);
  const [reviewMode, setReviewMode] = useState(false);
  const [reviewPlayIdx, setReviewPlayIdx] = useState(0);
  const tableSeat0Ref = useRef<HTMLDivElement>(null);
  const flyingStartRef = useRef<number | null>(null);

  // Sidebar accordion state
  const [groupsOpen, setGroupsOpen] = useState(true);
  const [legalOpen, setLegalOpen] = useState(true);
  const [allOpen, setAllOpen] = useState(false);

  // Mobile state
  const [compactHand, setCompactHand] = useState(false);
  const [mobileComboOpen, setMobileComboOpen] = useState(false);

  // Reset dismissed state when a new game starts
  useEffect(() => {
    if (!gameOver) setGameOverDismissed(false);
  }, [gameOver]);

  useEffect(() => {
    const mq = window.matchMedia("(max-width: 1023px)");
    const update = (e: MediaQueryListEvent | MediaQueryList) => setCompactHand(e.matches);
    update(mq);
    mq.addEventListener("change", update);
    return () => mq.removeEventListener("change", update);
  }, []);

  // Optimistic local groups — updated immediately on Group/Ungroup, synced from server on game_state
  const [localGroups, setLocalGroups] = useState<CardGroup[]>(gameState.groups);
  useEffect(() => {
    setLocalGroups((prev) => {
      const serverCardSets = new Set(
        gameState.groups.map((g) => [...g.cardIds].sort().join(","))
      );
      const pendingTemps = prev.filter(
        (g) =>
          g.id.startsWith("grp-temp-") &&
          !serverCardSets.has([...g.cardIds].sort().join(","))
      );
      return [...gameState.groups, ...pendingTemps];
    });
  }, [gameState.groups]);

  const groupedCardIds = useMemo(
    () => new Set(localGroups.flatMap((g) => g.cardIds)),
    [localGroups]
  );

  // --- Card selection ---
  const toggleCard = useCallback((id: string) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  const toggleCards = useCallback((ids: string[]) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      ids.forEach((id) => next.add(id));
      return next;
    });
  }, []);

  const matchingCombo = useMemo(
    () => findMatchingCombo(selectedIds, gameState.legal_moves, gameState.my_hand),
    [selectedIds, gameState.legal_moves, gameState.my_hand]
  );

  const handlePlay = useCallback(() => {
    if (!matchingCombo) return;
    const ids = matchingCombo.cards.map((c) => c.id);

    // Capture positions for fly animation
    const fromRects = ids.map((id) => {
      const el = document.querySelector(`[data-card-id="${id}"]`);
      return el?.getBoundingClientRect() ?? new DOMRect();
    });
    const toRect = tableSeat0Ref.current?.getBoundingClientRect() ?? new DOMRect();

    flyingStartRef.current = Date.now();
    setFlyingCards({ cards: matchingCombo.cards, fromRects, toRect });
    onPlayCards(ids);
    setSelectedIds(new Set());
  }, [matchingCombo, onPlayCards]);

  // Reset the flying clock to when the CSS transition actually starts (after double rAF),
  // so the 300ms minimum wait is measured from animation start, not from handlePlay.
  const handleFlyingArrived = useCallback(() => {
    flyingStartRef.current = Date.now();
  }, []);

  // Clear flying overlay once game state updates (server confirmed the play),
  // but always wait at least 300ms so the animation can complete.
  useEffect(() => {
    if (!flyingCards || flyingStartRef.current === null) return;
    const elapsed = Date.now() - flyingStartRef.current;
    const remaining = Math.max(0, 300 - elapsed);
    const t = setTimeout(() => {
      setFlyingCards(null);
      flyingStartRef.current = null;
    }, remaining);
    return () => clearTimeout(t);
  }, [gameState]); // eslint-disable-line react-hooks/exhaustive-deps

  // On server error: immediately clear any stuck flying animation and show a toast
  useEffect(() => {
    if (!latestError) return;
    const { key, message } = latestError;
    setFlyingCards(null);
    flyingStartRef.current = null;
    setToasts((prev) => [...prev, { id: key, text: message }]);
    const t = setTimeout(() => {
      setToasts((prev) => prev.filter((toast) => toast.id !== key));
    }, 3000);
    return () => clearTimeout(t);
  }, [latestError?.key]); // eslint-disable-line react-hooks/exhaustive-deps

  // On AFK auto-play: show toast explaining what happened
  useEffect(() => {
    if (!autoPlayed) return;
    const { key, leading } = autoPlayed;
    const text = leading
      ? "Time's up — random move played"
      : "Time's up — auto-passed";
    setToasts((prev) => [...prev, { id: key, text }]);
    const t = setTimeout(() => {
      setToasts((prev) => prev.filter((toast) => toast.id !== key));
    }, 3000);
    return () => clearTimeout(t);
  }, [autoPlayed?.key]); // eslint-disable-line react-hooks/exhaustive-deps

  const handlePass = useCallback(() => {
    onPass();
    setSelectedIds(new Set());
  }, [onPass]);

  const handleSelectCombo = useCallback((combo: ComboDTO) => {
    setSelectedIds(new Set(combo.cards.map((c) => c.id)));
  }, []);

  // --- Grouping (optimistic: update locally then sync to backend) ---
  const handleGroup = useCallback(() => {
    if (selectedIds.size === 0) return;
    const selectedCards = gameState.my_hand.filter((c) => selectedIds.has(c.id));
    const result = validateCombo(selectedCards);
    if (!result) return;
    const cardIds = selectedCards.map((c) => c.id);
    const idSet = new Set(cardIds);
    const newGroup: CardGroup = {
      id: `grp-temp-${Date.now()}`,
      cardIds,
      comboType: result.type,
      comboName: result.name,
    };
    setLocalGroups((prev) => [
      ...prev.filter((g) => !g.cardIds.some((cid) => idSet.has(cid))),
      newGroup,
    ]);
    onCreateGroup(cardIds, result.type, result.name);
    setSelectedIds(new Set());
  }, [selectedIds, gameState.my_hand, onCreateGroup]);

  const handleUngroup = useCallback(() => {
    if (selectedIds.size === 0) return;
    const toDelete = localGroups.find((g) => g.cardIds.some((cid) => selectedIds.has(cid)));
    if (toDelete) {
      setLocalGroups((prev) => prev.filter((g) => g.id !== toDelete.id));
      onDeleteGroup(toDelete.id);
    }
    setSelectedIds(new Set());
  }, [selectedIds, localGroups, onDeleteGroup]);

  const handleGroupClick = useCallback((group: CardGroup) => {
    setSelectedIds((prev) => {
      const allSelected = group.cardIds.every((id) => prev.has(id));
      if (allSelected) return new Set();
      return new Set(group.cardIds);
    });
  }, []);

  // --- Straight flush finder (computed by backend) ---
  const sfBySuit = useMemo(() => {
    const handById = new Map(gameState.my_hand.map((c) => [c.id, c]));
    const result: Record<number, { label: string; cards: CardDTO[] }[]> = {};
    [0, 1, 2, 3].forEach((suit) => {
      result[suit] = (gameState.sf_options[suit] ?? []).map(({ label, cardIds }) => ({
        label,
        cards: cardIds.map((id) => handById.get(id)).filter(Boolean) as CardDTO[],
      }));
    });
    return result;
  }, [gameState.sf_options, gameState.my_hand]);

  const handleFlushSelect = useCallback((cards: CardDTO[]) => {
    setSelectedIds(new Set(cards.map((c) => c.id)));
  }, []);

  // Gate layout changes behind animation to prevent shifts that move tableSeat0Ref mid-flight
  // In review mode always show hand + controls (game is over but we display historical hands)
  const handDone = !reviewMode && compactHand && gameState.my_hand.length === 0 && !flyingCards;

  // --- Layout data ---
  const partner = gameState.players.find((p) => p.seat === 2);
  const leftOpp = gameState.players.find((p) => p.seat === 1);
  const rightOpp = gameState.players.find((p) => p.seat === 3);

  // Review mode: flatten all plays across all tricks into a single navigable list
  const trickHistory = gameOver?.trick_history;
  const reviewSteps = useMemo(() => {
    if (!trickHistory) return [];
    const steps: { snapshot: typeof trickHistory[0]; playsSlice: typeof trickHistory[0]["plays"] }[] = [];
    for (const snap of trickHistory) {
      for (let i = 0; i < snap.plays.length; i++) {
        steps.push({ snapshot: snap, playsSlice: snap.plays.slice(0, i + 1) });
      }
    }
    return steps;
  }, [trickHistory]);
  const reviewTotal = reviewSteps.length;
  const currentStep = reviewMode && reviewSteps.length > 0 ? reviewSteps[reviewPlayIdx] : null;
  const snapshot = currentStep?.snapshot ?? null;
  const reviewHands = snapshot?.hands_before ?? {};
  const reviewWinnerSeat = snapshot?.winner_seat ?? null;
  // Crown only appears at the last play of a trick in review (not prematurely on earlier steps)
  const isLastPlayOfTrick = !reviewMode ? false :
    reviewPlayIdx === reviewTotal - 1 ||
    reviewSteps[reviewPlayIdx + 1]?.snapshot !== snapshot;
  // In review mode: Crown marks the trick winner (only at trick's last step); live: Crown marks trick lead
  const trickLeadOrWinner = reviewMode ? (isLastPlayOfTrick ? reviewWinnerSeat : null) : gameState.trick_lead_seat;
  // In review mode: use plays up to current step only
  const ta = reviewMode && currentStep ? lastPlayBySeat(currentStep.playsSlice) : gameState.trick_actions;
  // In review mode: remove cards the user has already played within the current trick
  const reviewMyHand = useMemo(() => {
    if (!reviewMode || !currentStep) return reviewHands["0"] ?? [];
    const base = reviewHands["0"] ?? [];
    const playedIds = new Set<string>();
    for (const p of currentStep.playsSlice) {
      if (p.seat === "0" && p.type === "play" && p.combo) {
        for (const card of p.combo.cards) playedIds.add(card.id);
      }
    }
    return base.filter((c) => !playedIds.has(c.id));
  }, [reviewMode, currentStep, reviewHands]);
  const fo = reviewMode ? [] : gameState.finish_order;
  const finishPos = (seat: number) => {
    const idx = fo.indexOf(seat);
    return idx >= 0 ? idx + 1 : null;
  };
  const ORDINAL = ["", "1st", "2nd", "3rd", "4th"];

  return (
    <div className="flex h-[100dvh] bg-background overflow-x-hidden">
      {/* Main board area */}
      <div className="flex-1 flex flex-col p-[5px] lg:px-6 lg:pt-2 lg:pb-1 gap-0 relative lg:justify-center">
        {/* Top-right buttons */}
        <div className="absolute top-2 right-2 lg:top-4 lg:right-4 z-20 flex items-center gap-2">
          {gameOver && gameOverDismissed && !reviewMode && (
            <button
              onClick={() => setGameOverDismissed(false)}
              className="text-text-secondary hover:text-foreground transition-colors text-xs font-medium"
              title="View results"
            >
              Results
            </button>
          )}
          <DropdownMenu.Root
            onOpenChange={(open) => {
              if (!open) { setForfeitConfirm(false); setAbortConfirm(false); }
            }}
          >
            <DropdownMenu.Trigger asChild>
              <button
                data-testid="game-menu-button"
                className="text-text-secondary hover:text-foreground transition-colors p-1"
                title="Game menu"
              >
                <MoreVertical size={18} />
              </button>
            </DropdownMenu.Trigger>
            <DropdownMenu.Portal>
              <DropdownMenu.Content
                align="end"
                sideOffset={4}
                className="bg-surface border border-border rounded-lg shadow-md p-1 min-w-[140px] z-30"
              >
                {/* Rules — always shown */}
                <DropdownMenu.Item asChild>
                  <a
                    href="/rules"
                    className="flex items-center gap-2 w-full px-3 py-2 text-xs font-medium text-foreground rounded hover:bg-muted outline-none transition-colors cursor-pointer"
                  >
                    <HelpCircle size={13} /> Rules
                  </a>
                </DropdownMenu.Item>

                {/* Abort — only before first move */}
                {onAbort && !abortConfirm && !forfeitConfirm && (
                  <DropdownMenu.Item
                    className="flex items-center gap-2 w-full text-left px-3 py-2 text-xs font-medium text-text-secondary rounded hover:bg-muted outline-none transition-colors cursor-pointer"
                    onSelect={(e) => { e.preventDefault(); setAbortConfirm(true); }}
                  >
                    <LogOut size={13} /> Abort Game
                  </DropdownMenu.Item>
                )}
                {onAbort && abortConfirm && (
                  <div className="flex flex-col gap-1.5 p-2">
                    <span className="text-[11px] text-text-secondary font-medium">AI takes your seat. No ELO change.</span>
                    <div className="flex gap-1.5">
                      <button
                        className="flex-1 py-1 bg-surface border border-border text-xs font-semibold rounded hover:border-foreground transition-colors cursor-pointer"
                        onClick={() => onAbort()}
                      >Confirm</button>
                      <button
                        className="flex-1 py-1 bg-background border border-border text-xs rounded hover:border-foreground transition-colors cursor-pointer"
                        onClick={() => setAbortConfirm(false)}
                      >Cancel</button>
                    </div>
                  </div>
                )}

                {/* Forfeit — only multiplayer */}
                {onForfeit && !forfeitConfirm && !abortConfirm && (
                  <DropdownMenu.Item
                    className="flex items-center gap-2 w-full text-left px-3 py-2 text-xs font-medium text-team-red rounded hover:bg-red-50 outline-none transition-colors cursor-pointer"
                    onSelect={(e) => { e.preventDefault(); setForfeitConfirm(true); }}
                  >
                    <Flag size={13} /> Forfeit Game
                  </DropdownMenu.Item>
                )}
                {onForfeit && forfeitConfirm && (
                  <div className="flex flex-col gap-1.5 p-2">
                    <span className="text-[11px] text-team-red font-medium">You will lose ELO.</span>
                    <div className="flex gap-1.5">
                      <button
                        className="flex-1 py-1 bg-team-red text-white text-xs font-semibold rounded hover:opacity-90 transition-opacity cursor-pointer"
                        onClick={() => onForfeit()}
                      >Confirm</button>
                      <button
                        className="flex-1 py-1 bg-background border border-border text-xs rounded hover:border-foreground transition-colors cursor-pointer"
                        onClick={() => setForfeitConfirm(false)}
                      >Cancel</button>
                    </div>
                  </div>
                )}
              </DropdownMenu.Content>
            </DropdownMenu.Portal>
          </DropdownMenu.Root>
        </div>
        {/* Reconnection banner */}
        {connectionStatus === "reconnecting" && (
          <div className="flex items-center justify-center gap-2 py-2 px-4 bg-accent/5 border border-accent/20 rounded-lg mb-2 w-1/2 mx-auto">
            <div className="w-3 h-3 border-2 border-accent border-t-transparent rounded-full animate-spin" />
            <span className="text-xs font-medium text-accent">Reconnecting to server...</span>
          </div>
        )}
        {connectionStatus === "disconnected" && (
          <div className="flex items-center justify-center gap-2 py-2 px-4 bg-team-red/5 border border-team-red/20 rounded-lg mb-2 w-1/2 mx-auto">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--team-red)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="shrink-0">
              <circle cx="12" cy="12" r="10" />
              <line x1="15" y1="9" x2="9" y2="15" />
              <line x1="9" y1="9" x2="15" y2="15" />
            </svg>
            <span className="text-xs font-medium text-team-red">Connection lost</span>
            <button onClick={() => window.location.reload()} className="text-xs font-semibold text-team-red underline underline-offset-2 hover:opacity-70 transition-opacity">Refresh</button>
          </div>
        )}

        {/* Play area: all players + table in a centered grid */}
        <div className="flex-1 flex items-start lg:items-center justify-center min-h-0">
          <div className="grid grid-cols-3 grid-rows-[auto_1fr] lg:grid-cols-[auto_672px_auto] lg:grid-rows-[auto_auto] gap-x-1 gap-y-3 lg:gap-x-10 lg:gap-y-10 items-start lg:items-center justify-items-center w-full h-full lg:h-auto lg:w-auto">
            {/* Partner (top center, spans column 2) */}
            <div className="col-start-2 row-start-1 h-7 lg:h-[82px] relative overflow-visible flex justify-center">
              {partner && (
                <div className="absolute bottom-0 left-1/2 -translate-x-1/2">
                  <OpponentPanel player={partner} thinking={aiThinking === 2} isActive={!reviewMode && gameState.current_player === 2} revealedHand={reviewMode ? (reviewHands["2"] ?? []) : (handDone ? undefined : gameState.partner_hand)} isReviewMode={reviewMode} />
                </div>
              )}
            </div>

            {/* Left opponent */}
            <div className="col-start-1 row-start-1 lg:row-start-2 h-7 lg:h-[82px] w-[60px] lg:w-[96px] relative overflow-visible">
              {leftOpp && (
                <div className="absolute top-0 left-0 lg:left-auto lg:right-0 lg:top-1/2 lg:-translate-y-1/2">
                  <OpponentPanel player={leftOpp} thinking={aiThinking === 1} isActive={!reviewMode && gameState.current_player === 1} revealedHand={reviewMode ? (reviewHands["1"] ?? []) : (handDone ? undefined : gameState.opponent_hands?.["1"])} isReviewMode={reviewMode} />
                </div>
              )}
            </div>

            {/* Table surface — all trick actions inside */}
            <div className="col-start-1 col-span-3 row-start-2 lg:col-start-2 lg:col-span-1 relative w-full h-full lg:w-[672px] lg:h-[360px] lg:rounded-2xl lg:border-2 lg:border-[#D9CFC2]/25 lg:bg-gradient-to-br lg:from-[#3D3329] lg:to-[#2A221A] lg:shadow-[inset_0_2px_24px_rgba(0,0,0,0.4),0_6px_20px_rgba(0,0,0,0.15)]">
              {/* Partner (top edge) */}
              <div data-testid="trick-seat-2" className="absolute top-3 left-0 right-0 flex justify-center">
                <div className="relative inline-flex">
                  {finishPos(2) && !ta?.["2"] ? (
                    <span className="px-2 py-0.5 rounded-full bg-white/10 text-[11px] font-semibold text-text-secondary lg:text-white/60">{ORDINAL[finishPos(2)!]}</span>
                  ) : (
                    <>
                      <TrickActionDisplay action={ta?.["2"] ?? null} size="sm" />
                      <Crown className={`absolute -top-2 -right-2 w-4 h-4 text-amber-500${trickLeadOrWinner === 2 ? "" : " invisible"}`} />
                    </>
                  )}
                </div>
              </div>
              {/* Left opp — mobile: centered under col 1 (1/6 from left); desktop: left edge */}
              <div data-testid="trick-seat-1" className="absolute left-[16.67%] -translate-x-1/2 top-0 bottom-0 flex items-center lg:left-3 lg:translate-x-0">
                <div className="relative inline-flex">
                  {finishPos(1) && !ta?.["1"] ? (
                    <span className="px-2 py-0.5 rounded-full bg-white/10 text-[11px] font-semibold text-text-secondary lg:text-white/60">{ORDINAL[finishPos(1)!]}</span>
                  ) : (
                    <>
                      <TrickActionDisplay action={ta?.["1"] ?? null} size="sm" />
                      <Crown className={`absolute -top-2 -right-2 w-4 h-4 text-amber-500${trickLeadOrWinner === 1 ? "" : " invisible"}`} />
                    </>
                  )}
                </div>
              </div>
              {/* Right opp — mobile: centered under col 3 (5/6 from left); desktop: right edge */}
              <div data-testid="trick-seat-3" className="absolute left-[83.33%] -translate-x-1/2 top-0 bottom-0 flex items-center lg:left-auto lg:right-3 lg:translate-x-0">
                <div className="relative inline-flex">
                  {finishPos(3) && !ta?.["3"] ? (
                    <span className="px-2 py-0.5 rounded-full bg-white/10 text-[11px] font-semibold text-text-secondary lg:text-white/60">{ORDINAL[finishPos(3)!]}</span>
                  ) : (
                    <>
                      <TrickActionDisplay action={ta?.["3"] ?? null} size="sm" />
                      <Crown className={`absolute -top-2 -right-2 w-4 h-4 text-amber-500${trickLeadOrWinner === 3 ? "" : " invisible"}`} />
                    </>
                  )}
                </div>
              </div>
              {/* You (bottom edge) */}
              <div ref={tableSeat0Ref} data-testid="trick-seat-0" className="absolute bottom-3 left-0 right-0 flex justify-center">
                <div className="relative inline-flex">
                  {finishPos(gameState.my_seat) && !ta?.["0"] ? (
                    <span className="px-2 py-0.5 rounded-full bg-white/10 text-[11px] font-semibold text-text-secondary lg:text-white/60">{ORDINAL[finishPos(gameState.my_seat)!]}</span>
                  ) : (
                    <>
                      {!flyingCards && <TrickActionDisplay action={ta?.["0"] ?? null} size="sm" />}
                      <Crown className={`absolute -top-2 -right-2 w-4 h-4 text-amber-500${trickLeadOrWinner === 0 ? "" : " invisible"}`} />
                    </>
                  )}
                </div>
              </div>
              {/* New trick label */}
              {!reviewMode && gameState.is_leading && !ta?.["0"] && !ta?.["1"] && !ta?.["2"] && !ta?.["3"] && (
                <div className="absolute inset-0 flex items-center justify-center">
                  <span className="text-sm text-text-secondary lg:text-white/40">New trick</span>
                </div>
              )}
            </div>

            {/* Right opponent */}
            <div className="col-start-3 row-start-1 lg:row-start-2 h-7 lg:h-[82px] w-[60px] lg:w-[96px] relative overflow-visible">
              {rightOpp && (
                <div className="absolute top-0 right-0 lg:right-auto lg:left-0 lg:top-1/2 lg:-translate-y-1/2">
                  <OpponentPanel player={rightOpp} thinking={aiThinking === 3} isActive={!reviewMode && gameState.current_player === 3} revealedHand={reviewMode ? (reviewHands["3"] ?? []) : (handDone ? undefined : gameState.opponent_hands?.["3"])} isReviewMode={reviewMode} />
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Your turn indicator + Controls — or Review nav bar */}
        <div className={`mt-auto lg:mt-0 py-0.5 lg:py-0 ${handDone ? "invisible pointer-events-none lg:visible lg:pointer-events-auto" : ""}`}>
          {reviewMode ? (
            <div data-testid="review-nav" className="flex items-center justify-center gap-3 py-2.5">
              <button
                data-testid="review-prev"
                disabled={reviewPlayIdx === 0}
                onClick={() => setReviewPlayIdx((i) => Math.max(0, i - 1))}
                className="w-9 h-9 flex items-center justify-center rounded-lg border border-border text-foreground hover:bg-surface disabled:opacity-30 disabled:cursor-not-allowed transition-colors text-base font-medium"
              >
                ←
              </button>
              <span data-testid="review-trick-counter" className="text-sm text-text-secondary tabular-nums w-44 text-center">
                Trick {snapshot?.trick_num ?? "—"} · Play {reviewPlayIdx + 1} of {reviewTotal}
              </span>
              <button
                data-testid="review-next"
                disabled={reviewPlayIdx === reviewTotal - 1}
                onClick={() => setReviewPlayIdx((i) => Math.min(reviewTotal - 1, i + 1))}
                className="w-9 h-9 flex items-center justify-center rounded-lg border border-border text-foreground hover:bg-surface disabled:opacity-30 disabled:cursor-not-allowed transition-colors text-base font-medium"
              >
                →
              </button>
              <button
                onClick={() => { setReviewMode(false); setGameOverDismissed(false); }}
                className="ml-2 px-3 py-1.5 text-xs font-medium text-text-secondary border border-border rounded-lg hover:text-foreground hover:border-foreground/30 transition-colors"
              >
                Done
              </button>
            </div>
          ) : (
            <>
              {/* Desktop-only turn label */}
              <div className={`hidden lg:flex items-center justify-center gap-1.5 pb-1 ${!gameState.is_my_turn ? "invisible" : ""}`}>
                <div className="w-1.5 h-1.5 rounded-full bg-accent" />
                <span className="text-[13px] font-medium text-accent">
                  {gameState.is_leading ? "Your turn to lead" : "Your turn to play"}
                </span>
              </div>
              <GameControls
                matchingCombo={matchingCombo}
                isLeading={gameState.is_leading}
                isMyTurn={gameState.is_my_turn}
                hasSelection={selectedIds.size > 0}
                turnDeadlineMs={gameState.turn_deadline_ms}
                onPlay={handlePlay}
                onPass={handlePass}
                onUnselect={() => setSelectedIds(new Set())}
                compact={compactHand}
                canGroup={selectedIds.size > 0 && !Array.from(selectedIds).some((id) => groupedCardIds.has(id))}
                canUngroup={localGroups.some((g) => g.cardIds.some((cid) => selectedIds.has(cid)))}
                onGroup={handleGroup}
                onUngroup={handleUngroup}
                onCombos={() => setMobileComboOpen(true)}
              />
            </>
          )}
        </div>

        {/* Player hand with groups — hidden on mobile once player is finished */}
        <div className={`py-0 lg:py-1 ${handDone ? "invisible pointer-events-none lg:visible lg:pointer-events-auto" : ""}`}>
          <PlayerHand
            cards={reviewMode ? reviewMyHand : gameState.my_hand}
            selectedIds={reviewMode ? new Set<string>() : selectedIds}
            onToggleCard={reviewMode ? () => {} : toggleCard}
            onToggleCards={reviewMode ? () => {} : toggleCards}
            groups={reviewMode ? [] : localGroups}
            groupedCardIds={reviewMode ? new Set<string>() : groupedCardIds}
            onGroupClick={reviewMode ? () => {} : handleGroupClick}
            hiddenIds={reviewMode ? undefined : (flyingCards ? new Set(flyingCards.cards.map((c) => c.id)) : undefined)}
            compact={compactHand}
          />
        </div>

        {/* Hand toolbar: desktop only */}
        <div className="hidden lg:flex py-1 items-center justify-center gap-2">
          <HandToolbar
            onFlushSelect={handleFlushSelect}
            sfBySuit={sfBySuit}
            onGroup={handleGroup}
            onUngroup={handleUngroup}
            canGroup={selectedIds.size > 0 && !Array.from(selectedIds).some((id) => groupedCardIds.has(id))}
            canUngroup={localGroups.some((g) =>
              g.cardIds.some((cid) => selectedIds.has(cid))
            )}
          />
        </div>
      </div>

      {/* Sidebar: desktop only, hidden once player is out of cards or in review mode */}
      <div className={`w-[280px] bg-surface border-l border-border p-5 overflow-y-auto flex-col gap-4 ${reviewMode || (!flyingCards && gameState.my_hand.length === 0) ? "hidden" : "hidden lg:flex"}`}>
        {/* Groups section */}
        <div className="flex flex-col gap-2">
          <button
            onClick={() => setGroupsOpen((o) => !o)}
            className="flex items-center justify-between w-full text-[13px] font-semibold text-text-secondary tracking-wider uppercase"
          >
            <span>Groups</span>
            {groupsOpen ? <ChevronDown size={14} strokeWidth={2} /> : <ChevronRight size={14} strokeWidth={2} />}
          </button>
          {groupsOpen && (
            localGroups.length === 0 ? (
              <span className="text-sm text-text-secondary">No groups yet</span>
            ) : (
              <div className="flex flex-col gap-1.5">
                {localGroups.map((group) => (
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
            )
          )}
        </div>

        <div className="border-t border-border" />

        {/* Legal combos section — excludes cards already in groups */}
        <div className="flex flex-col gap-2">
          <button
            onClick={() => setLegalOpen((o) => !o)}
            className="flex items-center justify-between w-full text-[13px] font-semibold text-text-secondary tracking-wider uppercase"
          >
            <span>Legal Combos</span>
            {legalOpen ? <ChevronDown size={14} strokeWidth={2} /> : <ChevronRight size={14} strokeWidth={2} />}
          </button>
          {legalOpen && (
            gameState.is_my_turn ? (
              <ComboBrowser
                legalMoves={gameState.legal_moves.filter(
                  (m) => !m.cards.every((c) => groupedCardIds.has(c.id))
                )}
                onSelectCombo={handleSelectCombo}
              />
            ) : (
              <span className="text-sm text-text-secondary">Not your turn</span>
            )
          )}
        </div>

        <div className="border-t border-border" />

        {/* All combos section — all valid combos from hand, ignoring current trick */}
        <div className="flex flex-col gap-2">
          <button
            onClick={() => setAllOpen((o) => !o)}
            className="flex items-center justify-between w-full text-[13px] font-semibold text-text-secondary tracking-wider uppercase"
          >
            <span>All Combos</span>
            {allOpen ? <ChevronDown size={14} strokeWidth={2} /> : <ChevronRight size={14} strokeWidth={2} />}
          </button>
          {allOpen && (
            <ComboBrowser
              legalMoves={(gameState.all_moves ?? []).filter(
                (m) => !m.cards.every((c) => groupedCardIds.has(c.id))
              )}
              onSelectCombo={handleSelectCombo}
            />
          )}
        </div>

      </div>

      {/* Mobile combo bottom sheet */}
      {mobileComboOpen && (
        <div className="lg:hidden fixed inset-0 z-50 flex flex-col justify-end">
          <div
            className="absolute inset-0 bg-black/40"
            onClick={() => setMobileComboOpen(false)}
          />
          <div className="relative bg-surface rounded-t-2xl p-5 max-h-[60dvh] overflow-y-auto flex flex-col gap-4">
            <div className="flex items-center justify-between">
              <span className="text-sm font-semibold text-foreground">Legal Combos</span>
              <button
                onClick={() => setMobileComboOpen(false)}
                className="text-xs text-text-secondary hover:text-foreground transition-colors"
              >
                Close
              </button>
            </div>
            {gameState.is_my_turn ? (
              <ComboBrowser
                legalMoves={gameState.legal_moves.filter(
                  (m) => !m.cards.every((c) => groupedCardIds.has(c.id))
                )}
                onSelectCombo={(combo) => {
                  handleSelectCombo(combo);
                  setMobileComboOpen(false);
                }}
              />
            ) : (
              <span className="text-sm text-text-secondary">Not your turn</span>
            )}
          </div>
        </div>
      )}

      {/* Toast notifications — server error feedback */}
      {toasts.length > 0 && (
        <div className="fixed bottom-6 left-1/2 -translate-x-1/2 z-[60] flex flex-col items-center gap-2 pointer-events-none">
          {toasts.map((toast) => (
            <div
              key={toast.id}
              className="bg-foreground text-background text-sm font-medium px-4 py-2.5 rounded-xl shadow-lg"
            >
              {toast.text}
            </div>
          ))}
        </div>
      )}

      {/* Card fly animation overlay */}
      {flyingCards && (
        <FlyingCards
          cards={flyingCards.cards}
          fromRects={flyingCards.fromRects}
          toRect={flyingCards.toRect}
          compact={compactHand}
          onArrived={handleFlyingArrived}
        />
      )}

      {/* Game over modal */}
      {gameOver && !gameOverDismissed && (
        <GameOverModal
          data={gameOver}
          humanSeat={gameState.my_seat}
          onPlayAgain={() => {
            setGameOverDismissed(true);
            (isMultiplayer && onRematch ? onRematch : onPlayAgain)?.();
          }}
          isMultiplayer={isMultiplayer}
          onDismiss={() => setGameOverDismissed(true)}
          onReview={() => {
            setReviewMode(true);
            setReviewPlayIdx(0);
            setGameOverDismissed(true);
          }}
        />
      )}
    </div>
  );
}
