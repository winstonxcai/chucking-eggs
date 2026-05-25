import { test, expect, type Page } from "@playwright/test";

// Backend API base — set API_URL env var when running against production
const API_BASE = process.env.API_URL || "http://localhost:8000";

const unique = (prefix: string) => `${prefix.slice(0, 10)}-${Math.random().toString(36).slice(2, 8)}`;

class TurnChangedError extends Error {}

async function claimPlayer(request: import("@playwright/test").APIRequestContext, username: string) {
  const response = await request.post(`${API_BASE}/api/auth/claim`, { data: { username } });
  expect(response.ok(), await response.text()).toBeTruthy();
  return response.json() as Promise<{
    player_id: string;
    player_token: string;
    username: string;
    elo: number;
  }>;
}

async function applyPlayerIdentity(
  ctx: import("@playwright/test").BrowserContext,
  player: { player_id: string; player_token: string; username: string; elo: number }
) {
  await ctx.addInitScript(
    ({ id, token, name, elo }: { id: string; token: string; name: string; elo: number }) => {
      localStorage.setItem("ce_player_id", id);
      localStorage.setItem("ce_player_token", token);
      localStorage.setItem("ce_username", name);
      localStorage.setItem("ce_elo", String(elo));
    },
    { id: player.player_id, token: player.player_token, name: player.username, elo: player.elo }
  );
}

function seededRandom(seed: number) {
  let state = seed >>> 0;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

function randomIndex(rand: () => number, count: number) {
  return Math.min(count - 1, Math.floor(rand() * count));
}

async function visibleElementCount(page: Page, selector: string) {
  return page.locator(`${selector}:visible`).count();
}

async function clickVisibleElementAt(page: Page, selector: string, index: number) {
  const element = page.locator(`${selector}:visible`).nth(index);
  return element.click({ timeout: 1_000 }).then(
    () => true,
    () => false
  );
}

async function isVisibleEnabledButton(page: Page, text: string) {
  return page
    .locator("button:visible")
    .filter({ hasText: new RegExp(`^${text}$`) })
    .first()
    .isEnabled()
    .catch(() => false);
}

async function clickVisibleEnabledButton(page: Page, text: string) {
  return page
    .locator("button:visible")
    .filter({ hasText: new RegExp(`^${text}$`) })
    .first()
    .click({ timeout: 1_000 })
    .then(
      () => true,
      () => false
    );
}

async function clickEnabledPlayButton(page: Page) {
  return page
    .locator('[data-testid="play-button"]:visible')
    .first()
    .click({ timeout: 1_000 })
    .then(
      () => true,
      () => false
    );
}

async function moveState(page: Page) {
  const [handCards, trickCards, passBadges, done, turnVisible] = await Promise.all([
    page.locator('[data-testid="player-hand"] [data-card-id]').count(),
    page.locator('[data-testid^="trick-seat"] [data-card-id]').count(),
    page.locator('[data-testid^="trick-seat"]:has-text("Pass")').count(),
    page.getByText(/Victory|Defeat/).count(),
    isPlayerTurn(page),
  ]);
  return { handCards, actionCount: trickCards + passBadges, done, turnVisible };
}

async function waitForSubmittedMove(page: Page, before: Awaited<ReturnType<typeof moveState>>) {
  const deadline = Date.now() + 3_000;
  while (Date.now() < deadline) {
    const after = await moveState(page);
    if (
      after.handCards !== before.handCards ||
      after.actionCount !== before.actionCount ||
      after.done !== before.done ||
      !after.turnVisible
    ) {
      return;
    }
    await page.waitForTimeout(25);
  }
  throw new Error("Submitted move did not produce an observable game-state update");
}

async function moveControlDiagnostics(page: Page) {
  return page.evaluate(() => {
    const isVisible = (element: Element) => {
      const rect = element.getBoundingClientRect();
      const style = window.getComputedStyle(element);
      return style.display !== "none" && style.visibility !== "hidden" && rect.width > 0 && rect.height > 0;
    };
    const buttons = Array.from(document.querySelectorAll("button")).map((button) => ({
      text: button.textContent?.trim() ?? "",
      disabled: button.disabled,
      visible: isVisible(button),
    }));
    return {
      buttons: buttons.filter((button) => button.visible).map((button) => `${button.text}${button.disabled ? " disabled" : ""}`),
      turnLabels: Array.from(document.querySelectorAll('[data-testid="turn-label"]')).map((element) => ({
        text: element.textContent?.trim() ?? "",
        visible: isVisible(element),
      })),
      comboTypeCount: Array.from(document.querySelectorAll('[data-testid^="combo-type-"]')).filter(isVisible).length,
      legalComboCount: Array.from(document.querySelectorAll('[data-testid="legal-combo"]')).filter(isVisible).length,
    };
  });
}

async function isPlayerTurn(page: Page) {
  return page.getByTestId("turn-label").isVisible().catch(() => false);
}

async function makeRandomLegalMove(page: Page, rand: () => number): Promise<"play" | "pass"> {
  if (!(await isPlayerTurn(page))) {
    throw new TurnChangedError("Turn changed before the move driver could act");
  }

  const before = await moveState(page);
  const canPass = await isVisibleEnabledButton(page, "Pass");

  const comboTypeCount = await visibleElementCount(page, '[data-testid^="combo-type-"]');

  if (canPass && (comboTypeCount === 0 || rand() < 0.35)) {
    if (await clickVisibleEnabledButton(page, "Pass")) {
      await waitForSubmittedMove(page, before);
      return "pass";
    }
    throw new TurnChangedError("Pass was enabled but the turn changed before it could be clicked");
  }

  let legalComboCount = await visibleElementCount(page, '[data-testid="legal-combo"]');
  if (legalComboCount === 0 && comboTypeCount > 0) {
    await clickVisibleElementAt(page, '[data-testid^="combo-type-"]', randomIndex(rand, comboTypeCount));
    await page.waitForTimeout(50);
    legalComboCount = await visibleElementCount(page, '[data-testid="legal-combo"]');
  }

  if (legalComboCount > 0 && await clickVisibleElementAt(page, '[data-testid="legal-combo"]', randomIndex(rand, legalComboCount))) {
    const playEnabled = await page
      .locator('[data-testid="play-button"]:visible')
      .first()
      .isEnabled({ timeout: 1_000 })
      .catch(() => false);
    if (playEnabled && await clickEnabledPlayButton(page)) {
      await waitForSubmittedMove(page, before);
      return "play";
    }
  }

  if (canPass && await clickVisibleEnabledButton(page, "Pass")) {
    await waitForSubmittedMove(page, before);
    return "pass";
  }

  const diagnostics = await moveControlDiagnostics(page);
  if (!diagnostics.turnLabels.some((label) => label.visible)) {
    throw new TurnChangedError("Turn changed before the move driver could act");
  }

  throw new Error(
    `No playable combo or pass action was available for the active player: ${JSON.stringify(
      diagnostics
    )}`
  );
}

async function driveRandomQuadMoves(pages: Page[], maxMoves: number, maxRuntimeMs = 20_000) {
  const rand = seededRandom(42);
  const moves: Array<"play" | "pass"> = [];
  const deadline = Date.now() + maxRuntimeMs;

  for (
    let attempts = 0;
    moves.length < maxMoves && attempts < maxMoves * 20 && Date.now() < deadline;
    attempts += 1
  ) {
    const gameOverCounts = await Promise.all(pages.map((page) => page.getByText(/Victory|Defeat/).count()));
    if (gameOverCounts.some((count) => count > 0)) break;

    let moved = false;
    for (const page of pages) {
      if (!(await isPlayerTurn(page))) continue;
      try {
        moves.push(await makeRandomLegalMove(page, rand));
      } catch (error) {
        if (!(error instanceof TurnChangedError)) throw error;
        await page.waitForTimeout(50);
        continue;
      }
      moved = true;
      break;
    }

    if (!moved) {
      await pages[0].waitForTimeout(50);
    }
  }

  return moves;
}

test("duo partner shows real username instead of 'Partner'", async ({ browser, request }) => {
  const u1 = unique("host");
  const u2 = unique("friend");

  // Register both players via the claim API so the backend has their usernames
  const [p1, p2] = await Promise.all([claimPlayer(request, u1), claimPlayer(request, u2)]);

  const ctx1 = await browser.newContext();
  const ctx2 = await browser.newContext();

  await ctx1.addInitScript(
    ({ id, token, name, elo }: { id: string; token: string; name: string; elo: number }) => {
      localStorage.setItem("ce_player_id", id);
      localStorage.setItem("ce_player_token", token);
      localStorage.setItem("ce_username", name);
      localStorage.setItem("ce_elo", String(elo));
    },
    { id: p1.player_id, token: p1.player_token, name: p1.username, elo: p1.elo }
  );
  await ctx2.addInitScript(
    ({ id, token, name, elo }: { id: string; token: string; name: string; elo: number }) => {
      localStorage.setItem("ce_player_id", id);
      localStorage.setItem("ce_player_token", token);
      localStorage.setItem("ce_username", name);
      localStorage.setItem("ce_elo", String(elo));
    },
    { id: p2.player_id, token: p2.player_token, name: p2.username, elo: p2.elo }
  );

  const page1 = await ctx1.newPage();
  const page2 = await ctx2.newPage();

  try {
    // Create duo room
    await page1.goto("/");
    await page1.getByRole("button", { name: /2-player/i }).click();
    await expect(page1).toHaveURL(/\/lobby/, { timeout: 10_000 });

    const roomCode = await page1.locator('[data-testid="room-code"]').textContent();
    const code = roomCode!.trim();

    // Player 2 joins
    await page2.goto("/join");
    await page2.locator('input[placeholder="XXXXXX"]').fill(code);
    await page2.getByRole("button", { name: /join game/i }).click();

    // Both reach game board
    await Promise.all([
      expect(page1).toHaveURL(/\/game/, { timeout: 20_000 }),
      expect(page2).toHaveURL(/\/game/, { timeout: 20_000 }),
    ]);
    await Promise.all([
      expect(page1.locator("[data-card-id]").first()).toBeVisible({ timeout: 15_000 }),
      expect(page2.locator("[data-card-id]").first()).toBeVisible({ timeout: 15_000 }),
    ]);

    // Player 1's board should show player 2's real username (not "Partner")
    await expect(page1.getByText(u2).last()).toBeVisible({ timeout: 10_000 });
  } finally {
    await ctx1.close();
    await ctx2.close();
  }
});

test("quad game: seeded room lets four real players make deterministic random moves", async ({ browser, request }) => {
  test.slow();

  const players = await Promise.all(
    Array.from({ length: 4 }, (_, i) => claimPlayer(request, unique(`quad-p${i + 1}`)))
  );

  const createRes = await request.post(`${API_BASE}/api/room/create`, {
    data: { mode: "quad", difficulty: "greedy", seed: 42 },
    headers: { "X-Player-Token": players[0].player_token },
  });
  expect(createRes.ok(), await createRes.text()).toBeTruthy();
  const creator = await createRes.json();

  const joiners = [];
  for (const player of players.slice(1)) {
    const joinRes = await request.post(`${API_BASE}/api/room/join/${creator.room_code}`, {
      headers: { "X-Player-Token": player.player_token },
    });
    expect(joinRes.ok(), await joinRes.text()).toBeTruthy();
    joiners.push(await joinRes.json());
  }

  const seats = [creator.seat, ...joiners.map((joiner) => joiner.seat)];
  expect([...seats].sort((a, b) => a - b)).toEqual([0, 1, 2, 3]);

  const contexts = await Promise.all(Array.from({ length: 4 }, () => browser.newContext()));
  await Promise.all(contexts.map((ctx, i) => applyPlayerIdentity(ctx, players[i])));
  const pages = await Promise.all(contexts.map((ctx) => ctx.newPage()));
  const sessions = [creator, ...joiners];

  try {
    await Promise.all(
      pages.map((page, i) =>
        page.goto(`/game?game_id=${creator.game_id}&seat=${sessions[i].seat}&token=${sessions[i].reconnect_token}`)
      )
    );

    await Promise.all(
      pages.map((page) => expect(page).toHaveURL(/\/game/, { timeout: 20_000 }))
    );
    await Promise.all(
      pages.map((page) => expect(page.locator("[data-card-id]").first()).toBeVisible({ timeout: 15_000 }))
    );

    const urlSeats = pages.map((page) => Number(new URL(page.url()).searchParams.get("seat")));
    expect([...urlSeats].sort((a, b) => a - b)).toEqual([0, 1, 2, 3]);

    for (let i = 0; i < pages.length; i++) {
      await expect(pages[i].getByText(players[i].username).last()).toBeVisible({ timeout: 10_000 });
    }

    const moves = await driveRandomQuadMoves(pages, 4);
    expect(moves.length).toBeGreaterThanOrEqual(2);
    expect(moves).toContain("play");

    const trickSignals = await Promise.all(
      pages.map(async (page) => {
        const [trickCards, passBadges, done] = await Promise.all([
          page.locator('[data-testid^="trick-seat"] [data-card-id]').count(),
          page.locator('[data-testid^="trick-seat"]:has-text("Pass")').count(),
          page.getByText(/Victory|Defeat/).count(),
        ]);
        return trickCards + passBadges + done;
      })
    );
    expect(trickSignals.reduce((total, signal) => total + signal, 0)).toBeGreaterThan(0);
  } finally {
    await Promise.all(contexts.map((ctx) => ctx.close().catch(() => {})));
  }
});

test("duo game: board advances under AFK config — requires HUMAN_TURN_TIMEOUT_S=0", { tag: "@zero-timeout" }, async ({ browser, request }) => {
  test.skip(process.env.HUMAN_TURN_TIMEOUT_S !== "0", "Needs short AFK timeout (HUMAN_TURN_TIMEOUT_S=0)");
  test.slow();

  const [p1, p2] = await Promise.all([
    claimPlayer(request, unique("full-host")),
    claimPlayer(request, unique("full-guest")),
  ]);

  const [ctx1, ctx2] = await Promise.all([browser.newContext(), browser.newContext()]);
  for (const [ctx, p] of [[ctx1, p1], [ctx2, p2]] as const) {
    await ctx.addInitScript(
      ({ id, token, name, elo }: { id: string; token: string; name: string; elo: number }) => {
        localStorage.setItem("ce_player_id", id);
        localStorage.setItem("ce_player_token", token);
        localStorage.setItem("ce_username", name);
        localStorage.setItem("ce_elo", String(elo));
      },
      { id: p.player_id, token: p.player_token, name: p.username, elo: p.elo }
    );
  }
  const [page1, page2] = await Promise.all([ctx1.newPage(), ctx2.newPage()]);

  try {
    // Create room via API with seed=42 for deterministic deal
    const roomRes = await request.post(`${API_BASE}/api/room/create`, {
      data: { mode: "duo", difficulty: "greedy", seed: 42 },
    });
    const room = await roomRes.json();

    // Host joins directly; guest joins via /join UI
    await page1.goto(`/game?game_id=${room.game_id}&seat=0&token=${room.reconnect_token}`);
    await page2.goto("/join");
    await page2.locator('input[placeholder="XXXXXX"]').fill(room.room_code);
    await page2.getByRole("button", { name: /join game/i }).click();

    await Promise.all([
      expect(page1).toHaveURL(/\/game/, { timeout: 20_000 }),
      expect(page2).toHaveURL(/\/game/, { timeout: 20_000 }),
    ]);
    await Promise.all([
      expect(page1.locator("[data-card-id]").first()).toBeVisible({ timeout: 15_000 }),
      expect(page2.locator("[data-card-id]").first()).toBeVisible({ timeout: 15_000 }),
    ]);

    // Bounded integration check: the game should advance under short-timeout config
    // without requiring a full multiplayer game to finish.
    await expect(async () => {
      const [p1Cards, p2Cards, p1Passes, p2Passes, p1Done, p2Done] = await Promise.all([
        page1.locator('[data-testid^="trick-seat"] [data-card-id]').count(),
        page2.locator('[data-testid^="trick-seat"] [data-card-id]').count(),
        page1.locator('[data-testid^="trick-seat"]:has-text("Pass")').count(),
        page2.locator('[data-testid^="trick-seat"]:has-text("Pass")').count(),
        page1.getByText(/Victory|Defeat/).count(),
        page2.getByText(/Victory|Defeat/).count(),
      ]);
      expect(p1Cards + p2Cards + p1Passes + p2Passes + p1Done + p2Done).toBeGreaterThan(0);
    }).toPass({ timeout: 30_000 });
  } finally {
    await ctx1.close();
    await ctx2.close();
  }
});

test("duo game: refreshing the game page reconnects to the same game", async ({ browser, request }) => {
  const p1 = await claimPlayer(request, unique("refresh-host"));
  const p2 = await claimPlayer(request, unique("refresh-guest"));

  const ctx1 = await browser.newContext();
  const ctx2 = await browser.newContext();

  await ctx1.addInitScript(
    ({ id, token, name, elo }: { id: string; token: string; name: string; elo: number }) => {
      localStorage.setItem("ce_player_id", id);
      localStorage.setItem("ce_player_token", token);
      localStorage.setItem("ce_username", name);
      localStorage.setItem("ce_elo", String(elo));
    },
    { id: p1.player_id, token: p1.player_token, name: p1.username, elo: p1.elo }
  );
  await ctx2.addInitScript(
    ({ id, token, name, elo }: { id: string; token: string; name: string; elo: number }) => {
      localStorage.setItem("ce_player_id", id);
      localStorage.setItem("ce_player_token", token);
      localStorage.setItem("ce_username", name);
      localStorage.setItem("ce_elo", String(elo));
    },
    { id: p2.player_id, token: p2.player_token, name: p2.username, elo: p2.elo }
  );

  const page1 = await ctx1.newPage();
  const page2 = await ctx2.newPage();

  try {
    const roomRes = await request.post(`${API_BASE}/api/room/create`, {
      data: { mode: "duo", difficulty: "greedy", seed: 42 },
      headers: { "X-Player-Token": p1.player_token },
    });
    expect(roomRes.ok()).toBeTruthy();
    const room = await roomRes.json();

    const joinRes = await request.post(`${API_BASE}/api/room/join/${room.room_code}`, {
      headers: { "X-Player-Token": p2.player_token },
    });
    expect(joinRes.ok()).toBeTruthy();
    const guest = await joinRes.json();

    await page1.goto(`/game?game_id=${room.game_id}&seat=${room.seat}&token=${room.reconnect_token}`);
    await page2.goto(`/game?game_id=${guest.game_id}&seat=${guest.seat}&token=${guest.reconnect_token}`);

    await Promise.all([
      expect(page1).toHaveURL(/\/game/, { timeout: 20_000 }),
      expect(page2).toHaveURL(/\/game/, { timeout: 20_000 }),
    ]);
    await expect(page1.locator("[data-card-id]").first()).toBeVisible({ timeout: 15_000 });

    // Refresh page1 — should reconnect to the same game
    await page1.reload();

    // Must still be on /game (not redirected to home or new game)
    await expect(page1).toHaveURL(/\/game/, { timeout: 10_000 });
    // Cards should be visible (reconnection succeeded)
    await expect(page1.locator("[data-card-id]").first()).toBeVisible({ timeout: 15_000 });
  } finally {
    await ctx1.close();
    await ctx2.close();
  }
});
