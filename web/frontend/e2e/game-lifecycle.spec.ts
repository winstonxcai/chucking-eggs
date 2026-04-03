/**
 * Game session lifecycle e2e tests — solo disposal + multiplayer forfeit system.
 *
 * These tests require the backend to be started with HUMAN_TURN_TIMEOUT_S=5
 * for the auto-forfeit test (Test 10). The playwright.config.ts passes that
 * env var when it starts the backend fresh.
 */
import { test, expect, type Page } from "@playwright/test";

const fakePlayer = () => {
  localStorage.setItem("ce_player_id", "test-player-00000000-0000-0000-0000-000000000000");
  localStorage.setItem("ce_username", "testbot");
  localStorage.setItem("ce_elo", "1200");
};

async function waitForCards(page: Page, timeout = 20_000) {
  await expect(page.locator("[data-card-id]").first()).toBeVisible({ timeout });
}

/** Wait for saveGame to flush to localStorage before navigating away. */
async function waitForActiveGameSaved(page: Page, timeout = 5_000) {
  await page.waitForFunction(
    () => localStorage.getItem("gd_game_id") !== null,
    { timeout }
  );
}

/** Navigate home via the sidebar link (client-side nav, triggers React cleanup). */
async function goHome(page: Page) {
  await page.locator('a[href="/"]').first().click();
  await expect(page).toHaveURL(/localhost:3000\/$/, { timeout: 5_000 });
}

// ---------------------------------------------------------------------------
// Test 1: Solo game disposal
// ---------------------------------------------------------------------------
test("solo game disposal — navigating away and back starts a fresh game", async ({ page }) => {
  await page.addInitScript(fakePlayer);

  await page.goto("/game?difficulty=easy");
  await waitForCards(page);

  const firstGameId = await page.evaluate(() => sessionStorage.getItem("gd_game_id"));
  expect(firstGameId).toBeTruthy();

  // Navigate home via client-side nav (sidebar Play link) — triggers React cleanup
  await goHome(page);

  const afterNavGameId = await page.evaluate(() => sessionStorage.getItem("gd_game_id"));
  expect(afterNavGameId).toBeNull();

  // Navigate back to game — should create a fresh game (no "Game not found" error)
  await page.goto("/game?difficulty=easy");
  await waitForCards(page, 25_000);

  await expect(page.getByText(/game not found/i)).not.toBeVisible();

  const secondGameId = await page.evaluate(() => sessionStorage.getItem("gd_game_id"));
  expect(secondGameId).toBeTruthy();
  expect(secondGameId).not.toBe(firstGameId);
});

// ---------------------------------------------------------------------------
// Test 2: Active game banner appears for multiplayer
// ---------------------------------------------------------------------------
test("active game banner appears when navigating home from a multiplayer game", async ({
  browser,
}) => {
  const ctx1 = await browser.newContext();
  const ctx2 = await browser.newContext();
  await ctx1.addInitScript(fakePlayer);
  await ctx2.addInitScript(fakePlayer);
  const page1 = await ctx1.newPage();
  const page2 = await ctx2.newPage();

  try {
    await page1.goto("/");
    await page1.getByRole("button", { name: /2-player/i }).click();
    await expect(page1).toHaveURL(/\/lobby/, { timeout: 10_000 });

    const roomCode = (await page1.locator("[data-testid='room-code']").textContent())?.trim();
    expect(roomCode).toMatch(/^[A-Z0-9]{6}$/i);

    await page2.goto("/join");
    await page2.locator('input[placeholder="XXXXXX"]').fill(roomCode!);
    await page2.getByRole("button", { name: /join game/i }).click();

    await Promise.all([
      expect(page1).toHaveURL(/\/game/, { timeout: 20_000 }),
      expect(page2).toHaveURL(/\/game/, { timeout: 20_000 }),
    ]);
    await waitForCards(page1);

    // Wait for saveGame to flush to localStorage before navigating
    await waitForActiveGameSaved(page1);

    // Navigate home via sidebar link (client-side nav)
    await goHome(page1);

    await expect(page1.getByText(/active game/i)).toBeVisible({ timeout: 5_000 });
    await expect(page1.getByRole("button", { name: /rejoin/i })).toBeVisible();
    await expect(page1.getByRole("button", { name: /forfeit/i })).toBeVisible();
  } finally {
    await ctx1.close();
    await ctx2.close();
  }
});

// ---------------------------------------------------------------------------
// Test 3: Rejoin from banner
// ---------------------------------------------------------------------------
test("rejoin button on home banner navigates back to the active game", async ({ browser }) => {
  const ctx1 = await browser.newContext();
  const ctx2 = await browser.newContext();
  await ctx1.addInitScript(fakePlayer);
  await ctx2.addInitScript(fakePlayer);
  const page1 = await ctx1.newPage();
  const page2 = await ctx2.newPage();

  try {
    await page1.goto("/");
    await page1.getByRole("button", { name: /2-player/i }).click();
    await expect(page1).toHaveURL(/\/lobby/, { timeout: 10_000 });

    const roomCode = (await page1.locator("[data-testid='room-code']").textContent())?.trim();
    await page2.goto("/join");
    await page2.locator('input[placeholder="XXXXXX"]').fill(roomCode!);
    await page2.getByRole("button", { name: /join game/i }).click();

    await expect(page1).toHaveURL(/\/game/, { timeout: 20_000 });
    await waitForCards(page1);
    await waitForActiveGameSaved(page1);

    const gameId = await page1.evaluate(() => localStorage.getItem("gd_game_id"));

    await goHome(page1);
    await expect(page1.getByRole("button", { name: /rejoin/i })).toBeVisible({ timeout: 5_000 });

    await page1.getByRole("button", { name: /rejoin/i }).click();
    await expect(page1).toHaveURL(/\/game/, { timeout: 10_000 });

    // Should reconnect to the same game
    await waitForCards(page1, 15_000);
    const rejoinedGameId = await page1.evaluate(() =>
      new URLSearchParams(window.location.search).get("game_id") ||
      sessionStorage.getItem("gd_game_id")
    );
    expect(rejoinedGameId).toBe(gameId);
  } finally {
    await ctx1.close();
    await ctx2.close();
  }
});

// ---------------------------------------------------------------------------
// Test 4: Forfeit confirmation dialog
// ---------------------------------------------------------------------------
test("forfeit button shows confirmation dialog; cancel keeps the banner", async ({ browser }) => {
  const ctx1 = await browser.newContext();
  const ctx2 = await browser.newContext();
  await ctx1.addInitScript(fakePlayer);
  await ctx2.addInitScript(fakePlayer);
  const page1 = await ctx1.newPage();
  const page2 = await ctx2.newPage();

  try {
    await page1.goto("/");
    await page1.getByRole("button", { name: /2-player/i }).click();
    await expect(page1).toHaveURL(/\/lobby/, { timeout: 10_000 });

    const roomCode = (await page1.locator("[data-testid='room-code']").textContent())?.trim();
    await page2.goto("/join");
    await page2.locator('input[placeholder="XXXXXX"]').fill(roomCode!);
    await page2.getByRole("button", { name: /join game/i }).click();

    await expect(page1).toHaveURL(/\/game/, { timeout: 20_000 });
    await waitForCards(page1);
    await waitForActiveGameSaved(page1);

    await goHome(page1);
    await expect(page1.getByRole("button", { name: /forfeit/i })).toBeVisible({ timeout: 5_000 });

    // Click forfeit — confirmation inline UI should appear
    await page1.getByRole("button", { name: /forfeit/i }).click();
    await expect(page1.getByText(/forfeit\?|you will lose elo/i)).toBeVisible({ timeout: 3_000 });

    // Cancel — banner back to normal
    await page1.getByRole("button", { name: /cancel/i }).click();
    await expect(page1.getByRole("button", { name: /rejoin/i })).toBeVisible();
    await expect(page1.getByText(/forfeit\?|you will lose elo/i)).not.toBeVisible();

    // Confirm forfeit this time
    await page1.getByRole("button", { name: /forfeit/i }).click();
    await expect(page1.getByText(/forfeit\?|you will lose elo/i)).toBeVisible({ timeout: 3_000 });
    await page1.getByRole("button", { name: /confirm/i }).click();

    // Banner should disappear and localStorage cleared
    await expect(page1.getByText(/active game/i)).not.toBeVisible({ timeout: 10_000 });
    const lsGameId = await page1.evaluate(() => localStorage.getItem("gd_game_id"));
    expect(lsGameId).toBeNull();
  } finally {
    await ctx1.close();
    await ctx2.close();
  }
});

// ---------------------------------------------------------------------------
// Test 5: Forfeit broadcast to other player
// ---------------------------------------------------------------------------
test("forfeiting player broadcasts game_forfeited and other player sees end screen", async ({
  browser,
}) => {
  test.setTimeout(60_000);
  const ctx1 = await browser.newContext();
  const ctx2 = await browser.newContext();
  await ctx1.addInitScript(fakePlayer);
  await ctx2.addInitScript(fakePlayer);
  const page1 = await ctx1.newPage();
  const page2 = await ctx2.newPage();

  try {
    await page1.goto("/");
    await page1.getByRole("button", { name: /2-player/i }).click();
    await expect(page1).toHaveURL(/\/lobby/, { timeout: 10_000 });

    const roomCode = (await page1.locator("[data-testid='room-code']").textContent())?.trim();
    await page2.goto("/join");
    await page2.locator('input[placeholder="XXXXXX"]').fill(roomCode!);
    await page2.getByRole("button", { name: /join game/i }).click();

    await Promise.all([
      expect(page1).toHaveURL(/\/game/, { timeout: 20_000 }),
      expect(page2).toHaveURL(/\/game/, { timeout: 20_000 }),
    ]);
    await Promise.all([waitForCards(page1), waitForCards(page2)]);

    // Player 1 forfeits via the in-game gear menu
    const menuBtn = page1.locator("[data-testid='game-menu-button']");
    await expect(menuBtn).toBeVisible({ timeout: 5_000 });
    await menuBtn.click();
    await page1.getByRole("button", { name: /forfeit game/i }).click();
    await page1.getByRole("button", { name: /confirm/i }).click();

    // Player 2 should see the forfeit screen
    await expect(page2.getByText(/forfeited/i)).toBeVisible({ timeout: 15_000 });
    // Auto-redirect for player 2 — should go home within ~4s
    await expect(page2).toHaveURL(/localhost:3000\/$/, { timeout: 8_000 });
  } finally {
    await ctx1.close();
    await ctx2.close();
  }
});

// ---------------------------------------------------------------------------
// Test 6: Lobby abort — free leave
// ---------------------------------------------------------------------------
test("canceling in lobby dissolves the room with no active game banner", async ({ page }) => {
  await page.addInitScript(fakePlayer);

  await page.goto("/");
  await page.getByRole("button", { name: /2-player/i }).click();
  await expect(page).toHaveURL(/\/lobby/, { timeout: 10_000 });

  await page.getByRole("button", { name: /cancel/i }).click();
  await expect(page).toHaveURL(/localhost:3000\/$/, { timeout: 8_000 });

  // No active game banner (lobby was pre-start, no ELO, no saved game)
  await expect(page.getByText(/active game/i)).not.toBeVisible();

  // sessionStorage is cleared
  const ssGameId = await page.evaluate(() => sessionStorage.getItem("gd_game_id"));
  expect(ssGameId).toBeNull();
});

// ---------------------------------------------------------------------------
// Test 7: Cannot start new multiplayer game while one is active
// ---------------------------------------------------------------------------
test("2-Player and 4-Player buttons are disabled while an active multiplayer game exists", async ({
  browser,
}) => {
  const ctx1 = await browser.newContext();
  const ctx2 = await browser.newContext();
  await ctx1.addInitScript(fakePlayer);
  await ctx2.addInitScript(fakePlayer);
  const page1 = await ctx1.newPage();
  const page2 = await ctx2.newPage();

  try {
    await page1.goto("/");
    await page1.getByRole("button", { name: /2-player/i }).click();
    await expect(page1).toHaveURL(/\/lobby/, { timeout: 10_000 });

    const roomCode = (await page1.locator("[data-testid='room-code']").textContent())?.trim();
    await page2.goto("/join");
    await page2.locator('input[placeholder="XXXXXX"]').fill(roomCode!);
    await page2.getByRole("button", { name: /join game/i }).click();

    await expect(page1).toHaveURL(/\/game/, { timeout: 20_000 });
    await waitForCards(page1);
    await waitForActiveGameSaved(page1);

    // Navigate home via sidebar link — banner is visible
    await goHome(page1);
    await expect(page1.getByText(/active game/i)).toBeVisible({ timeout: 5_000 });

    // Multiplayer creation buttons should be disabled
    const duoBtn = page1.getByRole("button", { name: /2-player/i });
    const quadBtn = page1.getByRole("button", { name: /4-player/i });
    await expect(duoBtn).toBeDisabled();
    await expect(quadBtn).toBeDisabled();
  } finally {
    await ctx1.close();
    await ctx2.close();
  }
});

// ---------------------------------------------------------------------------
// Test 8: In-game forfeit via gear menu
// ---------------------------------------------------------------------------
test("in-game gear menu allows forfeit with confirmation dialog", async ({ browser }) => {
  test.setTimeout(60_000);
  const ctx1 = await browser.newContext();
  const ctx2 = await browser.newContext();
  await ctx1.addInitScript(fakePlayer);
  await ctx2.addInitScript(fakePlayer);
  const page1 = await ctx1.newPage();
  const page2 = await ctx2.newPage();

  try {
    await page1.goto("/");
    await page1.getByRole("button", { name: /2-player/i }).click();
    await expect(page1).toHaveURL(/\/lobby/, { timeout: 10_000 });

    const roomCode = (await page1.locator("[data-testid='room-code']").textContent())?.trim();
    await page2.goto("/join");
    await page2.locator('input[placeholder="XXXXXX"]').fill(roomCode!);
    await page2.getByRole("button", { name: /join game/i }).click();

    await expect(page1).toHaveURL(/\/game/, { timeout: 20_000 });
    await waitForCards(page1);

    // Open gear menu
    const menuBtn = page1.locator("[data-testid='game-menu-button']");
    await expect(menuBtn).toBeVisible({ timeout: 5_000 });
    await menuBtn.click();

    // Dropdown with Forfeit Game
    await expect(page1.getByRole("button", { name: /forfeit game/i })).toBeVisible();
    await page1.getByRole("button", { name: /forfeit game/i }).click();

    // Confirmation text
    await expect(page1.getByText(/you will lose elo/i)).toBeVisible({ timeout: 3_000 });

    // Confirm
    await page1.getByRole("button", { name: /confirm/i }).click();

    // Should redirect home and clear game
    await expect(page1).toHaveURL(/localhost:3000\/$/, { timeout: 10_000 });
    const lsGameId = await page1.evaluate(() => localStorage.getItem("gd_game_id"));
    expect(lsGameId).toBeNull();
  } finally {
    await ctx1.close();
    await ctx2.close();
  }
});

// ---------------------------------------------------------------------------
// Test 9: Gear icon hidden in solo games
// ---------------------------------------------------------------------------
test("gear/menu icon is NOT shown in solo games", async ({ page }) => {
  await page.addInitScript(fakePlayer);

  await page.goto("/game?difficulty=easy");
  await waitForCards(page);

  // Gear menu button should not exist for solo games
  await expect(page.locator("[data-testid='game-menu-button']")).not.toBeVisible();
});

// ---------------------------------------------------------------------------
// Test 10: Auto-forfeit on disconnect timeout
// ---------------------------------------------------------------------------
test("disconnected multiplayer player is auto-forfeited after timeout — requires HUMAN_TURN_TIMEOUT_S=5", async ({
  browser,
}) => {
  test.skip(
    process.env.HUMAN_TURN_TIMEOUT_S !== "5",
    "Needs short AFK timeout — start backend with HUMAN_TURN_TIMEOUT_S=5"
  );
  test.slow();
  // Auto-forfeit grace period is 60s server-side. This test takes ~65s.
  test.setTimeout(120_000);

  const ctx1 = await browser.newContext();
  const ctx2 = await browser.newContext();
  await ctx1.addInitScript(fakePlayer);
  await ctx2.addInitScript(fakePlayer);
  const page1 = await ctx1.newPage();
  const page2 = await ctx2.newPage();

  try {
    await page1.goto("/");
    await page1.getByRole("button", { name: /2-player/i }).click();
    await expect(page1).toHaveURL(/\/lobby/, { timeout: 10_000 });

    const roomCode = (await page1.locator("[data-testid='room-code']").textContent())?.trim();
    await page2.goto("/join");
    await page2.locator('input[placeholder="XXXXXX"]').fill(roomCode!);
    await page2.getByRole("button", { name: /join game/i }).click();

    await Promise.all([
      expect(page1).toHaveURL(/\/game/, { timeout: 20_000 }),
      expect(page2).toHaveURL(/\/game/, { timeout: 20_000 }),
    ]);
    await Promise.all([waitForCards(page1), waitForCards(page2)]);

    // Simulate player 1 disconnect by closing their context (closes WebSocket)
    await ctx1.close();

    // Player 2 should receive the auto-forfeit broadcast within 65s
    await expect(page2.getByText(/forfeited/i)).toBeVisible({ timeout: 65_000 });
    // Auto-redirect to home
    await expect(page2).toHaveURL(/localhost:3000\/$/, { timeout: 8_000 });
  } finally {
    try { await ctx2.close(); } catch { /* already closed */ }
  }
});
