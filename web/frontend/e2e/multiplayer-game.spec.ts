import { test, expect } from "@playwright/test";

// Backend API base — set API_URL env var when running against production
const API_BASE = process.env.API_URL || "http://localhost:8000";

test("duo partner shows real username instead of 'Partner'", async ({ browser, request }) => {
  const ts = Date.now();
  const u1 = `host-${ts}`;
  const u2 = `friend-${ts}`;

  // Register both players via the claim API so the backend has their usernames
  const [r1, r2] = await Promise.all([
    request.post(`${API_BASE}/api/auth/claim`, { data: { username: u1 } }),
    request.post(`${API_BASE}/api/auth/claim`, { data: { username: u2 } }),
  ]);
  const p1 = await r1.json();
  const p2 = await r2.json();

  const ctx1 = await browser.newContext();
  const ctx2 = await browser.newContext();

  await ctx1.addInitScript(
    ({ id, name, elo }: { id: string; name: string; elo: number }) => {
      localStorage.setItem("ce_player_id", id);
      localStorage.setItem("ce_username", name);
      localStorage.setItem("ce_elo", String(elo));
    },
    { id: p1.player_id, name: p1.username, elo: p1.elo }
  );
  await ctx2.addInitScript(
    ({ id, name, elo }: { id: string; name: string; elo: number }) => {
      localStorage.setItem("ce_player_id", id);
      localStorage.setItem("ce_username", name);
      localStorage.setItem("ce_elo", String(elo));
    },
    { id: p2.player_id, name: p2.username, elo: p2.elo }
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
    await expect(page1.getByText(u2)).toBeVisible({ timeout: 10_000 });
  } finally {
    await ctx1.close();
    await ctx2.close();
  }
});

test("duo game: full game completes via AFK — requires HUMAN_TURN_TIMEOUT_S=5", async ({ browser, request }) => {
  test.skip(process.env.HUMAN_TURN_TIMEOUT_S !== "5", "Needs short AFK timeout (HUMAN_TURN_TIMEOUT_S=5)");
  test.slow();

  const ts = Date.now();
  const [r1, r2] = await Promise.all([
    request.post(`${API_BASE}/api/auth/claim`, { data: { username: `full-host-${ts}` } }),
    request.post(`${API_BASE}/api/auth/claim`, { data: { username: `full-guest-${ts}` } }),
  ]);
  const [p1, p2] = await Promise.all([r1.json(), r2.json()]);

  const [ctx1, ctx2] = await Promise.all([browser.newContext(), browser.newContext()]);
  for (const [ctx, p] of [[ctx1, p1], [ctx2, p2]] as const) {
    await ctx.addInitScript(
      ({ id, name, elo }: { id: string; name: string; elo: number }) => {
        localStorage.setItem("ce_player_id", id);
        localStorage.setItem("ce_username", name);
        localStorage.setItem("ce_elo", String(elo));
      },
      { id: p.player_id, name: p.username, elo: p.elo }
    );
  }
  const [page1, page2] = await Promise.all([ctx1.newPage(), ctx2.newPage()]);

  try {
    // Create room via API with seed=42 for deterministic deal
    const roomRes = await request.post(`${API_BASE}/api/room/create`, {
      data: { mode: "duo", difficulty: "easy", seed: 42 },
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

    // AFK timer auto-plays all turns — wait for game-over modal on both pages
    await Promise.all([
      expect(page1.getByText(/Victory!|Defeat/)).toBeVisible({ timeout: 300_000 }),
      expect(page2.getByText(/Victory!|Defeat/)).toBeVisible({ timeout: 300_000 }),
    ]);
  } finally {
    await ctx1.close();
    await ctx2.close();
  }
});

test("duo game: refreshing the game page reconnects to the same game", async ({ browser, request }) => {
  const ts = Date.now();
  const u1 = `refresh-host-${ts}`;
  const u2 = `refresh-guest-${ts}`;

  const [r1, r2] = await Promise.all([
    request.post(`${API_BASE}/api/auth/claim`, { data: { username: u1 } }),
    request.post(`${API_BASE}/api/auth/claim`, { data: { username: u2 } }),
  ]);
  const p1 = await r1.json();
  const p2 = await r2.json();

  const ctx1 = await browser.newContext();
  const ctx2 = await browser.newContext();

  await ctx1.addInitScript(
    ({ id, name, elo }: { id: string; name: string; elo: number }) => {
      localStorage.setItem("ce_player_id", id);
      localStorage.setItem("ce_username", name);
      localStorage.setItem("ce_elo", String(elo));
    },
    { id: p1.player_id, name: p1.username, elo: p1.elo }
  );
  await ctx2.addInitScript(
    ({ id, name, elo }: { id: string; name: string; elo: number }) => {
      localStorage.setItem("ce_player_id", id);
      localStorage.setItem("ce_username", name);
      localStorage.setItem("ce_elo", String(elo));
    },
    { id: p2.player_id, name: p2.username, elo: p2.elo }
  );

  const page1 = await ctx1.newPage();
  const page2 = await ctx2.newPage();

  try {
    // Create duo room, both reach game
    await page1.goto("/");
    await page1.getByRole("button", { name: /2-player/i }).click();
    await expect(page1).toHaveURL(/\/lobby/, { timeout: 10_000 });

    const roomCode = await page1.locator('[data-testid="room-code"]').textContent();
    await page2.goto("/join");
    await page2.locator('input[placeholder="XXXXXX"]').fill(roomCode!.trim());
    await page2.getByRole("button", { name: /join game/i }).click();

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
