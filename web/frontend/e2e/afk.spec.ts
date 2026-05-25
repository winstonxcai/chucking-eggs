import { test, expect } from "@playwright/test";

const API_BASE = process.env.API_URL || "http://localhost:8000";

const fakePlayer = () => {
  const suffix = `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  localStorage.setItem("ce_player_id", `test-player-${suffix}`);
  localStorage.setItem("ce_username", "testbot");
  localStorage.setItem("ce_elo", "1200");
};

test.describe.configure({ mode: "serial" });

test.beforeEach(async ({ page }) => {
  await page.addInitScript(fakePlayer);
});

async function gotoSeededSolo(page: import("@playwright/test").Page, request: import("@playwright/test").APIRequestContext) {
  const res = await request.post(`${API_BASE}/api/room/create`, {
    data: { mode: "solo", difficulty: "greedy", seed: 3 },
  });
  const room = await res.json();
  await page.goto(`/game?game_id=${room.game_id}&seat=0&token=${room.reconnect_token}`);
}

async function waitForRope(page: import("@playwright/test").Page, timeout = 30_000) {
  const deadline = Date.now() + timeout;
  const rope = page.locator("[data-testid='rope-timer']");
  while (Date.now() < deadline) {
    if (await rope.isVisible().catch(() => false)) return rope;
    const playControl = page.locator("button").filter({ hasText: /^(Select|Play)/ }).first();
    if (await playControl.isVisible().catch(() => false)) {
      await page.locator('[data-testid="player-hand"] [data-card-id]').first().click();
      const playButton = page.locator("button").filter({ hasText: /^Play/ }).first();
      if (await playButton.isEnabled().catch(() => false)) {
        await playButton.click();
      }
    }
    await page.waitForTimeout(500);
  }
  await expect(rope).toBeVisible({ timeout: 1 });
  return rope;
}

// ---------------------------------------------------------------------------
// (A) Frontend rope timer UI — uses page.clock to avoid real waiting
// ---------------------------------------------------------------------------

test("rope timer appears when it is your turn", async ({ page, request }) => {
  test.setTimeout(60_000);
  await gotoSeededSolo(page, request);

  // Wait for the game to load (cards deal)
  await expect(page.locator("[data-card-id]").first()).toBeVisible({ timeout: 15_000 });

  // Wait until the rope timer appears (it's your turn and deadline is set).
  // Give extra time — AI may play first before the human gets a turn.
  await waitForRope(page, 30_000);
});

test("rope timer turns red in last 15 seconds", async ({ page, request }) => {
  test.setTimeout(60_000);
  // Navigate first so the game is running, THEN install the fake clock.
  // Installing before navigation freezes setInterval before tick() starts,
  // which can prevent the rope from appearing.
  await gotoSeededSolo(page, request);

  await expect(page.locator("[data-card-id]").first()).toBeVisible({ timeout: 15_000 });
  await waitForRope(page, 30_000);

  // Freeze time now that the rope timer is active, then jump 76s forward
  // (14s remain → urgent / red)
  await page.clock.install({ time: Date.now() });
  await page.clock.fastForward(76_000);

  // The inner bar should have bg-team-red class
  const innerBar = page.locator("[data-testid='rope-timer'] > div");
  await expect(innerBar).toHaveClass(/bg-team-red/, { timeout: 2_000 });
});

// ---------------------------------------------------------------------------
// (B) Backend AFK auto-play — requires HUMAN_TURN_TIMEOUT_S=0 on the server
// ---------------------------------------------------------------------------

test("AFK auto-play fires after timeout — server must be started with HUMAN_TURN_TIMEOUT_S=0", { tag: "@zero-timeout" }, async ({ page, request }) => {
  // Skip unless the backend was started with the 5-second AFK override.
  // Locally: kill docker backend, then `npx playwright test e2e/afk.spec.ts` (Playwright starts fresh).
  // CI: playwright.config.ts passes HUMAN_TURN_TIMEOUT_S=0 to the backend webServer command.
  test.skip(process.env.HUMAN_TURN_TIMEOUT_S !== "0", "Needs short AFK timeout — start backend with HUMAN_TURN_TIMEOUT_S=0");
  test.slow(); // mark as potentially slow

  await gotoSeededSolo(page, request);
  await expect(page.locator("[data-card-id]").first()).toBeVisible({ timeout: 15_000 });

  // Wait for rope timer to appear (it's our turn)
  const rope = await waitForRope(page, 15_000);

  // Do NOT interact — wait for the 5-second backend timeout to trigger auto-play
  // Then the rope should disappear (no longer our turn)
  await expect(rope).not.toBeVisible({ timeout: 10_000 });
});
