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
    const playControl = page.getByTestId("play-button").first();
    if (await playControl.isVisible().catch(() => false)) {
      await page.locator('[data-testid="player-hand"] [data-card-id]').first().click({ timeout: 1_000 }).catch(() => {});
      const playButton = page.getByTestId("play-button").first();
      if (await playButton.isEnabled().catch(() => false)) {
        await playButton.click({ timeout: 1_000 }).catch(() => {});
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
  test.setTimeout(60_000);

  await gotoSeededSolo(page, request);
  await expect(page.locator("[data-card-id]").first()).toBeVisible({ timeout: 15_000 });

  // Seed 3 starts on the human seat. With HUMAN_TURN_TIMEOUT_S=0 the backend
  // may auto-play before Playwright can reliably click or observe the rope, so
  // assert the user-visible auto-play notification instead of driving the turn.
  await expect(page.getByTestId("toast").filter({ hasText: /Time's up/ })).toBeVisible({ timeout: 15_000 });
  await expect(page.getByTestId("rope-timer")).not.toBeVisible({ timeout: 15_000 });
});
