import { test, expect } from "@playwright/test";

const fakePlayer = () => {
  localStorage.setItem("ce_player_id", "test-player-00000000-0000-0000-0000-000000000000");
  localStorage.setItem("ce_username", "testbot");
  localStorage.setItem("ce_elo", "1200");
};

test.beforeEach(async ({ page }) => {
  await page.addInitScript(fakePlayer);
});

// ---------------------------------------------------------------------------
// (A) Frontend rope timer UI — uses page.clock to avoid real waiting
// ---------------------------------------------------------------------------

test("rope timer appears when it is your turn", async ({ page }) => {
  await page.clock.install();
  await page.goto("/game?difficulty=easy");

  // Wait for the game to load (cards deal)
  await expect(page.locator("[data-card-id]").first()).toBeVisible({ timeout: 15_000 });

  // Wait until the rope timer appears (it's your turn and deadline is set)
  await expect(page.locator("[data-testid='rope-timer']")).toBeVisible({ timeout: 15_000 });
});

test("rope timer turns red in last 15 seconds", async ({ page }) => {
  await page.clock.install();
  await page.goto("/game?difficulty=easy");

  await expect(page.locator("[data-card-id]").first()).toBeVisible({ timeout: 15_000 });
  await expect(page.locator("[data-testid='rope-timer']")).toBeVisible({ timeout: 15_000 });

  // Fast-forward 76 seconds — 14 seconds remain, should be urgent (red)
  await page.clock.fastForward(76_000);

  // The inner bar should have bg-team-red class
  const innerBar = page.locator("[data-testid='rope-timer'] > div");
  await expect(innerBar).toHaveClass(/bg-team-red/, { timeout: 2_000 });
});

// ---------------------------------------------------------------------------
// (B) Backend AFK auto-play — requires HUMAN_TURN_TIMEOUT_S=5 on the server
// ---------------------------------------------------------------------------

test("AFK auto-play fires after timeout — server must be started with HUMAN_TURN_TIMEOUT_S=5", async ({ page }) => {
  // Skip when running against a long-timeout server (e.g. local docker with default 90s)
  // This test only passes when playwright starts the backend fresh with the 5s override.
  test.slow(); // mark as potentially slow

  await page.goto("/game?difficulty=easy");
  await expect(page.locator("[data-card-id]").first()).toBeVisible({ timeout: 15_000 });

  // Wait for rope timer to appear (it's our turn)
  const rope = page.locator("[data-testid='rope-timer']");
  await expect(rope).toBeVisible({ timeout: 15_000 });

  // Do NOT interact — wait for the 5-second backend timeout to trigger auto-play
  // Then the rope should disappear (no longer our turn)
  await expect(rope).not.toBeVisible({ timeout: 10_000 });
});
