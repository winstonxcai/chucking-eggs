import { test, expect } from "@playwright/test";

const fakePlayer = () => {
  localStorage.setItem("ce_player_id", "test-player-00000000-0000-0000-0000-000000000000");
  localStorage.setItem("ce_username", "testbot");
  localStorage.setItem("ce_elo", "1200");
};

test.beforeEach(async ({ page }) => {
  await page.addInitScript(fakePlayer);
});

test("AI-played cards appear in trick zone on the board", async ({ page }) => {
  await page.goto("/game?difficulty=easy");
  await expect(page.locator("[data-card-id]").first()).toBeVisible({ timeout: 15_000 });

  // Wait for any AI seat to show a card in its trick zone
  const aiTrickCard = page.locator(
    '[data-testid="trick-seat-1"] [data-card-id], ' +
    '[data-testid="trick-seat-2"] [data-card-id], ' +
    '[data-testid="trick-seat-3"] [data-card-id]'
  );
  await expect(aiTrickCard.first()).toBeVisible({ timeout: 25_000 });
});

test("cards stay in trick zone after a subsequent player responds", async ({ page }) => {
  await page.goto("/game?difficulty=easy");
  await expect(page.locator("[data-card-id]").first()).toBeVisible({ timeout: 15_000 });

  // Wait for seat 1 (left opponent) to have cards in trick zone
  const seat1Zone = page.locator('[data-testid="trick-seat-1"] [data-card-id]');
  await expect(seat1Zone.first()).toBeVisible({ timeout: 25_000 });

  // Wait for seat 2 or 3 to also respond (play or pass)
  const otherZones = page.locator(
    '[data-testid="trick-seat-2"] [data-card-id], [data-testid="trick-seat-2"]:has-text("Pass"), ' +
    '[data-testid="trick-seat-3"] [data-card-id], [data-testid="trick-seat-3"]:has-text("Pass")'
  );
  await expect(otherZones.first()).toBeVisible({ timeout: 25_000 });

  // Seat 1's cards/action must STILL be visible — not wiped by the response
  await expect(seat1Zone.first()).toBeVisible();
});

test("pass action shows pill badge with hand icon", async ({ page }) => {
  await page.goto("/game?difficulty=easy");
  await expect(page.locator("[data-card-id]").first()).toBeVisible({ timeout: 15_000 });

  // Wait for any trick zone to show a pass badge (contains "Pass" text)
  const passBadge = page.locator('[data-testid^="trick-seat"]:has-text("Pass")');
  await expect(passBadge.first()).toBeVisible({ timeout: 60_000 });

  // Pill badge must contain an SVG (the Hand icon from lucide-react)
  await expect(passBadge.first().locator("svg")).toBeVisible();
});

test("GameOver modal appears after game completes — requires HUMAN_TURN_TIMEOUT_S=5", async ({ page }) => {
  test.skip(process.env.HUMAN_TURN_TIMEOUT_S !== "5", "Needs short AFK timeout — start backend with HUMAN_TURN_TIMEOUT_S=5");
  test.slow();

  await page.goto("/game?difficulty=easy");
  await expect(page.locator("[data-card-id]").first()).toBeVisible({ timeout: 15_000 });

  // AFK auto-play handles all turns; wait for game-over modal
  await expect(page.getByText(/Victory!|Defeat/)).toBeVisible({ timeout: 180_000 });
});
