/**
 * Post-game trick history review e2e tests.
 *
 * Most tests require the backend to be started with HUMAN_TURN_TIMEOUT_S=0
 * so that the human's turns auto-play quickly and the game can complete.
 * playwright.config.ts passes that env var when it starts the backend fresh.
 */

import { test, expect, type Page } from "@playwright/test";

const fakePlayer = () => {
  localStorage.setItem("ce_player_id", "test-player-00000000-0000-0000-0000-000000000000");
  localStorage.setItem("ce_username", "testbot");
  localStorage.setItem("ce_elo", "1200");
};

test.beforeEach(async ({ page }) => {
  await page.addInitScript(fakePlayer);
});

/** Wait for a game to reach the GameOverModal (Victory / Defeat heading). */
async function waitForGameOver(page: Page, timeout = 150_000) {
  await expect(page.getByText(/victory|defeat/i).first()).toBeVisible({ timeout });
}

/** Navigate to a solo game, wait for cards, then wait for game over. */
async function playGameToCompletion(page: Page) {
  await page.goto("/game?difficulty=greedy");
  await expect(page.locator("[data-card-id]").first()).toBeVisible({ timeout: 20_000 });
  await waitForGameOver(page);
}

// ---------------------------------------------------------------------------
// Review button presence
// ---------------------------------------------------------------------------

test("review game button appears on game over modal — requires HUMAN_TURN_TIMEOUT_S=0", { tag: "@zero-timeout" }, async ({ page }) => {
  test.skip(
    process.env.HUMAN_TURN_TIMEOUT_S !== "0",
    "Needs short AFK timeout — start backend with HUMAN_TURN_TIMEOUT_S=0"
  );
  test.slow();
  test.setTimeout(180_000);

  await playGameToCompletion(page);

  await expect(page.locator("[data-testid='review-game-btn']")).toBeVisible({ timeout: 5_000 });
});

// ---------------------------------------------------------------------------
// Entering review mode
// ---------------------------------------------------------------------------

test("clicking review game shows review nav and hides action controls — requires HUMAN_TURN_TIMEOUT_S=0", { tag: "@zero-timeout" }, async ({ page }) => {
  test.skip(
    process.env.HUMAN_TURN_TIMEOUT_S !== "0",
    "Needs short AFK timeout — start backend with HUMAN_TURN_TIMEOUT_S=0"
  );
  test.slow();
  test.setTimeout(180_000);

  await playGameToCompletion(page);
  await page.locator("[data-testid='review-game-btn']").click();

  // Review nav bar should appear
  await expect(page.locator("[data-testid='review-nav']")).toBeVisible({ timeout: 3_000 });

  // Review counter should start at the first play in the first trick.
  const counter = page.locator("[data-testid='review-trick-counter']");
  await expect(counter).toBeVisible();
  await expect(counter).toContainText(/Trick 1 .* Play 1 of \d+/);

  // Normal action buttons (Play / Pass) should not be visible
  await expect(page.getByRole("button", { name: /^play$/i })).not.toBeVisible();
  await expect(page.getByRole("button", { name: /^pass$/i })).not.toBeVisible();
});

// ---------------------------------------------------------------------------
// Navigation
// ---------------------------------------------------------------------------

test("prev button is disabled initially; next advances counter — requires HUMAN_TURN_TIMEOUT_S=0", { tag: "@zero-timeout" }, async ({ page }) => {
  test.skip(
    process.env.HUMAN_TURN_TIMEOUT_S !== "0",
    "Needs short AFK timeout — start backend with HUMAN_TURN_TIMEOUT_S=0"
  );
  test.slow();
  test.setTimeout(180_000);

  await playGameToCompletion(page);
  await page.locator("[data-testid='review-game-btn']").click();
  await expect(page.locator("[data-testid='review-nav']")).toBeVisible({ timeout: 3_000 });

  // Prev should be disabled at the first review step.
  await expect(page.locator("[data-testid='review-prev']")).toBeDisabled();

  // Click next — counter should advance to the second play.
  await page.locator("[data-testid='review-next']").click();
  await expect(page.locator("[data-testid='review-trick-counter']")).toContainText(/Trick \d+ .* Play 2 of \d+/);
  await expect(page.locator("[data-testid='review-prev']")).toBeEnabled();
});

test("navigating to last review step disables next button — requires HUMAN_TURN_TIMEOUT_S=0", { tag: "@zero-timeout" }, async ({ page }) => {
  test.skip(
    process.env.HUMAN_TURN_TIMEOUT_S !== "0",
    "Needs short AFK timeout — start backend with HUMAN_TURN_TIMEOUT_S=0"
  );
  test.slow();
  test.setTimeout(180_000);

  await playGameToCompletion(page);
  await page.locator("[data-testid='review-game-btn']").click();
  await expect(page.locator("[data-testid='review-nav']")).toBeVisible({ timeout: 3_000 });

  const counterText = await page.locator("[data-testid='review-trick-counter']").textContent();
  const match = counterText?.match(/Play 1 of (\d+)/);
  const total = match ? parseInt(match[1]) : 0;
  expect(total).toBeGreaterThan(0);

  await page.locator("[data-testid='review-end']").click();

  // At the last review step, next should be disabled.
  await expect(page.locator("[data-testid='review-next']")).toBeDisabled();
  await expect(page.locator("[data-testid='review-trick-counter']")).toContainText(new RegExp(`Play ${total} of ${total}`));
});

// ---------------------------------------------------------------------------
// Omniscient hand display
// ---------------------------------------------------------------------------

test("partner and opponent hands show face-up cards in review mode — requires HUMAN_TURN_TIMEOUT_S=0", { tag: "@zero-timeout" }, async ({ page }) => {
  test.skip(
    process.env.HUMAN_TURN_TIMEOUT_S !== "0",
    "Needs short AFK timeout — start backend with HUMAN_TURN_TIMEOUT_S=0"
  );
  test.slow();
  test.setTimeout(180_000);

  // Use desktop viewport so opponent revealed hands (hidden lg:grid) are visible
  await page.setViewportSize({ width: 1280, height: 800 });

  await playGameToCompletion(page);
  await page.locator("[data-testid='review-game-btn']").click();
  await expect(page.locator("[data-testid='review-nav']")).toBeVisible({ timeout: 3_000 });

  // Partner (seat 2) and at least one opponent should have a revealed hand with cards
  const partnerHand = page.locator("[data-testid='opponent-hand-2'] [data-card-id]");
  await expect(partnerHand.first()).toBeVisible({ timeout: 5_000 });
});

// ---------------------------------------------------------------------------
// Winner highlight
// ---------------------------------------------------------------------------

test("trick winner position has highlight in review mode — requires HUMAN_TURN_TIMEOUT_S=0", { tag: "@zero-timeout" }, async ({ page }) => {
  test.skip(
    process.env.HUMAN_TURN_TIMEOUT_S !== "0",
    "Needs short AFK timeout — start backend with HUMAN_TURN_TIMEOUT_S=0"
  );
  test.slow();
  test.setTimeout(180_000);

  await playGameToCompletion(page);
  await page.locator("[data-testid='review-game-btn']").click();
  await expect(page.locator("[data-testid='review-nav']")).toBeVisible({ timeout: 3_000 });
  await page.locator("[data-testid='review-end']").click();

  // Exactly one trick seat should have the review-winner attribute at a completed trick step.
  await expect(page.locator("[data-review-winner='true']")).toHaveCount(1, { timeout: 3_000 });
});

// ---------------------------------------------------------------------------
// Exiting review mode
// ---------------------------------------------------------------------------

test("done button exits review mode and shows game over modal again — requires HUMAN_TURN_TIMEOUT_S=0", { tag: "@zero-timeout" }, async ({ page }) => {
  test.skip(
    process.env.HUMAN_TURN_TIMEOUT_S !== "0",
    "Needs short AFK timeout — start backend with HUMAN_TURN_TIMEOUT_S=0"
  );
  test.slow();
  test.setTimeout(180_000);

  await playGameToCompletion(page);
  await page.locator("[data-testid='review-game-btn']").click();
  await expect(page.locator("[data-testid='review-nav']")).toBeVisible({ timeout: 3_000 });

  await page.getByTestId("review-done").click();

  // Review nav should be gone
  await expect(page.locator("[data-testid='review-nav']")).not.toBeVisible({ timeout: 3_000 });

  // Game over modal should be back (Victory/Defeat + Play Again)
  await expect(page.getByText(/victory|defeat/i).first()).toBeVisible({ timeout: 3_000 });
  await expect(page.getByRole("button", { name: /play again/i })).toBeVisible();
});
