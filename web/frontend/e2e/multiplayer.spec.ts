import { test, expect } from "@playwright/test";

test("two users can create and join a duo room and both reach the game board", async ({
  browser,
}) => {
  // Two isolated browser contexts = two separate users
  const ctx1 = await browser.newContext();
  const ctx2 = await browser.newContext();
  const page1 = await ctx1.newPage();
  const page2 = await ctx2.newPage();

  try {
    // --- User 1: Create a room ---
    await page1.goto("/");
    await page1.getByRole("button", { name: /create room/i }).click();

    // Wait for lobby — room code must be visible
    await expect(page1).toHaveURL(/\/lobby/, { timeout: 10_000 });
    const roomCode = await page1
      .locator('[data-testid="room-code"]')
      .textContent();
    expect(roomCode?.trim()).toMatch(/^[A-Z0-9]{6}$/i);

    // --- User 2: Join the room ---
    await page2.goto("/join");
    await page2.locator('input[placeholder="XXXXXX"]').fill(roomCode!.trim());
    await page2.getByRole("button", { name: /join game/i }).click();
    // page2 may pass through /lobby quickly — just wait for /game

    // --- Both users should reach /game (page1 via lobby redirect, page2 via join→lobby→game) ---
    await Promise.all([
      expect(page1).toHaveURL(/\/game/, { timeout: 20_000 }),
      expect(page2).toHaveURL(/\/game/, { timeout: 20_000 }),
    ]);

    // --- Both game boards should show dealt cards ---
    await expect(page1.locator("[data-card-id]").first()).toBeVisible({
      timeout: 15_000,
    });
    await expect(page2.locator("[data-card-id]").first()).toBeVisible({
      timeout: 15_000,
    });
  } finally {
    await ctx1.close();
    await ctx2.close();
  }
});
