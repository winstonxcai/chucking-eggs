import { test, expect } from "@playwright/test";

test("two users can create and join a duo room and both reach the game board", async ({
  browser,
}) => {
  const ctx1 = await browser.newContext();
  const ctx2 = await browser.newContext();
  const fakePlayer = () => {
    localStorage.setItem("ce_player_id", "test-player-00000000-0000-0000-0000-000000000000");
    localStorage.setItem("ce_username", "testbot");
    localStorage.setItem("ce_elo", "1200");
  };
  await ctx1.addInitScript(fakePlayer);
  await ctx2.addInitScript(fakePlayer);
  const page1 = await ctx1.newPage();
  const page2 = await ctx2.newPage();

  try {
    // --- User 1: Create a duo room ---
    await page1.goto("/");
    await page1.getByRole("button", { name: /2-player/i }).click();

    await expect(page1).toHaveURL(/\/lobby/, { timeout: 10_000 });
    const roomCode = await page1
      .locator('[data-testid="room-code"]')
      .textContent();
    expect(roomCode?.trim()).toMatch(/^[A-Z0-9]{6}$/i);

    // --- User 2: Join the room ---
    await page2.goto("/join");
    await page2.locator('input[placeholder="XXXXXX"]').fill(roomCode!.trim());
    await page2.getByRole("button", { name: /join game/i }).click();

    // --- Both should reach /game ---
    await Promise.all([
      expect(page1).toHaveURL(/\/game/, { timeout: 20_000 }),
      expect(page2).toHaveURL(/\/game/, { timeout: 20_000 }),
    ]);

    // --- Both boards show dealt cards ---
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

test("four users can create and join a quad room and all reach the game board", async ({
  browser,
}) => {
  const ctxs = await Promise.all([
    browser.newContext(),
    browser.newContext(),
    browser.newContext(),
    browser.newContext(),
  ]);
  const fakePlayer = () => {
    localStorage.setItem("ce_player_id", "test-player-00000000-0000-0000-0000-000000000000");
    localStorage.setItem("ce_username", "testbot");
    localStorage.setItem("ce_elo", "1200");
  };
  await Promise.all(ctxs.map((ctx) => ctx.addInitScript(fakePlayer)));
  const [page1, page2, page3, page4] = await Promise.all(
    ctxs.map((ctx) => ctx.newPage())
  );

  try {
    // --- User 1: Create a quad room ---
    await page1.goto("/");
    await page1.getByRole("button", { name: /4-player/i }).click();

    await expect(page1).toHaveURL(/\/lobby/, { timeout: 10_000 });
    const roomCode = await page1
      .locator('[data-testid="room-code"]')
      .textContent();
    expect(roomCode?.trim()).toMatch(/^[A-Z0-9]{6}$/i);
    const code = roomCode!.trim();

    // --- Users 2, 3, 4: Join sequentially (server assigns seats in order) ---
    for (const page of [page2, page3, page4]) {
      await page.goto("/join");
      await page.locator('input[placeholder="XXXXXX"]').fill(code);
      await page.getByRole("button", { name: /join game/i }).click();
    }

    // --- All four should reach /game ---
    await Promise.all(
      [page1, page2, page3, page4].map((p) =>
        expect(p).toHaveURL(/\/game/, { timeout: 20_000 })
      )
    );

    // --- All four boards show dealt cards ---
    await Promise.all(
      [page1, page2, page3, page4].map((p) =>
        expect(p.locator("[data-card-id]").first()).toBeVisible({
          timeout: 15_000,
        })
      )
    );
  } finally {
    await Promise.all(ctxs.map((ctx) => ctx.close()));
  }
});
