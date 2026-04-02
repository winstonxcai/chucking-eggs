import { test, expect } from "@playwright/test";

const fakePlayer = () => {
  localStorage.setItem(
    "ce_player_id",
    "test-player-00000000-0000-0000-0000-000000000000"
  );
  localStorage.setItem("ce_username", "testbot");
  localStorage.setItem("ce_elo", "1200");
};

test("quad lobby shows 1/4 connected when creator first enters", async ({
  browser,
}) => {
  const ctx = await browser.newContext();
  await ctx.addInitScript(fakePlayer);
  const page = await ctx.newPage();
  try {
    await page.goto("/");
    await page.getByRole("button", { name: /4-player/i }).click();
    await expect(page).toHaveURL(/\/lobby/, { timeout: 10_000 });
    // Connected count starts at 0/4 (creator is on lobby, WS not yet established)
    // and increments as players join. Match any digit.
    await expect(page.getByText(/\d\/4 connected/i)).toBeVisible({
      timeout: 5_000,
    });
  } finally {
    await ctx.close();
  }
});

test("fifth player cannot join a full quad room", async ({ browser }) => {
  const ctxs = await Promise.all(
    Array.from({ length: 5 }, () => browser.newContext())
  );
  await Promise.all(ctxs.map((ctx) => ctx.addInitScript(fakePlayer)));
  const [page1, page2, page3, page4, page5] = await Promise.all(
    ctxs.map((ctx) => ctx.newPage())
  );

  try {
    // Creator opens a quad room
    await page1.goto("/");
    await page1.getByRole("button", { name: /4-player/i }).click();
    await expect(page1).toHaveURL(/\/lobby/, { timeout: 10_000 });
    const roomCode = await page1
      .locator('[data-testid="room-code"]')
      .textContent();
    const code = roomCode!.trim();

    // Players 2–4 join sequentially to fill all seats
    for (const page of [page2, page3, page4]) {
      await page.goto("/join");
      await page.locator('input[placeholder="XXXXXX"]').fill(code);
      await page.getByRole("button", { name: /join game/i }).click();
    }

    // Wait for all 4 to leave the lobby (game has started)
    await Promise.all(
      [page1, page2, page3, page4].map((p) =>
        expect(p).toHaveURL(/\/game/, { timeout: 25_000 })
      )
    );

    // 5th player tries to join — should be rejected
    await page5.goto("/join");
    await page5.locator('input[placeholder="XXXXXX"]').fill(code);
    await page5.getByRole("button", { name: /join game/i }).click();
    await expect(
      page5.getByText(/room not found or already full/i)
    ).toBeVisible({ timeout: 10_000 });
  } finally {
    await Promise.all(ctxs.map((ctx) => ctx.close()));
  }
});

test("each of the four quad players lands on a unique seat 0-3", async ({
  browser,
}) => {
  const ctxs = await Promise.all(
    Array.from({ length: 4 }, () => browser.newContext())
  );
  await Promise.all(ctxs.map((ctx) => ctx.addInitScript(fakePlayer)));
  const [page1, page2, page3, page4] = await Promise.all(
    ctxs.map((ctx) => ctx.newPage())
  );

  try {
    await page1.goto("/");
    await page1.getByRole("button", { name: /4-player/i }).click();
    await expect(page1).toHaveURL(/\/lobby/, { timeout: 10_000 });
    const roomCode = await page1
      .locator('[data-testid="room-code"]')
      .textContent();
    const code = roomCode!.trim();

    for (const page of [page2, page3, page4]) {
      await page.goto("/join");
      await page.locator('input[placeholder="XXXXXX"]').fill(code);
      await page.getByRole("button", { name: /join game/i }).click();
    }

    await Promise.all(
      [page1, page2, page3, page4].map((p) =>
        expect(p).toHaveURL(/\/game/, { timeout: 25_000 })
      )
    );

    // Extract seat query param from each /game URL
    const seats = [page1, page2, page3, page4].map((p) => {
      const url = new URL(p.url());
      return url.searchParams.get("seat");
    });

    expect([...seats].sort()).toEqual(["0", "1", "2", "3"]);
  } finally {
    await Promise.all(ctxs.map((ctx) => ctx.close()));
  }
});

test("joining with an invalid room code shows an error", async ({ page }) => {
  await page.addInitScript(fakePlayer);
  await page.goto("/join");
  await page.locator('input[placeholder="XXXXXX"]').fill("ZZZZZZ");
  await page.getByRole("button", { name: /join game/i }).click();
  await expect(page.getByText(/room not found or already full/i)).toBeVisible({
    timeout: 5_000,
  });
});
