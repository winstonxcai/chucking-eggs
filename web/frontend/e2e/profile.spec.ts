import { test, expect } from "@playwright/test";

const API_PROFILE = "**/api/profile/**";

const baseFixture = {
  player: { _id: "abc123", username: "tiger_slayer", elo: 1350, games_played: 5 },
  games: [
    {
      played_at: "2026-04-03T06:00:00",
      players: [
        { display_name: "tiger_slayer", is_bot: false, elo_before: 1200, elo_after: 1280, team_result: "win" },
        { display_name: "Bot1", is_bot: true, elo_before: 1200, elo_after: 1150, team_result: "loss" },
      ],
    },
  ],
  elo_history: [
    { date: "4/3", elo: 1200 },
    { date: "4/3 14:00", elo: 1280 },
    { date: "4/3 15:30", elo: 1350 },
  ],
};

test("profile page renders player info", async ({ page }) => {
  await page.route(API_PROFILE, (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(baseFixture),
    })
  );

  await page.goto("/profile/tiger_slayer");

  await expect(page.getByRole("heading", { name: "tiger_slayer" })).toBeVisible();
  await expect(page.getByText("1350")).toBeVisible();
  await expect(page.getByText(/5 games/)).toBeVisible();
});

test("profile page shows Elo History chart when data is present", async ({ page }) => {
  await page.route(API_PROFILE, (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(baseFixture),
    })
  );

  await page.goto("/profile/tiger_slayer");

  await expect(page.getByText("Elo History")).toBeVisible();
  // Chart SVG should be rendered (recharts renders a <svg>)
  await expect(page.locator(".recharts-wrapper")).toBeVisible();
});

test("profile page shows no-games message when elo_history is empty", async ({ page }) => {
  await page.route(API_PROFILE, (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ ...baseFixture, games: [], elo_history: [] }),
    })
  );

  await page.goto("/profile/tiger_slayer");

  await expect(page.getByText("No games yet")).toBeVisible();
});

test("profile page shows error state for unknown player", async ({ page }) => {
  await page.route(API_PROFILE, (route) =>
    route.fulfill({
      status: 404,
      contentType: "application/json",
      body: JSON.stringify({ detail: "Player not found" }),
    })
  );

  await page.goto("/profile/ghost_user");

  await expect(page.getByText("Player not found")).toBeVisible();
});

test("profile page does not crash when elo_history is absent (regression)", async ({ page }) => {
  const errors: string[] = [];
  page.on("console", (msg) => {
    if (msg.type() === "error") errors.push(msg.text());
  });

  // Old API shape without elo_history
  const { elo_history: _omit, ...oldShape } = baseFixture;
  await page.route(API_PROFILE, (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(oldShape),
    })
  );

  await page.goto("/profile/tiger_slayer");
  await expect(page.getByRole("heading", { name: "tiger_slayer" })).toBeVisible();

  // No JS errors — specifically no "Cannot read properties of undefined" crash
  const crashErrors = errors.filter((e) => e.includes("Cannot read properties of undefined"));
  expect(crashErrors).toHaveLength(0);
});
