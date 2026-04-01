import { test, expect } from "@playwright/test";

// Inject a fake player into localStorage so the AppShell modal doesn't block navigation
test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    localStorage.setItem("ce_player_id", "test-player-00000000-0000-0000-0000-000000000000");
    localStorage.setItem("ce_username", "testbot");
    localStorage.setItem("ce_elo", "1200");
  });
});

// All 12 difficulty tiers ordered by calibrated Glicko-2 ELO (see runs/wr_matrix_v2/)
const difficulties = [
  "wjsd",
  "liuzha",
  "hulalala",
  "easy",
  "medium",
  "competition",
  "casual",
  "hard",
  "master",
  "yaoji",
  "jidan",
  "expert",
];

for (const difficulty of difficulties) {
  test(`loads solo game: ${difficulty}`, async ({ page }) => {
    await page.goto(`/game?difficulty=${difficulty}`);
    await expect(page.locator("[data-card-id]").first()).toBeVisible({
      timeout: 15000,
    });
  });
}
