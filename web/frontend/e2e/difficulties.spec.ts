import { test, expect } from "@playwright/test";

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
