import { test, expect } from "@playwright/test";

const fakePlayer = () => {
  localStorage.setItem("ce_player_id", "test-player-00000000-0000-0000-0000-000000000000");
  localStorage.setItem("ce_username", "testbot");
  localStorage.setItem("ce_elo", "1200");
};

async function waitForCards(page: import("@playwright/test").Page, timeout = 20_000) {
  await expect(page.locator("[data-card-id]").first()).toBeVisible({ timeout });
}

// ─── Test 1: Adding/removing border-pulse animation class does not shift panels ───
// Directly tests the CSS fix: programmatically toggles animate-border-pulse-green/red
// and asserts OpponentPanel positions are stable — no dependency on game progression.

test("OpponentPanel panels do not shift when border-pulse animation class is toggled", async ({ page }) => {
  await page.addInitScript(fakePlayer);
  await page.goto("/game?difficulty=greedy");
  await waitForCards(page);

  // Baseline rect snapshot of all opponent panel divs.
  const recsBefore = await page.evaluate(() => {
    const panels = Array.from(
      document.querySelectorAll<HTMLElement>('[class*="border-l-"][class*="rounded-xl"]')
    );
    return panels.map((el) => el.getBoundingClientRect().toJSON());
  });
  expect(recsBefore.length).toBeGreaterThan(0);

  // Simulate active-player change: add the green pulse class to the first panel.
  await page.evaluate(() => {
    const panels = Array.from(
      document.querySelectorAll<HTMLElement>('[class*="border-l-"][class*="rounded-xl"]')
    );
    panels[0]?.classList.add("animate-border-pulse-green");
  });
  await page.waitForTimeout(100); // one frame for any reflow

  const recsWithPulse = await page.evaluate(() => {
    const panels = Array.from(
      document.querySelectorAll<HTMLElement>('[class*="border-l-"][class*="rounded-xl"]')
    );
    return panels.map((el) => el.getBoundingClientRect().toJSON());
  });

  // Remove the class (player turn ends).
  await page.evaluate(() => {
    const panels = Array.from(
      document.querySelectorAll<HTMLElement>('[class*="border-l-"][class*="rounded-xl"]')
    );
    panels[0]?.classList.remove("animate-border-pulse-green");
  });
  await page.waitForTimeout(100);

  const recsWithoutPulse = await page.evaluate(() => {
    const panels = Array.from(
      document.querySelectorAll<HTMLElement>('[class*="border-l-"][class*="rounded-xl"]')
    );
    return panels.map((el) => el.getBoundingClientRect().toJSON());
  });

  // No panel should shift in either direction (1px rounding tolerance).
  for (let i = 0; i < recsBefore.length; i++) {
    expect(Math.abs(recsWithPulse[i].top    - recsBefore[i].top)).toBeLessThanOrEqual(1);
    expect(Math.abs(recsWithPulse[i].left   - recsBefore[i].left)).toBeLessThanOrEqual(1);
    expect(Math.abs(recsWithPulse[i].width  - recsBefore[i].width)).toBeLessThanOrEqual(1);
    expect(Math.abs(recsWithPulse[i].height - recsBefore[i].height)).toBeLessThanOrEqual(1);

    expect(Math.abs(recsWithoutPulse[i].top    - recsBefore[i].top)).toBeLessThanOrEqual(1);
    expect(Math.abs(recsWithoutPulse[i].left   - recsBefore[i].left)).toBeLessThanOrEqual(1);
    expect(Math.abs(recsWithoutPulse[i].width  - recsBefore[i].width)).toBeLessThanOrEqual(1);
    expect(Math.abs(recsWithoutPulse[i].height - recsBefore[i].height)).toBeLessThanOrEqual(1);
  }
});

// ─── Test 2: GameControls Play/Pass buttons do not shift when Unselect button appears ─

test("GameControls Play and Pass buttons do not shift when a card is selected or deselected", async ({ page }) => {
  await page.addInitScript(fakePlayer);
  await page.goto("/game?difficulty=greedy");
  await waitForCards(page);

  // Wait until it is the human player's turn so the controls row is visible.
  await page.waitForFunction(() => {
    const btns = Array.from(document.querySelectorAll<HTMLButtonElement>("button"));
    return btns.some((b) => /^(Select|Play)/.test(b.textContent?.trim() ?? "") && b.offsetParent !== null);
  }, { timeout: 30_000 });

  // Baseline: record Play and Pass button rects before any card selection.
  const recsBefore = await page.evaluate(() => {
    const btns = Array.from(document.querySelectorAll<HTMLButtonElement>("button"));
    const playBtn = btns.find(
      (b) => /^(Select|Play)/.test(b.textContent?.trim() ?? "")
    );
    const passBtn = btns.find((b) => b.textContent?.trim() === "Pass");
    if (!playBtn || !passBtn) return null;
    return {
      play: playBtn.getBoundingClientRect().toJSON(),
      pass: passBtn.getBoundingClientRect().toJSON(),
    };
  });
  expect(recsBefore).not.toBeNull();

  // Select the first card in hand — this makes hasSelection=true, showing the Unselect button.
  await page.locator("[data-card-id]").first().click();

  const recsAfterSelect = await page.evaluate(() => {
    const btns = Array.from(document.querySelectorAll<HTMLButtonElement>("button"));
    const playBtn = btns.find(
      (b) => /^(Select|Play)/.test(b.textContent?.trim() ?? "")
    );
    const passBtn = btns.find((b) => b.textContent?.trim() === "Pass");
    if (!playBtn || !passBtn) return null;
    return {
      play: playBtn.getBoundingClientRect().toJSON(),
      pass: passBtn.getBoundingClientRect().toJSON(),
    };
  });
  expect(recsAfterSelect).not.toBeNull();

  // Play and Pass must not have shifted.
  expect(Math.abs(recsAfterSelect!.play.top  - recsBefore!.play.top)).toBeLessThanOrEqual(1);
  expect(Math.abs(recsAfterSelect!.play.left - recsBefore!.play.left)).toBeLessThanOrEqual(1);
  expect(Math.abs(recsAfterSelect!.pass.top  - recsBefore!.pass.top)).toBeLessThanOrEqual(1);
  expect(Math.abs(recsAfterSelect!.pass.left - recsBefore!.pass.left)).toBeLessThanOrEqual(1);

  // Deselect the card — Unselect button goes invisible again.
  await page.locator("[data-card-id]").first().click();

  const recsAfterDeselect = await page.evaluate(() => {
    const btns = Array.from(document.querySelectorAll<HTMLButtonElement>("button"));
    const playBtn = btns.find(
      (b) => /^(Select|Play)/.test(b.textContent?.trim() ?? "")
    );
    const passBtn = btns.find((b) => b.textContent?.trim() === "Pass");
    if (!playBtn || !passBtn) return null;
    return {
      play: playBtn.getBoundingClientRect().toJSON(),
      pass: passBtn.getBoundingClientRect().toJSON(),
    };
  });
  expect(recsAfterDeselect).not.toBeNull();

  // Positions must return to baseline.
  expect(Math.abs(recsAfterDeselect!.play.top  - recsBefore!.play.top)).toBeLessThanOrEqual(1);
  expect(Math.abs(recsAfterDeselect!.play.left - recsBefore!.play.left)).toBeLessThanOrEqual(1);
  expect(Math.abs(recsAfterDeselect!.pass.top  - recsBefore!.pass.top)).toBeLessThanOrEqual(1);
  expect(Math.abs(recsAfterDeselect!.pass.left - recsBefore!.pass.left)).toBeLessThanOrEqual(1);
});
