import { test, expect } from "@playwright/test";

const API_BASE = process.env.API_URL || "http://localhost:8000";

const fakePlayer = () => {
  const suffix = `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  localStorage.setItem("ce_player_id", `test-player-${suffix}`);
  localStorage.setItem("ce_username", "testbot");
  localStorage.setItem("ce_elo", "1200");
};

test.describe.configure({ mode: "serial" });

test.beforeEach(async ({ page }) => {
  await page.addInitScript(fakePlayer);
});

async function gotoSeededSolo(page: import("@playwright/test").Page, request: import("@playwright/test").APIRequestContext) {
  const res = await request.post(`${API_BASE}/api/room/create`, {
    data: { mode: "solo", difficulty: "greedy", seed: 3 },
  });
  const room = await res.json();
  await page.goto(`/game?game_id=${room.game_id}&seat=0&token=${room.reconnect_token}`);
}

async function maybePlayIfHumanStarts(page: import("@playwright/test").Page) {
  if (!(await isHumanTurn(page))) return;
  await makeHumanMove(page);
}

async function isHumanTurn(page: import("@playwright/test").Page) {
  return page.getByTestId("turn-label").isVisible().catch(() => false);
}

async function makeHumanMove(page: import("@playwright/test").Page): Promise<"pass" | "play"> {
  await expect(page.getByTestId("turn-label")).toBeVisible({ timeout: 30_000 });
  const passButton = page.getByTestId("pass-button").first();
  const canPass = await expect(passButton).toBeEnabled({ timeout: 500 }).then(() => true).catch(() => false);
  if (canPass) {
    await passButton.click();
    return "pass";
  }

  const comboTypes = page.locator('[data-testid^="combo-type-"]');
  await expect(comboTypes.first()).toBeVisible({ timeout: 5_000 });
  const count = await comboTypes.count();
  let selectedCombo = false;
  for (let i = 0; i < count; i++) {
    await comboTypes.nth(i).click();
    const combo = page.getByTestId("legal-combo").first();
    if (await combo.isVisible({ timeout: 500 }).catch(() => false)) {
      await combo.click();
      selectedCombo = true;
      break;
    }
  }
  expect(selectedCombo).toBeTruthy();
  const playButton = page.getByTestId("play-button").first();
  await expect(playButton).toBeEnabled({ timeout: 2_000 });
  await playButton.click();
  return "play";
}

async function playFirstVisibleCombo(page: import("@playwright/test").Page) {
  await expect(page.getByTestId("turn-label")).toBeVisible({ timeout: 30_000 });
  const playButton = page.getByTestId("play-button").first();
  if (!(await playButton.isEnabled().catch(() => false))) {
    await page.locator('[data-testid="player-hand"] [data-card-id]').first().click();
  }
  await expect(playButton).toBeEnabled({ timeout: 2_000 });
  await playButton.click();
}

test("AI-played cards appear in trick zone on the board", async ({ page, request }) => {
  await gotoSeededSolo(page, request);
  await expect(page.locator("[data-card-id]").first()).toBeVisible({ timeout: 15_000 });
  await maybePlayIfHumanStarts(page);

  // Wait for any AI seat to show a card in its trick zone
  const aiTrickCard = page.locator(
    '[data-testid="trick-seat-1"] [data-card-id], ' +
    '[data-testid="trick-seat-2"] [data-card-id], ' +
    '[data-testid="trick-seat-3"] [data-card-id]'
  );
  const deadline = Date.now() + 25_000;
  while (Date.now() < deadline) {
    if (await aiTrickCard.first().isVisible().catch(() => false)) return;
    if (await isHumanTurn(page)) {
      await makeHumanMove(page);
    } else {
      await page.waitForTimeout(100);
    }
  }
  await expect(aiTrickCard.first()).toBeVisible({ timeout: 10_000 });
});

test("cards stay in trick zone after a subsequent player responds", async ({ page, request }) => {
  await gotoSeededSolo(page, request);
  await expect(page.locator("[data-card-id]").first()).toBeVisible({ timeout: 15_000 });
  await maybePlayIfHumanStarts(page);

  // Wait for any played cards in a trick zone.
  const firstPlayedCard = page.locator('[data-testid^="trick-seat"] [data-card-id]').first();
  await expect(firstPlayedCard).toBeVisible({ timeout: 25_000 });
  const firstCardId = await firstPlayedCard.getAttribute("data-card-id");

  // Wait for another visible trick action.
  try {
    await page.waitForFunction(() => {
      const zones = Array.from(document.querySelectorAll<HTMLElement>('[data-testid^="trick-seat"]'));
      return zones.filter((zone) => zone.textContent?.includes("Pass") || zone.querySelector("[data-card-id]")).length >= 2;
    }, { timeout: 8_000 });
  } catch {
    await makeHumanMove(page);
    await page.waitForFunction(() => {
      const zones = Array.from(document.querySelectorAll<HTMLElement>('[data-testid^="trick-seat"]'));
      return zones.filter((zone) => zone.textContent?.includes("Pass") || zone.querySelector("[data-card-id]")).length >= 2;
    }, { timeout: 25_000 });
  }

  // The first played card must still be visible, not wiped by the response.
  await expect(page.locator(`[data-card-id="${firstCardId}"]`).first()).toBeVisible();
});

test("pass action shows pill badge with hand icon", async ({ page, request }) => {
  await gotoSeededSolo(page, request);
  await expect(page.locator("[data-card-id]").first()).toBeVisible({ timeout: 15_000 });

  let madePass = false;
  for (let i = 0; i < 8; i++) {
    const move = await makeHumanMove(page);
    if (move === "pass") {
      madePass = true;
      break;
    }
  }
  expect(madePass).toBe(true);

  // Wait for any trick zone to show a pass badge.
  const passBadge = page.locator('[data-testid="trick-seat-0"]:has-text("Pass")');
  await expect(passBadge.first()).toBeVisible({ timeout: 30_000 });

  // Pill badge must contain the Hand SVG icon from lucide-react
  await expect(passBadge.first().locator("svg.lucide-hand")).toBeVisible();
});

test("invalid play protocol error is recoverable without desync", async ({ page, request }) => {
  await page.addInitScript(() => {
    const originalSend = WebSocket.prototype.send;
    let mutated = false;
    WebSocket.prototype.send = function patchedSend(data: string | ArrayBufferLike | Blob | ArrayBufferView) {
      if (!mutated && typeof data === "string") {
        try {
          const message = JSON.parse(data);
          if (message.type === "play_cards") {
            mutated = true;
            return originalSend.call(this, JSON.stringify({ ...message, card_ids: ["not-a-card"] }));
          }
        } catch {
          // Send the original payload if it is not JSON.
        }
      }
      return originalSend.call(this, data);
    };
  });

  await gotoSeededSolo(page, request);
  await expect(page.locator("[data-card-id]").first()).toBeVisible({ timeout: 15_000 });

  await playFirstVisibleCombo(page);
  await expect(page.getByTestId("toast").filter({ hasText: "Invalid combo" })).toBeVisible({ timeout: 5_000 });

  await playFirstVisibleCombo(page);
  await expect(page.locator('[data-testid^="trick-seat"] [data-card-id]').first()).toBeVisible({ timeout: 10_000 });
  await expect(page.getByText(/Game disconnected/i)).not.toBeVisible();
});

test("GameOver modal appears after game completes — requires HUMAN_TURN_TIMEOUT_S=0", { tag: "@zero-timeout" }, async ({ page }) => {
  test.skip(process.env.HUMAN_TURN_TIMEOUT_S !== "0", "Needs short AFK timeout — start backend with HUMAN_TURN_TIMEOUT_S=0");
  test.slow();

  await page.goto("/game?difficulty=greedy");
  await expect(page.locator("[data-card-id]").first()).toBeVisible({ timeout: 15_000 });

  // AFK auto-play handles all turns; wait for game-over modal
  await expect(page.getByText(/Victory|Defeat/)).toBeVisible({ timeout: 180_000 });
});
