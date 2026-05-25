import { test, expect, type BrowserContext, type Page } from "@playwright/test";

const API_BASE = process.env.API_URL || "http://localhost:8000";

const unique = (prefix: string) => `${prefix.slice(0, 10)}-${Math.random().toString(36).slice(2, 8)}`;

type Player = {
  player_id: string;
  player_token: string;
  username: string;
  elo: number;
};

async function claimPlayer(request: import("@playwright/test").APIRequestContext, username: string): Promise<Player> {
  const response = await request.post(`${API_BASE}/api/auth/claim`, { data: { username, is_test: true } });
  expect(response.ok(), await response.text()).toBeTruthy();
  return response.json() as Promise<Player>;
}

async function applyPlayerIdentity(ctx: BrowserContext, player: Player) {
  await ctx.addInitScript(
    ({ id, token, name, elo }: { id: string; token: string; name: string; elo: number }) => {
      localStorage.setItem("ce_player_id", id);
      localStorage.setItem("ce_player_token", token);
      localStorage.setItem("ce_username", name);
      localStorage.setItem("ce_elo", String(elo));
    },
    { id: player.player_id, token: player.player_token, name: player.username, elo: player.elo }
  );
}

async function createSignedRoom(
  request: import("@playwright/test").APIRequestContext,
  player: Player,
  mode: "duo" | "quad" = "duo",
  seed = 42
) {
  const response = await request.post(`${API_BASE}/api/room/create`, {
    data: { mode, difficulty: "greedy", seed },
    headers: { "X-Player-Token": player.player_token },
  });
  expect(response.ok(), await response.text()).toBeTruthy();
  return response.json();
}

async function joinSignedRoom(
  request: import("@playwright/test").APIRequestContext,
  roomCode: string,
  player: Player
) {
  const response = await request.post(`${API_BASE}/api/room/join/${roomCode}`, {
    headers: { "X-Player-Token": player.player_token },
  });
  expect(response.ok(), await response.text()).toBeTruthy();
  return response.json();
}

async function openGame(page: Page, session: { game_id: string; seat: number; reconnect_token: string }) {
  await page.goto(`/game?game_id=${session.game_id}&seat=${session.seat}&token=${encodeURIComponent(session.reconnect_token)}`);
  await expect(page).toHaveURL(/\/game/, { timeout: 20_000 });
  await expect(page.locator("[data-card-id]").first()).toBeVisible({ timeout: 20_000 });
}

async function createStartedDuo(
  browser: import("@playwright/test").Browser,
  request: import("@playwright/test").APIRequestContext
) {
  const [p1, p2] = await Promise.all([
    claimPlayer(request, unique("duo-a")),
    claimPlayer(request, unique("duo-b")),
  ]);
  const room = await createSignedRoom(request, p1, "duo");
  const guest = await joinSignedRoom(request, room.room_code, p2);
  const [ctx1, ctx2] = await Promise.all([browser.newContext(), browser.newContext()]);
  await Promise.all([applyPlayerIdentity(ctx1, p1), applyPlayerIdentity(ctx2, p2)]);
  const [page1, page2] = await Promise.all([ctx1.newPage(), ctx2.newPage()]);
  await Promise.all([openGame(page1, room), openGame(page2, guest)]);
  return { players: [p1, p2] as const, sessions: [room, guest] as const, contexts: [ctx1, ctx2] as const, pages: [page1, page2] as const };
}

test("wrong reconnect token shows an authorization error and no board access", async ({ browser, request }) => {
  const [p1, p2] = await Promise.all([
    claimPlayer(request, unique("wrong-a")),
    claimPlayer(request, unique("wrong-b")),
  ]);
  const room = await createSignedRoom(request, p1, "duo");
  const guest = await joinSignedRoom(request, room.room_code, p2);
  const ctx = await browser.newContext();
  await applyPlayerIdentity(ctx, p2);
  const page = await ctx.newPage();

  try {
    await page.goto(`/game?game_id=${room.game_id}&seat=${guest.seat}&token=${encodeURIComponent(room.reconnect_token)}`);
    await expect(page.getByText(/not authorized|Game disconnected/i)).toBeVisible({ timeout: 10_000 });
    await expect(page.locator("[data-card-id]").first()).not.toBeVisible();
  } finally {
    await ctx.close();
  }
});

test("disconnect and reconnect before takeover preserves seat identity", async ({ browser, request }) => {
  const { players, sessions, contexts, pages } = await createStartedDuo(browser, request);
  const [ctx1, ctx2] = contexts;
  const [, page2] = pages;

  try {
    await pages[0].close();
    const replacement = await ctx1.newPage();
    await openGame(replacement, sessions[0]);
    await expect(replacement.getByText(players[0].username).last()).toBeVisible({ timeout: 10_000 });
    await expect(page2.getByText(players[0].username).last()).toBeVisible({ timeout: 10_000 });
  } finally {
    await ctx1.close().catch(() => {});
    await ctx2.close().catch(() => {});
  }
});

test("duplicate tab for the same seat leaves the newest connection alive", async ({ browser, request }) => {
  const { contexts, sessions, pages } = await createStartedDuo(browser, request);
  const [ctx1, ctx2] = contexts;

  try {
    const newerTab = await ctx1.newPage();
    await openGame(newerTab, sessions[0]);
    await pages[0].close();
    await expect(newerTab.locator("[data-card-id]").first()).toBeVisible({ timeout: 10_000 });
    await expect(newerTab.getByText(/Game disconnected/i)).not.toBeVisible();
  } finally {
    await ctx1.close().catch(() => {});
    await ctx2.close().catch(() => {});
  }
});

test("active signed player is blocked from joining a second room", async ({ browser, request }) => {
  const { players, contexts } = await createStartedDuo(browser, request);
  const third = await claimPlayer(request, unique("other-host"));
  const otherRoom = await createSignedRoom(request, third, "duo");
  const joinPage = await contexts[0].newPage();

  try {
    await joinPage.goto("/join");
    await joinPage.locator('input[placeholder="XXXXXX"]').fill(otherRoom.room_code);
    await joinPage.getByRole("button", { name: /join game/i }).click();
    await expect(joinPage.getByText(/already_in_game/i)).toBeVisible({ timeout: 10_000 });
    await expect(joinPage).not.toHaveURL(/\/game/);
    expect(players[0].username).toBeTruthy();
  } finally {
    await Promise.all(contexts.map((ctx) => ctx.close().catch(() => {})));
  }
});

test("profile history reflects a signed forfeit game", async ({ browser, request }) => {
  const { players, contexts, pages } = await createStartedDuo(browser, request);
  const [page1] = pages;

  try {
    await page1.locator("[data-testid='game-menu-button']").click();
    await page1.getByRole("menuitem", { name: /forfeit game/i }).click();
    await page1.getByRole("button", { name: /confirm/i }).click();
    await expect(page1).toHaveURL(/\/$/, { timeout: 10_000 });

    await page1.goto(`/profile/${players[0].username}`);
    await expect(page1.getByRole("heading", { name: players[0].username })).toBeVisible({ timeout: 10_000 });
    await expect(page1.getByText(/1 game played/i)).toBeVisible();
    await expect(page1.getByText(/\-\d+/).first()).toBeVisible();
    await expect(page1.getByText(/you \+/i)).toBeVisible();
  } finally {
    await Promise.all(contexts.map((ctx) => ctx.close().catch(() => {})));
  }
});

test("rematch flow creates a new multiplayer room and both players land in it", async ({ browser, request }) => {
  const { players, sessions, contexts, pages } = await createStartedDuo(browser, request);
  const [page1, page2] = pages;
  const oldGameId = sessions[0].game_id;
  const contextsToClose: BrowserContext[] = [...contexts];

  try {
    const forfeit = await request.post(`${API_BASE}/api/room/${oldGameId}/forfeit`, {
      data: { player_id: players[0].player_id },
      headers: { "X-Player-Token": players[0].player_token },
    });
    expect(forfeit.ok(), await forfeit.text()).toBeTruthy();

    await Promise.all([
      expect(page1.getByText(/forfeited/i)).toBeVisible({ timeout: 10_000 }),
      expect(page2.getByText(/forfeited/i)).toBeVisible({ timeout: 10_000 }),
    ]);

    await Promise.all(contexts.map((ctx) => ctx.close().catch(() => {})));
    contextsToClose.length = 0;

    const rematchResponse = await request.post(`${API_BASE}/api/room/${oldGameId}/rematch`);
    expect(rematchResponse.ok(), await rematchResponse.text()).toBeTruthy();
    const rematch = await rematchResponse.json();

    const [rematchP1, rematchP2] = await Promise.all([
      joinSignedRoom(request, rematch.room_code, players[0]),
      joinSignedRoom(request, rematch.room_code, players[1]),
    ]);
    const rematchContexts = await Promise.all([browser.newContext(), browser.newContext()]);
    contextsToClose.push(...rematchContexts);
    await Promise.all([
      applyPlayerIdentity(rematchContexts[0], players[0]),
      applyPlayerIdentity(rematchContexts[1], players[1]),
    ]);
    const rematchPages = await Promise.all(rematchContexts.map((ctx) => ctx.newPage()));
    await Promise.all([openGame(rematchPages[0], rematchP1), openGame(rematchPages[1], rematchP2)]);

    await Promise.all([
      expect(rematchPages[0]).toHaveURL(new RegExp(`/game\\?game_id=${rematch.game_id}`), { timeout: 30_000 }),
      expect(rematchPages[1]).toHaveURL(new RegExp(`/game\\?game_id=${rematch.game_id}`), { timeout: 30_000 }),
    ]);
    const newGameIds = rematchPages.map((page) => new URL(page.url()).searchParams.get("game_id"));
    expect(new Set(newGameIds).size).toBe(1);
    expect(newGameIds[0]).not.toBe(oldGameId);
  } finally {
    await Promise.all(contextsToClose.map((ctx) => ctx.close().catch(() => {})));
  }
});

test("lobby expiry sends the browser home without a stale active-game banner @zero-timeout", { tag: "@zero-timeout" }, async ({ browser, request }) => {
  test.skip(process.env.LOBBY_TIMEOUT_S !== "5", "Needs short lobby timeout from the integration config");
  const player = await claimPlayer(request, unique("expiry"));
  const ctx = await browser.newContext();
  await applyPlayerIdentity(ctx, player);
  const page = await ctx.newPage();

  try {
    await page.goto("/");
    await page.getByRole("button", { name: /2-player/i }).click();
    await expect(page).toHaveURL(/\/lobby/, { timeout: 10_000 });
    const gameId = new URL(page.url()).searchParams.get("game_id");
    expect(gameId).toBeTruthy();

    await expect(page).toHaveURL(/\/$/, { timeout: 12_000 });
    await expect(page.getByText(/active game/i)).not.toBeVisible();
    await expect(async () => {
      const status = await request.get(`${API_BASE}/api/room/${gameId}/status`);
      expect(status.status()).toBe(404);
    }).toPass({ timeout: 10_000 });
  } finally {
    await ctx.close();
  }
});

test("slow full quad game with four real clients completes under AFK automation @zero-timeout @slow-full-game", { tag: ["@zero-timeout", "@slow-full-game"] }, async ({ browser, request }) => {
  test.skip(process.env.RUN_SLOW_E2E !== "true", "Set RUN_SLOW_E2E=true for nightly full-game coverage");
  test.skip(process.env.HUMAN_TURN_TIMEOUT_S !== "0", "Needs HUMAN_TURN_TIMEOUT_S=0");
  test.slow();
  test.setTimeout(300_000);

  const players = await Promise.all(Array.from({ length: 4 }, (_, i) => claimPlayer(request, unique(`full-q${i}`))));
  const room = await createSignedRoom(request, players[0], "quad", 42);
  const joiners = [];
  for (const player of players.slice(1)) {
    joiners.push(await joinSignedRoom(request, room.room_code, player));
  }

  const contexts = await Promise.all(players.map(() => browser.newContext()));
  await Promise.all(contexts.map((ctx, i) => applyPlayerIdentity(ctx, players[i])));
  const pages = await Promise.all(contexts.map((ctx) => ctx.newPage()));
  const sessions = [room, ...joiners];

  try {
    await Promise.all(pages.map((page, i) => openGame(page, sessions[i])));
    await Promise.all(pages.map((page) => expect(page.getByText(/Victory|Defeat/)).toBeVisible({ timeout: 240_000 })));
  } finally {
    await Promise.all(contexts.map((ctx) => ctx.close().catch(() => {})));
  }
});
