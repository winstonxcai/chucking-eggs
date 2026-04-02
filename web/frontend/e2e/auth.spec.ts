import { test, expect } from "@playwright/test";

const API_CLAIM = "**/api/auth/claim";

test("modal visible on first visit when no localStorage", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: /welcome to chucking eggs/i })).toBeVisible();
  await expect(page.getByRole("button", { name: /start playing/i })).toBeVisible();
});

test("start playing button disabled when username is empty", async ({ page }) => {
  await page.goto("/");
  const btn = page.getByRole("button", { name: /start playing/i });
  await expect(btn).toBeDisabled();
});

test("successful claim dismisses modal and shows username in sidebar", async ({ page }) => {
  await page.route(API_CLAIM, async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ player_id: "test-uuid-1234", username: "tiger_slayer", elo: 1200 }),
    });
  });

  await page.goto("/");
  await page.fill("#username", "tiger_slayer");
  await page.getByRole("button", { name: /start playing/i }).click();

  // Modal should be gone
  await expect(page.getByRole("heading", { name: /welcome to chucking eggs/i })).not.toBeVisible();
  // Sidebar should show username
  await expect(page.getByText("tiger_slayer")).toBeVisible();
});

test("409 conflict shows inline error", async ({ page }) => {
  await page.route(API_CLAIM, async (route) => {
    await route.fulfill({
      status: 409,
      contentType: "application/json",
      body: JSON.stringify({ detail: "Username already taken." }),
    });
  });

  await page.goto("/");
  await page.fill("#username", "takenuser");
  await page.getByRole("button", { name: /start playing/i }).click();

  await expect(page.getByText(/username already taken/i)).toBeVisible();
  // Modal should still be open
  await expect(page.getByRole("button", { name: /start playing/i })).toBeVisible();
});

test("email field is optional — claim succeeds without it", async ({ page }) => {
  let requestBody: Record<string, unknown> = {};
  await page.route(API_CLAIM, async (route) => {
    requestBody = JSON.parse(route.request().postData() ?? "{}");
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ player_id: "test-uuid-5678", username: "nomail_user", elo: 1200 }),
    });
  });

  await page.goto("/");
  await page.fill("#username", "nomail_user");
  // Do NOT fill email
  await page.getByRole("button", { name: /start playing/i }).click();

  await expect(page.getByRole("heading", { name: /welcome to chucking eggs/i })).not.toBeVisible();
  // Frontend sends email: null when field is empty (not omitted)
  expect(requestBody.email).toBeFalsy();
});
