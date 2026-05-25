import { defineConfig } from "@playwright/test";

const frontendPort = 3001;
const backendPort = 8001;
const frontendUrl = `http://127.0.0.1:${frontendPort}`;
const backendUrl = `http://127.0.0.1:${backendPort}`;

export default defineConfig({
  testDir: "./e2e",
  grepInvert: /@zero-timeout/,
  timeout: 30_000,
  expect: { timeout: 10_000 },
  use: {
    baseURL: process.env.BASE_URL || frontendUrl,
    headless: true,
  },
  webServer: [
    {
      command: `NEXT_DIST_DIR=.next-e2e NEXT_PUBLIC_API_URL=${backendUrl} npm run dev -- -H 127.0.0.1 -p ${frontendPort}`,
      url: frontendUrl,
      timeout: 60_000,
    },
    {
      command: [
        "PYTHONPATH=../../ml/src:.",
        "APP_ENV=test",
        "USE_MOCK_DB=true",
        "DATA_DIR=../../data/web_backend_e2e",
        `ALLOWED_ORIGINS=${frontendUrl}`,
        "HUMAN_TURN_TIMEOUT_S=60",
        "ACTION_PAUSE_S=0",
        "AI_THINK_PAUSE_S=0",
        `uv run uvicorn app.main:app --host 127.0.0.1 --port ${backendPort}`,
      ].join(" "),
      cwd: "../backend",
      url: `${backendUrl}/api/health`,
      timeout: 30_000,
    },
  ],
});
