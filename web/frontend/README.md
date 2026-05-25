# Chucking Eggs Frontend

Next.js frontend for the Guan Dan web app: solo play, multiplayer rooms, profile pages, leaderboard, and settings.

## Setup

Use Node 22 (`.nvmrc` is provided):

```bash
nvm use
npm install
npm run dev
```

Open `http://localhost:3000`. The frontend expects the FastAPI backend to be running separately unless you use `docker compose up` from the repo root.

## Common Commands

```bash
npm run dev
npm run lint
npm run build
npm run e2e
```

## Project Map

- `app/`: Next.js routes.
- `components/game/`: board, hand, controls, and in-game UI.
- `components/layout/`: app shell and navigation.
- `hooks/`: game socket and local player state.
- `lib/`: shared frontend types and card helpers.
- `e2e/`: Playwright coverage for game flow, auth, multiplayer, layout, and profile views.
