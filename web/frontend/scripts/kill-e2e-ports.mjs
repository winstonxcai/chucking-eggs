import { execFileSync } from "node:child_process";

const ports = [3001, 8001, 3010, 8010];
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function pidsForPort(port) {
  try {
    return execFileSync("lsof", ["-ti", `tcp:${port}`, "-sTCP:LISTEN"], {
      encoding: "utf8",
      stdio: ["ignore", "pipe", "ignore"],
    })
      .split(/\s+/)
      .map((pid) => Number(pid))
      .filter(Number.isInteger);
  } catch {
    return [];
  }
}

async function waitForPort(port, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (pidsForPort(port).length === 0) return true;
    await sleep(100);
  }
  return pidsForPort(port).length === 0;
}

for (const port of ports) {
  const pids = pidsForPort(port);

  for (const pid of pids) {
    try {
      process.kill(pid, "SIGTERM");
      console.log(`Stopped stale E2E server on port ${port} (pid ${pid})`);
    } catch {
      // Process may have exited between lsof and kill.
    }
  }
  if (pids.length === 0 || await waitForPort(port, 2_000)) continue;

  for (const pid of pidsForPort(port)) {
    try {
      process.kill(pid, "SIGKILL");
      console.log(`Force-stopped stale E2E server on port ${port} (pid ${pid})`);
    } catch {
      // Process may have exited between lsof and kill.
    }
  }
  if (!(await waitForPort(port, 2_000))) {
    throw new Error(`Port ${port} is still in use after cleanup`);
  }
}
