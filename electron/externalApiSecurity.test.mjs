import { createRequire } from "node:module";
import http from "node:http";
import { afterEach, describe, expect, it } from "vitest";

const require = createRequire(import.meta.url);
const WebSocket = require("ws");
const { createRequestGate, startExternalApiServer } = require("./externalApi.cjs");
const { GazeStateStore } = require("./externalStore.cjs");

const silent = { log() {}, warn() {}, error() {} };
let api = null;

async function start(options = {}) {
  api = startExternalApiServer({
    store: new GazeStateStore({ startedAt: "2026-11-27T00:00:00.000Z" }),
    host: "127.0.0.1",
    port: 0,
    logger: silent,
    ...options
  });
  if (!api.server.listening) {
    await new Promise((resolve) => api.server.once("listening", resolve));
  }
  return api.server.address().port;
}

afterEach(() => {
  api?.close();
  api = null;
});

function get(port, path, headers = {}) {
  return new Promise((resolve, reject) => {
    const req = http.request({ host: "127.0.0.1", port, path, headers }, (res) => {
      let body = "";
      res.on("data", (chunk) => (body += chunk));
      res.on("end", () => resolve({ status: res.statusCode, body }));
    });
    req.on("error", reject);
    req.end();
  });
}

function connect(port, path = "/events", headers = {}) {
  return new Promise((resolve) => {
    const socket = new WebSocket(`ws://127.0.0.1:${port}${path}`, { headers });
    socket.once("message", (data) => {
      socket.close();
      resolve({ ok: true, message: JSON.parse(String(data)) });
    });
    socket.once("unexpected-response", (_req, res) => resolve({ ok: false, status: res.statusCode }));
    socket.once("error", () => resolve({ ok: false, status: null }));
  });
}

describe("external API request gate", () => {
  it("allows native clients that send no Origin", async () => {
    const port = await start();
    expect((await get(port, "/health")).status).toBe(200);
    const ws = await connect(port);
    expect(ws.ok).toBe(true);
    expect(ws.message.type).toBe("snapshot");
  });

  it("rejects browser WebSocket handshakes from arbitrary web pages", async () => {
    const port = await start();
    expect(await connect(port, "/events", { Origin: "https://evil.example" })).toEqual({ ok: false, status: 403 });
    expect(await connect(port, "/events", { Origin: "null" })).toEqual({ ok: false, status: 403 });
    expect((await get(port, "/snapshot", { Origin: "https://evil.example" })).status).toBe(403);
  });

  it("allows explicitly allow-listed origins", async () => {
    const port = await start({ allowedOrigins: ["http://localhost:3000/"] });
    expect((await connect(port, "/events", { Origin: "http://localhost:3000" })).ok).toBe(true);
    expect((await connect(port, "/events", { Origin: "http://localhost:3001" })).status).toBe(403);
  });

  it("rejects DNS-rebinding Host headers on loopback binds", async () => {
    const port = await start();
    expect((await get(port, "/snapshot", { Host: `rebind.attacker.example:${port}` })).status).toBe(403);
    expect((await connect(port, "/events", { Host: `rebind.attacker.example:${port}` })).status).toBe(403);
    for (const host of [`localhost:${port}`, `127.0.0.1:${port}`, `[::1]:${port}`]) {
      expect((await get(port, "/health", { Host: host })).status).toBe(200);
    }
  });

  it("requires the token when one is configured", async () => {
    const port = await start({ token: "s3cret" });
    expect((await get(port, "/snapshot")).status).toBe(401);
    expect((await get(port, "/snapshot", { Authorization: "Bearer wrong" })).status).toBe(401);
    expect((await get(port, "/snapshot", { Authorization: "Bearer s3cret" })).status).toBe(200);
    expect((await connect(port, "/events")).status).toBe(401);
    expect((await connect(port, "/events?token=s3cret")).ok).toBe(true);
  });

  it("skips the Host check when bound to a non-loopback address", () => {
    const gate = createRequestGate({ host: "0.0.0.0" });
    expect(gate({ headers: { host: "192.168.1.20:47892" }, url: "/health" })).toEqual({ ok: true });
    expect(gate({ headers: { host: "192.168.1.20:47892", origin: "https://evil.example" }, url: "/" }).ok).toBe(false);
  });
});
