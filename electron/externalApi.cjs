const http = require("http");
const { WebSocketServer } = require("ws");
const { SCHEMA_VERSION } = require("./externalStore.cjs");

const SNAPSHOT_SLICES = new Set(["gaze", "display", "camera", "calibration", "runtime", "models", "status"]);
const LOOPBACK_HOSTS = new Set(["127.0.0.1", "localhost", "::1", "[::1]"]);

function isLoopbackHost(host) {
  return LOOPBACK_HOSTS.has(String(host || "").toLowerCase());
}

function hostnameOf(hostHeader) {
  const value = String(hostHeader || "").trim().toLowerCase();
  if (value.startsWith("[")) {
    const end = value.indexOf("]");
    return end > 0 ? value.slice(0, end + 1) : value;
  }
  const colon = value.lastIndexOf(":");
  return colon >= 0 ? value.slice(0, colon) : value;
}

/**
 * Request gate for the external API.
 *
 * - Host check (loopback binds only): rejects DNS-rebinding, where a web page on an attacker
 *   domain resolves to 127.0.0.1 and sends `Host: attacker.example`.
 * - Origin check: browsers always send `Origin` on WebSocket handshakes and cross-site fetches,
 *   and CORS does not protect WebSockets. Requests carrying an Origin are rejected unless it is
 *   in `allowedOrigins`. Native clients (curl, websocat, Python) send no Origin and are allowed.
 * - Optional token: `Authorization: Bearer <token>` or `?token=<token>` (browsers cannot set
 *   headers on WebSocket handshakes).
 */
function createRequestGate({ host, allowedOrigins = [], token = null } = {}) {
  const origins = new Set(allowedOrigins.map((origin) => String(origin).trim().replace(/\/+$/u, "")).filter(Boolean));
  const checkHost = isLoopbackHost(host);
  return (request) => {
    if (checkHost) {
      const requestHost = hostnameOf(request.headers.host);
      if (!isLoopbackHost(requestHost)) {
        return { ok: false, status: 403, error: "forbidden_host" };
      }
    }
    const origin = request.headers.origin;
    if (origin !== undefined && !origins.has(String(origin).replace(/\/+$/u, ""))) {
      return { ok: false, status: 403, error: "forbidden_origin" };
    }
    if (token) {
      const auth = String(request.headers.authorization || "");
      const url = new URL(request.url || "/", "http://127.0.0.1");
      const provided = auth.startsWith("Bearer ") ? auth.slice(7) : url.searchParams.get("token");
      if (!provided || !timingSafeEqualString(provided, token)) {
        return { ok: false, status: 401, error: "unauthorized" };
      }
    }
    return { ok: true };
  };
}

function timingSafeEqualString(a, b) {
  const crypto = require("crypto");
  const left = Buffer.from(String(a));
  const right = Buffer.from(String(b));
  return left.length === right.length && crypto.timingSafeEqual(left, right);
}

function sendJson(response, statusCode, payload) {
  const body = JSON.stringify(payload);
  response.writeHead(statusCode, {
    "content-type": "application/json; charset=utf-8",
    "cache-control": "no-store",
    "content-length": Buffer.byteLength(body)
  });
  response.end(body);
}

function createRequestHandler(store, gate = () => ({ ok: true })) {
  return (request, response) => {
    const verdict = gate(request);
    if (!verdict.ok) {
      sendJson(response, verdict.status, { error: verdict.error });
      return;
    }
    if (request.method !== "GET") {
      sendJson(response, 405, { error: "method_not_allowed" });
      return;
    }

    const url = new URL(request.url || "/", "http://127.0.0.1");
    if (url.pathname === "/health") {
      const snapshot = store.getSnapshot();
      sendJson(response, 200, {
        ok: true,
        schema_version: SCHEMA_VERSION,
        started_at: snapshot.started_at
      });
      return;
    }

    if (url.pathname === "/snapshot") {
      sendJson(response, 200, store.getSnapshot());
      return;
    }

    const sliceMatch = url.pathname.match(/^\/snapshot\/([a-z_]+)$/u);
    if (sliceMatch && SNAPSHOT_SLICES.has(sliceMatch[1])) {
      sendJson(response, 200, store.getSlice(sliceMatch[1]));
      return;
    }

    sendJson(response, 404, { error: "not_found" });
  };
}

function sendSocketJson(socket, payload) {
  if (socket.readyState === socket.OPEN) {
    socket.send(JSON.stringify(payload));
  }
}

function startExternalApiServer({ store, host, port, allowedOrigins = [], token = null, logger = console }) {
  const gate = createRequestGate({ host, allowedOrigins, token });
  if (!isLoopbackHost(host)) {
    logger.warn?.(
      `External API is bound to non-loopback host ${host}; gaze data is reachable from the network` +
        (token ? "" : " without a token (set --external-api-token)")
    );
  }
  const server = http.createServer(createRequestHandler(store, gate));
  const wss = new WebSocketServer({ noServer: true });

  wss.on("connection", (socket) => {
    sendSocketJson(socket, {
      type: "snapshot",
      snapshot: store.getSnapshot()
    });
    const unsubscribe = store.subscribe((event) => sendSocketJson(socket, event));
    socket.on("close", unsubscribe);
    socket.on("error", unsubscribe);
  });

  server.on("upgrade", (request, socket, head) => {
    const url = new URL(request.url || "/", `http://${host}`);
    if (url.pathname !== "/events") {
      socket.destroy();
      return;
    }
    const verdict = gate(request);
    if (!verdict.ok) {
      socket.write(`HTTP/1.1 ${verdict.status} ${verdict.status === 401 ? "Unauthorized" : "Forbidden"}\r\nConnection: close\r\nContent-Length: 0\r\n\r\n`);
      socket.destroy();
      return;
    }
    wss.handleUpgrade(request, socket, head, (ws) => {
      wss.emit("connection", ws, request);
    });
  });

  server.on("error", (error) => {
    logger.error(`External API failed: ${error.message}`);
  });

  server.listen(port, host, () => {
    const address = server.address();
    const actualPort = typeof address === "object" && address ? address.port : port;
    logger.log(`External API listening on http://${host}:${actualPort}`);
  });

  return {
    server,
    wss,
    close: () => {
      wss.close();
      server.close();
    }
  };
}

module.exports = {
  SNAPSHOT_SLICES,
  createRequestGate,
  createRequestHandler,
  startExternalApiServer
};
