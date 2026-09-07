/**
 * Any-language proxy demo (Node).
 *
 * Prerequisites:
 *   1. Start proxy:  python -m uvicorn agent_metering.proxy:app --port 8787
 *   2. Set OPENAI_API_KEY
 *
 *   node examples/proxy_node_example.mjs
 *
 * Uses fetch only (no npm deps). Sends X-User-Id for per-user metering.
 */

const PROXY =
  process.env.OPENAI_BASE_URL?.replace(/\/$/, "") ||
  "http://127.0.0.1:8787/proxy/openai/v1";
const API_KEY = process.env.OPENAI_API_KEY;
const USER_ID = process.env.AGENT_METERING_USER || "node_demo_user";

if (!API_KEY) {
  console.error("Set OPENAI_API_KEY");
  process.exit(1);
}

const url = `${PROXY}/chat/completions`;
const res = await fetch(url, {
  method: "POST",
  headers: {
    Authorization: `Bearer ${API_KEY}`,
    "Content-Type": "application/json",
    "X-User-Id": USER_ID,
    "X-Feature": "node_example",
  },
  body: JSON.stringify({
    model: "gpt-4o-mini",
    messages: [{ role: "user", content: "Say hi in one word." }],
  }),
});

const text = await res.text();
if (!res.ok) {
  console.error("Proxy error", res.status, text);
  process.exit(1);
}

console.log(text);
console.log(`\nMetered under X-User-Id=${USER_ID} (see agent_metering.db)`);
