// End-to-end OPFS KappaStore test in a real browser (Chromium via Playwright).
// Serves the harness over http://127.0.0.1 (a secure context, required for OPFS), then:
//   1) put bytes → κ; assert κ == address(bytes) (σ-axis), get(κ) round-trips the bytes;
//   2) reload the page → get(κ) still returns the bytes (OPFS persistence per-origin);
//   3) get(absent κ) → null (eviction-tolerant).
import http from "node:http";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { chromium } from "playwright";

const ROOT = path.dirname(fileURLToPath(import.meta.url));
const TYPES = { ".html": "text/html", ".js": "text/javascript", ".wasm": "application/wasm" };

const server = http.createServer(async (req, res) => {
  const rel = req.url === "/" ? "/index.html" : req.url.split("?")[0];
  try {
    const body = await readFile(path.join(ROOT, rel));
    res.writeHead(200, { "content-type": TYPES[path.extname(rel)] || "application/octet-stream" });
    res.end(body);
  } catch {
    res.writeHead(404).end("not found");
  }
});

function fail(msg) {
  console.error("OPFS-TEST: FAIL —", msg);
  process.exitCode = 1;
}

await new Promise((r) => server.listen(0, "127.0.0.1", r));
const port = server.address().port;
const url = `http://127.0.0.1:${port}/index.html`;

const browser = await chromium.launch();
const ctx = await browser.newContext();
const page = await ctx.newPage();
page.on("console", (m) => console.log("  [page]", m.text()));
page.on("pageerror", (e) => fail("pageerror: " + e.message));

try {
  await page.goto(url);
  await page.waitForFunction("window.__ready === true", null, { timeout: 15000 });

  // 1) put → κ; address() agreement; get round-trip.
  const r1 = await page.evaluate(async () => {
    const enc = new TextEncoder();
    const dec = new TextDecoder();
    const bytes = enc.encode("hello-from-opfs-in-a-real-browser");
    const k = await window.hg.opfs_put(bytes);
    const addr = window.hg.address(bytes);
    const got = await window.hg.opfs_get(k);
    return { k, addr, got: got ? dec.decode(got) : null };
  });
  if (r1.k !== r1.addr) fail(`κ mismatch: put=${r1.k} address=${r1.addr}`);
  else if (r1.got !== "hello-from-opfs-in-a-real-browser") fail(`round-trip got ${r1.got}`);
  else console.log("OPFS-TEST: put/get round-trip + σ-axis address OK; κ =", r1.k);

  // 2) reload → persistence (OPFS is per-origin, survives the reload).
  await page.reload();
  await page.waitForFunction("window.__ready === true", null, { timeout: 15000 });
  const r2 = await page.evaluate(async (k) => {
    const got = await window.hg.opfs_get(k);
    return got ? new TextDecoder().decode(got) : null;
  }, r1.k);
  if (r2 !== "hello-from-opfs-in-a-real-browser") fail(`after reload got ${r2}`);
  else console.log("OPFS-TEST: persisted across reload OK");

  // 3) absent κ → null.
  const r3 = await page.evaluate(async () => {
    const k = window.hg.address(new TextEncoder().encode("never-stored-opfs"));
    return await window.hg.opfs_get(k);
  });
  if (r3 !== null) fail("absent κ did not return null");
  else console.log("OPFS-TEST: absent κ → null OK");

  if (!process.exitCode) console.log("OPFS-TEST: PASS");
} catch (e) {
  fail(String(e));
} finally {
  await browser.close();
  server.close();
}
