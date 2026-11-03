// Render frames of render.html with headless Chromium.
//   node render.mjs stills <outDir> <t1,t2,...>      (seconds)
//   node render.mjs frames <outDir> [workers] [from] [to]
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";
const require = createRequire(import.meta.url);
const { chromium } = require("playwright");

const ROOT = path.dirname(fileURLToPath(import.meta.url));
const TYPES = { ".html": "text/html", ".json": "application/json", ".bin": "application/octet-stream" };
const server = http.createServer((req, res) => {
  const f = path.join(ROOT, decodeURIComponent(req.url.split("?")[0]));
  if (!f.startsWith(ROOT) || !fs.existsSync(f)) { res.writeHead(404); return res.end(); }
  res.writeHead(200, { "Content-Type": TYPES[path.extname(f)] || "application/octet-stream" });
  fs.createReadStream(f).pipe(res);
});
await new Promise(r => server.listen(0, "127.0.0.1", r));
const URL_ = `http://127.0.0.1:${server.address().port}/render.html${process.env.VARIANT ? "?v=" + process.env.VARIANT : ""}`;

const [mode, outDir, a3, a4, a5] = process.argv.slice(2);
fs.mkdirSync(outDir, { recursive: true });
const browser = await chromium.launch({ args: ["--disable-gpu-vsync", "--force-color-profile=srgb"] });

async function openPage() {
  const page = await browser.newPage({ viewport: { width: 1080, height: 1920 } });
  page.on("console", m => { if (m.type() === "error") console.error("[page]", m.text()); });
  page.on("pageerror", e => console.error("[pageerror]", e.message));
  await page.goto(URL_);
  await page.waitForFunction(() => window.READY === true, null, { timeout: 120000 });
  return page;
}
async function grab(page, n, file) {
  const b64 = await page.evaluate(n => { window.renderFrame(n); return document.getElementById("c").toDataURL("image/png").split(",")[1]; }, n);
  fs.writeFileSync(file, Buffer.from(b64, "base64"));
}

if (mode === "stills") {
  const page = await openPage();
  for (const s of a3.split(",")) {
    const n = Math.round(parseFloat(s) * 60);
    await grab(page, n, path.join(outDir, `t${s.padStart(5, "0")}.png`));
    console.log("still", s);
  }
} else {
  const total = await (await openPage()).evaluate(() => window.FRAMES);
  const workers = parseInt(a3 || "4"), from = parseInt(a4 || "0"), to = parseInt(a5 || String(total));
  let next = from, done = 0; const t0 = Date.now();
  await Promise.all(Array.from({ length: workers }, async () => {
    const page = await openPage();
    while (next < to) {
      const n = next++;
      await grab(page, n, path.join(outDir, `f${String(n).padStart(5, "0")}.png`));
      if (++done % 60 === 0) console.log(`${done}/${to - from} frames, ${((Date.now() - t0) / done).toFixed(0)} ms/frame`);
    }
  }));
}
await browser.close();
server.close();
