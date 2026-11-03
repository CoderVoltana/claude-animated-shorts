// Renders T = 2\pi\sqrt{L/g} with real KaTeX in headless Chromium.
// Writes one transparent PNG mask per glyph group (all pixel-aligned in the same box) so the
// video can animate the groups in one by one and make the radical sign glow on its own.
const { chromium } = require('playwright');
const katex = require('katex');
const path = require('path');
const fs = require('fs');

const OUT = path.join(__dirname, 'build', 'formula');
fs.mkdirSync(OUT, { recursive: true });

const tex = String.raw`\htmlClass{gT}{T}\htmlClass{gEq}{\,=\,}\htmlClass{g2pi}{2\pi}` +
  String.raw`\htmlClass{gRoot}{\sqrt{\htmlClass{gRad}{\htmlClass{gL}{L}/\htmlClass{gG}{g}}}}`;
const html = katex.renderToString(tex, { displayMode: false, trust: true, strict: false, output: 'html' });
const css = 'file://' + path.join(__dirname, 'node_modules/katex/dist/katex.min.css');

// group -> CSS that leaves only that group visible (white)
const groups = {
  all: '.katex{color:#fff}',
  T: '.gT{color:#fff}',
  eq: '.gEq{color:#fff}',
  twopi: '.g2pi{color:#fff}',
  root: '.gRoot{color:#fff} .gRad{color:transparent}',
  L: '.gL{color:#fff}',
  slash: '.gRad{color:#fff} .gL,.gG{color:transparent}',
  g: '.gG{color:#fff}',
};

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1400, height: 600 }, deviceScaleFactor: 1 });
  const meta = {};
  for (const [name, rule] of Object.entries(groups)) {
    const file = path.join(OUT, `_${name}.html`);
    fs.writeFileSync(file, `<!doctype html><html><head><link rel="stylesheet" href="${css}">
      <style>html,body{margin:0;background:transparent}
      #box{display:inline-block;width:max-content;padding:40px 50px;font-size:150px}
      .katex-display{margin:0}.katex{color:transparent} ${rule}</style></head>
      <body><div id="box">${html}</div></body></html>`);
    await page.goto('file://' + file);
    await page.evaluate(() => document.fonts.ready);
    const el = await page.$('#box');
    meta.box = await el.boundingBox();
    fs.unlinkSync(file);
    await el.screenshot({ path: path.join(OUT, `${name}.png`), omitBackground: true });
  }
  fs.writeFileSync(path.join(OUT, 'meta.json'), JSON.stringify({ tex, ...meta }, null, 1));
  await browser.close();
  console.log('formula masks ->', OUT);
})();
