// 논문 PDF에서 그림과 표를 잘라 PNG로 만든다. Paper Reading 노트에 원본 그림을 붙이기 위한 것.
// render-slides.mjs 와 같은 방식(vendor/pdf.js + playwright)이지만, 경로에 '#'이나 공백이 있어도 되고
// 쪽 전체가 아니라 지정한 영역만 고해상도로 잘라낸다.
//
// 사용법:
//   node scripts/paper-crop.mjs pages <pdf> <outdir>      쪽 전체를 scale 2로 렌더 (좌표 재기용)
//   node scripts/paper-crop.mjs crop <spec.json>          spec 대로 잘라내기
//
// spec.json: [{ "pdf": "...", "page": 3, "box": [x0, y0, x1, y1], "out": "img/pr2/x.png" }, ...]
// box 는 PDF point 단위(1pt = 1/72 inch), 왼쪽 위가 원점. pages 모드 png 의 픽셀 ÷ 2 가 곧 point 다.

import { chromium } from 'playwright';
import { createServer } from 'node:http';
import { createReadStream, existsSync, mkdirSync, statSync, readFileSync } from 'node:fs';
import { resolve, dirname, extname, sep } from 'node:path';

const [mode, a1, a2] = process.argv.slice(2);
if (!['pages', 'crop'].includes(mode)) {
  console.error('사용법: node scripts/paper-crop.mjs pages <pdf> <outdir> | crop <spec.json>');
  process.exit(1);
}

const ROOT = process.cwd();
const MIME = { '.mjs': 'text/javascript; charset=utf-8', '.pdf': 'application/pdf' };
const server = createServer((req, res) => {
  const rel = decodeURIComponent(req.url.split('?')[0]).replace(/^\/+/, '');
  const f = resolve(ROOT, rel);
  if (!(f === ROOT || f.startsWith(ROOT + sep)) || !existsSync(f) || !statSync(f).isFile()) return res.writeHead(404).end('nf');
  res.writeHead(200, { 'content-type': MIME[extname(f).toLowerCase()] || 'application/octet-stream' });
  createReadStream(f).pipe(res);
});
await new Promise(ok => server.listen(0, '127.0.0.1', ok));
const origin = `http://127.0.0.1:${server.address().port}`;
const url = p => origin + '/' + p.replace(/\\/g, '/').split('/').map(encodeURIComponent).join('/');

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1400, height: 900 } });
await page.route('**/__host', r => r.fulfill({ contentType: 'text/html', body: '<body style="margin:0;background:#fff"><canvas id="c"></canvas></body>' }));
await page.goto(origin + '/__host');
await page.evaluate(async origin => {
  const pdfjs = await import(origin + '/vendor/pdf.js/pdf.min.mjs');
  pdfjs.GlobalWorkerOptions.workerSrc = origin + '/vendor/pdf.js/pdf.worker.min.mjs';
  window.__pdfjs = pdfjs;
  window.__docs = {};
}, origin);

async function render(pdf, n, scale, box) {
  await page.evaluate(async ({ u, n, scale, box }) => {
    const docs = window.__docs;
    docs[u] = docs[u] || await window.__pdfjs.getDocument({ url: u }).promise;
    const p = await docs[u].getPage(n);
    const vp = p.getViewport({ scale });
    const full = document.createElement('canvas');
    full.width = vp.width; full.height = vp.height;
    const fctx = full.getContext('2d');
    fctx.fillStyle = '#fff'; fctx.fillRect(0, 0, full.width, full.height);
    await p.render({ canvasContext: fctx, viewport: vp }).promise;
    const c = document.getElementById('c');
    const [x0, y0, x1, y1] = box ? box.map(v => Math.round(v * scale)) : [0, 0, full.width, full.height];
    c.width = x1 - x0; c.height = y1 - y0;
    c.getContext('2d').drawImage(full, x0, y0, x1 - x0, y1 - y0, 0, 0, x1 - x0, y1 - y0);
  }, { u: url(pdf), n, scale, box });
}

if (mode === 'pages') {
  mkdirSync(a2, { recursive: true });
  const total = await page.evaluate(async u => {
    window.__docs[u] = window.__docs[u] || await window.__pdfjs.getDocument({ url: u }).promise;
    return window.__docs[u].numPages;
  }, url(a1));
  for (let n = 1; n <= total; n++) {
    await render(a1, n, 2, null);
    await page.locator('#c').screenshot({ path: `${a2}/p${String(n).padStart(2, '0')}.png` });
  }
  console.log(`${total} pages -> ${a2}/pNN.png (scale 2, px/2 = pt)`);
} else {
  const spec = JSON.parse(readFileSync(a1, 'utf8'));
  for (const s of spec) {
    mkdirSync(dirname(s.out), { recursive: true });
    await render(s.pdf, s.page, s.scale || 3, s.box);
    await page.locator('#c').screenshot({ path: s.out });
    console.log(s.out);
  }
}

await browser.close();
server.close();
