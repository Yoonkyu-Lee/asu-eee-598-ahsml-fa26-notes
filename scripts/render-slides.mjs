// 슬라이드 PDF의 쪽을 PNG로 렌더한다. 도해를 눈으로 확인하기 위한 것.
// poppler(pdftoppm)가 이 머신에 없어서 vendor/pdf.js + playwright 로 대신한다.
//
// 사용법: node scripts/render-slides.mjs slides/L01-course-intro.pdf 12-18
//         node scripts/render-slides.mjs slides/L01-course-intro.pdf        (전체)
// 결과:  shots/slides/<stem>-p12.png ...

import { chromium } from 'playwright';
import { createServer } from 'node:http';
import { createReadStream, existsSync, mkdirSync, statSync } from 'node:fs';
import { resolve, basename, extname, sep } from 'node:path';

const file = process.argv[2];
const range = process.argv[3];
if (!file) { console.error('사용법: node scripts/render-slides.mjs <pdf> [12-18]'); process.exit(1); }

const ROOT = process.cwd();
const MIME = { '.html': 'text/html; charset=utf-8', '.mjs': 'text/javascript; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.pdf': 'application/pdf' };
// 렌더 대상 페이지. pdf.js 는 모듈 import 를 쓰므로 file:// 이 아니라 http 로 열어야 한다.
const server = createServer((req, res) => {
  const rel = decodeURIComponent(req.url.split('?')[0]).replace(/^\/+/, '');
  const f = resolve(ROOT, rel);
  if (!(f === ROOT || f.startsWith(ROOT + sep)) || !existsSync(f) || !statSync(f).isFile()) return res.writeHead(404).end('nf');
  res.writeHead(200, { 'content-type': MIME[extname(f).toLowerCase()] || 'application/octet-stream' });
  createReadStream(f).pipe(res);
});
await new Promise(ok => server.listen(0, '127.0.0.1', ok));
const origin = `http://127.0.0.1:${server.address().port}`;

const stem = basename(file).replace(/\.pdf$/i, '');
mkdirSync('shots/slides', { recursive: true });

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1400, height: 900 } });
await page.route('**/__host', r => r.fulfill({ contentType: 'text/html', body: '<canvas id="c"></canvas>' }));
await page.goto(origin + '/__host');

const total = await page.evaluate(async ({ origin, file }) => {
  const pdfjs = await import(origin + '/vendor/pdf.js/pdf.min.mjs');
  pdfjs.GlobalWorkerOptions.workerSrc = origin + '/vendor/pdf.js/pdf.worker.min.mjs';
  window.__doc = await pdfjs.getDocument({ url: origin + '/' + file }).promise;
  return window.__doc.numPages;
}, { origin, file: file.replace(/\\/g, '/') });

let pages = [];
if (range) {
  const [a, b] = range.split('-').map(Number);
  for (let i = a; i <= (b || a); i++) pages.push(i);
} else {
  for (let i = 1; i <= total; i++) pages.push(i);
}

for (const n of pages) {
  await page.evaluate(async n => {
    const p = await window.__doc.getPage(n);
    const vp = p.getViewport({ scale: 1.6 });
    const c = document.getElementById('c');
    c.width = vp.width; c.height = vp.height;
    await p.render({ canvasContext: c.getContext('2d'), viewport: vp }).promise;
  }, n);
  await page.locator('#c').screenshot({ path: `shots/slides/${stem}-p${String(n).padStart(2, '0')}.png` });
}

console.log(`${pages.length} pages -> shots/slides/${stem}-pNN.png  (총 ${total}쪽)`);
await browser.close();
server.close();
