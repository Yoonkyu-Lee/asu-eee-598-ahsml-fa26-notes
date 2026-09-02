// 스크린샷의 일부만 잘라 본다. 전체 페이지 png 은 너무 길어서 한눈에 안 들어온다.
// 사용법: node scripts/crop.mjs shots/index-desktop.png 0 0 1280 900 [out.png]
import { chromium } from 'playwright';
import { readFileSync } from 'node:fs';
const [file, x, y, w, h, out] = process.argv.slice(2);
if (!file) { console.error('사용법: node scripts/crop.mjs <png> <x> <y> <w> <h> [out]'); process.exit(1); }
const b64 = readFileSync(file).toString('base64');
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: +w, height: +h } });
await page.setContent('<body style="margin:0"><canvas id="c"></canvas></body>');
await page.evaluate(async ({ b64, x, y, w, h }) => {
  const img = new Image();
  img.src = 'data:image/png;base64,' + b64;
  await img.decode();
  const c = document.getElementById('c');
  c.width = w; c.height = h;
  c.getContext('2d').drawImage(img, x, y, w, h, 0, 0, w, h);
}, { b64, x: +x, y: +y, w: +w, h: +h });
const dst = out || 'shots/_crop.png';
await page.locator('#c').screenshot({ path: dst });
console.log(dst);
await browser.close();
