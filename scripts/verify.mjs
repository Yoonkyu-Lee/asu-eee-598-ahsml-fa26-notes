// scripts/verify.mjs
// 사용법: node scripts/verify.mjs L03-conditional-probability.html
//
// 잡는 것: JS 콘솔 에러, SVG 텍스트가 viewBox를 벗어남, 중복 id,
//          해결되지 않는 url(#...) 참조, 모바일 가로 오버플로우,
//          데스크톱/모바일 전체 스크린샷 + 도해별 개별 스크린샷
// 못 잡는 것: 도형끼리 겹쳐서 글자가 안 읽히는 것, 설명의 질, 예제 난이도.
//            그건 도해별 스크린샷을 사람이 봐야 한다.

import { chromium } from 'playwright';
import { createServer } from 'node:http';
import { createReadStream, readFileSync, readdirSync, mkdirSync, existsSync, statSync } from 'node:fs';
import { resolve, basename, extname, join, relative, sep } from 'node:path';

const target = process.argv[2];
if (!target) {
  console.error('사용법: node scripts/verify.mjs <파일.html>');
  process.exit(1);
}
const abs = resolve(process.cwd(), target);
if (!existsSync(abs)) {
  console.error(`파일 없음: ${abs}`);
  process.exit(1);
}

const stem = basename(target).replace(/\.html$/, '');
mkdirSync('shots', { recursive: true });

// ── 정적 서버 ────────────────────────────────
// file://로 열면 브라우저가 PDF와 모듈 fetch를 막아서 슬라이드 리더를 검증할 수 없다.
// GitHub Pages와 같은 조건(http)으로 맞춘다.
const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.pdf': 'application/pdf',
  '.png': 'image/png',
  '.svg': 'image/svg+xml',
  '.json': 'application/json',
};
const ROOT = process.cwd();

function serve() {
  return new Promise(ok => {
    const server = createServer((req, res) => {
      const rel = decodeURIComponent(req.url.split('?')[0]).replace(/^\/+/, '');
      const file = resolve(ROOT, rel);
      // 루트 밖으로 나가는 요청은 막는다
      const inside = file === ROOT || file.startsWith(ROOT + sep);
      if (!inside || !existsSync(file) || !statSync(file).isFile()) {
        res.writeHead(404).end('not found');
        return;
      }
      res.writeHead(200, { 'content-type': MIME[extname(file).toLowerCase()] || 'application/octet-stream' });
      createReadStream(file).pipe(res);
    });
    server.listen(0, '127.0.0.1', () => ok({
      origin: `http://127.0.0.1:${server.address().port}`,
      close: () => new Promise(done => server.close(done)),
    }));
  });
}

const site = await serve();
const url = `${site.origin}/${relative(ROOT, abs).split(sep).join('/')}`;

// ── 색 리터럴 검사 (소스 기준) ────────────────
// 색을 하드코딩하면 다크모드에서 그 부분만 안 바뀐다. 화면으로는 잘 안 보여서
// 소스에서 잡는다. :root 정의부와 <mask> 안(알파 채널)은 예외다.
const litColors = (() => {
  let src = readFileSync(abs, 'utf8');
  src = src.replace(/<mask[\s\S]*?<\/mask>/g, '');                 // 마스크 = 알파 채널
  src = src.replace(/^:root[^{]*\{[\s\S]*?^\}/gm, '');             // 토큰 정의
  src = src.replace(/@media \(prefers-color-scheme[\s\S]*?\}\}/g, '');
  const hits = [];
  for (const m of src.matchAll(/(?:fill|stroke|color|background|border[a-z-]*)\s*[:=]\s*"?(#[0-9A-Fa-f]{3,8}\b|rgba?\(\s*\d+\s*,[^)]*\))/g)) {
    hits.push(m[1]);
  }
  return [...new Set(hits)];
})();

const MOBILE = 390;
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });

const consoleErrors = [];
page.on('console', m => { if (m.type() === 'error') consoleErrors.push(m.text()); });
page.on('pageerror', e => consoleErrors.push('pageerror: ' + e.message));

await page.goto(url, { waitUntil: 'networkidle' });
await page.waitForTimeout(400);

// ── 정적 검사 ────────────────────────────────
const report = await page.evaluate(() => {
  const out = { dupIds: [], badRefs: [], overflow: [], counts: {} };

  // 중복 id
  const seen = new Map();
  document.querySelectorAll('[id]').forEach(el => {
    seen.set(el.id, (seen.get(el.id) || 0) + 1);
  });
  seen.forEach((n, id) => { if (n > 1) out.dupIds.push(`${id} (${n}회)`); });

  // url(#...) 참조가 실제 존재하는지
  const refAttrs = ['clip-path', 'mask', 'fill', 'stroke', 'marker-end', 'marker-start', 'filter'];
  document.querySelectorAll('svg *').forEach(el => {
    refAttrs.forEach(a => {
      const v = el.getAttribute(a);
      if (!v) return;
      const m = /url\(#([^)]+)\)/.exec(v);
      if (m && !document.getElementById(m[1])) out.badRefs.push(`${a}="${v}"`);
    });
  });

  // SVG 텍스트가 viewBox 밖으로 나가는지
  document.querySelectorAll('svg').forEach((svg, si) => {
    const vb = svg.viewBox && svg.viewBox.baseVal;
    if (!vb || (!vb.width && !vb.height)) return;
    const label = svg.getAttribute('aria-label') || `svg#${si}`;
    svg.querySelectorAll('text').forEach(t => {
      let b;
      try { b = t.getBBox(); } catch { return; }
      if (b.width === 0 && b.height === 0) return;
      const pad = 1.5;
      const left = b.x < vb.x - pad;
      const right = b.x + b.width > vb.x + vb.width + pad;
      const top = b.y < vb.y - pad;
      const bottom = b.y + b.height > vb.y + vb.height + pad;
      if (left || right || top || bottom) {
        const side = [left && '왼쪽', right && '오른쪽', top && '위', bottom && '아래']
          .filter(Boolean).join('/');
        out.overflow.push(
          `[${label}] "${(t.textContent || '').trim().slice(0, 24)}" ${side} 이탈 ` +
          `(x=${b.x.toFixed(0)} w=${b.width.toFixed(0)} / viewBox w=${vb.width})`
        );
      }
    });
  });

  out.counts.sections = document.querySelectorAll('section').length;
  out.counts.figures = document.querySelectorAll('figure, .mini').length;
  out.counts.examples = document.querySelectorAll('.ex').length;
  out.counts.traps = document.querySelectorAll('.trap').length;
  out.counts.details = document.querySelectorAll('details').length;
  return out;
});

// ── 스크린샷: 데스크톱 / 모바일 전체 ──────────
// 이 저장소의 노트는 한 슬라이드도 건너뛰지 않아서 페이지가 아주 길다.
// 크롬은 대략 16384px 를 넘는 이미지를 못 만들고 fullPage 캡처가 통째로 실패한다
// (Protocol error: Unable to capture screenshot). 그래서 넘치면 잘라서 여러 장으로 찍는다.
const SHOT_MAX = 12000;
const shotFiles = [];

async function shootTall(prefix) {
  const h = await page.evaluate(() => document.documentElement.scrollHeight);
  if (h <= SHOT_MAX) {
    const p = `shots/${prefix}.png`;
    await page.screenshot({ path: p, fullPage: true });
    shotFiles.push(p);
    return;
  }
  const vp = page.viewportSize();
  const slice = SHOT_MAX;
  const n = Math.ceil(h / slice);
  await page.setViewportSize({ width: vp.width, height: slice });
  await page.waitForTimeout(200);
  for (let i = 0; i < n; i++) {
    await page.evaluate(y => window.scrollTo(0, y), i * slice);
    await page.waitForTimeout(120);
    const p = `shots/${prefix}-${String(i + 1).padStart(2, '0')}.png`;
    await page.screenshot({ path: p });
    shotFiles.push(p);
  }
  await page.setViewportSize(vp);
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.waitForTimeout(200);
}

await shootTall(`${stem}-desktop`);
await page.setViewportSize({ width: MOBILE, height: 844 });
await page.waitForTimeout(250);
await shootTall(`${stem}-mobile`);

// ── 모바일 가로 오버플로우 (풀이를 다 펼친 상태로) ──
// 풀이가 접혀 있을 때는 멀쩡하다가 펼치면 터지는 경우가 있어서 펼쳐놓고 잰다.
await page.evaluate(() => document.querySelectorAll('details').forEach(d => d.open = true));
await page.waitForTimeout(300);
const hoverflow = await page.evaluate(() => {
  const de = document.documentElement;
  const doc = de.scrollWidth;
  const view = de.clientWidth;
  const bad = [];
  if (doc > view + 1) {
    document.querySelectorAll('body *').forEach(el => {
      // SVG 내부 요소는 좌표계가 달라 의미 없는 값이 나온다
      if (el.tagName !== 'svg' && el.closest('svg')) return;
      // 가로 스크롤 컨테이너 안쪽은 의도된 것이므로 제외
      for (let p = el.parentElement; p && p !== document.body; p = p.parentElement) {
        if (getComputedStyle(p).overflowX === 'auto') return;
      }
      const b = el.getBoundingClientRect();
      if (b.right <= view + 1) return;
      // 더 안쪽 요소가 이미 걸렸으면 바깥 요소는 생략
      const inner = [...el.children].some(k => k.getBoundingClientRect().right > view + 1);
      if (inner) return;
      const sec = el.closest('section');
      bad.push(
        `<${el.tagName.toLowerCase()} class="${String(el.className).slice(0, 22)}"> ` +
        `right=${Math.round(b.right)} [${sec ? sec.id : '-'}] ` +
        `"${(el.textContent || '').replace(/\s+/g, ' ').trim().slice(0, 40)}"`
      );
    });
  }
  return { doc, view, bad: bad.slice(0, 8) };
});

// ── 슬라이드 리더 ────────────────────────────
// 앵커가 있는 파일만 검사한다. 리더는 1100px 이상에서만 뜨므로 넓은 뷰포트로 연다.
const reader = { anchors: 0, mounted: false, pages: 0, outOfRange: [], overlaps: [], backward: [], gaps: [], folded: [], lostLabel: [], starts: 0, marks: 0, ticks: 0, skipped: false };
{
  const anchorCount = await page.evaluate(() => document.querySelectorAll('[data-slide]').length);
  reader.anchors = anchorCount;
  if (anchorCount === 0) {
    reader.skipped = true;
  } else {
    await page.setViewportSize({ width: 1600, height: 1000 });
    await page.goto(url, { waitUntil: 'networkidle' });
    try {
      // 첫 쪽이 렌더될 때까지 기다린다
      await page.waitForSelector('#rdr canvas', { timeout: 20000 });
      reader.mounted = true;
    } catch { /* mounted=false로 아래에서 보고된다 */ }

    Object.assign(reader, await page.evaluate(() => {
      const txt = document.querySelector('#rdr-pg')?.textContent || '';
      const total = +(/\/\s*(\d+)/.exec(txt)?.[1] || 0);
      const list = [...document.querySelectorAll('[data-slide]')].map(el => {
        const m = /^(\d+)(?:\s*[-–~]\s*(\d+))?$/.exec(String(el.dataset.slide).trim());
        // 앵커는 id가 없는 div.sec-head인 경우가 많아 감싸는 섹션 id를 같이 보여준다.
        const sec = el.closest('section');
        const where = (sec ? sec.id + ' ' : '') + el.tagName.toLowerCase() +
          ' "' + (el.textContent || '').replace(/\s+/g, ' ').trim().slice(0, 26) + '"';
        const isSec = el.classList.contains('sec-head');
        if (!m) return { where, isSec, bad: true, raw: el.dataset.slide };
        return { where, isSec, from: +m[1], to: m[2] ? +m[2] : +m[1] };
      });
      const outOfRange = list
        .filter(a => a.bad || (total && (a.from < 1 || a.to > total || a.from > a.to)))
        .map(a => `${a.where} → data-slide="${a.bad ? a.raw : a.from + '-' + a.to}"`);
      const overlaps = [];
      for (let i = 0; i < list.length; i++) {
        for (let j = i + 1; j < list.length; j++) {
          const a = list[i], b = list[j];
          if (a.bad || b.bad) continue;
          if (a.from <= b.to && b.from <= a.to) {
            // 섹션과 그 안의 h3는 포함 관계라 정상. 완전 포함은 봐준다.
            const contains = (a.from <= b.from && b.to <= a.to) || (b.from <= a.from && a.to <= b.to);
            if (!contains) overlaps.push(`${a.where} [${a.from}-${a.to}] ↔ ${b.where} [${b.from}-${b.to}]`);
          }
        }
      }
      // 문서 순서대로 읽었을 때 슬라이드 번호가 뒤로 가는 곳.
      // 노트 구성이 슬라이드 진행과 어긋난다는 신호다. 의도한 재배열일 수도 있어 경고만 한다.
      const backward = [];
      for (let i = 1; i < list.length; i++) {
        const a = list[i - 1], b = list[i];
        if (a.bad || b.bad) continue;
        if (b.from < a.from) backward.push(`${a.where} [${a.from}] → ${b.where} [${b.from}]`);
      }
      // 어떤 앵커에도 안 걸린 쪽. 노트가 그 슬라이드를 아예 안 다룬다는 뜻이다.
      // 표지나 손글씨 삽입 페이지는 정상이지만, 범위를 짧게 잡아 빠뜨린 경우도 여기 걸린다.
      const covered = new Set();
      for (const a of list) {
        if (a.bad) continue;
        for (let i = a.from; i <= a.to; i++) covered.add(i);
      }
      const gaps = [];
      for (let i = 1; i <= total; i++) if (!covered.has(i)) gaps.push(i);

      // 노트의 판정선과 슬라이드의 판정선은 개수가 맞아야 한다.
      // 단, 기준은 앵커 개수가 아니라 "서로 다른 시작 쪽" 개수다.
      // 노트 앵커 둘이 같은 슬라이드 쪽에서 시작하면 PDF에는 그릴 자리가 하나뿐이라
      // 하나로 접힌다. 그건 정상이고, 그 외에 어긋나면 표시를 잃어버린 것이다.
      const starts = new Map();      // 시작 쪽 → 그 쪽에서 시작하는 앵커 이름들
      for (const a of list) {
        if (a.bad) continue;
        if (!starts.has(a.from)) starts.set(a.from, []);
        starts.get(a.from).push(a);
      }
      const marks = document.querySelectorAll('.rdr-mark').length;
      const ticks = document.querySelectorAll('.rdr-tick').length;
      // 접힌 자리 중, 섹션끼리 부딪힌 것은 라벨 이름을 실제로 잃는다.
      // 섹션 + h3 조합은 라벨이 이기므로 정보 손실이 없다.
      const folded = [], lostLabel = [];
      for (const [pg, v] of starts.entries()) {
        if (v.length < 2) continue;
        const line = `${pg}쪽: ${v.map(x => x.where).join('  |  ')}`;
        if (v.filter(x => x.isSec).length > 1) lostLabel.push(line);
        else folded.push(line);
      }

      return {
        pages: total, outOfRange, overlaps, backward, gaps,
        starts: starts.size, marks, ticks, folded, lostLabel,
      };
    }));
  }
}

// ── 스크린샷: 도해별 개별 (풀이 펼친 상태, 데스크톱 폭) ──
await page.setViewportSize({ width: 1280, height: 900 });
await page.goto(url, { waitUntil: 'networkidle' });
await page.evaluate(() => document.querySelectorAll('details').forEach(d => d.open = true));
await page.waitForTimeout(300);
await page.waitForTimeout(250);
const figs = await page.$$('figure, .play');
let shotCount = 0;
for (const el of figs) {
  const id = String(shotCount + 1).padStart(2, '0');
  try {
    await el.scrollIntoViewIfNeeded();
    await page.waitForTimeout(60);
    await el.screenshot({ path: `shots/${stem}-fig${id}.png` });
    shotCount++;
  } catch {
    // 렌더되지 않는 요소는 건너뛴다
  }
}

// ── 본문 링크가 다크모드에서 읽히는지 ──────────────────────────────
// 링크에 규칙을 안 주면 브라우저 기본 남색이 나온다. 라이트에서는 그럭저럭
// 읽히지만 다크에서는 배경에 묻는다. 노트에는 본문 링크가 없어서 드러나지
// 않던 결함이고, 공략 페이지처럼 링크가 본체인 문서에서 바로 터진다.
const linkContrast = [];
for (const theme of ['light', 'dark']) {
  await page.evaluate(t => { document.documentElement.dataset.theme = t; }, theme);
  await page.waitForTimeout(120);
  const bad = await page.evaluate((t) => {
    const lum = (c) => {
      const [r, g, b] = c.map(v => {
        const x = v / 255;
        return x <= 0.03928 ? x / 12.92 : Math.pow((x + 0.055) / 1.055, 2.4);
      });
      return 0.2126 * r + 0.7152 * g + 0.0722 * b;
    };
    const parse = (s) => (s.match(/[\d.]+/g) || []).slice(0, 3).map(Number);
    const bgOf = (el) => {
      for (let n = el; n; n = n.parentElement) {
        const c = getComputedStyle(n).backgroundColor;
        const a = (c.match(/[\d.]+/g) || [])[3];
        if (c && c !== 'transparent' && a !== '0') return parse(c);
      }
      return parse(getComputedStyle(document.body).backgroundColor || 'rgb(255,255,255)');
    };
    const out = [];
    for (const a of document.querySelectorAll('main a, .toc a')) {
      if (getComputedStyle(a).display === 'none') continue;
      const fg = parse(getComputedStyle(a).color);
      const bg = bgOf(a);
      const l1 = lum(fg), l2 = lum(bg);
      const ratio = (Math.max(l1, l2) + 0.05) / (Math.min(l1, l2) + 0.05);
      if (ratio < 3)
        out.push(t + ': ' + ratio.toFixed(2) + ':1  "' +
          (a.textContent || '').replace(/\s+/g, ' ').trim().slice(0, 34) + '"');
    }
    return [...new Set(out)];
  }, theme);
  linkContrast.push(...bad);
}
await page.evaluate(() => { delete document.documentElement.dataset.theme; });

// ── <details> 하나에 <summary>가 둘 이상인지 ──────────────────────
// 개폐 버튼이 되는 건 첫 번째 <summary> 하나뿐이다. 이중 언어를 만든다고
// <summary>를 두 벌 넣으면, 숨겨진 쪽이 첫 번째가 되는 언어에서
// 누를 수 있는 것이 사라져 내용에 영원히 도달할 수 없게 된다.
// 이중 언어는 <summary> 안에 <span lang>을 두 개 넣어서 해결한다.
const dupSummary = await page.evaluate(() =>
  [...document.querySelectorAll('details')]
    .filter(d => d.querySelectorAll(':scope > summary').length > 1)
    .map(d => (d.querySelector(':scope > summary')?.textContent || '')
      .replace(/\s+/g, ' ').trim().slice(0, 40)));

// 접힌 <details> 를 각 언어에서 실제로 열 수 있는지
await page.evaluate(() => document.querySelectorAll('details').forEach(d => d.open = false));
const unopenable = [];
for (const lang of ['ko', 'en']) {
  await page.evaluate(l => {
    document.documentElement.dataset.lang = l;
    document.documentElement.lang = l;
  }, lang);
  await page.waitForTimeout(120);
  const bad = await page.evaluate((l) =>
    [...document.querySelectorAll('details')].map((d, i) => {
      const dr = d.getBoundingClientRect();
      if (dr.width === 0 && dr.height === 0) return null;   // 블록째 숨겨진 건 정상
      const sum = d.querySelector(':scope > summary');
      if (!sum) return l + ': summary 자체가 없는 details';
      const r = sum.getBoundingClientRect();
      if (getComputedStyle(sum).display === 'none' || r.width === 0 || r.height === 0)
        return l + ': 개폐 버튼이 보이지 않는 details #' + (i + 1);
      return null;
    }).filter(Boolean), lang);
  unopenable.push(...bad);
}
await page.evaluate(() => {
  document.documentElement.dataset.lang = 'ko';
  document.documentElement.lang = 'ko';
  document.querySelectorAll('details').forEach(d => d.open = true);
});

// ── 도해 안에서 글자가 다른 것과 겹치는지 ────────────────────────
// viewBox 밖으로 나가는 것만 잡고 있었다. 안에서 겹치는 건 못 잡았고,
// 실제로 포함배제 도해의 +, −, = 가 옆 패널 원 안에 그려져 있었다.
// 좌표는 반드시 getBoundingClientRect 로 잰다. getBBox 는 조상의
// transform 을 반영하지 않아서 translate 로 배치한 패널들이 전부
// 같은 자리에 있는 것처럼 나온다.
const svgHits = await page.evaluate(() => {
  const textHits = [], shapeHits = [];
  const vis = e => getComputedStyle(e).display !== 'none';
  const grp = e => { let n = e.parentElement; while (n && n.tagName !== 'svg') { if (n.tagName === 'g') return n; n = n.parentElement; } return null; };
  const ov = (a, b) => a.left < b.right - 1 && b.left < a.right - 1 && a.top < b.bottom - 1 && b.top < a.bottom - 1;
  document.querySelectorAll('figure svg, .play svg, .nota svg').forEach(svg => {
    const name = (svg.getAttribute('aria-label') || '?').slice(0, 30);
    const texts = [...svg.querySelectorAll('text')].filter(vis)
      .map(t => ({ bb: t.getBoundingClientRect(), g: grp(t), txt: (t.textContent || '').trim().slice(0, 16) }))
      .filter(t => t.bb.width > 0.5);
    for (let i = 0; i < texts.length; i++) for (let j = i + 1; j < texts.length; j++)
      if (ov(texts[i].bb, texts[j].bb))
        textHits.push(name + ': "' + texts[i].txt + '" 와 "' + texts[j].txt + '"');
    const shapes = [...svg.querySelectorAll('circle,rect,ellipse,path')].filter(vis)
      .filter(x => { const f = x.getAttribute('fill') || getComputedStyle(x).fill; return f && f !== 'none' && !/rgba\(0, 0, 0, 0\)/.test(f); })
      .map(x => ({ bb: x.getBoundingClientRect(), g: grp(x), tag: x.tagName }));
    texts.forEach(t => shapes.forEach(sh => {
      if (t.g === sh.g) return;              // 같은 그룹이면 그 도형의 라벨이다
      if (ov(t.bb, sh.bb)) shapeHits.push(name + ': "' + t.txt + '" 가 ' + sh.tag + ' 위에');
    }));
  });
  return { textHits: [...new Set(textHits)], shapeHits: [...new Set(shapeHits)] };
});

// ── 리더가 켜졌을 때 본문이 읽을 수 있는 폭을 유지하는지 ──────────
// 리더는 폭을 차지하므로 화면이 좁으면 본문을 그만큼 밀어낸다. 실제로
// 1100px 에서 본문이 266px 까지 눌린 적이 있다. 모바일 기준(390px)보다
// 좁아서 표와 도해가 깨진다. 리더가 뜨는 가장 좁은 화면에서 재둔다.
let narrowMain = 0;
try {
  await page.setViewportSize({ width: 1220, height: 900 });
  await page.goto(url, { waitUntil: 'networkidle' });
  await page.waitForTimeout(900);
  narrowMain = await page.evaluate(() => {
    const m = document.querySelector('main');
    return m ? Math.round(m.getBoundingClientRect().width) : 0;
  });
} catch { narrowMain = 0; }

await browser.close();
await site.close();

// ── 결과 출력 ────────────────────────────────
const line = (s) => console.log(s);
let fail = 0;

line(`\n검증: ${target}`);
line(`섹션 ${report.counts.sections} · 도해 ${report.counts.figures} · ` +
     `예제 ${report.counts.examples} · 함정 ${report.counts.traps} · 접힌풀이 ${report.counts.details}`);

if (consoleErrors.length) {
  fail++; line(`\n[FAIL] JS 콘솔 에러 ${consoleErrors.length}건`);
  consoleErrors.slice(0, 8).forEach(e => line('  · ' + e));
} else line('\n[OK] JS 콘솔 에러 없음');

if (report.dupIds.length) {
  fail++; line(`\n[FAIL] 중복 id ${report.dupIds.length}건 (SVG 렌더링이 깨진다)`);
  report.dupIds.forEach(e => line('  · ' + e));
} else line('[OK] 중복 id 없음');

if (report.badRefs.length) {
  fail++; line(`\n[FAIL] 존재하지 않는 참조 ${report.badRefs.length}건`);
  report.badRefs.slice(0, 8).forEach(e => line('  · ' + e));
} else line('[OK] url(#...) 참조 정상');

if (report.overflow.length) {
  fail++; line(`\n[FAIL] SVG 텍스트 이탈 ${report.overflow.length}건`);
  report.overflow.slice(0, 15).forEach(e => line('  · ' + e));
} else line('[OK] SVG 텍스트 viewBox 안에 있음');

if (hoverflow.doc > hoverflow.view + 1) {
  fail++;
  line(`\n[FAIL] 모바일 가로 오버플로우 (문서 ${hoverflow.doc}px > 화면 ${hoverflow.view}px)`);
  hoverflow.bad.forEach(e => line('  · ' + e));
  line('  힌트: 긴 .m 수식(nowrap), 표의 min-content, 그리드 트랙 1fr을 의심할 것.');
} else line(`[OK] 모바일(${MOBILE}px) 가로 오버플로우 없음 (풀이 펼친 상태 기준)`);

if (litColors.length) {
  fail++;
  line(`
[FAIL] 하드코딩된 색 종 (다크모드에서 이 부분만 안 바뀐다)`);
  litColors.slice(0, 12).forEach(e => line("  · " + e));
  line("  디자인 토큰을 쓸 것: var(--blue), rgba(var(--blue-rgb),.16) 형태.");
} else line("[OK] 하드코딩된 색 없음 (전부 디자인 토큰)");
if (svgHits.textHits.length) {
  fail = 1;
  line('\n[FAIL] 도해 안에서 글자끼리 겹침 ' + svgHits.textHits.length + '건');
  svgHits.textHits.slice(0, 10).forEach(x => line('  · ' + x));
} else {
  line('[OK] 도해 글자끼리 겹치지 않음');
}
if (narrowMain && narrowMain < 480) {
  fail = 1;
  line('\n[FAIL] 리더가 뜨는 최소 폭(1220px)에서 본문이 ' + narrowMain + 'px 뿐이다');
  line('  모바일 기준(390px)에 가깝게 눌리면 표와 도해가 깨진다.');
  line('  reader.css 의 --rdr-w 를 줄이거나 리더 임계폭을 올릴 것.');
} else if (narrowMain) {
  line('[OK] 리더 최소 폭에서 본문 ' + narrowMain + 'px 확보');
}
if (dupSummary.length || unopenable.length) {
  fail = 1;
  line('\n[FAIL] 열 수 없는 <details> ' + (dupSummary.length + unopenable.length) + '건');
  dupSummary.forEach(x => line('  · <summary>가 2개 이상: "' + x + '"'));
  unopenable.forEach(x => line('  · ' + x));
  line('  개폐 버튼이 되는 건 첫 번째 <summary> 하나뿐이다. 이중 언어로 두 벌을 넣으면');
  line('  한쪽 언어에서 누를 것이 사라진다. <summary> 안에 <span lang>을 두 개 넣을 것.');
} else {
  line('[OK] 모든 <details> 가 양쪽 언어에서 열린다');
}
if (linkContrast.length) {
  fail = 1;
  line('\n[FAIL] 배경에 묻는 링크 ' + linkContrast.length + '건 (대비 3:1 미만)');
  linkContrast.slice(0, 12).forEach(x => line('  · ' + x));
  line('  링크에 색 규칙이 없으면 브라우저 기본 남색이 나와 다크모드에서 안 읽힌다.');
  line('  main a{color:var(--blue)} 같은 토큰 규칙을 줄 것.');
} else {
  line('[OK] 링크 대비 충분 (라이트/다크 양쪽)');
}

if (reader.skipped) {
  line('[--] 슬라이드 리더: data-slide 앵커가 없어 건너뜀');
} else if (!reader.mounted) {
  fail++;
  line(`\n[FAIL] 슬라이드 리더가 뜨지 않음 (앵커는 ${reader.anchors}개 있음)`);
  line(`  slides/${stem}.pdf 가 있는지, reader.js 태그가 들어갔는지 확인할 것.`);
} else {
  line(`[OK] 슬라이드 리더 동작 (앵커 ${reader.anchors}개 / PDF ${reader.pages}쪽)`);

  // 노트의 판정선과 슬라이드의 판정선 개수가 맞는지.
  // 앵커 개수가 아니라 "서로 다른 시작 쪽" 개수와 맞아야 한다.
  const shown = reader.marks + reader.ticks;
  if (shown === reader.starts) {
    line(`[OK] 판정선 개수 일치 (노트 시작 쪽 ${reader.starts}개 = PDF 표시 ${shown}개` +
         `: 라벨 ${reader.marks} + 눈금 ${reader.ticks})`);
    if (reader.folded.length) {
      line(`     같은 슬라이드 쪽에서 시작해 하나로 접힌 앵커 ${reader.folded.length}곳 (정상):`);
      reader.folded.slice(0, 6).forEach(e => line('       · ' + e));
    }
  } else {
    fail++;
    line(`\n[FAIL] 판정선 개수 불일치: 노트 시작 쪽 ${reader.starts}개 ≠ PDF 표시 ${shown}개`);
    line(`  라벨 ${reader.marks} + 눈금 ${reader.ticks}. 슬라이드쪽 표시가 조용히 사라졌다.`);
    line('  섹션 두 개가 같은 쪽에서 시작하면 라벨 하나를 잃는다. reader.js의 layout()을 볼 것.');
  }
}

if (reader.outOfRange?.length) {
  fail++;
  line(`\n[FAIL] data-slide가 PDF 쪽 범위를 벗어남 ${reader.outOfRange.length}건 (PDF는 ${reader.pages}쪽)`);
  reader.outOfRange.slice(0, 10).forEach(e => line('  · ' + e));
}
if (reader.overlaps?.length) {
  fail++;
  line(`\n[FAIL] data-slide 범위가 서로 겹침 ${reader.overlaps.length}건`);
  reader.overlaps.slice(0, 10).forEach(e => line('  · ' + e));
  line('  겹치면 역방향 동기화(PDF→노트)가 어느 쪽으로 갈지 정해지지 않는다.');
  line('  한 슬라이드는 그것을 실제로 다루는 곳 한 군데에만 앵커를 단다.');
}
if (reader.lostLabel?.length) {
  fail++;
  line(`\n[FAIL] 섹션 두 개가 같은 슬라이드 쪽에서 시작 ${reader.lostLabel.length}곳`);
  reader.lostLabel.forEach(e => line('  · ' + e));
  line('  슬라이드쪽에 라벨을 하나만 그릴 수 있어 섹션 이름 하나가 사라진다.');
  line('  두 섹션이 정말 같은 쪽에서 시작한다면 하나로 합치거나 범위를 다시 나눌 것.');
}
if (reader.gaps?.length) {
  // 실패로 치지 않는다. 표지나 손글씨 삽입 페이지는 노트가 안 다루는 게 정상이다.
  line(`\n[WARN] 어떤 앵커에도 안 걸린 슬라이드 ${reader.gaps.length}쪽: ${reader.gaps.join(', ')}`);
  line('  노트가 그 슬라이드를 안 다룬다는 뜻이다. 표지·삽입 페이지면 정상이지만,');
  line('  범위를 짧게 잡아 빠뜨린 것일 수도 있다. 해당 쪽을 열어서 확인할 것.');
}
if (svgHits.shapeHits.length) {
  // 실패로 치지 않는다. 원 안에 그 원의 라벨을 두는 건 정상이다.
  // 다른 그룹의 도형을 침범한 것만 모으지만 그래도 오탐이 섞인다.
  line('\n[WARN] 도해에서 글자가 다른 그룹의 도형 위에 있는 곳 ' + svgHits.shapeHits.length + '건');
  svgHits.shapeHits.slice(0, 8).forEach(x => line('  · ' + x));
  line('  그 도형의 라벨이면 정상이다. 연산자나 캡션이면 자리가 겹친 것이니 해당 도해를 열어볼 것.');
}
if (reader.backward?.length) {
  // 실패로 치지 않는다. 노트를 일부러 슬라이드와 다르게 배열할 수도 있다.
  line(`\n[WARN] 노트를 따라 내려가는데 슬라이드가 뒤로 가는 곳 ${reader.backward.length}건`);
  reader.backward.slice(0, 10).forEach(e => line('  · ' + e));
  line('  읽으면서 슬라이드가 앞뒤로 튄다. 의도한 재배열이 아니면 노트 순서를 슬라이드에 맞추는 게 낫다.');
}

line(`\n전체 스크린샷 (${shotFiles.length}장): ` + shotFiles.join(', '));
if (shotCount > 0) {
  line(`도해 스크린샷: shots/${stem}-fig01.png … -fig${String(shotCount).padStart(2, '0')}.png (${shotCount}장)`);
  line('도해 스크린샷은 반드시 한 장씩 열어서 눈으로 확인할 것.');
  line('도형 겹침(테두리가 글자를 관통하는 것)과 색 대비는 자동 검사가 못 잡는다.\n');
} else {
  line('도해 스크린샷: 없음 (figure / .play 요소가 없는 파일)\n');
}


// ── 다른 페이지를 가리키는 링크가 실제로 존재하는지 ────────────────
// 공략 페이지처럼 링크가 본체인 문서는 앵커 하나가 깨지면 근거를 잘못 가리킨다.
// 어긋난 슬라이드 앵커가 해로운 것과 같은 이유로 실패로 잡는다.
{
  const src = readFileSync(abs, 'utf8');
  const dir = resolve(abs, '..');
  const idCache = new Map();
  const idsOf = (f) => {
    if (!idCache.has(f)) {
      const p = resolve(dir, f);
      idCache.set(f, existsSync(p)
        ? new Set([...readFileSync(p, 'utf8').matchAll(/id="([^"]+)"/g)].map(m => m[1]))
        : null);
    }
    return idCache.get(f);
  };
  const broken = [];
  const seen = new Set();
  for (const m of src.matchAll(/href="([^":#]+\.html)(?:#([^"]+))?"/g)) {
    const [, file, id] = m;
    const key = file + (id ? '#' + id : '');
    if (seen.has(key)) continue;
    seen.add(key);
    const ids = idsOf(file);
    if (ids === null) broken.push(key + '   대상 파일이 없다');
    else if (id && !ids.has(id)) broken.push(key + '   그 id가 대상 파일에 없다');
  }
  if (broken.length) {
    fail = 1;
    line('\n[FAIL] 다른 페이지를 가리키는 링크 ' + broken.length + '개가 깨져 있다');
    broken.forEach(b => line('  · ' + b));
    line('  링크가 근거를 잘못 가리키면 없느니만 못하다. 대상 id를 확인할 것.');
  } else if (seen.size) {
    line('[OK] 페이지 간 링크 ' + seen.size + '개 전부 대상 존재');
  }
}

// ── HW 공략 페이지와 노트의 문제 번호가 어긋나지 않는지 ─────────────
// HW 매핑이 노트와 공략 페이지 양쪽에 있다. 한쪽만 고치면 조용히 갈라진다.
{
  const dir = resolve(abs, '..');
  const all = readdirSync(dir);
  for (const g of all.filter(f => /^HW\d+-.*\.html$/.test(f))) {
    const hw = g.match(/^HW(\d+)/)[1];
    const gsrc = readFileSync(resolve(dir, g), 'utf8');
    // 공략 페이지가 다루는 문제 번호. <span class="sec-num">P5</span>
    const covered = new Set([...gsrc.matchAll(/class="sec-num">P(\d+)</g)].map(m => +m[1]));
    // 노트가 언급하는 문제 번호. "HW1 Problem 5", "HW1 Problems 4–8", "HW1 4(b)"
    const cited = new Set();
    for (const f of all.filter(x => /^L\d+-.*\.html$/.test(x))) {
      const t = readFileSync(resolve(dir, f), 'utf8');
      const re = new RegExp('HW' + hw + '\\s*(?:Problems?\\s*)?(\\d+)(?:\\s*[–-]\\s*(\\d+))?', 'g');
      for (const m of t.matchAll(re)) {
        const a = +m[1], b = m[2] ? +m[2] : a;
        if (b < a || b - a > 12) continue;   // 범위가 아닌 숫자가 섞이는 것 방지
        for (let k = a; k <= b; k++) cited.add(k);
      }
    }
    const missing = [...cited].filter(n => !covered.has(n)).sort((a, b) => a - b);
    const extra = [...covered].filter(n => !cited.has(n)).sort((a, b) => a - b);
    if (missing.length || extra.length) {
      fail = 1;
      line('\n[FAIL] ' + g + ' 와 노트의 HW' + hw + ' 문제 번호가 어긋난다');
      if (missing.length) line('  노트는 언급하는데 공략에 없는 문제: ' + missing.join(', '));
      if (extra.length) line('  공략에는 있는데 노트가 안 짚는 문제: ' + extra.join(', '));
      line('  한쪽만 고치면 이렇게 갈라진다. 양쪽을 맞출 것.');
    } else if (covered.size) {
      line('[OK] ' + g + ' 의 문제 ' + covered.size + '개가 노트 언급과 일치');
    }
  }
}

process.exit(fail ? 1 : 0);
