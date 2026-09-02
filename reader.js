// reader.js — 노트 옆에 강의 슬라이드 PDF를 띄우고 스크롤을 동기화한다.
//
// 규칙 (CLAUDE.md "슬라이드 리더" 참조):
//  - 데스크톱 전용. 1220px 아래에서는 아무것도 하지 않고 PDF도 받지 않는다.
//  - 동기화는 구간 경계를 넘을 때만 일어난다. 한 구간 안에서는 양쪽 다 자유롭게 움직인다.
//  - 리더는 부가 기능이다. 실패하면 조용히 사라지고 노트 본문은 그대로 읽힌다.
//
// 앵커는 노트 안의 [data-slide]다. 값은 PDF 물리 쪽 번호(1부터).
//   <section id="s3" data-slide="9">      한 쪽짜리 구간
//   <section id="s5" data-slide="12-17">  12~17쪽을 한 구간으로 묶는다

const MIN_WIDTH = 1220;      // 이 아래로는 리더를 띄우지 않는다.
                             // 더 낮추면 본문이 500px 아래로 눌린다 (reader.css 참조)

// 동기화 기준선. 화면 위에서 이 비율 지점을 책갈피 선이 넘어가면 슬라이드가 넘어간다.
// 화면에 바늘로 표시되고 드래그로 옮길 수 있다. 저장하지 않는다(세션 한정).
let followLine = 0.28;
const FOLLOW_MIN = 0.06, FOLLOW_MAX = 0.6;

// 슬라이드쪽 기준선. 페이지 열의 위에서 이만큼 내려온 지점에 걸친 쪽을 "현재"로 본다.
// 노트쪽 바늘과 마주 보는 위치에 표시되고 역시 드래그로 옮길 수 있다.
let pdfLine = 40;
const PROG_MS = 2500;      // 프로그램 스크롤의 보조 타임아웃 (도착 판정이 주, 이건 안전장치)

const anchors = [...document.querySelectorAll('[data-slide]')]
  .map(el => {
    const raw = String(el.dataset.slide).trim();
    const m = /^(\d+)(?:\s*[-–~]\s*(\d+))?$/.exec(raw);
    if (!m) return null;
    const from = +m[1];
    return { el, from, to: m[2] ? +m[2] : from };
  })
  .filter(Boolean);

// 앵커가 없으면 이 페이지는 리더를 쓰지 않는다 (index.html 등).
if (anchors.length) {
  boot();
}

function boot() {
  if (window.innerWidth < MIN_WIDTH) {
    // 좁은 화면에서는 PDF를 받지 않는다. 넓어지면 그때 시작한다.
    const onResize = () => {
      if (window.innerWidth >= MIN_WIDTH) {
        window.removeEventListener('resize', onResize);
        boot();
      }
    };
    window.addEventListener('resize', onResize);
    return;
  }
  // start 자체가 실패하면(리더 UI조차 못 만든 경우) 그때만 걷어낸다.
  // PDF 로드 실패는 load() 안에서 재시도 버튼으로 처리한다.
  start().catch(() => teardown());
}

let ui = null;

function teardown() {
  if (ui) {
    ui.root.remove();
    ui.openBtn?.remove();
    ui.needle?.remove();
    document.body.classList.remove('rdr-on');
    ui = null;
  }
}

// 노트 파일명에서 슬라이드 경로를 만든다. L02-probability-space.html → slides/L02-probability-space.pdf
function pdfUrl() {
  const name = location.pathname.split('/').pop() || '';
  const stem = name.replace(/\.html?$/i, '');
  if (!/^L\d+-/.test(stem)) return null;
  return `slides/${stem}.pdf`;
}

function css(href) {
  return new Promise(resolve => {
    const l = document.createElement('link');
    l.rel = 'stylesheet';
    l.href = href;
    l.onload = l.onerror = () => resolve();
    document.head.appendChild(l);
  });
}

function buildUI() {
  const root = document.createElement('aside');
  root.id = 'rdr';
  root.innerHTML = `
    <div id="rdr-grab" title="드래그해서 폭 조절"></div>
    <div id="rdr-head">
      <button id="rdr-prev" title="이전 쪽">◀</button>
      <span id="rdr-pg">–</span>
      <button id="rdr-next" title="다음 쪽">▶</button>
      <span class="sp"></span>
      <button id="rdr-follow" class="on" title="노트를 스크롤하면 슬라이드가 따라온다">따라가기</button>
      <button id="rdr-close" title="리더 닫기">✕</button>
    </div>
    <div id="rdr-scroll"><div id="rdr-msg">
      <div class="rdr-line"></div>
      <div id="rdr-bar" class="is-indet"><i></i></div>
      <div class="rdr-sub"></div>
      <button id="rdr-retry" hidden></button>
    </div></div>
    <div id="rdr-pin" title="슬라이드쪽 기준선. 여기 걸린 쪽이 '현재'다. 다른 구간으로 넘어가면 노트가 따라온다. 드래그해서 옮길 수 있다."></div>`;
  document.body.appendChild(root);
  document.body.classList.add('rdr-on');

  const openBtn = document.createElement('button');
  openBtn.id = 'rdr-open';
  openBtn.textContent = L('슬라이드', 'slides');
  openBtn.hidden = true;
  document.body.appendChild(openBtn);

  const needle = document.createElement('div');
  needle.id = 'rdr-needle';
  needle.title = '동기화 기준선. 제목 옆 선이 여기를 넘어가면 슬라이드가 넘어간다. 드래그해서 옮길 수 있다.';
  document.body.appendChild(needle);

  return {
    root, openBtn, needle,
    scroll: root.querySelector('#rdr-scroll'),
    msg: root.querySelector('#rdr-msg'),
    pg: root.querySelector('#rdr-pg'),
    prev: root.querySelector('#rdr-prev'),
    next: root.querySelector('#rdr-next'),
    follow: root.querySelector('#rdr-follow'),
    close: root.querySelector('#rdr-close'),
    grab: root.querySelector('#rdr-grab'),
    pin: root.querySelector('#rdr-pin'),
    line: root.querySelector('#rdr-msg .rdr-line'),
    bar: root.querySelector('#rdr-bar'),
    sub: root.querySelector('#rdr-msg .rdr-sub'),
    retry: root.querySelector('#rdr-retry'),
  };
}

// 진행이 이만큼 멈춰 있으면 죽은 것으로 본다. 전체 시간이 아니라
// "마지막 진전 이후" 기준이다. 느린 회선에서 오래 걸리는 것과
// 아예 멈춘 것을 구분해야 하기 때문이다.
const STALL_MS = 25000;

const KB = n => (n < 1024 * 1024
  ? Math.round(n / 1024) + ' KB'
  : (n / 1024 / 1024).toFixed(1) + ' MB');

function setPhase(text, sub) {
  if (!ui?.line) return;
  ui.line.textContent = text;
  ui.sub.textContent = sub || '';
}

function setProgress(loaded, total) {
  if (!ui?.bar) return;
  if (total > 0) {
    ui.bar.classList.remove('is-indet');
    ui.bar.firstElementChild.style.width =
      Math.min(100, Math.round((loaded / total) * 100)) + '%';
    setPhase(L('슬라이드를 불러오는 중…', 'Loading slides…'),
             KB(loaded) + ' / ' + KB(total));
  } else {
    ui.bar.classList.add('is-indet');
    setPhase(L('슬라이드를 불러오는 중…', 'Loading slides…'), loaded ? KB(loaded) : '');
  }
}

function showError(why) {
  if (!ui?.line) return;
  ui.bar.hidden = true;
  ui.line.textContent = L('슬라이드를 불러오지 못했어.', 'The slides did not load.');
  ui.sub.textContent = why || '';
  ui.retry.textContent = L('다시 불러오기', 'Try again');
  ui.retry.hidden = false;
}

// 실패해도 리더를 지우지 않는다. 지워버리면 왜 안 뜨는지 알 수 없고
// 다시 시도할 방법도 없어진다. 예전에는 그냥 teardown 했다.
function runStart() {
  ui.retry.hidden = true;
  ui.bar.hidden = false;
  ui.bar.classList.add('is-indet');
  ui.bar.firstElementChild.style.width = '0';
  setPhase(L('슬라이드를 불러오는 중…', 'Loading slides…'), '');
  load().catch(err => showError(String(err?.message || err).slice(0, 120)));
}

async function start() {
  const url = pdfUrl();
  if (!url) return;

  await css(new URL('./reader.css', import.meta.url).href);
  ui = buildUI();
  wireChrome();
  ui.retry.addEventListener('click', runStart);

  // file://로 열면 브라우저가 PDF fetch를 막는다. 시도하면 콘솔 에러만 남으므로 아예 안 한다.
  if (location.protocol === 'file:') {
    ui.bar.hidden = true;
    ui.line.innerHTML = L(
      '로컬 파일(<code>file://</code>)로 열면 브라우저가 PDF를 못 읽어.<br><br>' +
      '슬라이드를 보려면 간단한 서버로 열어야 해:<br>' +
      '<code>npx serve .</code> 또는 <code>python -m http.server</code>',
      'Opened as a local file (<code>file://</code>), so the browser will not read the PDF.<br><br>' +
      'Serve the folder to see the slides:<br>' +
      '<code>npx serve .</code> or <code>python -m http.server</code>');
    return;
  }

  runStart();
}

async function load() {
  const url = pdfUrl();
  setProgress(0, 0);

  // 진전이 있을 때마다 시계를 되감는다. 느려서 오래 걸리는 것과 멈춘 것을
  // 총 경과 시간으로는 구분할 수 없다. pdf.js 모듈(1.7MB)을 받는 구간도
  // 덮어야 하므로 import 전에 건다.
  let task = null;
  let timer = null;
  let stalled = false;
  const armStall = () => {
    clearTimeout(timer);
    timer = setTimeout(() => {
      stalled = true;
      try { task?.destroy(); } catch {}
      showError(L('응답이 없어. 연결을 확인하고 다시 눌러줘.',
                  'No response. Check the connection and try again.'));
    }, STALL_MS);
  };
  armStall();

  try {
    const pdfjs = await import('./vendor/pdf.js/pdf.min.mjs');
    if (stalled) return;
    armStall();
    pdfjs.GlobalWorkerOptions.workerSrc = new URL('./vendor/pdf.js/pdf.worker.min.mjs', import.meta.url).href;

    task = pdfjs.getDocument({ url, isEvalSupported: false });
    task.onProgress = ({ loaded, total }) => {
      if (stalled) return;
      armStall();
      setProgress(loaded, total);
    };

    const doc = await task.promise;
    // 정체로 판정하고 destroy 한 뒤에 늦게 끝난 경우. 안내 문구를 덮지 않는다.
    if (stalled) return;
    await layout(doc);
  } catch (err) {
    // 정체 안내가 이미 떠 있으면 destroy 때문에 생긴 거부로 그것을 덮지 않는다.
    if (stalled) return;
    throw err;
  } finally {
    clearTimeout(timer);
  }
}

// ── 페이지 배치와 렌더 ──────────────────────────────
// 먼저 정확한 크기의 빈 상자를 깔아서 스크롤 높이를 확정하고,
// 실제 canvas 렌더는 화면에 가까워진 것만 한다.

let pages = [];          // { div, num, viewport, rendered, task }
let renderObserver = null;

// 앵커가 가리키는 노트 위치의 이름. 슬라이드쪽 구간 라벨에 쓴다.
function anchorTitle(a) {
  const el = a.el;
  if (el.classList?.contains('sec-head')) {
    const num = el.querySelector('.sec-num')?.textContent.trim() || '';
    // h2가 한국어판과 영어판 두 개일 수 있다. 화면에 보이는 쪽을 쓴다.
    const hs = [...el.querySelectorAll('h2')];
    const h2 = (hs.find(h => h.offsetParent !== null) || hs[0])?.textContent.trim() || '';
    return (num ? num + ' ' : '') + h2;
  }
  return el.textContent.trim();
}

async function layout(doc) {
  const n = doc.numPages;
  const width = ui.scroll.clientWidth - 24;   // padding 12px 양쪽
  ui.msg.remove();

  // 이전 렌더 정리
  renderObserver?.disconnect();
  pages.forEach(p => { try { p.task?.cancel(); } catch {} });
  pages = [];
  ui.scroll.querySelectorAll('.rdr-page,.rdr-mark,.rdr-tick').forEach(el => el.remove());

  // 구간 경계를 미리 계산한다.
  // 섹션 경계는 이름표를 달고, h3 하위 경계는 이름 없이 가는 눈금만 둔다.
  // 라벨을 다 달면 페이지 열이 글자로 빽빽해진다.
  const secMark = new Map();   // 쪽 번호 → 노트 섹션 이름
  const subTick = new Set();   // 쪽 번호 (h3 경계)
  const covered = new Set();   // 어떤 앵커든 걸린 쪽
  for (const a of anchors) {
    for (let i = a.from; i <= a.to; i++) covered.add(i);
    if (a.el.classList?.contains('sec-head')) {
      if (!secMark.has(a.from)) secMark.set(a.from, anchorTitle(a));
    } else {
      subTick.add(a.from);
    }
  }

  for (let i = 1; i <= n; i++) {
    const page = await doc.getPage(i);
    const base = page.getViewport({ scale: 1 });
    const scale = width / base.width;
    const vp = page.getViewport({ scale });

    if (secMark.has(i)) {
      const mk = document.createElement('div');
      mk.className = 'rdr-mark';
      mk.textContent = secMark.get(i);
      mk.title = `노트 ${secMark.get(i)} 가 여기서 시작한다`;
      ui.scroll.appendChild(mk);
    } else if (subTick.has(i)) {
      ui.scroll.appendChild(Object.assign(document.createElement('div'), { className: 'rdr-tick' }));
    }

    const div = document.createElement('div');
    div.className = 'rdr-page' + (covered.has(i) ? '' : ' is-bare');
    div.style.width = `${Math.round(vp.width)}px`;
    div.style.height = `${Math.round(vp.height)}px`;
    div.dataset.page = String(i);
    div.innerHTML = `<span class="rdr-num">${i}</span>` +
      (covered.has(i) ? '' : '<span class="rdr-bare">' + L('노트에 없음', 'not in notes') + '</span>');
    ui.scroll.appendChild(div);

    pages.push({ div, num: i, page, vp, rendered: false, task: null });
  }

  renderObserver = new IntersectionObserver(entries => {
    for (const e of entries) {
      if (!e.isIntersecting) continue;
      const p = pages[+e.target.dataset.page - 1];
      if (p && !p.rendered) render(p);
    }
  }, { root: ui.scroll, rootMargin: '300px 0px' });
  pages.forEach(p => renderObserver.observe(p.div));

  wireSync(n);
  relabel();
  syncNoteToPdf(true);
}

// 언어가 바뀌면 리더가 찍어둔 한국어/영어 라벨을 다시 쓴다.
// 페이지는 다시 그리지 않는다. 글자만 바꾼다.
function relabel() {
  if (!ui) return;
  // 구간 라벨: 앵커 순서와 마크 순서가 같으므로 짝지어 다시 쓴다.
  const secAnchors = anchors.filter(a => a.el.classList?.contains('sec-head'));
  const seen = new Set();
  const ordered = [];
  for (const a of secAnchors) { if (seen.has(a.from)) continue; seen.add(a.from); ordered.push(a); }
  ui.scroll.querySelectorAll('.rdr-mark').forEach((mk, i) => {
    if (!ordered[i]) return;
    const t = anchorTitle(ordered[i]);
    mk.textContent = t;
    mk.title = L('노트 ' + t + ' 가 여기서 시작한다', t + ' starts here');
  });
  ui.scroll.querySelectorAll('.rdr-bare').forEach(b => {
    b.textContent = L('노트에 없음', 'not in notes');
  });
  setFollow(autoFollow);
  ui.openBtn.textContent = L('슬라이드', 'slides');
  // 로딩 중이거나 실패한 상태로 언어를 바꿀 수 있다.
  // 성공하면 #rdr-msg 가 통째로 제거되므로 붙어 있을 때만 손댄다.
  if (!ui.line?.isConnected) return;
  if (!ui.retry.hidden) {
    ui.line.textContent = L('슬라이드를 불러오지 못했어.', 'The slides did not load.');
    ui.retry.textContent = L('다시 불러오기', 'Try again');
  } else if (ui.line.textContent) {
    ui.line.textContent = L('슬라이드를 불러오는 중…', 'Loading slides…');
  }
}
window.addEventListener('langchange', relabel);

function render(p) {
  p.rendered = true;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const canvas = document.createElement('canvas');
  canvas.width = Math.round(p.vp.width * dpr);
  canvas.height = Math.round(p.vp.height * dpr);
  p.div.insertBefore(canvas, p.div.firstChild);
  const ctx = canvas.getContext('2d', { alpha: false });
  p.task = p.page.render({
    canvasContext: ctx,
    viewport: p.vp,
    transform: dpr === 1 ? null : [dpr, 0, 0, dpr, 0, 0],
  });
  p.task.promise.catch(() => { p.rendered = false; });
}

// ── 동기화 ──────────────────────────────────────────

// 동기화는 "구간 경계를 넘을 때만" 일어난다.
//
// 한 주제가 슬라이드 여러 장에 걸치면(data-slide="12-17") 그 구간 안에서는
// 노트도 슬라이드도 자유롭게 움직여야 한다. 매 스크롤마다 구간의 첫 쪽으로
// 되돌리면 슬라이드를 읽을 수가 없다.
//
// 그래서 "지금 어느 구간에 있나"(actAnchor)를 하나의 상태로 두고,
// 그 값이 바뀔 때만 반대쪽을 움직인다. 구간이 바뀌어도 상대가 이미 새 구간
// 안에 있으면 건드리지 않는다.
let autoFollow = true;
let actAnchor = null;    // 현재 구간. 이게 바뀔 때만 동기화가 일어난다.
let sticky = null;       // 사용자가 직접 슬라이드를 잡아둔 구역 (아래 설명)
let curPage = 0;

// 루프 가드.
// 시간 창으로만 막으면 부드러운 스크롤이 창보다 오래 걸릴 때 남은 이벤트가
// "사용자 조작"으로 오인된다. 목표 위치에 도착했는지로 판정하고,
// 도달 불가능한 경우(끝까지 스크롤 등)를 위해 넉넉한 타임아웃만 보조로 둔다.
let progPdf = null;      // { target, until }
let progNote = null;

const now = () => performance.now();

// 프로그램이 낸 스크롤이면 true를 반환한다 (핸들러는 그대로 return).
function consumeProg(slot, current) {
  const p = slot === 'pdf' ? progPdf : progNote;
  if (!p) return false;
  if (Math.abs(current - p.target) < 2 || now() > p.until) {
    if (slot === 'pdf') progPdf = null; else progNote = null;
  }
  return true;
}

function scrollPdfTo(top, smooth) {
  const max = ui.scroll.scrollHeight - ui.scroll.clientHeight;
  const t = Math.max(0, Math.min(top, max));
  if (Math.abs(ui.scroll.scrollTop - t) < 2) { progPdf = null; return; }
  progPdf = { target: t, until: now() + PROG_MS };
  ui.scroll.scrollTo({ top: t, behavior: smooth ? 'smooth' : 'auto' });
}

function scrollNoteTo(top, smooth) {
  const max = document.documentElement.scrollHeight - window.innerHeight;
  const t = Math.max(0, Math.min(top, max));
  if (Math.abs(window.scrollY - t) < 2) { progNote = null; return; }
  progNote = { target: t, until: now() + PROG_MS };
  window.scrollTo({ top: t, behavior: smooth ? 'smooth' : 'auto' });
}

function pageTop(i) {
  const p = pages[i - 1];
  if (!p) return 0;
  return p.div.offsetTop - ui.scroll.offsetTop;
}

// 노트를 따라갈 때는 즉시 이동한다. 부드럽게 움직이면 스크롤 중에 목표가 계속
// 바뀌어서 리더가 뒤처지고, 도착 판정도 흔들린다.
function goToPage(n, smooth = false) {
  n = Math.max(1, Math.min(pages.length, n));
  scrollPdfTo(pageTop(n) - 12, smooth);
  setCurrent(n);
}

function setCurrent(n) {
  if (n === curPage) return;
  pages[curPage - 1]?.div.classList.remove('is-current');
  curPage = n;
  pages[n - 1]?.div.classList.add('is-current');
  ui.pg.innerHTML = `<b>${n}</b> / ${pages.length}`;
  ui.prev.disabled = n <= 1;
  ui.next.disabled = n >= pages.length;
}

// 언어 전환 때문에 같은 앵커가 한국어판과 영어판 두 벌로 존재한다.
// 숨겨진 쪽은 getBoundingClientRect()가 전부 0이라 "화면 맨 위"로 오인되므로
// 반드시 걸러야 한다. 보이는 것만 위치 기준으로 쓴다.
const shown = a => a.el.offsetParent !== null;

// 리더 UI도 노트의 언어 설정을 따른다.
const L = (ko, en) => (document.documentElement.dataset.lang === 'en' ? en : ko);

// 화면 위쪽 기준선에 걸린 마지막 앵커 = 지금 읽고 있는 구간.
function anchorFromNote() {
  const line = window.innerHeight * followLine;
  let found = null;
  for (const a of anchors) {
    if (!shown(a)) continue;
    if (a.el.getBoundingClientRect().top <= line) found = a;
    else break;
  }
  return found || anchors.find(shown) || anchors[0];
}

// 지금 리더가 보고 있는 쪽.
function pageFromPdf() {
  const top = ui.scroll.scrollTop + pdfLine;
  let n = 1;
  for (let i = 1; i <= pages.length; i++) {
    if (pageTop(i) <= top) n = i; else break;
  }
  return n;
}

const inRange = (a, n) => !!a && n >= a.from && n <= a.to;

// PDF 쪽 번호로 대응되는 구간을 찾는다.
// 포함하는 앵커가 여럿이면(섹션과 그 안의 h3) 더 좁은 쪽을 고른다. 더 구체적이기 때문이다.
// 어느 구간에도 안 들어가는 쪽(중간에 끼워진 페이지 등)은 그 앞의 마지막 앵커로 둔다.
function anchorForPage(n) {
  let best = null, fallback = null;
  for (const a of anchors) {
    if (!shown(a)) continue;           // 숨겨진 언어판은 위치를 못 준다
    if (inRange(a, n)) {
      if (!best || (a.to - a.from) <= (best.to - best.from)) best = a;
    }
    if (a.from <= n) fallback = a;
  }
  return best || fallback;
}

// 그 쪽을 포함하는 가장 넓은 구간. 사용자가 직접 슬라이드를 옮겼을 때
// "어디까지가 같은 주제인가"의 기준이 된다.
function widestForPage(n) {
  let best = null;
  for (const a of anchors) {
    if (inRange(a, n) && (!best || (a.to - a.from) > (best.to - best.from))) best = a;
  }
  return best;
}

const within = (x, s) => !!x && !!s && x.from >= s.from && x.to <= s.to;

// 노트 → 슬라이드. 구간이 바뀌었을 때만, 그리고 슬라이드가 그 구간 밖일 때만 움직인다.
function syncNoteToPdf(force = false) {
  if (!pages.length) return;
  if (!force && !autoFollow) return;
  const a = anchorFromNote();
  if (!a) return;
  const changed = a !== actAnchor;
  actAnchor = a;
  if (!force && !changed) return;          // 같은 구간 안에서는 가만히 둔다

  // 사용자가 직접 슬라이드를 옮겨둔 상태라면, 그 구역을 벗어나기 전까지 존중한다.
  // 섹션(12-17) 안을 읽는 중에 h3 하위 구간(14-15)으로 들어갔다고 해서
  // 사용자가 보고 있던 16쪽을 14쪽으로 되돌리면 안 된다.
  if (!force && within(a, sticky)) return;
  sticky = null;

  if (!force && inRange(a, curPage)) return;   // 이미 이 구간을 보고 있으면 그대로 둔다
  goToPage(a.from);
  pulseNeedle();                                // 방금 기준선을 넘어서 넘어갔다는 신호
}

// 동기화가 실제로 일어난 순간 바늘을 잠깐 강조한다.
// 왜 지금 슬라이드가 넘어갔는지 눈으로 이어지게 하려는 것.
const pulseTimer = { needle: null, pin: null };
function pulse(which) {
  const el = which === 'pin' ? ui?.pin : ui?.needle;
  if (!el) return;
  el.classList.add('is-hit');
  clearTimeout(pulseTimer[which]);
  pulseTimer[which] = setTimeout(() => el.classList.remove('is-hit'), 420);
}
const pulseNeedle = () => pulse('needle');

function placeNeedle() {
  if (ui?.needle) ui.needle.style.top = `${(followLine * 100).toFixed(2)}%`;
}
function placePin() {
  if (ui?.pin) ui.pin.style.top = `calc(var(--rdr-head) + ${Math.round(pdfLine)}px)`;
}

// 슬라이드 → 노트. 슬라이드가 현재 구간을 벗어났을 때만 움직인다.
// 스크롤이든 ◀▶ 버튼이든 같은 규칙을 따른다.
function alignNoteToPage(n) {
  if (!autoFollow) return;

  // 같은 구간 안에서는 노트를 건드리지 않는다. 대신 "사용자가 이 구역을 직접
  // 잡고 있다"고 기억해서, 노트가 그 구역 안에서 움직여도 되돌리지 않게 한다.
  if (inRange(actAnchor, n)) {
    sticky = widestForPage(n) || actAnchor;
    return;
  }

  const a = anchorForPage(n);
  if (!a || a === actAnchor) { sticky = widestForPage(n); return; }
  actAnchor = a;
  sticky = null;                           // 양쪽이 다시 맞춰졌다
  const y = a.el.getBoundingClientRect().top + window.scrollY - 18;
  if (Math.abs(y - window.scrollY) < 8) return;
  scrollNoteTo(y, true);
  pulse('pin');                            // 슬라이드가 기준선을 넘어 구간을 바꿨다는 신호
}

function syncPdfToNote() {
  const n = pageFromPdf();
  setCurrent(n);
  alignNoteToPage(n);
}

function wireSync() {
  let noteTick = false;
  window.addEventListener('scroll', () => {
    if (consumeProg('note', window.scrollY)) return;
    if (noteTick) return;
    noteTick = true;
    requestAnimationFrame(() => { noteTick = false; syncNoteToPdf(); });
  }, { passive: true });

  let pdfTick = false;
  // 사용자가 슬라이드를 직접 넘겨도 따라가기를 끄지 않는다.
  // 구간을 벗어나지 않는 한 아무 일도 일어나지 않고, 벗어나면 그건 "다음 주제로
  // 넘어갔다"는 뜻이므로 노트가 따라가는 게 맞다.
  ui.scroll.addEventListener('scroll', () => {
    if (consumeProg('pdf', ui.scroll.scrollTop)) return;
    if (pdfTick) return;
    pdfTick = true;
    requestAnimationFrame(() => { pdfTick = false; syncPdfToNote(); });
  }, { passive: true });

  window.addEventListener('resize', debounce(() => {
    if (window.innerWidth < MIN_WIDTH) return;
    relayoutWidth();
  }, 250));
}

function setFollow(on) {
  autoFollow = on;
  ui.follow.classList.toggle('on', on);
  ui.follow.textContent = on ? L('따라가기', 'follow') : L('따라가기 꺼짐', 'follow off');
}

// ── 크롬 (버튼, 폭 조절) ────────────────────────────

function wireChrome() {
  ui.prev.addEventListener('click', () => { goToPage(curPage - 1); alignNoteToPage(curPage); });
  ui.next.addEventListener('click', () => { goToPage(curPage + 1); alignNoteToPage(curPage); });
  ui.follow.addEventListener('click', () => {
    setFollow(!autoFollow);
    if (autoFollow) { sticky = null; syncNoteToPdf(true); }   // 켜면 지금 읽는 곳으로 맞춘다
  });
  ui.close.addEventListener('click', () => {
    ui.root.hidden = true;
    ui.openBtn.hidden = false;
    document.body.classList.remove('rdr-on');
  });
  ui.openBtn.addEventListener('click', () => {
    ui.root.hidden = false;
    ui.openBtn.hidden = true;
    document.body.classList.add('rdr-on');
    relayoutWidth();
    syncNoteToPdf(true);
  });

  // 기준선 바늘 드래그.
  // 끌면서 바로 동기화를 다시 판정한다. 바늘이 제목을 지나는 순간 슬라이드가
  // 넘어가는 게 보여야 이 장치가 뭘 하는지 알 수 있다.
  placeNeedle();
  let needleDrag = false;
  ui.needle.addEventListener('pointerdown', e => {
    needleDrag = true;
    ui.needle.setPointerCapture(e.pointerId);
    document.body.classList.add('rdr-needle-drag');
    e.preventDefault();
  });
  ui.needle.addEventListener('pointermove', e => {
    if (!needleDrag) return;
    const r = e.clientY / window.innerHeight;
    followLine = Math.max(FOLLOW_MIN, Math.min(FOLLOW_MAX, r));
    placeNeedle();
    syncNoteToPdf();
  });
  const endNeedle = e => {
    if (!needleDrag) return;
    needleDrag = false;
    try { ui.needle.releasePointerCapture(e.pointerId); } catch {}
    document.body.classList.remove('rdr-needle-drag');
  };
  ui.needle.addEventListener('pointerup', endNeedle);
  ui.needle.addEventListener('pointercancel', endNeedle);

  // 슬라이드쪽 기준선 드래그. 노트쪽과 같은 방식이되 단위가 %가 아니라 px다.
  // 페이지 열의 위에서부터의 거리라서 패널 높이가 바뀌어도 의미가 유지된다.
  placePin();
  let pinDrag = false;
  ui.pin.addEventListener('pointerdown', e => {
    pinDrag = true;
    ui.pin.setPointerCapture(e.pointerId);
    document.body.classList.add('rdr-pin-drag');
    e.preventDefault();
  });
  ui.pin.addEventListener('pointermove', e => {
    if (!pinDrag) return;
    const r = ui.scroll.getBoundingClientRect();
    pdfLine = Math.max(8, Math.min(r.height * 0.7, e.clientY - r.top));
    placePin();
    syncPdfToNote();
  });
  const endPin = e => {
    if (!pinDrag) return;
    pinDrag = false;
    try { ui.pin.releasePointerCapture(e.pointerId); } catch {}
    document.body.classList.remove('rdr-pin-drag');
  };
  ui.pin.addEventListener('pointerup', endPin);
  ui.pin.addEventListener('pointercancel', endPin);

  // 폭 조절
  let dragging = false;
  ui.grab.addEventListener('pointerdown', e => {
    dragging = true;
    ui.grab.setPointerCapture(e.pointerId);
    document.body.classList.add('rdr-resizing');
  });
  ui.grab.addEventListener('pointermove', e => {
    if (!dragging) return;
    const w = Math.max(340, Math.min(880, window.innerWidth - e.clientX));
    document.documentElement.style.setProperty('--rdr-w', `${Math.round(w)}px`);
  });
  const endDrag = e => {
    if (!dragging) return;
    dragging = false;
    try { ui.grab.releasePointerCapture(e.pointerId); } catch {}
    document.body.classList.remove('rdr-resizing');
    relayoutWidth();
  };
  ui.grab.addEventListener('pointerup', endDrag);
  ui.grab.addEventListener('pointercancel', endDrag);
}

// 패널 폭이 바뀌면 페이지 상자 크기를 다시 계산하고 렌더를 버린다.
const relayoutWidth = debounce(() => {
  if (!pages.length) return;
  const width = ui.scroll.clientWidth - 24;
  const keep = curPage;
  for (const p of pages) {
    const base = p.page.getViewport({ scale: 1 });
    p.vp = p.page.getViewport({ scale: width / base.width });
    p.div.style.width = `${Math.round(p.vp.width)}px`;
    p.div.style.height = `${Math.round(p.vp.height)}px`;
    try { p.task?.cancel(); } catch {}
    p.div.querySelector('canvas')?.remove();
    p.rendered = false;
  }
  // 보이는 것부터 다시 그린다
  pages.forEach(p => { renderObserver.unobserve(p.div); renderObserver.observe(p.div); });
  goToPage(keep, false);
}, 120);

function debounce(fn, ms) {
  let t;
  return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
}
