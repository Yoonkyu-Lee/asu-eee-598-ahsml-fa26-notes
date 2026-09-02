// 로컬에서 사이트를 띄운다. 슬라이드 리더까지 보려면 서버로 열어야 한다.
//
// 주의: `python -m http.server` 로는 리더가 안 뜬다.
// pdf.js 를 `import()` 로 받아오는데, python 이 .mjs 를 text/javascript 로 안 내보내서
// 브라우저가 모듈 로드를 거부한다. ("Failed to fetch dynamically imported module")
// 그래서 MIME 을 직접 지정하는 이 서버를 쓴다. verify.mjs 안의 서버와 같은 설정이다.
//
// 사용법: node scripts/serve.mjs [포트]   (기본 4599)

import { createServer } from 'node:http';
import { createReadStream, existsSync, statSync } from 'node:fs';
import { resolve, extname, sep } from 'node:path';

const PORT = Number(process.argv[2]) || 4599;
const ROOT = process.cwd();
const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.pdf': 'application/pdf',
  '.png': 'image/png',
  '.svg': 'image/svg+xml',
  '.json': 'application/json',
  '.txt': 'text/plain; charset=utf-8',
};

createServer((req, res) => {
  let rel = decodeURIComponent(req.url.split('?')[0]).replace(/^\/+/, '');
  if (rel === '') rel = 'index.html';
  const file = resolve(ROOT, rel);
  // 루트 밖으로 나가는 요청은 막는다
  if (!(file === ROOT || file.startsWith(ROOT + sep)) || !existsSync(file) || !statSync(file).isFile()) {
    res.writeHead(404, { 'content-type': 'text/plain; charset=utf-8' }).end('not found: ' + rel);
    return;
  }
  res.writeHead(200, { 'content-type': MIME[extname(file).toLowerCase()] || 'application/octet-stream' });
  createReadStream(file).pipe(res);
}).listen(PORT, '127.0.0.1', () => {
  console.log(`http://127.0.0.1:${PORT}/`);
});
