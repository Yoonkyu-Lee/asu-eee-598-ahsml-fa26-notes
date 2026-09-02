// 슬라이드 PDF의 쪽 수를 찍는다. data-slide 앵커를 달기 전에 확인용.
import { readFileSync, readdirSync } from 'node:fs';
const { getDocument } = await import('pdfjs-dist/legacy/build/pdf.mjs');
for (const f of readdirSync('slides').filter(n => n.endsWith('.pdf')).sort()) {
  const d = await getDocument({ data: new Uint8Array(readFileSync('slides/' + f)) }).promise;
  console.log(f, d.numPages);
}
