// 슬라이드 PDF의 쪽별 텍스트를 찍는다. data-slide 앵커를 PDF 쪽 기준으로
// 맞추기 위한 것. pptx 슬라이드 번호와 PDF 쪽 번호는 숨김 슬라이드 때문에 어긋난다.
import { readFileSync } from 'node:fs';
const { getDocument } = await import('pdfjs-dist/legacy/build/pdf.mjs');
const f = process.argv[2];
const d = await getDocument({ data: new Uint8Array(readFileSync(f)) }).promise;
for (let i = 1; i <= d.numPages; i++) {
  const p = await d.getPage(i);
  const tc = await p.getTextContent();
  const txt = tc.items.map(it => it.str).join(' ').replace(/\s+/g, ' ').trim();
  console.log(`\n--- p${i} ---\n${txt.slice(0, 900)}`);
}
