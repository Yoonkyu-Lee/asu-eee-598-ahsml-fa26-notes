# ASU EEE / CEN 598 — Hardware and Systems for Machine Learning (Fall 2026)

개인 학습 노트. https://yoonkyu-lee.github.io/asu-eee-598-ahsml-fa26-notes/

## 폴더 구조

| 경로 | 내용 |
|---|---|
| `docs/` | **공개 사이트** (GitHub Pages 소스). 이 안의 경로가 그대로 사이트 주소가 된다 |
| `docs/index.html`, `docs/glossary.html` | 목차, 용어집 |
| `docs/lectures/` | 강의 노트 (`L01-….html` …) |
| `docs/readings/` | 논문 읽기 노트 (`PR1-….html` …) |
| `docs/slides/` | 노트 옆 리더가 띄우는 강의 슬라이드 PDF |
| `docs/assets/` | 리더(`reader.js`, `reader.css`), pdf.js(`vendor/`), 그림(`img/`) |
| `docs/404.html` | 예전 루트 주소(`/L07-….html` 등)를 새 주소로 보낸다 |
| `hw/` | 과제 |
| `project/` | 파이널 프로젝트 |
| `scripts/` | 노트 검증·렌더링 도구와 Node 의존성 (`package.json`, `node_modules/`) |
| `materials/` | 강의 원본 PDF, 실라버스 (git에 안 올림) |
| `papers/` | 논문 PDF (git에 안 올림) |

## 도구

저장소 루트에서 실행한다. 처음 한 번은 `cd scripts && npm install`.

```bash
node scripts/serve.mjs                                  # docs/ 를 로컬 서버로 띄우기
node scripts/verify.mjs docs/lectures/L10-dataflow.html # 노트 검증
node scripts/render-slides.mjs docs/slides/L10-dataflow.pdf 1-3
```
