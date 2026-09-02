# EEE/CEN 598 AHSML 학습 노트 프로젝트

## 이 저장소가 하는 일

ASU EEE/CEN 598 (Hardware and Systems for Machine Learning, Jeff (Jun) Zhang, Fall 2026)
강의 슬라이드를 **한 장도 빠뜨리지 않고** 분해해서, ML을 아는 사람과 하드웨어를 아는 사람
어느 쪽이 읽어도 나머지 절반이 채워지는 개인 학습 노트 HTML로 재구성한다.
노트 옆에 원본 슬라이드 PDF를 띄우고 스크롤을 동기화해서, 지금 읽는 문단이
슬라이드 몇 쪽에서 나온 것인지 바로 보이게 한다.
GitHub Pages로 배포된다: https://yoonkyu-lee.github.io/asu-eee-598-ahsml-fa26-notes/

이 과목의 목표는 강사의 표현대로 **"a bridge between machine learning and hardware"** (L01 s2)다.
노트도 그 다리여야 한다. ML 쪽만 설명하거나 하드웨어 쪽만 설명하면 이 노트는 실패한 것이다.

## 사용자 배경

사용자(YK)는 UIUC Computer Engineering 학부를 졸업하고 ASU에서 MS 중인 한국인 학생이다.
디지털 설계 / 반도체 배경이 있고, ECE 385, 391, 408, 411, 444를 들었다.

**그리고 결정적으로, UIUC ECE 479 (IoT and Cognitive Computing, SP26)를 이미 들었다.**
`D:\Library\01 Immigration Documents\01 UIUC\4-2 SP26\ECE 479`

**이게 이 노트의 전제를 바꾼다.** 이 수업의 주제 목록 중 절반 이상을 사용자가 이미 봤다.
"처음 배우는 사람"을 상정하고 쓰면 노트의 대부분이 낭비된다.

### ECE 479에서 이미 본 것

강의는 5블록이었다: IoT 기초(L1-5) → 고전 ML(L6-8) → Deep Learning(L9-13) →
**HW 가속(L15-21)** → 보안(L22-24). 뒤 두 블록이 EEE 598과 정면으로 겹친다.

| EEE 598 예고 주제 | ECE 479에서 본 것 | 노트가 할 일 |
|---|---|---|
| DNN basics, training/inference | Lec08-09 (perceptron, MLP, backprop, CNN), conv 출력 dim·파라미터 수 공식 | **다시 안 가르친다.** 연결만 짚고 지나간다 |
| Roofline, memory hierarchy | Lec19. OI, ridge point, mem-bound vs compute-bound **완전히 익혔다** | 복습 대신 **왜 이 모델이 그렇게 생겼는지**로 들어간다 |
| Quantization | Lec12. PTQ vs QAT, STE, Deep Compression 3단, BNN XNOR-popcount | **INT8 너머**가 새 내용이다. per-channel, outlier, LLM 양자화 |
| Pruning, compression | Lec12. magnitude pruning, retrain 필수, KD, FC가 conv보다 redundant | 구조적 sparsity와 하드웨어 지원이 새 내용 |
| Systolic array | Lec11에서 Edge TPU를 systolic array로 소개받음. **개념만** | **여기가 진짜 새 내용이다.** dataflow 분류, 매핑, 재사용 분석 |
| GPU 아키텍처 | Lec15-16. warp, divergence, shared memory tiling, Tensor Core, FP32/FP16/BF16/TF32/FP8 | 다시 안 가르친다. 숫자 형식 표는 링크로 대신한다 |
| FPGA / HLS | Lec17-18. LUT/FF/BRAM/DSP, `PIPELINE II`, `ARRAY_PARTITION`, tiling, critical path | 다시 안 가르친다 |
| Compilation, tiling, mapping | Lec19에서 Tm·Tn unroll 수준까지 | **스케줄링 공간 탐색과 dataflow 표기법**이 새 내용 |
| TinyML, TFLite | Lec11. TFLite FlatBuffer, Edge TPU INT8 전용, 4 TOPS@2W | 다시 안 가르친다 |
| Transformer, LLM | Lec21. attention, √d_k, multi-head, O(L²d), BERT vs GPT | **LLM 추론 최적화**(KV cache, batching, serving)가 새 내용 |
| Near-data processing | **안 봤다** | 처음부터 쓴다 |
| Reliability, fault tolerance | Lec22-24는 보안이었고 fault tolerance는 아니다 | 처음부터 쓴다 |
| AutoML, NAS-HW codesign | Lec19에 co-design 개념만 | 거의 새 내용 |
| MLaaS, serving system | **안 봤다** | 처음부터 쓴다 |

**규칙: 이미 본 것은 `.callout`으로 "ECE 479 Lec19에서 roofline으로 본 그 이야기다"라고
한 줄 짚고, 지면은 새 내용에 쓴다.** 익숙한 것을 다시 설명하면 노트를 안 읽게 된다.

**단, 두 가지 예외가 있다.**
1. **학부 수준과 대학원 수준의 깊이가 다른 곳.** ECE 479는 시험용 공식으로 배웠다
   (`Separable / Standard = 1/d_j + 1/k²`). EEE 598은 **왜 그 식이 나오는지와
   그 결과 하드웨어가 어떻게 생겨야 하는지**를 묻는다. 여기는 다시 판다.
2. **1차 문헌.** ECE 479는 슬라이드로 배웠고 논문을 안 읽었다.
   EEE 598은 원 논문이 과제다. **개념은 알아도 그 개념이 어느 논문에서 어떻게 나왔는지는
   모른다.** 노트가 개념과 논문을 이어줘야 한다.

### 실제로 해본 것

ECE 479 Lab 3에서 **end-to-end edge 배포를 끝까지 해봤다.**
(`ECE 479\lab3`, 부산물로 Tauri 데스크톱 앱 `lab3\sinograph_explorer`)

98,169-class CJK 문자 인식기를 PyTorch(RTX 4080) → Keras → TFLite INT8 →
Edge TPU 컴파일 → Raspberry Pi 5 + Coral USB로 배포했다.

**이 경험이 이 과목의 핵심 교훈을 이미 몸으로 겪은 것이다.**

- **on-chip SRAM 용량이 전부를 결정한다.** v3(59 MB INT8)는 Coral의 8 MiB SRAM에
  안 들어가서 **87%가 USB로 스트리밍**됐고, 그 결과 matmul 가속이 PCIe latency에 묻혔다.
  "연산보다 데이터 이동이 비싸다"를 슬라이드가 아니라 실측으로 겪었다.
- **양자화는 조용히 망가진다.** 50 MB FC head가 symmetric INT8 calibration에서
  전 입력이 한 토큰으로 붕괴했다. **PTQ가 왜 layer마다 다르게 동작하는지**의 실물 사례다.
- **아키텍처를 바꿔서 푼다.** classifier head를 128-d ArcFace embedding + cosine NN으로
  바꿔서 11 MB / 41-41 ops mapped / INT8 손실 0.00pp를 얻었다.
- **Pareto로 논증할 줄 안다.** Tesseract, EasyOCR, cnocr, Manga-OCR과
  size × latency × accuracy × coverage로 비교했다.

**노트를 쓸 때 이 경험을 기준점으로 쓴다.** 특히 `.scale`(숫자 감각)에서 그렇다.
"8 MiB on-chip이면 무엇이 안 들어가는가"를 사용자는 이미 안다. 거기서 출발하면 된다.

**다만 남의 수업 결과물이므로 이 저장소에 코드나 수치를 옮겨 싣지 않는다.**
비유와 기준점으로만 쓰고, 필요하면 "직접 해봤던 그 상황"이라고 언급하는 선에서 그친다.

## 앞선 두 노트와 무엇이 다른가

이 프로젝트는 `D:\Engineering\asu-cen-598-addv-fa26-notes`의 구조를 물려받았고,
그쪽은 다시 `D:\Engineering\asu-eee-554-notes`에서 왔다.
리더, 검증 스크립트, 디자인 토큰, 이중 언어 장치는 같은 것을 쓴다.
**하지만 노트가 채워야 하는 간극이 다르고, 결정적으로 학사 제약이 다르다.**

| | EEE 554 (확률) | CEN 598 (ADDV) | **EEE 598 (AHSML)** |
|---|---|---|---|
| 강사 | Hajek 교재 기반 | Aman Arora | **Jeff (Jun) Zhang** |
| 막히는 지점 | 수식이 안 읽힘 | 용어가 설명 없이 지나감 | **개념 절반은 ECE 479에서 이미 봤다.** 없는 건 깊이와 1차 문헌 |
| 선수 노출 | 없음 | 없음 | **UIUC ECE 479를 들었다.** 아래 [사용자 배경](#사용자-배경) 참조 |
| 핵심 장치 | `.nota` 기호 뜯어보기 | `.jargon` 용어 뜯어보기 | **`.nota` + `.jargon` + `.scale` 셋 다** |
| 평가 | 숙제 + 시험 | Lab 45 / Exam 30 / Quiz 20 / Part 5 | **Reading 15 / HW 25 / Project 50 / Part 10** |
| AI 도구 | 언급 없음 | **허용** (사용 내역 제출) | **전면 금지.** 아래 참조 |
| 교재 | Hajek 한 권 | 없음 | **세 권 있음** |
| 슬라이드 | PDF 배포 | PPTX → 변환 필요 | **PDF 배포. 변환 불필요** |
| 이 저장소가 다루는 것 | 모듈 노트 + HW 공략 | 강의 노트 + Lab 공략 | **강의 노트 + Paper Reading 보조.** HW·Project는 안 다룬다 |

앞선 두 저장소의 규약을 기계적으로 복사하지 않는다. 위 표에서 갈라지는 것은 이 문서가 이긴다.

## 이 저장소의 최상위 제약: AI 사용 금지

**이게 다른 무엇보다 먼저다.** 실라버스가 이렇게 적고 있다.

> Use of AI Tools Not Permitted. This course's assignments are designed for you to demonstrate
> your independent learning without assistance from Artificial Intelligence (AI) tools. Unless an
> assignment explicitly states otherwise, the use of AI tools is prohibited for any course
> assignments.

그리고 Paper Reading #1 Canvas 공지가 한 번 더 못을 박았다.

> Any use of ChatGPT or any generative AI tool for the paper review form will
> automatically receive a ZERO grade.

CEN 598은 AI를 허용하고 사용 내역만 적게 했다. **이 과목은 정반대다.**
앞 저장소의 습관을 그대로 가져오면 학사 제재로 직결된다.

### 선이 어디인가

**사용자와 합의한 범위다. 임의로 넓히지 않는다.**

| 대상 | 이 저장소가 하는 일 |
|---|---|
| 강의 슬라이드 | **블로그화한다.** 제출물이 아니라 개인 학습 노트다. AI가 개입해도 되는 영역 |
| Paper Reading (15%) | **리뷰를 대신 쓰지 않는다.** 읽는 법, 막힌 부분 설명, 사용자가 쓴 글의 영어 로컬라이징까지 (아래 [Paper Reading 처리 원칙](#paper-reading-처리-원칙) 참조) |
| Homework (25%) | **다루지 않는다.** 공략 페이지도 만들지 않는다. "Done solo"라고 슬라이드가 못박았다 (L01 s9) |
| Research Project (50%) | **다루지 않는다.** 3인 팀 과제이고 논문 제출까지 가는 결과물이다 |

**Homework와 Project를 다뤄달라는 요청이 나중에 오면, 그때 다시 확인한다.**
이 문서를 조용히 고쳐서 범위를 넓히지 않는다.

### 노트가 과제에 새는 것도 막는다

강의 노트는 허용 영역이지만, **그 안에 과제 답이 들어가면 경계가 무너진다.**

- 노트의 예제 문제는 **슬라이드에 있는 것이나 교재의 표준 예제**로 만든다.
  Homework 문제와 숫자만 바꾼 것을 만들지 않는다.
- Homework 문제지를 읽고 그것에 맞춰 노트를 쓰지 않는다.
  노트는 슬라이드를 따라간다. 우연히 겹치는 것과 겨냥하는 것은 다르다.
- **논문 리뷰 본문에 들어갈 문장을 노트에 미리 써두지 않는다.** 우회로가 된다.

## 경로

| 용도 | 경로 |
|---|---|
| 저장소 | `D:\Engineering\asu-eee-598-ahsml-fa26-notes` |
| 강의 자료 원본 (읽기 전용) | `D:\Library\01 Immigration Documents\02 ASU\FA26\EEE 598  Advanced Hardware and Systems for Machine Learning` |
| 강의 운영 | ASU Canvas (Announcements, 슬라이드, 과제, Readings), Ed Discussion |
| **선수 과목 자료 (읽기 전용)** | `D:\Library\01 Immigration Documents\01 UIUC\4-2 SP26\ECE 479` |
| 앞선 노트 저장소 (틀의 출처) | `D:\Engineering\asu-cen-598-addv-fa26-notes`, `D:\Engineering\asu-eee-554-notes` |

ECE 479 폴더에서 볼 만한 것:

| 경로 | 쓸모 |
|---|---|
| `Lecture/Lecture_Index.md` | 24강 전체 요약. **어느 주제를 어느 깊이로 봤는지 여기서 확인한다** |
| `Lecture/notes/` | 강의별 상세 노트 |
| `Final_Cheatsheet.md` | 공식과 함정 목록. 사용자가 실제로 외운 것 |
| `Lecture/Lec*.pdf` | 원본 슬라이드. 그림이 필요할 때만 |
| `lab3/README.md` | end-to-end edge 배포 프로젝트 기록 |

**읽기 전용이다.** 그리고 **필요할 때만 펼친다.** 새 강의를 쓰기 전에
`Lecture_Index.md`와 `Final_Cheatsheet.md`만 확인해서 "이건 이미 봤나"를 판정하면 충분하다.
매번 전부 훑지 않는다.

경로의 `EEE 598` 뒤에 **공백이 두 칸**이다. 자동완성 없이 손으로 적으면 틀린다.

Drive 미러 폴더는 **절대 수정하지 않는다.** 읽기만 한다.

**Drive 원본은 학기 중에 갱신된다.** 슬라이드가 추가되고, Paper Reading 폴더가 주마다 늘어난다.
**노트를 쓰거나 고치기 전에 반드시 폴더를 다시 훑고 쪽수를 대조한다.**
낡은 사본으로 앵커를 달면 전부 한 칸씩 어긋난다.

```bash
node scripts/pagecount.mjs
```

### 슬라이드가 PDF로 온다

CEN 598은 PPTX라 변환 스크립트와 숨김 슬라이드 처리가 필요했다. **여기는 PDF다.**
`pptx2pdf.ps1`, `pptx-text.py`, `hidden-slides.py`를 이 저장소로 가져오지 않았다.
**발표자 노트도 없다.** CEN 598에서 발표자 노트가 채워주던 자리를 여기서는
교재와 Canvas 공지가 대신한다.

Drive의 PDF를 `slides/`로 복사하면서 노트와 stem을 맞춰 이름을 바꾼다.
`EEE 598 - Lecture 1 - Introduction.pdf` → `slides/L01-introduction.pdf`.

## 교재와 내용의 근거

**CEN 598과 달리 이 과목은 교재가 있다.** 실라버스가 셋을 지정했다.

| 약칭 | 책 |
|---|---|
| **Sze** | *Efficient Processing of Deep Neural Networks*, Sze, Chen, Yang, Emer, 2020 (Synthesis Lectures, 온라인) |
| **Goodfellow** | *Deep Learning*, Goodfellow, Bengio, Courville, 2016 |
| **Nielsen** | *Neural Networks and Deep Learning*, Nielsen, 2015 (온라인) |

**Sze가 이 과목의 중심 교재다.** 슬라이드의 accelerator, dataflow, 에너지 분석은
사실상 Sze의 서술을 따라간다. L02-3 슬라이드는 Goodfellow의 그림을 직접 인용한다
(s4, s5, s8에 "Deep Learning, Goodfellow et. al." 이 찍혀 있다).

검증 근거의 순서는 이렇다.

1. **슬라이드 PDF.** 1차 근거.
2. **지정 교재.** 슬라이드가 줄여 쓴 것을 펴는 곳. 슬라이드에 출처가 찍혀 있으면 그 책을 따라간다.
3. **Canvas 공지.** 슬라이드보다 **최신이다.** 아래 참조.
4. **원 논문.** 슬라이드가 인용한 논문 (Golden Age, AlexNet, Eyeriss, TPU 논문 등).
   숫자와 용어의 최종 심판.

**슬라이드 밖에서 가져온 것은 문맥으로 그게 보충이라는 걸 알 수 있게 쓴다.**
`data-slide`를 안 붙이는 것으로 이미 절반은 표시되지만, 본문에서도 "슬라이드에는 그림만
있는데, Sze 책이 이 그림을 이렇게 설명한다" 같은 식으로 밝힌다.

### 슬라이드는 재활용 덱이다

**L01 footer에 `2025 Fall`, L02-3 footer에 `2023 Fall`이 찍혀 있다.**
강사가 이전 학기 덱을 그대로 쓴다. 그래서:

- **날짜, 마감, 정책은 슬라이드를 믿지 않는다.** Canvas 공지와 실라버스가 이긴다.
  실제로 슬라이드 표지의 날짜(8/20, 8/25)는 2026 학기 실제 강의일과 다를 수 있다.
- 숫자가 오래된 것도 있다. L01 s38의 V100은 2017년 GPU다. **슬라이드가 그 시점의 예시로
  쓴 것이므로 최신 칩으로 바꿔 적지 않는다.** 대신 "지금 기준으로는 어디쯤인지"를
  보충으로 한 줄 붙인다. 그건 슬라이드 밖 내용이므로 앵커를 달지 않는다.
- **연도가 찍힌 그래프는 그래프가 끝나는 지점을 노트에 밝힌다.** 독자가 "지금도 이런가"를
  묻게 되는 지점이다.

## 파일 명명

```
index.html                       허브 (강의 목록)
glossary.html                    약어·용어 사전
L01-introduction.html            Lecture 1
L02-ml-basics.html               Lecture 2-3
L{NN}-{kebab-case-영문주제}.html

PR{N}-{kebab-case-주제}.html     Paper Reading 노트 (아래 참조)

slides/L01-introduction.pdf      Lecture 1 슬라이드 (노트와 같은 stem)
slides/L02-ml-basics.pdf

reader.js  reader.css            슬라이드 리더 (모든 노트가 공유)
vendor/pdf.js/                   pdf.js 런타임
```

`{NN}`은 **슬라이드 파일이 스스로 밝히는 Lecture 번호**를 따른다.
**한 PDF가 두 강의를 덮으면 앞 번호를 stem으로 쓰고 제목에 범위를 적는다.**
`EEE 598 - Lecture 2-3 - ML Basics.pdf` → `L02-ml-basics.html`, 제목은 `Lecture 2-3`.
파일을 둘로 쪼개지 않는다. 슬라이드가 하나면 리더도 하나다.

### scripts/

| 스크립트 | 하는 일 |
|---|---|
| `verify.mjs` | 노트 검증. exit 0 이어야 통과 |
| `pagecount.mjs` | `slides/*.pdf`의 쪽 수 |
| `pdftext.mjs` | PDF **쪽별** 텍스트. `data-slide` 앵커를 달기 전 대조용 |
| `render-slides.mjs` | PDF 쪽을 PNG로 렌더 → `shots/slides/` |
| `crop.mjs` | 스크린샷 일부만 잘라 보기 |
| `serve.mjs` | 로컬 서버. 리더 확인용 |

## 한 슬라이드도 건너뛰지 않는다

**이 저장소가 앞선 두 프로젝트와 가장 크게 다른 작성 규칙이다.**

앞선 노트들은 슬라이드를 개념 단위로 묶어서 요약했다. **여기서는 그렇게 하지 않는다.**

- **모든 슬라이드가 노트 어딘가에서 실제로 다뤄져야 한다.** 묶어서 지나가지 않는다.
  "s25~s33은 응용 사례들이다" 한 줄로 아홉 장을 처리하는 것이 금지 대상이다.
- **그림만 있는 슬라이드가 이 규칙의 핵심 대상이다.** 이 덱은 그림 전용 슬라이드가 많다
  (L01 s17, s23, s25, s28~s33, s35~s41). 텍스트 추출로는 제목 한 줄밖에 안 나온다.
  **렌더해서 눈으로 보고, 그림이 무엇을 보여주는지 서술한다.**
  - 축이 무엇이고 단위가 무엇인지
  - 어느 구역에 무엇이 있는지 (좌상단 / 가운데 / 우하단)
  - 그래프면 어디서 꺾이는지, 기울기가 어디서 바뀌는지
  - 사진이면 무엇을 찍은 것이고 어디를 보라는 것인지
  - 그리고 **강사가 이걸 왜 여기 놨는지**
- **분량이 길어도 된다. 잡담이 섞여도 된다.** 이 수업은 AI 하드웨어의 세계관을 얻으려고
  듣는 것이다. 배경 이야기가 본론일 때가 있다.
- **단, 톤은 유지한다.** 앞선 두 프로젝트의 "이해가 쉽도록 하는 대화 톤"이다.
  길다는 것이 딱딱해도 된다는 뜻이 아니다. 길수록 더 말하듯이 써야 읽힌다.

### 그래도 노트로 옮기지 않는 것

로지스틱스 슬라이드는 예외다. Office hours 시간, Ed Discussion 사용법, 강사 이메일,
Apporto 접속 안내, 성적 분포표(L01 s14) 같은 것.

**`index.html`의 공지 배너**에 모으고 강의 노트에는 넣지 않는다. 유효기간이 있는 정보다.
그 쪽들은 앵커가 안 걸리므로 리더에서 `노트에 없음` 배지가 붙는다. 이게 정상이다.

**다만 학습에 영향을 주는 로지스틱스는 노트에 넣는다.** 예를 들어:
- L01 s9 (평가 비중): Project 50%라는 사실이 "이 수업을 어떻게 들어야 하는가"를 정한다.
- L01 s10 (Paper Reading 여섯 질문): 이게 사실상 리뷰 작성 가이드다. 반드시 노트에 넣는다.
- L01 s12-13 (PDK와 컴퓨팅 자원): Project에서 실제로 쓸 것이다.
- L01 s16 (Top conferences): 논문을 읽을 때 어느 학회 것인지가 곧 신뢰도 정보다.

경계는 이렇다. **"학기가 끝나도 쓸모가 있는가."** 있으면 노트, 없으면 배너.

## 언어와 문체

- **모든 본문은 한국어.** 반말체, 설명하듯이. "~야", "~거야", "~돼".
- **em dash(—) 금지.** 콜론, 쉼표, 마침표를 쓴다. (SVG 안이나 구분선 용도는 예외)
- **업계 용어와 약어는 영어를 유지한다.** inference, training, quantization, pruning, dataflow,
  systolic array, roofline, tiling, dequantization, sparsity, MAC, PE, TOPS, HBM, SRAM.
  **번역해서 쓰면 안 된다.** 사용자가 실제로 부딪힐 논문, 코드, 면접은 전부 영어다.
  "수축기 배열"처럼 옮기지도 않고 "시스톨릭"처럼 음차하지도 않는다.
  영어 그대로 쓰고 처음 나올 때 `.jargon`으로 푼다.
- 반면 **일반 명사는 한국어로 쓴다.** "여기서 데이터를 다시 읽어야 한다", "이 지점에서 갈린다".
  용어는 영어, 서술은 한국어. 이 경계를 흐리지 않는다.
- 과장 금지. "놀랍게도", "혁명적으로" 같은 수사보다 정확한 서술.
- **논문 제목과 저자는 원문 그대로.** "AlexNet 논문"이라고만 쓰지 말고 처음 한 번은
  *ImageNet Classification with Deep Convolutional Neural Networks* (Krizhevsky et al., NIPS 2012)로
  적는다. 논문을 찾아갈 수 있어야 한다.

### 숫자를 옮길 때

이 과목은 숫자가 곧 논지다. **틀리면 논지가 뒤집힌다.**

- **슬라이드의 숫자를 반올림하지 않는다.** `46,225 mm²`를 "약 4.6만"으로 바꾸지 않는다.
- **단위를 붙인다.** TOPS인지 TOPS/W인지, FLOP인지 FLOP/s인지가 전부 다른 이야기다.
  슬라이드가 단위를 빠뜨렸으면 **빠뜨렸다고 적는다.** 추측해서 채우지 않는다.
- **출처 연도를 같이 적는다.** 위 [재활용 덱](#슬라이드는-재활용-덱이다) 참조.

## 강의 노트의 5단 구조

각 개념 섹션은 이 순서를 반복한다.

1. **목차와 요약** — 이 강의가 무슨 일을 하는지, 오늘 나오는 용어와 기호가 뭔지,
   나중에 어디서 쓰이는지. 섹션 00에 배치.
   **"오늘 나오는 약어"와 "오늘 나오는 기호" 두 목록**을 두고 그중 뭐가 제일 중요한지 명시.
   그리고 **"이미 아는 것 / 새로 나오는 것"을 갈라 놓는다.**
   ECE 479에서 본 것은 한 줄로 확인만 하고 넘어가고, 새것에 지면을 준다.
   독자가 00번만 읽고도 **어디를 건너뛸 수 있는지** 알아야 한다.
2. **정의와 뜯어보기** — 슬라이드의 정의를 그대로 `.def`로. 그리고 바로 뒤에:
   - 용어면 `.jargon`
   - 수식이면 `.nota`
   - 숫자면 `.scale`
3. **비주얼** — **데이터가 어디에 있고 어디로 움직이는지 그린다.**
   이 과목의 그림은 대부분 "무엇이 어느 메모리에 있고 어느 연산기로 가는가"다.
   그리고 **그림만 있는 원본 슬라이드를 말로 옮기는 자리**이기도 하다.
4. **감각 문제** — `<details>`로 답을 접어둔다.
   계산(이 layer의 MAC 수는?)과 판단(이건 memory bound인가 compute bound인가) 둘 다.
   **Homework 문제를 겨냥하지 않는다.** 위 [AI 사용 금지](#노트가-과제에-새는-것도-막는다) 참조.
5. **함정과 혼동** — 헷갈리는 이웃 용어, 단위 함정, 흔한 오해. `.trap` 블록.

마지막 섹션은 항상 **정리 & 다음 강의**: 한 장 요약 표 + 다음 강의 예고.

## 세 가지 뜯어보기 장치

앞선 두 저장소는 각각 하나씩만 썼다. **이 과목은 셋이 다 필요하다.**
용어를 몰라도 막히고, 수식을 몰라도 막히고, **숫자가 큰 건지 작은 건지 몰라도 막힌다.**

### `.jargon` — 용어 뜯어보기

CEN 598에서 그대로 가져온다. 사용자가 **"그 말이 어느 세계의 말인지 몰라서"** 막히는 것을 푼다.

```html
<details class="jargon" open>
  <summary><span lang="ko">용어 뜯어보기 · TOPS</span><span lang="en">Unpacking the jargon · TOPS</span></summary>
  <div class="body">
    <table class="parts">
      <tr><td>풀네임</td><td>Tera Operations Per Second</td></tr>
      <tr><td>한 줄</td><td>1초에 정수 연산을 몇 조 번 하는지</td></tr>
      <tr><td>언제</td><td>inference 칩의 성능을 광고할 때</td></tr>
      <tr><td>누가</td><td>칩 벤더의 스펙 시트. 논문에서는 잘 안 쓴다</td></tr>
      <tr><td>헷갈리는 것</td><td>TFLOPS와 다르다. 정수냐 부동소수점이냐가 갈린다. TOPS/W는 또 다른 지표다</td></tr>
    </table>
  </div>
</details>
```

- **기본은 펼쳐둔다(`open`).** 모르는 사람에게 안 보이면 있으나 마나다.
- **다섯 줄을 채운다.** 풀네임 / 한 줄 정의 / 언제 / 누가 / 헷갈리는 것.
- 약어는 **풀네임을 반드시 적는다.** MAC을 "곱셈 누적"이라고만 적고 넘어가지 않는다.
  Multiply-Accumulate라는 걸 알아야 논문에서 마주쳤을 때 읽힌다.
- 같은 용어를 두 번 풀지 않는다. **처음 나오는 곳에서 풀고, 뒤에서는 그리로 링크한다.**
  `verify.mjs`가 링크 대상이 실제로 있는지 검사한다.

**이 과목의 시그니처 함정은 "같은 단어가 ML 쪽과 하드웨어 쪽에서 다른 뜻인 경우"다.**

| 단어 | ML 쪽 | 하드웨어 쪽 |
|---|---|---|
| kernel | convolution filter | GPU에서 도는 함수 |
| batch | 한 번에 처리하는 샘플 수 | 무관 |
| bandwidth | 무관 | 초당 옮길 수 있는 바이트 |
| scaling | 모델을 키우는 것 | 공정 미세화 |
| inference | 학습 끝난 모델로 예측 | (같음) |
| latency | 응답 시간 | 파이프라인 단계 지연 |

**이런 짝은 `.jargon` 안에 묻지 말고 `.trap`으로 따로 뺀다.**

### `.nota` — 기호 뜯어보기

EEE 554에서 가져온다. **수식이 나오는 강의에만 쓴다.** L02-3의 cost function,
gradient descent, regularization이 첫 대상이다.

```html
<details class="nota" open>
  <summary><span lang="ko">기호 뜯어보기</span><span lang="en">Unpacking the notation</span></summary>
  <div class="body">
    <table class="parts">
      <tr><td class="m">J(θ)</td><td>파라미터 θ일 때의 cost. 학습이 줄이려는 값</td></tr>
      <tr><td class="m">∇<sub>θ</sub>J</td><td>θ 각 성분에 대한 J의 기울기. θ와 같은 모양의 벡터</td></tr>
      <tr><td class="m">α</td><td>learning rate. 기울기를 얼마나 믿고 움직일지</td></tr>
    </table>
  </div>
</details>
```

- **기호마다 한 줄씩.** 아래 첨자, 위 첨자, 굵은 글씨가 뭘 뜻하는지 빠뜨리지 않는다.
- **모양을 밝힌다.** 스칼라인지 벡터인지 행렬인지. 이 과목에서는 그게 곧
  "메모리에서 몇 바이트인가"로 이어진다.
- 슬라이드의 수식이 **PDF 텍스트 추출에서 깨진다.** L02-3 s13, s16, s17이 실제로 깨졌다
  (`= 0 + 1`처럼 첨자가 사라진다). **수식이 있는 쪽은 반드시 렌더해서 눈으로 읽는다.**

### `.scale` — 숫자 감각

**이 저장소가 새로 만드는 장치다.** 앞선 두 프로젝트에는 없다.

이 과목의 슬라이드는 숫자로 논증한다. `1.2 trillion transistors`, `46,225 mm²`,
`300W TDP`, `4 TOPS`, `2 TOPS/W`, `21 billion transistors`. **그런데 이 숫자가 큰 건지
작은 건지를 슬라이드가 알려주지 않는다.** 비교 대상이 없으면 아무 의미도 전달되지 않는다.

```html
<div class="scale">
  <table class="parts">
    <tr><td>Cerebras WSE 다이 면적</td><td class="m">46,225 mm²</td><td>A4 용지의 약 1/13, 손바닥만하다</td></tr>
    <tr><td>NVIDIA V100</td><td class="m">815 mm²</td><td>레티클 한계 근처. 보통 칩 중 가장 큰 축</td></tr>
    <tr><td>보통 모바일 SoC</td><td class="m">~100 mm²</td><td>기준점</td></tr>
  </table>
</div>
```

- **항상 기준점을 하나 이상 같이 준다.** 혼자 있는 숫자는 정보가 아니다.
- **기준점은 사용자가 이미 아는 것으로 잡는다.** 1순위는 **직접 만져본 것**이다.
  Coral Edge TPU (8 MiB on-chip SRAM, 4 TOPS @ 2W), Raspberry Pi 5 CPU,
  RTX 4080 Laptop, 11 MB INT8 모델, 25 ms/char.
  그 다음이 학부에서 본 것(V100, Jetson Xavier), 그 다음이 노트북·스마트폰.
  위 [실제로 해본 것](#실제로-해본-것) 참조.
- **비교는 같은 단위끼리만 한다.** mm²를 W와 비교하지 않는다.
- 슬라이드에 없는 기준점을 가져오는 것이므로 **`data-slide`를 붙이지 않는다.**
- **출처가 필요하면 밝힌다.** "보통 모바일 SoC ~100 mm²"처럼 어림수를 쓸 때는
  어림수라고 적는다. 정확한 숫자인 척하지 않는다.

## 약어 사전 (`glossary.html`)

`.jargon`은 **강의 → 용어** 방향이다. 읽다가 "아 MAC이 이거구나"를 알려준다.
**논문을 펴놓고 앉았을 때는 방향이 반대다.** "PE가 뭐였지"에 답하는 곳이 따로 필요하다.

이 과목은 특히 그렇다. **Paper Reading이 주마다 있고, 논문은 약어를 안 풀고 쓴다.**

- **알파벳순으로 정렬한다.** 강의순이 아니다.
- 항목마다: **풀네임 / 한 줄 정의 / 처음 나온 강의로 가는 링크.**
- **ML 쪽 약어와 하드웨어 쪽 약어를 섞어 둔다.** 분리하지 않는다.
  이 과목의 요점이 둘이 한 문장 안에 나온다는 것이다.
- 검색 상자를 둔다. 브라우저 저장소는 쓰지 않는다.
- `data-slide`를 붙이지 않고 `reader.js`도 넣지 않는다.
- **강의 노트를 쓸 때마다 같이 갱신한다.** 나중에 몰아서 하면 반드시 빠뜨린다.

## 함정 섹션 작성 원칙

- **혼동되는 이웃 용어를 짝으로 잡는다.**
  Training vs. Inference. Latency vs. Throughput. FLOPS vs. FLOP.
  Quantization vs. Pruning. Accelerator vs. GPU. Model size vs. Activation size.
  **"둘 다 맞는 말인데 가리키는 게 다른" 경우가 제일 위험하다.**
- **구별 기준을 하나로 못 박는다.** "언제 하는가"인지 "무엇을 줄이는가"인지 "누가 쓰는가"인지.
  기준을 안 밝히고 나열하면 읽고 나서도 구별이 안 된다.
- **ML 쪽 뜻과 하드웨어 쪽 뜻이 갈리는 단어는 반드시 짚는다.** 위 표 참조.
- **단위 함정을 짚는다.** 이 과목 고유의 함정이다.
  TOPS와 TOPS/W, FLOP과 FLOP/s, bit와 byte, GB와 GiB, 그리고 **MAC 1회 = FLOP 2회**.
- **"왜 이걸 배우는지"를 함정 섹션 마지막에 넣으면 효과가 좋다.**

## 강의 간 연결

**필수 작업이다.** 이 과목은 알고리즘 → 최적화 → 하드웨어 → 컴파일러 → 시스템으로
층을 올라간다. **아래층을 잊으면 위층이 왜 그렇게 생겼는지 설명이 안 된다.**

- `.callout` 블록으로 "Lecture N의 X가 여기서 일한다" 형태로 쓴다.
- **`.callout`은 ECE 479로도 건다.** "ECE 479 Lec19에서 roofline으로 본 그 이야기다"처럼.
  이게 지면을 아끼는 장치이자 사용자가 가장 빨리 이해하는 경로다.
  다만 **ECE 479 슬라이드 쪽수를 인용하지 않는다.** 강의 번호와 주제까지만.
  남의 수업 자료이고, 이 저장소는 Public이다.
- **L01 s42가 이 과목 전체의 지도다.** 이 슬라이드에 세 가지가 같이 있다.
  왼쪽에 층 셋 (better algorithms/applications, better runtimes/softwares, better hardware),
  오른쪽 재활용 기호의 세 꼭짓점에 지표 셋 (Accuracy, Performance, Efficiency),
  왼쪽 아래에 파란 원 세 개로 그린 선순환 (Bigger Data → Better Algorithm → More Compute → Bigger Data).
  **선순환은 s24에 이미 나온 그림을 다시 쓴 것이다.** 두 곳을 연결해서 쓴다.
  새 강의를 쓸 때 **"이 강의는 어느 층이고 어느 지표를 움직이는가"**를 00번 섹션에서 먼저 말한다.
- 마지막 정리 섹션에 이전 강의 ↔ 현재 강의 대응표를 넣으면 좋다.

## Paper Reading 처리 원칙

**Reading이 15%이고, 주마다 한두 편이 강의 전에 마감된다** (L01 s10).
이 저장소에서 가장 조심해야 하는 영역이다.

### 절대 하지 않는 것

**리뷰 폼의 내용을 대신 쓰지 않는다.** 초안도, 뼈대도, "이렇게 쓰면 어때"도 안 된다.
Canvas 공지가 자동 0점이라고 명시했다. 요약(A), 강점(B), 약점(C), 저자 코멘트(L),
질문(M), 개선안(N) 전부 사용자가 쓴다.

**점수 항목(D~I)을 대신 고르지 않는다.** Novelty 몇 점, Overall merit 몇 점을
제안하지 않는다. 그게 리뷰어의 판단이고 이 과제가 재려는 것이다.

사용자가 "그냥 초안만 잡아줘"라고 해도 **하지 않는다.** 그 요청은 거절하고
아래의 허용 항목으로 대신 돕는다. 이건 사용자와 합의된 규칙이다.

### 하는 것

| 역할 | 구체적으로 |
|---|---|
| **읽는 법 가이드** | 논문을 어떤 순서로 읽는지, 어디를 먼저 보는지, 무엇을 메모하는지 |
| **막힌 부분 설명** | "이 그림의 y축이 뭔지 모르겠어", "이 baseline이 왜 불공정하다는 거야" |
| **배경 지식** | 논문이 전제하는 선행 연구, 약어, 그 시절 맥락 |
| **영어 로컬라이징** | 사용자가 쓴 감상문을 **쉬운 영어로** 다듬는 것. 내용은 사용자 것 |
| **대화로 궁금증 풀기** | 직접·간접으로 연관된 주제 질의응답 |
| **기억해둘 지식 기록** | 대화에서 나온 값진 것을 메모리 문서로 남기고 블로그 컨텐츠에 넣기 |
| **리뷰 평가** | 사용자가 다 쓴 뒤에 읽고 평가한다. 아래 참조 |

**막히는 지점이 개념이 아니다.** 사용자는 ECE 479에서 이 분야의 개념을 이미 훑었다
(위 [사용자 배경](#사용자-배경) 참조). 그래서 "attention이 뭐야"에서 막히지 않는다.
**막히는 건 논문이라는 장르 자체다.** 어디부터 읽는지, 어느 그림이 논지이고 어느 그림이
장식인지, baseline이 공정한지 어떻게 판단하는지, "incremental"이라고 쓰려면 무엇을
근거로 대야 하는지. **읽는 법 가이드는 개념 설명이 아니라 이쪽에 지면을 쓴다.**

**로컬라이징의 선.** 사용자가 한국어나 어색한 영어로 쓴 문장을 자연스러운 영어로 옮기는 것은
번역이지 저술이 아니다. **하지만 문장을 늘리거나, 없던 논거를 넣거나, 문단을 재배열해서
논지를 강화하면 그건 저술이다.** 옮기고 나서 원문에 없던 주장이 생겼는지 확인한다.
생겼으면 지운다.

### 리뷰 평가와 블로그화

**강의 슬라이드와 순서가 반대다.** 슬라이드는 바로 블로그화하지만, 논문은 이렇게 간다.

1. 사용자가 논문을 읽고 리뷰를 쓴다.
2. **내가 읽고 평가한다.** 논문의 실제 내용과 대조해서 사실 오류가 있는지,
   놓친 핵심 기여가 있는지, 약점 지적이 근거 있는지.
3. 적절하다고 판단되면 **그때** `PR{N}-*.html`로 블로그화한다.
4. Canvas 점수가 나오면 사용자가 알려준다. **교수의 채점 성향을 추정해서 이 문서에 반영한다.**

**2번에서 리뷰를 고쳐주지 않는다.** 무엇이 어긋났는지 알려주고 사용자가 고친다.

### `PR{N}-*.html`에 무엇을 싣는가

**제출한 리뷰 원문을 그대로 싣지 않는다.** repo가 Public이라 인터넷에 게시하는 것이고,
같은 과제를 듣는 다른 학생이 볼 수 있다. 학문적 정직성 문제로 직결된다.

싣는 것:
- **논문 자체에 대한 정리.** 문제, 방법, 평가, 결과. 이건 학습 노트다.
- **읽으면서 막혔던 것과 어떻게 풀었는지.** 이게 이 페이지의 진짜 값어치다.
- **배경 지식.** 그 시절 맥락, 선행 연구, 이 논문 이후에 뭐가 나왔는지.
- **읽는 법에서 배운 것.** 다음 논문에 쓸 수 있는 것.

싣지 않는 것:
- 리뷰 폼 원문, 점수 항목의 선택, 제출한 코멘트
- **강사가 준 Sample review.** Canvas 공지가 "Please don't share it outside the class!!"라고
  명시했다. 인용도 하지 않는다.
- **논문 PDF.** 저작권이 있고 Drive에 있는 사본이다. **링크만 건다.**

`data-slide`를 붙이지 않고 `reader.js`도 넣지 않는다. 슬라이드를 다루는 페이지가 아니다.

### 리뷰 폼의 구조

과제는 HotCRP 형식의 텍스트 폼이다. 항목이 이렇다.

| 항목 | 내용 | 형식 |
|---|---|---|
| A | Paper summary (문제 1-2 / 핵심 통찰 1-2 / 기여 2-3) | 서술 |
| B | Strengths | 불릿 |
| C | Weaknesses | 불릿 |
| D | Novelty | 1-4 택1 |
| E | Evaluation | 1-4 택1 |
| F | Impact | 1-4 택1 |
| G | Writing quality | 1-4 택1 |
| I | Overall merit (ISCA 기준 절대평가) | 1-4 택1 |
| L | Comments for authors | 서술, 가장 길다 |
| M | Questions for authors' response | 불릿 |
| N | 개선 아이디어 / next steps | 서술 |
| O | Comments for the Instructor | 서술 |

**L01 s10이 사실상 이 폼의 작성 가이드다.** 강사가 읽으면서 생각하라고 한 여섯 가지:
biggest challenge와 motivation / proposed methodology와 그게 말이 되는 이유 /
evaluation 방법 (도구, 데이터셋, 실험) / 문제·아이디어·평가에 대한 내 평가 /
future work / 내가 가진 질문. **이 여섯이 A, B, C, L, M, N에 그대로 대응한다.**

**폼의 마감은 강의 전이다.** 실라버스 말고 Canvas 공지가 날짜의 기준이다.

## HTML 기술 규격

### 자체완결형

**CDN 금지.** 네트워크에서 뭔가를 받아오는 코드를 쓰지 않는다.
노트 본문의 CSS, SVG, JS는 전부 그 HTML 파일 하나에 인라인한다.
오프라인에서 열어도 노트는 완전히 동작해야 한다.

예외는 PDF 리더 하나뿐이다. `vendor/pdf.js/`와 `slides/*.pdf`는 repo 안에 있는
로컬 파일이므로 "외부 의존성 없음"은 유지된다. 다만 **리더는 없어도 되는 부가 기능이다.**

### 수식 표기

**MathJax나 KaTeX를 쓰지 않는다** (CDN 금지). EEE 554와 같은 방식으로 손으로 조판한다.

```html
<span class="m">J(θ)</span>                       <!-- 인라인 수식 -->
<div class="eq"><span class="m">θ ← θ − α∇<sub>θ</sub>J(θ)</span></div>   <!-- 독립 수식 -->
```

- `.m`은 수식용 폰트(세리프 이탤릭 계열). `.eq`는 가운데 정렬 블록.
- 유니코드 수학 기호를 그대로 쓴다: `θ α ∇ Σ ∂ ≈ ≤ ×`.
- **아래 첨자는 `<sub>`, 위 첨자는 `<sup>`.** 유니코드 첨자(`ₓ`, `²`)는 폰트에 따라
  안 나오는 것이 있다. `mm²`처럼 흔한 것만 예외.
- 행렬이나 여러 줄 유도가 필요하면 **SVG로 그린다.** 아스키 아트를 쓰지 않는다.

### 코드 표기

이 과목은 PyTorch와 Verilog가 둘 다 나온다.

```html
<code>torch.nn.Conv2d</code>
<pre class="code"><span class="k">for</span> n <span class="k">in range</span>(N):  <span class="c"># batch</span>
  <span class="k">for</span> k <span class="k">in range</span>(K):  <span class="c"># output channel</span>
    ...</pre>
```

- **하이라이팅은 손으로 `<span>`을 넣는다.** 라이브러리를 쓰지 않는다.
  칠하는 것은 키워드(`.k`), 문자열(`.s`), 주석(`.c`), 숫자(`.n`) 넷뿐이다.
- **색은 토큰으로만 준다.** 하드코딩하면 다크모드에서 안 바뀐다.
- `<pre class="code">`는 **가로 스크롤을 자기 안에서 처리한다.** `overflow-x:auto`.
- **loop nest는 주석으로 각 차원이 뭔지 밝힌다.** 이 과목의 코드는 대부분 loop nest이고,
  어느 loop가 어느 차원인지가 곧 dataflow다.

### 디자인 토큰 (모든 파일 공통)

색마다 **RGB 성분을 따로 둔다.** 도해에서 같은 색을 여러 농도로 쓰기 때문이다.

```css
--blue-rgb:18,80,196;
--blue:rgb(var(--blue-rgb));
/* 농도 변형은 rgba(var(--blue-rgb),.16) 형태로 쓴다 */
```

`index.html`, `glossary.html`, 모든 `L{NN}` 파일의 `:root` 블록은
**한 글자도 다르지 않게 동일**하다. 여기 없는 변수를 새로 만들지 않는다.

**출발점은 `asu-cen-598-addv-fa26-notes/L01-course-intro.html`의 `<style>` 블록이다.**
거기에 EEE 554의 `.nota` / `.m` / `.eq` 규칙과 이 저장소의 `.scale`을 더한다.
그 뒤로는 이 저장소 안에서 복사해서 쓴다.

```bash
python -c "import io; s=io.open('L01-introduction.html',encoding='utf-8').read();   io.open('.build-style.html','w',encoding='utf-8').write(s[s.index('<style>'):s.index('</style>')+8])"
```

**SVG 안에서도 CSS 변수를 쓴다.** `fill="var(--blue)"`는 정상 동작한다.

**색을 하드코딩하지 않는다.** `verify.mjs`가 소스에서 리터럴 색을 잡아 실패시킨다.
**예외는 `<mask>` 안뿐이다.** 마스크의 `#000`/`#fff`는 색이 아니라 알파 채널이다.

### 의미 고정 색상

**앞선 두 저장소와 의미가 다르다.** 554는 집합 A/B/C, CEN 598은 직군과 단계였다.
**여기는 "연산이냐 데이터냐"다.** 이 과목의 중심 긴장이 거기 있기 때문이다.

| 대상 | 색 |
|---|---|
| 모델과 알고리즘 (DNN, layer, weight, activation, gradient) | `--blue` |
| 연산 하드웨어 (PE, MAC, systolic array, ALU, datapath) | `--violet` |
| **메모리와 데이터 이동 (DRAM, HBM, SRAM, buffer, register file, bandwidth)** | `--brown` |
| 소프트웨어 스택 (framework, compiler, runtime, mapping, scheduler) | `--pink` |
| 이득 (speedup, energy saving, 정답, 개선) | `--green` |
| 비용과 손실 (energy, latency, accuracy drop, 버그, 함정) | `--amber` |
| 이 강의 범위 밖 / 나중에 배울 것 | `--ink3` |

**이 규칙이 지켜지면 독자가 색만 보고 "이건 연산 쪽" / "이건 메모리 쪽"을 읽게 된다.**
이 과목의 결론은 거의 항상 **"연산보다 데이터 이동이 비싸다"**로 수렴한다.
violet과 brown이 그 이야기를 그림마다 반복해줘야 한다. 이게 이 노트의 시그니처다.

**규칙을 어기면 인접한 도해끼리 색이 충돌한다.** 새 도해를 그릴 때뿐 아니라
**기존 도해를 손볼 때도 색이 규칙에 맞는지 확인한다.**
자동 검사는 색이 토큰인지만 보지 의미까지는 못 본다.

### 다크모드

기본은 OS 설정(`prefers-color-scheme`)을 따르고, 좌하단 버튼으로 덮어쓸 수 있다.

- 다크 팔레트는 `@media (prefers-color-scheme:dark){:root:not([data-theme="light"])}`와
  `:root[data-theme="dark"]` 두 곳에 같은 값을 적는다.
- **의미 고정 색상의 의미는 유지하고 밝기만 올린다.**
- 토글 상태는 **저장하지 않는다.** 브라우저 저장소 금지 규칙 때문이다.
- 슬라이드 PDF는 원본 문서라 다크모드에서도 흰 종이 그대로 둔다.

### 블록 클래스

| 클래스 | 용도 | 시각 |
|---|---|---|
| `.def` | 정의 | 검은 좌측 바 |
| `.jargon` | 용어 뜯어보기 (기본 펼침) | 점선 테두리 + 회색 배경 |
| `.nota` | 기호 뜯어보기 (기본 펼침) | 점선 테두리 + 회색 배경 |
| `.scale` | **숫자 감각** | 회색 배경 카드, 우측 정렬 숫자 열 |
| `.trap` | 함정, 혼동되는 용어 | amber 배경 + 좌측 바 |
| `.ex` | 감각 문제 (내부에 `<details>` 답) | 흰 카드, 파란 태그 |
| `.callout` | 강의 간 연결, 강조 | 상하 실선 |
| `pre.code` | 코드 블록 | 회색 배경, 가로 스크롤 |
| `.eq` | 독립 수식 | 가운데 정렬 |
| `figure` + `figcaption` | SVG 도해 | 흰 카드 |
| `.mini` | 작은 SVG 카드 (grid2/3/4 안에) | 흰 카드 |
| `.play` | 인터랙티브 도구 | 흰 카드 |

### SVG 규칙

- 항상 `viewBox` 지정, `width`/`height` 속성은 쓰지 않는다.
- `role="img"` + `aria-label` 필수.
- 텍스트는 `.svgtxt`(본문), `.svglab`(모노, 라벨), `.svgnum`(모노, 숫자) 클래스 사용.
  **수식이 들어가는 도해에서는 `.svgtxt`를 세리프로 둔다.** 블록도만 있는 도해에서는
  산세리프. 한 파일 안에서 섞여도 되지만 한 도해 안에서는 통일한다.
- **크기와 색은 반드시 인라인 `style`로 준다. 프레젠테이션 속성은 클래스에 진다.**
  `style="font-size:24px"`로 쓴다.
- **좌표는 짐작하지 말고 재서 맞춘다.** `<g transform="translate(…)">`가 있는 도해에서는
  `getBBox()`가 조상의 transform을 반영하지 않으므로 `getBoundingClientRect()`를 쓴다.
- **화살표는 방향과 의미를 라벨로 밝힌다.** 이 과목 도해는 데이터 흐름이 주인공이다.
  **화살표에 "무엇이 몇 바이트" 흐르는지 적을 수 있으면 적는다.** 그게 이 과목의 핵심이다.
- `<defs>` 안의 id는 파일 내에서 유일해야 한다.
- 도형 밖으로 텍스트가 나가지 않게 좌표를 계산한다. **검증 스크립트가 이걸 잡는다.**

### 그림만 있는 슬라이드를 옮길 때

**원본 그림을 그대로 베끼려 하지 않는다.** 두 가지 이유다.
저작권 문제이고, 원본이 이미 리더에 떠 있어서 베낀 그림은 중복이다.

대신:
1. **본문으로 서술한다.** 위 [한 슬라이드도 건너뛰지 않는다](#한-슬라이드도-건너뛰지-않는다) 참조.
2. **필요하면 그 그림이 말하려는 구조만 다시 그린다.** 사진이나 스크린샷을 재현하지 말고
   그 안의 논리(무엇이 무엇보다 크다, 무엇이 무엇으로 간다)를 도해로 만든다.
3. 슬라이드에 그림 출처가 찍혀 있으면 **출처를 노트에도 적는다.**
   L01은 Eyeriss tutorial, Reagen et al. 2017, J. Dean 2018 등이 찍혀 있다.

### 인터랙티브 요소

강의마다 하나씩, 그 강의의 핵심 개념을 손으로 만져볼 수 있는 도구를 만든다.

원칙: **결과를 보여주는 게 아니라 사용자가 조작해서 발견하게 한다.**

이 과목에 맞는 형태:
- **숫자 계산기.** layer 파라미터(H, W, C, K, R, S)를 슬라이더로 바꾸면
  MAC 수, weight 바이트, activation 바이트가 같이 움직인다.
  "왜 FC layer는 memory bound이고 conv layer는 compute bound인가"가 여기서 보인다.
- **Roofline 탐색기.** arithmetic intensity를 움직이면 어느 지붕에 걸리는지 보여준다.
- **Quantization 손실 체험.** bit width를 낮추면 값이 어떻게 뭉개지는지.
- **Dataflow 비교기.** weight/output/row stationary에서 무엇이 몇 번 읽히는지 세어준다.
- **분류기.** 항목을 주고 "이건 compute bound인가 memory bound인가"를 고르게 하고 채점.

브라우저 저장소(localStorage 등) 사용 금지. 모든 상태는 JS 변수로만.

## 슬라이드 리더 (데스크톱 전용)

노트 오른쪽에 강의 슬라이드 PDF를 띄우고, **노트를 스크롤하면 슬라이드가 따라온다.**
`reader.js` / `reader.css` / `vendor/pdf.js/`는 앞선 저장소에서 그대로 가져온 것이고
수정하지 않았다. **동작 규칙과 구현 세부는 `asu-eee-554-notes/CLAUDE.md`의
"슬라이드 리더" 절이 여전히 유효하다.** 고치기 전에 그쪽을 읽는다.

노트는 `<script src="reader.js" type="module"></script>` 한 줄만 넣는다.
`reader.css`는 `reader.js`가 직접 주입하므로 `<link>`를 쓰지 않는다.

`file://`로 열면 브라우저가 PDF fetch를 막는다. 로컬에서 리더까지 보려면 서버로 열어야 한다.

```bash
node scripts/serve.mjs      # http://127.0.0.1:4599/
```

**`python -m http.server` 를 쓰면 안 된다.** `.mjs` 를 `text/javascript` 로 안 내보내서
브라우저가 pdf.js 모듈 로드를 거부한다. `verify.mjs` 는 자기 서버를 띄우므로 영향이 없다.

### 앵커 규칙

동기화의 기반은 노트 안에 박아둔 슬라이드 쪽수다. **노트를 쓸 때 같이 넣는다.**

```html
<div class="sec-head" data-slide="14"><span class="sec-num">03</span><h2>Domain-Specific Accelerators</h2></div>
<div class="sec-head" data-slide="26-28">…</div>   <!-- 여러 쪽에 걸친 구간 -->
<h3 data-slide="17">Deep Learning Publications</h3> <!-- 섹션보다 잘게 끊고 싶을 때 -->
```

**앵커는 `<section>`이 아니라 `.sec-head`(와 `h3`)에 붙인다.** CSS가 `attr(data-slide)`로
출처 라벨을 직접 만들기 때문이고, `attr()`은 자기 자신의 속성만 읽는다.

- `data-slide`는 **1부터 시작하는 PDF 쪽 번호**다.
  **슬라이드에 인쇄된 번호와 다르다.** 이 덱은 표지에 번호가 없고, L01은 s29부터
  인쇄 번호가 PDF 쪽수보다 1 크다 (PDF 29쪽에 "30"이 찍혀 있다). 강사가 슬라이드를
  지우면서 번호를 안 고친 것이다. **항상 PDF 쪽수를 쓴다.** 인쇄 번호를 믿지 않는다.
  앵커를 달기 전에 `node scripts/pdftext.mjs`로 해당 쪽 텍스트를 확인한다.
- **한 슬라이드도 건너뛰지 않는 규칙 때문에, 앵커가 안 걸린 쪽이 거의 없어야 한다.**
  로지스틱스 슬라이드만 예외다. `verify.mjs`가 미포함 쪽을 경고로 알려주는데,
  **이 저장소에서는 그 경고를 진지하게 본다.** CEN 598과 다른 점이다.
- 슬라이드에 없는 내용(보충, 함정, `.scale`, `.jargon`의 "누가/언제" 줄, 인터랙티브 도구)에는
  **붙이지 않는다.** 앵커가 없는 구간은 리더가 직전 앵커를 유지한다.
- **한 슬라이드는 그것을 실제로 다루는 곳 한 군데에만 앵커를 단다.**
  범위가 겹치면 역방향 동기화가 어디로 갈지 정해지지 않는다. `verify.mjs`가 겹침을 잡는다.
- **노트를 따라 내려갈 때 슬라이드 번호가 뒤로 가면 안 된다.** 경고가 나오면
  앵커를 고치는 게 아니라 **노트 순서를 슬라이드에 맞추는 걸 먼저 검토한다.**

### 슬라이드를 눈으로 봐야 한다

**텍스트 추출만으로 쓰면 반드시 틀린다.** 이 덱은 특히 그렇다.

- **그림 전용 슬라이드가 절반 가까이다.** L01 s17, s23, s25, s28~s33, s35~s41은
  텍스트 추출에 제목 한 줄밖에 안 나온다.
- **수식이 깨진다.** L02-3 s13, s16, s17의 첨자가 통째로 사라진다.
- **표의 열 순서가 뒤섞인다.** L01 s33의 기술 노드 표가 실제로 뒤섞여 나온다.

```bash
node scripts/render-slides.mjs slides/L01-introduction.pdf 25-33
```

**모든 쪽을 렌더해서 열어본다.** 이 저장소는 한 장도 건너뛰지 않으므로
"도해가 있는 쪽만"이 아니라 **전부**다.

## 강의 목록

**Canvas의 Schedule이 기준이다. 실라버스의 topic 목록은 순서가 아니다.**
학기 초라 전체 일정이 아직 안 나왔다. 나오면 이 표를 채우고 `index.html`의 뼈대로 쓴다.

현재 배포된 것:

| # | 슬라이드 원본 | 노트 | 쪽 | 상태 |
|---|---|---|---|---|
| 1 | `EEE 598 - Lecture 1 - Introduction.pdf` | `L01-introduction.html` | 43 | 완료 |
| 2-3 | `EEE 598 - Lecture 2-3 - ML Basics.pdf` | `L02-ml-basics.html` | 21 | 완료 |

**두 노트 모두 슬라이드 전 쪽에 앵커가 걸려 있다** (미포함 0쪽).
새 노트를 쓴 뒤에는 아래 명령으로 그것을 확인한다.

```bash
python -c "import io,re; s=io.open('L0N-....html',encoding='utf-8').read(); \
cov=set(); [cov.update(range(int(a),int(b or a)+1)) for a,b in re.findall(r'data-slide=\"(\d+)(?:-(\d+))?\"',s)]; \
print(sorted(set(range(1,N+1))-cov))"
```

실라버스와 L01 s7이 예고한 주제 (순서 미확정):

- Deep Neural Network Basics: training과 inference
- 컴퓨터 구조 개념: performance, energy efficiency, parallelism, locality, memory hierarchy, roofline
- 모델 최적화: quantization, pruning
- Custom hardware 1: systolic array, CGRA
- Custom hardware 2: near data processing
- Custom hardware 3: reliability와 fault tolerance
- Compilation: tiling, mapping, data orchestration
- AutoML: neural architecture와 hardware co-design
- TinyML: hardware-aware, resource-constrained DNN 최적화, TFLite
- Systems for ML: cloud serving, TensorFlow Serving
- 최신 주제: recommendation model, LLM

**슬라이드 없이 노트를 쓰지 않는다.** `index.html`에서 `soon` 상태로 두고 기다린다.

## 워크플로우

새 강의 노트를 만들 때:

1. **Drive 원본이 갱신됐는지 확인한다.** 슬라이드가 새로 올라왔거나 바뀌었을 수 있다.
   Canvas 공지도 같이 본다. 슬라이드가 재활용 덱이라 공지가 최신이다.
2. **PDF를 `slides/L{NN}-{topic}.pdf`로 복사한다.** 노트와 stem을 맞춘다.
   `node scripts/pagecount.mjs`로 쪽 수를 확인한다.
3. **`node scripts/pdftext.mjs`로 쪽별 텍스트를 뽑아 읽는다.**
   **읽으면서 어느 개념이 몇 쪽인지 기록해둔다.** 6번에서 `data-slide`로 쓴다.
4. **`node scripts/render-slides.mjs`로 전 쪽을 렌더해서 한 장씩 본다.**
   건너뛰지 않는다. 그림 전용 쪽에 무엇이 있는지 여기서 정해진다.
5. **오늘 나오는 새 용어, 새 기호, 새 숫자를 먼저 뽑는다.**
   용어는 풀네임/한 줄/언제/누가/헷갈리는 것을 채울 수 있는지,
   기호는 모양(스칼라/벡터/행렬)을 말할 수 있는지,
   숫자는 비교 기준점을 댈 수 있는지 확인한다. 못 채우면 아직 이해 못 한 것이다.
   **그리고 각각을 ECE 479에서 봤는지 판정한다.**
   `ECE 479\Lecture\Lecture_Index.md`와 `Final_Cheatsheet.md`를 대조한다.
   본 것은 확인용 한 줄 + `.callout`, 안 본 것은 `.jargon`/`.nota`/`.scale` 전체.
   **이 판정이 노트의 분량 배분을 정한다.** 건너뛰면 아는 이야기만 길게 쓰게 된다.
6. **`L{NN}-{topic}.html`을 작성한다.** 기존 파일의 `<style>` 블록을 그대로 복사해서 시작한다.
   각 `.sec-head`에 `data-slide`를 붙인다. 슬라이드에 없는 보충에는 붙이지 않는다.
7. **`glossary.html`을 갱신한다.** 이 강의에서 새로 푼 용어를 전부 넣는다. 미루지 않는다.
8. `node scripts/verify.mjs L{NN}-{topic}.html` 실행. exit code 0이어야 통과다.
   - `shots/` 폴더에 전체 스크린샷과 도해별 개별 스크린샷 `-fig01.png…`이 생긴다.
   - **전체 스크린샷은 여러 장으로 잘려 나온다.** 이 저장소의 노트는 한 슬라이드도
     건너뛰지 않아서 페이지가 매우 길다 (L01이 데스크톱 33,525px, 모바일 54,014px).
     크롬은 대략 16,384px를 넘는 이미지를 못 만들고 `fullPage` 캡처가
     `Protocol error: Unable to capture screenshot`으로 통째로 실패한다.
     그래서 `verify.mjs`의 `shootTall()`이 12,000px 단위로 잘라서 찍는다
     (`-desktop-01.png`, `-desktop-02.png`, …). **이 동작을 fullPage 한 장으로 되돌리면
     노트가 길어지는 순간 검증이 다시 죽는다.**
   - **도해 스크린샷을 한 장씩 다 열어본다.** 자동 검사가 통과해도 건너뛰지 않는다.
   - **미포함 슬라이드 경고를 반드시 확인한다.** 이 저장소에서는 그게 규칙 위반 신호다.
9. **슬라이드 리더가 실제로 따라오는지 확인한다.** `node scripts/serve.mjs`로 띄우고
   노트를 위에서 아래로 훑는다.

   ```js
   document.getElementById('s12').scrollIntoView();
   document.querySelector('aside').innerText.split('\n').slice(0,3).join(' ')
   ```

   **어긋난 앵커는 있으나 마나가 아니라 적극적으로 해롭다.** 근거를 잘못 가리킨다.
   **포트를 바꿔가며 확인한다.** 잘못된 MIME 으로 받은 `.mjs` 는 브라우저 캐시에 남는다.
10. 문제가 있으면 고치고 8번 반복.
11. **`index.html`을 갱신한다:**
    - 해당 강의 카드의 `<div class="mod soon">` → `<a class="mod" href="...">`
    - `<span class="status wait">준비 중</span>` → `<span class="status done">읽기</span>`
    - 닫는 `</div>` → `</a>`
    - 태그 목록과 헤더의 강의 수 chip 갱신
12. 커밋하고 푸시한다. 커밋 메시지는 한 줄, 영어: `Add Lecture 4: Quantization`
    **슬라이드 PDF가 노트보다 늦게 들어가면 안 된다.** 노트만 먼저 올라가면 리더가 404를 받는다.

Paper Reading 노트를 만들 때는 위 [리뷰 평가와 블로그화](#리뷰-평가와-블로그화)를 따른다.
**사용자가 제출을 마친 뒤에만 만든다.**

## 검증

`verify.mjs`는 저장소 루트를 **임시 http 서버로 띄우고** 그 주소로 연다.
GitHub Pages와 같은 조건으로 맞추기 위해서다.

**자동으로 잡는 것** (하나라도 걸리면 exit 1):
- 페이지 로드 시 JS 콘솔 에러
- 소스에 하드코딩된 색 리터럴 (`<mask>` 안과 `:root` 정의부는 예외)
- SVG `<text>`가 부모 viewBox를 벗어나는 경우
- 중복된 DOM id, 해결되지 않는 `url(#...)` 참조. **`<a>` 안에 `<a>`를 넣으면 걸린다.**
- 모바일(390px) 가로 오버플로우. **`<details>`를 전부 펼친 상태로 잰다.**
  `.jargon`과 `.nota`가 기본 펼침이라 여기가 특히 중요하다. `pre.code`도 걸린다.
- 슬라이드 리더가 실제로 떠서 첫 쪽을 렌더하는지
- `data-slide` 값이 PDF 쪽 수를 벗어나는지
- `data-slide` 범위가 서로 겹치는지 (포함 관계는 정상, 부분 겹침만 잡는다)
- 다른 페이지를 가리키는 `href="x.html#id"`의 대상이 실제로 있는지.
  **`glossary.html` ↔ 강의 노트 링크가 여기 걸린다.**
- 본문과 목차 링크가 **라이트/다크 양쪽에서** 배경 대비 3:1 이상인지
- 모든 `<details>`가 **양쪽 언어에서 열리는지** (`<summary>`가 2개면 실패)
- 도해 안에서 **글자끼리 겹치는지**
- 리더가 뜨는 최소 폭에서 본문이 480px 이상 남는지

**경고만 하는 것** (exit code에 영향 없음):
- 도해에서 글자가 다른 그룹의 도형 위에 있는 곳
- 노트를 따라 내려가는데 슬라이드 번호가 뒤로 가는 곳
- **어떤 앵커에도 안 걸린 슬라이드.** 위에 적었듯 이 저장소에서는 이 경고를 진지하게 본다.

`verify.mjs`에 `HW*.html`의 문제 번호를 대조하는 블록이 남아 있다 (620-655줄).
**이 저장소는 HW 페이지를 만들지 않으므로 그 검사는 돌지 않는다.** 지우지 않고 둔다.

**전혀 못 잡는 것**: 설명의 질, 용어 풀이가 실제로 이해되는지, 숫자 감각이 맞는지,
**그리고 한 슬라이드를 통째로 빠뜨렸는지도 자동으로는 못 잡는다** (앵커 경고가 힌트일 뿐).
사용자 피드백이 유일한 신호다. **완성했다고 단정하지 말고 확인을 요청한다.**

### 모바일 오버플로우가 났을 때

| 원인 | 방어 |
|---|---|
| 그리드 트랙이 `1fr`이라 넓은 자식이 레이아웃 전체를 밀어냄 | `grid-template-columns:minmax(0,1fr)` |
| 표의 min-content가 화면보다 넓음 | `@media(max-width:900px){table{display:block;overflow-x:auto}}` |
| 긴 코드 줄이 안 접힘 | `pre.code{overflow-x:auto}` |
| `.jargon .parts` / `.nota .parts` 첫 칸이 `nowrap` | `@media(max-width:900px){.parts td:first-child{white-space:normal}}` |
| **`.scale`의 숫자 열이 길어서 3열이 안 들어감** | `@media(max-width:900px){.scale .parts{display:block}}` |

마지막 것이 이 저장소에서 새로 생긴 것이다.

## 커밋하는 것과 안 하는 것

repo는 **Public**이고 GitHub Pages로 서빙된다. 여기 올리는 건 인터넷에 게시하는 것이다.

**사용자에게 실라버스의 Copyright 조항을 알렸고, 슬라이드를 커밋하기로 결정했다.**
결정은 사용자의 것이므로 그대로 진행하되, **footer에 저작권 귀속을 반드시 밝힌다.**
이 결정을 조용히 뒤집지 않는다.

| 파일 | 커밋 |
|---|---|
| `slides/L{NN}-*.pdf` (강의 슬라이드) | O |
| `vendor/pdf.js/` | O |
| **논문 PDF** | **X.** 3자 저작물이다. 링크만 건다 |
| **Sample review.txt** | **X.** 강사가 반 밖으로 공유하지 말라고 명시했다 |
| **Review Form.txt** | X. 제출 양식이다. 구조 요약만 노트에 적는다 |
| **제출한 리뷰 원문** | **X.** 학문적 정직성 |
| Homework 문제지, 제출 코드 | **X** |
| Project 자료 | **X.** 팀 과제라 다른 사람 저작물이 섞인다 |
| `shots/`, `node_modules/` | X |

`.gitignore`:
```
node_modules/
shots/
package-lock.json
*.pdf
!slides/*.pdf
```

`*.pdf`로 전부 막고 `!slides/*.pdf`로 슬라이드만 되살린다.
**논문 PDF를 저장소 안에 두지 않는다.** `.gitignore`가 막아주더라도 애초에 복사하지 않는다.

커밋 전에 `git status`로 의도하지 않은 파일이 스테이징됐는지 확인한다.

### footer 문구

각 노트와 `index.html` footer에 이렇게 밝힌다:

- 강의 슬라이드의 저작권은 담당 강사(Jeff (Jun) Zhang)와 ASU에 있다는 것
- 이 사이트는 수강생이 만든 개인 학습 자료이고 공식 강의 자료가 아니라는 것
- 노트 본문은 슬라이드를 재구성하고 보충한 것이라는 것
- **지정 교재 세 권을 적는다.** CEN 598과 다른 점이다. 교재가 실제로 있다.

## 상단 미니 헤더

노트가 길어서 한참 내리면 큰 제목이 안 보인다. 목차 맨 위에 작은 헤더를 둔다.

- 내용: `← 강의 목록` 링크 / `LECTURE N` / 영문 제목.
  **제목은 `header.top`의 `h1.title`에 있는 영문 그대로 쓴다.**
- **목차(`nav.toc`)가 이미 sticky이므로 그 안에 넣는다.**
- `header.top`의 `h1.title`을 IntersectionObserver로 감시해서, 화면에서 사라지면 `.on`을 붙인다.
- 모바일에서는 목차가 static이라 숨긴다.
- **강의 목록으로 돌아가는 링크는 헤더와 미니 헤더 양쪽에 있어야 한다.**

## 이중 언어 (한국어 / 영어)

한 파일 안에 두 언어를 나란히 두고 보이는 쪽만 남긴다. 좌하단 버튼으로 전환한다.

```html
<p lang="ko">…</p>
<p lang="en">…</p>
```

- **도해·앵커·CSS·리더를 공유하므로 두 언어의 구조가 어긋날 수 없다.** 파일이 커지는 게 비용이다.
- SVG 텍스트도 같은 좌표로 두 벌 겹쳐 둔다.
- **언어 규칙에는 `!important`가 필요하다.**
- **`<details>` 하나에 `<summary>`를 두 벌 넣지 않는다.** 개폐 버튼이 되는 건 첫 번째
  `<summary>` 하나뿐이라, 두 벌을 넣으면 숨겨지는 언어에서 **내용에 영원히 도달할 수 없다.**
  `<summary><span lang="ko">…</span><span lang="en">…</span></summary>`로 쓴다.
  **`.jargon`과 `.nota`가 전부 이 형태다.**
- **`data-slide`가 붙은 `h3`를 두 벌로 만들면 앵커가 중복된다.** 숨겨진 쪽은
  `getBoundingClientRect()`가 전부 0이라 리더가 "화면 맨 위"로 오인한다.
- JS가 찍는 문자열은 `L(ko, en)` 헬퍼로 처리하고, `langchange` 이벤트에서 다시 그린다.

### 영어 문체

**번역문이 아니라 처음부터 영어로 쓴 글이어야 한다.** 한국어 어순을 따라가지 않는다.

- 2인칭 대화체. 축약형(you'll, that's, it's)을 쓴다.
- em dash 금지는 영어에도 적용한다.
- **업계 용어는 원래대로 영어 유지.**
- **`.jargon`의 영어판은 한국어판보다 짧아진다.** 풀네임과 한 줄 정의는 영어권 독자에게
  이미 절반이 자명하기 때문이다. **그래도 "언제 / 누가 / 헷갈리는 것"은 반드시 남긴다.**
- **`.scale`의 영어판은 기준점을 바꿔야 할 수도 있다.** "A4 용지"는 미국 독자에게
  기준이 아니다. Letter 규격이나 다른 기준으로 바꾼다.

### 검사

번역 누락은 눈으로 못 잡는다. 스크립트로 확인한다.

- 영어 모드로 바꾸고 `lang="ko"` 조상이 없는데 한글이 남은 요소를 찾는다.
- **`<html lang="ko">` 때문에 `closest('[lang="ko"]')`가 모든 요소에서 참이 된다.**
  루트를 제외해야 한다.
- **SVG 요소에는 `offsetParent`가 없다.** 가시성은 계산된 `display`로 판정한다.
