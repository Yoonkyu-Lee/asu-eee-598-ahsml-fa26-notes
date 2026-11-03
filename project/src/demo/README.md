# 데모: 박스 다섯 대 (CP21 → CP24)

진짜 박스 프로세스 다섯 개가 `127.0.0.1:8601`–`8605`에 떠서 서로 메시지를 주고받는다 (Ed25519, 증명, RLN, 복호화, MiniLM·CLIP 임베딩). 운영자 서버는 없다. 설계와 결과는 [cp/CP21.md](../../cp/CP21.md), [cp/CP24.md](../../cp/CP24.md).

- CP24부터 박스는 설계대로 돈다: 증명 먼저(O-b), 텍스트 단위 + 사진 개수 한도(I5), 실시간 128토큰(F1), 인코더 우선순위(텍스트 먼저, 오래된 그룹 먼저). 박스 0–2는 오래된 그룹, 3–4는 새 그룹. 신원은 박스가 직접 뽑는다. CP21 박스는 `--cp21`.
- **여러 컴퓨터에서 각자 박스 하나씩**: [LAN.md](LAN.md) (영어, 팀원용).

## 준비 (git에 없는 것)
- `project/.venv` (데모만이면 `src/demo/requirements.txt`)
- `project/models/all-MiniLM-L6-v2`, `project/models/clip-vit-base-patch32` (`src/demo/fetch_models.py`가 받고 SHA-256 확인)
- `project/data/coco/val2017_224.npy`, `val2017_index.json` ("아무 사진" 단추와 장면 실행기용. 없어도 화면에서 내 사진 파일을 올릴 수 있음)
- 포트 8601–8605가 비어 있을 것. CPU만 쓴다 (박스마다 1스레드).

## 실행
`project/`에서:
```
.venv/Scripts/python.exe src/demo/run_demo.py           # 글·사진 몇 개와 방 메시지 하나를 넣고 시작
.venv/Scripts/python.exe src/demo/run_demo.py --empty   # 빈 상태로 시작
```
다섯 박스가 모델을 다 불러오면 주소 다섯 개가 찍힌다. 브라우저 탭 여러 개로 `http://127.0.0.1:8601/` … `8605/`을 연다. 끝낼 때는 Ctrl+C (박스 프로세스도 같이 꺼짐).

## 화면에서 해 볼 것
| 해 볼 것 | 어디서 | 보이는 것 |
|---|---|---|
| 글쓰기 | Write (sub A / sub B, 사진 첨부 가능) | 다른 박스 피드에 퍼짐. sub B는 박스 0, 2, 4만 받음 |
| 추천·비추천·삭제 | Feed의 ▲ ▼ delete | 집계가 모든 박스에 반영, 삭제하면 모든 박스에서 사라짐 |
| 텍스트 검색 | Search → Text | MiniLM 임베딩으로 이 박스의 색인 검색 |
| 말로 사진 찾기 | Search → Photos by words | CLIP으로 찾은 사진 격자 |
| 방 R | Room R | 박스 1, 2, 3만 읽음. 나머지는 "키가 없어 못 읽음" |
| 이 박스의 작업 | This box's work | 작업 종류별 개수와 가상 칩 시간 (추정: 칩 모델) |
| 남은 몫 | Write 아래 | 이 창에서 남은 텍스트 번호·사진·투표 (CP24) |
| 한도 넘기기 | Limits and slashing | 같은 번호에 글 두 개 → 다른 박스가 비밀을 복원해 알림, 이 박스는 빨간 배너, 이후 글은 버려짐 (CP24) |

## 장면 실행기 (화면 없음, 약 2분)
```
.venv/Scripts/python.exe src/demo/x2.py           # CP21 박스 → results/cp21/cp21.json
.venv/Scripts/python.exe src/demo/x2.py --cp24    # CP24 박스와 DayNet24, D7–D9 추가 → results/cp24/x2.json
.venv/Scripts/python.exe src/demo/lan_check.py    # 여러 컴퓨터 모드를 이 컴퓨터의 사설 IP로 (L1–L7) → results/cp24/lan_check.json
```
박스 다섯 개에 장면 145개(CP24는 146개)를 넣고 같은 장면을 시뮬레이터로도 돌려, 박스별 작업 수를 비교하고 기능 시험을 돌린다.

## 확인한 것 (2026-11-01, 두 번째 PC, CPU만)
다섯 박스가 약 10초 만에 떴다. API로 확인했다:
- 글은 다섯 박스 모두로 퍼지고, sub B 글은 박스 0, 2, 4에만 간다.
- 추천은 모든 박스에서 +1이 된다.
- 작성자가 아닌 박스가 삭제하려 하면 403이 나고, 작성자가 삭제하면 모든 박스에서 사라진다.
- 방 R은 박스 1, 2, 3에서만 읽힌다. 박스 0이 방에 보내려 하면 403이 난다.
- 텍스트 검색과 사진 검색("a dog" → 개 사진 캡션이 1등)이 결과를 낸다.

## 로그
`logs/cp21_demo_box{0..4}.log`, 설정 `logs/cp21_demo_cfg.json` (방 키 포함, git 제외). 박스가 안 뜨면 해당 박스 로그부터 본다.
