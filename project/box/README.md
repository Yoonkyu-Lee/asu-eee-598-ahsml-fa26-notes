# 노드 박스 (에뮬): 박스 다섯 대를 이 PC에서, 한 화면에 (CP21 → CP32)

진짜 박스 프로세스 다섯 개가 `127.0.0.1:8621`–`8625`에 떠서 서로 메시지를 주고받는다 (Ed25519, 증명, RLN, 복호화, MiniLM·CLIP 임베딩). 운영자 서버는 없다. 칩은 진짜가 아니라 모델이다: 박스가 하는 일마다 "칩이었으면 몇 초"를 [chip/model/chip.py](../chip/model/chip.py)에서 가져와 더한다 (가상 가속기). 설계와 지난 결과는 [journal/cp/CP21.md](../journal/cp/CP21.md), [CP24](../journal/cp/CP24.md), 박스끼리 맞추는 규칙은 [docs/protocol.md](../docs/protocol.md).

- 동기화: 기본값 `--exchange recon` (CP29 집합 맞추기 + CP30 한 번만 가져오기). `--floor-skip`은 CP30의 F′ (옵션), `--exchange x2`는 CP26 방식.
- CP24부터 박스는 설계대로 돈다: 증명 먼저(O-b), 텍스트 단위 + 사진 개수 한도(I5), 실시간 128토큰(F1), 인코더 우선순위(텍스트 먼저, 오래된 그룹 먼저). 신원은 박스가 직접 뽑는다.
- CP32: 박스 다섯은 `lan.py`(여러 컴퓨터 모드와 같은 코드)를 루프백 주소로 띄운 것이다. 예전 `run_demo.py`는 동기화를 안 정해 최신 설계를 안 돌아서 `legacy/box/`로 옮김.
- **여러 컴퓨터에서 각자 박스 하나씩**: [LAN.md](LAN.md) (영어, 팀원용).

## 준비 (git에 없는 것)
- `project/.venv` (데모만이면 `box/requirements.txt`)
- `project/models/all-MiniLM-L6-v2`, `project/models/clip-vit-base-patch32` (`box/fetch_models.py`가 받고 SHA-256 확인)
- `project/data/coco/val2017_224.npy`, `val2017_index.json` ("아무 사진" 단추와 장면 실행기용. 없어도 화면에서 내 사진 파일을 올릴 수 있음)
- 포트 8620–8625가 비어 있을 것. CPU만 쓴다 (박스마다 1스레드).

## 실행: 한 화면에 다섯 사용자 (CP32)
`project/`에서:
```
.venv/Scripts/python.exe box/stage.py                 # 그리고 브라우저로 http://127.0.0.1:8620/
.venv/Scripts/python.exe box/stage.py --headless      # 같은 이야기를 화면 없이 -> results/demo/story_headless.json
.venv/Scripts/python.exe box/stage.py --real-timers   # 박스 기본 시계 (바닥 10–30분, probe cap 5분)
```
- 한 페이지에 사용자 다섯(Ana, Ben, Cho, Dev, Eve)의 클라가 나란히 뜬다. 각 칸은 그 사람 박스의 진짜 화면이다 (iframe). 이름은 무대가 붙인 이름표일 뿐, 메시지에는 작성자 이름이 없다.
- **▶ 이야기 재생**: 15단계 (`box/story.py`). 각 사용자의 클라가 스스로 입력하고 누른다 (무대가 postMessage로 "이걸 해라"를 보내면 클라가 자기 화면에서 하고, 자기 박스만 부른다). 단계마다 무대가 다섯 박스를 읽어 약속대로인지 ✓/✗.
- 아무 칸에서나 직접 써도 되고, 칸마다 끄기/켜기, 위에 망 쪼개기/다시 붙이기, 처음부터.
- 시계는 기본으로 시험 값 (바닥 20–40 s, probe cap 20 s, `tests/sync_check.py`와 같음). 화면 아래에 표시된다.
- 박스 상태는 `state/stage/box<i>/` (매번 비우고 시작, `--keep`이면 이어서). 모두 127.0.0.1에서만 열린다. Ctrl+C로 다 꺼짐.
- 칸 하나만 브라우저로 따로 보고 싶으면 `http://127.0.0.1:8621/` … `8625/`.

## 화면에서 해 볼 것
| 해 볼 것 | 어디서 | 보이는 것 |
|---|---|---|
| 글쓰기 | Write (sub A / sub B, 사진 첨부 가능) | 다른 박스 피드에 퍼짐. sub B 글은 박스 0, 2, 4 화면에만 뜸 (단 H12: 다른 박스도 봉투를 받아 검증·저장함, CP32) |
| 추천·비추천·삭제 | Feed의 ▲ ▼ delete | 집계가 모든 박스에 반영, 삭제하면 모든 박스에서 사라짐 |
| 텍스트 검색 | Search → Text | MiniLM 임베딩으로 이 박스의 색인 검색 |
| 말로 사진 찾기 | Search → Photos by words | CLIP으로 찾은 사진 격자 |
| 방 R | Room R | 암호를 들은 Ana·Ben·Dev(박스 0, 1, 3)만 읽음. 나머지는 봉투를 중계만 |
| 이 박스의 작업 | This box's work | 작업 종류별 개수와 가상 칩 시간 (추정: 칩 모델) |
| 남은 몫 | Write 아래 | 이 창에서 남은 텍스트 번호·사진·투표 (CP24) |
| 한도 넘기기 | Limits and slashing | 같은 번호에 글 두 개 → 다른 박스가 비밀을 복원해 알림, 이 박스는 빨간 배너, 이후 글은 버려짐 (CP24) |

## 시험 (화면 없음)
```
.venv/Scripts/python.exe -W ignore tests/sync_check.py      # 박스 4개, 링크 끊기·시계·강제 종료·위조, 대본 12개 (약 12분) -> results/tests/sync_check.json
.venv/Scripts/python.exe tests/lan_check.py <사설 주소>     # 여러 컴퓨터 모드를 이 PC 주소로 (L1–L7, 약 2분)            -> results/tests/lan_check.json
.venv/Scripts/python.exe tests/persist_check.py <사설 주소> # 껐다 켜도 같은 회원·지운 글 (P1–P4, 약 5분)               -> results/tests/persist_check.json
```
CP21–CP24의 장면 실행기(`x2.py`, 박스와 시뮬레이터 칸칸 대조)는 `legacy/src/demo/`에 있고 태그 `cp30`에서 돈다.

## 확인한 것 (CP32, 2026-11-17, CPU만)
이야기 15단계를 머리 없이(H) 한 번, 브라우저 꼭두각시로(B) 두 번: 14단계 통과, **3단계 실패 = H12** (서브 B에 없는 박스도 서브 B 글을 받음, `results/demo/`). 끄기·켜기 단추로 다시 켠 박스는 같은 신원으로 돌아옴.

## 확인한 것 (2026-11-01, 두 번째 PC, CPU만, CP21 박스)
다섯 박스가 약 10초 만에 떴다. API로 확인했다:
- 글은 다섯 박스 모두로 퍼지고, sub B 글은 박스 0, 2, 4에만 간다.
- 추천은 모든 박스에서 +1이 된다.
- 작성자가 아닌 박스가 삭제하려 하면 403이 나고, 작성자가 삭제하면 모든 박스에서 사라진다.
- 방 R은 박스 1, 2, 3에서만 읽힌다. 박스 0이 방에 보내려 하면 403이 난다.
- 텍스트 검색과 사진 검색("a dog" → 개 사진 캡션이 1등)이 결과를 낸다.

## 로그
`logs/cp32_stage_box{0..4}.log`, 박스 설정 `logs/lan_cfg_box{0..4}.json` (방 키 포함, git 제외). 박스가 안 뜨면 해당 박스 로그부터 본다.
