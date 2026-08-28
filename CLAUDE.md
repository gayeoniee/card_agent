# card_agent 규칙

개인 레포지만 **dev 에 들어갈 수 있는 상태**여야 하므로 팀 규칙을 따른다.
근거: `DAENGS_dev/docs/collaboration.md`, `DAENGS_APP/docs/collaboration.md`,
`DAENGS_dev/CLAUDE.md`, `DAENGS_dev/docs/decisions.md`.

## 환경

- Python **3.12 고정** (`requires-python = ">=3.12,<3.13"` + `.python-version`). D-001.
- **`pip` 을 쓰지 않는다.** 의존성은 `uv add` / `uv remove`, `uv.lock` 은 커밋한다.
  `pyproject.toml` 의 의존성을 손으로 고치면 lock 과 어긋난다.
- 저장소와 무관한 일회성 스크립트는 **PEP 723 인라인 의존성 + `uv run --no-project`**.
  `tools/contact_sheet.py` 처럼 `src/` 를 부르는 도구는 프로젝트 환경에서 돌린다.

## 브랜치·PR

- 작업 브랜치는 `dev` 에서 따고 PR 로 `dev` 에 머지한다. `main` 은 릴리즈 스냅샷이라
  작업하지 않고, **default 를 `main` 으로 바꾸지 않는다.**
- **PR 은 끝난 뒤가 아니라 시작할 때 draft 로 연다.** 빈 커밋 하나 올리고 draft PR.
- **PR 제목은 `타입: 무엇을`** (feat / fix / docs / refactor / build / chore).
- **커밋 메시지는 한글 서술형, 접두사 없음.** 제목에 무엇이 어떻게 잘못돼 있었는지
  (`~던 것`), 본문에 왜 그렇게 고쳤는지와 무엇을 재봤는지. **PR 제목 규칙을 커밋에
  적용하지 않는다** — 둘은 다른 규칙이다.
- 한 PR = 한 Iteration = 구현 1~2일 분량.

## 커밋하지 않는 것

- 강아지 **사진**, 생성된 카드·음악, 누끼 모델 가중치. `.gitignore` 로 막는다.
- 카드 원화(수 MB)는 git 밖에 둔다 — `DAENGS_dev/tools/art-src/` 와 같은 취급이라
  `templates/cards/` 를 무시한다.
- `.env` · API 키. `.env.example` 만 커밋하고 실제 변수와 일치시킨다.

## 이 폴더 고유

- **dev 의 무엇도 import 하지 않는다.** config·DB·모델을 부르지 않고 `scene.json`
  하나로만 말한다. 이것이 폴더째 옮길 수 있게 하는 유일한 조건이다.
- **계약의 출처는 `src/contract.py`** 하나다. 필드 이름은 앱의 `ImmersiveScene` ·
  `DexCard` 를 그대로 따라간다. 새 어휘를 만들지 않는다. 문서에 스키마를 두 번 적지
  않는다.
- ⚠ **`serve.py` 에 `from __future__ import annotations` 를 넣지 말 것.**
  `UploadFile` 이 문자열 어노테이션이 되어 pydantic 이 이름을 못 찾고 500 으로 죽는다.
  `skin-screening/CLAUDE.md` 가 "실제로 당했다" 고 적어 둔 함정이고, 같은 모양의
  서버라 그대로 있다.
- **그림·소리 판단은 사람이 한다** (앱 협업규칙 1절). 화풍이 붙는지, 루프 이음새가
  들리는지는 만들어서 실기기에서 본다. 코드가 "괜찮아 보인다" 고 적지 않는다.
- 되돌리기 번거로운 결정은 `docs/decisions.md` 에 **`CA-0xx`** 로 적는다. dev 의
  `D-0xx` 와 부딪히지 않게 접두사를 갈라 둔다.

## 재사용할 것 (새로 만들지 말 것)

| 쓸 것 | 어디에 | 무엇을 |
| --- | --- | --- |
| `DAENGS_APP/tools/convert_audio.py` | 앱 레포 | OGG 변환 규격과 이유. `BLOCK = 1<<16` 블록 단위 전달까지 그대로 |
| `DAENGS_dev/tools/neo-hologram-layers.py` | dev 레포 | 누끼 다듬기(`trim`)·카드 모서리 알파(`card`) 로직 |
| `DAENGS_APP/tools/isoasset.py pastel` | 앱 레포 | 톤 강제 통일. `pixelize.py` 의 원본 |
| `ImmersiveScene` · `DexCard` | `DAENGS_APP/.../ui/dex/` | JSON 필드 이름의 출처 |
| `skin-screening/serve.py` | dev 레포 | `--mock` 플래그 · 포트 안 여는 서비스 모양 |
