# card_agent

강아지 **얼굴 사진 · 이름 · 생일** 세 개로 그 강아지 전용 카드 한 장과 이머시브 장면
값을 뽑는다. 카드 디자인과 이머시브 구조는 지금 앱의 것을 그대로 쓰고, 채워 넣는
값만 개인화한다.

바깥과는 **`scene.json` 하나로만** 말한다. dev 레포의 config·DB·모델을 하나도 부르지
않는 독립 폴더라, 개인 레포에서 굴리다가 폴더째 dev 에 넣을 수 있다. 사본 추적은
`pyproject.toml` 의 `version` 한 줄로 한다.

## 파이프라인

```
입력   photo.jpg  ·  name "네옹"  ·  birthday 2023-05-14
  │
  ├─① 누끼        rembg → 알파 있는 강아지 + bbox          src/cutout.py
  ├─② 털색        누끼 안쪽만 k-means → accent · accent2    src/coat.py
  ├─③ 작물        생일 월 → 12작물 중 1 (룰 테이블)         src/crops.py
  ├─④ 화풍 맞춤   양자화·포스터라이즈                       src/pixelize.py
  ├─⑤ 카드 합성   카드 그림창에 contain 으로 얹음            src/compose.py
  ├─⑥ 장면 값     seed=강아지 id · place 문구 · motes/dew    src/scene.py
  └─⑦ 음악        provider → 루프 이음새 → OGG             src/music/, src/loop.py
  │
출력   scene.json  +  card.webp  +  subject.webp  +  bgm.ogg
```

①~⑥ 은 오프라인·0원·즉시고, **⑦만 네트워크·유료·느리다.** 그래서 ⑦이 실패해도
카드는 나온다 — `bgm` 이 `null` 이면 앱의 `SceneMusic` 이 조용히 아무것도 안 한다.

## 돌리는 법

```bash
uv sync                       # 3.12 고정. pip 은 쓰지 않는다
uv run pytest                 # 계약 왕복 · 12달 매핑 · fit 산출

# 한 장 뽑기 (음악은 mock — 네트워크·비용 없음)
uv run python -m src.pipeline \
  --photo photos/neong.jpg --name 네옹 --birthday 2023-05-14 \
  --assets templates/cards --out out/neong

# 서비스로 (포트를 바깥에 열지 않는다)
uv run python serve.py --mock
```

누끼(rembg)는 무겁고 가중치를 받아야 해서 갈라 두었다. 필요할 때만:

```bash
uv sync --extra cutout
```

없으면 `cutout.py` 가 곱게 실패한다 — 알파가 이미 있는 PNG 를 넣으면 누끼 없이도
나머지 파이프라인은 그대로 돈다.

## 서비스로 쓸 때

생성이 느리므로 **동기로 기다리지 않는다.** 접수하고 job id 를 주고, 상태는
물어보게 한다. 기다리게 했다가는 nginx 60초 타임아웃에 걸려 우리가 내지 않은 HTML
오류 페이지를 사용자가 받는다 (D-021 이 `/ask` 예열에서 실측한 함정).

```
POST /cards               photo(파일) · name · birthday(YYYY-MM-DD)  → 202 {id, status_url}
GET  /cards/{id}          queued / running / done / failed  (설비가 없으면 503)
GET  /cards/{id}/files/…  card.webp · subject.webp · scene.json · bgm.ogg
GET  /healthz             누끼 설비·음악 provider·원화 유무
```

`--mock` 이면 음악은 mock 이고, 카드 원화가 없으면 자리표 카드를 만들어 쓴다.
그래서 아무 설비 없이도 파이프라인 전체가 도는지 볼 수 있다.

## 카드 원화는 어디에 있나

`SAJOYO/DAENGS_CARDS` (공개) 의 `art/` 다. 원화는 git 밖에 두므로 거기서 받아
`templates/cards/` 에 넣는다.

```bash
git clone --depth 1 https://github.com/SAJOYO/DAENGS_CARDS /tmp/cards
cp /tmp/cards/art/cabbage-*.webp templates/cards/
```

그림창 좌표는 **재지 않는다.** 원화가 그림 영역만 투명하게 지운 완성 카드라
알파의 구멍이 곧 좌표다.

```bash
uv run python tools/art_window.py templates/cards/*-card-frame.webp
```

## 지금 비어 있는 것

- **⚠ 그림창이 뚫린 프레임은 배추 한 장뿐이다.** 나머지 11장은 도감의 `art/*.webp`
  가 그림·프레임·인쇄문구가 한 장에 다 구워진 통짜라 사진을 얹을 구멍이 없다.
  **지금 실제로 뽑을 수 있는 카드는 12월생(배추) 하나다**
- **`inset`·`anchor_y`** — 배추만 앱 배치에 맞춰 뒀다. 나머지는 "꽉 채워 가운데"
  기본값이고, 프레임이 생기면 카드마다 사람이 봐야 한다
- **앱의 `seedOf` 대조** — Kotlin `String.hashCode` 규약으로 구현했고 앱 구현을 직접
  보고 맞춘 것은 아니다 (CA-007). 다르면 `src/scene.py` 의 `seed_of` 한 함수만 고친다
- **음악 서비스** — 지금은 mock 뿐이다. 실제 연동은 별도 카드이고 첫 항목은 상업 이용
  라이선스 확인이다
- **강아지 테이블** — 이름·생일을 담을 DB 스키마가 아직 없어서 씨의 재료로
  `"이름:생일"` 을 쓴다

## 사람이 봐야 하는 곳

그림 판단은 AI 에게 위임하지 않는다 (앱 협업규칙 1절). 다음 둘은 **만들어서 봐야**
판정된다.

- 사진과 픽셀아트 프레임이 **한 화면에서 붙는가.** `tools/contact_sheet.py` 로 화풍
  옵션을 나란히 뽑아 고른다. 원본 크기가 아니라 **화면 크기에서** 본다.
- 생성 음악의 **루프 이음새.** 30초를 두 바퀴 이상 들어야 "툭" 이 들린다.

## 문서

- `src/contract.py` — `scene.json` 계약의 **유일한 출처.** 문서와 어긋나면 코드가 맞다
- `docs/decisions.md` — 되돌리기 번거로운 결정 (`CA-0xx`)
- `CLAUDE.md` — 이 폴더에서 지켜야 할 규칙
