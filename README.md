# card_agent

강아지 **얼굴 사진 · 이름 · 생일** 세 개로 그 강아지 전용 카드 한 장과 이머시브 장면
값을 뽑는다. 카드 디자인과 이머시브 구조는 지금 앱의 것을 그대로 쓰고, 채워 넣는
값만 개인화한다.

**카드는 12장 중 랜덤이다** (CA-017). 부를 때마다 새로 뽑으므로 같은 강아지도 다시
돌리면 다른 카드가 나온다. 뽑은 결과는 `scene.json` 에 박힌다 — 그게 기록이라 따로
저장할 DB 가 필요 없다. 생일은 장면 값과 카드 번호판에 쓰이고 **어느 카드가 나올지는
안 정한다.**

바깥과는 **`scene.json` 하나로만** 말한다. dev 레포의 config·DB·모델을 하나도 부르지
않는 독립 폴더라, 개인 레포에서 굴리다가 폴더째 dev 에 넣을 수 있다. 사본 추적은
`pyproject.toml` 의 `version` 한 줄로 한다.

## 파이프라인

```
입력   photo.jpg  ·  name "네옹"  ·  birthday 2023-05-14
  │
  ├─① 누끼        rembg → 알파 있는 강아지 + bbox          src/cutout.py
  ├─② 털색        누끼 안쪽만 k-means → accent · accent2    src/coat.py
  ├─③ 작물        12작물 중 **랜덤** (뽑기)                 src/crops.py
  ├─④ 화풍 맞춤   양자화·포스터라이즈                       src/pixelize.py
  ├─⑤ 카드 합성   카드 그림창에 contain 으로 얹음            src/compose.py
  ├─⑥ 장면 값     seed=강아지 id · place 문구 · motes/dew    src/scene.py
  └─⑦ 음악        provider → 루프 이음새 → OGG             src/music/, src/loop.py
  │
출력   scene.json  +  card.webp  +  subject.webp  +  bgm.ogg
```

①~⑥ 은 오프라인·0원·즉시고, **⑦만 네트워크·유료·느리다.** 그래서 ⑦이 실패해도
카드는 나온다 — `bgm` 이 `null` 이면 앱의 `SceneMusic` 이 조용히 아무것도 안 한다.

## 카드 원화가 두 종류다 — 길도 둘이다

```
*-card-frame.webp   그림 영역이 통째로 비어 있다   → 강아지 **전체**를 contain
                    templates/windows.toml  ·  compose_card()

*-card-slots.webp   채소는 인쇄돼 있고 얼굴 자리만 → 강아지 **얼굴만** cover
                    templates/holes.toml    ·  compose_face_hole()
```

새로 받은 12장이 뒤쪽이다. 앱 PR #19 가 코틀린으로 채소를 그려 보고 "종이 오린 꼴"
이라며 되돌린 뒤, **채소는 저쪽이 그리고 우리는 자리만 비운다** 로 정한 방식이다
(CA-013).

얼굴 갈래는 앞에 한 단계가 더 붙는다 — 사진 어디가 얼굴인지 알아야 한다.

```
사진 → 키포인트 → 얼굴 상자·목선·가로 기준점 → 누끼 → 구멍에 → 카드
       dogpose.py   facebox.py                cutout.py  compose.py
```

```bash
uv sync --extra facebox                       # opencv 하나. 가중치를 안 받는다
uv run python tools/fetch_dog_pose.py         # 데이터셋 386MB, 한 번만

# ★ 관문 — 학습 전에 정답 라벨로 얼굴 상자가 쓸 만한지 본다
uv run python tools/verify_face_box.py --split val --sweep

# 강아지 하나를 카드 12장 전부에
uv run python tools/make_face_card.py --from-label --dog n02090622_2518 --sheet
```

**모델은 아직 없다.** 여기 결과는 전부 정답 라벨 기준이고, 그건 "모델이 완벽했을 때의
상한" 이다. 학습은 `tools/train_dog_pose.py` 를 Colab 에서 돌린다 — GPU 가 필요하고
**AGPL-3.0 이 그때 들어온다** (CA-016).

### 관문에서 나온 것 (val 1,703장, 정답 라벨)

| 지표 | 값 | 통과선 |
| --- | --- | --- |
| 가시성 (얼굴 kp ≥3) | **97.5%** | 85% |
| 재현율 (상자가 얼굴 kp 를 전부 담음) | **100%** | 97% |
| 오염률 (구멍 안 · 목선 위에 몸통 kp) | **19.7%** | 20% 이하 |

숫자는 `docs/verify-report.json` 에 그대로 있다 (마진 스윕 27칸 포함). 마진은 짐작이
아니라 그 27칸을 훑어 골랐고, `recall` 은 27칸 전부 1.000 이라 **고르는 데는 못 쓴다** —
튜닝 신호가 아니라 계산이 안 깨졌다는 확인이다.

구멍 좌표는 사람이 안 잰다. 받은 카드가 이미 뚫려 있어서 **구멍이 곧 좌표다.**

```bash
uv run python tools/measure_holes.py --punch --write
```

잰 값이 맞는지는 확인됐다 — 배추·고구마가 앱 `CardSlots.kt` 의 실측값과 1.5%p 안쪽으로
맞는다 (`tests/test_holes.py` 가 그걸 지킨다).

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

## 지금 비어 있는 것

- **그림창 좌표 11장** — `templates/windows.toml`(직사각)은 배추 값만 앱에서 왔고,
  나머지는 자리값이다. 얼굴 구멍(`templates/holes.toml`)은 12장 다 실측이다
- **학습된 포즈 모델** — 얼굴 상자는 지금 **정답 라벨**로만 검증했다. 폰으로 찍은
  실사 사진으로는 아직 안 봤고, `facebox.CHIN_DROP` 은 맞춰 볼 정답이 없어 눈으로
  고른 값이다 (CA-015)
- **카드 글자(이름·번호)** — 프레임 배치가 12장 공통이 아니라 자리를 따로 재야 한다.
  피망·당근·가지·단호박은 검은 프레임에 제목이 가운데 온다
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
- 얼굴이 **구멍에 제대로 앉았는가.** `tools/make_face_card.py --sheet` 가 120·72·40dp
  로 나란히 낸다. 구멍 크기가 카드마다 3.4배까지 차이 나서 (피망 rx 18.4% · 상추 9.97%),
  **40dp 에서 피망은 개로 읽히는데 상추·브로콜리는 안 읽힌다.**

## 문서

- `src/contract.py` — `scene.json` 계약의 **유일한 출처.** 문서와 어긋나면 코드가 맞다
- `docs/decisions.md` — 되돌리기 번거로운 결정 (`CA-0xx`)
- `CLAUDE.md` — 이 폴더에서 지켜야 할 규칙
