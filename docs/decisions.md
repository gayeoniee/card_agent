# 결정 기록 (card_agent)

되돌리기 번거로운 결정만 적는다. 번호는 `CA-0xx` — dev 의 `D-0xx` 와 부딪히지 않게
접두사를 갈라 둔다.

---

## CA-001 · Python 3.12 고정, 의존성은 uv 로만

`requires-python = ">=3.12,<3.13"` 과 `.python-version` 을 같이 둔다. `pip` 을 쓰지
않고 `uv add`/`uv remove` 로만 고치며 `uv.lock` 을 커밋한다.

dev 의 D-001 과 같은 이유다. 여기서 판을 달리 깔면 폴더째 dev 로 옮기는 순간
락파일이 두 벌이 된다.

## CA-002 · dev 레포를 부르지 않는다 — 계약은 `scene.json` 하나

`card_agent` 는 dev 의 config·DB·모델을 하나도 import 하지 않는다. 바깥과는
`scene.json` 한 장으로만 말하고, 그 출처는 `src/contract.py` 하나다.

개인 레포에서 굴리다가 폴더째 dev 에 넣을 수 있어야 한다는 것이 이 설계의 목적이다.
dev 를 한 줄이라도 부르는 순간 그 성질이 사라진다. 사본 추적은 명시 대신
`pyproject.toml` 의 `version` 으로 한다.

이 결정의 대가: 산책 데이터 같은 dev 쪽 값을 장면에 못 쓴다. `scene.weather` ·
`scene.time_of_day` 자리만 optional 로 비워 두고, 채우는 일은 나중 카드로 넘긴다.

## CA-003 · 계약 필드 이름은 앱을 따라간다

`statLabel` · `fit` · `motes` 처럼 앱의 `ImmersiveScene` · `DexCard` 에 이미 있는
이름을 그대로 쓴다. 파이썬 쪽 이름(`stat_label`)은 alias 로만 오간다.

새 어휘를 만들면 앱에 붙일 때 이름 대조표가 하나 더 생긴다. 그 표는 아무도
갱신하지 않는다. 모르는 필드는 `extra="forbid"` 로 거절해서, 이름이 어긋난 JSON 이
조용히 통과하지 못하게 한다.

## CA-004 · 생일 월 → 작물은 룰 테이블이다 (모델 아님)

카드가 정확히 12장이고 달도 12개라 1:1 로 떨어진다. `templates/crops.toml` 표
하나로 끝내고 모델을 쓰지 않는다.

표의 `no` · `statLabel` · `stat` · `foil` 중 **확인된 값은 배추 = No.01 하나뿐**이다.
나머지 11장은 앱의 실제 `DexCard` 목록을 손에 넣기 전의 자리값이고, 목록이 오면
TOML 만 고치면 된다 — 파이프라인은 건드릴 것이 없다.
