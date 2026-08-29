"""이머시브 장면 값 — seed · place · motes/dew.

`ImmersiveScene` 이 이미 기본값 있는 data class 라서 새 구조를 짤 것이 없다. 씨를
강아지 id 로 바꾸는 것만으로 유저마다 다른 장면이 나온다.

산책 데이터 같은 dev 쪽 값은 쓰지 않는다 (CA-002). `weather` · `time_of_day` 자리만
비워 둔다.
"""

import random
from datetime import date

from src.contract import Rect, Scene
from src.crops import Crop

# 앱의 seedOf 와 같은 값을 내야 먼지·이슬 배치가 앱에서 기대한 대로 굳는다.
# 앱 구현을 직접 보고 맞췄다 (Immersive.kt:113):
#
#     fun seedOf(text: String): Int = text.fold(7) { h, c -> h * 31 + c.code }
#
# **시작값이 0 이 아니라 7 이다.** Java 의 String.hashCode 규약으로 지레짐작했다가
# 전부 다른 값을 내고 있었다. 나머지(31 진법 · UTF-16 코드 단위 · 32비트 넘침)는 같다.
_INT32 = 1 << 32
_INT31 = 1 << 31
# 앱의 fold 시작값. 0 이 아니다.
_SEED_INIT = 7

# 장면 문구 후보. 씨로 고르므로 강아지마다 고정된다.
FEATURES = (
    "이슬 맺힌 텃밭",
    "이슬 맺힌 {crop}밭",
    "바람 드는 이랑",
    "볕 잘 드는 두렁",
    "비 갠 뒤의 고랑",
    "서리 내린 {crop}밭",
)
TIMES = (
    "해 뜨기 직전",
    "이른 아침",
    "볕이 앉은 한낮",
    "해 질 무렵",
    "달 뜬 뒤",
)

# 앱의 기본값(motes 52 · leaves 7 · dew 15)을 가운데 두고 벌린 범위다.
MOTES_RANGE = (40, 64)
LEAVES_RANGE = (5, 10)
DEW_RANGE = (10, 20)


def seed_of(text: str) -> int:
    """앱의 `seedOf` 와 같은 값 (Immersive.kt:113).

    UTF-16 코드 단위로 센다 — 이름에 이모지가 들어가면 파이썬의 코드포인트와
    갈라지는 자리다.
    """
    h = _SEED_INIT
    units = text.encode("utf-16-be")
    for i in range(0, len(units), 2):
        code = (units[i] << 8) | units[i + 1]
        h = (h * 31 + code) & (_INT32 - 1)
    return h - _INT32 if h >= _INT31 else h


def dog_id(name: str, birthday: date) -> str:
    """강아지 하나를 가리키는 문자열.

    강아지 테이블이 아직 없다(스키마에 존재하지 않는다). 지금은 이름과 생일을 붙여
    쓰고, DB 가 생기면 그 id 를 `build_scene(dog_key=...)` 로 넣으면 된다 — 그때
    씨가 바뀌므로 장면 배치도 한 번 바뀐다.
    """
    return f"{name.strip()}:{birthday.isoformat()}"


def has_batchim(word: str) -> bool:
    """마지막 글자에 받침이 있는가. '네옹이의' 와 '코코의' 를 가르는 것."""
    if not word:
        return False
    code = ord(word[-1])
    if not (0xAC00 <= code <= 0xD7A3):
        return False
    return (code - 0xAC00) % 28 != 0


def possessive(name: str) -> str:
    """'네옹' → '네옹이의', '코코' → '코코의'."""
    name = name.strip()
    return f"{name}이의" if has_batchim(name) else f"{name}의"


def place_of(name: str, crop: Crop, seed: int) -> str:
    rnd = random.Random(seed)
    feature = rnd.choice(FEATURES).format(crop=crop.korean)
    time = rnd.choice(TIMES)
    return f"{possessive(name)} {feature} · {time}"


def build_scene(
    *,
    name: str,
    birthday: date,
    crop: Crop,
    accent: str,
    accent2: str,
    fit: Rect,
    window: Rect,
    card: str,
    subject: str | None = None,
    frame: str | None = None,
    back: str | None = None,
    bgm: str | None = None,
    dog_key: str | None = None,
) -> Scene:
    """장면 값 한 벌. 같은 강아지면 언제 돌려도 같은 값이 나온다."""
    key = dog_key or dog_id(name, birthday)
    seed = seed_of(key)
    rnd = random.Random(seed)

    return Scene(
        place=place_of(name, crop, seed),
        seed=seed,
        motes=rnd.randint(*MOTES_RANGE),
        leaves=rnd.randint(*LEAVES_RANGE),
        dew=rnd.randint(*DEW_RANGE),
        accent=accent,
        accent2=accent2,
        back=back,
        subject=subject,
        card=card,
        frame=frame,
        fit=fit,
        window=window,
        bgm=bgm,
        # dev 를 부르지 않으므로 비운다 (CA-002). 자리는 계약에 있다.
        weather=None,
        time_of_day=None,
    )
