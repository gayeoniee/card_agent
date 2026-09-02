from datetime import date

from src.contract import Rect
from src.crops import crop_by_id
from src.scene import (
    DEW_RANGE,
    MOTES_RANGE,
    build_scene,
    dog_id,
    has_batchim,
    place_of,
    possessive,
    seed_of,
)

RECT = Rect(x=1, y=1, w=1, h=1)


def scene(name="네옹", birthday=date(2023, 5, 14), **kw):
    return build_scene(
        name=name, birthday=birthday, crop=crop_by_id("danhobak"),   # 카드는 랜덤이라(CA-017) 여기서는 한 장을 못박는다
        accent="#EEBE93", accent2="#F5D9BE", fit=RECT, window=RECT,
        card="card.webp", **kw,
    )


def test_해시가_kotlin_String_hashCode_와_같다():
    """앱의 seedOf 와 값이 같아야 먼지·이슬 배치가 기대한 대로 굳는다."""
    assert seed_of("") == 0
    assert seed_of("abc") == 96354
    assert seed_of("Aa") == seed_of("BB") == 2112          # 유명한 충돌
    assert seed_of("polygenelubricants") == -2147483648    # 32비트로 넘친 값


def test_이모지_이름도_UTF16_으로_센다():
    """파이썬 코드포인트로 세면 앱과 갈라지는 자리 — 대리쌍 둘로 세야 한다."""
    assert seed_of("🐶") == 0xD83D * 31 + 0xDC36


def test_강아지가_다르면_씨가_다르다():
    assert scene("네옹").seed != scene("코코").seed
    assert scene("네옹", date(2023, 5, 14)).seed != scene("네옹", date(2022, 5, 14)).seed


def test_같은_강아지면_언제_돌려도_같다():
    assert scene() == scene()


def test_받침에_따라_조사가_바뀐다():
    assert possessive("네옹") == "네옹이의"
    assert possessive("코코") == "코코의"
    assert possessive("뭉치") == "뭉치의"
    assert possessive("봄") == "봄이의"
    assert has_batchim("강") and not has_batchim("가")


def test_한글이_아닌_이름은_받침이_없는_것으로_본다():
    assert possessive("Neo") == "Neo의"


def test_장소_문구에_이름과_때가_들어간다():
    place = place_of("네옹", crop_by_id("danhobak"), seed_of("네옹:2023-05-14"))
    assert place.startswith("네옹이의 ")
    assert " · " in place


def test_먼지와_이슬이_정해진_범위_안이다():
    for name in ("네옹", "코코", "보리", "뭉치", "까미", "하양"):
        s = scene(name)
        assert MOTES_RANGE[0] <= s.motes <= MOTES_RANGE[1]
        assert DEW_RANGE[0] <= s.dew <= DEW_RANGE[1]


def test_dev_를_부르지_않으므로_날씨_자리는_비어_있다():
    s = scene()
    assert s.weather is None and s.time_of_day is None


def test_음악이_없으면_bgm_은_null_이다():
    assert scene().bgm is None
    assert scene(bgm="bgm.ogg").bgm == "bgm.ogg"


def test_DB_가_생기면_그_id_로_씨를_바꿀_수_있다():
    assert scene(dog_key="dog-1234").seed == seed_of("dog-1234")
    assert dog_id("네옹", date(2023, 5, 14)) == "네옹:2023-05-14"
