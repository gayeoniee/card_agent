"""카드 12장 표와 뽑기.

표가 깨져 있으면 **읽을 때 죽어야 한다.** 뒤에서 죽으면 카드가 한 장 빠진 채로
서비스가 돌고, 그건 그 카드가 뽑힐 때까지 아무도 모른다.
"""

import random

import pytest

from src.crops import CARD_COUNT, CropTableError, crop_by_id, draw_crop, load_table


def test_카드가_열두장이다():
    assert len(load_table()) == CARD_COUNT


def test_도감번호와_id_가_겹치지_않는다():
    crops = load_table().values()
    assert sorted(c.no for c in crops) == list(range(1, CARD_COUNT + 1))
    assert len({c.id for c in crops}) == CARD_COUNT


def test_배추는_No_01_이다():
    """앱 `DexCards.kt` 의 첫 줄. 표가 앱에서 온 것인지 보는 눈이다 (CA-012)."""
    assert crop_by_id("cabbage").no == 1
    assert crop_by_id("cabbage").korean == "배추"


def test_표에_없는_id_는_곱게_알린다():
    with pytest.raises(CropTableError, match="모르는 카드"):
        crop_by_id("kimchi")


def test_계약의_Card_로_넘어간다():
    card = crop_by_id("tomato").to_card(art="card.webp")
    assert card.id == "tomato" and card.art == "card.webp"
    assert card.stat_label  # statLabel alias 로 오간다


# -- 뽑기 -------------------------------------------------------------------


def test_뽑은_카드는_표_안의_것이다():
    ids = {c.id for c in load_table().values()}
    for _ in range(50):
        assert draw_crop().id in ids


def test_같은_rng_를_주면_재현된다():
    """테스트가 고정할 수 있어야 한다. 서비스는 씨를 안 준다 (CA-017)."""
    a = [draw_crop(random.Random(7)).id for _ in range(5)]
    b = [draw_crop(random.Random(7)).id for _ in range(5)]
    assert a == b


def test_뽑기는_한_장에_고이지_않는다():
    """생일로 정하던 때와 달리 **부를 때마다 새로 뽑는다** (CA-017).

    12장 중 균등이라 200번이면 한 장만 나올 확률은 사실상 0 이다.
    """
    got = {draw_crop().id for _ in range(200)}
    assert len(got) > 1
    # 균등이면 200번에 12장이 다 나오는 것이 정상이다. 하나라도 빠지면 표를
    # 일부만 뽑고 있다는 뜻이라 그때 봐야 한다.
    assert got == {c.id for c in load_table().values()}


# -- 깨진 표 ----------------------------------------------------------------


def _write(tmp_path, rows: str):
    path = tmp_path / "crops.toml"
    path.write_text(rows, encoding="utf-8")
    return path


ONE = '''
[[crop]]
no = 1
id = "cabbage"
korean = "배추"
name = "Cabbage Neo"
statLabel = "CRUNCH"
stat = 820
foil = "Prism"
'''


def test_장수가_모자라면_읽을_때_죽는다(tmp_path):
    with pytest.raises(CropTableError, match="12장이 아니다"):
        load_table(_write(tmp_path, ONE))


def test_칸이_비면_읽을_때_죽는다(tmp_path):
    with pytest.raises(CropTableError, match="foil"):
        load_table(_write(tmp_path, ONE.replace('foil = "Prism"\n', "")))


def test_id_가_겹치면_읽을_때_죽는다(tmp_path):
    with pytest.raises(CropTableError, match="겹친다"):
        load_table(_write(tmp_path, ONE + ONE.replace("no = 1", "no = 2")))


def test_도감번호가_1부터_12를_안_덮으면_죽는다(tmp_path):
    twelve = "".join(
        ONE.replace("no = 1", f"no = {n}").replace('"cabbage"', f'"c{n}"')
        for n in range(1, 13)
    )
    broken = twelve.replace("no = 12", "no = 13")
    with pytest.raises(CropTableError, match="한 번씩 덮지 않는다"):
        load_table(_write(tmp_path, broken))
