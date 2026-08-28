from datetime import date

import pytest

from src.crops import CropTableError, crop_for_birthday, crop_for_month, load_table


def test_열두달이_빠짐없이_매핑된다():
    table = load_table()
    assert sorted(table) == list(range(1, 13))


def test_도감번호와_id_가_겹치지_않는다():
    crops = load_table().values()
    assert sorted(c.no for c in crops) == list(range(1, 13))
    assert len({c.id for c in crops}) == 12


def test_배추는_No_01_이다():
    """앱의 이머시브가 No.01 배추다 — 표에서 확인된 유일한 값."""
    baechu = next(c for c in load_table().values() if c.id == "baechu")
    assert baechu.no == 1


def test_5월생은_단호박_카드다():
    crop = crop_for_birthday(date(2023, 5, 14))
    assert (crop.no, crop.id, crop.stat_label, crop.stat, crop.foil) == (
        5,
        "danhobak",
        "CRUNCH",
        840,
        "Oilslick",
    )


def test_연도와_날짜는_카드를_바꾸지_않는다():
    assert crop_for_birthday(date(2019, 5, 1)) == crop_for_birthday(date(2024, 5, 31))


def test_월_밖의_값은_거절한다():
    for bad in (0, 13, -1):
        with pytest.raises(ValueError):
            crop_for_month(bad)


def test_카드로_바꾸면_계약을_만족한다():
    card = crop_for_month(5).to_card(art="card.webp")
    assert card.stat_label == "CRUNCH"
    assert card.model_dump(by_alias=True)["statLabel"] == "CRUNCH"


def test_달이_비면_표를_읽다가_죽는다(tmp_path):
    broken = tmp_path / "crops.toml"
    broken.write_text(
        '[[crop]]\nmonth = 1\nno = 1\nid = "a"\nkorean = "가"\nname = "A"\n'
        'statLabel = "X"\nstat = 1\nfoil = "F"\n',
        encoding="utf-8",
    )
    with pytest.raises(CropTableError):
        load_table(broken)
