"""얼굴 구멍 표와 구멍 합성.

표가 깨져 있으면 **읽을 때 죽어야 한다.** 뒤에서 죽으면 "카드는 나왔는데 구멍이
비어 있다" 가 되고, 그건 사람이 눈으로 보기 전까지 아무도 모른다.
"""

import numpy as np
import pytest
from PIL import Image

from src.compose import compose_face_hole, paste_in_hole
from src.crops import load_table
from src.cutout import Face, face_from
from src.holes import Hole, HoleCard, HoleTableError, hole_card, load_holes

# -- 표 ---------------------------------------------------------------------


def test_카드_열두장이_다_있다():
    assert sorted(load_holes()) == sorted(c.id for c in load_table().values())


def test_배추와_고구마는_앱에서_쓰던_값과_맞는다():
    """앱 `CardSlots.kt` 가 실기기에서 쓰던 두 값. 재는 방식이 맞는지의 근거다.

    사람이 잰 것이 아니라 `tools/measure_holes.py` 가 뚫린 자리를 읽은 값이라,
    이 둘이 어긋나면 재는 방식부터 틀린 것이다.
    """
    for card_id, want in (
        ("cabbage", (52.84, 44.71, 19.81, 14.27)),
        ("sweet-potato", (48.84, 40.76, 12.13, 8.80)),
    ):
        f = hole_card(card_id).face
        got = (f.cx, f.cy, f.rx, f.ry)
        assert all(abs(a - b) < 1.5 for a, b in zip(got, want)), f"{card_id}: {got} vs {want}"


def test_구멍이_카드_밖으로_나가면_읽을_때_죽는다(tmp_path):
    bad = tmp_path / "holes.toml"
    bad.write_text(
        "[cabbage]\nsize = [100, 100]\n"
        "face = { cx = 95, cy = 50, rx = 20, ry = 20 }\n",
        encoding="utf-8",
    )
    with pytest.raises(HoleTableError, match="밖으로"):
        load_holes(bad)


def test_카드가_모자라면_읽을_때_죽는다(tmp_path):
    bad = tmp_path / "holes.toml"
    bad.write_text(
        "[cabbage]\nsize = [100, 100]\n"
        "face = { cx = 50, cy = 50, rx = 20, ry = 20 }\n",
        encoding="utf-8",
    )
    with pytest.raises(HoleTableError, match="구멍이 없다"):
        load_holes(bad)


def test_모르는_카드는_곱게_알린다():
    with pytest.raises(HoleTableError, match="모르는 카드"):
        hole_card("kimchi")


# -- 합성 -------------------------------------------------------------------


def fake_face(size: int = 200) -> Face:
    """가운데가 어두운 정사각 크롭. 누끼가 도는 최소한의 그림."""
    a = np.full((size, size, 3), 235, np.uint8)
    a[40:170, 50:150] = 70
    return face_from(Image.fromarray(a), 0.85, anchor_x=100.0, chin_y=165.0)


def fake_slots(card: HoleCard, size: tuple[int, int] = (300, 400)) -> Image.Image:
    """구멍만 알파로 뚫린 카드 흉내 (`*-card-slots.webp` 와 같은 모양)."""
    from PIL import ImageDraw

    art = Image.new("RGBA", size, (40, 90, 50, 255))
    d = ImageDraw.Draw(art)
    for hole in (card.face, card.avatar):
        if hole is None:
            continue
        cx, cy, rx, ry = hole.px(size)
        d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=(0, 0, 0, 0))
    return art


def test_구멍이_얼굴로_채워진다():
    """구멍 안이 비면 카드에 검은 초승달로 보인다. 남는 자리가 없어야 한다."""
    card = hole_card("cabbage")
    art = fake_slots(card)
    got = compose_face_hole(art, fake_face(), card)

    cx, cy, rx, ry = card.face.px(got.card.size)
    alpha = np.asarray(got.card.getchannel("A"))
    yy, xx = np.mgrid[0:alpha.shape[0], 0:alpha.shape[1]]
    # 구멍 안쪽 80% 만 본다 — 가장자리는 페더링이라 반투명이 정상이다.
    inside = ((xx - cx) / (rx * 0.8)) ** 2 + ((yy - cy) / (ry * 0.8)) ** 2 <= 1
    assert inside.any()
    assert alpha[inside].min() > 200, "구멍 안이 비었다"


def test_얼굴은_카드_아래에_깔린다():
    """위에 얹으면 오려 붙인 스티커가 된다. 구멍 **바깥**은 원화 색이어야 한다."""
    card = hole_card("cabbage")
    got = compose_face_hole(fake_slots(card), fake_face(), card)
    assert got.card.convert("RGBA").getpixel((2, 2))[:3] == (40, 90, 50)


def test_아바타가_없어도_돈다():
    card = hole_card("cabbage")
    only_face = HoleCard(id=card.id, size=card.size, face=card.face, avatar=None)
    got = compose_face_hole(fake_slots(only_face), fake_face(), only_face)
    assert got.card.size == (300, 400)


def test_window_는_구멍의_사각형이다():
    """`Scene.window` 는 계약이 이미 가진 필드다. 새 어휘를 만들지 않는다 (CA-003)."""
    card = hole_card("cabbage")
    got = compose_face_hole(fake_slots(card), fake_face(), card)
    f = card.face
    assert got.window.x == pytest.approx(f.cx - f.rx, abs=0.01)
    assert got.window.w == pytest.approx(f.rx * 2, abs=0.01)


def test_구멍을_채우는_것은_턱_위쪽뿐이다():
    """스케일을 **턱 위 높이**로 잰다. 턱 아래(목·가슴)까지 넣으면 구멍 위가 빈다.

    턱이 낮을수록(= 턱 위가 넉넉할수록) 덜 키워도 구멍이 차므로 결과가 작아진다.
    턱 아래까지 셈에 넣으면 이 차이가 사라진다 — 그게 가지·피망·상추에서 구멍 위가
    검은 초승달로 남던 자리다.
    """
    card = hole_card("cabbage")
    art = fake_slots(card)
    base = fake_face()

    def fit_for(chin: float):
        face = Face(plain=base.plain, outlined=base.outlined, core=base.core,
                    cut=base.cut, anchor_x=base.anchor_x, chin_y=chin)
        return paste_in_hole(Image.new("RGBA", art.size, (0, 0, 0, 0)), face, card.face)

    # 세로가 스케일을 정하는 자리에서만 갈린다 — 턱 위 높이가 core 폭보다 작을 때다.
    assert fit_for(130.0).w < fit_for(100.0).w
