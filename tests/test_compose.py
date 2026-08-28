import pytest
from PIL import Image

from src.compose import (
    CardArtMissing,
    WindowTableError,
    compose_card,
    contain,
    load_windows,
    open_frame,
    round_corners,
    save_webp,
    template_for,
    trim_alpha,
)
from src.contract import Rect
from src.crops import load_table
from src.pixelize import STYLES
from tests.fakes import fake_dog, fake_frame


def test_카드_열두장에_모두_그림창이_있다():
    windows = load_windows()
    assert {c.id for c in load_table().values()} <= set(windows)


def test_배추_창_좌표는_앱_값_그대로다():
    """window 는 우리가 정하는 값이 아니라 앱이 쓰는 상수다 (cards.mjs 의 scene.window)."""
    window = template_for("cabbage").window
    assert (window.x, window.y, window.w, window.h) == (4.91, 10.28, 90.51, 81.58)


def test_배추_구멍은_프레임에서_뽑은_값이다():
    """자를 대지 않는다 — 알파의 구멍이 곧 좌표다."""
    art = template_for("cabbage").art
    assert template_for("cabbage").measured is True
    # 앱이 손으로 잰 window 와 0.4%p 안에서 맞는다
    for 뽑은값, 잰값 in ((art.x, 4.91), (art.y, 10.28), (art.w, 90.51), (art.h, 81.58)):
        assert abs(뽑은값 - 잰값) < 0.4


def test_원화_이름은_지정하지_않으면_id_에서_짓는다():
    assert template_for("cabbage").frame == "cabbage-card-frame.webp"
    assert template_for("danhobak").frame == "danhobak-card-frame.webp"


def test_프레임이_있는_카드는_아직_배추뿐이다():
    """나머지 11장은 통짜 webp 라 사진을 얹을 구멍이 없다 — 자리값으로 표시해 둔다."""
    잰것 = [t.id for t in load_windows().values() if t.measured]
    assert 잰것 == ["cabbage"]


def test_contain_은_비율을_지키고_가운데에_넣는다():
    x, y, w, h = contain((200, 100), (10, 20, 100, 100))
    assert (w, h) == (100, 50)          # 가로가 먼저 닿는다
    assert (x, y) == (10, 45)           # 세로로 가운데
    assert abs(w / h - 200 / 100) < 0.01


def test_contain_은_창_밖으로_나가지_않는다():
    for subject in [(1, 900), (900, 1), (37, 41), (1000, 1000)]:
        x, y, w, h = contain(subject, (5, 7, 120, 90))
        assert 5 <= x and 7 <= y and x + w <= 125 and y + h <= 97


def test_fit_은_재는_값이_아니라_코드가_뱉는_값이다():
    """그림창을 상수로 두고 누끼를 맞추면 fit 이 나온다 — 방향을 뒤집은 자리."""
    window = template_for("cabbage").art
    composed = compose_card(fake_frame(window), fake_dog(), window)
    fit = composed.fit
    assert window.x <= fit.x and window.y <= fit.y
    assert fit.x + fit.w <= window.x + window.w + 0.01
    assert fit.y + fit.h <= window.y + window.h + 0.01
    assert composed.art_window == window


def test_fit_이_원본_비율을_지킨다():
    window = template_for("cabbage").art
    frame = fake_frame(window, (600, 840))
    dog = fake_dog((400, 200))          # trim 하면 320x120 짜리 타원
    composed = compose_card(frame, dog, window)
    trimmed = trim_alpha(dog)
    want = trimmed.width / trimmed.height
    got = (composed.fit.w * 600) / (composed.fit.h * 840)
    assert abs(got - want) / want < 0.02


def test_강아지가_그림창_안에만_보인다():
    """원화를 위에 얹으므로 창 밖으로 삐져나오면 안 된다."""
    window = Rect(x=25, y=25, w=50, h=50)
    frame = fake_frame(window, (400, 400))
    card = compose_card(frame, fake_dog((300, 300)), window).card
    assert card.getpixel((10, 10))[:3] == (26, 48, 30)      # 창 밖은 원화 색
    assert card.getpixel((200, 200))[3] == 255              # 창 안은 채워졌다


def test_화풍은_그림창_크기로_줄인_뒤_입힌다():
    """먼저 입히고 줄이면 칸이 보간으로 뭉개진다."""
    window = template_for("cabbage").art
    frame = fake_frame(window)
    plain = compose_card(frame, fake_dog(), window)
    pixel = compose_card(frame, fake_dog(), window, style=STYLES["pixel-hard"])
    assert plain.fit == pixel.fit                            # 화풍이 자리를 바꾸지 않는다
    assert len(pixel.subject.convert("RGB").getcolors(1 << 16)) < len(
        plain.subject.convert("RGB").getcolors(1 << 16)
    )


def test_trim_은_투명한_여백을_잘라_낸다():
    dog = fake_dog((400, 300), margin=40)
    assert trim_alpha(dog).size == (321, 221)


def test_trim_은_알파_먼지에_속지_않는다():
    dog = fake_dog((400, 300), margin=40)
    dog.putpixel((1, 1), (255, 255, 255, 3))     # 눈에 안 보이는 먼지
    assert trim_alpha(dog).size == (321, 221)


def test_모서리를_깎으면_귀퉁이가_투명해진다():
    card = Image.new("RGBA", (200, 300), (255, 0, 0, 255))
    out = round_corners(card, 10)
    assert out.getpixel((0, 0))[3] == 0
    assert out.getpixel((100, 150))[3] == 255


def test_원화가_없으면_곱게_알린다(tmp_path):
    with pytest.raises(CardArtMissing):
        open_frame(template_for("cabbage"), tmp_path)


def test_그림창이_카드_밖이면_표를_읽다가_죽는다(tmp_path):
    broken = tmp_path / "windows.toml"
    broken.write_text("[baechu]\nx = 50\ny = 0\nw = 90\nh = 10\n", encoding="utf-8")
    with pytest.raises(WindowTableError):
        load_windows(broken)


def test_카드가_빠지면_표를_읽다가_죽는다(tmp_path):
    broken = tmp_path / "windows.toml"
    broken.write_text("[baechu]\nx = 1\ny = 1\nw = 90\nh = 90\n", encoding="utf-8")
    with pytest.raises(WindowTableError):
        load_windows(broken)


def test_webp_는_알파를_잃지_않는다(tmp_path):
    window = template_for("cabbage").art
    card = compose_card(fake_frame(window), fake_dog(), window, corner_radius_pct=6).card
    path = save_webp(card, tmp_path / "card.webp")
    assert Image.open(path).convert("RGBA").getpixel((0, 0))[3] == 0
