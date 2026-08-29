from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.compose import template_for
from src.contract import Rect
from src.title import TitleUnavailable, blank_title, draw_title, personalize_title

CARD = Path("templates/cards/cabbage-card.webp")
FRAME = Path("templates/cards/cabbage-card-frame.webp")
FONT = Path("templates/fonts/Cinzel-Bold.ttf")

원화있음 = pytest.mark.skipif(not (CARD.exists() and FRAME.exists()),
                          reason="도감 원화가 없다 — README 참고")
설비있음 = pytest.mark.skipif(not FONT.exists(), reason="폰트가 없다 — uv sync --extra title")

BOX = Rect(x=24.69, y=3.82, w=44.81, h=5.87)


def 밝은픽셀(img: Image.Image, box: Rect) -> int:
    w, h = img.size
    crop = img.convert("RGB").crop((round(box.x / 100 * w), round(box.y / 100 * h),
                                   round((box.x + box.w) / 100 * w),
                                   round((box.y + box.h) / 100 * h)))
    return int((np.asarray(crop).mean(axis=2) > 150).sum())


def test_제목바_좌표가_표에_있다():
    assert template_for("cabbage").title is not None


@원화있음
def test_지우면_흰_글자가_사라진다():
    card = Image.open(CARD)
    assert 밝은픽셀(blank_title(card, BOX), BOX) < 밝은픽셀(card, BOX) * 0.05


@원화있음
def test_지워도_알파를_잃지_않는다():
    """RGB 로만 작업하면 그림창이 막혀 강아지가 통째로 가린다."""
    frame = Image.open(FRAME).convert("RGBA")
    지운것 = blank_title(frame, BOX)
    assert 지운것.mode == "RGBA"
    assert 지운것.getchannel("A").getextrema()[0] == 0          # 구멍이 살아 있다
    before = np.asarray(frame.getchannel("A"))
    after = np.asarray(지운것.getchannel("A"))
    assert np.array_equal(before, after)


@원화있음
@설비있음
def test_이름을_찍으면_글자가_다시_생긴다():
    card = Image.open(CARD)
    나온것 = personalize_title(card, BOX, "Neong Neo")
    assert 밝은픽셀(나온것, BOX) > 밝은픽셀(card, BOX) * 0.4


@원화있음
@설비있음
def test_긴_이름은_바_안에서_줄인다():
    card = Image.open(CARD)
    긴것 = draw_title(blank_title(card, BOX), BOX, "Wolfgang Amadeus Neo")
    w, h = 긴것.size
    x1 = round((BOX.x + BOX.w) / 100 * w)
    옆칸 = 긴것.convert("RGB").crop((x1 + 6, round(BOX.y / 100 * h),
                                  x1 + 40, round((BOX.y + BOX.h) / 100 * h)))
    # 은색 테두리 바깥으로 글자가 새지 않는다 (원본과 같은 자리라 흰 글자가 없어야 한다)
    assert (np.asarray(옆칸).mean(axis=2) > 240).mean() < 0.05


@원화있음
def test_설비가_없으면_곱게_실패한다(monkeypatch):
    monkeypatch.setattr("src.title.FONTS_DIR", Path("없는폴더"))
    with pytest.raises(TitleUnavailable):
        draw_title(Image.open(CARD), BOX, "Neong Neo", fonts_dir=Path("없는폴더"))
