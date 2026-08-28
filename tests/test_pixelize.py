import numpy as np
import pytest
from PIL import Image

from src.pixelize import STYLES, apply_style, scaled, style_by_name, variants
from tests.fakes import fake_dog


def colors(img: Image.Image) -> int:
    return len(img.convert("RGB").getcolors(1 << 16))


def test_모든_후보가_크기를_바꾸지_않는다():
    dog = fake_dog((240, 180))
    for style in variants():
        assert apply_style(dog, style).size == dog.size


def test_양자화하면_색이_준다():
    dog = Image.effect_mandelbrot((128, 128), (-2, -1.5, 1, 1.5), 40).convert("RGBA")
    assert colors(apply_style(dog, STYLES["pixel-hard"])) <= 16


def test_알파는_이진화되어_반투명_가장자리가_없다():
    dog = fake_dog((160, 120))
    out = apply_style(dog, STYLES["pixel"])
    assert set(np.unique(np.asarray(out.getchannel("A"))).tolist()) <= {0, 255}


def test_raw_는_알파를_건드리지_않는다():
    dog = fake_dog((80, 60))
    dog.putpixel((0, 0), (10, 10, 10, 77))
    assert apply_style(dog, STYLES["raw"]).getpixel((0, 0))[3] == 77


def test_칸을_뭉치면_같은_색_덩어리가_커진다():
    dog = Image.effect_mandelbrot((120, 120), (-2, -1.5, 1, 1.5), 60).convert("RGBA")
    소 = colors(apply_style(dog, STYLES["pixel"]))
    대 = colors(apply_style(dog, STYLES["pixel-hard"]))
    assert 대 < 소


def test_파스텔은_톤을_눕힌다():
    dog = fake_dog((80, 60), color=(200, 40, 40))
    before = dog.getpixel((40, 30))[:3]
    after = apply_style(dog, STYLES["pastel"]).getpixel((40, 30))[:3]
    assert min(after) > min(before)      # 흰 쪽으로 섞였다


def test_칸_크기는_카드_크기에_비례로_커진다():
    assert scaled(STYLES["pixel"], 2.0).block == 6
    assert scaled(STYLES["pixel"], 0.1).block == 1      # 0 으로는 안 내려간다


def test_모르는_화풍은_이름을_알려_주고_죽는다():
    with pytest.raises(ValueError, match="모르는 화풍"):
        style_by_name("없는것")
