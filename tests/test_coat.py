import pytest
from PIL import Image, ImageDraw

from src.coat import LIGHTEN, MIN_V, coat_colors, lighten, to_hex
from src.contract import Scene
from tests.fakes import fake_dog


def rgb(hexstr: str) -> tuple[int, int, int]:
    return tuple(int(hexstr[i:i + 2], 16) for i in (1, 3, 5))


def test_털색이_그대로_accent_가_된다():
    coat = coat_colors(fake_dog((400, 300), color=(238, 190, 147)))
    assert coat.accent == "#EEBE93"


def test_배경은_세지_않는다():
    """알파 밖에 요란한 색을 깔아 두어도 뽑히면 안 된다."""
    dog = fake_dog((300, 220), color=(200, 160, 120))
    bg = Image.new("RGBA", dog.size, (0, 255, 0, 0))     # 알파 0 인 초록
    bg.alpha_composite(dog)
    coat = coat_colors(bg)
    r, g, b = rgb(coat.accent)
    assert not (g > r and g > b)


def test_accent2_는_accent_의_밝은_짝이다():
    coat = coat_colors(fake_dog(color=(120, 90, 70)))
    assert rgb(coat.accent2) == tuple(
        round(c + (255 - c) * LIGHTEN) for c in rgb(coat.accent)
    )
    assert sum(rgb(coat.accent2)) > sum(rgb(coat.accent))


def test_검은_개도_빛으로_쓸_수_있는_색이_나온다():
    """털색 그대로 쓰면 이머시브에서 아무것도 안 보인다."""
    coat = coat_colors(Image.new("RGBA", (60, 60), (8, 8, 10, 255)))
    assert max(rgb(coat.accent)) / 255 >= MIN_V - 0.01


def test_같은_사진이면_같은_색이_나온다():
    dog = fake_dog((320, 240))
    draw = ImageDraw.Draw(dog)
    draw.ellipse((40, 40, 120, 120), fill=(60, 70, 200, 255))
    assert coat_colors(dog) == coat_colors(dog)


def test_계약이_받는_모양의_hex_다():
    coat = coat_colors(fake_dog())
    Scene(place="ㅁ", seed=1, motes=1, dew=1, accent=coat.accent, accent2=coat.accent2,
          card="c.webp", fit={"x": 1, "y": 1, "w": 1, "h": 1},
          window={"x": 1, "y": 1, "w": 1, "h": 1})


def test_팔레트는_큰_군집부터다():
    dog = fake_dog((300, 200), color=(210, 170, 130))
    shares = [share for _, share in coat_colors(dog).palette]
    assert shares == sorted(shares, reverse=True)
    assert sum(shares) == pytest.approx(1.0, abs=0.01)


def test_누끼가_비면_알려_준다():
    with pytest.raises(ValueError):
        coat_colors(Image.new("RGBA", (10, 10), (0, 0, 0, 0)))


def test_hex_변환은_범위를_넘지_않는다():
    assert to_hex((300, -5, 12.6)) == "#FF000D"
    assert to_hex(lighten((0, 0, 0), 1.0)) == "#FFFFFF"
