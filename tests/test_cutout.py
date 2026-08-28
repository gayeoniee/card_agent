import pytest
from PIL import Image

from src import cutout as cutout_mod
from src.cutout import Cutout, CutoutUnavailable, cutout, has_alpha
from tests.fakes import fake_dog


def test_알파가_이미_있으면_rembg_를_부르지_않는다(monkeypatch):
    """이 길이 있어야 누끼 설비 없이도 나머지 파이프라인이 돈다."""
    monkeypatch.setattr(cutout_mod, "_session", lambda model: pytest.fail("불리면 안 된다"))
    result = cutout(fake_dog((300, 200)))
    assert result.used_rembg is False
    assert result.image.size == (251, 151)      # 여백이 잘렸다


def test_원본에서의_자리와_비율을_알려_준다():
    result = cutout(fake_dog((400, 200)))
    left, top, right, bottom = result.bbox
    assert result.source_size == (400, 200)
    assert 0 < result.coverage < 1
    assert (right - left) <= 400 and (bottom - top) <= 200


def test_알파가_없는_사진은_누끼로_넘어간다(monkeypatch):
    photo = Image.new("RGB", (64, 64), (120, 90, 60))
    called = {}

    def fake_session(model):
        called["model"] = model
        return object()

    monkeypatch.setattr(cutout_mod, "_session", fake_session)
    monkeypatch.setattr(cutout_mod, "_clean", lambda img: img)

    class FakeRembg:
        @staticmethod
        def remove(img, session=None):
            out = img.convert("RGBA")
            out.putalpha(Image.new("L", out.size, 255))
            return out

    monkeypatch.setitem(__import__("sys").modules, "rembg", FakeRembg)
    result = cutout(photo)
    assert result.used_rembg is True
    assert called["model"] == "u2net"


def test_설비가_없으면_곱게_실패한다(monkeypatch):
    def missing(model):
        raise CutoutUnavailable("rembg 가 없다")

    monkeypatch.setattr(cutout_mod, "_session", missing)
    with pytest.raises(CutoutUnavailable):
        cutout(Image.new("RGB", (32, 32), (10, 20, 30)))


def test_결과가_통째로_비면_알려_준다(monkeypatch):
    empty = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    with pytest.raises(CutoutUnavailable, match="비었다"):
        cutout(empty)


def test_알파_먼지는_없는_것으로_센다():
    dog = fake_dog((200, 150))
    dog.putpixel((0, 0), (9, 9, 9, 3))
    assert cutout(dog).bbox[0] > 0


def test_전부_불투명한_RGBA_는_알파가_없는_것이다():
    assert has_alpha(Image.new("RGBA", (8, 8), (1, 2, 3, 255))) is False
    assert has_alpha(fake_dog((40, 40))) is True


def test_coverage_는_넓이_비율이다():
    c = Cutout(image=Image.new("RGBA", (1, 1)), bbox=(0, 0, 50, 40),
               source_size=(100, 80), used_rembg=False)
    assert c.coverage == pytest.approx(0.25)
