from datetime import date

import pytest
from PIL import Image

from src import pipeline
from src.compose import CardArtMissing, template_for
from src.contract import SceneDoc
from src.crops import crop_for_month
from src.pixelize import STYLES
from tests.fakes import fake_dog, fake_frame


@pytest.fixture()
def cards_dir(tmp_path):
    """카드 원화는 git 밖이라 테스트 안에서 만든다."""
    cards = tmp_path / "cards"
    cards.mkdir()
    for crop in crop_for_month.__globals__["load_table"]().values():
        template = template_for(crop.id)
        fake_frame(template.window).save(cards / template.frame)
    return cards


def test_사진_이름_생일_셋이면_결과가_나온다(tmp_path, cards_dir):
    result = pipeline.run(
        fake_dog((640, 480), color=(238, 190, 147)), "네옹", date(2023, 5, 14),
        out_dir=tmp_path / "out", cards_dir=cards_dir, style=STYLES["pixel"],
    )
    assert result.card_path.exists() and result.subject_path.exists()
    assert result.scene_path.exists()
    assert result.bgm_path is None


def test_나온_JSON_이_계약을_만족한다(tmp_path, cards_dir):
    result = pipeline.run(
        fake_dog(color=(238, 190, 147)), "네옹", date(2023, 5, 14),
        out_dir=tmp_path / "out", cards_dir=cards_dir,
    )
    doc = SceneDoc.load(result.scene_path)
    assert doc == result.doc
    assert doc.card.no == 5 and doc.card.id == "danhobak"
    assert doc.scene.accent == "#EEBE93"
    assert doc.scene.bgm is None                    # 음악 없이도 완결된다
    assert doc.scene.card == "card.webp"
    assert doc.agent_version


def test_생일이_카드를_고른다(tmp_path, cards_dir):
    for month, want in ((5, "danhobak"), (12, "cabbage"), (1, "pepper")):
        result = pipeline.run(
            fake_dog(), "네옹", date(2023, month, 3),
            out_dir=tmp_path / f"out{month}", cards_dir=cards_dir,
        )
        assert result.crop.id == want


def test_카드를_직접_고를_수도_있다(tmp_path, cards_dir):
    result = pipeline.run(
        fake_dog(), "네옹", date(2023, 5, 14), card_id="cucumber",
        out_dir=tmp_path / "out", cards_dir=cards_dir,
    )
    assert result.crop.id == "cucumber" and result.doc.card.no == 8


def test_fit_이_그림창_안에_들어간다(tmp_path, cards_dir):
    result = pipeline.run(
        fake_dog((500, 700)), "코코", date(2023, 7, 1),
        out_dir=tmp_path / "out", cards_dir=cards_dir,
    )
    fit, window = result.doc.scene.fit, result.doc.scene.window
    assert window.x <= fit.x and window.y <= fit.y
    assert fit.x + fit.w <= window.x + window.w + 0.01
    assert fit.y + fit.h <= window.y + window.h + 0.01


def test_같은_입력이면_같은_결과가_나온다(tmp_path, cards_dir):
    kw = dict(cards_dir=cards_dir, style=STYLES["pixel"])
    a = pipeline.run(fake_dog(), "네옹", date(2023, 5, 14), out_dir=tmp_path / "a", **kw)
    b = pipeline.run(fake_dog(), "네옹", date(2023, 5, 14), out_dir=tmp_path / "b", **kw)
    assert a.doc == b.doc
    assert a.card_path.read_bytes() == b.card_path.read_bytes()


def test_원화가_없으면_곱게_실패한다(tmp_path):
    with pytest.raises(CardArtMissing):
        pipeline.run(fake_dog(), "네옹", date(2023, 5, 14),
                     out_dir=tmp_path / "out", cards_dir=tmp_path / "없는폴더")


def test_모르는_카드_id_는_알려_주고_죽는다(tmp_path, cards_dir):
    with pytest.raises(ValueError, match="모르는 카드"):
        pipeline.run(fake_dog(), "네옹", date(2023, 5, 14), card_id="없는것",
                     out_dir=tmp_path / "out", cards_dir=cards_dir)


def test_CLI_로도_돈다(tmp_path, cards_dir, capsys):
    photo = tmp_path / "dog.png"
    fake_dog().save(photo)
    code = pipeline.main([
        "--photo", str(photo), "--name", "네옹", "--birthday", "2023-05-14",
        "--out", str(tmp_path / "out"), "--assets", str(cards_dir),
    ])
    assert code == 0
    assert "Danhobak" in capsys.readouterr().out
    assert (tmp_path / "out" / "scene.json").exists()


def test_사진에_알파가_없어도_넣을_수_있다(tmp_path, cards_dir, monkeypatch):
    """rembg 가 있는 환경에서만 도는 길이라, 부르는지까지만 본다."""
    photo = Image.new("RGB", (200, 200), (180, 140, 100))
    calls = {}

    from src.cutout import cutout as real_cutout

    def fake_cutout(img, **kw):
        calls["called"] = True
        return real_cutout(fake_dog())

    monkeypatch.setattr(pipeline, "cutout", fake_cutout)
    pipeline.run(photo, "네옹", date(2023, 5, 14),
                 out_dir=tmp_path / "out", cards_dir=cards_dir)
    assert calls["called"]


def test_mock_음악까지_붙으면_bgm_이_난다(tmp_path, cards_dir):
    from src.music.mock import MockMusic

    result = pipeline.run(
        fake_dog(), "네옹", date(2023, 5, 14), out_dir=tmp_path / "out",
        cards_dir=cards_dir, music=MockMusic(), music_seconds=4,
    )
    assert result.bgm_path is not None and result.bgm_path.exists()
    assert result.doc.scene.bgm == "bgm.ogg"


def test_음악이_실패해도_카드는_나온다(tmp_path, cards_dir):
    """⑦만 네트워크·유료다. 여기서 죽으면 설계가 틀린 것이다."""
    from src.music.mock import FailingMusic

    result = pipeline.run(
        fake_dog(), "네옹", date(2023, 5, 14), out_dir=tmp_path / "out",
        cards_dir=cards_dir, music=FailingMusic(),
    )
    assert result.bgm_path is None
    assert result.doc.scene.bgm is None
    assert result.card_path.exists() and result.scene_path.exists()


def test_CLI_의_mock_음악_옵션이_돈다(tmp_path, cards_dir):
    photo = tmp_path / "dog.png"
    fake_dog().save(photo)
    assert pipeline.main([
        "--photo", str(photo), "--name", "코코", "--birthday", "2022-09-02",
        "--out", str(tmp_path / "out"), "--assets", str(cards_dir),
        "--music", "mock", "--seconds", "4",
    ]) == 0
    assert (tmp_path / "out" / "bgm.ogg").exists()


def test_음악이_무슨_예외로_죽어도_카드는_나온다(tmp_path, cards_dir):
    """provider 는 아직 없는 물건이라 무엇으로 실패할지 모른다 — 타임아웃도 카드를 못 막는다."""

    class 터지는Provider:
        def generate(self, prompt, seconds):
            raise TimeoutError("실제 서비스가 낼 법한 예외")

    result = pipeline.run(
        fake_dog(), "네옹", date(2023, 5, 14), out_dir=tmp_path / "out",
        cards_dir=cards_dir, music=터지는Provider(),
    )
    assert result.bgm_path is None
    assert result.scene_path.exists() and result.doc.scene.bgm is None
