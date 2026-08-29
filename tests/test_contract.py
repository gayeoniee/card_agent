from datetime import date

import pytest
from pydantic import ValidationError

from src.contract import Card, Dog, Rect, Scene, SceneDoc


def sample() -> SceneDoc:
    return SceneDoc(
        agent_version="0.1.0",
        dog=Dog(name="네옹", birthday=date(2023, 5, 14)),
        card=Card(
            no=5,
            id="danhobak",
            name="Danhobak Neo",
            stat_label="CRUNCH",
            stat=840,
            foil="Oilslick",
            art="card.webp",
        ),
        scene=Scene(
            place="네옹이의 이슬 맺힌 텃밭 · 해 뜨기 직전",
            seed=812734,
            motes=52,
            leaves=7,
            dew=15,
            accent="#EEBE93",
            accent2="#F5D9BE",
            back="back.webp",
            subject="subject.webp",
            card="card.webp",
            frame="frame.webp",
            fit=Rect(x=6.06, y=14.15, w=87.43, h=62.70),
            window=Rect(x=4.91, y=10.28, w=90.51, h=81.58),
            bgm="bgm.ogg",
        ),
    )


def test_왕복해도_값이_그대로다():
    doc = sample()
    again = SceneDoc.model_validate_json(doc.to_json())
    assert again == doc


def test_직렬화_이름이_앱_필드명과_같다():
    """앱이 statLabel·schema 로 읽는다. 파이썬 이름이 새어 나가면 안 된다."""
    payload = sample().model_dump(mode="json", by_alias=True)
    assert payload["schema"] == 1
    assert payload["card"]["statLabel"] == "CRUNCH"
    assert "stat_label" not in payload["card"]
    assert payload["scene"]["fit"] == {"x": 6.06, "y": 14.15, "w": 87.43, "h": 62.70}


def test_음악이_없어도_계약이_선다():
    """⑦이 실패해도 카드는 나온다 — bgm null 이 정상 값이다."""
    doc = sample()
    doc.scene.bgm = None
    assert SceneDoc.model_validate_json(doc.to_json()).scene.bgm is None


def test_산책_데이터_자리는_비워_둔다():
    scene = sample().scene
    assert scene.weather is None and scene.time_of_day is None


def test_색은_여섯자리_hex_만_받는다():
    with pytest.raises(ValidationError):
        Scene(**{**sample().scene.model_dump(), "accent": "#FFF"})


def test_모르는_필드는_거절한다():
    """앱과 이름이 어긋난 채 조용히 통과하는 것을 막는다."""
    with pytest.raises(ValidationError):
        Dog(name="네옹", birthday=date(2023, 5, 14), nickname="넹")


def test_파일로_쓰고_다시_읽는다(tmp_path):
    path = sample().write(tmp_path / "scene.json")
    assert SceneDoc.load(path) == sample()
    assert path.read_text(encoding="utf-8").endswith("\n")
    assert "네옹" in path.read_text(encoding="utf-8")  # ensure_ascii 로 뭉개지지 않는다
