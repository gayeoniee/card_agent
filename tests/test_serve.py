import io
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import serve
from src.compose import template_for
from src.crops import load_table
from tests.fakes import fake_dog, fake_frame


def png_bytes(size=(320, 240)) -> bytes:
    buf = io.BytesIO()
    fake_dog(size).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def cards_dir(tmp_path) -> Path:
    cards = tmp_path / "cards"
    cards.mkdir()
    for crop in load_table().values():
        template = template_for(crop.id)
        fake_frame(template.window).save(cards / template.frame)
    return cards


@pytest.fixture()
def client(tmp_path, cards_dir):
    app = serve.create_app(cards_dir=cards_dir, out_dir=tmp_path / "out", mock=True)
    with TestClient(app) as c:
        yield c


def wait(client, job_id, want="done", timeout=60.0):
    """상태를 물어본다. 접수 응답을 기다리게 하지 않는 설계라 여기서 돈다."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        res = client.get(f"/cards/{job_id}")
        body = res.json()
        if body["status"] in ("done", "failed"):
            assert body["status"] == want, body
            return body
        time.sleep(0.05)
    raise AssertionError("일감이 안 끝났다")


def test_접수는_기다리지_않고_202_로_돌아온다(client):
    """기다리게 했다가 nginx 60초 타임아웃에 걸린 것이 D-021 이다."""
    res = client.post("/cards", files={"photo": ("dog.png", png_bytes(), "image/png")},
                      data={"name": "네옹", "birthday": "2023-05-14"})
    assert res.status_code == 202
    body = res.json()
    assert body["status"] == "queued"
    assert body["status_url"] == f"/cards/{body['id']}"


def test_상태를_물어보면_결과가_나온다(client):
    res = client.post("/cards", files={"photo": ("dog.png", png_bytes(), "image/png")},
                      data={"name": "네옹", "birthday": "2023-05-14"})
    job = wait(client, res.json()["id"])
    # 카드는 랜덤이라(CA-017) 어느 것인지가 아니라 **표 안의 것인지**를 본다.
    assert job["scene"]["card"]["id"] in {c.id for c in load_table().values()}
    assert job["scene"]["schema"] == 1
    assert "card.webp" in job["files"]


def test_만들어진_파일을_내려받는다(client):
    res = client.post("/cards", files={"photo": ("dog.png", png_bytes(), "image/png")},
                      data={"name": "네옹", "birthday": "2023-05-14"})
    job_id = res.json()["id"]
    wait(client, job_id)
    card = client.get(f"/cards/{job_id}/files/card.webp")
    assert card.status_code == 200 and card.content[:4] == b"RIFF"
    assert client.get(f"/cards/{job_id}/files/scene.json").json()["schema"] == 1


def test_아무_파일이나_내주지_않는다(client):
    res = client.post("/cards", files={"photo": ("dog.png", png_bytes(), "image/png")},
                      data={"name": "네옹", "birthday": "2023-05-14"})
    job_id = res.json()["id"]
    wait(client, job_id)
    assert client.get(f"/cards/{job_id}/files/photo.bin").status_code == 404
    assert client.get(f"/cards/{job_id}/files/scene.json").status_code == 200


def test_생일이_이상하면_받지_않는다(client):
    res = client.post("/cards", files={"photo": ("dog.png", png_bytes(), "image/png")},
                      data={"name": "네옹", "birthday": "2023년 5월"})
    assert res.status_code == 400


def test_이름이_비면_받지_않는다(client):
    res = client.post("/cards", files={"photo": ("dog.png", png_bytes(), "image/png")},
                      data={"name": "   ", "birthday": "2023-05-14"})
    assert res.status_code == 400


def test_사진이_아니면_받지_않는다(client):
    쓰레기 = "이건 사진이 아니다".encode("utf-8")
    res = client.post("/cards", files={"photo": ("dog.png", 쓰레기, "image/png")},
                      data={"name": "네옹", "birthday": "2023-05-14"})
    assert res.status_code == 400


def test_없는_일감은_404(client):
    assert client.get("/cards/없는것").status_code == 404


def test_설비가_없으면_503_성격으로_알린다(tmp_path):
    """원화가 없는 것은 요청이 틀린 게 아니다."""
    app = serve.create_app(cards_dir=tmp_path / "없는폴더", out_dir=tmp_path / "out")
    with TestClient(app) as client:
        res = client.post("/cards", files={"photo": ("dog.png", png_bytes(), "image/png")},
                          data={"name": "네옹", "birthday": "2023-05-14"})
        job_id = res.json()["id"]
        deadline = time.time() + 30
        while time.time() < deadline:
            got = client.get(f"/cards/{job_id}")
            if got.status_code == 503:
                assert got.json()["kind"] == "unavailable"
                return
            time.sleep(0.05)
        raise AssertionError("503 이 안 나왔다")


def test_healthz_가_설비_상태를_알려_준다(client):
    body = client.get("/healthz").json()
    assert body["ok"] is True
    assert body["music"] == "mock"
    assert body["cards_ready"] is True
    assert body["agent_version"]


def test_mock_은_원화가_없어도_자리표로_돈다(tmp_path):
    """`serve.py --mock` 이 설비 없이 전체 파이프라인을 도는 조건."""
    app = serve.create_app(cards_dir=tmp_path / "없는폴더", out_dir=tmp_path / "out", mock=True)
    with TestClient(app) as client:
        res = client.post("/cards", files={"photo": ("dog.png", png_bytes(), "image/png")},
                          data={"name": "코코", "birthday": "2022-09-02"})
        job_id = res.json()["id"]
        deadline = time.time() + 60
        while time.time() < deadline:
            body = client.get(f"/cards/{job_id}").json()
            if body["status"] in ("done", "failed"):
                assert body["status"] == "done", body
                assert body["scene"]["scene"]["bgm"] == "bgm.ogg"
                return
            time.sleep(0.05)
        raise AssertionError("일감이 안 끝났다")


def test_어노테이션을_문자열로_만들지_않는다():
    """serve.py 에 from __future__ import annotations 를 넣으면 UploadFile 이
    문자열 어노테이션이 되어 pydantic 이 이름을 못 찾고 500 으로 죽는다.

    설명에 적어 두는 것으로는 안 지켜지길래(실제로 당한 함정이다) 테스트로 박는다.
    """
    import ast

    tree = ast.parse(Path(serve.__file__).read_text(encoding="utf-8"))
    futures = [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "__future__"
        for alias in node.names
    ]
    assert "annotations" not in futures


def test_일감_id_로_바깥_파일을_꺼낼_수_없다(client, tmp_path):
    """받은 문자열을 그대로 경로로 이으면 out_dir 바깥이 나간다."""
    (tmp_path / "out").mkdir(parents=True, exist_ok=True)
    (tmp_path / "out" / "scene.json").write_text('{"몰래":"가져가면 안 되는 것"}', encoding="utf-8")

    for 나쁜_id in ("..", "%2e%2e", "../..", "a/../.."):
        res = client.get(f"/cards/{나쁜_id}/files/scene.json")
        assert res.status_code == 404, (나쁜_id, res.status_code, res.text)


def test_모르는_일감의_파일은_안_준다(client):
    assert client.get("/cards/0123456789ab/files/card.webp").status_code == 404


def test_이름이_계약보다_길면_접수에서_거절한다(client, tmp_path):
    """여기서 안 거르면 유료인 음악 단계까지 다 돌고 마지막에야 죽는다."""
    res = client.post("/cards", files={"photo": ("dog.png", png_bytes(), "image/png")},
                      data={"name": "네" * 21, "birthday": "2023-05-14"})
    assert res.status_code == 400
    assert not [p for p in (tmp_path / "out").iterdir() if p.name != "_placeholder-cards"]


def test_거절한_요청은_일감도_파일도_안_남긴다(client, tmp_path):
    쓰레기 = "이건 사진이 아니다".encode("utf-8")
    for _ in range(5):
        res = client.post("/cards", files={"photo": ("dog.png", 쓰레기, "image/png")},
                          data={"name": "네옹", "birthday": "2023-05-14"})
        assert res.status_code == 400
    남은_것 = [p for p in (tmp_path / "out").iterdir() if p.name != "_placeholder-cards"]
    assert 남은_것 == []


def test_원화가_한_장만_있어도_어느_카드가_뽑히든_돈다(tmp_path):
    """확인된 배추 원화만 있는 폴더에서 나머지 11장이 503 으로 죽으면 안 된다.

    카드는 랜덤이라(CA-017) **어느 것이 뽑힐지 고를 수 없다.** 그래서 여러 번 눌러
    여러 카드가 나오게 하고, 그 전부가 성공하는지를 본다. 자리표 카드를 12장 전부에
    만들어 두는 `placeholder_cards()` 가 이걸 받쳐 준다.
    """
    cards = tmp_path / "cards"
    cards.mkdir()
    배추 = template_for("cabbage")
    fake_frame(배추.window).save(cards / 배추.frame)

    표 = {c.id for c in load_table().values()}
    app = serve.create_app(cards_dir=cards, out_dir=tmp_path / "out", mock=True)
    나온_카드 = set()
    with TestClient(app) as client:
        for _ in range(8):
            res = client.post("/cards", files={"photo": ("dog.png", png_bytes(), "image/png")},
                              data={"name": "네옹", "birthday": "2023-05-14"})
            job = wait(client, res.json()["id"])
            assert job["status"] == "done", job
            나온_카드.add(job["scene"]["card"]["id"])
    assert 나온_카드 <= 표
    assert len(나온_카드) > 1, "여덟 번 눌렀는데 한 장만 나왔다 — 뽑기가 고정돼 있다"


def test_진짜_원화가_있으면_자리표_대신_그것을_쓴다(tmp_path):
    cards = tmp_path / "cards"
    cards.mkdir()
    배추 = template_for("cabbage")
    fake_frame(배추.window).save(cards / 배추.frame)

    serve.create_app(cards_dir=cards, out_dir=tmp_path / "out", mock=True)
    복사본 = tmp_path / "out" / "_placeholder-cards" / 배추.frame
    assert 복사본.read_bytes() == (cards / 배추.frame).read_bytes()
