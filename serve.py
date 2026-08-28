"""card_agent 서비스 — 사진·이름·생일을 받아 카드 한 장을 만든다.

⚠ 이 파일에 `from __future__ import annotations` 를 넣지 말 것. `UploadFile` 이
문자열 어노테이션이 되어 pydantic 이 이름을 못 찾고 500 으로 죽는다.
`skin-screening/CLAUDE.md` 가 "실제로 당했다" 고 적어 둔 함정이고, 같은 모양의
서버라 그대로 있다.

**동기로 기다리지 않는다.** 접수하고 job id 를 주고, 상태는 물어보게 한다. 생성이
느린데 기다리게 했다가는 nginx 60초 타임아웃에 걸려, 우리가 내지 않은 HTML 오류
페이지를 사용자가 받는다 (D-021 이 /ask 예열에서 실측한 함정).

    uv run python serve.py --mock      # 네트워크·비용 없이 전체 파이프라인
"""

import argparse
import os
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, ImageDraw, UnidentifiedImageError

from src import pipeline
from src.compose import CARDS_DIR, CardArtMissing, load_windows
from src.cutout import CutoutUnavailable
from src.pixelize import DEFAULT_STYLE, style_by_name
from src.version import agent_version

# 환경변수 이름은 .env.example 과 반드시 같이 고친다.
ENV_ASSETS = "CARD_AGENT_ASSETS"
ENV_OUT = "CARD_AGENT_OUT"
ENV_HOST = "CARD_AGENT_HOST"
ENV_PORT = "CARD_AGENT_PORT"
ENV_STYLE = "CARD_AGENT_STYLE"

MAX_PHOTO_BYTES = 12 * 1024 * 1024
SERVED_FILES = (pipeline.CARD_FILE, pipeline.SUBJECT_FILE, pipeline.SCENE_FILE, pipeline.BGM_FILE)


class Jobs:
    """접수한 일감. 프로세스가 죽으면 같이 사라진다 — 지금은 그걸로 충분하다."""

    def __init__(self):
        self._lock = threading.Lock()
        self._jobs: dict[str, dict] = {}

    def create(self) -> str:
        job_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._jobs[job_id] = {"id": job_id, "status": "queued"}
        return job_id

    def update(self, job_id: str, **fields) -> None:
        with self._lock:
            self._jobs[job_id].update(fields)

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None


def placeholder_cards(into: Path) -> Path:
    """mock 전용 임시 원화. 카드 원화는 git 밖이라 없을 수 있다.

    진짜 카드로 오해하지 않도록 대놓고 PLACEHOLDER 라고 적어 둔다.
    """
    into.mkdir(parents=True, exist_ok=True)
    for card_id, template in load_windows().items():
        path = into / template.frame
        if path.exists():
            continue
        size = (600, 840)
        frame = Image.new("RGBA", size, (26, 48, 30, 255))
        draw = ImageDraw.Draw(frame)
        w = template.window
        box = (round(w.x / 100 * size[0]), round(w.y / 100 * size[1]),
               round((w.x + w.w) / 100 * size[0]), round((w.y + w.h) / 100 * size[1]))
        frame.paste((0, 0, 0, 0), box)
        draw.text((16, size[1] - 28), f"PLACEHOLDER · {card_id}", fill=(210, 210, 210, 255))
        frame.save(path)
    return into


def create_app(*, cards_dir: Path, out_dir: Path, style_name: str = DEFAULT_STYLE,
               mock: bool = False) -> FastAPI:
    app = FastAPI(title="card_agent", version=agent_version())
    jobs = Jobs()
    # 파이프라인은 CPU 를 쓴다. 이벤트 루프에서 돌리면 상태 조회까지 같이 멈춘다.
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="card")
    style = style_by_name(style_name)

    music_kind = "mock" if mock else "none"
    if mock:
        cards_dir = placeholder_cards(out_dir / "_placeholder-cards") if not cards_dir.exists() else cards_dir

    def work(job_id: str, photo_path: Path, name: str, birthday: date):
        jobs.update(job_id, status="running")
        try:
            music = None
            if mock:
                from src.music.mock import MockMusic

                music = MockMusic()
            result = pipeline.run(
                Image.open(photo_path), name, birthday,
                out_dir=out_dir / job_id, cards_dir=cards_dir, style=style, music=music,
            )
            jobs.update(
                job_id,
                status="done",
                scene=result.doc.model_dump(mode="json", by_alias=True),
                files=[f for f in SERVED_FILES if (out_dir / job_id / f).exists()],
            )
        except (CutoutUnavailable, CardArtMissing) as exc:
            # 설비가 없는 것이지 요청이 틀린 게 아니다.
            jobs.update(job_id, status="failed", reason=str(exc), kind="unavailable")
        except Exception as exc:
            traceback.print_exc()
            jobs.update(job_id, status="failed", reason=str(exc), kind="error")

    @app.get("/healthz")
    def healthz():
        from src.cutout import available

        return {
            "ok": True,
            "agent_version": agent_version(),
            "cutout": available(),
            "music": music_kind,
            "cards_dir": str(cards_dir),
            "cards_ready": cards_dir.exists(),
        }

    @app.post("/cards", status_code=202)
    async def create_card(photo: UploadFile, name: str = Form(...), birthday: str = Form(...)):
        """접수만 하고 바로 돌려준다. 기다리게 하지 않는다 (D-021)."""
        try:
            when = date.fromisoformat(birthday)
        except ValueError:
            raise HTTPException(400, "birthday 는 YYYY-MM-DD 여야 한다") from None
        if not name.strip():
            raise HTTPException(400, "name 이 비었다")

        blob = await photo.read()
        if not blob:
            raise HTTPException(400, "photo 가 비었다")
        if len(blob) > MAX_PHOTO_BYTES:
            raise HTTPException(413, f"사진이 너무 크다 ({len(blob)} bytes)")

        job_id = jobs.create()
        job_dir = out_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        photo_path = job_dir / "photo.bin"
        photo_path.write_bytes(blob)
        try:
            Image.open(photo_path).verify()
        except (UnidentifiedImageError, OSError):
            raise HTTPException(400, "사진을 읽지 못했다") from None

        pool.submit(work, job_id, photo_path, name.strip(), when)
        return {"id": job_id, "status": "queued", "status_url": f"/cards/{job_id}"}

    @app.get("/cards/{job_id}")
    def read_card(job_id: str):
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "그런 일감이 없다")
        if job["status"] == "failed" and job.get("kind") == "unavailable":
            # 설비가 없는 것이라 503 성격이다. 다시 보내도 지금은 안 된다.
            return JSONResponse(job, status_code=503)
        return job

    @app.get("/cards/{job_id}/files/{filename}")
    def read_file(job_id: str, filename: str):
        if filename not in SERVED_FILES:
            raise HTTPException(404, "그런 파일은 안 준다")
        path = out_dir / job_id / filename
        if not path.exists():
            raise HTTPException(404, "아직 없다")
        return FileResponse(path)

    return app


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="card_agent 서비스")
    ap.add_argument("--host", default=os.environ.get(ENV_HOST, "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get(ENV_PORT, "8099")))
    ap.add_argument("--assets", type=Path,
                    default=Path(os.environ.get(ENV_ASSETS, str(CARDS_DIR))))
    ap.add_argument("--out", type=Path, default=Path(os.environ.get(ENV_OUT, "out")))
    ap.add_argument("--style", default=os.environ.get(ENV_STYLE, DEFAULT_STYLE))
    ap.add_argument("--mock", action="store_true",
                    help="음악은 mock, 원화가 없으면 자리표 카드를 만들어 쓴다")
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    app = create_app(cards_dir=args.assets, out_dir=args.out,
                     style_name=args.style, mock=args.mock)
    # 바깥에 직접 열지 않는다. compose 로 들여올 때도 profiles 뒤에 꺼둔 채 시작한다 (D-024).
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
