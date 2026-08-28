"""파이프라인 — 사진·이름·생일 셋을 넣으면 결과 한 벌이 나온다.

    photo.jpg · "네옹" · 2023-05-14
      → scene.json · card.webp · subject.webp (· bgm.ogg)

①~⑥ 은 오프라인·0원·즉시고 ⑦(음악)만 네트워크·유료·느리다. 그래서 음악은 없어도
되는 단계로 두었다 — 실패하면 `bgm` 이 null 이고, 앱의 SceneMusic 이 조용히
아무것도 안 한다.
"""

import argparse
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from PIL import Image

from src.coat import coat_colors
from src.compose import CARDS_DIR, compose_card, open_frame, save_webp, template_for
from src.contract import SceneDoc
from src.crops import Crop, crop_for_birthday, crop_for_month
from src.cutout import cutout
from src.pixelize import DEFAULT_STYLE, PixelStyle, style_by_name
from src.version import agent_version

CARD_FILE = "card.webp"
SUBJECT_FILE = "subject.webp"
SCENE_FILE = "scene.json"
BGM_FILE = "bgm.ogg"

# 카드 모서리를 깎는 정도(짧은 변에 대한 %). 원화가 이미 둥글면 0 이어도 된다.
CORNER_RADIUS_PCT = 0.0


@dataclass(frozen=True)
class Result:
    doc: SceneDoc
    out_dir: Path
    card_path: Path
    subject_path: Path
    scene_path: Path
    bgm_path: Path | None
    crop: Crop


def run(
    photo: Image.Image,
    name: str,
    birthday: date,
    *,
    out_dir: Path,
    cards_dir: Path = CARDS_DIR,
    style: PixelStyle | None = None,
    card_id: str | None = None,
    music=None,
    music_seconds: int = 30,
    dog_key: str | None = None,
) -> Result:
    """한 장 뽑는다. 그림 단계에서 실패하면 그대로 올린다 — 카드 없이 나갈 이유가 없다."""
    from src.scene import build_scene       # scene 이 crops 를 부르므로 여기서

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ③ 작물 — 생일 월 하나로 정해진다
    crop = crop_for_month(_card_month(card_id)) if card_id else crop_for_birthday(birthday)

    # ① 누끼 → ② 털색
    cut = cutout(photo)
    coat = coat_colors(cut.image)

    # ④⑤ 화풍 맞춤 + 카드 합성 (fit 은 여기서 나온다)
    template = template_for(crop.id)
    frame = open_frame(template, cards_dir)
    composed = compose_card(
        frame, cut.image, template.window,
        style=style, corner_radius_pct=CORNER_RADIUS_PCT,
    )

    card_path = save_webp(composed.card, out_dir / CARD_FILE)
    subject_path = save_webp(composed.subject, out_dir / SUBJECT_FILE)

    # ⑦ 음악 — 유일한 생성·유일한 유료. 없어도 결과는 완결된다
    bgm_path = None
    if music is not None:
        bgm_path = _make_music(music, out_dir, crop, name, music_seconds)

    # ⑥ 장면 값
    scene = build_scene(
        name=name,
        birthday=birthday,
        crop=crop,
        accent=coat.accent,
        accent2=coat.accent2,
        fit=composed.fit,
        window=composed.window,
        card=CARD_FILE,
        subject=SUBJECT_FILE,
        frame=template.frame,
        bgm=BGM_FILE if bgm_path else None,
        dog_key=dog_key,
    )

    doc = SceneDoc(
        agent_version=agent_version(),
        dog={"name": name, "birthday": birthday},
        card=crop.to_card(art=CARD_FILE),
        scene=scene,
    )
    scene_path = doc.write(out_dir / SCENE_FILE)

    return Result(
        doc=doc, out_dir=out_dir, card_path=card_path, subject_path=subject_path,
        scene_path=scene_path, bgm_path=bgm_path, crop=crop,
    )


def _card_month(card_id: str) -> int:
    from src.crops import load_table

    for month, crop in load_table().items():
        if crop.id == card_id:
            return month
    raise ValueError(f"모르는 카드 id 다: {card_id}")


def _make_music(music, out_dir: Path, crop: Crop, name: str, seconds: int) -> Path | None:
    """음악만은 실패해도 카드를 세운다. 여기서 삼키는 유일한 예외다.

    예외 종류를 좁히지 않는다. provider 는 아직 없는 물건이라 실제 서비스가 붙으면
    타임아웃·연결 끊김·라이브러리 고유 예외가 무엇으로든 올라온다. 그중 하나라도
    빠져나가면 card.webp·subject.webp 는 이미 쓰인 뒤 scene.json 만 없는 상태로
    죽는다 — CA-009 가 막으려던 바로 그 경우다.
    """
    from src.loop import write_loop_ogg
    from src.music.base import prompt_for

    try:
        wav = music.generate(prompt_for(crop.korean, name), seconds)
        return write_loop_ogg(wav, out_dir / BGM_FILE)
    except Exception as exc:
        print(f"음악 없이 간다: {type(exc).__name__}: {exc}", file=sys.stderr)
        return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="사진·이름·생일 → 카드 한 장과 장면")
    ap.add_argument("--photo", required=True, type=Path)
    ap.add_argument("--name", required=True)
    ap.add_argument("--birthday", required=True, help="YYYY-MM-DD")
    ap.add_argument("--out", type=Path, default=Path("out"))
    ap.add_argument("--assets", type=Path, default=CARDS_DIR, help="카드 원화 폴더")
    ap.add_argument("--style", default=DEFAULT_STYLE, help="화풍 이름 (src/pixelize.py)")
    ap.add_argument("--card", default=None, help="작물을 직접 고른다 (생일 대신)")
    ap.add_argument("--music", choices=("none", "mock"), default="none",
                    help="mock 은 네트워크·비용 없이 파이프라인 전체를 돌린다")
    ap.add_argument("--seconds", type=int, default=30)
    args = ap.parse_args(argv)

    music = None
    if args.music == "mock":
        from src.music.mock import MockMusic

        music = MockMusic()

    result = run(
        Image.open(args.photo),
        args.name,
        date.fromisoformat(args.birthday),
        out_dir=args.out,
        cards_dir=args.assets,
        style=style_by_name(args.style),
        card_id=args.card,
        music=music,
        music_seconds=args.seconds,
    )
    print(f"{result.scene_path} — No.{result.crop.no} {result.crop.name} "
          f"({result.crop.korean}), bgm={'있음' if result.bgm_path else '없음'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
