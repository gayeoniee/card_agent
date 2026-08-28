"""비교 시트 — 화풍 후보를 나란히 뽑아 사람이 고른다.

그림 판단은 AI 에게 위임하지 않는다 (앱 협업규칙 1절). 사진과 픽셀아트 프레임이
한 화면에서 붙는지는 만들어서 봐야 판정되므로, 후보를 한 장에 늘어놓는 것까지만
한다.

원본 크기가 아니라 **화면 크기에서** 본다. HISTORY.md 11절이 "눈이 3픽셀이 되면서
무너졌다" 고 적은 자리다 — 그래서 칸 너비 기본값이 실제 카드가 화면에서 차지하는
크기다.

    uv run python tools/contact_sheet.py \
        --subject out/neong/subject.png --card danhobak \
        --assets templates/cards --out out/sheet.png
"""

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.compose import compose_card, open_frame, template_for  # noqa: E402
from src.pixelize import STYLES, variants  # noqa: E402

# 카드가 실기기 화면에서 차지하는 대략의 너비(px). 원본 크기로 보면 안 되는 이유는
# 모듈 설명 참고.
DEFAULT_CELL_WIDTH = 420
LABEL_HEIGHT = 34
PAD = 16


def build_sheet(frame: Image.Image, subject: Image.Image, window, styles,
                cell_width: int = DEFAULT_CELL_WIDTH, columns: int = 3) -> Image.Image:
    cells = []
    for style in styles:
        composed = compose_card(frame, subject, window, style=style)
        card = composed.card
        h = round(card.height * cell_width / card.width)
        cells.append((style.name, composed.fit, card.resize((cell_width, h), Image.Resampling.LANCZOS)))

    cell_h = max(c[2].height for c in cells) + LABEL_HEIGHT
    rows = (len(cells) + columns - 1) // columns
    sheet = Image.new(
        "RGBA",
        (columns * cell_width + (columns + 1) * PAD, rows * cell_h + (rows + 1) * PAD),
        (18, 18, 20, 255),
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=18)

    for i, (name, fit, card) in enumerate(cells):
        col, row = i % columns, i // columns
        x = PAD + col * (cell_width + PAD)
        y = PAD + row * (cell_h + PAD)
        sheet.alpha_composite(card, (x, y))
        draw.text(
            (x, y + card.height + 8),
            f"{name}   fit=({fit.x}, {fit.y}, {fit.w}, {fit.h})",
            fill=(235, 235, 235, 255),
            font=font,
        )
    return sheet


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="화풍 후보 비교 시트")
    ap.add_argument("--subject", required=True, type=Path, help="알파 있는 강아지 PNG")
    ap.add_argument("--card", required=True, help="카드 id (templates/crops.toml)")
    ap.add_argument("--assets", type=Path, default=None, help="카드 원화 폴더")
    ap.add_argument("--frame", type=Path, default=None, help="카드 원화를 직접 지정")
    ap.add_argument("--styles", default=",".join(STYLES), help="쉼표로 구분한 화풍 이름")
    ap.add_argument("--cell-width", type=int, default=DEFAULT_CELL_WIDTH)
    ap.add_argument("--columns", type=int, default=3)
    ap.add_argument("--out", type=Path, default=Path("out/contact-sheet.png"))
    args = ap.parse_args(argv)

    template = template_for(args.card)
    if args.frame is not None:
        frame = Image.open(args.frame).convert("RGBA")
    elif args.assets is not None:
        frame = open_frame(template, args.assets)
    else:
        frame = open_frame(template)

    subject = Image.open(args.subject).convert("RGBA")
    styles = variants([s.strip() for s in args.styles.split(",") if s.strip()])

    sheet = build_sheet(frame, subject, template.window, styles,
                        cell_width=args.cell_width, columns=args.columns)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.convert("RGB").save(args.out)
    print(f"{args.out} — {len(styles)}개 후보. 실기기 화면 크기로 보고 고른다")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
