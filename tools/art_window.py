"""카드 프레임에서 그림창 좌표를 뽑아 templates/windows.toml 에 넣을 줄로 찍는다.

"카드당 좌표 4개, 12장이면 48개 숫자를 사람이 한 번 잰다" 고 계획에 적었는데, 원화가
**그림 영역만 투명하게 지운 완성 카드**라 잴 필요가 없었다. 알파의 구멍이 곧 그
좌표다.

    uv run python tools/art_window.py templates/cards/*.webp
"""

import argparse
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.compose import extract_art_window  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="프레임 알파에서 그림창 좌표 뽑기")
    ap.add_argument("frames", nargs="+", type=Path)
    ap.add_argument("--threshold", type=int, default=200, help="이 아래 알파를 구멍으로 본다")
    args = ap.parse_args(argv)

    for path in args.frames:
        frame = Image.open(path)
        rect = extract_art_window(frame, threshold=args.threshold)
        card_id = path.stem.replace("-card-frame", "")
        print(f"[{card_id}]")
        print(f'frame = "{path.name}"   # {frame.size[0]}x{frame.size[1]}')
        print(f"art = {{ x = {rect.x}, y = {rect.y}, w = {rect.w}, h = {rect.h} }}")
        print("measured = true")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
