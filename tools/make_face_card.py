"""사진 한 장 → 강아지 **얼굴만** 따서 야채 카드 구멍에 끼운 PNG.

    uv run python tools/make_face_card.py --from-label --n 12
    uv run python tools/make_face_card.py --from-label --dog n02090622_2518

`src/pipeline.py` 가 아니다. 저쪽은 강아지 **전체**를 직사각 그림창에 넣는 길이고
(`*-card-frame.webp`), 이쪽은 **얼굴만** 타원 구멍에 넣는 길이다
(`*-card-slots.webp`). 두 원화가 성질이 달라서 길도 둘이다 — `src/holes.py` 설명 참고.

## 키포인트를 어디서 받나

    --from-label   dog-pose val 이미지 + **정답 라벨**   ← 지금은 이것
    --model x.pt   학습된 YOLO pose 로 추론              ← Colab 학습 뒤

이렇게 갈라 두면 **모델 없이도 합성 결과를 눈으로 볼 수 있고**, 나중에 모델이 붙을
때 파이프라인 나머지는 한 줄도 안 바뀐다. 정답 라벨은 "모델이 완벽했을 때"라서
여기서 나온 카드가 이 접근의 **상한**이다.

## 카드는 랜덤이다

CA-017. `--seed` 는 어느 강아지를 볼지와 어느 카드가 나올지를 같이 고정한다 — 도구가
같은 그림을 다시 내야 눈으로 비교가 되기 때문이고, 서비스 쪽(`src/pipeline.py`)은
씨를 안 준다. `--dog` 는 뽑기를 아예 비켜서 **강아지 하나를 12장 전부에** 넣는다 —
같은 얼굴이 카드마다 어떻게 달라 보이는지는 그렇게 놓고 봐야 보인다.

## 축소해서도 본다

원본 크기로만 보면 문제가 안 보인다. 앱에서 카드는 120·72·40dp 로도 그려지는데,
HISTORY.md 11절이 "눈이 3픽셀이 되면서 무너졌다" 고 적은 자리다. `--sheet` 가 세
크기를 나란히 낸다.
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import compose  # noqa: E402
from src import cutout  # noqa: E402
from src import dogpose as kp  # noqa: E402
from src import facebox as geo  # noqa: E402
from src.compose import CARDS_DIR  # noqa: E402
from src.crops import draw_crop  # noqa: E402
from src.dogpose import DATASET_ROOT, load_split  # noqa: E402
from src.holes import HoleCard, hole_card, load_holes  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "out" / "face-cards"

#: 앱에서 카드가 그려지는 크기(dp). 화면 밀도 3배로 잡아 픽셀로 바꾼다.
CHECK_DP = (120, 72, 40)
DP_TO_PX = 3


def face_of(photo: Image.Image, kps: np.ndarray) -> cutout.Face | None:
    """사진 + 키포인트 → 카드에 끼울 얼굴. 상자를 못 만들면 `None`.

    `None` 은 실패가 아니라 **물러설 자리**다 — 앱에서는 `GuideFrameScreen` 이
    받아서 사람이 직접 맞춘다.
    """
    iw, ih = photo.size
    box = geo.face_box(kps, iw, ih)
    if box is None:
        return None
    neck = geo.neck_t(kps, box, ih)

    left, top, right, bottom = box.to_pixels(iw, ih)
    crop = photo.crop((left, top, right, bottom))

    # 구멍에 맞출 기준점을 **키포인트에서** 준다. 크롭 안 픽셀로 옮겨서 넘긴다.
    ax = geo.face_anchor_x(kps, iw)
    chin = kps[kp.CHIN]
    return cutout.face_from(
        crop, neck,
        anchor_x=None if ax is None else ax - left,
        chin_y=float(chin[1]) * ih - top if chin[2] > 0 else None,
    )


def size_sheet(card: Image.Image, path: Path) -> None:
    """같은 카드를 120·72·40dp 로 나란히. **이 크기에서 얼굴이 읽혀야 한다.**"""
    shots = []
    for dp in CHECK_DP:
        h = dp * DP_TO_PX
        shots.append(card.resize((round(h * card.width / card.height), h), Image.Resampling.LANCZOS))
    pad = 16
    sheet = Image.new(
        "RGBA",
        (sum(s.width for s in shots) + pad * (len(shots) + 1), shots[0].height + pad * 2),
        (24, 22, 20, 255),
    )
    x = pad
    for s in shots:
        sheet.alpha_composite(s, (x, pad))
        x += s.width + pad
    sheet.convert("RGB").save(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-label", action="store_true",
                    help="dog-pose val 의 정답 라벨로 돌린다")
    ap.add_argument("--model", type=Path, help="학습된 YOLO pose 가중치 (아직 없음)")
    ap.add_argument("--split", default="val")
    ap.add_argument("--root", type=Path, default=DATASET_ROOT)
    ap.add_argument("--assets", type=Path, default=CARDS_DIR,
                    help="카드 원화 자리. git 밖이다")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dog", help="이 강아지 하나로 **카드 전부**를 채운다 (val 의 stem)")
    ap.add_argument("--sheet", action="store_true", help="축소 크기 비교판도 낸다")
    args = ap.parse_args()

    if args.model:
        print("아직 학습된 가중치가 없습니다. tools/train_dog_pose.py 를 Colab 에서\n"
              "돌린 뒤 이 길이 열립니다. 지금은 --from-label 로 확인하세요.", file=sys.stderr)
        return 2
    if not args.from_label:
        ap.error("--from-label 또는 --model 중 하나가 필요합니다")

    samples = [s for s in load_split(args.split, args.root) if s.sane()]
    if not samples:
        print(f"{args.split} 가 비었습니다. tools/fetch_dog_pose.py 를 먼저 돌리세요.",
              file=sys.stderr)
        return 2

    if args.dog:
        pick = next((s for s in samples if s.stem == args.dog), None)
        if pick is None:
            print(f"{args.dog} 를 {args.split} 에서 못 찾았습니다.", file=sys.stderr)
            return 2
        # 강아지 하나를 카드 전부에. 뽑기 화면이 실제로 그렇게 생겼다 — 얼굴은 한
        # 장이고 카드만 바뀐다 (앱 PR #19: "누끼는 카드가 아니라 강아지에 붙는다").
        jobs = [(pick, c) for c in load_holes().values()]
    else:
        rng = random.Random(args.seed)
        chosen = rng.sample(samples, min(args.n, len(samples)))
        # 카드는 **랜덤이다** (CA-017). --seed 는 어느 강아지를 볼지와 어느 카드가
        # 나올지를 같이 고정한다 — 도구가 같은 그림을 다시 낼 수 있어야 눈으로
        # 비교가 된다. 서비스 쪽은 씨를 안 준다.
        jobs = [(s, hole_card(draw_crop(rng).id)) for s in chosen]

    OUT.mkdir(parents=True, exist_ok=True)
    made = skipped = ellipsed = missing = 0
    faces: dict[str, cutout.Face | None] = {}

    for s, card in jobs:
        if s.stem not in faces:
            # **누끼는 한 번만 뜬다.** 카드를 몇 장 뽑아도 얼굴은 한 장이다.
            faces[s.stem] = face_of(Image.open(s.image).convert("RGB"), s.kps)
        face = faces[s.stem]
        if face is None:
            skipped += 1
            print(f"  {s.stem}: 얼굴 키포인트가 모자라 물러섬 (앱에서는 수동 상자)")
            continue
        if not face.cut:
            ellipsed += 1

        art_path = card.art_path(args.assets)
        if not art_path.exists():
            missing += 1
            print(f"  {card.id}: 원화가 없다 ({art_path}) — 원화는 git 밖에 둔다")
            continue

        got = compose.compose_face_hole(Image.open(art_path), face, card)
        path = OUT / f"{s.stem}__{card.id}.png"
        got.card.convert("RGB").save(path)
        if args.sheet:
            size_sheet(got.card, OUT / f"{s.stem}__{card.id}__sizes.png")
        made += 1
        print(f"  {s.stem} → {card.id}"
              f"{'' if face.cut else '  (누끼 실패 → 타원으로 물러섬)'}")

    print()
    print(f"  {made}장 만듦 · {skipped}장 물러섬 · 원화 없음 {missing}장 · "
          f"타원 폴백 {ellipsed}장")
    print(f"  → {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
