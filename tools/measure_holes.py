"""자리를 비운 카드 12장에서 **구멍 좌표를 재서** `templates/holes.toml` 로 낸다.

    uv run tools/measure_windows.py            # 재서 toml 출력 + 확인용 그림
    uv run tools/measure_windows.py --write    # templates/holes.toml 에 쓴다

## 왜 재야 하나

`templates/holes.toml` 은 스스로 이렇게 적어 뒀다.

> ⚠ 확인된 값은 배추(No.01) 하나뿐이다. 나머지 11장은 같은 프레임 배치를 가정한
>   자리값이라, 카드 원화를 놓고 한 번씩 재서 고쳐야 한다.

그 원화가 이제 왔다. 그리고 **12장이 이미 뚫려 있어서** 사람이 자를 댈 필요가 없다 —
구멍이 곧 좌표다. 뚫린 자리를 찾아 읽기만 하면 된다.

## 뚫린 자리를 어떻게 찾나

받은 판은 JPG 라 알파가 없다. 대신 구멍이 색으로 남아 있다.

    큰 얼굴창   **검정**  (0,0,0)      가운데 언저리의 둥근 덩어리
    아바타 원   **흰색**  (255,255,255) 왼쪽 위 구석

검정만 보고 제일 큰 덩어리를 고르면 **틀린다.** 피망·당근·가지·단호박은 프레임
자체가 검은색이라 카드 테두리가 통째로 제일 큰 검정 덩어리가 된다. 그래서
**사진 가장자리에 닿는 덩어리는 버리고**, 둥근 정도로 한 번 더 거른다.

## 알파는 우리가 다시 만든다

앱은 구멍이 **알파로 뚫려** 있어야 뒤에 깐 얼굴이 비친다 (`CardSlots.kt`). 받은
JPG 는 구멍이 까맣게 칠해져 있을 뿐이라, 잰 원을 그대로 알파에서 빼서 WebP 로
다시 낸다 — `punch_card_slots.py` 가 하던 마지막 단계와 같다.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = pathlib.Path(__file__).resolve().parents[1]
CARDS = ROOT / "templates" / "cards"
OUT = ROOT / "out" / "holes"

#: 이 값보다 어두우면 "뚫린 검정". JPG 압축이 가장자리를 흐려서 여유를 둔다.
BLACK_MAX = 24
#: 이 값보다 밝으면 "뚫린 흰색"(아바타).
WHITE_MIN = 232

#: 얼굴창 반지름이 카드 **짧은 변** 대비 이 범위여야 한다.
FACE_R_MIN, FACE_R_MAX = 0.06, 0.32
#: 아바타 원은 훨씬 작다.
AVATAR_R_MIN, AVATAR_R_MAX = 0.02, 0.14


def biggest_circle(mask: np.ndarray, r_min: float, r_max: float) -> dict | None:
    """[mask] 안에 들어가는 **가장 큰 원**. 중심과 반지름(픽셀).

    ## 외접상자를 안 쓴다

    처음에는 덩어리의 외접상자를 구멍으로 봤는데, **가지에서 무너졌다.** 가지 몸통이
    거의 검정이라 구멍이 몸통과 한 덩어리로 붙었고, 상자가 755x670 이 되면서 둥근
    정도가 0.34 로 떨어져 필터에 걸렸다.

    거리 변환은 붙은 조각에 안 흔들린다. 화소마다 "가장 가까운 바깥까지의 거리" 를
    재는데, 그 최댓값이 곧 **덩어리 안에 들어가는 제일 큰 원의 반지름**이다. 가늘게
    붙은 몸통은 거리가 작아서 최댓값을 못 만든다.

    가장자리 띠(피망·당근의 검은 프레임)도 같은 이유로 자연히 빠진다 — 띠는 얇아서
    내접원이 작다.
    """
    m = mask.astype(np.uint8)
    # 사진 가장자리에 닿은 덩어리가 밖으로 이어진 것처럼 안 보이게 한 줄 두른다.
    # 안 두르면 프레임 띠의 거리가 밖으로 무한정 자라는 것처럼 계산된다.
    m = cv2.copyMakeBorder(m, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    _, r, _, (cx, cy) = cv2.minMaxLoc(dist)
    if not (r_min <= r <= r_max):
        return None
    return {"cx": float(cx) - 1, "cy": float(cy) - 1, "r": float(r)}


def measure(path: pathlib.Path) -> dict | None:
    rgb = np.asarray(Image.open(path).convert("RGB"))
    h, w = rgb.shape[:2]

    short = min(w, h)
    face = biggest_circle(rgb.max(axis=2) < BLACK_MAX,
                          FACE_R_MIN * short, FACE_R_MAX * short)
    if face is None:
        print(f"  {path.stem}: 얼굴창을 못 찾았다", file=sys.stderr)
        return None

    # 아바타는 **왼쪽 위 구석**에 있다. 그 구석만 잘라서 찾으면 카드 안쪽의 흰
    # 하이라이트(피망·오이의 반짝임)에 안 끌린다.
    corner = np.zeros((h, w), bool)
    corner[: int(h * 0.25), : int(w * 0.35)] = True
    avatar = biggest_circle((rgb.min(axis=2) > WHITE_MIN) & corner,
                            AVATAR_R_MIN * short, AVATAR_R_MAX * short)

    def pct(d: dict | None) -> dict | None:
        if d is None:
            return None
        # **구멍은 픽셀에서 원이다.** 카드가 3:4 라 퍼센트로 옮기면 rx ≠ ry 가 된다 —
        # 앱의 `Hole` 도 같은 단위(카드 크기 대비 %)를 쓰므로 그대로 맞다.
        return {
            "cx": round(d["cx"] / w * 100, 2), "cy": round(d["cy"] / h * 100, 2),
            "rx": round(d["r"] / w * 100, 2), "ry": round(d["r"] / h * 100, 2),
        }

    return {"size": (w, h), "face": pct(face), "avatar": pct(avatar)}


#: 뚫는 원을 잰 것보다 이만큼 키운다.
#:
#: JPG 라 구멍 가장자리가 압축으로 흐려져 있어서, 잰 반지름 그대로 뚫으면 **검은
#: 테가 한 줄 남는다.** 카드에 얹으면 얼굴 둘레에 그림자처럼 보인다. 조금 키워서
#: 그 띠까지 먹는다 — `punch_card_slots.py` 의 SHRINK 가 같은 이유로 1.07 이다.
PUNCH_GROW = 1.035


def punch_alpha(src: pathlib.Path, got: dict, dst: pathlib.Path) -> None:
    """잰 원만큼 **알파를 깎아** RGBA 로 다시 낸다.

    받은 판은 구멍이 까맣게(아바타는 희게) 칠해져 있을 뿐이다. 앱과 `compose.py`
    는 얼굴을 카드 **아래**에 깔고 카드를 그 위에 얹으므로, 구멍이 알파로 뚫려
    있지 않으면 얼굴이 아예 안 보인다.
    """
    im = Image.open(src).convert("RGBA")
    w, h = im.size
    hole = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(hole)
    for k in ("face", "avatar"):
        g = got[k]
        if g is None:
            continue
        cx, cy = g["cx"] / 100 * w, g["cy"] / 100 * h
        rx = g["rx"] / 100 * w * PUNCH_GROW
        ry = g["ry"] / 100 * h * PUNCH_GROW
        d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=255)
    hole = hole.filter(ImageFilter.GaussianBlur(max(1, int(min(w, h) * 0.002))))

    # **덮어쓰지 않고 뺀다.** 덮어쓰면 원래 투명하던 바깥까지 불투명해진다.
    a = np.asarray(im.getchannel("A"), dtype=np.int16)
    a = np.clip(a - np.asarray(hole, dtype=np.int16), 0, 255).astype(np.uint8)
    im.putalpha(Image.fromarray(a, "L"))
    im.save(dst, "WEBP", quality=92, method=6)


def overlay(path: pathlib.Path, got: dict, dst: pathlib.Path) -> None:
    """잰 원을 카드 위에 그려 둔다. **숫자만 보고 넘어가지 않는다.**"""
    im = Image.open(path).convert("RGB")
    w, h = im.size
    d = ImageDraw.Draw(im)
    for hole, color in ((got["face"], (80, 255, 120)), (got["avatar"], (255, 120, 200))):
        if hole is None:
            continue
        cx, cy = hole["cx"] / 100 * w, hole["cy"] / 100 * h
        rx, ry = hole["rx"] / 100 * w, hole["ry"] / 100 * h
        d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], outline=color, width=6)
    im.thumbnail((420, 10000))
    im.save(dst)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="templates/holes.toml 에 쓴다")
    ap.add_argument("--punch", action="store_true",
                    help="잰 원만큼 알파를 뚫어 RGBA WebP 로 다시 낸다")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    rows, tiles = {}, []
    for path in sorted(CARDS.glob("*-card-slots.jpg")):
        card_id = path.stem.replace("-card-slots", "")
        got = measure(path)
        if got is None:
            continue
        rows[card_id] = got
        tile = OUT / f"{card_id}.png"
        overlay(path, got, tile)
        tiles.append(tile)
        if args.punch:
            punch_alpha(path, got, CARDS / f"{card_id}-card-slots.webp")
        f, a = got["face"], got["avatar"]
        av = "—" if a is None else f"({a['cx']:5.2f}, {a['cy']:5.2f}) r {a['rx']:4.2f}"
        print(f"  {card_id:14} {got['size'][0]}x{got['size'][1]}  "
              f"얼굴 ({f['cx']:5.2f}, {f['cy']:5.2f}) r {f['rx']:5.2f}×{f['ry']:5.2f}   "
              f"아바타 {av}")

    # 확인용 한 장.
    if tiles:
        cols = 4
        first = Image.open(tiles[0])
        cw, ch = first.size
        sheet = Image.new("RGB", (cw * cols, ch * ((len(tiles) + cols - 1) // cols)), (20, 20, 20))
        for i, t in enumerate(tiles):
            sheet.paste(Image.open(t), ((i % cols) * cw, (i // cols) * ch))
        sheet.save(OUT / "_all.png")
        print(f"\n  → {OUT / '_all.png'}  **눈으로 볼 것**")

    text = render_toml(rows)
    if args.write:
        (ROOT / "templates" / "holes.toml").write_text(text)
        print(f"  → templates/holes.toml")
    else:
        print()
        print(text)
    return 0


def render_toml(rows: dict) -> str:
    head = """# 카드 12장의 구멍 좌표. 카드 크기에 대한 퍼센트 (ImmersiveScene 과 같은 단위).
#
# **사람이 자를 댄 값이 아니라 tools/measure_windows.py 가 잰 값이다.** 받은 카드는
# 이미 뚫려 있어서 구멍이 곧 좌표다 — 큰 얼굴창은 검게, 아바타 원은 희게 비어 있고,
# 그 덩어리의 중심과 반지름을 그대로 읽었다. 카드가 바뀌면 다시 돌리면 된다.
#
# id      : DexCards.kt 의 DexCard.id 와 같다 (앱이 어휘의 출처다)
# cx,cy   : 구멍 중심 (%)
# rx,ry   : 구멍 반지름 (%)
#
# ⚠ 프레임 배치가 12장 공통이 아니다. 피망·당근·가지·단호박은 검은 프레임에
#   제목이 가운데 오고 번호판이 아래에 있다. 글자 자리를 12장 공통 상수로 두면
#   그 넷에서 엉뚱한 자리에 글자가 찍힌다.
"""
    body = []
    for card_id, got in rows.items():
        f, a = got["face"], got["avatar"]
        body.append(f"\n[{card_id}]\nsize = [{got['size'][0]}, {got['size'][1]}]")
        body.append(f"face = {{ cx = {f['cx']}, cy = {f['cy']}, rx = {f['rx']}, ry = {f['ry']} }}")
        if a:
            body.append(f"avatar = {{ cx = {a['cx']}, cy = {a['cy']}, rx = {a['rx']}, ry = {a['ry']} }}")
        else:
            body.append("# avatar = 못 찾음 — 손으로 재야 한다")
    return head + "\n".join(body) + "\n"


if __name__ == "__main__":
    sys.exit(main())
