"""**관문.** 정답 라벨로 얼굴 상자와 목선이 쓸 만한지 숫자로 답한다.

    uv run tools/verify_face_box.py --split val --sweep

## 왜 모델 없이 먼저 이걸 하나

모델을 학습하는 데는 GPU 로 두 시간이 든다. 그런데 학습이 아무리 잘 돼도 **정답
키포인트로조차 얼굴 상자가 안 나온다면** 그 두 시간은 통째로 버리는 것이다.
정답 라벨은 "모델이 완벽했을 때의 상한" 이라서, 여기서 안 되면 어디서도 안 된다.

PR #19 가 "누끼가 8할이라 0단계에서 먼저 보고, 안 되면 멈추고 다시 상의한다" 고
한 것과 같은 자세다.

## IoU 를 왜 안 쓰나

dog-pose 에는 **정답 얼굴 상자가 없다.** 몸통 상자와 키포인트뿐이다. 그래서 IoU 를
잴 대상이 아예 없고, 대신 네 가지 대리 지표를 쓴다.

    1 가시성  얼굴 키포인트가 3개 이상 보이는 이미지 비율   ← 이게 낮으면 접근 자체가 끝
    2 재현율  상자가 보이는 얼굴 키포인트를 **전부** 담는 비율
    3 오염률  **카드 구멍 안 · 목선 위**에 몸통 키포인트가 드는 비율 ← 실제로 보이는 것만
    4 넓이비  얼굴상자 / 몸통상자. 분포가 갈라지면 견종별로 갈린다는 뜻

2 와 3 은 **서로 반대로 움직인다.** 마진을 키우면 재현율이 오르고 오염률도 오른다.
`--sweep` 이 그 트레이드오프를 격자로 찍어서, 기본 마진값에 근거를 남긴다.

## 숫자가 통과해도 컨택트시트를 본다

좌표 지표는 상자가 얼굴을 담았는지는 알려 주지만 **그림이 쓸 만한지는 못 잡는다.**
60장을 뽑아 상자·목선·키포인트를 그린 한 장을 같이 낸다.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import dogpose as kp  # noqa: E402
from src import facebox as geo  # noqa: E402
from src.dogpose import DATASET_ROOT, load_split  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "out" / "verify"

#: 통과선. 밑돌면 멈추고 다시 상의한다.
#:
#: 오염은 **카드에 실제로 보이는 것**만 센다 — 구멍(타원) 안이면서 목선 위.
#: 상자 네 모서리와 목선 아래는 각각 구멍 밖·페이드 아래라 카드에 안 나타난다.
#: 처음에는 상자 전체로 셌는데, 그러면 보이지도 않는 앞발 때문에 마진을 줄이게
#: 되고 그 대가로 이마가 잘린다.
GATE = {"coverage": 0.85, "recall": 0.97, "contamination_hole": 0.20}

#: `--sweep` 이 훑을 격자.
SWEEP_TOP = (0.30, 0.45, 0.60)
SWEEP_SIDE = (0.15, 0.25, 0.35)
SWEEP_BOTTOM = (0.35, 0.55, 0.75)

SHEET_COLS, SHEET_ROWS, SHEET_CELL = 10, 6, 220


def pct_str(v: float | None) -> str:
    """`None` 이어도 요약이 안 죽게. throat 이 한 장도 없는 부분집합에서 실제로 터졌다."""
    return "—" if v is None else f"{v:.1%}"


def measure(samples, margins: geo.Margins) -> dict:
    """한 마진 설정으로 전체를 훑어 지표를 낸다.

    오염률을 **셋으로 나눠 낸다.** 좁아지는 순서다.

        contamination           상자 안에 몸통 키포인트가 있다
        contamination_visible   그중 목선 위에 있다 (아래는 페이드가 녹인다)
        contamination_hole      그중 구멍(타원) 안이다  ← 카드에 실제로 보이는 것

    셋을 안 가르면 "턱 밑에 앞발이 있는 초상 사진" 이 전부 실패로 세어진다. 그러면
    오염률을 낮추려고 마진을 줄이게 되고, **그 대가로 이마가 잘린다.** 실제로 상자
    전체로 재면 51.6%, 구멍 안으로 재면 23.9% 로 두 배 넘게 벌어진다.
    """
    n = made = recall_ok = contaminated = visible_contam = hole_contam = 0
    area_ratios: list[float] = []
    # 오염이 **어느 키포인트에서** 나는지 따로 센다.
    #
    # 오염률 하나만 보면 "상자를 줄여야 한다" 로만 읽힌다. 그런데 범인이 앞다리
    # 팔꿈치라면 그건 상자가 큰 게 아니라 **얼굴 옆에 앞발이 있는 사진**이고,
    # 꼬리라면 그건 개가 몸을 말고 있는 것이다. 대책이 정반대다.
    offenders = {i: 0 for i in kp.BODY}

    for s, (iw, ih) in samples:
        n += 1
        box = geo.face_box(s.kps, iw, ih, margins)
        if box is None:
            continue
        made += 1

        face_pts = geo.visible(s.kps, kp.FACE)
        if all(box.contains(float(x), float(y)) for x, y in face_pts):
            recall_ok += 1
        hits = geo.body_hits(s.kps, box)
        if hits:
            contaminated += 1
        neck = geo.neck_t(s.kps, box, ih)
        above = geo.body_hits(s.kps, box, above=neck)
        if above:
            visible_contam += 1
        # ★ 실제로 카드에 보이는 것: **구멍(타원) 안 + 목선 위**
        in_hole = geo.body_hits(s.kps, box, above=neck, hole=True)
        if in_hole:
            hole_contam += 1
        for i in in_hole:
            offenders[i] += 1

        body = geo.body_box(s.kps)
        if body is not None and body.w * body.h > 0:
            area_ratios.append((box.w * box.h) / (body.w * body.h))

    return {
        "n": n,
        "coverage": made / n if n else 0.0,
        # 재현율·오염률은 **상자가 만들어진 것들 중에서** 잰다. 못 만든 것을 섞으면
        # 두 가지 다른 실패(못 찾음 · 잘못 찾음)가 한 숫자로 뭉개진다.
        "recall": recall_ok / made if made else 0.0,
        "contamination": contaminated / made if made else 0.0,
        "contamination_visible": visible_contam / made if made else 0.0,
        "contamination_hole": hole_contam / made if made else 0.0,
        "area_ratio_median": statistics.median(area_ratios) if area_ratios else None,
        "area_ratio_iqr": (
            [
                round(float(np.percentile(area_ratios, 25)), 4),
                round(float(np.percentile(area_ratios, 75)), 4),
            ]
            if area_ratios
            else None
        ),
        "offenders": {
            kp.NAMES[i]: round(c / made, 4)
            for i, c in sorted(offenders.items(), key=lambda kv: -kv[1])
            if c and made
        },
    }


def label_census(samples) -> dict:
    """키포인트별로 **실제로 찍혀 있는 비율.**

    이걸 안 세고 시작했다가 `throat` 로 목선을 잡는 설계를 통째로 세웠는데,
    train·val 8,476장에 `throat` 이 한 장도 없었다. 지표보다 이게 먼저다.
    """
    a = np.stack([s.kps for s, _ in samples])
    vis = (a[:, :, 2] > 0).mean(axis=0)
    return {kp.NAMES[i]: round(float(vis[i]), 4) for i in range(kp.N_KEYPOINTS)}


def neck_report(samples) -> dict:
    """목선을 어느 갈래로 얼마나 정하는지, 그리고 앱의 기본값 0.74 가 맞는지."""
    from_throat = from_chin = fallback = 0
    ts: list[float] = []

    for s, (iw, ih) in samples:
        box = geo.face_box(s.kps, iw, ih)
        if box is None:
            continue
        ts.append(geo.neck_t(s.kps, box, ih))

        if s.kps[kp.THROAT][2] > 0:
            from_throat += 1
        elif s.kps[kp.CHIN][2] > 0 and len(geo.visible(s.kps, kp.EAR_BASES)) > 0:
            from_chin += 1
        else:
            fallback += 1

    n = max(from_throat + from_chin + fallback, 1)
    pct = lambda a, q: round(float(np.percentile(a, q)), 4) if a else None
    return {
        # 이 셋의 비율이 목선을 **얼마나 믿을 수 있는지** 그 자체다.
        "source_throat": round(from_throat / n, 4),
        "source_chin_ears": round(from_chin / n, 4),
        "source_fallback": round(fallback / n, 4),
        "neck_t": {
            "median": pct(ts, 50),
            "p10": pct(ts, 10),
            "p90": pct(ts, 90),
            "app_fallback": geo.NECK_FALLBACK,
        },
        "chin_drop_in_code": geo.CHIN_DROP,
    }


def contact_sheet(samples, seed: int, path: Path) -> None:
    """무작위 60장에 키포인트·얼굴상자·목선을 그려 한 장으로."""
    picks = random.Random(seed).sample(samples, min(SHEET_COLS * SHEET_ROWS, len(samples)))
    sheet = Image.new("RGB", (SHEET_COLS * SHEET_CELL, SHEET_ROWS * SHEET_CELL), (24, 22, 20))

    for i, (s, (iw, ih)) in enumerate(picks):
        im = Image.open(s.image).convert("RGB")
        d = ImageDraw.Draw(im)
        r = max(2, min(iw, ih) // 160)

        for idx in kp.FACE:
            x, y, v = s.kps[idx]
            if v > 0:
                d.ellipse([x * iw - r, y * ih - r, x * iw + r, y * ih + r], fill=(255, 90, 140))
        for idx in (kp.THROAT, kp.WITHERS):
            x, y, v = s.kps[idx]
            if v > 0:
                d.ellipse([x * iw - r, y * ih - r, x * iw + r, y * ih + r], fill=(90, 200, 255))

        box = geo.face_box(s.kps, iw, ih)
        if box is not None:
            l, t, rr, b = box.to_pixels(iw, ih)
            d.rectangle([l, t, rr, b], outline=(120, 255, 120), width=max(2, r))
            # 목선 — 이 아래가 카드에서 녹는다
            ny = t + (b - t) * geo.neck_t(s.kps, box, ih)
            d.line([l, ny, rr, ny], fill=(255, 210, 60), width=max(2, r))
        else:
            d.line([0, 0, iw, ih], fill=(255, 70, 70), width=max(3, r))

        im.thumbnail((SHEET_CELL, SHEET_CELL))
        col, row = i % SHEET_COLS, i // SHEET_COLS
        sheet.paste(
            im,
            (col * SHEET_CELL + (SHEET_CELL - im.width) // 2,
             row * SHEET_CELL + (SHEET_CELL - im.height) // 2),
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--root", type=Path, default=DATASET_ROOT)
    ap.add_argument("--limit", type=int, default=0, help="빠르게 볼 때만")
    ap.add_argument("--sweep", action="store_true", help="마진 격자를 훑는다")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"{args.split} 읽는 중 …")
    raw = list(load_split(args.split, args.root))
    if args.limit:
        raw = raw[: args.limit]

    # 어긋난 라벨은 **세는 대신 걸러낸다.** 섞어 두면 우리 계산이 틀린 것인지
    # 라벨이 틀린 것인지 구분이 안 된다. 몇 장을 걸렀는지는 따로 보고한다.
    kept = [s for s in raw if s.sane()]
    dropped = len(raw) - len(kept)

    samples = [(s, s.size()) for s in kept]
    print(f"  {len(raw)}장 중 {dropped}장은 라벨이 어긋나 뺐습니다 ({dropped / len(raw):.1%})")

    base = measure(samples, geo.DEFAULT_MARGINS)

    # **마진 0 의 오염률이 바닥이다.** 얼굴 키포인트만으로 만든 최소 상자에서도
    # 앞발이 들어온다면 그건 우리 마진 탓이 아니라 사진이 그런 것이다 (Stanford
    # Dogs 는 대부분 얼굴이 화면을 채우는 초상 사진이라 발·꼬리가 턱 옆에 온다).
    # 이 줄이 없으면 오염률을 보고 마진만 계속 줄이다가 이마를 잘라 먹는다.
    floor = measure(samples, geo.Margins(top=0.0, side=0.0, bottom=0.0))
    necks = neck_report(samples)

    report = {
        "split": args.split,
        "images_total": len(raw),
        "images_dropped_bad_label": dropped,
        "margins": vars(geo.DEFAULT_MARGINS),
        "label_census": label_census(samples),
        "metrics": base,
        "metrics_no_margin": floor,
        "neck": necks,
        "gate": GATE,
    }

    if args.sweep:
        rows = []
        for top in SWEEP_TOP:
            for side in SWEEP_SIDE:
                for bottom in SWEEP_BOTTOM:
                    m = geo.Margins(top=top, side=side, bottom=bottom)
                    r = measure(samples, m)
                    rows.append(
                        {"top": top, "side": side, "bottom": bottom,
                         "recall": round(r["recall"], 4),
                         "contamination_hole": round(r["contamination_hole"], 4),
                         "area_ratio_median": r["area_ratio_median"]}
                    )
        report["sweep"] = rows

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    contact_sheet(samples, args.seed, OUT / "contact_sheet.png")

    # -- 사람이 읽을 요약 ----------------------------------------------------
    print()
    print(f"  1 가시성   {base['coverage']:.1%}   (통과선 {GATE['coverage']:.0%})")
    print(f"  2 재현율   {base['recall']:.1%}   (통과선 {GATE['recall']:.0%})")
    print(f"  3 오염률   구멍 안 {base['contamination_hole']:.1%}   "
          f"(통과선 {GATE['contamination_hole']:.0%} 이하)  ·  "
          f"목선 위 {base['contamination_visible']:.1%}  ·  "
          f"상자 전체 {base['contamination']:.1%}")
    print(f"  4 넓이비   중앙값 {base['area_ratio_median']:.3f}  IQR {base['area_ratio_iqr']}")
    print()
    print(f"  마진 0 일 때의 바닥   재현율 {floor['recall']:.1%} · "
          f"구멍 안 오염률 {floor['contamination_hole']:.1%}")
    top3 = list(base["offenders"].items())[:3]
    print("  오염 범인 상위  " + " · ".join(f"{n} {v:.1%}" for n, v in top3))
    print()
    print(f"  목선 출처   throat {pct_str(necks['source_throat'])} · "
          f"턱+귀밑 {pct_str(necks['source_chin_ears'])} · "
          f"폴백 {pct_str(necks['source_fallback'])}")
    print(f"        상자 안 위치 중앙값 {necks['neck_t']['median']} "
          f"(p10 {necks['neck_t']['p10']} · p90 {necks['neck_t']['p90']} · "
          f"앱 기본값 {geo.NECK_FALLBACK})")

    if "sweep" in report:
        print()
        print("  마진 스윕 (재현율 ↑ 좋음 · 구멍 안 오염률 ↓ 좋음)")
        print("    top  side bottom |  재현율  오염률  넓이비")
        for r in report["sweep"]:
            print(f"    {r['top']:.2f} {r['side']:.2f} {r['bottom']:.4}   |  "
                  f"{r['recall']:.3f}   {r['contamination_hole']:.3f}  "
                  f"{r['area_ratio_median']:.3f}")

    passed = (
        base["coverage"] >= GATE["coverage"]
        and base["recall"] >= GATE["recall"]
        and base["contamination_hole"] <= GATE["contamination_hole"]
    )
    print()
    print(f"  → {OUT / 'report.json'}")
    print(f"  → {OUT / 'contact_sheet.png'}  **눈으로 볼 것**")
    print()
    print("  관문: " + ("통과" if passed else "미달 — 멈추고 다시 상의"))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
