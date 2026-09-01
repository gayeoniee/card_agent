"""YOLO11 pose 를 dog-pose 로 학습한다. **이 컨테이너에서는 안 돈다 — Colab 용이다.**

    uv sync --extra train
    uv run tools/train_dog_pose.py --arm face --epochs 100

## 왜 여기서 안 돌리나

학습 이미지가 6,773장이고 GPU 가 있어야 한다. T4 한 장에서 100 에폭이 대략 1.5~2시간,
CPU 면 며칠이다. 그래서 이 저장소에서는 **정답 라벨로 기하를 먼저 검증하고**
(`verify_face_box.py`), 학습은 Colab 에 이 파일을 올려 돌린다.

## 갈래 둘을 같은 val 로 비교한다

    --arm full   24개 그대로. 라벨을 안 건드린다
    --arm face   **얼굴 6개만** 남기고 라벨을 다시 쓴다

`face` 가 그럴듯한 이유: 우리는 24개 중 6개만 쓴다 (나머지 중 4개는 이 데이터셋에
아예 라벨이 없고, 14개는 다리·꼬리다). 헤드가 작아지면 폰에서 빠르고, 학습 신호가
우리가 쓰는 키포인트에 몰린다.

`face` 가 나쁠 수 있는 이유: 몸통 키포인트가 사라지면 **자세라는 맥락**이 사라진다.
옆모습이나 얼굴이 가린 사진에서 "이 덩어리가 개의 머리다" 를 몸이 알려 주는데,
그 힌트가 없어진다.

**어느 쪽이 이겼는지는 숫자로 적는다.** 같은 val 셋에서 pose mAP50-95 와, 우리가
실제로 쓰는 값(`verify_face_box.py` 의 지표)을 둘 다 본다 — mAP 가 높아도 얼굴
상자가 나빠질 수 있고, 우리에게 중요한 것은 뒤쪽이다.

## ⚠️ 라이선스

`ultralytics` 는 **AGPL-3.0** 이고, Ultralytics 는 그 코드로 학습한 가중치도 AGPL
파생물로 본다. `.tflite` 만 APK 에 넣어도 조항이 따라온다. dog-pose 는 Stanford
Dogs(연구용 조건) 기반이라 상용 배포용 가중치의 학습 데이터로 쓰는 것도 회색지대다.
**실험은 이대로 가되, 앱에 실을 때 다시 판단한다** — README 의 라이선스 절 참고.
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src import dogpose as kp  # noqa: E402
from src.dogpose import DATASET_ROOT  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: `--arm face` 가 남기는 키포인트. **눈은 여기 없다** — 라벨이 없기 때문이다.
FACE_KEEP: tuple[int, ...] = kp.FACE


def write_face_dataset(src: pathlib.Path, dst: pathlib.Path) -> pathlib.Path:
    """24개 라벨을 6개로 줄여 다시 쓴다. 이미지는 심볼릭 링크로 재활용한다."""
    dst.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        (dst / "labels" / split).mkdir(parents=True, exist_ok=True)
        img_dir = dst / "images" / split
        if not img_dir.exists():
            img_dir.symlink_to(src / "images" / split, target_is_directory=True)

        for p in (src / "labels" / split).glob("*.txt"):
            out = []
            for line in p.read_text().splitlines():
                f = line.split()
                if len(f) != 5 + kp.N_KEYPOINTS * 3:
                    continue
                head = f[:5]
                kps = [f[5 + i * 3 : 8 + i * 3] for i in range(kp.N_KEYPOINTS)]
                out.append(" ".join(head + [v for i in FACE_KEEP for v in kps[i]]))
            (dst / "labels" / split / p.name).write_text("\n".join(out) + "\n")

    # 좌우 뒤집기 증강을 쓰려면 flip_idx 가 있어야 한다. **없으면 뒤집은 사진에서
    # 왼쪽 귀 자리에 오른쪽 귀 정답이 들어가 학습이 스스로 망가진다.**
    order = list(FACE_KEEP)
    flip = list(range(len(order)))
    for a, b in kp.FLIP_PAIRS:
        if a in order and b in order:
            flip[order.index(a)] = order.index(b)
            flip[order.index(b)] = order.index(a)

    names = "\n".join(f"    - {kp.NAMES[i]}" for i in order)
    yaml = dst / "dog-face-pose.yaml"
    yaml.write_text(
        f"""# tools/train_dog_pose.py 가 만든 것. 손으로 고치지 말 것.
path: {dst}
train: images/train
val: images/val

kpt_shape: [{len(order)}, 3]
flip_idx: {flip}

names:
  0: dog

kpt_names:
  0:
{names}
"""
    )
    return yaml


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("full", "face"), default="face")
    ap.add_argument("--model", default="yolo11n-pose.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--root", type=pathlib.Path, default=DATASET_ROOT)
    ap.add_argument("--export", action="store_true", help="끝나고 TFLite(int8)로 내보낸다")
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print(
            "ultralytics 가 없습니다. `uv sync --extra train` 으로 켜세요.\n"
            "**AGPL-3.0 이 여기서 들어옵니다** — 이 파일 맨 위의 라이선스 메모를 읽으세요.",
            file=sys.stderr,
        )
        return 2

    if args.arm == "face":
        data = write_face_dataset(args.root, ROOT / ".data" / "dog-face-pose")
        print(f"얼굴 {len(FACE_KEEP)}개짜리 라벨을 새로 썼습니다 → {data}")
    else:
        data = "dog-pose.yaml"  # ultralytics 가 들고 있는 것

    model = YOLO(args.model)
    model.train(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=f"dogface-{args.arm}",
    )
    metrics = model.val()
    print(f"\n  pose mAP50-95 = {metrics.pose.map:.4f}   (갈래: {args.arm})")
    print("  **이 숫자만 보고 고르지 말 것.** 같은 가중치로 tools/verify_face_box.py 를")
    print("  다시 돌려 얼굴 상자 지표를 보는 것이 우리에게 중요한 쪽이다.")

    if args.export:
        # int8 TFLite. 앱은 minSdk 26 이고 모델을 APK 에 넣을 것이라 작아야 한다.
        path = model.export(format="tflite", int8=True, imgsz=args.imgsz)
        dest = ROOT / "out" / "models" / pathlib.Path(path).name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(path, dest)
        print(f"  → {dest}  ({dest.stat().st_size / 1e6:.1f} MB)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
