import argparse
import shutil
from pathlib import Path

import cv2


def read_yolo_detect_label(label_path: Path):
    """
    기존 YOLO detect 라벨을 읽는다.
    한 줄 형식:
    class x_center y_center width height
    모든 값은 0~1 정규화 좌표라고 가정한다.
    """
    items = []

    if not label_path.exists():
        return items

    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 5:
            continue

        cls = int(float(parts[0]))
        x = float(parts[1])
        y = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])

        items.append({
            "cls": cls,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "area": w * h,
        })

    return items


def clamp01(v: float) -> float:
    """좌표가 0~1 범위를 벗어나지 않도록 제한한다."""
    return max(0.0, min(1.0, v))


def make_union_bbox(boxes, margin_ratio: float):
    """
    4개 내부 ID bbox를 모두 포함하는 하나의 pose object bbox를 만든다.
    여기서 bbox는 marker 전체가 아니라, 내부 ID 구조 전체를 감싸는 bbox이다.
    """
    x1_list = [b["x"] - b["w"] / 2 for b in boxes]
    y1_list = [b["y"] - b["h"] / 2 for b in boxes]
    x2_list = [b["x"] + b["w"] / 2 for b in boxes]
    y2_list = [b["y"] + b["h"] / 2 for b in boxes]

    x1 = min(x1_list)
    y1 = min(y1_list)
    x2 = max(x2_list)
    y2 = max(y2_list)

    bw = x2 - x1
    bh = y2 - y1

    # bbox가 너무 타이트하지 않도록 학습 라벨 bbox만 약간 확장한다.
    # 이건 추론 후처리가 아니라 학습용 pose object bbox 정의이다.
    mx = bw * margin_ratio
    my = bh * margin_ratio

    x1 = clamp01(x1 - mx)
    y1 = clamp01(y1 - my)
    x2 = clamp01(x2 + mx)
    y2 = clamp01(y2 + my)

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1

    return cx, cy, w, h


def draw_check_image(image_path: Path, out_path: Path, pose_values):
    """
    pose 라벨이 제대로 생성됐는지 확인하기 위한 개별 이미지 저장.
    contact sheet가 아니라 파일별 확인용 이미지이다.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return

    H, W = img.shape[:2]

    _, bx, by, bw, bh, *kpts = pose_values

    # object bbox 표시
    x1 = int((bx - bw / 2) * W)
    y1 = int((by - bh / 2) * H)
    x2 = int((bx + bw / 2) * W)
    y2 = int((by + bh / 2) * H)
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)

    names = ["square", "rect1", "rect2", "rect3"]

    for i in range(4):
        px = kpts[i * 3 + 0]
        py = kpts[i * 3 + 1]
        vis = int(kpts[i * 3 + 2])

        if vis <= 0:
            continue

        cx = int(px * W)
        cy = int(py * H)

        cv2.circle(img, (cx, cy), 5, (0, 255, 255), -1)
        cv2.putText(
            img,
            names[i],
            (cx + 6, cy - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


def convert_split(src_root: Path, out_root: Path, split: str, margin_ratio: float, check_dir: Path | None):
    """
    train 또는 val split 하나를 YOLO-pose 형식으로 변환한다.
    기존 detect label의 class0~3 중심을 pose keypoint0~3으로 사용한다.
    """
    src_img_dir = src_root / "images" / split
    src_lbl_dir = src_root / "labels" / split

    out_img_dir = out_root / "images" / split
    out_lbl_dir = out_root / "labels" / split

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    image_files = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        image_files.extend(src_img_dir.glob(ext))

    image_files = sorted(image_files)

    success = 0
    failed = 0

    for img_path in image_files:
        label_path = src_lbl_dir / f"{img_path.stem}.txt"
        items = read_yolo_detect_label(label_path)

        # class별 후보를 모은다.
        by_class = {0: [], 1: [], 2: [], 3: []}
        for item in items:
            if item["cls"] in by_class:
                by_class[item["cls"]].append(item)

        # class0~3이 모두 있어야 pose 라벨 생성 가능
        if any(len(by_class[c]) == 0 for c in range(4)):
            failed += 1
            print(f"[FAILED] {split}/{img_path.name}: missing class among 0,1,2,3")
            continue

        # 같은 class가 여러 개 있으면 면적이 가장 큰 bbox를 사용한다.
        # clean 가상데이터에서는 보통 class별 1개가 정상이다.
        selected = []
        for c in range(4):
            selected.append(max(by_class[c], key=lambda b: b["area"]))

        bx, by, bw, bh = make_union_bbox(selected, margin_ratio)

        # pose label:
        # class x y w h px0 py0 v0 px1 py1 v1 px2 py2 v2 px3 py3 v3
        # visibility=2는 보이는 keypoint로 둔다.
        pose_values = [
            0,
            bx, by, bw, bh,
            selected[0]["x"], selected[0]["y"], 2,
            selected[1]["x"], selected[1]["y"], 2,
            selected[2]["x"], selected[2]["y"], 2,
            selected[3]["x"], selected[3]["y"], 2,
        ]

        out_label_path = out_lbl_dir / f"{img_path.stem}.txt"
        out_label_path.write_text(
            " ".join(
                str(v) if isinstance(v, int) else f"{v:.6f}"
                for v in pose_values
            ) + "\n",
            encoding="utf-8",
        )

        shutil.copy2(img_path, out_img_dir / img_path.name)

        if check_dir is not None:
            check_path = check_dir / split / f"{img_path.stem}_pose_check.jpg"
            draw_check_image(img_path, check_path, pose_values)

        success += 1

    print(f"[{split}] success={success}, failed={failed}")


def write_yaml(out_root: Path):
    """
    Ultralytics YOLO-pose 학습용 data.yaml 생성.
    keypoint 4개, 각 keypoint는 x,y,visibility 3개 값을 가진다.
    """
    yaml_text = f"""path: {out_root.as_posix()}
train: images/train
val: images/val

names:
  0: marker

kpt_shape: [4, 3]
flip_idx: [0, 1, 2, 3]
"""

    (out_root / "data.yaml").write_text(yaml_text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dataset_root", required=True, help="기존 YOLO 4-class detect 데이터셋 루트")
    parser.add_argument("--out_dataset_root", required=True, help="생성할 YOLO-pose 데이터셋 루트")
    parser.add_argument("--margin_ratio", type=float, default=0.05, help="pose object bbox 확장 비율")
    parser.add_argument("--check_dir", default="", help="pose 라벨 확인 이미지 저장 폴더")
    parser.add_argument("--clean", action="store_true", help="출력 폴더가 있으면 삭제 후 새로 생성")
    args = parser.parse_args()

    src_root = Path(args.src_dataset_root)
    out_root = Path(args.out_dataset_root)
    check_dir = Path(args.check_dir) if args.check_dir else None

    if args.clean and out_root.exists():
        shutil.rmtree(out_root)

    if args.clean and check_dir is not None and check_dir.exists():
        shutil.rmtree(check_dir)

    convert_split(src_root, out_root, "train", args.margin_ratio, check_dir)
    convert_split(src_root, out_root, "val", args.margin_ratio, check_dir)
    write_yaml(out_root)

    print(f"[DONE] pose dataset saved: {out_root}")
    print(f"[DONE] yaml saved: {out_root / 'data.yaml'}")


if __name__ == "__main__":
    main()