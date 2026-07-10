from pathlib import Path
import argparse
import csv
import random
import cv2


# ============================================================
# 현재 YOLO dataset 라벨 확인 코드
#
# 목적:
#   yolo_dataset/images/train, val 이미지 위에
#   yolo_dataset/labels/train, val의 bbox를 다시 그려서 확인한다.
#
# 확인 항목:
#   1. 이미지당 bbox 개수
#   2. class 0,1,2,3이 각각 1개씩 있는지
#   3. class 0이 정사각형 위치인지
#   4. class 1~3이 정사각형 기준 시계방향으로 들어갔는지
#
# 출력:
#   label_check_current/train/*.png
#   label_check_current/val/*.png
#   label_check_current/label_summary.csv
# ============================================================

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

COLORS = {
    0: (255, 0, 0),      # class 0: blue
    1: (0, 255, 255),    # class 1: yellow
    2: (255, 255, 255),  # class 2: white
    3: (0, 255, 0),      # class 3: green
}


def read_yolo_label(label_path: Path):
    """YOLO txt 라벨 읽기"""
    labels = []

    if not label_path.exists():
        return labels

    with open(label_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            parts = line.strip().split()

            if len(parts) < 5:
                continue

            cls = int(float(parts[0]))
            x = float(parts[1])
            y = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])

            labels.append({
                "line_idx": line_idx,
                "class_id": cls,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
            })

    return labels


def yolo_to_xyxy(label, image_w, image_h):
    """YOLO 정규화 좌표를 pixel bbox로 변환"""
    x = label["x"]
    y = label["y"]
    w = label["w"]
    h = label["h"]

    x1 = int(round((x - w / 2.0) * image_w))
    y1 = int(round((y - h / 2.0) * image_h))
    x2 = int(round((x + w / 2.0) * image_w))
    y2 = int(round((y + h / 2.0) * image_h))

    x1 = max(0, min(image_w - 1, x1))
    y1 = max(0, min(image_h - 1, y1))
    x2 = max(0, min(image_w - 1, x2))
    y2 = max(0, min(image_h - 1, y2))

    return x1, y1, x2, y2


def draw_labels(image, labels):
    """이미지 위에 현재 YOLO 라벨 bbox 표시"""
    vis = image.copy()
    image_h, image_w = image.shape[:2]

    for label in labels:
        cls = label["class_id"]
        color = COLORS.get(cls, (0, 0, 255))

        x1, y1, x2, y2 = yolo_to_xyxy(label, image_w, image_h)

        cx = int(round(label["x"] * image_w))
        cy = int(round(label["y"] * image_h))

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.circle(vis, (cx, cy), 4, color, -1)

        text = f"class {cls}"

        cv2.putText(
            vis,
            text,
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    return vis


def collect_images(image_dir: Path):
    """이미지 수집"""
    return sorted([
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ])


def check_one_split(dataset_root: Path, out_root: Path, split: str, max_images: int, seed: int):
    """train 또는 val 하나 확인"""
    image_dir = dataset_root / "images" / split
    label_dir = dataset_root / "labels" / split
    out_dir = out_root / split

    out_dir.mkdir(parents=True, exist_ok=True)

    if not image_dir.exists():
        raise FileNotFoundError(f"이미지 폴더 없음: {image_dir}")

    if not label_dir.exists():
        raise FileNotFoundError(f"라벨 폴더 없음: {label_dir}")

    images = collect_images(image_dir)

    # 너무 많으면 일부만 랜덤 확인
    if max_images > 0 and len(images) > max_images:
        rng = random.Random(seed)
        images = rng.sample(images, max_images)
        images = sorted(images)

    rows = []

    for idx, img_path in enumerate(images):
        stem = img_path.stem
        label_path = label_dir / f"{stem}.txt"

        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)

        if image is None:
            rows.append({
                "split": split,
                "stem": stem,
                "status": "image_read_failed",
            })
            continue

        labels = read_yolo_label(label_path)
        classes = [x["class_id"] for x in labels]

        class_counts = {c: classes.count(c) for c in [0, 1, 2, 3]}

        # 정상 기준: bbox 4개, class 0~3 각각 1개
        is_ok = (
            len(labels) == 4 and
            class_counts[0] == 1 and
            class_counts[1] == 1 and
            class_counts[2] == 1 and
            class_counts[3] == 1
        )

        vis = draw_labels(image, labels)
        out_path = out_dir / f"{stem}_label_check.png"
        cv2.imwrite(str(out_path), vis)

        rows.append({
            "split": split,
            "stem": stem,
            "status": "ok" if is_ok else "check_needed",
            "num_labels": len(labels),
            "class0_count": class_counts[0],
            "class1_count": class_counts[1],
            "class2_count": class_counts[2],
            "class3_count": class_counts[3],
            "image_file": str(img_path),
            "label_file": str(label_path),
            "check_image": str(out_path),
        })

        print(
            f"[{split}] {idx + 1}/{len(images)} {stem} | "
            f"labels={len(labels)} "
            f"c0={class_counts[0]} c1={class_counts[1]} "
            f"c2={class_counts[2]} c3={class_counts[3]} "
            f"{'OK' if is_ok else 'CHECK'}"
        )

    return rows


def save_csv(csv_path: Path, rows):
    """CSV 저장"""
    if not rows:
        return

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fields = sorted(set().union(*[row.keys() for row in rows]))

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="YOLO dataset root 경로"
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="라벨 확인 이미지 저장 폴더"
    )

    parser.add_argument(
        "--max_images",
        type=int,
        default=0,
        help="split별 확인 이미지 개수. 0이면 전체 확인"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="랜덤 샘플링 seed"
    )

    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    out_root = Path(args.out_dir)

    print("========== CONFIG ==========")
    print(f"dataset_root: {dataset_root}")
    print(f"out_dir:      {out_root}")
    print(f"max_images:   {args.max_images}")
    print("============================")

    all_rows = []

    for split in ["train", "val"]:
        rows = check_one_split(
            dataset_root=dataset_root,
            out_root=out_root,
            split=split,
            max_images=args.max_images,
            seed=args.seed,
        )
        all_rows.extend(rows)

    csv_path = out_root / "label_summary.csv"
    save_csv(csv_path, all_rows)

    print("\n========== DONE ==========")
    print(f"summary: {csv_path}")
    print("===========================")


if __name__ == "__main__":
    main()