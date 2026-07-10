from pathlib import Path
import argparse
import shutil


# ============================================================
# YOLO 4-class 라벨을 2-class 라벨로 변환하는 코드
#
# 기존 class:
#   0 = square_id
#   1 = clockwise_id_1
#   2 = clockwise_id_2
#   3 = clockwise_id_3
#
# 변환 후 class:
#   0 = square_id
#   1 = rect_id
#
# 변환 규칙:
#   기존 0 -> 0
#   기존 1,2,3 -> 1
#
# 이미지:
#   그대로 복사
#
# 라벨:
#   class 번호만 변환
# ============================================================

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def collect_images(image_dir: Path):
    """이미지 파일 수집"""
    images = []

    if not image_dir.exists():
        raise FileNotFoundError(f"이미지 폴더 없음: {image_dir}")

    for path in image_dir.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            images.append(path)

    return sorted(images)


def convert_label_file(src_label: Path, dst_label: Path):
    """4-class YOLO 라벨을 2-class YOLO 라벨로 변환"""
    if not src_label.exists():
        raise FileNotFoundError(f"라벨 파일 없음: {src_label}")

    out_lines = []

    with open(src_label, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()

            # YOLO 학습 라벨은 class x y w h 형식
            if len(parts) < 5:
                continue

            old_class = int(float(parts[0]))

            if old_class == 0:
                new_class = 0
            elif old_class in [1, 2, 3]:
                new_class = 1
            else:
                # 예상 외 class는 무시
                continue

            x_center = parts[1]
            y_center = parts[2]
            box_w = parts[3]
            box_h = parts[4]

            out_lines.append(f"{new_class} {x_center} {y_center} {box_w} {box_h}")

    if len(out_lines) == 0:
        raise RuntimeError(f"변환된 라벨이 비어 있습니다: {src_label}")

    dst_label.parent.mkdir(parents=True, exist_ok=True)

    with open(dst_label, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + "\n")


def process_split(split: str, src_dataset: Path, dst_dataset: Path, apply: bool):
    """train 또는 val 처리"""
    src_image_dir = src_dataset / "images" / split
    src_label_dir = src_dataset / "labels" / split

    dst_image_dir = dst_dataset / "images" / split
    dst_label_dir = dst_dataset / "labels" / split

    images = collect_images(src_image_dir)

    print(f"\n========== {split.upper()} ==========")
    print(f"source images: {len(images)}")

    converted = 0
    failed = 0

    if apply:
        dst_image_dir.mkdir(parents=True, exist_ok=True)
        dst_label_dir.mkdir(parents=True, exist_ok=True)

    for img_path in images:
        stem = img_path.stem
        src_label = src_label_dir / f"{stem}.txt"

        dst_image = dst_image_dir / img_path.name
        dst_label = dst_label_dir / f"{stem}.txt"

        try:
            if apply:
                shutil.copy2(str(img_path), str(dst_image))
                convert_label_file(src_label, dst_label)

            converted += 1
            print(f"[OK] {split} {stem}")

        except Exception as e:
            failed += 1
            print(f"[FAIL] {split} {stem}: {e}")

    print(f"[{split}] converted: {converted}")
    print(f"[{split}] failed:    {failed}")

    return converted, failed


def write_data_yaml(dst_dataset: Path):
    """2-class YOLO data.yaml 생성"""
    yaml_text = f"""path: {dst_dataset.resolve().as_posix()}
train: images/train
val: images/val

names:
  0: square_id
  1: rect_id
"""

    yaml_path = dst_dataset / "data.yaml"

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_text)

    print(f"\ndata.yaml saved: {yaml_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--src_dataset",
        type=str,
        default="../yolo_dataset",
        help="기존 4-class YOLO 데이터셋"
    )

    parser.add_argument(
        "--dst_dataset",
        type=str,
        default="../yolo_dataset_2class",
        help="생성할 2-class YOLO 데이터셋"
    )

    parser.add_argument(
        "--apply",
        action="store_true",
        help="실제로 데이터셋 생성"
    )

    args = parser.parse_args()

    src_dataset = Path(args.src_dataset)
    dst_dataset = Path(args.dst_dataset)

    if not src_dataset.exists():
        raise FileNotFoundError(f"src_dataset 없음: {src_dataset}")

    print("========== CONFIG ==========")
    print(f"src_dataset: {src_dataset.resolve()}")
    print(f"dst_dataset: {dst_dataset.resolve()}")
    print(f"apply:       {args.apply}")
    print("============================")

    total_converted = 0
    total_failed = 0

    for split in ["train", "val"]:
        converted, failed = process_split(split, src_dataset, dst_dataset, args.apply)
        total_converted += converted
        total_failed += failed

    if args.apply:
        write_data_yaml(dst_dataset)

    print("\n========== TOTAL RESULT ==========")
    print(f"converted: {total_converted}")
    print(f"failed:    {total_failed}")

    if not args.apply:
        print("\n현재는 미리보기 모드입니다. 실제 생성하려면 --apply를 붙이세요.")

    print("==================================")


if __name__ == "__main__":
    main()