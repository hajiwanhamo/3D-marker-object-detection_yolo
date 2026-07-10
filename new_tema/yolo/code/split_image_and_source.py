from pathlib import Path
import argparse
import random
import shutil

# ============================================================
# YOLO 이미지 + source_label 동시 train/val 분할 코드
#
# 목적:
# 1. yolo_dataset/images/train 이미지 중 20%를 images/val로 이동
# 2. 같은 stem의 YOLO label txt가 있으면 labels/train -> labels/val로 이동
# 3. 같은 stem의 source 파일을 source/train -> source/val로 이동
#
# 예:
#   aug_000000.png
#   aug_000000_marker_top_id_uv.npy
#   aug_000000_marker_meta.json
#
# 핵심:
#   이미지와 source_label이 서로 다른 split으로 갈라지지 않도록
#   이미지 이동 기준에 맞춰 source 파일도 같이 이동
# ============================================================

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def collect_images(image_dir: Path):
    """train 이미지 목록 수집"""
    if not image_dir.exists():
        raise FileNotFoundError(f"이미지 폴더를 찾을 수 없습니다: {image_dir}")

    images = []
    for path in image_dir.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            images.append(path)

    return sorted(images)


def count_images(image_dir: Path):
    """이미지 개수 확인"""
    if not image_dir.exists():
        return 0

    return sum(
        1 for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def move_file(src: Path, dst: Path, apply: bool):
    """파일 이동"""
    if not src.exists():
        return False

    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        raise FileExistsError(f"이동 대상 파일이 이미 존재합니다: {dst}")

    if apply:
        shutil.move(str(src), str(dst))
        print(f"[MOVE] {src} -> {dst}")
    else:
        print(f"[DRY-RUN MOVE] {src} -> {dst}")

    return True


def move_source_files(stem: str, source_train_dir: Path, source_val_dir: Path, apply: bool):
    """
    같은 stem을 가진 source 파일 이동

    기본적으로 아래 형태를 모두 이동:
    aug_000000_marker_*
    """
    matched_files = sorted(source_train_dir.glob(f"{stem}_marker_*"))

    if len(matched_files) == 0:
        return 0

    moved_count = 0

    for src_path in matched_files:
        dst_path = source_val_dir / src_path.name
        move_file(src_path, dst_path, apply)
        moved_count += 1

    return moved_count


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        default="../yolo_dataset",
        help="YOLO 데이터셋 루트 폴더"
    )

    parser.add_argument(
        "--source",
        type=str,
        default="../source",
        help="source_label 루트 폴더. 내부에 train, val 폴더가 있어야 함"
    )

    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="검증 데이터 비율"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="랜덤 고정값"
    )

    parser.add_argument(
        "--apply",
        action="store_true",
        help="실제로 파일 이동"
    )

    parser.add_argument(
        "--allow_existing_val",
        action="store_true",
        help="val 폴더에 이미지가 이미 있어도 추가 분할 허용"
    )

    args = parser.parse_args()

    dataset_root = Path(args.dataset)
    source_root = Path(args.source)

    img_train_dir = dataset_root / "images" / "train"
    img_val_dir = dataset_root / "images" / "val"

    label_train_dir = dataset_root / "labels" / "train"
    label_val_dir = dataset_root / "labels" / "val"

    source_train_dir = source_root / "train"
    source_val_dir = source_root / "val"

    if not dataset_root.exists():
        raise FileNotFoundError(f"YOLO dataset 폴더를 찾을 수 없습니다: {dataset_root}")

    if not source_train_dir.exists():
        raise FileNotFoundError(f"source/train 폴더를 찾을 수 없습니다: {source_train_dir}")

    img_val_dir.mkdir(parents=True, exist_ok=True)
    label_train_dir.mkdir(parents=True, exist_ok=True)
    label_val_dir.mkdir(parents=True, exist_ok=True)
    source_val_dir.mkdir(parents=True, exist_ok=True)

    existing_val_count = count_images(img_val_dir)

    if existing_val_count > 0 and not args.allow_existing_val:
        raise RuntimeError(
            f"images/val에 이미 이미지가 {existing_val_count}개 있습니다. "
            f"중복 분할 방지를 위해 중단합니다. "
            f"추가 분할이 목적이면 --allow_existing_val 옵션을 사용하세요."
        )

    images = collect_images(img_train_dir)

    if len(images) == 0:
        raise RuntimeError(f"train 이미지가 없습니다: {img_train_dir}")

    val_count = int(len(images) * args.val_ratio)

    if val_count <= 0:
        raise RuntimeError(
            f"val_count가 0입니다. image_count={len(images)}, val_ratio={args.val_ratio}"
        )

    random.seed(args.seed)
    random.shuffle(images)

    val_images = images[:val_count]

    print("========== SPLIT CONFIG ==========")
    print(f"mode: {'APPLY' if args.apply else 'DRY-RUN'}")
    print(f"dataset: {dataset_root.resolve()}")
    print(f"source:  {source_root.resolve()}")
    print(f"train images before: {len(images)}")
    print(f"val ratio: {args.val_ratio}")
    print(f"val move count: {len(val_images)}")
    print("==================================")

    missing_labels = []
    missing_source = []
    moved_source_total = 0

    for img_path in val_images:
        stem = img_path.stem

        # 1. 이미지 이동
        dst_img_path = img_val_dir / img_path.name
        move_file(img_path, dst_img_path, args.apply)

        # 2. YOLO label txt가 이미 있으면 같이 이동
        label_path = label_train_dir / f"{stem}.txt"
        dst_label_path = label_val_dir / f"{stem}.txt"

        if label_path.exists():
            move_file(label_path, dst_label_path, args.apply)
        else:
            missing_labels.append(f"{stem}.txt")

        # 3. source_label 파일 이동
        moved_source_count = move_source_files(
            stem,
            source_train_dir,
            source_val_dir,
            args.apply
        )

        if moved_source_count == 0:
            missing_source.append(stem)
        else:
            moved_source_total += moved_source_count

    train_after = count_images(img_train_dir)
    val_after = count_images(img_val_dir)

    source_train_count = len(list(source_train_dir.glob("*")))
    source_val_count = len(list(source_val_dir.glob("*")))

    print("\n========== SPLIT RESULT ==========")
    print(f"이동 대상 이미지 수: {len(val_images)}")
    print(f"현재 train 이미지 수: {train_after}")
    print(f"현재 val 이미지 수: {val_after}")
    print(f"이동된 source 파일 수: {moved_source_total}")
    print(f"현재 source/train 파일 수: {source_train_count}")
    print(f"현재 source/val 파일 수: {source_val_count}")
    print(f"YOLO label txt 누락 수: {len(missing_labels)}")
    print(f"source 누락 stem 수: {len(missing_source)}")

    if missing_labels:
        print("\nYOLO label txt 누락 예시:")
        for name in missing_labels[:20]:
            print(name)

    if missing_source:
        print("\nsource 파일 누락 예시:")
        for stem in missing_source[:20]:
            print(stem)

    if not args.apply:
        print("\n현재는 미리보기 모드입니다. 실제 이동하려면 --apply를 붙이세요.")

    print("==================================")


if __name__ == "__main__":
    main()