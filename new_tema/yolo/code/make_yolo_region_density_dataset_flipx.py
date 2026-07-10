from pathlib import Path
import argparse
import csv
import random
import shutil

import cv2
import numpy as np


# ============================================================
# make_yolo_density_region_dataset_flipx.py
#
# 목적:
#   좌우반전 수정이 끝난 기존 4-class YOLO 가상 데이터셋에서
#   실해역처럼 "영역별 포인트 밀도 불균일"만 반영한 데이터셋을 생성한다.
#
# 반드시 지키는 조건:
#   - 실해역 데이터는 학습에 사용하지 않는다.
#   - bbox 라벨은 수정하지 않고 그대로 복사한다.
#   - bbox 확장 없음.
#   - 가림, 타원 삭제, scanline, stripe, 빗살형 결손 없음.
#   - class 0 정사각형과 class 1~3 직사각형의 밀도 유지율을 따로 설정한다.
#
# 기존 단순 density-only와 다른 점:
#   - 픽셀마다 완전 독립 랜덤 제거만 하지 않는다.
#   - bbox 내부에 부드러운 랜덤 밀도장(local density field)을 만든다.
#   - 같은 ID 내부에서도 어떤 영역은 많이 남고, 어떤 영역은 적게 남는다.
#   - 방향성 줄무늬가 아니라 비방향성 영역별 밀도 불균일이다.
# ============================================================


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def ensure_dir(path: Path):
    """폴더 생성"""
    path.mkdir(parents=True, exist_ok=True)


def clean_outputs(paths):
    """출력 폴더 삭제"""
    for p in paths:
        if p is None or str(p).strip() == "":
            continue

        path = Path(p)

        if path.exists():
            shutil.rmtree(str(path))


def collect_images(image_dir: Path):
    """이미지 목록 수집"""
    if not image_dir.exists():
        raise FileNotFoundError(f"이미지 폴더 없음: {image_dir}")

    return sorted([
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ])


def read_yolo_labels(label_path: Path):
    """YOLO txt 라벨 읽기"""
    labels = []

    if not label_path.exists():
        return labels

    with open(label_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            parts = line.strip().split()

            if len(parts) < 5:
                continue

            labels.append({
                "line_idx": line_idx,
                "class_id": int(float(parts[0])),
                "x": float(parts[1]),
                "y": float(parts[2]),
                "w": float(parts[3]),
                "h": float(parts[4]),
                "raw": line.strip(),
            })

    return labels


def yolo_to_xyxy(label, image_w: int, image_h: int):
    """YOLO 정규화 bbox를 pixel xyxy로 변환"""
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

    if x2 <= x1:
        x2 = min(image_w - 1, x1 + 1)

    if y2 <= y1:
        y2 = min(image_h - 1, y1 + 1)

    return x1, y1, x2, y2


def object_mask_from_crop(crop, threshold: int):
    """
    crop 내부에서 실제 포인트가 있는 픽셀만 추출.
    검은 배경은 제외한다.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return gray > int(threshold)


def make_local_density_field(h: int, w: int, rng: random.Random, args):
    """
    방향성 없는 영역별 밀도장을 생성한다.

    원리:
      1. 작은 랜덤 grid 생성
      2. crop 크기로 bicubic resize
      3. Gaussian blur로 부드럽게 만듦
      4. 평균 1 근처로 정규화
      5. base_keep_ratio에 곱해서 픽셀별 keep probability로 사용

    이 방식은 scanline/stripe가 아니라 부드러운 영역별 밀도 차이를 만든다.
    """
    if h <= 1 or w <= 1:
        return np.ones((h, w), dtype=np.float32)

    grid_h = rng.randint(args.region_grid_min, args.region_grid_max)
    grid_w = rng.randint(args.region_grid_min, args.region_grid_max)

    np_rng = np.random.default_rng(rng.randint(0, 2**31 - 1))

    # log-normal을 사용해 영역별 밀도 차이를 자연스럽게 만든다.
    low = np_rng.normal(loc=0.0, scale=args.region_sigma, size=(grid_h, grid_w)).astype(np.float32)
    field = np.exp(low).astype(np.float32)

    field = cv2.resize(field, (w, h), interpolation=cv2.INTER_CUBIC)

    # 방향성 구조가 생기지 않도록 2D Gaussian blur 적용
    blur_k = max(3, int(round(min(h, w) * args.region_blur_ratio)))
    if blur_k % 2 == 0:
        blur_k += 1

    blur_k = min(blur_k, 31)
    if blur_k >= 3:
        field = cv2.GaussianBlur(field, (blur_k, blur_k), 0)

    mean_val = float(np.mean(field))
    if mean_val > 1e-8:
        field = field / mean_val

    # 너무 과도한 지역 밀도 차이 방지
    field = np.clip(field, args.local_density_min, args.local_density_max)

    return field.astype(np.float32)


def apply_region_density_dropout_to_bbox(image, label, rng: random.Random, args):
    """
    bbox 내부 객체 픽셀에 영역별 밀도 불균일 dropout 적용.

    bbox는 그대로 유지하고, 픽셀 제거만 수행한다.
    """
    out = image
    image_h, image_w = out.shape[:2]

    cls = int(label["class_id"])
    x1, y1, x2, y2 = yolo_to_xyxy(label, image_w, image_h)

    crop = out[y1:y2 + 1, x1:x2 + 1].copy()

    if crop.size == 0:
        return out, 0, 0, 1.0, 1.0, 1.0

    obj_mask = object_mask_from_crop(crop, args.pixel_threshold)
    obj_count = int(obj_mask.sum())

    if obj_count <= 0:
        return out, 0, 0, 1.0, 1.0, 1.0

    if cls == 0:
        base_keep = rng.uniform(args.square_keep_min, args.square_keep_max)
    else:
        base_keep = rng.uniform(args.rect_keep_min, args.rect_keep_max)

    h, w = obj_mask.shape[:2]
    density_field = make_local_density_field(h, w, rng, args)

    # 픽셀별 keep probability
    keep_prob = base_keep * density_field
    keep_prob = np.clip(keep_prob, args.keep_prob_min, args.keep_prob_max)

    np_rng = np.random.default_rng(rng.randint(0, 2**31 - 1))
    rand_map = np_rng.random(obj_mask.shape).astype(np.float32)

    keep_mask = rand_map < keep_prob
    remove_mask = obj_mask & (~keep_mask)

    crop[remove_mask] = (0, 0, 0)

    if args.intensity_jitter:
        remain_mask = obj_mask & keep_mask

        if int(remain_mask.sum()) > 0:
            factor = rng.uniform(args.intensity_min, args.intensity_max)
            tmp = crop.astype(np.float32)
            tmp[remain_mask] *= factor
            crop = np.clip(tmp, 0, 255).astype(np.uint8)

    out[y1:y2 + 1, x1:x2 + 1] = crop

    kept_count = int((obj_mask & keep_mask).sum())
    actual_keep = kept_count / max(obj_count, 1)

    return out, obj_count, kept_count, base_keep, actual_keep, float(np.std(density_field))


def draw_labels(image, labels):
    """확인 이미지용 bbox 표시"""
    out = image.copy()
    image_h, image_w = out.shape[:2]

    colors = {
        0: (255, 0, 0),      # square
        1: (0, 255, 255),
        2: (255, 255, 255),
        3: (0, 255, 0),
    }

    for label in labels:
        cls = int(label["class_id"])
        color = colors.get(cls, (0, 0, 255))
        x1, y1, x2, y2 = yolo_to_xyxy(label, image_w, image_h)

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            out,
            f"class {cls}",
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    return out


def make_check_image(original, density_img, labels):
    """원본/영역별 밀도저하 이미지 비교"""
    left = draw_labels(original, labels)
    right = draw_labels(density_img, labels)

    cv2.putText(left, "original", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(right, "region_density", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    return np.concatenate([left, right], axis=1)


def copy_label(src_label: Path, dst_label: Path):
    """라벨 txt 복사"""
    ensure_dir(dst_label.parent)
    shutil.copy2(str(src_label), str(dst_label))


def copy_labels_source_for_stem(src_source_dir: Path, dst_source_dir: Path, src_stem: str, dst_stem: str):
    """labels_source의 uv/meta 파일 복사"""
    ensure_dir(dst_source_dir)

    copied = 0

    pairs = [
        (src_source_dir / f"{src_stem}_top_id_uv.npy", dst_source_dir / f"{dst_stem}_top_id_uv.npy"),
        (src_source_dir / f"{src_stem}_meta.json", dst_source_dir / f"{dst_stem}_meta.json"),
    ]

    for src, dst in pairs:
        if src.exists():
            shutil.copy2(str(src), str(dst))
            copied += 1

    return copied


def write_data_yaml(out_dataset_root: Path):
    """4-class YOLO data.yaml 저장"""
    lines = [
        f"path: {out_dataset_root.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        "",
        "names:",
        "  0: square_id",
        "  1: clockwise_id_1",
        "  2: clockwise_id_2",
        "  3: clockwise_id_3",
    ]

    with open(out_dataset_root / "data.yaml", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def process_one_image(img_path: Path, label_path: Path, split: str, idx: int, variant_idx: int, args):
    """이미지 1장 처리"""
    image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)

    if image is None:
        raise RuntimeError(f"이미지 읽기 실패: {img_path}")

    labels = read_yolo_labels(label_path)

    if len(labels) == 0:
        raise RuntimeError(f"라벨 없음 또는 비어 있음: {label_path}")

    rng_seed = args.seed + idx * 1009 + variant_idx * 9176 + (0 if split == "train" else 100000)
    rng = random.Random(rng_seed)

    out_img = image.copy()
    object_stats = []

    for label in labels:
        out_img, obj_count, kept_count, base_keep, actual_keep, field_std = apply_region_density_dropout_to_bbox(
            out_img,
            label,
            rng,
            args,
        )

        object_stats.append({
            "class_id": int(label["class_id"]),
            "obj_pixels": obj_count,
            "kept_pixels": kept_count,
            "base_keep": base_keep,
            "actual_keep": actual_keep,
            "density_field_std": field_std,
        })

    return image, out_img, labels, object_stats


def process_split(split: str, args, rows):
    """train/val split 처리"""
    src_dataset_root = Path(args.src_dataset_root)
    src_labels_source_root = Path(args.src_labels_source_root) if args.src_labels_source_root else None

    out_dataset_root = Path(args.out_dataset_root)
    out_labels_source_root = Path(args.out_labels_source_root) if args.out_labels_source_root else None
    check_root = Path(args.check_dir)

    src_image_dir = src_dataset_root / "images" / split
    src_label_dir = src_dataset_root / "labels" / split
    src_source_dir = src_labels_source_root / split if src_labels_source_root else None

    out_image_dir = out_dataset_root / "images" / split
    out_label_dir = out_dataset_root / "labels" / split
    out_source_dir = out_labels_source_root / split if out_labels_source_root else None
    check_dir = check_root / split

    ensure_dir(out_image_dir)
    ensure_dir(out_label_dir)
    ensure_dir(check_dir)

    if out_source_dir is not None:
        ensure_dir(out_source_dir)

    images = collect_images(src_image_dir)

    print(f"\n========== {split.upper()} ==========")
    print(f"images: {len(images)}")

    created = 0
    failed = 0
    check_saved = 0

    for idx, img_path in enumerate(images):
        stem = img_path.stem
        label_path = src_label_dir / f"{stem}.txt"

        try:
            for variant_idx in range(args.variants_per_image):
                if args.keep_original_name and args.variants_per_image == 1:
                    out_stem = stem
                else:
                    out_stem = f"{stem}_regiondensity{variant_idx + 1:02d}"

                original, density_img, labels, object_stats = process_one_image(
                    img_path=img_path,
                    label_path=label_path,
                    split=split,
                    idx=idx,
                    variant_idx=variant_idx,
                    args=args,
                )

                out_img_path = out_image_dir / f"{out_stem}.png"
                out_label_path = out_label_dir / f"{out_stem}.txt"

                cv2.imwrite(str(out_img_path), density_img)
                copy_label(label_path, out_label_path)

                copied_source = 0

                if src_source_dir is not None and out_source_dir is not None:
                    copied_source = copy_labels_source_for_stem(
                        src_source_dir=src_source_dir,
                        dst_source_dir=out_source_dir,
                        src_stem=stem,
                        dst_stem=out_stem,
                    )

                if check_saved < args.check_max_per_split:
                    check_img = make_check_image(original, density_img, labels)
                    cv2.imwrite(str(check_dir / f"{out_stem}_check.png"), check_img)
                    check_saved += 1

                class_stats = {}

                for s in object_stats:
                    cid = int(s["class_id"])
                    class_stats[f"class{cid}_base_keep"] = s["base_keep"]
                    class_stats[f"class{cid}_actual_keep"] = s["actual_keep"]
                    class_stats[f"class{cid}_pixels"] = s["obj_pixels"]
                    class_stats[f"class{cid}_kept"] = s["kept_pixels"]
                    class_stats[f"class{cid}_density_field_std"] = s["density_field_std"]

                rows.append({
                    "split": split,
                    "src_stem": stem,
                    "out_stem": out_stem,
                    "variant": variant_idx + 1,
                    "status": "ok",
                    "labels": len(labels),
                    "copied_source_files": copied_source,
                    **class_stats,
                })

                created += 1

            print(f"[OK] {split} {idx + 1}/{len(images)} {stem}")

        except Exception as e:
            failed += 1
            rows.append({
                "split": split,
                "src_stem": stem,
                "out_stem": "",
                "variant": "",
                "status": f"fail: {e}",
            })
            print(f"[FAIL] {split} {idx + 1}/{len(images)} {stem}: {e}")

    print(f"[{split}] created={created}, failed={failed}, check_saved={check_saved}")

    return created, failed


def save_summary(rows, check_dir: Path):
    """summary csv 저장"""
    ensure_dir(check_dir)

    if len(rows) == 0:
        return

    path = check_dir / "region_density_generation_summary.csv"
    fields = sorted(set().union(*[row.keys() for row in rows]))

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--src_dataset_root", type=str, required=True, help="기존 flipx YOLO dataset")
    parser.add_argument("--src_labels_source_root", type=str, default="", help="기존 labels_source 경로")

    parser.add_argument("--out_dataset_root", type=str, required=True, help="출력 region density YOLO dataset")
    parser.add_argument("--out_labels_source_root", type=str, default="", help="출력 labels_source 경로")
    parser.add_argument("--check_dir", type=str, required=True, help="check 이미지 저장 폴더")

    parser.add_argument("--variants_per_image", type=int, default=1)
    parser.add_argument("--keep_original_name", action="store_true", help="variants_per_image=1일 때 원본 파일명 유지")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check_max_per_split", type=int, default=40)
    parser.add_argument("--clean", action="store_true")

    # class별 기본 밀도 유지율
    parser.add_argument("--square_keep_min", type=float, default=0.35)
    parser.add_argument("--square_keep_max", type=float, default=0.70)
    parser.add_argument("--rect_keep_min", type=float, default=0.60)
    parser.add_argument("--rect_keep_max", type=float, default=0.88)

    # 영역별 밀도장 파라미터
    parser.add_argument("--region_grid_min", type=int, default=3)
    parser.add_argument("--region_grid_max", type=int, default=6)
    parser.add_argument("--region_sigma", type=float, default=0.55)
    parser.add_argument("--region_blur_ratio", type=float, default=0.08)
    parser.add_argument("--local_density_min", type=float, default=0.35)
    parser.add_argument("--local_density_max", type=float, default=1.85)

    # 최종 keep probability 제한
    parser.add_argument("--keep_prob_min", type=float, default=0.05)
    parser.add_argument("--keep_prob_max", type=float, default=0.98)

    # 객체 픽셀 기준
    parser.add_argument("--pixel_threshold", type=int, default=8)

    # 밝기 조정은 기본 비활성
    parser.add_argument("--intensity_jitter", action="store_true")
    parser.add_argument("--intensity_min", type=float, default=0.85)
    parser.add_argument("--intensity_max", type=float, default=1.15)

    args = parser.parse_args()

    if args.clean:
        clean_outputs([
            args.out_dataset_root,
            args.out_labels_source_root,
            args.check_dir,
        ])

    ensure_dir(Path(args.out_dataset_root))
    ensure_dir(Path(args.check_dir))

    if args.out_labels_source_root:
        ensure_dir(Path(args.out_labels_source_root))

    print("========== CONFIG ==========")
    print(f"src_dataset_root:       {args.src_dataset_root}")
    print(f"src_labels_source_root: {args.src_labels_source_root}")
    print(f"out_dataset_root:       {args.out_dataset_root}")
    print(f"out_labels_source_root: {args.out_labels_source_root}")
    print(f"check_dir:              {args.check_dir}")
    print(f"variants_per_image:     {args.variants_per_image}")
    print(f"square_keep:            {args.square_keep_min} ~ {args.square_keep_max}")
    print(f"rect_keep:              {args.rect_keep_min} ~ {args.rect_keep_max}")
    print(f"region_grid:            {args.region_grid_min} ~ {args.region_grid_max}")
    print(f"region_sigma:           {args.region_sigma}")
    print("real data used:         False")
    print("bbox changed:           False")
    print("occlusion/erase used:   False")
    print("scanline/stripe used:   False")
    print("============================")

    rows = []
    total_created = 0
    total_failed = 0

    for split in ["train", "val"]:
        created, failed = process_split(split, args, rows)
        total_created += created
        total_failed += failed

    write_data_yaml(Path(args.out_dataset_root))
    save_summary(rows, Path(args.check_dir))

    print("\n========== TOTAL RESULT ==========")
    print(f"created: {total_created}")
    print(f"failed:  {total_failed}")
    print(f"summary: {Path(args.check_dir) / 'region_density_generation_summary.csv'}")
    print("==================================")


if __name__ == "__main__":
    main()
