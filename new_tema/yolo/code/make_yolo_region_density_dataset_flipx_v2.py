from pathlib import Path
import argparse
import csv
import random
import shutil

import cv2
import numpy as np


# ============================================================
# make_yolo_region_density_dataset_flipx_v2.py
#
# 목적:
#   flipx 수정이 끝난 YOLO 가상 데이터셋에서 실해역과 비슷한
#   "영역별 포인트 밀도 불균일" 이미지를 생성한다.
#
# 핵심 반영:
#   1. class 0 정사각형 ID
#      - 중심부는 점처럼 비교적 밀집
#      - 주변부는 강하게 누락
#
#   2. class 1~3 직사각형 ID
#      - 전체 형상은 유지
#      - 내부 일부 구간은 밀도가 낮아져 끊기거나 누락된 것처럼 보임
#
# 금지:
#   - 실해역 데이터 학습 사용 없음
#   - bbox 변경 없음
#   - 라벨 변경 없음
#   - bbox 확장 없음
#   - scanline/stripe/빗살형 결손 없음
#   - 큰 타원형 가림 없음
# ============================================================


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def clean_outputs(paths):
    for p in paths:
        if p is None or str(p).strip() == "":
            continue
        path = Path(p)
        if path.exists():
            shutil.rmtree(str(path))


def collect_images(image_dir: Path):
    if not image_dir.exists():
        raise FileNotFoundError(f"이미지 폴더 없음: {image_dir}")

    return sorted([
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ])


def read_yolo_labels(label_path: Path):
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
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return gray > int(threshold)


def make_region_field(h: int, w: int, rng: random.Random, sigma: float,
                      grid_min: int, grid_max: int, blur_ratio: float,
                      local_min: float, local_max: float):
    """
    방향성 없는 영역별 밀도장 생성.
    줄무늬가 아니라 blob/patch 형태의 밀도 차이를 만든다.
    """
    if h <= 1 or w <= 1:
        return np.ones((h, w), dtype=np.float32)

    grid_h = rng.randint(grid_min, grid_max)
    grid_w = rng.randint(grid_min, grid_max)

    np_rng = np.random.default_rng(rng.randint(0, 2**31 - 1))
    raw = np_rng.normal(loc=0.0, scale=sigma, size=(grid_h, grid_w)).astype(np.float32)
    field = np.exp(raw).astype(np.float32)

    field = cv2.resize(field, (w, h), interpolation=cv2.INTER_CUBIC)

    blur_k = max(3, int(round(min(h, w) * blur_ratio)))
    if blur_k % 2 == 0:
        blur_k += 1
    blur_k = min(blur_k, 31)

    if blur_k >= 3:
        field = cv2.GaussianBlur(field, (blur_k, blur_k), 0)

    mean_val = float(np.mean(field))
    if mean_val > 1e-8:
        field = field / mean_val

    field = np.clip(field, local_min, local_max)
    return field.astype(np.float32)


def make_square_center_field(h: int, w: int, rng: random.Random, args):
    """
    정사각형 ID용 밀도장.
    중심부는 강하게 남기고 주변부는 낮게 만든다.
    """
    if h <= 1 or w <= 1:
        return np.ones((h, w), dtype=np.float32)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    cx = (w - 1) * (0.5 + rng.uniform(-args.square_center_jitter, args.square_center_jitter))
    cy = (h - 1) * (0.5 + rng.uniform(-args.square_center_jitter, args.square_center_jitter))

    # 정사각형 내부 중심부가 점처럼 밀집되도록 Gaussian 중심 밀도 생성
    sx = max(1.0, w * rng.uniform(args.square_center_sigma_min, args.square_center_sigma_max))
    sy = max(1.0, h * rng.uniform(args.square_center_sigma_min, args.square_center_sigma_max))

    gaussian = np.exp(-(((xx - cx) ** 2) / (2 * sx * sx) + ((yy - cy) ** 2) / (2 * sy * sy))).astype(np.float32)

    # 중심부 keep은 높고, 외곽 keep은 낮음
    center_keep = rng.uniform(args.square_center_keep_min, args.square_center_keep_max)
    outer_keep = rng.uniform(args.square_outer_keep_min, args.square_outer_keep_max)

    field = outer_keep + (center_keep - outer_keep) * gaussian

    # 중심부도 완전히 매끈하지 않게 약한 지역 밀도장 곱함
    local = make_region_field(
        h=h,
        w=w,
        rng=rng,
        sigma=args.square_region_sigma,
        grid_min=args.square_region_grid_min,
        grid_max=args.square_region_grid_max,
        blur_ratio=args.region_blur_ratio,
        local_min=args.local_density_min,
        local_max=args.local_density_max,
    )

    field = field * local
    field = np.clip(field, args.keep_prob_min, args.keep_prob_max)

    return field.astype(np.float32)


def make_rect_region_field(h: int, w: int, rng: random.Random, args):
    """
    직사각형 ID용 밀도장.
    방향성 줄무늬 없이 영역별로 포인트가 많이 남는 곳/적게 남는 곳을 만든다.
    """
    base_keep = rng.uniform(args.rect_keep_min, args.rect_keep_max)

    local = make_region_field(
        h=h,
        w=w,
        rng=rng,
        sigma=args.rect_region_sigma,
        grid_min=args.rect_region_grid_min,
        grid_max=args.rect_region_grid_max,
        blur_ratio=args.region_blur_ratio,
        local_min=args.local_density_min,
        local_max=args.local_density_max,
    )

    field = base_keep * local

    # 너무 균일한 경우를 줄이기 위해 아주 낮은 밀도 patch를 몇 개 추가
    # 큰 가림이 아니라, 내부 일부가 성기게 보이도록 keep probability만 낮춤
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    patch_count = rng.randint(args.rect_low_patch_min, args.rect_low_patch_max)

    for _ in range(patch_count):
        pcx = rng.uniform(0.15, 0.85) * max(1, w - 1)
        pcy = rng.uniform(0.15, 0.85) * max(1, h - 1)

        psx = max(1.0, w * rng.uniform(args.rect_patch_sigma_min, args.rect_patch_sigma_max))
        psy = max(1.0, h * rng.uniform(args.rect_patch_sigma_min, args.rect_patch_sigma_max))

        patch = np.exp(-(((xx - pcx) ** 2) / (2 * psx * psx) + ((yy - pcy) ** 2) / (2 * psy * psy))).astype(np.float32)
        reduce_strength = rng.uniform(args.rect_patch_reduce_min, args.rect_patch_reduce_max)

        field = field * (1.0 - reduce_strength * patch)

    field = np.clip(field, args.keep_prob_min, args.keep_prob_max)

    return field.astype(np.float32)


def apply_density_to_bbox(image, label, rng: random.Random, args):
    """
    bbox 내부 객체 픽셀에 class별 밀도장 적용.
    """
    out = image
    image_h, image_w = out.shape[:2]

    cls = int(label["class_id"])
    x1, y1, x2, y2 = yolo_to_xyxy(label, image_w, image_h)

    crop = out[y1:y2 + 1, x1:x2 + 1].copy()

    if crop.size == 0:
        return out, 0, 0, 1.0

    obj_mask = object_mask_from_crop(crop, args.pixel_threshold)
    obj_count = int(obj_mask.sum())

    if obj_count <= 0:
        return out, 0, 0, 1.0

    h, w = obj_mask.shape[:2]

    if cls == 0:
        keep_prob = make_square_center_field(h, w, rng, args)
    else:
        keep_prob = make_rect_region_field(h, w, rng, args)

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

    return out, obj_count, kept_count, actual_keep


def draw_labels(image, labels):
    out = image.copy()
    image_h, image_w = out.shape[:2]

    colors = {
        0: (255, 0, 0),
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
    left = draw_labels(original, labels)
    right = draw_labels(density_img, labels)

    cv2.putText(left, "original", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(right, "density_v2", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    return np.concatenate([left, right], axis=1)


def copy_label(src_label: Path, dst_label: Path):
    ensure_dir(dst_label.parent)
    shutil.copy2(str(src_label), str(dst_label))


def copy_labels_source_for_stem(src_source_dir: Path, dst_source_dir: Path, src_stem: str, dst_stem: str):
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
        out_img, obj_count, kept_count, actual_keep = apply_density_to_bbox(
            out_img,
            label,
            rng,
            args,
        )

        object_stats.append({
            "class_id": int(label["class_id"]),
            "obj_pixels": obj_count,
            "kept_pixels": kept_count,
            "actual_keep": actual_keep,
        })

    return image, out_img, labels, object_stats


def process_split(split: str, args, rows):
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
                    out_stem = f"{stem}_densityv2_{variant_idx + 1:02d}"

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
                    class_stats[f"class{cid}_actual_keep"] = s["actual_keep"]
                    class_stats[f"class{cid}_pixels"] = s["obj_pixels"]
                    class_stats[f"class{cid}_kept"] = s["kept_pixels"]

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
    ensure_dir(check_dir)

    if len(rows) == 0:
        return

    path = check_dir / "region_density_v2_generation_summary.csv"
    fields = sorted(set().union(*[row.keys() for row in rows]))

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--src_dataset_root", type=str, required=True)
    parser.add_argument("--src_labels_source_root", type=str, default="")

    parser.add_argument("--out_dataset_root", type=str, required=True)
    parser.add_argument("--out_labels_source_root", type=str, default="")
    parser.add_argument("--check_dir", type=str, required=True)

    parser.add_argument("--variants_per_image", type=int, default=1)
    parser.add_argument("--keep_original_name", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check_max_per_split", type=int, default=40)
    parser.add_argument("--clean", action="store_true")

    # square: 중심 점 밀집 + 주변 누락
    parser.add_argument("--square_center_keep_min", type=float, default=0.75)
    parser.add_argument("--square_center_keep_max", type=float, default=0.98)
    parser.add_argument("--square_outer_keep_min", type=float, default=0.08)
    parser.add_argument("--square_outer_keep_max", type=float, default=0.35)
    parser.add_argument("--square_center_sigma_min", type=float, default=0.18)
    parser.add_argument("--square_center_sigma_max", type=float, default=0.32)
    parser.add_argument("--square_center_jitter", type=float, default=0.10)
    parser.add_argument("--square_region_sigma", type=float, default=0.35)
    parser.add_argument("--square_region_grid_min", type=int, default=3)
    parser.add_argument("--square_region_grid_max", type=int, default=5)

    # rect: 형상 유지 + 내부 구간별 밀도 불균일/부분 끊김
    parser.add_argument("--rect_keep_min", type=float, default=0.50)
    parser.add_argument("--rect_keep_max", type=float, default=0.82)
    parser.add_argument("--rect_region_sigma", type=float, default=0.70)
    parser.add_argument("--rect_region_grid_min", type=int, default=3)
    parser.add_argument("--rect_region_grid_max", type=int, default=6)
    parser.add_argument("--rect_low_patch_min", type=int, default=1)
    parser.add_argument("--rect_low_patch_max", type=int, default=3)
    parser.add_argument("--rect_patch_sigma_min", type=float, default=0.12)
    parser.add_argument("--rect_patch_sigma_max", type=float, default=0.28)
    parser.add_argument("--rect_patch_reduce_min", type=float, default=0.25)
    parser.add_argument("--rect_patch_reduce_max", type=float, default=0.65)

    # common
    parser.add_argument("--region_blur_ratio", type=float, default=0.08)
    parser.add_argument("--local_density_min", type=float, default=0.35)
    parser.add_argument("--local_density_max", type=float, default=1.85)
    parser.add_argument("--keep_prob_min", type=float, default=0.03)
    parser.add_argument("--keep_prob_max", type=float, default=0.98)
    parser.add_argument("--pixel_threshold", type=int, default=8)

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
    print(f"out_dataset_root:       {args.out_dataset_root}")
    print(f"check_dir:              {args.check_dir}")
    print(f"variants_per_image:     {args.variants_per_image}")
    print("square: center dense, outer sparse")
    print("rect: region density uneven, no stripe")
    print("real data used:         False")
    print("bbox changed:           False")
    print("scanline/stripe used:   False")
    print("occlusion/erase used:   False")
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
    print(f"summary: {Path(args.check_dir) / 'region_density_v2_generation_summary.csv'}")
    print("==================================")


if __name__ == "__main__":
    main()
