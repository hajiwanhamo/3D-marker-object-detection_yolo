from pathlib import Path
import argparse
import json
import math
import random
import shutil

import cv2
import numpy as np


# ============================================================
# 가상데이터 projection 결과만 사용해서 YOLO 데이터셋 생성
#
# 핵심 조건:
#   - 실해역 데이터 절대 사용하지 않음
#   - 가상데이터 projection 폴더 하나만 입력
#   - 입력 가상데이터를 8:2 비율로 train/val 분할
#   - YOLO 이미지는 *_color.png만 사용
#   - 라벨 생성용 *_top_id_uv.npy, *_meta.json은 labels_source에 저장
#   - 라벨은 flipx가 반영된 uv/meta 기준으로 생성
#   - 여기서는 추가 좌우반전하지 않음
#
# 출력:
#   yolo_dataset/images/train/*.png
#   yolo_dataset/images/val/*.png
#   yolo_dataset/labels/train/*.txt
#   yolo_dataset/labels/val/*.txt
#   labels_source/train/*_top_id_uv.npy
#   labels_source/train/*_meta.json
#   labels_source/val/*_top_id_uv.npy
#   labels_source/val/*_meta.json
#   label_check/train/*_label_check.png
#   label_check/val/*_label_check.png
#
# label_mode:
#   2class:
#       0 = square_id
#       1 = rect_id
#
#   4class:
#       0 = square_id
#       1 = square 기준 시계방향 첫 번째 rect
#       2 = square 기준 시계방향 두 번째 rect
#       3 = square 기준 시계방향 세 번째 rect
# ============================================================


def load_meta(meta_path: Path) -> dict:
    """meta.json 읽기"""
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_stem_from_color(color_path: Path):
    """
    *_color.png에서 YOLO용 기본 파일명 추출.

    예:
      aug_000000_marker_color.png -> yolo_stem=aug_000000, raw_stem=aug_000000_marker
      aug_000000_color.png        -> yolo_stem=aug_000000, raw_stem=aug_000000
    """
    stem = color_path.stem

    if stem.endswith("_color"):
        raw_stem = stem[:-len("_color")]
    else:
        raw_stem = stem

    if raw_stem.endswith("_marker"):
        yolo_stem = raw_stem[:-len("_marker")]
    else:
        yolo_stem = raw_stem

    return yolo_stem, raw_stem


def find_uv_meta(proj_dir: Path, yolo_stem: str, raw_stem: str):
    """
    projection 결과 폴더에서 top_id_uv/meta 파일 찾기.
    """
    uv_candidates = [
        proj_dir / f"{raw_stem}_top_id_uv.npy",
        proj_dir / f"{yolo_stem}_top_id_uv.npy",
        proj_dir / f"{yolo_stem}_marker_top_id_uv.npy",
    ]

    meta_candidates = [
        proj_dir / f"{raw_stem}_meta.json",
        proj_dir / f"{yolo_stem}_meta.json",
        proj_dir / f"{yolo_stem}_marker_meta.json",
    ]

    uv_path = None
    meta_path = None

    for p in uv_candidates:
        if p.exists():
            uv_path = p
            break

    for p in meta_candidates:
        if p.exists():
            meta_path = p
            break

    if uv_path is None:
        raise FileNotFoundError(f"top_id_uv 파일 없음: yolo_stem={yolo_stem}, raw_stem={raw_stem}")

    if meta_path is None:
        raise FileNotFoundError(f"meta 파일 없음: yolo_stem={yolo_stem}, raw_stem={raw_stem}")

    return uv_path, meta_path


def scan_projection_items(src_proj: Path):
    """
    가상데이터 projection 폴더에서 *_color.png 기준으로 샘플 목록 생성.
    """
    if not src_proj.exists():
        raise FileNotFoundError(f"src_proj 폴더 없음: {src_proj}")

    color_files = sorted(src_proj.glob("*_color.png"))

    if len(color_files) == 0:
        raise RuntimeError(f"*_color.png 파일 없음: {src_proj}")

    items = []

    for color_path in color_files:
        yolo_stem, raw_stem = normalize_stem_from_color(color_path)
        uv_path, meta_path = find_uv_meta(src_proj, yolo_stem, raw_stem)

        items.append({
            "yolo_stem": yolo_stem,
            "raw_stem": raw_stem,
            "color_path": color_path,
            "uv_path": uv_path,
            "meta_path": meta_path,
        })

    return items


def split_items(items, train_ratio: float, seed: int):
    """
    가상데이터 전체를 train/val로 8:2 분할.
    """
    items = list(items)
    rng = random.Random(seed)
    rng.shuffle(items)

    n_total = len(items)
    n_train = int(round(n_total * train_ratio))

    if n_train <= 0 or n_train >= n_total:
        raise RuntimeError(f"분할 불가: total={n_total}, train={n_train}")

    train_items = items[:n_train]
    val_items = items[n_train:]

    train_items = sorted(train_items, key=lambda x: x["yolo_stem"])
    val_items = sorted(val_items, key=lambda x: x["yolo_stem"])

    return train_items, val_items


def load_uv(uv_path: Path):
    """
    top_id_uv.npy 읽기.
    앞 2열만 사용.
    """
    uv = np.load(str(uv_path))

    if uv.ndim != 2 or uv.shape[1] < 2:
        raise RuntimeError(f"uv 파일 형식 오류: {uv_path}, shape={uv.shape}")

    uv = uv[:, :2].astype(np.float64)

    valid = np.isfinite(uv[:, 0]) & np.isfinite(uv[:, 1])
    uv = uv[valid]

    if len(uv) == 0:
        raise RuntimeError(f"유효한 uv 포인트 없음: {uv_path}")

    return uv


def uv_to_pixel(uv: np.ndarray, meta: dict):
    """
    uv 좌표를 image pixel 좌표로 변환.

    중요:
      projection 단계에서 이미 flip_x가 uv/meta에 반영되어 있어야 함.
      여기서는 추가 좌우반전하지 않음.

    공식:
      px = (u - u_min) / pixel_size_u_m
      py = (v_max - v) / pixel_size_v_m
    """
    u = uv[:, 0]
    v = uv[:, 1]

    u_min = float(meta["u_min"])
    v_max = float(meta["v_max"])
    pixel_size_u_m = float(meta["pixel_size_u_m"])
    pixel_size_v_m = float(meta["pixel_size_v_m"])

    px = (u - u_min) / pixel_size_u_m
    py = (v_max - v) / pixel_size_v_m

    return np.stack([px, py], axis=1)


def make_point_mask(pixel_xy: np.ndarray, image_size: int, point_radius: int):
    """
    pixel 좌표로 내부 ID 포인트 마스크 생성.
    """
    mask = np.zeros((image_size, image_size), dtype=np.uint8)

    px = np.round(pixel_xy[:, 0]).astype(np.int32)
    py = np.round(pixel_xy[:, 1]).astype(np.int32)

    valid = (
        (px >= 0) & (px < image_size) &
        (py >= 0) & (py < image_size)
    )

    px = px[valid]
    py = py[valid]

    for x, y in zip(px, py):
        cv2.circle(mask, (int(x), int(y)), int(point_radius), 255, -1)

    return mask, int(valid.sum()), int(len(valid) - valid.sum())


def clean_mask(mask: np.ndarray, close_kernel: int, dilate_iter: int):
    """
    점 형태 마스크를 component로 묶기 위한 morphology 처리.
    """
    k = int(close_kernel)

    if k < 1:
        return mask

    kernel = np.ones((k, k), dtype=np.uint8)
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    if int(dilate_iter) > 0:
        cleaned = cv2.dilate(cleaned, kernel, iterations=int(dilate_iter))

    return cleaned


def extract_components(mask: np.ndarray, min_area: int):
    """
    연결 성분 추출.
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    components = []

    for label_id in range(1, num_labels):
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        w = int(stats[label_id, cv2.CC_STAT_WIDTH])
        h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        cx, cy = centroids[label_id]

        if area < int(min_area):
            continue

        aspect = max(w / max(h, 1), h / max(w, 1))

        components.append({
            "label_id": int(label_id),
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "area": area,
            "cx": float(cx),
            "cy": float(cy),
            "aspect": float(aspect),
        })

    components = sorted(components, key=lambda c: c["area"], reverse=True)
    return components


def select_four_components(components):
    """
    내부 ID 4개 component 선택.
    기본은 면적 상위 4개.
    """
    if len(components) < 4:
        raise RuntimeError(f"내부 ID component가 4개보다 적습니다. 현재 개수: {len(components)}")

    return components[:4]


def assign_classes(components, label_mode: str):
    """
    component에 class 부여.

    square:
      aspect가 1에 가장 가까운 component를 class 0으로 선택.

    rect:
      square 기준 시계방향 순서로 정렬.

    label_mode=2class:
      square=0, rect=1

    label_mode=4class:
      square=0, rect=1,2,3
    """
    if len(components) != 4:
        raise RuntimeError(f"assign_classes에는 component 4개가 필요합니다. 현재 {len(components)}개")

    square_idx = int(np.argmin([abs(c["aspect"] - 1.0) for c in components]))
    square = components[square_idx]

    rects = [c for i, c in enumerate(components) if i != square_idx]

    center_x = float(np.mean([c["cx"] for c in components]))
    center_y = float(np.mean([c["cy"] for c in components]))

    square_angle = math.atan2(square["cy"] - center_y, square["cx"] - center_x) % (2.0 * math.pi)

    rect_pairs = []

    for r in rects:
        angle = math.atan2(r["cy"] - center_y, r["cx"] - center_x) % (2.0 * math.pi)
        rel_angle = (angle - square_angle) % (2.0 * math.pi)
        rect_pairs.append((rel_angle, r))

    rect_pairs = sorted(rect_pairs, key=lambda x: x[0])

    assigned = []

    square_out = dict(square)
    square_out["class_id"] = 0
    assigned.append(square_out)

    for idx, (_, r) in enumerate(rect_pairs):
        rect_out = dict(r)

        if label_mode == "2class":
            rect_out["class_id"] = 1
        else:
            rect_out["class_id"] = idx + 1

        assigned.append(rect_out)

    return assigned


def bbox_to_yolo(comp, image_w: int, image_h: int, expand_px: int):
    """
    component bbox를 YOLO txt 형식으로 변환.
    """
    x1 = int(comp["x"]) - int(expand_px)
    y1 = int(comp["y"]) - int(expand_px)
    x2 = int(comp["x"]) + int(comp["w"]) + int(expand_px)
    y2 = int(comp["y"]) + int(comp["h"]) + int(expand_px)

    x1 = max(0, min(image_w - 1, x1))
    y1 = max(0, min(image_h - 1, y1))
    x2 = max(0, min(image_w - 1, x2))
    y2 = max(0, min(image_h - 1, y2))

    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)

    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0

    return (
        cx / image_w,
        cy / image_h,
        bw / image_w,
        bh / image_h,
        (x1, y1, x2, y2),
    )


def draw_label_check(image: np.ndarray, assigned, image_w: int, image_h: int, expand_px: int):
    """
    라벨 확인 이미지 생성.
    """
    vis = image.copy()

    colors = {
        0: (255, 0, 0),      # square: blue
        1: (0, 255, 255),    # rect: yellow
        2: (255, 255, 255),
        3: (0, 255, 0),
    }

    for comp in assigned:
        class_id = int(comp["class_id"])
        _, _, _, _, box = bbox_to_yolo(comp, image_w, image_h, expand_px)
        x1, y1, x2, y2 = box

        color = colors.get(class_id, (0, 255, 0))

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.circle(vis, (int(comp["cx"]), int(comp["cy"])), 4, color, -1)

        cv2.putText(
            vis,
            f"class {class_id}",
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    return vis


def copy_color_image_as_rgb(src_color: Path, dst_image: Path):
    """
    YOLO 입력용 컬러 이미지만 저장.
    """
    image = cv2.imread(str(src_color), cv2.IMREAD_COLOR)

    if image is None:
        raise RuntimeError(f"컬러 이미지 읽기 실패: {src_color}")

    dst_image.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst_image), image)

    return image


def write_label_txt(label_path: Path, assigned, image_w: int, image_h: int, expand_px: int):
    """
    YOLO label txt 저장.
    """
    lines = []

    for comp in assigned:
        x_center, y_center, width, height, _ = bbox_to_yolo(
            comp,
            image_w,
            image_h,
            expand_px,
        )

        class_id = int(comp["class_id"])

        lines.append(
            f"{class_id} "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{width:.6f} "
            f"{height:.6f}"
        )

    label_path.parent.mkdir(parents=True, exist_ok=True)

    with open(label_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_data_yaml(dataset_root: Path, label_mode: str):
    """
    YOLO data.yaml 생성.
    """
    if label_mode == "2class":
        names = {
            0: "square_id",
            1: "rect_id",
        }
    else:
        names = {
            0: "square_id",
            1: "clockwise_id_1",
            2: "clockwise_id_2",
            3: "clockwise_id_3",
        }

    lines = [
        f"path: {dataset_root.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        "",
        "names:",
    ]

    for k, v in names.items():
        lines.append(f"  {k}: {v}")

    yaml_path = dataset_root / "data.yaml"

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[YAML] {yaml_path}")


def clean_output_dirs(dataset_root: Path, labels_source_root: Path, label_check_root: Path):
    """
    기존 출력 제거.
    입력 projection 폴더는 절대 삭제하지 않음.
    """
    for p in [dataset_root, labels_source_root, label_check_root]:
        if p.exists():
            shutil.rmtree(str(p))


def process_items(split: str, items, dataset_root: Path, labels_source_root: Path,
                  label_check_root: Path, args):
    """
    train 또는 val item 처리.
    """
    image_out_dir = dataset_root / "images" / split
    label_out_dir = dataset_root / "labels" / split
    source_out_dir = labels_source_root / split
    check_out_dir = label_check_root / split

    image_out_dir.mkdir(parents=True, exist_ok=True)
    label_out_dir.mkdir(parents=True, exist_ok=True)
    source_out_dir.mkdir(parents=True, exist_ok=True)
    check_out_dir.mkdir(parents=True, exist_ok=True)

    success = 0
    failed = 0
    failure_lines = []

    print(f"\n========== {split.upper()} ==========")
    print(f"items: {len(items)}")

    for idx, item in enumerate(items):
        yolo_stem = item["yolo_stem"]
        color_path = item["color_path"]
        uv_path = item["uv_path"]
        meta_path = item["meta_path"]

        try:
            meta = load_meta(meta_path)
            image_size = int(meta.get("image_size", args.image_size))

            # 1. YOLO 학습용 컬러 이미지만 images에 저장
            dst_image = image_out_dir / f"{yolo_stem}.png"
            image = copy_color_image_as_rgb(color_path, dst_image)

            image_h, image_w = image.shape[:2]

            if image_w != image_h:
                raise RuntimeError(f"정사각형 이미지가 아닙니다. image_w={image_w}, image_h={image_h}")

            if image_w != image_size:
                print(f"[WARN] {yolo_stem}: meta image_size={image_size}, 실제 image_size={image_w}")

            # 2. 라벨 생성용 source 파일을 labels_source에 저장
            dst_uv = source_out_dir / f"{yolo_stem}_top_id_uv.npy"
            dst_meta = source_out_dir / f"{yolo_stem}_meta.json"

            shutil.copy2(str(uv_path), str(dst_uv))
            shutil.copy2(str(meta_path), str(dst_meta))

            # 3. uv/meta 기준으로 label 생성
            uv = load_uv(uv_path)
            pixel_xy = uv_to_pixel(uv, meta)

            raw_mask, valid_count, invalid_count = make_point_mask(
                pixel_xy,
                image_w,
                args.point_radius,
            )

            cleaned_mask = clean_mask(
                raw_mask,
                args.close_kernel,
                args.dilate_iter,
            )

            components = extract_components(
                cleaned_mask,
                args.min_area,
            )

            selected = select_four_components(components)
            assigned = assign_classes(selected, args.label_mode)

            # 4. YOLO txt 저장
            label_path = label_out_dir / f"{yolo_stem}.txt"

            write_label_txt(
                label_path,
                assigned,
                image_w,
                image_h,
                args.bbox_expand_px,
            )

            # 5. label check 저장
            check_img = draw_label_check(
                image,
                assigned,
                image_w,
                image_h,
                args.bbox_expand_px,
            )

            check_path = check_out_dir / f"{yolo_stem}_label_check.png"
            mask_path = check_out_dir / f"{yolo_stem}_mask_check.png"

            cv2.imwrite(str(check_path), check_img)
            cv2.imwrite(str(mask_path), cleaned_mask)

            success += 1

            print(
                f"[OK] {split} {idx + 1}/{len(items)} {yolo_stem} | "
                f"components={len(components)} | valid_uv={valid_count}"
            )

        except Exception as e:
            failed += 1
            msg = f"[FAIL] {split} {idx + 1}/{len(items)} {yolo_stem}: {e}"
            failure_lines.append(msg)
            print(msg)

            # 실패 시 불완전 파일 제거
            for p in [
                image_out_dir / f"{yolo_stem}.png",
                label_out_dir / f"{yolo_stem}.txt",
                source_out_dir / f"{yolo_stem}_top_id_uv.npy",
                source_out_dir / f"{yolo_stem}_meta.json",
                check_out_dir / f"{yolo_stem}_label_check.png",
                check_out_dir / f"{yolo_stem}_mask_check.png",
            ]:
                if p.exists():
                    p.unlink()

    return success, failed, failure_lines


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--src_proj",
        type=str,
        required=True,
        help="가상데이터 projection 결과 폴더. 이 폴더 하나를 8:2로 train/val 분할함.",
    )

    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--labels_source_root", type=str, required=True)
    parser.add_argument("--label_check_root", type=str, required=True)

    parser.add_argument(
        "--label_mode",
        choices=["2class", "4class"],
        default="2class",
    )

    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--point_radius", type=int, default=2)
    parser.add_argument("--close_kernel", type=int, default=5)
    parser.add_argument("--dilate_iter", type=int, default=1)
    parser.add_argument("--min_area", type=int, default=30)
    parser.add_argument("--bbox_expand_px", type=int, default=5)

    parser.add_argument(
        "--clean",
        action="store_true",
        help="기존 dataset_root, labels_source_root, label_check_root 삭제 후 재생성",
    )

    parser.add_argument(
        "--allow_real_word",
        action="store_true",
        help="경로에 real 단어가 포함되어도 실행 허용. 기본은 차단.",
    )

    args = parser.parse_args()

    src_proj = Path(args.src_proj)
    dataset_root = Path(args.dataset_root)
    labels_source_root = Path(args.labels_source_root)
    label_check_root = Path(args.label_check_root)

    # 실해역 데이터가 학습 데이터셋에 들어가는 사고 방지
    if "real" in str(src_proj).lower() and not args.allow_real_word:
        raise RuntimeError(
            "src_proj 경로에 'real' 단어가 포함되어 있습니다. "
            "학습용 yolo_dataset에는 실해역 데이터가 들어가면 안 됩니다. "
            "정말 가상데이터 경로가 맞다면 --allow_real_word를 붙여 실행하세요."
        )

    print("========== CONFIG ==========")
    print(f"src_proj:           {src_proj}")
    print(f"dataset_root:       {dataset_root}")
    print(f"labels_source_root: {labels_source_root}")
    print(f"label_check_root:   {label_check_root}")
    print(f"label_mode:         {args.label_mode}")
    print(f"train_ratio:        {args.train_ratio}")
    print(f"seed:               {args.seed}")
    print("split:              single virtual source -> train/val = 8:2")
    print("image source:       *_color.png only")
    print("label source:       *_top_id_uv.npy + *_meta.json")
    print("extra flip:         False")
    print("============================")

    if args.clean:
        clean_output_dirs(dataset_root, labels_source_root, label_check_root)

    dataset_root.mkdir(parents=True, exist_ok=True)
    labels_source_root.mkdir(parents=True, exist_ok=True)
    label_check_root.mkdir(parents=True, exist_ok=True)

    items = scan_projection_items(src_proj)
    train_items, val_items = split_items(items, args.train_ratio, args.seed)

    print("\n========== SPLIT ==========")
    print(f"total: {len(items)}")
    print(f"train: {len(train_items)}")
    print(f"val:   {len(val_items)}")
    print("===========================")

    total_success = 0
    total_failed = 0
    all_failures = []

    s, f, failures = process_items(
        split="train",
        items=train_items,
        dataset_root=dataset_root,
        labels_source_root=labels_source_root,
        label_check_root=label_check_root,
        args=args,
    )

    total_success += s
    total_failed += f
    all_failures.extend(failures)

    s, f, failures = process_items(
        split="val",
        items=val_items,
        dataset_root=dataset_root,
        labels_source_root=labels_source_root,
        label_check_root=label_check_root,
        args=args,
    )

    total_success += s
    total_failed += f
    all_failures.extend(failures)

    write_data_yaml(dataset_root, args.label_mode)

    failure_log = label_check_root / "label_generation_failures.txt"

    with open(failure_log, "w", encoding="utf-8") as f:
        for line in all_failures:
            f.write(line + "\n")

    print("\n========== TOTAL RESULT ==========")
    print(f"success: {total_success}")
    print(f"failed:  {total_failed}")
    print(f"failure_log: {failure_log}")
    print("==================================")


if __name__ == "__main__":
    main()