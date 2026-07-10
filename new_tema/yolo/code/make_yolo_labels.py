from pathlib import Path
import argparse
import json
import math
import traceback

import cv2
import numpy as np


# ============================================================
# YOLO 전체 라벨 생성 코드
#
# 목적:
#   yolo_dataset/images/train, images/val 이미지 기준으로
#   source/train, source/val 안의
#   *_marker_top_id_uv.npy + *_marker_meta.json을 사용하여
#   YOLO detect용 txt 라벨 생성
#
# class 규칙:
#   class 0 = 정사각형 ID
#   class 1 = 정사각형 기준 시계방향 첫 번째 직사각형
#   class 2 = 정사각형 기준 시계방향 두 번째 직사각형
#   class 3 = 정사각형 기준 시계방향 세 번째 직사각형
#
# 출력:
#   yolo_dataset/labels/train/*.txt
#   yolo_dataset/labels/val/*.txt
#   label_check/train/*_label_check.png 일부 저장
#   label_check/val/*_label_check.png 일부 저장
# ============================================================

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def load_meta(meta_path: Path) -> dict:
    """meta.json 파일 읽기"""
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def uv_to_pixel(uv: np.ndarray, meta: dict):
    """
    uv 좌표를 이미지 pixel 좌표로 변환

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
    """pixel 좌표를 이용해 내부 ID 포인트 마스크 생성"""
    mask = np.zeros((image_size, image_size), dtype=np.uint8)

    px = np.round(pixel_xy[:, 0]).astype(np.int32)
    py = np.round(pixel_xy[:, 1]).astype(np.int32)

    valid = (
        (px >= 0) & (px < image_size) &
        (py >= 0) & (py < image_size)
    )

    px_valid = px[valid]
    py_valid = py[valid]

    for x, y in zip(px_valid, py_valid):
        cv2.circle(mask, (int(x), int(y)), point_radius, 255, -1)

    return mask, int(valid.sum()), int((~valid).sum())


def clean_mask(mask: np.ndarray, close_kernel: int, dilate_iter: int):
    """점 형태 마스크를 내부 ID 덩어리로 연결"""
    kernel = np.ones((close_kernel, close_kernel), dtype=np.uint8)

    cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    if dilate_iter > 0:
        cleaned = cv2.dilate(cleaned, kernel, iterations=dilate_iter)

    return cleaned


def extract_components(mask: np.ndarray, min_area: int):
    """연결 성분 추출 및 bbox 계산"""
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

        if area < min_area:
            continue

        components.append({
            "label_id": label_id,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "area": area,
            "cx": float(cx),
            "cy": float(cy),
        })

    # 큰 영역 우선 정렬
    components = sorted(components, key=lambda d: d["area"], reverse=True)

    return components


def select_four_components(components):
    """내부 ID 4개 component 선택"""
    if len(components) < 4:
        raise RuntimeError(
            f"내부 ID component가 4개보다 적습니다. 현재 개수: {len(components)}"
        )

    return components[:4]


def image_clockwise_angle(cx: float, cy: float, ox: float, oy: float):
    """
    이미지 좌표계 기준 각도 계산

    이미지 좌표계:
        x 오른쪽 증가
        y 아래쪽 증가

    이 좌표계에서 atan2(dy, dx)는 화면 기준 시계방향 각도로 사용 가능
    """
    dx = cx - ox
    dy = cy - oy
    return math.atan2(dy, dx)


def normalize_angle_positive(angle: float):
    """각도를 0 ~ 2pi 범위로 변환"""
    return angle % (2.0 * math.pi)


def assign_classes(components):
    """
    class 부여

    class 0:
        정사각형 ID

    class 1~3:
        정사각형 기준 시계방향 순서의 직사각형 ID
    """
    if len(components) != 4:
        raise RuntimeError(f"class 부여에는 component 4개가 필요합니다. 현재 개수: {len(components)}")

    # ------------------------------------------------------------
    # 1. 정사각형 ID 찾기
    #    bbox 가로/세로 비율이 1에 가장 가까운 component
    # ------------------------------------------------------------
    square_scores = []

    for idx, comp in enumerate(components):
        w = max(comp["w"], 1)
        h = max(comp["h"], 1)

        ratio = w / h
        square_score = abs(math.log(ratio))

        square_scores.append((square_score, idx))

    _, square_idx = min(square_scores)
    square_comp = components[square_idx]

    # ------------------------------------------------------------
    # 2. 내부 ID 4개 중심의 평균을 기준 중심으로 사용
    # ------------------------------------------------------------
    center_x = float(np.mean([c["cx"] for c in components]))
    center_y = float(np.mean([c["cy"] for c in components]))

    # ------------------------------------------------------------
    # 3. 기준 방향 = 중심에서 정사각형 ID로 향하는 방향
    # ------------------------------------------------------------
    square_angle = image_clockwise_angle(
        square_comp["cx"],
        square_comp["cy"],
        center_x,
        center_y
    )

    # ------------------------------------------------------------
    # 4. 나머지 3개를 정사각형 기준 시계방향 상대각으로 정렬
    # ------------------------------------------------------------
    other_items = []

    for idx, comp in enumerate(components):
        if idx == square_idx:
            continue

        comp_angle = image_clockwise_angle(
            comp["cx"],
            comp["cy"],
            center_x,
            center_y
        )

        rel_angle = normalize_angle_positive(comp_angle - square_angle)

        other_items.append({
            "rel_angle": rel_angle,
            "comp": comp
        })

    other_items = sorted(other_items, key=lambda item: item["rel_angle"])

    # ------------------------------------------------------------
    # 5. class 번호 부여
    # ------------------------------------------------------------
    assigned = []

    square_copy = square_comp.copy()
    square_copy["class_id"] = 0
    square_copy["rel_angle_from_square"] = 0.0
    assigned.append(square_copy)

    for class_id, item in enumerate(other_items, start=1):
        comp_copy = item["comp"].copy()
        comp_copy["class_id"] = class_id
        comp_copy["rel_angle_from_square"] = item["rel_angle"]
        assigned.append(comp_copy)

    assigned = sorted(assigned, key=lambda d: d["class_id"])

    return assigned


def bbox_to_yolo(comp, image_w: int, image_h: int, expand_px: int):
    """component bbox를 YOLO 형식으로 변환"""
    x1 = comp["x"] - expand_px
    y1 = comp["y"] - expand_px
    x2 = comp["x"] + comp["w"] + expand_px
    y2 = comp["y"] + comp["h"] + expand_px

    x1 = max(0, min(image_w - 1, x1))
    y1 = max(0, min(image_h - 1, y1))
    x2 = max(0, min(image_w - 1, x2))
    y2 = max(0, min(image_h - 1, y2))

    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)

    x_center = (x1 + x2) / 2.0 / image_w
    y_center = (y1 + y2) / 2.0 / image_h
    width = bw / image_w
    height = bh / image_h

    return x_center, y_center, width, height, (x1, y1, x2, y2)


def draw_check_image(image, assigned_components, image_w: int, image_h: int, expand_px: int):
    """bbox와 class 번호를 원본 이미지 위에 표시"""
    vis = image.copy()

    for comp in assigned_components:
        _, _, _, _, box = bbox_to_yolo(
            comp,
            image_w,
            image_h,
            expand_px
        )

        x1, y1, x2, y2 = box
        class_id = comp["class_id"]

        cv2.rectangle(
            vis,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 255, 0),
            2
        )

        cv2.putText(
            vis,
            f"class {class_id}",
            (int(x1), max(20, int(y1) - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        cv2.circle(
            vis,
            (int(comp["cx"]), int(comp["cy"])),
            4,
            (255, 0, 0),
            -1
        )

    return vis


def collect_images(image_dir: Path):
    """이미지 목록 수집"""
    if not image_dir.exists():
        raise FileNotFoundError(f"이미지 폴더 없음: {image_dir}")

    images = []

    for path in image_dir.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            images.append(path)

    return sorted(images)


def make_label_for_one_image(
    image_path: Path,
    source_dir: Path,
    label_dir: Path,
    check_dir: Path,
    save_check: bool,
    point_radius: int,
    close_kernel: int,
    dilate_iter: int,
    min_area: int,
    bbox_expand_px: int,
    overwrite: bool,
    apply: bool
):
    """이미지 1장에 대한 YOLO 라벨 생성"""
    stem = image_path.stem

    uv_path = source_dir / f"{stem}_marker_top_id_uv.npy"
    meta_path = source_dir / f"{stem}_marker_meta.json"
    txt_path = label_dir / f"{stem}.txt"
    check_path = check_dir / f"{stem}_label_check.png"
    mask_path = check_dir / f"{stem}_mask_check.png"

    if not uv_path.exists():
        raise FileNotFoundError(f"top_id_uv 파일 없음: {uv_path}")

    if not meta_path.exists():
        raise FileNotFoundError(f"meta 파일 없음: {meta_path}")

    if txt_path.exists() and not overwrite:
        return {
            "status": "skip_exists",
            "stem": stem,
            "message": f"라벨이 이미 존재합니다: {txt_path}"
        }

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image is None:
        raise RuntimeError(f"이미지를 읽지 못했습니다: {image_path}")

    image_h, image_w = image.shape[:2]

    if image_w != image_h:
        raise RuntimeError(f"정사각형 이미지가 아닙니다. image_w={image_w}, image_h={image_h}")

    uv = np.load(str(uv_path))

    if uv.ndim != 2 or uv.shape[1] < 2:
        raise RuntimeError(f"uv 파일 형식 오류: {uv_path}, shape={uv.shape}")

    uv = uv[:, :2].astype(np.float64)

    meta = load_meta(meta_path)

    image_size_meta = int(meta.get("image_size", image_w))

    if image_size_meta != image_w:
        print(f"[WARN] {stem}: meta image_size={image_size_meta}, actual={image_w}")

    pixel_xy = uv_to_pixel(uv, meta)

    raw_mask, valid_count, invalid_count = make_point_mask(
        pixel_xy,
        image_w,
        point_radius
    )

    cleaned_mask = clean_mask(
        raw_mask,
        close_kernel,
        dilate_iter
    )

    components = extract_components(
        cleaned_mask,
        min_area
    )

    selected = select_four_components(components)
    assigned = assign_classes(selected)

    txt_lines = []

    for comp in assigned:
        x_center, y_center, width, height, _ = bbox_to_yolo(
            comp,
            image_w,
            image_h,
            bbox_expand_px
        )

        class_id = comp["class_id"]

        # YOLO label 값 범위 확인
        vals = [x_center, y_center, width, height]
        if not all(0.0 <= v <= 1.0 for v in vals):
            raise RuntimeError(f"YOLO 좌표가 0~1 범위를 벗어났습니다: {stem}, values={vals}")

        txt_lines.append(
            f"{class_id} "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{width:.6f} "
            f"{height:.6f}"
        )

    class_ids = [comp["class_id"] for comp in assigned]

    if sorted(class_ids) != [0, 1, 2, 3]:
        raise RuntimeError(f"class_id 구성이 잘못되었습니다: {stem}, class_ids={class_ids}")

    if apply:
        label_dir.mkdir(parents=True, exist_ok=True)

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(txt_lines) + "\n")

        if save_check:
            check_dir.mkdir(parents=True, exist_ok=True)

            check_img = draw_check_image(
                image,
                assigned,
                image_w,
                image_h,
                bbox_expand_px
            )

            cv2.imwrite(str(check_path), check_img)
            cv2.imwrite(str(mask_path), cleaned_mask)

    return {
        "status": "ok",
        "stem": stem,
        "component_count": len(components),
        "valid_points": valid_count,
        "invalid_points": invalid_count,
        "txt_path": str(txt_path),
        "check_saved": save_check
    }


def process_split(
    split: str,
    dataset_root: Path,
    source_root: Path,
    check_root: Path,
    check_count: int,
    point_radius: int,
    close_kernel: int,
    dilate_iter: int,
    min_area: int,
    bbox_expand_px: int,
    overwrite: bool,
    apply: bool
):
    """train 또는 val 전체 처리"""
    image_dir = dataset_root / "images" / split
    label_dir = dataset_root / "labels" / split
    source_dir = source_root / split
    check_dir = check_root / split

    images = collect_images(image_dir)

    if not source_dir.exists():
        raise FileNotFoundError(f"source/{split} 폴더 없음: {source_dir}")

    print(f"\n========== {split.upper()} START ==========")
    print(f"image_dir: {image_dir}")
    print(f"source_dir: {source_dir}")
    print(f"label_dir: {label_dir}")
    print(f"image_count: {len(images)}")

    success = []
    skipped = []
    failed = []

    for idx, image_path in enumerate(images):
        save_check = idx < check_count

        try:
            result = make_label_for_one_image(
                image_path=image_path,
                source_dir=source_dir,
                label_dir=label_dir,
                check_dir=check_dir,
                save_check=save_check,
                point_radius=point_radius,
                close_kernel=close_kernel,
                dilate_iter=dilate_iter,
                min_area=min_area,
                bbox_expand_px=bbox_expand_px,
                overwrite=overwrite,
                apply=apply
            )

            if result["status"] == "ok":
                success.append(result["stem"])
                print(f"[OK] {split} {idx + 1}/{len(images)} {result['stem']}")
            elif result["status"] == "skip_exists":
                skipped.append(result["stem"])
                print(f"[SKIP] {split} {idx + 1}/{len(images)} {result['stem']}")

        except Exception as e:
            failed.append({
                "stem": image_path.stem,
                "error": str(e),
                "traceback": traceback.format_exc()
            })
            print(f"[FAIL] {split} {idx + 1}/{len(images)} {image_path.stem}: {e}")

    print(f"========== {split.upper()} RESULT ==========")
    print(f"success: {len(success)}")
    print(f"skipped: {len(skipped)}")
    print(f"failed:  {len(failed)}")

    return {
        "split": split,
        "image_count": len(images),
        "success": success,
        "skipped": skipped,
        "failed": failed
    }


def save_failure_log(log_path: Path, results: list, apply: bool):
    """실패 로그 저장"""
    lines = []

    for result in results:
        split = result["split"]

        for item in result["failed"]:
            lines.append(f"[{split}] {item['stem']}")
            lines.append(item["error"])
            lines.append("")

    if not lines:
        lines.append("No failed samples.")

    if apply:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    print(f"\nfailure log: {log_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        default="../yolo_dataset",
        help="YOLO 데이터셋 폴더"
    )

    parser.add_argument(
        "--source",
        type=str,
        default="../source",
        help="source 폴더. 내부에 train, val 폴더가 있어야 함"
    )

    parser.add_argument(
        "--check_root",
        type=str,
        default="../label_check",
        help="라벨 검증 이미지 저장 폴더"
    )

    parser.add_argument(
        "--check_count",
        type=int,
        default=20,
        help="train/val 각각 저장할 label_check 이미지 개수"
    )

    parser.add_argument(
        "--point_radius",
        type=int,
        default=2,
        help="uv 포인트를 마스크에 찍을 때 사용할 반지름"
    )

    parser.add_argument(
        "--close_kernel",
        type=int,
        default=5,
        help="마스크 연결용 kernel 크기"
    )

    parser.add_argument(
        "--dilate_iter",
        type=int,
        default=1,
        help="마스크 dilation 반복 횟수"
    )

    parser.add_argument(
        "--min_area",
        type=int,
        default=30,
        help="component 최소 면적"
    )

    parser.add_argument(
        "--bbox_expand_px",
        type=int,
        default=5,
        help="bbox 확장 pixel"
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="기존 txt 라벨이 있어도 덮어쓰기"
    )

    parser.add_argument(
        "--apply",
        action="store_true",
        help="실제로 txt와 check 이미지를 저장"
    )

    args = parser.parse_args()

    dataset_root = Path(args.dataset)
    source_root = Path(args.source)
    check_root = Path(args.check_root)

    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset 폴더 없음: {dataset_root}")

    if not source_root.exists():
        raise FileNotFoundError(f"source 폴더 없음: {source_root}")

    print("========== CONFIG ==========")
    print(f"dataset:       {dataset_root.resolve()}")
    print(f"source:        {source_root.resolve()}")
    print(f"check_root:    {check_root.resolve()}")
    print(f"check_count:   {args.check_count}")
    print(f"point_radius:  {args.point_radius}")
    print(f"close_kernel:  {args.close_kernel}")
    print(f"dilate_iter:   {args.dilate_iter}")
    print(f"min_area:      {args.min_area}")
    print(f"bbox_expand:   {args.bbox_expand_px}")
    print(f"overwrite:     {args.overwrite}")
    print(f"apply:         {args.apply}")
    print("============================")

    results = []

    for split in ["train", "val"]:
        result = process_split(
            split=split,
            dataset_root=dataset_root,
            source_root=source_root,
            check_root=check_root,
            check_count=args.check_count,
            point_radius=args.point_radius,
            close_kernel=args.close_kernel,
            dilate_iter=args.dilate_iter,
            min_area=args.min_area,
            bbox_expand_px=args.bbox_expand_px,
            overwrite=args.overwrite,
            apply=args.apply
        )

        results.append(result)

    fail_log_path = check_root / "label_generation_failures.txt"
    save_failure_log(fail_log_path, results, args.apply)

    total_success = sum(len(r["success"]) for r in results)
    total_skipped = sum(len(r["skipped"]) for r in results)
    total_failed = sum(len(r["failed"]) for r in results)

    print("\n========== TOTAL RESULT ==========")
    print(f"total success: {total_success}")
    print(f"total skipped: {total_skipped}")
    print(f"total failed:  {total_failed}")

    if not args.apply:
        print("\n현재는 미리보기 모드입니다.")
        print("실제로 txt와 검증 이미지를 저장하려면 --apply를 붙이세요.")

    print("==================================")


if __name__ == "__main__":
    main()