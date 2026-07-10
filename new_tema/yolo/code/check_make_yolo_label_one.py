from pathlib import Path
import argparse
import json
import math

import cv2
import numpy as np


# ============================================================
# 1개 샘플 YOLO 라벨 생성 검증 코드
#
# 입력:
#   one_sample/aug_000000.png
#   one_sample/aug_000000_marker_top_id_uv.npy
#   one_sample/aug_000000_marker_meta.json
#
# 출력:
#   one_sample_result/aug_000000.txt
#   one_sample_result/aug_000000_label_check.png
#   one_sample_result/aug_000000_mask_check.png
#
# class 부여 규칙:
#   class 0 = 정사각형 ID
#   class 1 = 정사각형 기준 시계방향 첫 번째 직사각형
#   class 2 = 정사각형 기준 시계방향 두 번째 직사각형
#   class 3 = 정사각형 기준 시계방향 세 번째 직사각형
# ============================================================


def load_meta(meta_path: Path) -> dict:
    """meta.json 파일 읽기"""
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def uv_to_pixel(uv: np.ndarray, meta: dict):
    """
    uv 좌표를 이미지 pixel 좌표로 변환

    변환식:
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
    """연결 성분을 추출하고 bbox 계산"""
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

    이 좌표계에서 atan2(dy, dx)는 화면 기준 시계방향 각도로 사용할 수 있음.
    """
    dx = cx - ox
    dy = cy - oy
    return math.atan2(dy, dx)


def normalize_angle_positive(angle: float):
    """각도를 0 ~ 2pi 범위로 변환"""
    two_pi = 2.0 * math.pi
    return angle % two_pi


def assign_classes(components):
    """
    class 부여

    1. 정사각형 ID를 class 0으로 지정
       - bbox 가로/세로 비율이 1에 가장 가까운 component 사용

    2. 나머지 직사각형 3개는 정사각형 기준 시계방향 순서로 class 1~3 부여
       - 전체 중심에서 정사각형 중심으로 향하는 방향을 기준 방향으로 설정
       - 전체 중심에서 각 직사각형 중심으로 향하는 방향의 상대각 계산
       - 상대각이 작은 순서대로 class 1, 2, 3 부여
    """
    if len(components) != 4:
        raise RuntimeError(f"class 부여에는 component 4개가 필요합니다. 현재 개수: {len(components)}")

    # ------------------------------------------------------------
    # 1. 정사각형 ID 찾기
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
    # 2. 전체 중심 계산
    # ------------------------------------------------------------
    center_x = float(np.mean([c["cx"] for c in components]))
    center_y = float(np.mean([c["cy"] for c in components]))

    # ------------------------------------------------------------
    # 3. 정사각형 방향을 기준 방향으로 설정
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
            "angle": comp_angle,
            "comp": comp
        })

    other_items = sorted(other_items, key=lambda item: item["rel_angle"])

    # ------------------------------------------------------------
    # 5. class 부여
    # ------------------------------------------------------------
    assigned = []

    square_comp = square_comp.copy()
    square_comp["class_id"] = 0
    square_comp["rel_angle_from_square"] = 0.0
    assigned.append(square_comp)

    for class_id, item in enumerate(other_items, start=1):
        comp = item["comp"].copy()
        comp["class_id"] = class_id
        comp["rel_angle_from_square"] = item["rel_angle"]
        assigned.append(comp)

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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sample_dir",
        type=str,
        default="../one_sample",
        help="1개 샘플 입력 폴더"
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="../one_sample_result",
        help="결과 저장 폴더"
    )

    parser.add_argument(
        "--stem",
        type=str,
        default="aug_000000",
        help="샘플 기본 이름"
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

    args = parser.parse_args()

    sample_dir = Path(args.sample_dir)
    out_dir = Path(args.out_dir)
    stem = args.stem

    out_dir.mkdir(parents=True, exist_ok=True)

    image_path = sample_dir / f"{stem}.png"
    uv_path = sample_dir / f"{stem}_marker_top_id_uv.npy"
    meta_path = sample_dir / f"{stem}_marker_meta.json"

    if not image_path.exists():
        raise FileNotFoundError(f"이미지 파일 없음: {image_path}")

    if not uv_path.exists():
        raise FileNotFoundError(f"top_id_uv 파일 없음: {uv_path}")

    if not meta_path.exists():
        raise FileNotFoundError(f"meta 파일 없음: {meta_path}")

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image is None:
        raise RuntimeError(f"이미지를 읽지 못했습니다: {image_path}")

    image_h, image_w = image.shape[:2]

    uv = np.load(str(uv_path))

    if uv.ndim != 2 or uv.shape[1] < 2:
        raise RuntimeError(f"uv 파일 형식이 잘못되었습니다. shape={uv.shape}")

    uv = uv[:, :2].astype(np.float64)

    meta = load_meta(meta_path)

    image_size_meta = int(meta.get("image_size", image_w))

    if image_w != image_h:
        raise RuntimeError(f"정사각형 이미지가 아닙니다. image_w={image_w}, image_h={image_h}")

    if image_size_meta != image_w:
        print(f"[WARN] meta image_size={image_size_meta}, 실제 이미지 크기={image_w}")

    pixel_xy = uv_to_pixel(uv, meta)

    raw_mask, valid_count, invalid_count = make_point_mask(
        pixel_xy,
        image_w,
        args.point_radius
    )

    cleaned_mask = clean_mask(
        raw_mask,
        args.close_kernel,
        args.dilate_iter
    )

    components = extract_components(
        cleaned_mask,
        args.min_area
    )

    print("========== CHECK INFO ==========")
    print(f"image_path: {image_path}")
    print(f"uv_path: {uv_path}")
    print(f"meta_path: {meta_path}")
    print(f"image_size: {image_w} x {image_h}")
    print(f"uv points: {len(uv)}")
    print(f"valid pixel points: {valid_count}")
    print(f"invalid pixel points: {invalid_count}")
    print(f"detected components: {len(components)}")

    for i, comp in enumerate(components[:10]):
        print(
            f"component {i}: "
            f"x={comp['x']}, y={comp['y']}, "
            f"w={comp['w']}, h={comp['h']}, "
            f"area={comp['area']}, "
            f"cx={comp['cx']:.2f}, cy={comp['cy']:.2f}"
        )

    selected = select_four_components(components)
    assigned = assign_classes(selected)

    print("========== CLASS ASSIGN ==========")
    for comp in assigned:
        deg = math.degrees(float(comp.get("rel_angle_from_square", 0.0)))
        print(
            f"class {comp['class_id']}: "
            f"x={comp['x']}, y={comp['y']}, "
            f"w={comp['w']}, h={comp['h']}, "
            f"cx={comp['cx']:.2f}, cy={comp['cy']:.2f}, "
            f"rel_angle_from_square={deg:.2f} deg"
        )

    txt_lines = []

    for comp in assigned:
        x_center, y_center, width, height, _ = bbox_to_yolo(
            comp,
            image_w,
            image_h,
            args.bbox_expand_px
        )

        class_id = comp["class_id"]

        txt_lines.append(
            f"{class_id} "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{width:.6f} "
            f"{height:.6f}"
        )

    txt_path = out_dir / f"{stem}.txt"

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(txt_lines) + "\n")

    check_img = draw_check_image(
        image,
        assigned,
        image_w,
        image_h,
        args.bbox_expand_px
    )

    check_path = out_dir / f"{stem}_label_check.png"
    mask_path = out_dir / f"{stem}_mask_check.png"

    cv2.imwrite(str(check_path), check_img)
    cv2.imwrite(str(mask_path), cleaned_mask)

    print("========== OUTPUT ==========")
    print(f"YOLO txt saved: {txt_path}")
    print(f"check image saved: {check_path}")
    print(f"mask image saved: {mask_path}")
    print("============================")


if __name__ == "__main__":
    main()