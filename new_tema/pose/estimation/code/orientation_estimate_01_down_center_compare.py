from pathlib import Path
from datetime import datetime
from collections import Counter
import csv
import math

import cv2
import numpy as np

# ============================================================
# 기존 첫 번째 방향 추정 코드 import
# 방향 추정은 여기서 가져온 결과를 그대로 사용한다.
# CENTER만 새 방식으로 교체한다.
# ============================================================

import orientation_estimate_01_down as base


# ============================================================
# 설정
# ============================================================

POSE_ROOT = Path("/Users/hajiwan/Desktop/object_detection/new_tema/pose")
NEWTEMA_ROOT = Path("/Users/hajiwan/Desktop/object_detection/new_tema")
REALDATA_DIR = NEWTEMA_ROOT / "yolov2/realdata/range_sweep_down_10sets/01_down"

DIR_RADIUS_RATIO = 0.22
CENTER_RADIUS = 6
DIRECTION_RADIUS = 8


# ============================================================
# UV 변환 후보
# marker_all_uv.npy는 대략 -0.5~0.5 부근
# YOLO label은 0~1 normalized image 좌표
# ============================================================

UV_MODES = [
    "xy",
    "xy_flip_y",
    "flip_x_y",
    "flip_xy",
    "swap",
    "swap_flip_y",
    "swap_flip_x",
    "swap_flip_xy",
]


def uv_to_norm_by_mode(uv, mode):
    """
    uv 좌표를 YOLO normalized image 좌표로 변환한다.

    mode:
    - xy:            x=u+0.5, y=v+0.5
    - xy_flip_y:     x=u+0.5, y=0.5-v
    - flip_x_y:      x=0.5-u, y=v+0.5
    - flip_xy:       x=0.5-u, y=0.5-v
    - swap 계열:     u/v 축 교환
    """
    uv = np.asarray(uv, dtype=np.float64)
    u = uv[:, 0]
    v = uv[:, 1]

    if mode == "xy":
        x, y = u + 0.5, v + 0.5
    elif mode == "xy_flip_y":
        x, y = u + 0.5, 0.5 - v
    elif mode == "flip_x_y":
        x, y = 0.5 - u, v + 0.5
    elif mode == "flip_xy":
        x, y = 0.5 - u, 0.5 - v
    elif mode == "swap":
        x, y = v + 0.5, u + 0.5
    elif mode == "swap_flip_y":
        x, y = v + 0.5, 0.5 - u
    elif mode == "swap_flip_x":
        x, y = 0.5 - v, u + 0.5
    elif mode == "swap_flip_xy":
        x, y = 0.5 - v, 0.5 - u
    else:
        raise ValueError(f"unknown uv mode: {mode}")

    return np.stack([x, y], axis=1)


def image_stem_to_real_base(stem):
    """
    pose 입력 이미지:
        1_19_2026_162815_complete_denoise_marker_color

    realdata 파일:
        1_19_2026_162815_complete_denoise_marker_marker_all_uv.npy
    """
    if stem.endswith("_color"):
        return stem[:-len("_color")]
    return stem


def load_marker_all_uv(stem):
    """
    해당 이미지와 대응되는 marker_all_uv.npy 로드.
    """
    real_base = image_stem_to_real_base(stem)
    p = REALDATA_DIR / f"{real_base}_marker_all_uv.npy"

    if not p.exists():
        return None, p

    arr = np.load(p).astype(np.float64)

    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"[ERROR] marker_all_uv shape 이상: {p}, {arr.shape}")

    return arr, p


# ============================================================
# polygon 내부 판정
# ============================================================

def points_in_polygon(points, poly):
    """
    points: (N, 2), normalized image 좌표
    poly: YOLO segmentation polygon normalized 좌표
    """
    if len(points) == 0 or len(poly) < 3:
        return np.zeros((len(points),), dtype=bool)

    x = points[:, 0]
    y = points[:, 1]

    inside = np.zeros(len(points), dtype=bool)

    xj, yj = poly[-1]

    for xi, yi in poly:
        cond = ((yi > y) != (yj > y))
        x_intersect = (xj - xi) * (y - yi) / ((yj - yi) + 1e-12) + xi
        inside ^= cond & (x < x_intersect)
        xj, yj = xi, yi

    return inside


def score_uv_mode(items, mode):
    """
    npy + image/label 방식에서 사용할 uv 변환 mode 선택.

    marker_all_uv 점을 image normalized 좌표로 변환했을 때
    YOLO polygon 안에 많이 들어가는 mode를 선택한다.

    이건 CENTER 좌표 변환을 위해서만 사용한다.
    방향 추정에는 사용하지 않는다.
    """
    total_hits = 0
    total_points = 0

    for item in items:
        marker_uv = item["marker_uv"]
        detections = item["detections"]

        if marker_uv is None or len(marker_uv) == 0 or len(detections) == 0:
            continue

        pts = uv_to_norm_by_mode(marker_uv, mode)

        in_img = (
            (pts[:, 0] >= 0.0) & (pts[:, 0] <= 1.0) &
            (pts[:, 1] >= 0.0) & (pts[:, 1] <= 1.0)
        )

        pts = pts[in_img]

        if len(pts) == 0:
            continue

        any_hit = np.zeros(len(pts), dtype=bool)

        for det in detections:
            any_hit |= points_in_polygon(pts, det["points"])

        total_hits += int(any_hit.sum())
        total_points += int(len(pts))

    ratio = total_hits / total_points if total_points > 0 else 0.0
    return total_hits, total_points, ratio


def choose_uv_mode_for_npy_image(items):
    """
    npy + image/label 정보 기반 CENTER 계산에 사용할 uv mode 자동 선택.
    """
    rows = []

    for mode in UV_MODES:
        hits, total, ratio = score_uv_mode(items, mode)
        rows.append((mode, hits, total, ratio))

    rows = sorted(rows, key=lambda x: (x[1], x[3]), reverse=True)
    return rows[0][0], rows


# ============================================================
# CENTER 계산
# ============================================================

def marker_center_uv_obb(marker_uv):
    """
    marker_all_uv.npy만 가지고 OBB 중심을 계산한다.
    """
    if marker_uv is None or len(marker_uv) == 0:
        return None

    pts = marker_uv.astype(np.float32)

    if len(pts) >= 5:
        rect = cv2.minAreaRect(pts)
        return np.array(rect[0], dtype=np.float64)

    return np.mean(marker_uv, axis=0)


def center_from_npy_image(marker_uv, uv_mode, image_shape):
    """
    비교 1:
    npy + image/label 정보 기반 CENTER.

    - marker_all_uv의 OBB 중심 계산
    - label overlap으로 선택한 uv_mode로 image 좌표 변환
    """
    center_uv = marker_center_uv_obb(marker_uv)

    if center_uv is None:
        return None

    h, w = image_shape[:2]
    norm = uv_to_norm_by_mode(center_uv.reshape(1, 2), uv_mode)[0]
    px = np.array([norm[0] * w, norm[1] * h], dtype=np.float64)

    return {
        "center_uv": center_uv,
        "center_norm": norm,
        "center_px": px,
        "center_method": f"NPY_IMAGE_OBB_{uv_mode}",
    }


def center_from_npy_only(marker_uv, image_shape):
    """
    비교 2:
    npy 정보만으로 CENTER.

    - marker_all_uv의 OBB 중심 계산
    - 좌표계 자동 선택 없이 고정 변환 사용
      x = u + 0.5
      y = v + 0.5

    주의:
    이 방식은 label/image overlap으로 flip/swap을 보정하지 않는다.
    """
    center_uv = marker_center_uv_obb(marker_uv)

    if center_uv is None:
        return None

    h, w = image_shape[:2]
    norm = uv_to_norm_by_mode(center_uv.reshape(1, 2), "xy")[0]
    px = np.array([norm[0] * w, norm[1] * h], dtype=np.float64)

    return {
        "center_uv": center_uv,
        "center_norm": norm,
        "center_px": px,
        "center_method": "NPY_ONLY_OBB_FIXED_XY",
    }


# ============================================================
# 방향점 생성
# ============================================================

def make_direction_points_with_center(center_px, north_vec_norm, image_shape):
    """
    기존 코드가 계산한 north_vec은 그대로 사용한다.
    CENTER만 새 center_px로 교체한다.
    """
    h, w = image_shape[:2]

    north_vec_px = np.array(
        [north_vec_norm[0] * w, north_vec_norm[1] * h],
        dtype=np.float64,
    )

    norm = float(np.linalg.norm(north_vec_px))

    if norm < 1e-12:
        return None

    north_unit = north_vec_px / norm
    east_unit = np.array([-north_unit[1], north_unit[0]], dtype=np.float64)

    radius_px = min(w, h) * DIR_RADIUS_RATIO

    N = center_px + north_unit * radius_px
    S = center_px - north_unit * radius_px
    E = center_px + east_unit * radius_px
    W = center_px - east_unit * radius_px

    angle_deg = math.degrees(math.atan2(-north_unit[1], north_unit[0]))

    return {
        "CENTER": center_px,
        "N": N,
        "E": E,
        "S": S,
        "W": W,
        "angle_deg": angle_deg,
    }


# ============================================================
# 시각화
# ============================================================

def draw_text(canvas, lines):
    x = 15
    y = 26

    for i, line in enumerate(lines):
        yy = y + i * 23

        cv2.putText(
            canvas,
            line,
            (x, yy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            canvas,
            line,
            (x, yy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )


def draw_orientation(image, result, dir_points, method_name):
    """
    CENTER/N/E/S/W만 표시.
    class 점과 C0~C3는 추가로 그리지 않는다.
    """
    canvas = image.copy()

    center = tuple(np.round(dir_points["CENTER"]).astype(int))

    cv2.circle(canvas, center, CENTER_RADIUS, (255, 255, 255), -1)
    cv2.putText(
        canvas,
        "CENTER",
        (center[0] + 8, center[1] + 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    colors = {
        "N": (0, 0, 255),
        "E": (0, 255, 0),
        "S": (255, 0, 0),
        "W": (0, 255, 255),
    }

    for key in ["N", "E", "S", "W"]:
        p = tuple(np.round(dir_points[key]).astype(int))
        cv2.circle(canvas, p, DIRECTION_RADIUS, colors[key], -1)
        cv2.putText(
            canvas,
            key,
            (p[0] + 8, p[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            colors[key],
            2,
            cv2.LINE_AA,
        )

    lines = [
        result.get("status", "UNKNOWN"),
        f"conf={result.get('confidence', 0.0):.3f}",
        method_name,
        "DIR=original v1 / CENTER=npy",
    ]

    draw_text(canvas, lines)
    return canvas


def draw_fail_image(image, result, center_data, method_name):
    canvas = image.copy()

    if center_data is not None:
        center = tuple(np.round(center_data["center_px"]).astype(int))
        cv2.circle(canvas, center, CENTER_RADIUS, (255, 255, 255), -1)
        cv2.putText(
            canvas,
            "CENTER",
            (center[0] + 8, center[1] + 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    draw_text(canvas, [result.get("status", "UNKNOWN"), method_name, result.get("reason", "")])
    return canvas


def list_to_str(values):
    return "|".join(str(v) for v in values)


# ============================================================
# 출력
# ============================================================

def make_output_dir():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = POSE_ROOT / f"orientation_output_01_down_center_compare_{ts}"
    img_npy_image = out_dir / "images_npy_image_center"
    img_npy_only = out_dir / "images_npy_only_center"

    if out_dir.exists():
        raise FileExistsError(f"[ERROR] 출력 폴더 이미 존재: {out_dir}")

    img_npy_image.mkdir(parents=True, exist_ok=False)
    img_npy_only.mkdir(parents=True, exist_ok=False)

    return out_dir, img_npy_image, img_npy_only


def main():
    input_run, input_dir, label_dir = base.find_latest_input_dir()
    out_dir, img_npy_image_dir, img_npy_only_dir = make_output_dir()

    print(f"[INFO] POSE_ROOT    = {POSE_ROOT}")
    print(f"[INFO] REALDATA_DIR = {REALDATA_DIR}")
    print(f"[INFO] INPUT_RUN    = {input_run}")
    print(f"[INFO] INPUT_DIR    = {input_dir}")
    print(f"[INFO] LABEL_DIR    = {label_dir}")
    print(f"[INFO] OUTPUT_DIR   = {out_dir}")

    txt_paths = sorted(label_dir.glob("*.txt"))

    if len(txt_paths) == 0:
        raise FileNotFoundError(f"[ERROR] label txt 없음: {label_dir}")

    items = []

    # ------------------------------------------------------------
    # 1. 기존 v1 입력 로딩 + marker_all_uv 로딩
    # ------------------------------------------------------------
    for txt_path in txt_paths:
        stem = txt_path.stem
        image_path = base.find_image_path(input_dir, stem)

        if image_path is None:
            print(f"[WARN] 이미지 없음: {stem}")
            continue

        image = cv2.imread(str(image_path))

        if image is None:
            print(f"[WARN] 이미지 읽기 실패: {image_path}")
            continue

        detections = base.read_label_file(txt_path)
        reps = base.select_representative_by_class(detections)

        marker_uv, marker_uv_path = load_marker_all_uv(stem)

        items.append({
            "stem": stem,
            "txt_path": txt_path,
            "image_path": image_path,
            "image_shape": image.shape,
            "detections": detections,
            "reps": reps,
            "marker_uv": marker_uv,
            "marker_uv_path": marker_uv_path,
        })

    if len(items) == 0:
        raise RuntimeError("[ERROR] 처리 가능한 이미지 없음")

    # ------------------------------------------------------------
    # 2. 기존 v1 template 생성
    # 방향 추정은 그대로 사용
    # ------------------------------------------------------------
    template, used_template_images = base.build_layout_template(items)

    print(f"[INFO] v1 template 생성 이미지 수 = {used_template_images}")
    print("[INFO] v1 template offsets:")
    for cls in sorted(template.keys()):
        print(f"  class{cls}: {template[cls]}")

    # ------------------------------------------------------------
    # 3. npy + image/label 방식용 uv mode 선택
    # ------------------------------------------------------------
    uv_mode, uv_mode_rows = choose_uv_mode_for_npy_image(items)

    print("[INFO] UV mode scores for NPY+IMAGE center:")
    for mode, hits, total, ratio in uv_mode_rows:
        print(f"  {mode:14s} hits={hits:8d} total={total:8d} ratio={ratio:.6f}")

    print(f"[INFO] selected UV mode = {uv_mode}")

    mode_csv = out_dir / "uv_mode_scores_npy_image_center.csv"

    with mode_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["mode", "hits", "total", "ratio", "selected"])
        for mode, hits, total, ratio in uv_mode_rows:
            writer.writerow([mode, hits, total, ratio, mode == uv_mode])

    # ------------------------------------------------------------
    # 4. 방향 추정 + 두 방식 CENTER 적용
    # ------------------------------------------------------------
    result_csv = out_dir / "orientation_center_compare_results.csv"

    fieldnames = [
        "image_name",
        "method",
        "status",
        "reason",
        "detected_classes",
        "used_classes",
        "square_detected",
        "square_estimated",
        "old_center_x",
        "old_center_y",
        "new_center_x",
        "new_center_y",
        "center_uv_u",
        "center_uv_v",
        "north_x",
        "north_y",
        "east_x",
        "east_y",
        "south_x",
        "south_y",
        "west_x",
        "west_y",
        "angle_deg",
        "confidence",
        "center_method",
        "marker_uv_path",
    ]

    status_counter = Counter()

    with result_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for item in items:
            image = cv2.imread(str(item["image_path"]))

            # 기존 v1 방향 추정 그대로
            result = base.estimate_orientation(item["reps"], template)
            status_counter[result.get("status", "UNKNOWN")] += 1

            old_center_px = None
            if "marker_center" in result:
                h, w = image.shape[:2]
                old_center_px = np.array(
                    [result["marker_center"][0] * w, result["marker_center"][1] * h],
                    dtype=np.float64,
                )

            center_variants = [
                (
                    "NPY_IMAGE_CENTER",
                    center_from_npy_image(item["marker_uv"], uv_mode, image.shape),
                    img_npy_image_dir,
                ),
                (
                    "NPY_ONLY_CENTER",
                    center_from_npy_only(item["marker_uv"], image.shape),
                    img_npy_only_dir,
                ),
            ]

            for method_name, center_data, image_out_dir in center_variants:
                row = {
                    "image_name": item["image_path"].name,
                    "method": method_name,
                    "status": result.get("status", ""),
                    "reason": result.get("reason", ""),
                    "detected_classes": list_to_str(result.get("detected_classes", [])),
                    "used_classes": list_to_str(result.get("used_classes", [])),
                    "square_detected": result.get("square_detected", False),
                    "square_estimated": result.get("square_estimated", False),
                    "old_center_x": "" if old_center_px is None else float(old_center_px[0]),
                    "old_center_y": "" if old_center_px is None else float(old_center_px[1]),
                    "new_center_x": "",
                    "new_center_y": "",
                    "center_uv_u": "",
                    "center_uv_v": "",
                    "north_x": "",
                    "north_y": "",
                    "east_x": "",
                    "east_y": "",
                    "south_x": "",
                    "south_y": "",
                    "west_x": "",
                    "west_y": "",
                    "angle_deg": "",
                    "confidence": result.get("confidence", 0.0),
                    "center_method": "" if center_data is None else center_data["center_method"],
                    "marker_uv_path": str(item["marker_uv_path"]),
                }

                if center_data is not None:
                    row["new_center_x"] = float(center_data["center_px"][0])
                    row["new_center_y"] = float(center_data["center_px"][1])
                    row["center_uv_u"] = float(center_data["center_uv"][0])
                    row["center_uv_v"] = float(center_data["center_uv"][1])

                # 방향 벡터는 기존 v1 result["north_vec"] 그대로 사용
                if center_data is not None and "north_vec" in result:
                    dir_points = make_direction_points_with_center(
                        center_data["center_px"],
                        result["north_vec"],
                        image.shape,
                    )

                    if dir_points is not None:
                        drawn = draw_orientation(image, result, dir_points, method_name)
                        out_img = image_out_dir / f"{item['stem']}_{method_name}.jpg"
                        cv2.imwrite(str(out_img), drawn)

                        row.update({
                            "north_x": float(dir_points["N"][0]),
                            "north_y": float(dir_points["N"][1]),
                            "east_x": float(dir_points["E"][0]),
                            "east_y": float(dir_points["E"][1]),
                            "south_x": float(dir_points["S"][0]),
                            "south_y": float(dir_points["S"][1]),
                            "west_x": float(dir_points["W"][0]),
                            "west_y": float(dir_points["W"][1]),
                            "angle_deg": float(dir_points["angle_deg"]),
                        })
                    else:
                        fail_img = draw_fail_image(
                            image,
                            {"status": "FAIL_BAD_DIRECTION_VECTOR", "reason": "original north_vec failed"},
                            center_data,
                            method_name,
                        )
                        out_img = image_out_dir / f"{item['stem']}_{method_name}.jpg"
                        cv2.imwrite(str(out_img), fail_img)

                else:
                    fail_img = draw_fail_image(image, result, center_data, method_name)
                    out_img = image_out_dir / f"{item['stem']}_{method_name}.jpg"
                    cv2.imwrite(str(out_img), fail_img)

                writer.writerow(row)

    print("[DONE] center compare 완료")
    print(f"[DONE] OUTPUT_DIR: {out_dir}")
    print(f"[DONE] NPY+IMAGE images: {img_npy_image_dir}")
    print(f"[DONE] NPY ONLY images: {img_npy_only_dir}")
    print(f"[DONE] CSV: {result_csv}")
    print(f"[DONE] UV mode CSV: {mode_csv}")
    print("[DONE] original direction status summary:")

    for k, v in sorted(status_counter.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
