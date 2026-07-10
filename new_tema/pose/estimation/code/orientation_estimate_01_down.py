from pathlib import Path
from datetime import datetime
import csv
import math

import cv2
import numpy as np


# ============================================================
# 설정값
# - 원본 yolov2/result 폴더는 절대 접근하지 않음
# - pose 안에 복사된 입력만 사용
# - 출력은 항상 timestamp 새 폴더 생성
# ============================================================

POSE_ROOT = Path("/Users/hajiwan/Desktop/object_detection/new_tema/pose")

# 가장 최근에 복사한 orientation_input 폴더 자동 사용
INPUT_RUN_PREFIX = "orientation_input_01_down_yolo11n_D3_real_empirical_v2_spatial_train160_val40_epoch60_conf030_"

# 방향점 표시 반경: 이미지 짧은 변 기준 비율
DIR_RADIUS_RATIO = 0.22

# 시각화 점 크기
POINT_RADIUS = 5
CENTER_RADIUS = 6
DIRECTION_RADIUS = 8

# class0 = square = North 기준
SQUARE_CLASS_ID = 0


# ============================================================
# 유틸 함수
# ============================================================

def find_latest_input_dir():
    """
    pose 폴더 안에서 가장 최근 orientation_input 폴더를 찾는다.
    원본 YOLO 결과 폴더는 절대 사용하지 않는다.
    """
    candidates = sorted(
        [p for p in POSE_ROOT.iterdir() if p.is_dir() and p.name.startswith(INPUT_RUN_PREFIX)],
        key=lambda p: p.name,
    )

    if not candidates:
        raise FileNotFoundError(f"[ERROR] pose 안에 입력 폴더가 없습니다: {POSE_ROOT}")

    input_run = candidates[-1]
    input_dir = input_run / "01_down"
    label_dir = input_dir / "labels"

    if not input_dir.exists():
        raise FileNotFoundError(f"[ERROR] 01_down 폴더가 없습니다: {input_dir}")

    if not label_dir.exists():
        raise FileNotFoundError(f"[ERROR] labels 폴더가 없습니다: {label_dir}")

    return input_run, input_dir, label_dir


def make_output_dir():
    """
    항상 새로운 timestamp 출력 폴더를 만든다.
    기존 출력 폴더를 덮어쓰지 않는다.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = POSE_ROOT / f"orientation_output_01_down_{ts}"
    image_out_dir = out_dir / "images"

    if out_dir.exists():
        raise FileExistsError(f"[ERROR] 출력 폴더가 이미 존재합니다. 덮어쓰기 금지: {out_dir}")

    image_out_dir.mkdir(parents=True, exist_ok=False)
    return out_dir, image_out_dir


def polygon_area(points):
    """
    polygon 면적 계산.
    points: normalized 좌표, shape=(N, 2)
    """
    if len(points) < 3:
        return 0.0

    x = points[:, 0]
    y = points[:, 1]
    return float(abs(0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))))


def polygon_center(points):
    """
    polygon 중심 계산.
    현재 YOLO segmentation polygon은 점 수가 많으므로 평균 중심을 사용한다.
    """
    if len(points) == 0:
        return np.array([np.nan, np.nan], dtype=np.float64)

    return points.mean(axis=0).astype(np.float64)


def parse_yolo_seg_line(line):
    """
    YOLO segmentation txt 한 줄 파싱.
    현재 형식:
        class x1 y1 x2 y2 ... confidence

    마지막 값은 confidence로 처리한다.
    """
    parts = line.strip().split()
    if len(parts) < 8:
        return None

    cls = int(float(parts[0]))
    vals = [float(v) for v in parts[1:]]

    # 좌표 + confidence 구조
    # vals = 2N 좌표 + 1 confidence
    if len(vals) % 2 == 1:
        conf = float(vals[-1])
        coords = vals[:-1]
    else:
        # 혹시 confidence가 없는 txt도 읽을 수 있게 방어
        conf = 1.0
        coords = vals

    if len(coords) < 6 or len(coords) % 2 != 0:
        return None

    points = np.array(coords, dtype=np.float64).reshape(-1, 2)
    center = polygon_center(points)
    area = polygon_area(points)

    return {
        "cls": cls,
        "points": points,
        "center": center,
        "area": area,
        "conf": conf,
    }


def read_label_file(txt_path):
    """
    label txt 파일을 읽고 detection 리스트 반환.
    """
    detections = []

    with txt_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            det = parse_yolo_seg_line(line)
            if det is not None:
                detections.append(det)

    return detections


def find_image_path(input_dir, stem):
    """
    label 파일명과 같은 stem의 이미지 파일 찾기.
    """
    for ext in [".jpg", ".jpeg", ".png"]:
        p = input_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def select_representative_by_class(detections):
    """
    class별 대표 detection 선택.
    1차 버전에서는 confidence가 가장 높은 detection을 대표로 사용한다.
    """
    reps = {}

    for det in detections:
        cls = det["cls"]

        # 면적이 0인 잘못된 polygon은 제외
        if det["area"] <= 0:
            continue

        if cls not in reps:
            reps[cls] = det
        else:
            if det["conf"] > reps[cls]["conf"]:
                reps[cls] = det

    return reps


def mean_center_from_reps(reps):
    """
    대표 detection들의 중심 평균으로 marker 중심을 계산한다.
    """
    centers = [det["center"] for det in reps.values() if np.all(np.isfinite(det["center"]))]

    if len(centers) == 0:
        return None

    return np.mean(np.stack(centers, axis=0), axis=0)


def rotate_to_canonical(offset, north_unit):
    """
    direct square 케이스에서 현재 이미지의 offset을 canonical 좌표로 변환.
    canonical 좌표에서는 square 방향이 +Y가 되도록 정렬한다.
    """
    # north_unit 방향을 canonical +Y로 둔다.
    # x축은 north를 기준으로 오른쪽 방향 성분.
    e_y = north_unit
    e_x = np.array([-north_unit[1], north_unit[0]], dtype=np.float64)

    return np.array([
        float(np.dot(offset, e_x)),
        float(np.dot(offset, e_y)),
    ], dtype=np.float64)


def build_layout_template(all_items):
    """
    class0이 검출된 이미지들을 이용해 class 배치 템플릿을 만든다.

    방식:
    1. class0 있는 이미지 선택
    2. marker_center 계산
    3. class0 방향을 north 기준으로 두고 canonical 좌표계로 회전
    4. class별 offset median으로 template 생성
    """
    offsets_by_class = {0: [], 1: [], 2: [], 3: []}

    used_images = 0

    for item in all_items:
        reps = item["reps"]

        if SQUARE_CLASS_ID not in reps:
            continue

        if len(reps) < 3:
            continue

        marker_center = mean_center_from_reps(reps)
        if marker_center is None:
            continue

        square_center = reps[SQUARE_CLASS_ID]["center"]
        north_vec = square_center - marker_center
        north_norm = np.linalg.norm(north_vec)

        if north_norm < 1e-6:
            continue

        north_unit = north_vec / north_norm

        # scale normalization: 각 class offset 거리의 median 사용
        distances = []
        for det in reps.values():
            d = np.linalg.norm(det["center"] - marker_center)
            if d > 1e-6:
                distances.append(d)

        if len(distances) == 0:
            continue

        scale = float(np.median(distances))
        if scale < 1e-6:
            continue

        for cls, det in reps.items():
            if cls not in offsets_by_class:
                continue

            offset = det["center"] - marker_center
            canon = rotate_to_canonical(offset, north_unit) / scale
            offsets_by_class[cls].append(canon)

        used_images += 1

    template = {}

    for cls, offsets in offsets_by_class.items():
        if len(offsets) == 0:
            continue

        template[cls] = np.median(np.stack(offsets, axis=0), axis=0)

    if SQUARE_CLASS_ID not in template:
        raise RuntimeError("[ERROR] class0 기반 template 생성 실패")

    return template, used_images


def estimate_similarity_transform(template_points, observed_points):
    """
    template 좌표를 observed 좌표에 맞추는 2D similarity transform 추정.

    q = scale * R * p + t

    template_points: shape=(N, 2)
    observed_points: shape=(N, 2)
    """
    P = np.asarray(template_points, dtype=np.float64)
    Q = np.asarray(observed_points, dtype=np.float64)

    if len(P) != len(Q):
        raise ValueError("template_points와 observed_points 개수가 다릅니다.")

    if len(P) < 2:
        return None

    mu_p = P.mean(axis=0)
    mu_q = Q.mean(axis=0)

    P0 = P - mu_p
    Q0 = Q - mu_q

    denom = float(np.sum(P0 ** 2))
    if denom < 1e-12:
        return None

    H = P0.T @ Q0
    U, S, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    # reflection 방지
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    scale = float(np.sum(S) / denom)
    t = mu_q - scale * (mu_p @ R.T)

    pred = scale * (P @ R.T) + t
    residual = float(np.mean(np.linalg.norm(pred - Q, axis=1)))

    return scale, R, t, residual


def transform_point(point, scale, R, t):
    """
    template 좌표의 한 점을 observed 좌표계로 변환.
    """
    p = np.asarray(point, dtype=np.float64)
    return scale * (p @ R.T) + t


def estimate_orientation(reps, template):
    """
    한 이미지의 방향 추정.

    CASE 1:
        class0 있음 → 직접 North
    CASE 2:
        class0 없음, class1/2/3 존재 → template matching으로 square 추정
    CASE 3:
        class 부족 → LOW_CONFIDENCE
    """
    detected_classes = sorted(list(reps.keys()))
    used_classes = []

    if len(reps) == 0:
        return {
            "status": "FAIL_NO_DETECTION",
            "reason": "no detections",
            "detected_classes": detected_classes,
            "used_classes": used_classes,
        }

    # ------------------------------------------------------------
    # CASE 1. class0 직접 검출
    # ------------------------------------------------------------
    if SQUARE_CLASS_ID in reps:
        marker_center = mean_center_from_reps(reps)
        square_center = reps[SQUARE_CLASS_ID]["center"]

        if marker_center is None:
            return {
                "status": "FAIL_BAD_GEOMETRY",
                "reason": "marker center failed",
                "detected_classes": detected_classes,
                "used_classes": used_classes,
            }

        north_vec = square_center - marker_center
        norm = float(np.linalg.norm(north_vec))

        if norm < 1e-6:
            return {
                "status": "FAIL_BAD_GEOMETRY",
                "reason": "north vector too small",
                "detected_classes": detected_classes,
                "used_classes": used_classes,
            }

        used_classes = sorted(list(reps.keys()))
        mean_conf = float(np.mean([det["conf"] for det in reps.values()]))

        return {
            "status": "OK_DIRECT_SQUARE",
            "reason": "class0 square detected",
            "detected_classes": detected_classes,
            "used_classes": used_classes,
            "square_detected": True,
            "square_estimated": False,
            "marker_center": marker_center,
            "square_center": square_center,
            "north_vec": north_vec,
            "confidence": mean_conf,
            "residual": 0.0,
        }

    # ------------------------------------------------------------
    # CASE 2/3. class0 없음 → template matching으로 square 추정
    # ------------------------------------------------------------
    common_classes = sorted([cls for cls in reps.keys() if cls in template and cls != SQUARE_CLASS_ID])

    if len(common_classes) < 2:
        return {
            "status": "FAIL_NO_ENOUGH_CLASSES",
            "reason": "class0 missing and fewer than 2 usable classes",
            "detected_classes": detected_classes,
            "used_classes": common_classes,
            "square_detected": False,
            "square_estimated": False,
        }

    template_points = np.stack([template[cls] for cls in common_classes], axis=0)
    observed_points = np.stack([reps[cls]["center"] for cls in common_classes], axis=0)

    transform = estimate_similarity_transform(template_points, observed_points)

    if transform is None:
        return {
            "status": "FAIL_BAD_GEOMETRY",
            "reason": "similarity transform failed",
            "detected_classes": detected_classes,
            "used_classes": common_classes,
            "square_detected": False,
            "square_estimated": False,
        }

    scale, R, t, residual = transform

    marker_center = t
    estimated_square = transform_point(template[SQUARE_CLASS_ID], scale, R, t)
    north_vec = estimated_square - marker_center
    norm = float(np.linalg.norm(north_vec))

    if norm < 1e-6:
        return {
            "status": "FAIL_BAD_GEOMETRY",
            "reason": "estimated north vector too small",
            "detected_classes": detected_classes,
            "used_classes": common_classes,
            "square_detected": False,
            "square_estimated": True,
        }

    mean_conf = float(np.mean([reps[cls]["conf"] for cls in common_classes]))

    # residual이 작고 사용 class가 많을수록 confidence 증가
    residual_score = max(0.0, 1.0 - residual / max(scale, 1e-6))
    class_score = min(1.0, len(common_classes) / 3.0)
    confidence = mean_conf * residual_score * class_score

    if len(common_classes) >= 3:
        status = "OK_ESTIMATED_SQUARE"
        reason = "class0 missing, estimated by class1/2/3 template matching"
    else:
        status = "LOW_CONFIDENCE"
        reason = "class0 missing, estimated with only 2 classes"

    return {
        "status": status,
        "reason": reason,
        "detected_classes": detected_classes,
        "used_classes": common_classes,
        "square_detected": False,
        "square_estimated": True,
        "marker_center": marker_center,
        "square_center": estimated_square,
        "north_vec": north_vec,
        "confidence": confidence,
        "residual": residual,
    }


def make_direction_points(marker_center_norm, north_vec_norm, image_shape):
    """
    marker 중심과 north vector로 N/E/S/W 점 계산.
    좌표는 pixel 기준으로 반환한다.
    """
    h, w = image_shape[:2]

    marker_px = np.array([marker_center_norm[0] * w, marker_center_norm[1] * h], dtype=np.float64)
    north_px = np.array([north_vec_norm[0] * w, north_vec_norm[1] * h], dtype=np.float64)

    norm = float(np.linalg.norm(north_px))
    if norm < 1e-6:
        return None

    north_unit = north_px / norm

    # 이미지 좌표계: x 오른쪽, y 아래쪽
    # north가 위쪽이면 east는 오른쪽이 되어야 하므로 [-y, x]
    east_unit = np.array([-north_unit[1], north_unit[0]], dtype=np.float64)

    radius_px = min(w, h) * DIR_RADIUS_RATIO

    N = marker_px + north_unit * radius_px
    S = marker_px - north_unit * radius_px
    E = marker_px + east_unit * radius_px
    W = marker_px - east_unit * radius_px

    # angle_deg: 이미지 y축을 위쪽 좌표계로 바꾼 뒤 +x 기준 각도
    angle_deg = math.degrees(math.atan2(-north_unit[1], north_unit[0]))

    return {
        "marker": marker_px,
        "N": N,
        "E": E,
        "S": S,
        "W": W,
        "north_unit": north_unit,
        "angle_deg": angle_deg,
    }


def draw_orientation(image, reps, result, dir_points):
    """
    방향 추정 결과를 이미지에 표시.
    화살표는 사용하지 않고 점과 텍스트만 표시한다.
    """
    canvas = image.copy()

    # class 중심점 표시
    class_colors = {
        0: (0, 255, 255),    # square
        1: (255, 0, 0),
        2: (0, 255, 0),
        3: (0, 0, 255),
    }

    h, w = canvas.shape[:2]

    for cls, det in reps.items():
        c = det["center"]
        p = (int(round(c[0] * w)), int(round(c[1] * h)))
        color = class_colors.get(cls, (200, 200, 200))
        cv2.circle(canvas, p, POINT_RADIUS, color, -1)
        cv2.putText(
            canvas,
            f"C{cls}",
            (p[0] + 6, p[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    # marker center
    marker = tuple(np.round(dir_points["marker"]).astype(int))
    cv2.circle(canvas, marker, CENTER_RADIUS, (255, 255, 255), -1)
    cv2.putText(
        canvas,
        "CENTER",
        (marker[0] + 8, marker[1] + 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    # N/E/S/W 점
    direction_colors = {
        "N": (0, 0, 255),
        "E": (0, 255, 0),
        "S": (255, 0, 0),
        "W": (0, 255, 255),
    }

    for label in ["N", "E", "S", "W"]:
        p = tuple(np.round(dir_points[label]).astype(int))
        color = direction_colors[label]
        cv2.circle(canvas, p, DIRECTION_RADIUS, color, -1)
        cv2.putText(
            canvas,
            label,
            (p[0] + 8, p[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )

    # status 표시
    status = result.get("status", "UNKNOWN")
    reason = result.get("reason", "")
    conf = result.get("confidence", 0.0)

    cv2.putText(
        canvas,
        f"{status} | conf={conf:.3f}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        reason[:80],
        (20, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return canvas


def list_to_str(values):
    return "|".join(str(v) for v in values)


def main():
    input_run, input_dir, label_dir = find_latest_input_dir()
    out_dir, image_out_dir = make_output_dir()

    print(f"[INFO] POSE_ROOT  = {POSE_ROOT}")
    print(f"[INFO] INPUT_RUN  = {input_run}")
    print(f"[INFO] INPUT_DIR  = {input_dir}")
    print(f"[INFO] LABEL_DIR  = {label_dir}")
    print(f"[INFO] OUTPUT_DIR = {out_dir}")

    # ------------------------------------------------------------
    # 1. 전체 label 읽기
    # ------------------------------------------------------------
    all_items = []

    txt_paths = sorted(label_dir.glob("*.txt"))

    if len(txt_paths) == 0:
        raise FileNotFoundError(f"[ERROR] txt label 파일이 없습니다: {label_dir}")

    for txt_path in txt_paths:
        stem = txt_path.stem
        image_path = find_image_path(input_dir, stem)

        if image_path is None:
            print(f"[WARN] 이미지 없음, skip: {stem}")
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[WARN] 이미지 읽기 실패, skip: {image_path}")
            continue

        detections = read_label_file(txt_path)
        reps = select_representative_by_class(detections)

        all_items.append({
            "stem": stem,
            "txt_path": txt_path,
            "image_path": image_path,
            "image_shape": image.shape,
            "detections": detections,
            "reps": reps,
        })

    if len(all_items) == 0:
        raise RuntimeError("[ERROR] 처리 가능한 이미지가 없습니다.")

    # ------------------------------------------------------------
    # 2. class0 직접 검출 케이스로 layout template 생성
    # ------------------------------------------------------------
    template, used_template_images = build_layout_template(all_items)

    print(f"[INFO] template 생성에 사용한 이미지 수 = {used_template_images}")
    print("[INFO] template offsets:")
    for cls in sorted(template.keys()):
        print(f"  class{cls}: {template[cls]}")

    # ------------------------------------------------------------
    # 3. 각 이미지 방향 추정 + 시각화 + CSV 저장
    # ------------------------------------------------------------
    csv_path = out_dir / "orientation_results.csv"

    fieldnames = [
        "image_name",
        "status",
        "detected_classes",
        "used_classes",
        "square_detected",
        "square_estimated",
        "marker_center_x",
        "marker_center_y",
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
        "residual",
        "reason",
    ]

    summary_count = {}

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for item in all_items:
            image = cv2.imread(str(item["image_path"]))
            reps = item["reps"]

            result = estimate_orientation(reps, template)
            status = result.get("status", "UNKNOWN")
            summary_count[status] = summary_count.get(status, 0) + 1

            row = {
                "image_name": item["image_path"].name,
                "status": status,
                "detected_classes": list_to_str(result.get("detected_classes", [])),
                "used_classes": list_to_str(result.get("used_classes", [])),
                "square_detected": result.get("square_detected", False),
                "square_estimated": result.get("square_estimated", False),
                "marker_center_x": "",
                "marker_center_y": "",
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
                "residual": result.get("residual", ""),
                "reason": result.get("reason", ""),
            }

            if "marker_center" in result and "north_vec" in result:
                dir_points = make_direction_points(
                    result["marker_center"],
                    result["north_vec"],
                    image.shape,
                )

                if dir_points is not None:
                    drawn = draw_orientation(image, reps, result, dir_points)
                    out_img_path = image_out_dir / f"{item['stem']}_orientation.jpg"
                    cv2.imwrite(str(out_img_path), drawn)

                    row.update({
                        "marker_center_x": float(dir_points["marker"][0]),
                        "marker_center_y": float(dir_points["marker"][1]),
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
                    row["status"] = "FAIL_BAD_DIRECTION_VECTOR"
                    row["reason"] = "direction point calculation failed"
            else:
                # 실패 케이스도 원본 이미지에 status만 표시해서 저장
                fail_image = image.copy()
                cv2.putText(
                    fail_image,
                    f"{status}",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                out_img_path = image_out_dir / f"{item['stem']}_orientation.jpg"
                cv2.imwrite(str(out_img_path), fail_image)

            writer.writerow(row)

    print("[DONE] 방향 추정 완료")
    print(f"[DONE] 결과 폴더: {out_dir}")
    print(f"[DONE] 결과 이미지: {image_out_dir}")
    print(f"[DONE] CSV: {csv_path}")
    print("[DONE] status summary:")
    for k, v in sorted(summary_count.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
