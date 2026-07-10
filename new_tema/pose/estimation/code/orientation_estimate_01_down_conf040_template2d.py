from pathlib import Path
from datetime import datetime
from itertools import product
from collections import defaultdict, Counter
import csv
import math

import cv2
import numpy as np


# ============================================================
# GT-template 기반 2D 방향추정 코드
# ------------------------------------------------------------
# 목적:
# - npy/xyz 사용하지 않음
# - YOLO conf040 예측 결과만 대상으로 2D 방향추정
# - GT train/val label에서 class0/1/2/3의 상대 배치 template 생성
# - conf040 예측 후보 중 template 구조와 가장 맞는 조합 선택
# - CENTER / N / E / S / W를 2D 이미지에 표시
#
# 핵심:
# - confidence 최고 detection을 무조건 대표로 쓰지 않음
# - class별 모든 후보 조합을 만들고 구조 residual이 가장 작은 조합 선택
# - class0(square)이 없으면 class1/2/3이 모두 있을 때만 square 위치 추정
# ============================================================


# ============================================================
# 경로 설정
# ============================================================

POSE_ROOT = Path("/Users/hajiwan/Desktop/object_detection/new_tema/pose")

GT_DATASET_DIR = Path(
    "/Users/hajiwan/Desktop/object_detection/new_tema/"
    "yolov2/dataset/dataset11/"
    "damage_manual_poly_D3_real_empirical_v2_spatial_train160_val40"
)

CONF040_DIR = POSE_ROOT / "yolo11n_D3_real_empirical_v2_spatial_train160_val40_epoch60_01_down_conf040"


# ============================================================
# 파라미터
# ------------------------------------------------------------
# 구조 residual은 GT template과 YOLO 예측 조합이 얼마나 비슷한지 나타낸다.
# 값이 작을수록 template 구조와 잘 맞는다.
# ============================================================

MAX_CANDIDATES_PER_CLASS = 5

# 3개 이상 class가 있을 때 구조를 정상으로 인정할 residual 기준
STRUCTURE_TH_WITH_SQUARE = 0.55
STRUCTURE_TH_NO_SQUARE = 0.55

# 1등 조합과 2등 조합의 residual 차이가 너무 작으면 square 후보가 애매하다고 판단
AMBIGUOUS_GAP_TH = 0.08

# 방향점 표시 반경
DIR_RADIUS_RATIO = 0.22

CENTER_RADIUS = 6
DIRECTION_RADIUS = 8
SELECTED_CLASS_RADIUS = 5


# ============================================================
# 기본 유틸
# ============================================================

def polygon_area(points):
    """
    normalized polygon 면적 계산.
    points: (N, 2)
    """
    if points is None or len(points) < 3:
        return 0.0

    x = points[:, 0]
    y = points[:, 1]

    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def polygon_center(points):
    """
    polygon 중심 계산.
    segmentation polygon은 복잡할 수 있으므로 단순 평균 중심을 사용한다.
    """
    if points is None or len(points) == 0:
        return None

    return np.mean(points, axis=0)


def read_yolo_seg_label(txt_path):
    """
    YOLO segmentation label txt 읽기.

    지원 형식:
    - GT label:
      class x1 y1 x2 y2 ...
    - predict label with conf:
      class x1 y1 x2 y2 ... conf

    반환:
    [
      {
        "cls": int,
        "poly": np.ndarray(N,2),
        "center": np.ndarray(2,),
        "area": float,
        "conf": float,
        "path": str,
        "line_id": int
      },
      ...
    ]
    """
    detections = []

    if not txt_path.exists():
        return detections

    with txt_path.open("r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) < 7:
                continue

            cls = int(float(parts[0]))
            values = [float(v) for v in parts[1:]]

            # values 길이가 홀수이면 마지막 값은 confidence로 판단
            if len(values) % 2 == 1:
                conf = float(values[-1])
                coords = values[:-1]
            else:
                conf = 1.0
                coords = values

            if len(coords) < 6 or len(coords) % 2 != 0:
                continue

            poly = np.asarray(coords, dtype=np.float64).reshape(-1, 2)
            center = polygon_center(poly)
            area = polygon_area(poly)

            if center is None:
                continue

            detections.append({
                "cls": cls,
                "poly": poly,
                "center": center,
                "area": area,
                "conf": conf,
                "path": str(txt_path),
                "line_id": line_id,
            })

    return detections


def group_by_class(detections):
    grouped = defaultdict(list)

    for det in detections:
        grouped[det["cls"]].append(det)

    return grouped


def find_image_path(image_dir, stem):
    """
    txt stem과 같은 이름의 이미지 파일 탐색.
    """
    exts = [".jpg", ".jpeg", ".png", ".bmp"]

    for ext in exts:
        p = image_dir / f"{stem}{ext}"
        if p.exists():
            return p

    # YOLO 결과 이미지가 root에 있고 stem이 약간 다를 경우를 대비한 fallback
    matches = []
    for ext in exts:
        matches.extend(image_dir.glob(f"{stem}*{ext}"))

    if matches:
        return sorted(matches)[0]

    return None


def find_conf040_paths():
    """
    pose 안에 복사된 conf040 결과 폴더에서 image/label 경로를 찾는다.
    """
    if not CONF040_DIR.exists():
        raise FileNotFoundError(f"[ERROR] conf040 폴더 없음: {CONF040_DIR}")

    # 일반적인 YOLO predict 결과: root/images 없이 root에 jpg, root/labels
    label_candidates = [
        CONF040_DIR / "labels",
        CONF040_DIR / "01_down" / "labels",
    ]

    label_dir = None
    for p in label_candidates:
        if p.exists() and any(p.glob("*.txt")):
            label_dir = p
            break

    if label_dir is None:
        raise FileNotFoundError(f"[ERROR] conf040 labels 폴더를 찾지 못함: {CONF040_DIR}")

    # 이미지 경로 후보
    image_candidates = [
        CONF040_DIR,
        CONF040_DIR / "01_down",
        CONF040_DIR / "images",
        CONF040_DIR / "01_down" / "images",
    ]

    image_dir = None
    for p in image_candidates:
        if p.exists() and any(p.glob("*.jpg")):
            image_dir = p
            break

    if image_dir is None:
        raise FileNotFoundError(f"[ERROR] conf040 image 폴더를 찾지 못함: {CONF040_DIR}")

    return image_dir, label_dir


# ============================================================
# similarity transform
# ------------------------------------------------------------
# template 좌표 X를 관측 좌표 Y에 맞춘다.
# Y = scale * R * X + t
# ============================================================

def fit_similarity_transform(X, Y):
    """
    X: template points, shape (K,2)
    Y: observed points, shape (K,2)

    반환:
    scale, R, t, pred, residual_mean, residual_norm

    K=1이면 회전/스케일을 결정할 수 없으므로 None 반환.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    if len(X) != len(Y) or len(X) < 2:
        return None

    mx = X.mean(axis=0)
    my = Y.mean(axis=0)

    Xc = X - mx
    Yc = Y - my

    var_x = float(np.sum(Xc ** 2))

    if var_x < 1e-12:
        return None

    H = Xc.T @ Yc

    U, S, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    # reflection 방지
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    scale = float(np.sum(S) / var_x)

    if scale <= 1e-12:
        return None

    t = my - scale * (R @ mx)

    pred = (scale * (R @ X.T)).T + t

    errors = np.linalg.norm(pred - Y, axis=1)
    residual_mean = float(np.mean(errors))

    y_radius = float(np.sqrt(np.mean(np.sum((Y - my) ** 2, axis=1))))
    residual_norm = residual_mean / max(y_radius, 1e-9)

    return {
        "scale": scale,
        "R": R,
        "t": t,
        "pred": pred,
        "residual_mean": residual_mean,
        "residual_norm": residual_norm,
    }


def apply_transform(point, transform):
    """
    template 좌표 point를 fitted image normalized 좌표로 변환.
    """
    p = np.asarray(point, dtype=np.float64)
    return transform["scale"] * (transform["R"] @ p) + transform["t"]


# ============================================================
# GT template 생성
# ============================================================

def canonicalize_gt_centers(class_centers):
    """
    GT 4개 class 중심을 canonical 좌표로 변환한다.

    절차:
    1. 4개 class 중심 평균을 임시 center로 둔다.
    2. class0 방향을 canonical North 방향인 (0, -1)에 맞춰 회전한다.
    3. RMS 반경으로 scale 정규화한다.

    class0을 North 기준으로 고정하면,
    class1/2/3의 상대 위치가 누적 평균 가능해진다.
    """
    pts = np.stack([class_centers[c] for c in [0, 1, 2, 3]], axis=0)

    center = pts.mean(axis=0)
    rel = pts - center

    north_vec = rel[0]

    n = float(np.linalg.norm(north_vec))
    if n < 1e-12:
        return None

    current_angle = math.atan2(north_vec[1], north_vec[0])
    target_angle = -math.pi / 2.0

    rot_angle = target_angle - current_angle

    c = math.cos(rot_angle)
    s = math.sin(rot_angle)

    R = np.array([[c, -s], [s, c]], dtype=np.float64)

    rel_rot = (R @ rel.T).T

    scale = float(np.sqrt(np.mean(np.sum(rel_rot ** 2, axis=1))))

    if scale < 1e-12:
        return None

    rel_norm = rel_rot / scale

    return {
        cls: rel_norm[i]
        for i, cls in enumerate([0, 1, 2, 3])
    }


def build_gt_template():
    """
    GT labels/train + labels/val에서 4개 class가 모두 있는 sample만 사용해
    class 상대 배치 template을 만든다.
    """
    label_dirs = [
        GT_DATASET_DIR / "labels" / "train",
        GT_DATASET_DIR / "labels" / "val",
    ]

    for d in label_dirs:
        if not d.exists():
            raise FileNotFoundError(f"[ERROR] GT label 폴더 없음: {d}")

    canonical_samples = []
    used_files = []
    skipped_files = []

    for label_dir in label_dirs:
        for txt_path in sorted(label_dir.glob("*.txt")):
            detections = read_yolo_seg_label(txt_path)
            grouped = group_by_class(detections)

            # 0~3 class가 모두 있어야 template sample로 사용
            if not all(cls in grouped and len(grouped[cls]) > 0 for cls in [0, 1, 2, 3]):
                skipped_files.append((txt_path.name, "missing_class"))
                continue

            class_centers = {}

            # GT에서 중복이 있을 경우 면적이 가장 큰 polygon을 사용
            for cls in [0, 1, 2, 3]:
                det = max(grouped[cls], key=lambda d: d["area"])
                class_centers[cls] = det["center"]

            canonical = canonicalize_gt_centers(class_centers)

            if canonical is None:
                skipped_files.append((txt_path.name, "canonical_failed"))
                continue

            canonical_samples.append(canonical)
            used_files.append(str(txt_path))

    if len(canonical_samples) == 0:
        raise RuntimeError("[ERROR] GT template 생성에 사용할 label이 없음")

    template = {}

    for cls in [0, 1, 2, 3]:
        arr = np.stack([s[cls] for s in canonical_samples], axis=0)
        template[cls] = np.mean(arr, axis=0)

    # template 중심을 다시 0으로 보정
    template_center = np.stack([template[c] for c in [0, 1, 2, 3]], axis=0).mean(axis=0)

    for cls in [0, 1, 2, 3]:
        template[cls] = template[cls] - template_center

    # GT sample을 template에 다시 맞춰 GT residual 통계 계산
    residuals = []

    for s in canonical_samples:
        X = np.stack([template[c] for c in [0, 1, 2, 3]], axis=0)
        Y = np.stack([s[c] for c in [0, 1, 2, 3]], axis=0)

        tr = fit_similarity_transform(X, Y)
        if tr is not None:
            residuals.append(tr["residual_norm"])

    residuals = np.asarray(residuals, dtype=np.float64)

    return {
        "template": template,
        "used_files": used_files,
        "skipped_files": skipped_files,
        "gt_residual_mean": float(np.mean(residuals)) if len(residuals) else "",
        "gt_residual_p95": float(np.percentile(residuals, 95)) if len(residuals) else "",
        "num_samples": len(canonical_samples),
    }


# ============================================================
# 후보 조합 생성 및 선택
# ============================================================

def make_candidate_combinations(grouped):
    """
    class별 후보 detection 조합 생성.

    각 class 후보는 conf 높은 순서로 MAX_CANDIDATES_PER_CLASS개까지만 사용한다.
    """
    classes = sorted([c for c in grouped.keys() if c in [0, 1, 2, 3]])

    if len(classes) == 0:
        return []

    candidate_lists = []

    for cls in classes:
        dets = sorted(
            grouped[cls],
            key=lambda d: (d["conf"], d["area"]),
            reverse=True,
        )[:MAX_CANDIDATES_PER_CLASS]

        candidate_lists.append(dets)

    combos = []

    for selected in product(*candidate_lists):
        combo = {det["cls"]: det for det in selected}
        combos.append(combo)

    return combos


def evaluate_combo(combo, template):
    """
    하나의 class 조합을 GT template과 비교한다.
    """
    classes = sorted(combo.keys())

    if len(classes) < 2:
        return None

    X = np.stack([template[c] for c in classes], axis=0)
    Y = np.stack([combo[c]["center"] for c in classes], axis=0)

    transform = fit_similarity_transform(X, Y)

    if transform is None:
        return None

    # confidence 보조 점수
    conf_mean = float(np.mean([combo[c]["conf"] for c in classes]))

    # 최종 선택 score:
    # residual 우선, confidence는 동점 근처에서만 보조적으로 반영
    score = transform["residual_norm"] - 0.03 * conf_mean

    return {
        "classes": classes,
        "combo": combo,
        "transform": transform,
        "residual_norm": transform["residual_norm"],
        "residual_mean": transform["residual_mean"],
        "conf_mean": conf_mean,
        "score": score,
    }


def select_best_combo(grouped, template):
    """
    전체 후보 조합 중 template 구조와 가장 잘 맞는 조합 선택.
    """
    combos = make_candidate_combinations(grouped)

    evaluated = []

    for combo in combos:
        ev = evaluate_combo(combo, template)
        if ev is not None:
            evaluated.append(ev)

    evaluated = sorted(evaluated, key=lambda r: (r["score"], r["residual_norm"]))

    if not evaluated:
        return None, []

    return evaluated[0], evaluated


# ============================================================
# 방향 계산
# ============================================================

def estimate_orientation_from_selection(grouped, template):
    """
    conf040 예측 결과에서 최종 방향을 추정한다.

    규칙:
    - class0(square)이 있으면 class0을 North 기준으로 사용
    - class0이 여러 개 또는 다른 class 중복이 있으면 template residual로 조합 선택
    - class0이 없고 class1/2/3이 모두 있으면 class0 위치를 template으로 추정
    - class0이 없고 class가 2개 이하이면 실패
    """
    detected_classes = sorted([c for c in grouped.keys() if c in [0, 1, 2, 3]])
    counts = {c: len(grouped[c]) for c in detected_classes}

    if len(detected_classes) == 0:
        return {
            "status": "FAIL_NO_DETECTION",
            "reason": "no class0-3 detection",
            "detected_classes": detected_classes,
            "used_classes": [],
        }

    has_square = 0 in detected_classes

    # square가 없고 class1/2/3이 모두 없으면 방향 추정하지 않음
    if not has_square:
        if not all(c in detected_classes for c in [1, 2, 3]):
            return {
                "status": "FAIL_NO_SQUARE_AND_INSUFFICIENT_CLASSES",
                "reason": "class0 missing and class1/2/3 not all detected",
                "detected_classes": detected_classes,
                "used_classes": detected_classes,
            }

    # 후보 조합 선택
    best, ranked = select_best_combo(grouped, template)

    if best is None:
        return {
            "status": "FAIL_TEMPLATE_FIT",
            "reason": "similarity transform failed",
            "detected_classes": detected_classes,
            "used_classes": [],
        }

    used_classes = best["classes"]
    residual = best["residual_norm"]
    transform = best["transform"]

    has_duplicate = any(v > 1 for v in counts.values())

    # 2개 class만 있고 square 포함인 경우는 구조 residual로 검증이 거의 불가능하다.
    # 그래도 square가 있으므로 방향은 만들되 LOW 신뢰로 표시한다.
    if has_square and len(used_classes) == 2:
        status = "LOW_CONFIDENCE_WITH_SQUARE_2CLASS"
    elif has_square:
        if residual > STRUCTURE_TH_WITH_SQUARE:
            return {
                "status": "REJECT_BAD_STRUCTURE_WITH_SQUARE",
                "reason": f"residual_norm={residual:.4f} > {STRUCTURE_TH_WITH_SQUARE}",
                "detected_classes": detected_classes,
                "used_classes": used_classes,
                "residual_norm": residual,
                "residual_mean": best["residual_mean"],
                "confidence": best["conf_mean"],
            }

        if set(used_classes) == {0, 1, 2, 3} and not has_duplicate:
            status = "OK_FULL_4CLASS"
        elif has_duplicate:
            status = "OK_SQUARE_WITH_DUPLICATES"
        elif len(used_classes) >= 3:
            status = "OK_PARTIAL_WITH_SQUARE"
        else:
            status = "LOW_CONFIDENCE_WITH_SQUARE"

    else:
        # square 없음: class1/2/3 모두 있을 때만 여기까지 옴
        if residual > STRUCTURE_TH_NO_SQUARE:
            return {
                "status": "REJECT_BAD_STRUCTURE_NO_SQUARE",
                "reason": f"residual_norm={residual:.4f} > {STRUCTURE_TH_NO_SQUARE}",
                "detected_classes": detected_classes,
                "used_classes": used_classes,
                "residual_norm": residual,
                "residual_mean": best["residual_mean"],
                "confidence": best["conf_mean"],
            }

        status = "ESTIMATED_SQUARE_3CLASS"

    # square 후보가 여러 개이고 1등/2등 차이가 작으면 경고
    ambiguous = False

    if has_square and counts.get(0, 0) > 1 and len(ranked) >= 2:
        gap = ranked[1]["residual_norm"] - ranked[0]["residual_norm"]
        if gap < AMBIGUOUS_GAP_TH:
            ambiguous = True
            status = "WARN_AMBIGUOUS_SQUARE"

    # template center는 canonical 원점이므로 transform을 적용하면 image normalized center
    marker_center = apply_transform(np.array([0.0, 0.0], dtype=np.float64), transform)

    # square가 직접 선택된 경우: 선택된 class0 중심 사용
    if 0 in best["combo"]:
        square_center = best["combo"][0]["center"]
        square_estimated = False
    else:
        # square가 없는 경우: template class0 위치를 transform으로 복원
        square_center = apply_transform(template[0], transform)
        square_estimated = True

    north_vec = square_center - marker_center

    if float(np.linalg.norm(north_vec)) < 1e-12:
        return {
            "status": "FAIL_BAD_NORTH_VECTOR",
            "reason": "square center and marker center are identical",
            "detected_classes": detected_classes,
            "used_classes": used_classes,
        }

    return {
        "status": status,
        "reason": "",
        "detected_classes": detected_classes,
        "used_classes": used_classes,
        "counts": counts,
        "selected": best["combo"],
        "transform": transform,
        "marker_center": marker_center,
        "square_center": square_center,
        "square_estimated": square_estimated,
        "north_vec": north_vec,
        "residual_norm": residual,
        "residual_mean": best["residual_mean"],
        "confidence": best["conf_mean"],
        "ambiguous": ambiguous,
        "ranked_count": len(ranked),
    }


def make_direction_points(result, image_shape):
    """
    normalized CENTER/N vector를 pixel 좌표의 CENTER/N/E/S/W로 변환.
    """
    h, w = image_shape[:2]

    center_norm = result["marker_center"]
    north_vec_norm = result["north_vec"]

    center_px = np.array([center_norm[0] * w, center_norm[1] * h], dtype=np.float64)

    north_vec_px = np.array([north_vec_norm[0] * w, north_vec_norm[1] * h], dtype=np.float64)

    n = float(np.linalg.norm(north_vec_px))
    if n < 1e-12:
        return None

    north_unit = north_vec_px / n
    east_unit = np.array([-north_unit[1], north_unit[0]], dtype=np.float64)

    radius_px = min(w, h) * DIR_RADIUS_RATIO

    N = center_px + north_unit * radius_px
    S = center_px - north_unit * radius_px
    E = center_px + east_unit * radius_px
    W = center_px - east_unit * radius_px

    # image 좌표는 y가 아래로 증가하므로 angle 표시는 -y 기준으로 계산
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


def draw_orientation(image, result, dir_points, debug=False):
    canvas = image.copy()

    center = tuple(np.round(dir_points["CENTER"]).astype(int))

    cv2.circle(canvas, center, CENTER_RADIUS, (255, 255, 255), -1)
    cv2.putText(
        canvas,
        "CENTER",
        (center[0] + 8, center[1] + 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
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

    # debug 이미지에서는 선택된 class 중심도 표시
    if debug and "selected" in result:
        h, w = image.shape[:2]

        cls_colors = {
            0: (0, 0, 255),
            1: (0, 255, 0),
            2: (255, 0, 0),
            3: (0, 255, 255),
        }

        for cls, det in result["selected"].items():
            c = det["center"]
            p = (int(round(c[0] * w)), int(round(c[1] * h)))

            cv2.circle(canvas, p, SELECTED_CLASS_RADIUS, cls_colors.get(cls, (255, 255, 255)), -1)
            cv2.putText(
                canvas,
                f"C{cls}",
                (p[0] + 6, p[1] + 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                cls_colors.get(cls, (255, 255, 255)),
                2,
                cv2.LINE_AA,
            )

        if result.get("square_estimated", False):
            sq = result["square_center"]
            p = (int(round(sq[0] * w)), int(round(sq[1] * h)))
            cv2.circle(canvas, p, SELECTED_CLASS_RADIUS + 2, (255, 255, 255), 2)
            cv2.putText(
                canvas,
                "EST_C0",
                (p[0] + 6, p[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    lines = [
        result.get("status", "UNKNOWN"),
        f"res={result.get('residual_norm', '')}",
        f"conf={result.get('confidence', '')}",
        "GT-template 2D",
    ]

    draw_text(canvas, lines)
    return canvas


def draw_fail_image(image, result):
    canvas = image.copy()

    lines = [
        result.get("status", "UNKNOWN"),
        result.get("reason", ""),
        f"det={result.get('detected_classes', [])}",
    ]

    draw_text(canvas, lines)
    return canvas


# ============================================================
# 출력 관련
# ============================================================

def make_output_dirs():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = POSE_ROOT / f"orientation_output_01_down_conf040_template2d_{ts}"

    img_dir = out_dir / "images_orientation"
    debug_dir = out_dir / "images_debug_selection"

    if out_dir.exists():
        raise FileExistsError(f"[ERROR] 출력 폴더 이미 존재: {out_dir}")

    img_dir.mkdir(parents=True, exist_ok=False)
    debug_dir.mkdir(parents=True, exist_ok=False)

    return out_dir, img_dir, debug_dir


def list_to_str(values):
    return "|".join(str(v) for v in values)


def format_float(v):
    if v == "" or v is None:
        return ""
    try:
        return f"{float(v):.6f}"
    except Exception:
        return str(v)


# ============================================================
# main
# ============================================================

def main():
    image_dir, label_dir = find_conf040_paths()
    out_dir, image_out_dir, debug_out_dir = make_output_dirs()

    print(f"[INFO] GT_DATASET_DIR = {GT_DATASET_DIR}")
    print(f"[INFO] CONF040_DIR     = {CONF040_DIR}")
    print(f"[INFO] image_dir      = {image_dir}")
    print(f"[INFO] label_dir      = {label_dir}")
    print(f"[INFO] OUTPUT_DIR     = {out_dir}")

    gt_info = build_gt_template()
    template = gt_info["template"]

    print("[INFO] GT template samples:", gt_info["num_samples"])
    print("[INFO] GT residual mean:", gt_info["gt_residual_mean"])
    print("[INFO] GT residual p95 :", gt_info["gt_residual_p95"])
    print("[INFO] template:")
    for cls in [0, 1, 2, 3]:
        print(f"  class{cls}: {template[cls]}")

    # template 저장
    template_csv = out_dir / "gt_template_2d.csv"
    with template_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "template_x", "template_y"])
        for cls in [0, 1, 2, 3]:
            writer.writerow([cls, float(template[cls][0]), float(template[cls][1])])

    txt_paths = sorted(label_dir.glob("*.txt"))

    if len(txt_paths) == 0:
        raise FileNotFoundError(f"[ERROR] conf040 label txt 없음: {label_dir}")

    result_csv = out_dir / "orientation_results_conf040_template2d.csv"

    fieldnames = [
        "image_name",
        "status",
        "reason",
        "detected_classes",
        "used_classes",
        "class0_count",
        "class1_count",
        "class2_count",
        "class3_count",
        "square_estimated",
        "center_x",
        "center_y",
        "square_x",
        "square_y",
        "north_x",
        "north_y",
        "east_x",
        "east_y",
        "south_x",
        "south_y",
        "west_x",
        "west_y",
        "angle_deg",
        "residual_norm",
        "residual_mean",
        "confidence",
        "ranked_count",
    ]

    status_counter = Counter()

    with result_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for txt_path in txt_paths:
            stem = txt_path.stem
            image_path = find_image_path(image_dir, stem)

            if image_path is None:
                print(f"[WARN] 이미지 없음: {stem}")
                continue

            image = cv2.imread(str(image_path))

            if image is None:
                print(f"[WARN] 이미지 읽기 실패: {image_path}")
                continue

            detections = read_yolo_seg_label(txt_path)
            grouped = group_by_class(detections)

            result = estimate_orientation_from_selection(grouped, template)
            status_counter[result["status"]] += 1

            h, w = image.shape[:2]

            counts = result.get("counts", {})
            detected_classes = result.get("detected_classes", [])
            used_classes = result.get("used_classes", [])

            row = {
                "image_name": image_path.name,
                "status": result.get("status", ""),
                "reason": result.get("reason", ""),
                "detected_classes": list_to_str(detected_classes),
                "used_classes": list_to_str(used_classes),
                "class0_count": counts.get(0, 0),
                "class1_count": counts.get(1, 0),
                "class2_count": counts.get(2, 0),
                "class3_count": counts.get(3, 0),
                "square_estimated": result.get("square_estimated", ""),
                "center_x": "",
                "center_y": "",
                "square_x": "",
                "square_y": "",
                "north_x": "",
                "north_y": "",
                "east_x": "",
                "east_y": "",
                "south_x": "",
                "south_y": "",
                "west_x": "",
                "west_y": "",
                "angle_deg": "",
                "residual_norm": format_float(result.get("residual_norm", "")),
                "residual_mean": format_float(result.get("residual_mean", "")),
                "confidence": format_float(result.get("confidence", "")),
                "ranked_count": result.get("ranked_count", ""),
            }

            if "marker_center" in result and "north_vec" in result:
                dir_points = make_direction_points(result, image.shape)

                if dir_points is not None:
                    drawn = draw_orientation(image, result, dir_points, debug=False)
                    debug_drawn = draw_orientation(image, result, dir_points, debug=True)

                    cv2.imwrite(str(image_out_dir / f"{stem}_orientation.jpg"), drawn)
                    cv2.imwrite(str(debug_out_dir / f"{stem}_debug.jpg"), debug_drawn)

                    center_px = dir_points["CENTER"]
                    square_norm = result["square_center"]
                    square_px = np.array([square_norm[0] * w, square_norm[1] * h], dtype=np.float64)

                    row.update({
                        "center_x": format_float(center_px[0]),
                        "center_y": format_float(center_px[1]),
                        "square_x": format_float(square_px[0]),
                        "square_y": format_float(square_px[1]),
                        "north_x": format_float(dir_points["N"][0]),
                        "north_y": format_float(dir_points["N"][1]),
                        "east_x": format_float(dir_points["E"][0]),
                        "east_y": format_float(dir_points["E"][1]),
                        "south_x": format_float(dir_points["S"][0]),
                        "south_y": format_float(dir_points["S"][1]),
                        "west_x": format_float(dir_points["W"][0]),
                        "west_y": format_float(dir_points["W"][1]),
                        "angle_deg": format_float(dir_points["angle_deg"]),
                    })

                else:
                    fail_drawn = draw_fail_image(image, {
                        "status": "FAIL_BAD_DIRECTION_POINTS",
                        "reason": "make_direction_points returned None",
                        "detected_classes": detected_classes,
                    })
                    cv2.imwrite(str(image_out_dir / f"{stem}_orientation.jpg"), fail_drawn)
                    cv2.imwrite(str(debug_out_dir / f"{stem}_debug.jpg"), fail_drawn)

            else:
                fail_drawn = draw_fail_image(image, result)
                cv2.imwrite(str(image_out_dir / f"{stem}_orientation.jpg"), fail_drawn)
                cv2.imwrite(str(debug_out_dir / f"{stem}_debug.jpg"), fail_drawn)

            writer.writerow(row)

    print("[DONE] GT-template 2D orientation 완료")
    print(f"[DONE] OUTPUT_DIR: {out_dir}")
    print(f"[DONE] images_orientation: {image_out_dir}")
    print(f"[DONE] images_debug_selection: {debug_out_dir}")
    print(f"[DONE] CSV: {result_csv}")
    print(f"[DONE] template CSV: {template_csv}")
    print("[DONE] status summary:")

    for k, v in sorted(status_counter.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
