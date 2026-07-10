from pathlib import Path
import argparse
import csv
import math
from itertools import product

import cv2
import numpy as np


# ============================================================
# YOLO 실해역 예측 결과 후처리 코드
#
# 목적:
#   YOLO detect predict 결과에서 발생하는 문제를 줄인다.
#
# 해결 대상:
#   1. 하나의 실제 ID에 여러 class가 동시에 부여되는 문제
#   2. 같은 class가 여러 위치에 중복 검출되는 문제
#   3. 노이즈 조각 bbox가 최종 방향추정에 들어가는 문제
#
# 입력:
#   real_images/*.png
#   result/predict_xxx/labels/*.txt
#
# 출력:
#   postprocess_xxx/selected_labels/*.txt
#   postprocess_xxx/selected_vis/*.png
#   postprocess_xxx/postprocess_summary.csv
#
# 최종 선택 규칙:
#   class 0 = 정사각형 ID
#   class 1 = 정사각형 기준 시계방향 첫 번째 직사각형
#   class 2 = 정사각형 기준 시계방향 두 번째 직사각형
#   class 3 = 정사각형 기준 시계방향 세 번째 직사각형
# ============================================================


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


CLASS_NAMES = {
    0: "square_id",
    1: "clockwise_id_1",
    2: "clockwise_id_2",
    3: "clockwise_id_3",
}


COLORS = {
    0: (255, 0, 0),      # blue
    1: (255, 255, 0),    # cyan
    2: (255, 255, 255),  # white
    3: (0, 255, 255),    # yellow
}


def collect_images(image_dir: Path):
    """이미지 파일 수집"""
    images = []

    for path in image_dir.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            images.append(path)

    return sorted(images)


def read_yolo_prediction_txt(txt_path: Path):
    """
    YOLO predict txt 읽기

    지원 형식:
        class x_center y_center width height
        class x_center y_center width height confidence
    """
    preds = []

    if not txt_path.exists():
        return preds

    with open(txt_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            parts = line.strip().split()

            if len(parts) not in [5, 6]:
                continue

            class_id = int(float(parts[0]))
            x_center = float(parts[1])
            y_center = float(parts[2])
            box_w = float(parts[3])
            box_h = float(parts[4])
            conf = float(parts[5]) if len(parts) == 6 else 1.0

            if class_id not in [0, 1, 2, 3]:
                continue

            preds.append({
                "id": f"{txt_path.stem}_{line_idx}",
                "class_id": class_id,
                "x_center": x_center,
                "y_center": y_center,
                "box_w": box_w,
                "box_h": box_h,
                "conf": conf,
            })

    return preds


def norm_to_xyxy(pred, image_w: int, image_h: int):
    """정규화 bbox를 pixel 좌표로 변환"""
    x_center = pred["x_center"] * image_w
    y_center = pred["y_center"] * image_h
    box_w = pred["box_w"] * image_w
    box_h = pred["box_h"] * image_h

    x1 = int(round(x_center - box_w / 2.0))
    y1 = int(round(y_center - box_h / 2.0))
    x2 = int(round(x_center + box_w / 2.0))
    y2 = int(round(y_center + box_h / 2.0))

    x1 = max(0, min(image_w - 1, x1))
    y1 = max(0, min(image_h - 1, y1))
    x2 = max(0, min(image_w - 1, x2))
    y2 = max(0, min(image_h - 1, y2))

    return x1, y1, x2, y2


def box_area(pred):
    """정규화 bbox 면적"""
    return max(0.0, pred["box_w"]) * max(0.0, pred["box_h"])


def box_aspect(pred):
    """bbox 장축/단축 비율"""
    w = max(pred["box_w"], 1e-6)
    h = max(pred["box_h"], 1e-6)
    return max(w / h, h / w)


def iou_box(a, b):
    """정규화 bbox IoU 계산"""
    ax1 = a["x_center"] - a["box_w"] / 2.0
    ay1 = a["y_center"] - a["box_h"] / 2.0
    ax2 = a["x_center"] + a["box_w"] / 2.0
    ay2 = a["y_center"] + a["box_h"] / 2.0

    bx1 = b["x_center"] - b["box_w"] / 2.0
    by1 = b["y_center"] - b["box_h"] / 2.0
    bx2 = b["x_center"] + b["box_w"] / 2.0
    by2 = b["y_center"] + b["box_h"] / 2.0

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    inter = iw * ih
    union = box_area(a) + box_area(b) - inter

    if union <= 0:
        return 0.0

    return inter / union


def center_dist(a, b):
    """bbox 중심 거리"""
    dx = a["x_center"] - b["x_center"]
    dy = a["y_center"] - b["y_center"]
    return math.sqrt(dx * dx + dy * dy)


def angle_clockwise_from_center(pred, center_x: float, center_y: float):
    """
    이미지 좌표계 기준 시계방향 각도 계산
    x 오른쪽 증가, y 아래쪽 증가
    """
    dx = pred["x_center"] - center_x
    dy = pred["y_center"] - center_y
    return math.atan2(dy, dx) % (2.0 * math.pi)


def angle_diff_positive(angle: float):
    """0~2pi 범위로 정규화"""
    return angle % (2.0 * math.pi)


def square_score(pred):
    """
    class 0 후보용 정사각형 점수
    1에 가까울수록 좋음
    """
    aspect = box_aspect(pred)
    score = 1.0 / (1.0 + abs(aspect - 1.0))
    return score


def rectangle_score(pred):
    """
    class 1~3 후보용 직사각형 점수
    길쭉할수록 어느 정도 유리하지만 과도하면 감점
    """
    aspect = box_aspect(pred)

    # 직사각형이면 보통 1.6 이상이 유리
    if aspect < 1.2:
        return 0.45

    if aspect <= 4.5:
        return min(1.0, aspect / 3.0)

    return 0.65


def filter_candidates(preds, args):
    """confidence, 면적 기준 1차 후보 필터링"""
    filtered = []

    for p in preds:
        class_id = p["class_id"]

        if class_id == 0:
            if p["conf"] < args.conf0:
                continue
        else:
            if p["conf"] < args.conf_rect:
                continue

        area = box_area(p)

        if area < args.min_area:
            continue

        if area > args.max_area:
            continue

        filtered.append(p)

    return filtered


def keep_topk_by_class(preds, topk: int):
    """class별 confidence 상위 후보만 유지"""
    by_class = {0: [], 1: [], 2: [], 3: []}

    for p in preds:
        by_class[p["class_id"]].append(p)

    for class_id in by_class:
        by_class[class_id] = sorted(
            by_class[class_id],
            key=lambda x: x["conf"],
            reverse=True
        )[:topk]

    return by_class


def compute_combination_score(combo, args):
    """
    하나의 조합 class0~3에 대한 구조 점수 계산

    combo:
        {0: pred0, 1: pred1, 2: pred2, 3: pred3}
    """
    preds = [combo[i] for i in [0, 1, 2, 3]]

    score = 0.0
    penalty = 0.0

    # ------------------------------------------------------------
    # 1. confidence 점수
    # ------------------------------------------------------------
    conf_score = sum(p["conf"] for p in preds) / 4.0
    score += args.w_conf * conf_score

    # ------------------------------------------------------------
    # 2. class0 정사각형성
    # ------------------------------------------------------------
    score += args.w_square * square_score(combo[0])

    # ------------------------------------------------------------
    # 3. class1~3 직사각형성
    # ------------------------------------------------------------
    rect_score = sum(rectangle_score(combo[i]) for i in [1, 2, 3]) / 3.0
    score += args.w_rect * rect_score

    # ------------------------------------------------------------
    # 4. 중복 bbox / 같은 실제 ID에 여러 class가 붙는 현상 감점
    # ------------------------------------------------------------
    for i in range(4):
        for j in range(i + 1, 4):
            pi = preds[i]
            pj = preds[j]

            iou = iou_box(pi, pj)
            dist = center_dist(pi, pj)

            if iou > args.max_pair_iou:
                penalty += args.w_overlap * (iou - args.max_pair_iou)

            if dist < args.min_center_dist:
                penalty += args.w_close * (args.min_center_dist - dist)

    # ------------------------------------------------------------
    # 5. 정사각형 기준 시계방향 class 순서 점수
    # ------------------------------------------------------------
    cx = sum(p["x_center"] for p in preds) / 4.0
    cy = sum(p["y_center"] for p in preds) / 4.0

    square_angle = angle_clockwise_from_center(combo[0], cx, cy)

    rel_angles = {}

    for class_id in [1, 2, 3]:
        ang = angle_clockwise_from_center(combo[class_id], cx, cy)
        rel_angles[class_id] = angle_diff_positive(ang - square_angle)

    # 기대 조건:
    # class1 < class2 < class3
    order_penalty = 0.0

    if not (rel_angles[1] < rel_angles[2] < rel_angles[3]):
        order_penalty += 1.0

    # 각도가 너무 몰려 있으면 감점
    angle_values = [rel_angles[1], rel_angles[2], rel_angles[3]]
    angle_gaps = [
        angle_values[1] - angle_values[0],
        angle_values[2] - angle_values[1],
    ]

    for gap in angle_gaps:
        if gap < args.min_angle_gap_rad:
            order_penalty += (args.min_angle_gap_rad - gap)

    penalty += args.w_order * order_penalty

    # ------------------------------------------------------------
    # 6. class0 크기 과대 감점
    #    정사각형 ID는 보통 직사각형 ID보다 작아야 함
    # ------------------------------------------------------------
    area0 = box_area(combo[0])
    rect_areas = [box_area(combo[i]) for i in [1, 2, 3]]
    median_rect_area = float(np.median(rect_areas))

    if median_rect_area > 0:
        ratio = area0 / median_rect_area

        if ratio > args.max_square_rect_area_ratio:
            penalty += args.w_area_ratio * (ratio - args.max_square_rect_area_ratio)

    # ------------------------------------------------------------
    # 최종 점수
    # ------------------------------------------------------------
    final_score = score - penalty

    return final_score, {
        "score": score,
        "penalty": penalty,
        "conf_score": conf_score,
        "rect_score": rect_score,
        "square_score": square_score(combo[0]),
        "rel_angle_1": rel_angles[1],
        "rel_angle_2": rel_angles[2],
        "rel_angle_3": rel_angles[3],
    }


def select_best_structural_combo(preds, args):
    """
    YOLO 후보들 중 class0~3 구조적으로 가장 타당한 조합 선택
    """
    filtered = filter_candidates(preds, args)
    by_class = keep_topk_by_class(filtered, args.topk)

    # class0 후보가 없으면 실패
    if len(by_class[0]) == 0:
        return None, "no_class0_candidate", by_class, None

    # class1~3 후보가 모두 있어야 4개 조합 가능
    for class_id in [1, 2, 3]:
        if len(by_class[class_id]) == 0:
            return None, f"no_class{class_id}_candidate", by_class, None

    best_combo = None
    best_score = -1e9
    best_detail = None

    for p0, p1, p2, p3 in product(
        by_class[0],
        by_class[1],
        by_class[2],
        by_class[3],
    ):
        ids = [p0["id"], p1["id"], p2["id"], p3["id"]]

        # 같은 bbox 후보가 중복 사용되는 경우 방지
        if len(set(ids)) < 4:
            continue

        combo = {
            0: p0,
            1: p1,
            2: p2,
            3: p3,
        }

        combo_score, detail = compute_combination_score(combo, args)

        if combo_score > best_score:
            best_score = combo_score
            best_combo = combo
            best_detail = detail
            best_detail["final_score"] = combo_score

    if best_combo is None:
        return None, "no_valid_combo", by_class, None

    return best_combo, "ok", by_class, best_detail


def write_selected_txt(out_txt_path: Path, combo):
    """최종 선택 bbox txt 저장"""
    lines = []

    for class_id in [0, 1, 2, 3]:
        p = combo[class_id]

        lines.append(
            f"{class_id} "
            f"{p['x_center']:.6f} "
            f"{p['y_center']:.6f} "
            f"{p['box_w']:.6f} "
            f"{p['box_h']:.6f} "
            f"{p['conf']:.6f}"
        )

    with open(out_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def draw_candidates(image, preds, image_w: int, image_h: int):
    """전체 YOLO 후보 시각화"""
    vis = image.copy()

    for p in preds:
        class_id = p["class_id"]
        x1, y1, x2, y2 = norm_to_xyxy(p, image_w, image_h)

        color = COLORS.get(class_id, (0, 255, 0))

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1)
        cv2.putText(
            vis,
            f"{class_id}:{p['conf']:.2f}",
            (x1, max(15, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA
        )

    return vis


def draw_selected(image, combo, image_w: int, image_h: int):
    """후처리 최종 선택 bbox 시각화"""
    vis = image.copy()

    centers = []

    for class_id in [0, 1, 2, 3]:
        p = combo[class_id]
        x1, y1, x2, y2 = norm_to_xyxy(p, image_w, image_h)

        color = COLORS.get(class_id, (0, 255, 0))

        cx = int(round(p["x_center"] * image_w))
        cy = int(round(p["y_center"] * image_h))

        centers.append((class_id, cx, cy))

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.circle(vis, (cx, cy), 4, color, -1)

        cv2.putText(
            vis,
            f"class {class_id} {p['conf']:.2f}",
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA
        )

    # 중심점 연결선 표시
    centers_sorted = sorted(centers, key=lambda x: x[0])

    for i in range(len(centers_sorted)):
        c1 = centers_sorted[i]
        c2 = centers_sorted[(i + 1) % len(centers_sorted)]

        cv2.line(
            vis,
            (c1[1], c1[2]),
            (c2[1], c2[2]),
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )

    return vis


def process_one_image(image_path: Path, pred_label_dir: Path, out_dirs: dict, args):
    """이미지 1장 후처리"""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image is None:
        raise RuntimeError(f"이미지를 읽지 못했습니다: {image_path}")

    image_h, image_w = image.shape[:2]
    stem = image_path.stem

    pred_txt_path = pred_label_dir / f"{stem}.txt"
    preds = read_yolo_prediction_txt(pred_txt_path)

    if len(preds) == 0:
        return {
            "stem": stem,
            "status": "no_prediction_txt_or_empty",
            "num_raw": 0,
            "num_filtered": 0,
            "selected": False,
        }

    combo, status, by_class, detail = select_best_structural_combo(preds, args)

    raw_vis = draw_candidates(image, preds, image_w, image_h)
    cv2.imwrite(str(out_dirs["candidate_vis"] / f"{stem}_candidates.png"), raw_vis)

    if combo is None:
        return {
            "stem": stem,
            "status": status,
            "num_raw": len(preds),
            "num_filtered": sum(len(v) for v in by_class.values()),
            "selected": False,
        }

    out_txt_path = out_dirs["selected_labels"] / f"{stem}.txt"
    write_selected_txt(out_txt_path, combo)

    selected_vis = draw_selected(image, combo, image_w, image_h)
    cv2.imwrite(str(out_dirs["selected_vis"] / f"{stem}_selected.png"), selected_vis)

    result = {
        "stem": stem,
        "status": "ok",
        "num_raw": len(preds),
        "num_filtered": sum(len(v) for v in by_class.values()),
        "selected": True,
    }

    for class_id in [0, 1, 2, 3]:
        p = combo[class_id]
        result[f"class{class_id}_conf"] = p["conf"]
        result[f"class{class_id}_x"] = p["x_center"]
        result[f"class{class_id}_y"] = p["y_center"]
        result[f"class{class_id}_w"] = p["box_w"]
        result[f"class{class_id}_h"] = p["box_h"]

    if detail is not None:
        for k, v in detail.items():
            result[k] = v

    return result


def save_summary_csv(summary_path: Path, rows):
    """후처리 요약 CSV 저장"""
    if len(rows) == 0:
        return

    fieldnames = sorted(set().union(*[row.keys() for row in rows]))

    with open(summary_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image_dir",
        type=str,
        default="../real_images",
        help="실해역 이미지 폴더"
    )

    parser.add_argument(
        "--pred_label_dir",
        type=str,
        required=True,
        help="YOLO predict 결과 labels 폴더"
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="후처리 결과 저장 폴더"
    )

    # ------------------------------------------------------------
    # 후보 필터 파라미터
    # ------------------------------------------------------------
    parser.add_argument("--conf0", type=float, default=0.20, help="class0 최소 confidence")
    parser.add_argument("--conf_rect", type=float, default=0.25, help="class1~3 최소 confidence")
    parser.add_argument("--min_area", type=float, default=0.0002, help="최소 bbox area")
    parser.add_argument("--max_area", type=float, default=0.45, help="최대 bbox area")
    parser.add_argument("--topk", type=int, default=8, help="class별 유지할 후보 수")

    # ------------------------------------------------------------
    # 구조 점수 파라미터
    # ------------------------------------------------------------
    parser.add_argument("--min_center_dist", type=float, default=0.045, help="서로 다른 ID 중심 최소 거리")
    parser.add_argument("--max_pair_iou", type=float, default=0.35, help="서로 다른 ID bbox 최대 허용 IoU")
    parser.add_argument("--min_angle_gap_deg", type=float, default=20.0, help="class1~3 사이 최소 각도 차이")
    parser.add_argument("--max_square_rect_area_ratio", type=float, default=1.20, help="class0 area / rect median area 최대 권장값")

    # ------------------------------------------------------------
    # 점수 가중치
    # ------------------------------------------------------------
    parser.add_argument("--w_conf", type=float, default=2.0)
    parser.add_argument("--w_square", type=float, default=0.7)
    parser.add_argument("--w_rect", type=float, default=0.6)
    parser.add_argument("--w_overlap", type=float, default=4.0)
    parser.add_argument("--w_close", type=float, default=5.0)
    parser.add_argument("--w_order", type=float, default=2.5)
    parser.add_argument("--w_area_ratio", type=float, default=0.8)

    args = parser.parse_args()

    args.min_angle_gap_rad = math.radians(args.min_angle_gap_deg)

    image_dir = Path(args.image_dir)
    pred_label_dir = Path(args.pred_label_dir)
    out_dir = Path(args.out_dir)

    if not image_dir.exists():
        raise FileNotFoundError(f"image_dir 없음: {image_dir}")

    if not pred_label_dir.exists():
        raise FileNotFoundError(f"pred_label_dir 없음: {pred_label_dir}")

    out_dirs = {
        "selected_labels": out_dir / "selected_labels",
        "selected_vis": out_dir / "selected_vis",
        "candidate_vis": out_dir / "candidate_vis",
    }

    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    images = collect_images(image_dir)

    print("========== CONFIG ==========")
    print(f"image_dir:      {image_dir.resolve()}")
    print(f"pred_label_dir: {pred_label_dir.resolve()}")
    print(f"out_dir:        {out_dir.resolve()}")
    print(f"image_count:    {len(images)}")
    print(f"conf0:          {args.conf0}")
    print(f"conf_rect:      {args.conf_rect}")
    print("============================")

    rows = []
    ok_count = 0
    fail_count = 0

    for idx, image_path in enumerate(images):
        try:
            row = process_one_image(
                image_path=image_path,
                pred_label_dir=pred_label_dir,
                out_dirs=out_dirs,
                args=args
            )

            rows.append(row)

            if row.get("selected", False):
                ok_count += 1
                print(f"[OK] {idx + 1}/{len(images)} {image_path.stem}")
            else:
                fail_count += 1
                print(f"[FAIL] {idx + 1}/{len(images)} {image_path.stem}: {row['status']}")

        except Exception as e:
            fail_count += 1
            rows.append({
                "stem": image_path.stem,
                "status": f"exception: {e}",
                "selected": False,
            })
            print(f"[EXCEPTION] {idx + 1}/{len(images)} {image_path.stem}: {e}")

    summary_path = out_dir / "postprocess_summary.csv"
    save_summary_csv(summary_path, rows)

    print("\n========== RESULT ==========")
    print(f"total images: {len(images)}")
    print(f"selected ok:  {ok_count}")
    print(f"failed:       {fail_count}")
    print(f"summary csv:  {summary_path}")
    print("============================")


if __name__ == "__main__":
    main()