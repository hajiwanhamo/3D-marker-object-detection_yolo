from pathlib import Path
import argparse
import csv
import math
from itertools import combinations

import cv2
import numpy as np


# ============================================================
# 2-class YOLO 실해역 후처리 코드
#
# YOLO 예측 class:
#   0 = square_id
#   1 = rect_id
#
# 최종 출력 class:
#   0 = square_id
#   1 = square 기준 시계방향 첫 번째 rect
#   2 = square 기준 시계방향 두 번째 rect
#   3 = square 기준 시계방향 세 번째 rect
#
# 중요 조건:
#   - YOLO 모델과 YOLO detect 결과를 유지한다.
#   - square는 반드시 YOLO class 0 후보에서만 선택한다.
#   - rect는 반드시 YOLO class 1 후보에서만 선택한다.
#   - rect 3개의 세부 class 1,2,3만 후처리에서 시계방향으로 재부여한다.
# ============================================================

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

COLORS = {
    0: (255, 0, 0),      # class 0 square: blue
    1: (255, 255, 0),    # class 1 rect: cyan
    2: (255, 255, 255),  # class 2 rect: white
    3: (0, 255, 255),    # class 3 rect: yellow
}


def collect_images(image_dir: Path):
    """실해역 이미지 파일 목록 수집"""
    if not image_dir.exists():
        raise FileNotFoundError(f"image_dir 없음: {image_dir}")

    images = []

    for p in image_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            images.append(p)

    return sorted(images)


def read_yolo_txt(txt_path: Path):
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

            # 2-class 모델 결과만 사용
            if class_id not in [0, 1]:
                continue

            preds.append({
                "id": f"{txt_path.stem}_{line_idx}",
                "class_id": class_id,
                "x": float(parts[1]),
                "y": float(parts[2]),
                "w": float(parts[3]),
                "h": float(parts[4]),
                "conf": float(parts[5]) if len(parts) == 6 else 1.0,
            })

    return preds


def box_area(p):
    """정규화 bbox 면적 계산"""
    return max(0.0, p["w"]) * max(0.0, p["h"])


def box_aspect(p):
    """bbox 장축/단축 비율 계산"""
    w = max(p["w"], 1e-8)
    h = max(p["h"], 1e-8)
    return max(w / h, h / w)


def norm_to_xyxy(p, image_w: int, image_h: int):
    """정규화 bbox를 pixel 좌표로 변환"""
    x1 = int(round((p["x"] - p["w"] / 2.0) * image_w))
    y1 = int(round((p["y"] - p["h"] / 2.0) * image_h))
    x2 = int(round((p["x"] + p["w"] / 2.0) * image_w))
    y2 = int(round((p["y"] + p["h"] / 2.0) * image_h))

    x1 = max(0, min(image_w - 1, x1))
    y1 = max(0, min(image_h - 1, y1))
    x2 = max(0, min(image_w - 1, x2))
    y2 = max(0, min(image_h - 1, y2))

    return x1, y1, x2, y2


def iou(a, b):
    """정규화 bbox IoU 계산"""
    ax1 = a["x"] - a["w"] / 2.0
    ay1 = a["y"] - a["h"] / 2.0
    ax2 = a["x"] + a["w"] / 2.0
    ay2 = a["y"] + a["h"] / 2.0

    bx1 = b["x"] - b["w"] / 2.0
    by1 = b["y"] - b["h"] / 2.0
    bx2 = b["x"] + b["w"] / 2.0
    by2 = b["y"] + b["h"] / 2.0

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
    """bbox 중심점 거리 계산"""
    dx = a["x"] - b["x"]
    dy = a["y"] - b["y"]
    return math.sqrt(dx * dx + dy * dy)


def square_shape_score(p):
    """정사각형성 점수. 1에 가까울수록 정사각형에 가까움."""
    aspect = box_aspect(p)
    return 1.0 / (1.0 + abs(aspect - 1.0))


def rect_shape_score(p):
    """직사각형성 점수"""
    aspect = box_aspect(p)

    if aspect < 1.15:
        return 0.25

    if aspect <= 4.5:
        return min(1.0, aspect / 3.0)

    return 0.65


def clockwise_angle(p, center_x: float, center_y: float):
    """
    이미지 좌표계 기준 시계방향 각도 계산
    x는 오른쪽 증가, y는 아래쪽 증가
    """
    dx = p["x"] - center_x
    dy = p["y"] - center_y
    return math.atan2(dy, dx) % (2.0 * math.pi)


def filter_candidates(preds, args):
    """
    YOLO class를 유지한 상태로 square 후보와 rect 후보 분리
    class 0만 square 후보
    class 1만 rect 후보
    """
    square_candidates = []
    rect_candidates = []

    for p in preds:
        area = box_area(p)
        aspect = box_aspect(p)

        if area < args.min_area or area > args.max_area:
            continue

        if p["class_id"] == 0:
            if p["conf"] < args.conf_square:
                continue

            if aspect > args.square_max_aspect:
                continue

            if area > args.square_max_area:
                continue

            square_candidates.append(p)

        elif p["class_id"] == 1:
            if p["conf"] < args.conf_rect:
                continue

            rect_candidates.append(p)

    square_candidates = sorted(
        square_candidates,
        key=lambda p: (
            args.w_conf * p["conf"]
            + args.w_square_shape * square_shape_score(p)
            - args.w_square_area * box_area(p)
        ),
        reverse=True
    )[:args.topk_square]

    rect_candidates = sorted(
        rect_candidates,
        key=lambda p: (
            args.w_conf * p["conf"]
            + args.w_rect_shape * rect_shape_score(p)
            - args.w_rect_area * box_area(p)
        ),
        reverse=True
    )[:args.topk_rect]

    return square_candidates, rect_candidates


def score_combo(square, rects, args):
    """
    square 1개 + rect 3개 조합 점수 계산
    """
    all_items = [square] + list(rects)

    score = 0.0
    penalty = 0.0

    square_area = box_area(square)
    rect_areas = [box_area(r) for r in rects]
    rect_median_area = float(np.median(rect_areas)) if len(rect_areas) > 0 else 0.0

    if rect_median_area <= 0:
        return -1e9, None, {"fail_reason": "invalid_rect_area"}

    square_rect_ratio = square_area / rect_median_area

    # square가 rect보다 너무 크면 조합 제외
    if square_rect_ratio > args.hard_max_square_rect_area_ratio:
        return -1e9, None, {
            "fail_reason": "square_too_large",
            "square_rect_ratio": square_rect_ratio,
        }

    # confidence 점수
    conf_score = sum(p["conf"] for p in all_items) / 4.0
    score += args.w_conf * conf_score

    # shape 점수
    sq_shape = square_shape_score(square)
    rect_shape = sum(rect_shape_score(r) for r in rects) / 3.0

    score += args.w_square_shape * sq_shape
    score += args.w_rect_shape * rect_shape

    # square 면적 감점
    penalty += args.w_square_area * square_area

    if square_rect_ratio > args.soft_max_square_rect_area_ratio:
        penalty += args.w_area_ratio * (square_rect_ratio - args.soft_max_square_rect_area_ratio)

    # 서로 다른 후보 간 중복/근접 감점
    max_pair_iou = 0.0
    min_pair_dist = 999.0

    for a, b in combinations(all_items, 2):
        d = center_dist(a, b)
        ov = iou(a, b)

        max_pair_iou = max(max_pair_iou, ov)
        min_pair_dist = min(min_pair_dist, d)

        if ov > args.max_iou:
            penalty += args.w_overlap * (ov - args.max_iou)

        if d < args.min_center_dist:
            penalty += args.w_close * (args.min_center_dist - d)

    # rect끼리 많이 겹치면 감점
    for a, b in combinations(rects, 2):
        ov = iou(a, b)

        if ov > args.rect_rect_max_iou:
            penalty += args.w_rect_rect_overlap * (ov - args.rect_rect_max_iou)

    # square와 rect가 많이 겹치면 감점
    for r in rects:
        ov = iou(square, r)

        if ov > args.square_rect_max_iou:
            penalty += args.w_square_rect_overlap * (ov - args.square_rect_max_iou)

    # square 기준 시계방향 정렬
    center_x = sum(p["x"] for p in all_items) / 4.0
    center_y = sum(p["y"] for p in all_items) / 4.0

    square_angle = clockwise_angle(square, center_x, center_y)

    rel_rects = []

    for r in rects:
        rel_angle = (clockwise_angle(r, center_x, center_y) - square_angle) % (2.0 * math.pi)
        rel_rects.append((rel_angle, r))

    rel_rects = sorted(rel_rects, key=lambda x: x[0])
    rel_angles = [x[0] for x in rel_rects]

    min_angle_gap = 999.0

    for i in range(len(rel_angles) - 1):
        gap = rel_angles[i + 1] - rel_angles[i]
        min_angle_gap = min(min_angle_gap, gap)

        if gap < args.min_angle_gap_rad:
            penalty += args.w_angle_gap * (args.min_angle_gap_rad - gap)

    final_score = score - penalty

    detail = {
        "score": score,
        "penalty": penalty,
        "final_score": final_score,
        "conf_score": conf_score,
        "square_shape": sq_shape,
        "rect_shape": rect_shape,
        "square_area": square_area,
        "rect_median_area": rect_median_area,
        "square_rect_ratio": square_rect_ratio,
        "max_pair_iou": max_pair_iou,
        "min_pair_dist": min_pair_dist,
        "min_angle_gap_deg": math.degrees(min_angle_gap) if min_angle_gap < 900 else 999.0,
    }

    return final_score, rel_rects, detail


def select_best(preds, args):
    """최적 square 1개 + rect 3개 선택"""
    square_candidates, rect_candidates = filter_candidates(preds, args)

    if len(square_candidates) == 0:
        return None, "no_square_candidate", square_candidates, rect_candidates, None

    if len(rect_candidates) < 3:
        return None, "less_than_3_rect_candidates", square_candidates, rect_candidates, None

    best_selected = None
    best_score = -1e9
    best_detail = None

    for square in square_candidates:
        for rects in combinations(rect_candidates, 3):
            combo_score, rel_rects, detail = score_combo(square, rects, args)

            if combo_score > best_score:
                best_score = combo_score
                best_detail = detail

                if rel_rects is not None:
                    best_selected = {
                        0: square,
                        1: rel_rects[0][1],
                        2: rel_rects[1][1],
                        3: rel_rects[2][1],
                    }

    if best_selected is None:
        return None, "no_valid_combo", square_candidates, rect_candidates, best_detail

    if best_score < args.min_final_score:
        return None, "low_final_score", square_candidates, rect_candidates, best_detail

    return best_selected, "ok", square_candidates, rect_candidates, best_detail


def write_final_txt(path: Path, selected):
    """최종 4-class 결과 txt 저장"""
    lines = []

    for final_class in [0, 1, 2, 3]:
        p = selected[final_class]

        lines.append(
            f"{final_class} "
            f"{p['x']:.6f} "
            f"{p['y']:.6f} "
            f"{p['w']:.6f} "
            f"{p['h']:.6f} "
            f"{p['conf']:.6f}"
        )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def draw_candidates(image, preds, image_w: int, image_h: int):
    """YOLO 원본 후보 시각화"""
    vis = image.copy()

    for p in preds:
        x1, y1, x2, y2 = norm_to_xyxy(p, image_w, image_h)
        color = (255, 0, 0) if p["class_id"] == 0 else (0, 255, 255)

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1)
        cv2.putText(
            vis,
            f"{p['class_id']}:{p['conf']:.2f}",
            (x1, max(15, y1 - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA
        )

    return vis


def draw_selected(image, selected, image_w: int, image_h: int):
    """후처리 최종 선택 결과 시각화"""
    vis = image.copy()
    centers = []

    for final_class in [0, 1, 2, 3]:
        p = selected[final_class]
        x1, y1, x2, y2 = norm_to_xyxy(p, image_w, image_h)

        color = COLORS[final_class]
        cx = int(round(p["x"] * image_w))
        cy = int(round(p["y"] * image_h))

        centers.append((final_class, cx, cy))

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.circle(vis, (cx, cy), 4, color, -1)

        cv2.putText(
            vis,
            f"class {final_class} {p['conf']:.2f}",
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA
        )

    centers = sorted(centers, key=lambda x: x[0])

    for i in range(len(centers)):
        a = centers[i]
        b = centers[(i + 1) % len(centers)]

        cv2.line(
            vis,
            (a[1], a[2]),
            (b[1], b[2]),
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )

    return vis


def process_one(image_path: Path, label_dir: Path, out_dirs: dict, args):
    """이미지 1장 후처리"""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image is None:
        raise RuntimeError(f"이미지를 읽지 못했습니다: {image_path}")

    image_h, image_w = image.shape[:2]
    stem = image_path.stem

    pred_txt = label_dir / f"{stem}.txt"
    preds = read_yolo_txt(pred_txt)

    row = {
        "stem": stem,
        "num_raw": len(preds),
        "selected": False,
    }

    if len(preds) == 0:
        row["status"] = "no_prediction"
        return row

    candidate_vis = draw_candidates(image, preds, image_w, image_h)
    cv2.imwrite(str(out_dirs["candidate_vis"] / f"{stem}_candidates.png"), candidate_vis)

    selected, status, square_candidates, rect_candidates, detail = select_best(preds, args)

    row["status"] = status
    row["num_square_candidates"] = len(square_candidates)
    row["num_rect_candidates"] = len(rect_candidates)

    if detail:
        row.update(detail)

    if selected is None:
        return row

    row["selected"] = True

    out_txt = out_dirs["selected_labels"] / f"{stem}.txt"
    write_final_txt(out_txt, selected)

    selected_vis = draw_selected(image, selected, image_w, image_h)
    cv2.imwrite(str(out_dirs["selected_vis"] / f"{stem}_selected.png"), selected_vis)

    for final_class in [0, 1, 2, 3]:
        p = selected[final_class]

        row[f"class{final_class}_x"] = p["x"]
        row[f"class{final_class}_y"] = p["y"]
        row[f"class{final_class}_w"] = p["w"]
        row[f"class{final_class}_h"] = p["h"]
        row[f"class{final_class}_conf"] = p["conf"]
        row[f"class{final_class}_area"] = box_area(p)
        row[f"class{final_class}_aspect"] = box_aspect(p)

    return row


def save_csv(path: Path, rows):
    """summary csv 저장"""
    if not rows:
        return

    fieldnames = sorted(set().union(*[row.keys() for row in rows]))

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_dir", type=str, default="../real_images")
    parser.add_argument("--pred_label_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--conf_square", type=float, default=0.08)
    parser.add_argument("--conf_rect", type=float, default=0.12)
    parser.add_argument("--min_area", type=float, default=0.0002)
    parser.add_argument("--max_area", type=float, default=0.50)

    parser.add_argument("--square_max_aspect", type=float, default=1.90)
    parser.add_argument("--square_max_area", type=float, default=0.10)

    parser.add_argument("--topk_square", type=int, default=10)
    parser.add_argument("--topk_rect", type=int, default=20)

    parser.add_argument("--min_center_dist", type=float, default=0.035)
    parser.add_argument("--max_iou", type=float, default=0.45)
    parser.add_argument("--rect_rect_max_iou", type=float, default=0.40)
    parser.add_argument("--square_rect_max_iou", type=float, default=0.30)
    parser.add_argument("--min_angle_gap_deg", type=float, default=15.0)

    parser.add_argument("--soft_max_square_rect_area_ratio", type=float, default=1.00)
    parser.add_argument("--hard_max_square_rect_area_ratio", type=float, default=1.50)

    parser.add_argument("--min_final_score", type=float, default=0.00)

    parser.add_argument("--w_conf", type=float, default=2.0)
    parser.add_argument("--w_square_shape", type=float, default=1.2)
    parser.add_argument("--w_rect_shape", type=float, default=0.7)
    parser.add_argument("--w_square_area", type=float, default=2.0)
    parser.add_argument("--w_rect_area", type=float, default=0.3)
    parser.add_argument("--w_close", type=float, default=5.0)
    parser.add_argument("--w_overlap", type=float, default=4.0)
    parser.add_argument("--w_rect_rect_overlap", type=float, default=5.0)
    parser.add_argument("--w_square_rect_overlap", type=float, default=6.0)
    parser.add_argument("--w_angle_gap", type=float, default=2.0)
    parser.add_argument("--w_area_ratio", type=float, default=1.5)

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
        "candidate_vis": out_dir / "candidate_vis",
        "selected_vis": out_dir / "selected_vis",
        "selected_labels": out_dir / "selected_labels",
    }

    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    images = collect_images(image_dir)

    print("========== CONFIG ==========")
    print(f"image_dir:      {image_dir.resolve()}")
    print(f"pred_label_dir: {pred_label_dir.resolve()}")
    print(f"out_dir:        {out_dir.resolve()}")
    print(f"image_count:    {len(images)}")
    print(f"conf_square:    {args.conf_square}")
    print(f"conf_rect:      {args.conf_rect}")
    print("============================")

    rows = []
    ok = 0
    fail = 0

    for idx, image_path in enumerate(images):
        try:
            row = process_one(image_path, pred_label_dir, out_dirs, args)
            rows.append(row)

            if row["selected"]:
                ok += 1
                print(f"[OK] {idx + 1}/{len(images)} {image_path.stem}")
            else:
                fail += 1
                print(f"[FAIL] {idx + 1}/{len(images)} {image_path.stem}: {row['status']}")

        except Exception as e:
            fail += 1
            rows.append({
                "stem": image_path.stem,
                "status": f"exception: {e}",
                "selected": False,
            })
            print(f"[EXCEPTION] {idx + 1}/{len(images)} {image_path.stem}: {e}")

    summary_path = out_dir / "postprocess_2class_summary.csv"
    save_csv(summary_path, rows)

    print("\n========== RESULT ==========")
    print(f"selected ok: {ok}")
    print(f"failed:      {fail}")
    print(f"summary:     {summary_path}")
    print("============================")


if __name__ == "__main__":
    main()