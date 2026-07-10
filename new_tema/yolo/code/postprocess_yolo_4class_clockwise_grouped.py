from pathlib import Path
import argparse
import csv
import math
from itertools import combinations

import cv2
import numpy as np


# ============================================================
# 4-class YOLO 실해역 후처리 - 위치 기반 ID 그룹 병합 버전
#
# 목적:
#   하나의 실제 내부 ID 영역에 class 1/2/3이 동시에 붙는 문제를 줄인다.
#
# 입력 YOLO class:
#   0 = square_id
#   1 = clockwise_id_1
#   2 = clockwise_id_2
#   3 = clockwise_id_3
#
# 후처리 핵심:
#   1. class 0 후보는 square 후보로 분리
#   2. class 1~3 후보는 rect 후보로 통합
#   3. rect 후보들을 위치 기준으로 먼저 그룹화
#      - 같은 실제 ID 위에 붙은 여러 class bbox를 하나의 그룹으로 묶음
#   4. 그룹마다 대표 bbox 1개만 선택
#   5. rect 대표 그룹 3개 선택
#   6. square 기준 시계방향으로 최종 class 1,2,3 재부여
#
# 출력:
#   selected_labels/*.txt
#   selected_vis/*.png
#   grouped_candidate_vis/*.png
#   postprocess_4class_clockwise_grouped_summary.csv
# ============================================================


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

FINAL_COLORS = {
    0: (255, 0, 0),      # square: blue
    1: (255, 255, 0),    # rect1: cyan
    2: (255, 255, 255),  # rect2: white
    3: (0, 255, 255),    # rect3: yellow
}


def collect_images(image_dir: Path):
    """이미지 목록 수집"""
    if not image_dir.exists():
        raise FileNotFoundError(f"image_dir 없음: {image_dir}")

    return sorted([
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ])


def read_yolo_txt(txt_path: Path):
    """
    YOLO txt 읽기.

    지원 형식:
      class x y w h
      class x y w h conf
    """
    preds = []

    if not txt_path.exists():
        return preds

    with open(txt_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            parts = line.strip().split()

            if len(parts) not in [5, 6]:
                continue

            raw_class = int(float(parts[0]))

            if raw_class not in [0, 1, 2, 3]:
                continue

            preds.append({
                "id": f"{txt_path.stem}_{line_idx}",
                "raw_class": raw_class,
                "x": float(parts[1]),
                "y": float(parts[2]),
                "w": float(parts[3]),
                "h": float(parts[4]),
                "conf": float(parts[5]) if len(parts) == 6 else 1.0,
                "line_idx": line_idx,
            })

    return preds


def box_area(p):
    """정규화 bbox 면적"""
    return max(0.0, p["w"]) * max(0.0, p["h"])


def box_aspect(p):
    """bbox 장축/단축 비율"""
    w = max(p["w"], 1e-8)
    h = max(p["h"], 1e-8)
    return max(w / h, h / w)


def norm_to_xyxy(p, image_w: int, image_h: int):
    """YOLO 정규화 bbox를 pixel 좌표로 변환"""
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
    """bbox IoU 계산"""
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
    """bbox 중심 거리"""
    dx = a["x"] - b["x"]
    dy = a["y"] - b["y"]
    return math.sqrt(dx * dx + dy * dy)


def center_inside(a, b):
    """
    a의 중심이 b bbox 내부에 있는지 확인.
    정규화 좌표 기준.
    """
    bx1 = b["x"] - b["w"] / 2.0
    by1 = b["y"] - b["h"] / 2.0
    bx2 = b["x"] + b["w"] / 2.0
    by2 = b["y"] + b["h"] / 2.0

    return (bx1 <= a["x"] <= bx2) and (by1 <= a["y"] <= by2)


def square_shape_score(p):
    """정사각형성 점수"""
    aspect = box_aspect(p)
    return 1.0 / (1.0 + abs(aspect - 1.0))


def rect_shape_score(p):
    """직사각형성 점수"""
    aspect = box_aspect(p)

    if aspect < 1.10:
        return 0.20

    if aspect <= 5.0:
        return min(1.0, aspect / 3.0)

    return 0.60


def area_prior_score(area: float, target: float):
    """목표 면적에 가까울수록 높은 점수"""
    if target <= 0:
        return 0.0

    diff = abs(area - target) / target
    return 1.0 / (1.0 + diff)


def clockwise_angle(p, center_x: float, center_y: float):
    """
    이미지 좌표계 기준 시계방향 각도.
    y가 아래로 증가하므로 atan2(dy, dx)를 사용.
    """
    dx = p["x"] - center_x
    dy = p["y"] - center_y
    return math.atan2(dy, dx) % (2.0 * math.pi)


def square_score(p, args):
    """square 후보 점수"""
    area = box_area(p)

    return (
        args.w_square_conf * p["conf"]
        + args.w_square_shape * square_shape_score(p)
        + args.w_square_area_prior * area_prior_score(area, args.square_target_area)
        - args.w_square_area_penalty * max(0.0, area - args.square_target_area)
    )


def rect_score(p, args):
    """rect 후보 점수"""
    return (
        args.w_rect_conf * p["conf"]
        + args.w_rect_shape * rect_shape_score(p)
        - args.w_rect_area_penalty * box_area(p)
    )


def filter_square_candidates(preds, args):
    """class 0 후보만 square 후보로 사용"""
    out = []

    for p in preds:
        if p["raw_class"] != 0:
            continue

        area = box_area(p)
        aspect = box_aspect(p)

        if p["conf"] < args.conf_square:
            continue

        if area < args.square_min_area:
            continue

        if area > args.square_max_area:
            continue

        if aspect > args.square_max_aspect:
            continue

        out.append(p)

    return sorted(out, key=lambda p: square_score(p, args), reverse=True)[:args.topk_square]


def filter_rect_candidates(preds, args):
    """
    class 1,2,3 후보를 rect 후보로 통합.
    여기서는 raw class를 최종 class로 믿지 않는다.
    """
    out = []

    for p in preds:
        if p["raw_class"] not in [1, 2, 3]:
            continue

        area = box_area(p)

        if p["conf"] < args.conf_rect:
            continue

        if area < args.rect_min_area:
            continue

        if area > args.rect_max_area:
            continue

        out.append(p)

    return sorted(out, key=lambda p: rect_score(p, args), reverse=True)


def same_physical_id(a, b, args):
    """
    두 bbox가 같은 실제 ID 영역을 가리키는지 판단.

    IoU만 쓰면 실해역에서 bbox 크기가 다를 때 실패할 수 있으므로
    중심 거리와 center-inside 조건을 함께 사용한다.
    """
    ov = iou(a, b)
    dist = center_dist(a, b)

    if ov >= args.group_iou:
        return True

    if dist <= args.group_center_dist:
        return True

    # 한 bbox 중심이 다른 bbox 내부에 있고 중심도 너무 멀지 않으면 같은 ID로 간주
    if (center_inside(a, b) or center_inside(b, a)) and dist <= args.group_center_inside_dist:
        return True

    return False


def group_candidates(cands, args):
    """
    후보 bbox들을 위치 기준으로 그룹화.
    같은 실제 ID에 붙은 여러 class bbox를 하나의 그룹으로 묶는다.
    """
    n = len(cands)

    if n == 0:
        return []

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra = find(a)
        rb = find(b)

        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if same_physical_id(cands[i], cands[j], args):
                union(i, j)

    groups_dict = {}

    for i in range(n):
        r = find(i)
        groups_dict.setdefault(r, []).append(cands[i])

    groups = []

    for group_idx, members in enumerate(groups_dict.values()):
        # 대표 bbox는 rect score가 가장 높은 bbox로 선택
        rep = sorted(members, key=lambda p: rect_score(p, args), reverse=True)[0].copy()

        # 그룹 정보 추가
        rep["group_id"] = group_idx
        rep["group_size"] = len(members)
        rep["group_raw_classes"] = ",".join(sorted(set(str(m["raw_class"]) for m in members)))
        rep["group_max_conf"] = max(m["conf"] for m in members)
        rep["group_mean_conf"] = float(np.mean([m["conf"] for m in members]))

        groups.append({
            "group_id": group_idx,
            "members": members,
            "rep": rep,
        })

    groups = sorted(groups, key=lambda g: rect_score(g["rep"], args), reverse=True)

    return groups


def score_combo(square, rect_reps, args):
    """
    square 1개 + rect group 대표 3개 조합 점수 계산.
    """
    all_items = [square] + list(rect_reps)

    score = 0.0
    penalty = 0.0

    square_area = box_area(square)
    rect_areas = [box_area(r) for r in rect_reps]
    rect_median_area = float(np.median(rect_areas)) if len(rect_areas) > 0 else 0.0

    if rect_median_area <= 0:
        return -1e18, None, {"fail_reason": "invalid_rect_area"}

    # square 점수
    score += square_score(square, args)

    # rect 점수
    rect_conf_mean = float(np.mean([r["conf"] for r in rect_reps]))
    rect_shape_mean = float(np.mean([rect_shape_score(r) for r in rect_reps]))
    group_size_mean = float(np.mean([r.get("group_size", 1) for r in rect_reps]))

    score += args.w_rect_conf * rect_conf_mean
    score += args.w_rect_shape * rect_shape_mean

    # 같은 ID에 여러 예측이 모인 그룹은 오히려 대표성이 있다고 보고 약간 가산
    score += args.w_group_size_bonus * min(group_size_mean, args.group_size_bonus_cap)

    # square가 rect보다 너무 크면 제거
    square_rect_ratio = square_area / rect_median_area

    if square_rect_ratio > args.hard_max_square_rect_area_ratio:
        return -1e18, None, {
            "fail_reason": "square_too_large_vs_rect",
            "square_rect_ratio": square_rect_ratio,
        }

    if square_rect_ratio > args.soft_max_square_rect_area_ratio:
        penalty += args.w_area_ratio * (square_rect_ratio - args.soft_max_square_rect_area_ratio)

    max_pair_iou = 0.0
    min_pair_dist = 999.0

    # 선택된 4개가 서로 같은 ID처럼 겹치거나 너무 가까우면 감점
    for a, b in combinations(all_items, 2):
        ov = iou(a, b)
        dist = center_dist(a, b)

        max_pair_iou = max(max_pair_iou, ov)
        min_pair_dist = min(min_pair_dist, dist)

        if ov > args.max_iou:
            penalty += args.w_overlap * (ov - args.max_iou)

        if dist < args.min_center_dist:
            penalty += args.w_close * (args.min_center_dist - dist)

    # rect끼리 중복 방지
    for a, b in combinations(rect_reps, 2):
        ov = iou(a, b)

        if ov > args.rect_rect_max_iou:
            penalty += args.w_rect_rect_overlap * (ov - args.rect_rect_max_iou)

    # square와 rect 중복 방지
    for r in rect_reps:
        ov = iou(square, r)

        if ov > args.square_rect_max_iou:
            penalty += args.w_square_rect_overlap * (ov - args.square_rect_max_iou)

    # 시계방향 정렬
    center_x = float(np.mean([p["x"] for p in all_items]))
    center_y = float(np.mean([p["y"] for p in all_items]))

    square_angle = clockwise_angle(square, center_x, center_y)

    rect_pairs = []

    for r in rect_reps:
        angle = clockwise_angle(r, center_x, center_y)
        rel_angle = (angle - square_angle) % (2.0 * math.pi)
        rect_pairs.append((rel_angle, r))

    rect_pairs = sorted(rect_pairs, key=lambda x: x[0])

    rel_angles = [a for a, _ in rect_pairs]

    # rect 3개가 같은 방향에 몰리면 감점
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
        "square_conf": square["conf"],
        "square_area": square_area,
        "square_aspect": box_aspect(square),
        "square_rect_ratio": square_rect_ratio,
        "rect_conf_mean": rect_conf_mean,
        "rect_shape_mean": rect_shape_mean,
        "rect_median_area": rect_median_area,
        "group_size_mean": group_size_mean,
        "max_pair_iou": max_pair_iou,
        "min_pair_dist": min_pair_dist,
        "min_angle_gap_deg": math.degrees(min_angle_gap) if min_angle_gap < 900 else 999.0,
    }

    return final_score, rect_pairs, detail


def select_best(preds, args):
    """
    최적 square 1개 + rect group 3개 선택.
    """
    square_candidates = filter_square_candidates(preds, args)
    rect_candidates = filter_rect_candidates(preds, args)

    rect_groups = group_candidates(rect_candidates, args)
    rect_group_reps = [g["rep"] for g in rect_groups][:args.topk_rect_groups]

    if len(square_candidates) == 0:
        return None, "no_square_candidate", square_candidates, rect_candidates, rect_groups, None

    if len(rect_group_reps) < 3:
        return None, "less_than_3_rect_groups", square_candidates, rect_candidates, rect_groups, None

    best_selected = None
    best_score = -1e18
    best_detail = None

    for square in square_candidates:
        for rects in combinations(rect_group_reps, 3):
            # 서로 다른 group_id만 허용
            group_ids = [r["group_id"] for r in rects]
            if len(set(group_ids)) != 3:
                continue

            combo_score, rect_pairs, detail = score_combo(square, rects, args)

            if combo_score > best_score:
                best_score = combo_score
                best_detail = detail

                if rect_pairs is not None:
                    best_selected = {
                        0: square,
                        1: rect_pairs[0][1],
                        2: rect_pairs[1][1],
                        3: rect_pairs[2][1],
                    }

    if best_selected is None:
        return None, "no_valid_combo", square_candidates, rect_candidates, rect_groups, best_detail

    if best_score < args.min_final_score:
        return None, "low_final_score", square_candidates, rect_candidates, rect_groups, best_detail

    return best_selected, "ok", square_candidates, rect_candidates, rect_groups, best_detail


def write_selected_txt(path: Path, selected):
    """
    최종 선택 결과 txt 저장.
    class 0~3 각 1개.
    """
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

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def draw_grouped_candidates(image, preds, square_candidates, rect_groups, image_w: int, image_h: int):
    """
    그룹화된 후보 시각화.
    - raw bbox는 얇게 표시
    - rect group 대표 bbox는 굵게 표시
    """
    vis = image.copy()

    # raw prediction 표시
    for p in preds:
        x1, y1, x2, y2 = norm_to_xyxy(p, image_w, image_h)

        if p["raw_class"] == 0:
            color = (255, 0, 0)
        elif p["raw_class"] == 1:
            color = (255, 255, 0)
        elif p["raw_class"] == 2:
            color = (255, 255, 255)
        else:
            color = (0, 255, 255)

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1)

    # square 후보 표시
    for p in square_candidates:
        x1, y1, x2, y2 = norm_to_xyxy(p, image_w, image_h)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(
            vis,
            f"S cand {p['conf']:.2f}",
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    # rect group 대표 표시
    for g in rect_groups:
        rep = g["rep"]
        x1, y1, x2, y2 = norm_to_xyxy(rep, image_w, image_h)

        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 255), 3)
        cv2.putText(
            vis,
            f"G{rep['group_id']} n={rep['group_size']} raw={rep['group_raw_classes']} conf={rep['conf']:.2f}",
            (x1, min(image_h - 5, y2 + 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 200, 255),
            1,
            cv2.LINE_AA,
        )

    return vis


def draw_selected(image, selected, image_w: int, image_h: int):
    """
    최종 선택 결과 시각화.
    """
    vis = image.copy()
    centers = []

    for final_class in [0, 1, 2, 3]:
        p = selected[final_class]
        x1, y1, x2, y2 = norm_to_xyxy(p, image_w, image_h)
        color = FINAL_COLORS[final_class]

        cx = int(round(p["x"] * image_w))
        cy = int(round(p["y"] * image_h))

        centers.append((final_class, cx, cy))

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.circle(vis, (cx, cy), 4, color, -1)

        raw_text = f"raw{p.get('raw_class', -1)}"
        group_text = f"G{p.get('group_id', -1)}"

        cv2.putText(
            vis,
            f"class {final_class} {raw_text} {group_text} {p['conf']:.2f}",
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            color,
            2,
            cv2.LINE_AA,
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
            cv2.LINE_AA,
        )

    return vis


def process_one(image_path: Path, pred_label_dir: Path, out_dirs: dict, args):
    """
    이미지 1장 후처리.
    """
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image is None:
        raise RuntimeError(f"이미지 읽기 실패: {image_path}")

    image_h, image_w = image.shape[:2]
    stem = image_path.stem

    txt_path = pred_label_dir / f"{stem}.txt"
    preds = read_yolo_txt(txt_path)

    row = {
        "stem": stem,
        "raw_total": len(preds),
        "raw_class0": len([p for p in preds if p["raw_class"] == 0]),
        "raw_class1": len([p for p in preds if p["raw_class"] == 1]),
        "raw_class2": len([p for p in preds if p["raw_class"] == 2]),
        "raw_class3": len([p for p in preds if p["raw_class"] == 3]),
        "selected": False,
    }

    if len(preds) == 0:
        row["status"] = "no_prediction"
        return row

    selected, status, square_candidates, rect_candidates, rect_groups, detail = select_best(preds, args)

    row["status"] = status
    row["square_candidates"] = len(square_candidates)
    row["rect_raw_candidates"] = len(rect_candidates)
    row["rect_groups"] = len(rect_groups)

    if detail:
        row.update(detail)

    grouped_vis = draw_grouped_candidates(
        image=image,
        preds=preds,
        square_candidates=square_candidates,
        rect_groups=rect_groups,
        image_w=image_w,
        image_h=image_h,
    )

    cv2.imwrite(str(out_dirs["grouped_candidate_vis"] / f"{stem}_grouped_candidates.png"), grouped_vis)

    if selected is None:
        return row

    row["selected"] = True

    write_selected_txt(out_dirs["selected_labels"] / f"{stem}.txt", selected)

    selected_vis = draw_selected(
        image=image,
        selected=selected,
        image_w=image_w,
        image_h=image_h,
    )

    cv2.imwrite(str(out_dirs["selected_vis"] / f"{stem}_selected.png"), selected_vis)

    for final_class in [0, 1, 2, 3]:
        p = selected[final_class]

        row[f"class{final_class}_raw_class"] = p.get("raw_class", -1)
        row[f"class{final_class}_group_id"] = p.get("group_id", -1)
        row[f"class{final_class}_group_size"] = p.get("group_size", 1)
        row[f"class{final_class}_group_raw_classes"] = p.get("group_raw_classes", str(p.get("raw_class", -1)))
        row[f"class{final_class}_conf"] = p["conf"]
        row[f"class{final_class}_x"] = p["x"]
        row[f"class{final_class}_y"] = p["y"]
        row[f"class{final_class}_w"] = p["w"]
        row[f"class{final_class}_h"] = p["h"]
        row[f"class{final_class}_area"] = box_area(p)
        row[f"class{final_class}_aspect"] = box_aspect(p)

    return row


def save_csv(path: Path, rows):
    """CSV 저장"""
    if not rows:
        return

    fields = sorted(set().union(*[r.keys() for r in rows]))

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--pred_label_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    # 이미 conf=0.7/iou=0.3으로 생성된 YOLO txt를 입력으로 쓰므로 여기서는 추가 conf 필터를 기본 0으로 둔다.
    parser.add_argument("--conf_square", type=float, default=0.0)
    parser.add_argument("--conf_rect", type=float, default=0.0)

    # square 후보 기준
    parser.add_argument("--square_min_area", type=float, default=0.0008)
    parser.add_argument("--square_target_area", type=float, default=0.0045)
    parser.add_argument("--square_max_area", type=float, default=0.020)
    parser.add_argument("--square_max_aspect", type=float, default=2.20)

    # rect 후보 기준
    parser.add_argument("--rect_min_area", type=float, default=0.0010)
    parser.add_argument("--rect_max_area", type=float, default=0.120)

    # rect 후보 그룹화 기준
    parser.add_argument("--group_iou", type=float, default=0.18)
    parser.add_argument("--group_center_dist", type=float, default=0.060)
    parser.add_argument("--group_center_inside_dist", type=float, default=0.100)

    # 후보 개수
    parser.add_argument("--topk_square", type=int, default=8)
    parser.add_argument("--topk_rect_groups", type=int, default=15)

    # 선택된 최종 4개 간의 구조 조건
    parser.add_argument("--min_center_dist", type=float, default=0.025)
    parser.add_argument("--max_iou", type=float, default=0.60)
    parser.add_argument("--rect_rect_max_iou", type=float, default=0.50)
    parser.add_argument("--square_rect_max_iou", type=float, default=0.45)
    parser.add_argument("--min_angle_gap_deg", type=float, default=8.0)

    # square/rect 면적비
    parser.add_argument("--soft_max_square_rect_area_ratio", type=float, default=0.90)
    parser.add_argument("--hard_max_square_rect_area_ratio", type=float, default=1.50)

    # 최종 점수 기준
    parser.add_argument("--min_final_score", type=float, default=-1.0)

    # square 점수 가중치
    parser.add_argument("--w_square_conf", type=float, default=1.0)
    parser.add_argument("--w_square_shape", type=float, default=1.5)
    parser.add_argument("--w_square_area_prior", type=float, default=2.0)
    parser.add_argument("--w_square_area_penalty", type=float, default=2.0)

    # rect 점수 가중치
    parser.add_argument("--w_rect_conf", type=float, default=1.5)
    parser.add_argument("--w_rect_shape", type=float, default=0.8)
    parser.add_argument("--w_rect_area_penalty", type=float, default=0.3)

    # 그룹 점수
    parser.add_argument("--w_group_size_bonus", type=float, default=0.15)
    parser.add_argument("--group_size_bonus_cap", type=float, default=4.0)

    # 구조 penalty
    parser.add_argument("--w_area_ratio", type=float, default=1.0)
    parser.add_argument("--w_overlap", type=float, default=3.0)
    parser.add_argument("--w_close", type=float, default=3.0)
    parser.add_argument("--w_rect_rect_overlap", type=float, default=5.0)
    parser.add_argument("--w_square_rect_overlap", type=float, default=5.0)
    parser.add_argument("--w_angle_gap", type=float, default=1.0)

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
        "grouped_candidate_vis": out_dir / "grouped_candidate_vis",
        "selected_vis": out_dir / "selected_vis",
        "selected_labels": out_dir / "selected_labels",
    }

    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    images = collect_images(image_dir)

    print("========== CONFIG ==========")
    print(f"image_dir:      {image_dir}")
    print(f"pred_label_dir: {pred_label_dir}")
    print(f"out_dir:        {out_dir}")
    print(f"image_count:    {len(images)}")
    print("rule: class0 square fixed, class1~3 grouped as physical rect IDs, clockwise reassignment")
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
                print(
                    f"[OK] {idx + 1}/{len(images)} {image_path.stem} | "
                    f"square_candidates={row.get('square_candidates', '')} "
                    f"rect_raw={row.get('rect_raw_candidates', '')} "
                    f"rect_groups={row.get('rect_groups', '')}"
                )
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

    summary_path = out_dir / "postprocess_4class_clockwise_grouped_summary.csv"
    save_csv(summary_path, rows)

    print("\n========== RESULT ==========")
    print(f"selected ok: {ok}")
    print(f"failed:      {fail}")
    print(f"summary:     {summary_path}")
    print("============================")


if __name__ == "__main__":
    main()