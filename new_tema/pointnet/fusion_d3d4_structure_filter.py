#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fusion_d3d4_structure_filter.py

목적:
- D4 모델 결과에서 square(class0)만 사용
- D3 모델 결과에서 rect(class1, class2, class3)만 사용
- 각 이미지별로 class0~3 후보를 1개씩 선택
- 단순 confidence가 아니라 위치/크기/비율/구조 조건으로 필터링
- overlay 이미지와 summary CSV 저장

전제:
- YOLO segment predict 실행 시 save_txt=True save_conf=True로 labels/*.txt가 존재해야 함.
- YOLO segmentation txt 형식:
  class x1 y1 x2 y2 ... conf
"""

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np


# 시각화 색상: OpenCV BGR
COLORS = {
    0: (0, 255, 255),   # square: yellow
    1: (0, 0, 255),     # rect1: red
    2: (0, 255, 0),     # rect2: green
    3: (255, 0, 0),     # rect3: blue
}


CLASS_NAMES = {
    0: "square_D4",
    1: "rect1_D3",
    2: "rect2_D3",
    3: "rect3_D3",
}


def read_yolo_seg_label(label_path: Path, image_w: int, image_h: int):
    """
    YOLO segmentation label txt를 읽어서 polygon 후보 목록으로 변환.
    save_conf=True 기준 마지막 값은 confidence로 처리.
    """
    candidates = []

    if not label_path.exists():
        return candidates

    lines = label_path.read_text(encoding="utf-8").strip().splitlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 8:
            continue

        cls = int(float(parts[0]))
        nums = [float(x) for x in parts[1:]]

        # save_conf=True면 마지막 값이 confidence
        if len(nums) % 2 == 1:
            conf = float(nums[-1])
            coords = nums[:-1]
        else:
            conf = 1.0
            coords = nums

        if len(coords) < 6:
            continue

        pts = np.array(coords, dtype=np.float32).reshape(-1, 2)
        pts[:, 0] *= float(image_w)
        pts[:, 1] *= float(image_h)

        # 이미지 범위 보정
        pts[:, 0] = np.clip(pts[:, 0], 0, image_w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, image_h - 1)

        area = abs(float(cv2.contourArea(pts)))
        if area <= 1.0:
            continue

        x, y, w, h = cv2.boundingRect(pts.astype(np.int32))
        cx = float(x + w / 2.0)
        cy = float(y + h / 2.0)

        aspect = float(max(w, h) / max(min(w, h), 1))
        area_norm = float(area / max(image_w * image_h, 1))

        candidates.append({
            "cls": cls,
            "conf": conf,
            "pts": pts,
            "area": area,
            "area_norm": area_norm,
            "bbox": (int(x), int(y), int(w), int(h)),
            "center": (cx, cy),
            "aspect": aspect,
        })

    return candidates


def candidate_score(c):
    """
    후보 점수 계산.
    class0은 square 형태 선호.
    class1~3은 rect 형태 선호.
    """
    cls = c["cls"]
    conf = c["conf"]
    aspect = max(float(c["aspect"]), 1.0)
    area_norm = float(c["area_norm"])

    if cls == 0:
        # square는 aspect가 1에 가까울수록 좋고, 너무 큰 영역이면 감점
        aspect_penalty = abs(math.log(aspect))
        area_penalty = max(area_norm - 0.015, 0.0) * 20.0
        score = conf - 0.45 * aspect_penalty - area_penalty
    else:
        # rect는 너무 정사각형이면 감점, 너무 큰 body 영역이면 감점
        rect_bonus = min(aspect, 5.0) / 5.0
        square_like_penalty = max(1.25 - aspect, 0.0)
        area_penalty = max(area_norm - 0.035, 0.0) * 12.0
        score = conf + 0.15 * rect_bonus - 0.35 * square_like_penalty - area_penalty

    return float(score)


def polygon_iou_bbox(a, b):
    """
    빠른 중복 제거용 bbox IoU.
    mask IoU까지는 아니고, 같은 위치 중복 후보 제거용.
    """
    ax, ay, aw, ah = a["bbox"]
    bx, by, bw, bh = b["bbox"]

    ax2 = ax + aw
    ay2 = ay + ah
    bx2 = bx + bw
    by2 = by + bh

    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(ix2 - ix1, 0)
    ih = max(iy2 - iy1, 0)

    inter = iw * ih
    union = aw * ah + bw * bh - inter

    if union <= 0:
        return 0.0

    return float(inter / union)


def select_one_per_class(candidates_by_class):
    """
    class별 후보 중 1개 선택.
    우선 candidate_score 기준으로 선택.
    """
    selected = {}

    for cls, candidates in candidates_by_class.items():
        if not candidates:
            continue

        ranked = sorted(
            candidates,
            key=lambda c: candidate_score(c),
            reverse=True,
        )

        selected[cls] = ranked[0]

    return selected


def structure_filter(selected, args):
    """
    선택된 class0~3 후보가 같은 마커 구조 안에 있는지 검사.

    여기서는 GT 위치를 모르는 상태이므로 다음 조건만 사용:
    - class0 square가 rect들보다 과도하게 크면 reject
    - class0과 rect가 bbox 기준 과도하게 겹치면 reject
    - 중심점들이 한 덩어리 구조 안에 있어야 함
    - 너무 멀리 떨어진 후보가 있으면 reject
    """
    status = "PASS"
    reasons = []

    present_classes = sorted(selected.keys())

    if len(present_classes) < args.min_classes:
        status = "FAIL"
        reasons.append(f"classes<{args.min_classes}")

    if 0 not in selected:
        status = "FAIL"
        reasons.append("no_square")

    rects = [selected[c] for c in [1, 2, 3] if c in selected]

    # square 크기 검사
    if 0 in selected and rects:
        sq = selected[0]
        rect_area_med = float(np.median([r["area"] for r in rects]))
        if rect_area_med > 0:
            ratio = sq["area"] / rect_area_med
            if ratio > args.max_square_to_rect_area_ratio:
                status = "FAIL"
                reasons.append(f"square_too_large_ratio={ratio:.3f}")

    # square와 rect가 지나치게 겹치면 square가 rect 내부 조각일 가능성 있음
    if 0 in selected:
        sq = selected[0]
        for r in rects:
            iou = polygon_iou_bbox(sq, r)
            if iou > args.max_square_rect_bbox_iou:
                status = "FAIL"
                reasons.append(f"square_rect_overlap_cls{r['cls']}_iou={iou:.3f}")

    # 중심점 구조 검사
    if len(selected) >= 3:
        centers = np.array([selected[c]["center"] for c in selected.keys()], dtype=np.float64)

        cx = float(np.mean(centers[:, 0]))
        cy = float(np.mean(centers[:, 1]))

        dists = np.sqrt((centers[:, 0] - cx) ** 2 + (centers[:, 1] - cy) ** 2)
        med_dist = float(np.median(dists) + 1e-6)
        max_dist = float(np.max(dists))

        if max_dist > args.max_center_spread_ratio * med_dist and max_dist > args.min_abs_center_spread_px:
            status = "FAIL"
            reasons.append(f"center_spread=max{max_dist:.1f}_med{med_dist:.1f}")

    return status, ";".join(reasons) if reasons else "ok"


def draw_overlay(image, selected, status, reason):
    """
    선택 후보를 이미지 위에 표시.
    """
    out = image.copy()

    for cls, c in selected.items():
        pts = c["pts"].astype(np.int32)
        color = COLORS.get(cls, (255, 255, 255))

        cv2.polylines(out, [pts], isClosed=True, color=color, thickness=2)

        x, y, w, h = c["bbox"]
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 1)

        cx, cy = c["center"]
        cv2.circle(out, (int(cx), int(cy)), 4, color, -1)

        text = f"{cls}:{c['conf']:.2f}"
        cv2.putText(
            out,
            text,
            (x, max(y - 5, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    status_color = (0, 255, 0) if status == "PASS" else (0, 0, 255)
    cv2.putText(
        out,
        f"{status} | {reason}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        status_color,
        2,
        cv2.LINE_AA,
    )

    return out


def find_source_images(source_dir: Path):
    """
    원본 images_color 안의 이미지 목록 수집.
    """
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    return sorted([p for p in source_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])


def label_path_for(pred_set_dir: Path, image_path: Path):
    """
    YOLO predict 결과 labels 경로 추정.
    이미지 stem 기준 labels/stem.txt.
    """
    return pred_set_dir / "labels" / f"{image_path.stem}.txt"


def process_set(set_name, src_img_dir, d3_set_dir, d4_set_dir, out_set_dir, writer, args):
    """
    하나의 down set 처리.
    """
    out_pass = out_set_dir / "pass"
    out_fail = out_set_dir / "fail"
    out_all = out_set_dir / "all"

    out_pass.mkdir(parents=True, exist_ok=True)
    out_fail.mkdir(parents=True, exist_ok=True)
    out_all.mkdir(parents=True, exist_ok=True)

    images = find_source_images(src_img_dir)

    pass_count = 0
    fail_count = 0

    for img_path in images:
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            continue

        h, w = img.shape[:2]

        d3_label = label_path_for(d3_set_dir, img_path)
        d4_label = label_path_for(d4_set_dir, img_path)

        d3_candidates = read_yolo_seg_label(d3_label, w, h)
        d4_candidates = read_yolo_seg_label(d4_label, w, h)

        # Fusion 규칙:
        # D4에서는 class0만 사용
        # D3에서는 class1~3만 사용
        candidates_by_class = {
            0: [c for c in d4_candidates if c["cls"] == 0],
            1: [c for c in d3_candidates if c["cls"] == 1],
            2: [c for c in d3_candidates if c["cls"] == 2],
            3: [c for c in d3_candidates if c["cls"] == 3],
        }

        selected = select_one_per_class(candidates_by_class)
        status, reason = structure_filter(selected, args)

        overlay = draw_overlay(img, selected, status, reason)

        out_name = img_path.name
        cv2.imwrite(str(out_all / out_name), overlay)

        if status == "PASS":
            cv2.imwrite(str(out_pass / out_name), overlay)
            pass_count += 1
        else:
            cv2.imwrite(str(out_fail / out_name), overlay)
            fail_count += 1

        row = {
            "set": set_name,
            "image": img_path.name,
            "status": status,
            "reason": reason,
            "num_selected": len(selected),
            "has_c0_square_D4": int(0 in selected),
            "has_c1_rect1_D3": int(1 in selected),
            "has_c2_rect2_D3": int(2 in selected),
            "has_c3_rect3_D3": int(3 in selected),
        }

        for cls in [0, 1, 2, 3]:
            if cls in selected:
                c = selected[cls]
                row[f"c{cls}_conf"] = f"{c['conf']:.6f}"
                row[f"c{cls}_area"] = f"{c['area']:.3f}"
                row[f"c{cls}_aspect"] = f"{c['aspect']:.3f}"
                row[f"c{cls}_cx"] = f"{c['center'][0]:.3f}"
                row[f"c{cls}_cy"] = f"{c['center'][1]:.3f}"
            else:
                row[f"c{cls}_conf"] = ""
                row[f"c{cls}_area"] = ""
                row[f"c{cls}_aspect"] = ""
                row[f"c{cls}_cx"] = ""
                row[f"c{cls}_cy"] = ""

        writer.writerow(row)

    print(f"[SET] {set_name} | pass={pass_count} | fail={fail_count} | total={pass_count + fail_count}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--src_root", required=True, help="range_sweep_down_10sets")
    parser.add_argument("--d3_root", required=True, help="D3_yolo11n_damage_v1_epoch30 predict root")
    parser.add_argument("--d4_root", required=True, help="yolo11n_damage_D4_epoch30 predict root")
    parser.add_argument("--out_root", required=True)

    parser.add_argument("--sets", nargs="+", default=["0_down", "01_down", "02_down"])

    # 구조 필터 옵션
    parser.add_argument("--min_classes", type=int, default=4)
    parser.add_argument("--max_square_to_rect_area_ratio", type=float, default=0.85)
    parser.add_argument("--max_square_rect_bbox_iou", type=float, default=0.30)
    parser.add_argument("--max_center_spread_ratio", type=float, default=3.0)
    parser.add_argument("--min_abs_center_spread_px", type=float, default=120.0)

    args = parser.parse_args()

    src_root = Path(args.src_root)
    d3_root = Path(args.d3_root)
    d4_root = Path(args.d4_root)
    out_root = Path(args.out_root)

    out_root.mkdir(parents=True, exist_ok=True)

    csv_path = out_root / "fusion_structure_summary.csv"

    fieldnames = [
        "set",
        "image",
        "status",
        "reason",
        "num_selected",
        "has_c0_square_D4",
        "has_c1_rect1_D3",
        "has_c2_rect2_D3",
        "has_c3_rect3_D3",
    ]

    for cls in [0, 1, 2, 3]:
        fieldnames += [
            f"c{cls}_conf",
            f"c{cls}_area",
            f"c{cls}_aspect",
            f"c{cls}_cx",
            f"c{cls}_cy",
        ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for set_name in args.sets:
            src_img_dir = src_root / set_name / "images_color"
            d3_set_dir = d3_root / set_name
            d4_set_dir = d4_root / set_name

            if not src_img_dir.exists():
                print(f"[SKIP] source images 없음: {src_img_dir}")
                continue

            if not d3_set_dir.exists():
                print(f"[SKIP] D3 set 없음: {d3_set_dir}")
                continue

            if not d4_set_dir.exists():
                print(f"[SKIP] D4 set 없음: {d4_set_dir}")
                continue

            out_set_dir = out_root / set_name

            process_set(
                set_name=set_name,
                src_img_dir=src_img_dir,
                d3_set_dir=d3_set_dir,
                d4_set_dir=d4_set_dir,
                out_set_dir=out_set_dir,
                writer=writer,
                args=args,
            )

    print("")
    print("[DONE]")
    print("out_root:", out_root)
    print("summary:", csv_path)


if __name__ == "__main__":
    main()
