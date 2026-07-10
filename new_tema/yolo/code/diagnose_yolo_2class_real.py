from pathlib import Path
import argparse
import csv
import math

import cv2


# ============================================================
# YOLO 2-class 실해역 추론 결과 진단 코드
#
# 목적:
#   학습/데이터 생성/라벨 수정 없이,
#   현재 YOLO predict 결과 txt만 분석한다.
#
# YOLO class:
#   0 = square_id
#   1 = rect_id
#
# 분석 항목:
#   - 이미지별 square 후보 수
#   - 이미지별 rect 후보 수
#   - square 최대 confidence
#   - rect 상위 confidence
#   - square bbox 면적/비율
#   - 후처리 실패 원인 추정
#
# 출력:
#   out_dir/diagnosis_summary.csv
#   out_dir/diagnosis_vis/*.png
# ============================================================

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def collect_images(image_dir: Path):
    """이미지 목록 수집"""
    if not image_dir.exists():
        raise FileNotFoundError(f"image_dir 없음: {image_dir}")

    images = []

    for path in image_dir.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            images.append(path)

    return sorted(images)


def read_yolo_txt(txt_path: Path):
    """
    YOLO predict txt 읽기

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

            cls = int(float(parts[0]))

            if cls not in [0, 1]:
                continue

            pred = {
                "id": f"{txt_path.stem}_{line_idx}",
                "class_id": cls,
                "x": float(parts[1]),
                "y": float(parts[2]),
                "w": float(parts[3]),
                "h": float(parts[4]),
                "conf": float(parts[5]) if len(parts) == 6 else 1.0,
            }

            preds.append(pred)

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


def filter_square_candidates(preds, args):
    """후처리 기준으로 square 후보 필터링"""
    out = []

    for p in preds:
        if p["class_id"] != 0:
            continue

        area = box_area(p)
        aspect = box_aspect(p)

        if p["conf"] < args.conf_square:
            continue

        if area < args.min_area:
            continue

        if area > args.square_max_area:
            continue

        if aspect > args.square_max_aspect:
            continue

        out.append(p)

    return sorted(out, key=lambda q: q["conf"], reverse=True)


def filter_rect_candidates(preds, args):
    """후처리 기준으로 rect 후보 필터링"""
    out = []

    for p in preds:
        if p["class_id"] != 1:
            continue

        area = box_area(p)

        if p["conf"] < args.conf_rect:
            continue

        if area < args.min_area:
            continue

        if area > args.max_area:
            continue

        out.append(p)

    return sorted(out, key=lambda q: q["conf"], reverse=True)


def get_top(preds):
    """confidence 기준 최상위 후보 반환"""
    if not preds:
        return None

    return sorted(preds, key=lambda q: q["conf"], reverse=True)[0]


def diagnose_one(stem: str, preds, args):
    """이미지 1장 진단"""
    raw_square = [p for p in preds if p["class_id"] == 0]
    raw_rect = [p for p in preds if p["class_id"] == 1]

    filtered_square = filter_square_candidates(preds, args)
    filtered_rect = filter_rect_candidates(preds, args)

    top_square_raw = get_top(raw_square)
    top_square_filtered = get_top(filtered_square)

    top_rects = sorted(raw_rect, key=lambda q: q["conf"], reverse=True)[:5]
    top_rects_filtered = sorted(filtered_rect, key=lambda q: q["conf"], reverse=True)[:5]

    row = {
        "stem": stem,
        "raw_total": len(preds),
        "raw_square_count": len(raw_square),
        "raw_rect_count": len(raw_rect),
        "filtered_square_count": len(filtered_square),
        "filtered_rect_count": len(filtered_rect),
        "diagnosis": "",
    }

    # raw square 정보
    if top_square_raw is not None:
        row["top_square_raw_conf"] = top_square_raw["conf"]
        row["top_square_raw_area"] = box_area(top_square_raw)
        row["top_square_raw_aspect"] = box_aspect(top_square_raw)
        row["top_square_raw_x"] = top_square_raw["x"]
        row["top_square_raw_y"] = top_square_raw["y"]
        row["top_square_raw_w"] = top_square_raw["w"]
        row["top_square_raw_h"] = top_square_raw["h"]
    else:
        row["top_square_raw_conf"] = ""
        row["top_square_raw_area"] = ""
        row["top_square_raw_aspect"] = ""

    # filtered square 정보
    if top_square_filtered is not None:
        row["top_square_filtered_conf"] = top_square_filtered["conf"]
        row["top_square_filtered_area"] = box_area(top_square_filtered)
        row["top_square_filtered_aspect"] = box_aspect(top_square_filtered)
        row["top_square_filtered_x"] = top_square_filtered["x"]
        row["top_square_filtered_y"] = top_square_filtered["y"]
        row["top_square_filtered_w"] = top_square_filtered["w"]
        row["top_square_filtered_h"] = top_square_filtered["h"]
    else:
        row["top_square_filtered_conf"] = ""
        row["top_square_filtered_area"] = ""
        row["top_square_filtered_aspect"] = ""

    # rect confidence 상위 기록
    for i in range(5):
        key = f"top_rect_raw_{i + 1}_conf"
        row[key] = top_rects[i]["conf"] if i < len(top_rects) else ""

    for i in range(5):
        key = f"top_rect_filtered_{i + 1}_conf"
        row[key] = top_rects_filtered[i]["conf"] if i < len(top_rects_filtered) else ""

    # 원인 분류
    if len(preds) == 0:
        row["diagnosis"] = "no_yolo_prediction"
    elif len(raw_square) == 0:
        row["diagnosis"] = "no_raw_square_candidate"
    elif len(filtered_square) == 0:
        # raw square는 있지만 필터에서 탈락
        reasons = []

        for p in raw_square:
            if p["conf"] < args.conf_square:
                reasons.append("square_conf_low")
            if box_area(p) < args.min_area:
                reasons.append("square_area_too_small")
            if box_area(p) > args.square_max_area:
                reasons.append("square_area_too_large")
            if box_aspect(p) > args.square_max_aspect:
                reasons.append("square_aspect_bad")

        row["diagnosis"] = "+".join(sorted(set(reasons))) if reasons else "square_filtered_unknown"
    elif len(raw_rect) < 3:
        row["diagnosis"] = "less_than_3_raw_rect"
    elif len(filtered_rect) < 3:
        row["diagnosis"] = "less_than_3_filtered_rect"
    else:
        # 후보는 충분함
        # 여기서도 square가 큰지 확인
        sq = top_square_filtered
        sq_area = box_area(sq)
        sq_aspect = box_aspect(sq)

        if sq_area > args.warn_square_area:
            row["diagnosis"] = "candidate_ok_but_square_large_warning"
        elif sq_aspect > args.warn_square_aspect:
            row["diagnosis"] = "candidate_ok_but_square_aspect_warning"
        else:
            row["diagnosis"] = "candidate_ok"

    return row


def draw_diagnosis(image, preds, args):
    """
    진단용 시각화 이미지 생성
    파란색: raw square 후보
    노란색: raw rect 후보
    두꺼운 녹색: 필터 통과 square 후보
    두꺼운 흰색: 필터 통과 rect 후보
    """
    image_h, image_w = image.shape[:2]
    vis = image.copy()

    filtered_square = filter_square_candidates(preds, args)
    filtered_rect = filter_rect_candidates(preds, args)

    filtered_ids = set([p["id"] for p in filtered_square + filtered_rect])

    # raw 후보 먼저 그림
    for p in preds:
        x1, y1, x2, y2 = norm_to_xyxy(p, image_w, image_h)

        if p["class_id"] == 0:
            color = (255, 0, 0)      # blue
        else:
            color = (0, 255, 255)    # yellow

        thickness = 1

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(
            vis,
            f"{p['class_id']}:{p['conf']:.2f}",
            (x1, max(15, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA
        )

    # 필터 통과 후보 강조
    for p in filtered_square:
        x1, y1, x2, y2 = norm_to_xyxy(p, image_w, image_h)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(
            vis,
            f"S_OK {p['conf']:.2f}",
            (x1, min(image_h - 5, y2 + 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    for p in filtered_rect[:5]:
        x1, y1, x2, y2 = norm_to_xyxy(p, image_w, image_h)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 255), 2)

    return vis


def save_csv(path: Path, rows):
    """CSV 저장"""
    if not rows:
        return

    fieldnames = sorted(set().union(*[r.keys() for r in rows]))

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

    # 후처리 기준과 동일하거나 비슷하게 설정
    parser.add_argument("--conf_square", type=float, default=0.08)
    parser.add_argument("--conf_rect", type=float, default=0.12)
    parser.add_argument("--min_area", type=float, default=0.0002)
    parser.add_argument("--max_area", type=float, default=0.50)
    parser.add_argument("--square_max_area", type=float, default=0.10)
    parser.add_argument("--square_max_aspect", type=float, default=1.90)

    # 경고 기준
    parser.add_argument("--warn_square_area", type=float, default=0.060)
    parser.add_argument("--warn_square_aspect", type=float, default=1.60)

    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    pred_label_dir = Path(args.pred_label_dir)
    out_dir = Path(args.out_dir)
    vis_dir = out_dir / "diagnosis_vis"

    if not pred_label_dir.exists():
        raise FileNotFoundError(f"pred_label_dir 없음: {pred_label_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    images = collect_images(image_dir)

    print("========== CONFIG ==========")
    print(f"image_dir:      {image_dir.resolve()}")
    print(f"pred_label_dir: {pred_label_dir.resolve()}")
    print(f"out_dir:        {out_dir.resolve()}")
    print(f"image_count:    {len(images)}")
    print("============================")

    rows = []

    for idx, image_path in enumerate(images):
        stem = image_path.stem
        pred_txt = pred_label_dir / f"{stem}.txt"

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

        if image is None:
            row = {
                "stem": stem,
                "diagnosis": "image_read_failed",
            }
            rows.append(row)
            print(f"[FAIL] {idx + 1}/{len(images)} {stem}: image_read_failed")
            continue

        preds = read_yolo_txt(pred_txt)
        row = diagnose_one(stem, preds, args)
        rows.append(row)

        vis = draw_diagnosis(image, preds, args)
        cv2.imwrite(str(vis_dir / f"{stem}_diagnosis.png"), vis)

        print(
            f"[{idx + 1}/{len(images)}] {stem} | "
            f"raw_square={row['raw_square_count']} "
            f"raw_rect={row['raw_rect_count']} "
            f"filtered_square={row['filtered_square_count']} "
            f"filtered_rect={row['filtered_rect_count']} "
            f"diagnosis={row['diagnosis']}"
        )

    csv_path = out_dir / "diagnosis_summary.csv"
    save_csv(csv_path, rows)

    print("\n========== DONE ==========")
    print(f"summary: {csv_path}")
    print(f"vis_dir: {vis_dir}")
    print("==========================")

    # 전체 요약
    counts = {}

    for row in rows:
        key = row.get("diagnosis", "unknown")
        counts[key] = counts.get(key, 0) + 1

    print("\n========== DIAGNOSIS COUNT ==========")
    for k, v in sorted(counts.items(), key=lambda x: x[0]):
        print(f"{k}: {v}")
    print("=====================================")


if __name__ == "__main__":
    main()