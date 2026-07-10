import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


# ============================================================
# YOLO segmentation txt parsing
# ============================================================

def parse_yolo_seg_txt(txt_path: Path):
    """
    YOLO segmentation label txt를 파싱한다.

    지원 형식:
    1) class x1 y1 x2 y2 ... xn yn
    2) class x1 y1 x2 y2 ... xn yn conf

    반환:
        [
            {
                "cls": int,
                "pts": np.ndarray(N, 2),  # normalized xy
                "conf": float or None
            },
            ...
        ]
    """
    detections = []

    if not txt_path.exists():
        return detections

    lines = txt_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts = line.split()

        try:
            vals = [float(x) for x in parts]
        except ValueError:
            continue

        if len(vals) < 7:
            continue

        cls_id = int(vals[0])
        rest = vals[1:]

        conf = None

        # save_conf=True인 경우 마지막 값이 confidence이고,
        # 나머지 좌표 개수는 짝수여야 한다.
        if len(rest) >= 7 and len(rest) % 2 == 1 and 0.0 <= rest[-1] <= 1.0:
            conf = float(rest[-1])
            coords = rest[:-1]
        else:
            coords = rest

        if len(coords) < 6 or len(coords) % 2 != 0:
            continue

        pts = np.array(coords, dtype=np.float32).reshape(-1, 2)
        pts[:, 0] = np.clip(pts[:, 0], 0.0, 1.0)
        pts[:, 1] = np.clip(pts[:, 1], 0.0, 1.0)

        detections.append(
            {
                "cls": cls_id,
                "pts": pts,
                "conf": conf,
            }
        )

    return detections


# ============================================================
# Geometry metrics
# ============================================================

def polygon_area_norm(pts: np.ndarray):
    """normalized polygon 면적을 계산한다."""
    if pts is None or len(pts) < 3:
        return 0.0

    x = pts[:, 0]
    y = pts[:, 1]

    area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    return area


def polygon_centroid(pts: np.ndarray):
    """polygon 중심 좌표를 normalized xy로 계산한다."""
    if pts is None or len(pts) == 0:
        return 0.0, 0.0

    return float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))


def bbox_metrics(pts: np.ndarray):
    """axis-aligned bbox 기반 기본 metric 계산."""
    if pts is None or len(pts) == 0:
        return 0.0, 0.0, 0.0

    x1 = float(np.min(pts[:, 0]))
    x2 = float(np.max(pts[:, 0]))
    y1 = float(np.min(pts[:, 1]))
    y2 = float(np.max(pts[:, 1]))

    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)

    short = max(min(bw, bh), 1e-8)
    long = max(bw, bh)

    aspect = long / short

    return bw, bh, aspect


def min_area_rect_aspect(pts: np.ndarray):
    """
    회전된 bounding rectangle 기준 aspect ratio 계산.
    rect class가 길쭉한지, square class가 compact한지 보는 보조 지표.
    """
    if pts is None or len(pts) < 3:
        return 0.0

    p = pts.astype(np.float32)
    rect = cv2.minAreaRect(p)
    w, h = rect[1]

    if w <= 1e-8 or h <= 1e-8:
        return 0.0

    return float(max(w, h) / max(min(w, h), 1e-8))


def detection_metrics(det):
    """단일 detection의 metric 계산."""
    pts = det["pts"]

    area = polygon_area_norm(pts)
    cx, cy = polygon_centroid(pts)
    bw, bh, bbox_aspect = bbox_metrics(pts)
    rotated_aspect = min_area_rect_aspect(pts)

    return {
        "area": area,
        "cx": cx,
        "cy": cy,
        "bbox_w": bw,
        "bbox_h": bh,
        "bbox_aspect": bbox_aspect,
        "rotated_aspect": rotated_aspect,
    }


# ============================================================
# File indexing
# ============================================================

def infer_set_name_from_path(path: Path):
    """
    YOLO predict 결과 구조에서 set name 추정.

    일반 구조:
        predict_root/01_fixed/labels/xxx.txt
        predict_root/01_fixed/xxx.png
    """
    parts = list(path.parts)

    if "labels" in parts:
        idx = parts.index("labels")
        if idx > 0:
            return parts[idx - 1]

    return path.parent.name


def collect_images(pred_root: Path):
    """
    prediction root 안의 이미지 파일을 수집한다.
    labels 폴더 안은 제외한다.
    """
    images = []

    for p in pred_root.rglob("*"):
        if not p.is_file():
            continue

        if p.suffix.lower() not in IMG_EXTS:
            continue

        if "labels" in p.parts:
            continue

        images.append(p)

    return sorted(images)


def build_image_map(pred_root: Path):
    """
    이미지 검색용 map 생성.
    key:
        (set_name, stem)
        stem
    """
    img_map_set = {}
    img_map_stem = {}

    for img_path in collect_images(pred_root):
        set_name = img_path.parent.name
        stem = img_path.stem

        img_map_set[(set_name, stem)] = img_path

        if stem not in img_map_stem:
            img_map_stem[stem] = img_path

    return img_map_set, img_map_stem


def find_image_for_label(label_path: Path, img_map_set, img_map_stem):
    """label txt에 대응하는 이미지 경로를 찾는다."""
    set_name = infer_set_name_from_path(label_path)
    stem = label_path.stem

    if (set_name, stem) in img_map_set:
        return img_map_set[(set_name, stem)]

    return img_map_stem.get(stem, None)


# ============================================================
# Analysis
# ============================================================

def analyze_one_model(model_name: str, pred_root: Path):
    """
    하나의 YOLO predict 결과 폴더를 분석한다.

    반환:
        per_image_rows
        per_det_rows
    """
    img_map_set, img_map_stem = build_image_map(pred_root)

    # 이미지 전체 목록 기준으로 0 detection 이미지까지 포함
    image_items = []
    for (set_name, stem), img_path in img_map_set.items():
        image_items.append((set_name, stem, img_path))

    image_items = sorted(image_items)

    # label txt map
    label_map = {}

    for txt_path in pred_root.rglob("labels/*.txt"):
        if not txt_path.is_file():
            continue

        set_name = infer_set_name_from_path(txt_path)
        label_map[(set_name, txt_path.stem)] = txt_path

    per_image_rows = []
    per_det_rows = []

    for set_name, stem, img_path in image_items:
        txt_path = label_map.get((set_name, stem), None)

        detections = parse_yolo_seg_txt(txt_path) if txt_path is not None else []

        class_counts = {0: 0, 1: 0, 2: 0, 3: 0}

        for det_idx, det in enumerate(detections):
            cls_id = int(det["cls"])
            class_counts[cls_id] = class_counts.get(cls_id, 0) + 1

            m = detection_metrics(det)

            per_det_rows.append(
                {
                    "model": model_name,
                    "set": set_name,
                    "image": img_path.name,
                    "stem": stem,
                    "det_idx": det_idx,
                    "cls": cls_id,
                    "conf": "" if det["conf"] is None else f"{det['conf']:.6f}",
                    "area": f"{m['area']:.8f}",
                    "cx": f"{m['cx']:.6f}",
                    "cy": f"{m['cy']:.6f}",
                    "bbox_w": f"{m['bbox_w']:.6f}",
                    "bbox_h": f"{m['bbox_h']:.6f}",
                    "bbox_aspect": f"{m['bbox_aspect']:.6f}",
                    "rotated_aspect": f"{m['rotated_aspect']:.6f}",
                    "image_path": str(img_path),
                    "label_path": "" if txt_path is None else str(txt_path),
                }
            )

        present_classes = [c for c in [0, 1, 2, 3] if class_counts.get(c, 0) > 0]
        dup_classes = [c for c in [0, 1, 2, 3] if class_counts.get(c, 0) > 1]

        total_det = len(detections)

        per_image_rows.append(
            {
                "model": model_name,
                "set": set_name,
                "image": img_path.name,
                "stem": stem,
                "total_det": total_det,
                "unique_cls_count": len(present_classes),
                "has_all4": int(len(present_classes) == 4),
                "has_ge3": int(len(present_classes) >= 3),
                "has_detection": int(total_det > 0),
                "missing_classes": ",".join(str(c) for c in [0, 1, 2, 3] if class_counts.get(c, 0) == 0),
                "duplicate_classes": ",".join(str(c) for c in dup_classes),
                "cls0_count": class_counts.get(0, 0),
                "cls1_count": class_counts.get(1, 0),
                "cls2_count": class_counts.get(2, 0),
                "cls3_count": class_counts.get(3, 0),
                "image_path": str(img_path),
                "label_path": "" if txt_path is None else str(txt_path),
            }
        )

    return per_image_rows, per_det_rows


def summarize_model(per_image_rows, per_det_rows):
    """모델별 summary 생성."""
    by_model = {}

    for row in per_image_rows:
        model = row["model"]

        if model not in by_model:
            by_model[model] = {
                "model": model,
                "total_images": 0,
                "detected_images": 0,
                "no_detection_images": 0,
                "total_detections": 0,
                "avg_det_per_image": 0.0,
                "all4_images": 0,
                "ge3_images": 0,
                "cls0_images": 0,
                "cls1_images": 0,
                "cls2_images": 0,
                "cls3_images": 0,
                "cls0_dup_images": 0,
                "cls1_dup_images": 0,
                "cls2_dup_images": 0,
                "cls3_dup_images": 0,
                "cls0_detections": 0,
                "cls1_detections": 0,
                "cls2_detections": 0,
                "cls3_detections": 0,
            }

        s = by_model[model]

        s["total_images"] += 1
        s["detected_images"] += int(row["has_detection"])
        s["all4_images"] += int(row["has_all4"])
        s["ge3_images"] += int(row["has_ge3"])

        total_det = int(row["total_det"])
        s["total_detections"] += total_det

        for c in [0, 1, 2, 3]:
            cnt = int(row[f"cls{c}_count"])
            s[f"cls{c}_detections"] += cnt

            if cnt > 0:
                s[f"cls{c}_images"] += 1

            if cnt > 1:
                s[f"cls{c}_dup_images"] += 1

    for model, s in by_model.items():
        s["no_detection_images"] = s["total_images"] - s["detected_images"]

        if s["total_images"] > 0:
            s["avg_det_per_image"] = s["total_detections"] / s["total_images"]

    return list(by_model.values())


def summarize_set(per_image_rows):
    """모델별/세트별 summary 생성."""
    by_key = {}

    for row in per_image_rows:
        key = (row["model"], row["set"])

        if key not in by_key:
            by_key[key] = {
                "model": row["model"],
                "set": row["set"],
                "total_images": 0,
                "detected_images": 0,
                "total_detections": 0,
                "all4_images": 0,
                "ge3_images": 0,
                "cls0_images": 0,
                "cls1_images": 0,
                "cls2_images": 0,
                "cls3_images": 0,
            }

        s = by_key[key]

        s["total_images"] += 1
        s["detected_images"] += int(row["has_detection"])
        s["total_detections"] += int(row["total_det"])
        s["all4_images"] += int(row["has_all4"])
        s["ge3_images"] += int(row["has_ge3"])

        for c in [0, 1, 2, 3]:
            if int(row[f"cls{c}_count"]) > 0:
                s[f"cls{c}_images"] += 1

    return list(by_key.values())


# ============================================================
# Visualization
# ============================================================

COLOR_MAP = {
    0: (0, 0, 255),
    1: (0, 255, 0),
    2: (255, 128, 0),
    3: (255, 0, 255),
}


def draw_detections(img_path: Path, label_path: Path, title: str, max_size: int = 420):
    """이미지 위에 detection polygon을 그린다."""
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)

    if img is None:
        canvas = np.zeros((max_size, max_size, 3), dtype=np.uint8)
        cv2.putText(canvas, "IMAGE READ FAIL", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return canvas

    h, w = img.shape[:2]

    detections = parse_yolo_seg_txt(label_path) if label_path and Path(label_path).exists() else []

    for det in detections:
        cls_id = int(det["cls"])
        pts = det["pts"].copy()
        pts[:, 0] *= w
        pts[:, 1] *= h

        poly = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
        color = COLOR_MAP.get(cls_id, (255, 255, 255))

        cv2.polylines(img, [poly], True, color, 2)

        cx, cy = np.mean(pts[:, 0]), np.mean(pts[:, 1])
        conf_txt = "" if det["conf"] is None else f" {det['conf']:.2f}"
        cv2.putText(
            img,
            f"c{cls_id}{conf_txt}",
            (int(cx), int(cy)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    # 크기 축소
    scale = min(max_size / max(h, w), 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    # 제목 영역 추가
    th = 38
    canvas = np.zeros((img.shape[0] + th, img.shape[1], 3), dtype=np.uint8)
    canvas[th:, :] = img
    cv2.putText(canvas, title[:80], (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    return canvas


def make_contact_sheet(rows, out_path: Path, title_key: str, max_items: int = 24, cols: int = 4):
    """선택된 이미지들로 contact sheet 생성."""
    if not rows:
        return False

    items = rows[:max_items]
    thumbs = []

    for row in items:
        img_path = Path(row["image_path"])
        label_path = Path(row["label_path"]) if row.get("label_path") else None

        title = (
            f"{row['model']} | {row['set']} | det={row['total_det']} "
            f"| cls=({row['cls0_count']},{row['cls1_count']},{row['cls2_count']},{row['cls3_count']})"
        )

        thumbs.append(draw_detections(img_path, label_path, title))

    if not thumbs:
        return False

    max_h = max(t.shape[0] for t in thumbs)
    max_w = max(t.shape[1] for t in thumbs)

    padded = []

    for t in thumbs:
        pad = np.zeros((max_h, max_w, 3), dtype=np.uint8)
        pad[:t.shape[0], :t.shape[1]] = t
        padded.append(pad)

    rows_img = []
    for i in range(0, len(padded), cols):
        row_imgs = padded[i:i + cols]

        while len(row_imgs) < cols:
            row_imgs.append(np.zeros((max_h, max_w, 3), dtype=np.uint8))

        rows_img.append(np.hstack(row_imgs))

    sheet = np.vstack(rows_img)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet)

    return True


def make_visual_reports(per_image_rows, out_dir: Path):
    """
    모델별 대표 케이스 시트 생성.
    """
    vis_dir = out_dir / "contact_sheets"
    vis_dir.mkdir(parents=True, exist_ok=True)

    models = sorted(set(row["model"] for row in per_image_rows))

    for model in models:
        model_rows = [r for r in per_image_rows if r["model"] == model]

        # 4개 class 모두 나온 케이스
        all4 = [r for r in model_rows if int(r["has_all4"]) == 1]
        all4 = sorted(all4, key=lambda r: int(r["total_det"]), reverse=True)
        make_contact_sheet(all4, vis_dir / f"{model}_all4.jpg", "all4")

        # 3개 이상 class 나온 케이스
        ge3 = [r for r in model_rows if int(r["has_ge3"]) == 1 and int(r["has_all4"]) == 0]
        ge3 = sorted(ge3, key=lambda r: int(r["total_det"]), reverse=True)
        make_contact_sheet(ge3, vis_dir / f"{model}_ge3_not_all4.jpg", "ge3")

        # 중복이 많은 케이스
        dup = []
        for r in model_rows:
            max_dup = max(
                int(r["cls0_count"]),
                int(r["cls1_count"]),
                int(r["cls2_count"]),
                int(r["cls3_count"]),
            )

            if max_dup >= 3:
                dup.append(r)

        dup = sorted(dup, key=lambda r: int(r["total_det"]), reverse=True)
        make_contact_sheet(dup, vis_dir / f"{model}_high_duplicate.jpg", "duplicate")

        # class2 누락 케이스
        c2_missing = [r for r in model_rows if int(r["has_detection"]) == 1 and int(r["cls2_count"]) == 0]
        c2_missing = sorted(c2_missing, key=lambda r: int(r["total_det"]), reverse=True)
        make_contact_sheet(c2_missing, vis_dir / f"{model}_class2_missing.jpg", "class2_missing")


# ============================================================
# CSV / TXT output
# ============================================================

def write_csv(path: Path, rows):
    """dict row list를 CSV로 저장한다."""
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def write_report(out_dir: Path, model_summary):
    """사람이 바로 읽을 수 있는 summary txt 저장."""
    lines = []
    lines.append("============================================")
    lines.append("INITIAL PREDICTION COMPARISON SUMMARY")
    lines.append("============================================")
    lines.append("")

    sorted_summary = sorted(
        model_summary,
        key=lambda s: (
            int(s["all4_images"]),
            int(s["ge3_images"]),
            int(s["detected_images"]),
            -int(s["total_detections"]),
        ),
        reverse=True,
    )

    for s in sorted_summary:
        lines.append(f"[{s['model']}]")
        lines.append(f"  total_images        : {s['total_images']}")
        lines.append(f"  detected_images     : {s['detected_images']}")
        lines.append(f"  no_detection_images : {s['no_detection_images']}")
        lines.append(f"  total_detections    : {s['total_detections']}")
        lines.append(f"  avg_det_per_image   : {s['avg_det_per_image']:.3f}")
        lines.append(f"  all4_images         : {s['all4_images']}")
        lines.append(f"  ge3_images          : {s['ge3_images']}")
        lines.append(f"  cls0 images/dets    : {s['cls0_images']} / {s['cls0_detections']}")
        lines.append(f"  cls1 images/dets    : {s['cls1_images']} / {s['cls1_detections']}")
        lines.append(f"  cls2 images/dets    : {s['cls2_images']} / {s['cls2_detections']}")
        lines.append(f"  cls3 images/dets    : {s['cls3_images']} / {s['cls3_detections']}")
        lines.append(f"  cls0 dup images     : {s['cls0_dup_images']}")
        lines.append(f"  cls1 dup images     : {s['cls1_dup_images']}")
        lines.append(f"  cls2 dup images     : {s['cls2_dup_images']}")
        lines.append(f"  cls3 dup images     : {s['cls3_dup_images']}")
        lines.append("")

    lines.append("--------------------------------------------")
    lines.append("판단 기준")
    lines.append("--------------------------------------------")
    lines.append("1. all4_images가 많을수록 방향 추정 후보가 많음.")
    lines.append("2. ge3_images가 많으면 4개 ID 복원 가능성이 있음.")
    lines.append("3. total_detections가 너무 많으면 중복/오인이 많을 수 있음.")
    lines.append("4. detected_images가 낮으면 후처리할 후보 자체가 부족함.")
    lines.append("5. 최종 기준 모델은 all4/ge3와 중복 정도를 같이 보고 선택해야 함.")
    lines.append("")

    (out_dir / "summary_report.txt").write_text("\n".join(lines), encoding="utf-8")


def parse_model_arg(item: str):
    """
    --pred 인자 파싱.
    형식:
        name=/path/to/predict_root
    """
    if "=" not in item:
        raise ValueError(f"--pred 형식 오류: {item} / 예: noise_v1=/path/to/root")

    name, path = item.split("=", 1)
    name = name.strip()
    path = Path(path.strip())

    if not name:
        raise ValueError(f"모델 이름이 비어 있습니다: {item}")

    return name, path


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pred",
        action="append",
        required=True,
        help="비교할 predict 결과. 형식: name=/path/to/predict_root",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="분석 결과 저장 폴더",
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_per_image = []
    all_per_det = []

    for item in args.pred:
        model_name, pred_root = parse_model_arg(item)

        if not pred_root.exists():
            print(f"[SKIP] predict root 없음: {model_name} -> {pred_root}")
            continue

        print(f"[ANALYZE] {model_name}: {pred_root}")

        per_image, per_det = analyze_one_model(model_name, pred_root)

        print(f"  images={len(per_image)}, detections={len(per_det)}")

        all_per_image.extend(per_image)
        all_per_det.extend(per_det)

    model_summary = summarize_model(all_per_image, all_per_det)
    set_summary = summarize_set(all_per_image)

    write_csv(out_dir / "per_image.csv", all_per_image)
    write_csv(out_dir / "per_detection.csv", all_per_det)
    write_csv(out_dir / "model_summary.csv", model_summary)
    write_csv(out_dir / "set_summary.csv", set_summary)

    write_report(out_dir, model_summary)
    make_visual_reports(all_per_image, out_dir)

    print("============================================")
    print("[DONE] 초기 결과 비교 분석 완료")
    print("OUT:", out_dir)
    print("SUMMARY:", out_dir / "summary_report.txt")
    print("MODEL CSV:", out_dir / "model_summary.csv")
    print("SET CSV:", out_dir / "set_summary.csv")
    print("IMAGE CSV:", out_dir / "per_image.csv")
    print("DET CSV:", out_dir / "per_detection.csv")
    print("SHEETS:", out_dir / "contact_sheets")
    print("============================================")


if __name__ == "__main__":
    main()
