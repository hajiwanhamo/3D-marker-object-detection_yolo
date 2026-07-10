#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
convert_labelme_poly_to_yolo_seg.py

목적:
- LabelMe에서 생성한 polygon JSON을 YOLO segmentation txt 라벨로 변환
- images/train/*.json, images/val/*.json을 읽음
- labels/train/*.txt, labels/val/*.txt 생성
- class0~3이 이미지마다 각각 1개씩 존재하는지 검사
- QC overlay 이미지 생성
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

# LabelMe에서 입력한 이름을 YOLO class id로 변환
LABEL_MAP = {
    "class0": 0, "0": 0, "square": 0,
    "class1": 1, "1": 1, "rect1": 1,
    "class2": 2, "2": 2, "rect2": 2,
    "class3": 3, "3": 3, "rect3": 3,
}

CLASS_NAMES = {
    0: "class0",
    1: "class1",
    2: "class2",
    3: "class3",
}


def read_image(path: Path):
    """특수문자 경로 대응 이미지 읽기"""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_image(path: Path, img):
    """특수문자 경로 대응 이미지 저장"""
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix.lower(), img)
    if not ok:
        raise RuntimeError(f"이미지 저장 실패: {path}")
    buf.tofile(str(path))


def list_images(img_dir: Path):
    """이미지 파일 목록"""
    if not img_dir.exists():
        return []
    return sorted([p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS])


def polygon_area(points):
    """polygon 면적 계산"""
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) < 3:
        return 0.0
    return float(abs(cv2.contourArea(pts)))


def rectangle_to_polygon(points):
    """
    LabelMe rectangle이 있을 경우 2점을 4점 polygon으로 변환.
    단, 권장 방식은 Create Polygons 사용.
    """
    if len(points) != 2:
        return None

    x1, y1 = points[0]
    x2, y2 = points[1]

    xmin, xmax = sorted([x1, x2])
    ymin, ymax = sorted([y1, y2])

    return [
        [xmin, ymin],
        [xmax, ymin],
        [xmax, ymax],
        [xmin, ymax],
    ]


def normalize_points(points, width, height):
    """pixel 좌표를 YOLO normalized 좌표로 변환"""
    pts = np.asarray(points, dtype=np.float32).copy()

    pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)

    pts[:, 0] = pts[:, 0] / max(width, 1)
    pts[:, 1] = pts[:, 1] / max(height, 1)

    pts[:, 0] = np.clip(pts[:, 0], 0.0, 1.0)
    pts[:, 1] = np.clip(pts[:, 1], 0.0, 1.0)

    return pts


def draw_qc(img, polygons, out_path: Path):
    """라벨 확인용 overlay 이미지 생성"""
    vis = img.copy()

    colors = {
        0: (0, 255, 255),   # class0
        1: (0, 200, 0),     # class1
        2: (255, 0, 0),     # class2
        3: (0, 0, 255),     # class3
    }

    for cls_id, pts in polygons:
        pts_i = np.asarray(pts, dtype=np.int32).reshape(-1, 1, 2)
        color = colors.get(cls_id, (255, 255, 255))

        cv2.polylines(vis, [pts_i], True, color, 2)

        x, y = pts_i.reshape(-1, 2)[0]
        cv2.putText(
            vis,
            CLASS_NAMES[cls_id],
            (int(x), max(0, int(y) - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    write_image(out_path, vis)


def convert_one_image(img_path: Path, out_label_path: Path, qc_path: Path, strict_one_per_class: bool):
    """이미지 1장에 대응하는 LabelMe JSON을 YOLO txt로 변환"""
    img = read_image(img_path)
    if img is None:
        return False, [f"이미지 읽기 실패: {img_path}"]

    height, width = img.shape[:2]
    json_path = img_path.with_suffix(".json")

    errors = []

    if not json_path.exists():
        return False, [f"JSON 없음: {json_path}"]

    try:
        data = json.loads(json_path.read_text(encoding="utf-8", errors="ignore"))
    except Exception as e:
        return False, [f"JSON 읽기 실패: {json_path} | {e}"]

    shapes = data.get("shapes", [])

    label_lines = []
    qc_polygons = []
    class_counts = {0: 0, 1: 0, 2: 0, 3: 0}

    for shape in shapes:
        raw_label = str(shape.get("label", "")).strip()
        label_key = raw_label.lower()

        if label_key not in LABEL_MAP:
            errors.append(f"알 수 없는 label '{raw_label}': {json_path}")
            continue

        cls_id = LABEL_MAP[label_key]
        shape_type = shape.get("shape_type", "polygon")
        points = shape.get("points", [])

        if shape_type == "rectangle":
            points = rectangle_to_polygon(points)
            if points is None:
                errors.append(f"rectangle 변환 실패: {json_path}")
                continue

        elif shape_type != "polygon":
            errors.append(f"지원하지 않는 shape_type '{shape_type}': {json_path}")
            continue

        if points is None or len(points) < 3:
            errors.append(f"class{cls_id} polygon point 수 부족: {json_path}")
            continue

        if polygon_area(points) < 1.0:
            errors.append(f"class{cls_id} polygon 면적 너무 작음: {json_path}")
            continue

        norm = normalize_points(points, width, height)

        vals = [str(cls_id)]
        for x, y in norm:
            vals.append(f"{float(x):.6f}")
            vals.append(f"{float(y):.6f}")

        label_lines.append(" ".join(vals))
        qc_polygons.append((cls_id, points))
        class_counts[cls_id] += 1

    if strict_one_per_class:
        for cls_id in [0, 1, 2, 3]:
            if class_counts[cls_id] != 1:
                errors.append(
                    f"{img_path.name}: class{cls_id} 개수 {class_counts[cls_id]}개 "
                    f"(정상 기준: 1개)"
                )

    if errors:
        return False, errors

    # class id 순서대로 저장
    label_lines = sorted(label_lines, key=lambda s: int(s.split()[0]))

    out_label_path.parent.mkdir(parents=True, exist_ok=True)
    out_label_path.write_text("\n".join(label_lines) + "\n", encoding="utf-8")

    draw_qc(img, qc_polygons, qc_path)

    return True, []


def write_data_yaml(dataset_root: Path):
    """YOLO data.yaml 생성"""
    text = f"""path: {dataset_root}
train: images/train
val: images/val

names:
  0: class0
  1: class1
  2: class2
  3: class3
"""
    (dataset_root / "data.yaml").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--strict-one-per-class", action="store_true")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)

    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset root 없음: {dataset_root}")

    all_rows = []
    all_errors = []

    for split in ["train", "val"]:
        img_dir = dataset_root / "images" / split
        label_dir = dataset_root / "labels" / split
        qc_dir = dataset_root / "qc" / split

        label_dir.mkdir(parents=True, exist_ok=True)
        qc_dir.mkdir(parents=True, exist_ok=True)

        # 기존 txt 라벨 제거
        for old_txt in label_dir.glob("*.txt"):
            old_txt.unlink()

        images = list_images(img_dir)

        for img_path in images:
            out_label_path = label_dir / f"{img_path.stem}.txt"
            qc_path = qc_dir / f"{img_path.stem}_qc.png"

            ok, errors = convert_one_image(
                img_path=img_path,
                out_label_path=out_label_path,
                qc_path=qc_path,
                strict_one_per_class=args.strict_one_per_class,
            )

            row = {
                "split": split,
                "stem": img_path.stem,
                "image": str(img_path),
                "json": str(img_path.with_suffix(".json")),
                "label_txt": str(out_label_path),
                "qc": str(qc_path),
                "status": "ok" if ok else "error",
            }
            all_rows.append(row)

            if errors:
                for e in errors:
                    all_errors.append(f"[{split}] {e}")

    write_data_yaml(dataset_root)

    report_path = dataset_root / "labelme_to_yolo_report.csv"
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "stem", "image", "json", "label_txt", "qc", "status"],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    if all_errors:
        err_path = dataset_root / "labelme_to_yolo_errors.txt"
        err_path.write_text("\n".join(all_errors), encoding="utf-8")

        print("[ERROR] 변환 실패")
        print(f"errors: {err_path}")
        print(f"report: {report_path}")
        raise SystemExit(1)

    print("[DONE] LabelMe JSON -> YOLO segmentation 변환 완료")
    print(f"dataset: {dataset_root}")
    print(f"report:  {report_path}")
    print(f"qc:      {dataset_root / 'qc'}")
    print(f"yaml:    {dataset_root / 'data.yaml'}")


if __name__ == "__main__":
    main()
