import argparse
import csv
import shutil
from pathlib import Path

import cv2
import numpy as np


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def list_images(img_dir: Path):
    """이미지 폴더 내부 이미지 목록 반환"""
    if not img_dir.exists():
        return []
    return sorted([p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS])


def read_image_size(img_path: Path):
    """이미지 크기 읽기"""
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"[ERROR] 이미지 읽기 실패: {img_path}")
    h, w = img.shape[:2]
    return w, h


def load_yolo_segments(label_path: Path, w: int, h: int):
    """
    YOLO segmentation label txt 읽기.
    한 줄 = 하나의 instance.
    """
    objects = []

    if not label_path.exists():
        return objects

    lines = label_path.read_text(encoding="utf-8", errors="ignore").splitlines()

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
        coords = vals[1:]

        if len(coords) % 2 != 0:
            continue

        pts = np.array(coords, dtype=np.float32).reshape(-1, 2)
        pts[:, 0] *= w
        pts[:, 1] *= h

        objects.append((cls_id, pts))

    return objects


def simplify_polygon(pts: np.ndarray):
    """
    polygon point 수를 적당히 줄인다.
    너무 단순화되어 3점 미만이면 원본 유지.
    """
    if pts is None or len(pts) < 3:
        return pts

    contour = pts.astype(np.float32).reshape(-1, 1, 2)
    peri = cv2.arcLength(contour, True)

    if peri <= 1e-6:
        return pts

    approx = cv2.approxPolyDP(
        contour,
        epsilon=0.0025 * peri,
        closed=True,
    ).reshape(-1, 2)

    if len(approx) < 3:
        return pts

    return approx.astype(np.float32)


def merge_polygons(polygons):
    """
    같은 class가 여러 instance로 들어간 경우 하나로 병합한다.

    현재 목적:
    - class0 square 중복 instance 제거
    - class1/class2도 중복이 있으면 1개 instance로 정리

    방식:
    - 같은 class의 모든 polygon vertex를 모음
    - convex hull로 하나의 polygon 생성
    """
    if len(polygons) == 1:
        return simplify_polygon(polygons[0])

    pts_all = np.concatenate(polygons, axis=0).astype(np.float32)

    if len(pts_all) < 3:
        return pts_all

    hull = cv2.convexHull(pts_all.reshape(-1, 1, 2)).reshape(-1, 2)
    hull = simplify_polygon(hull)

    return hull.astype(np.float32)


def pts_to_yolo_line(cls_id: int, pts: np.ndarray, w: int, h: int):
    """absolute pixel polygon을 YOLO normalized segmentation line으로 변환"""
    if pts is None or len(pts) < 3:
        return None

    pts = pts.astype(np.float32).copy()
    pts[:, 0] = np.clip(pts[:, 0] / max(w, 1), 0.0, 1.0)
    pts[:, 1] = np.clip(pts[:, 1] / max(h, 1), 0.0, 1.0)

    vals = [str(int(cls_id))]

    for x, y in pts:
        vals.append(f"{float(x):.6f}")
        vals.append(f"{float(y):.6f}")

    return " ".join(vals)


def backup_dataset(dataset_root: Path, backup_root: Path):
    """
    dataset11/standard 전체 백업.
    이미 백업 폴더가 있으면 덮어쓰지 않고 중단한다.
    """
    if backup_root.exists():
        raise FileExistsError(
            f"[ERROR] 백업 폴더가 이미 존재합니다. 삭제하거나 다른 이름 사용 필요: {backup_root}"
        )

    shutil.copytree(dataset_root, backup_root)


def process_split(split: str, dataset_root: Path, report_rows):
    """
    dataset_root/labels/{split} 내부 txt를 직접 수정한다.
    """
    img_dir = dataset_root / "images" / split
    lbl_dir = dataset_root / "labels" / split

    images = list_images(img_dir)

    before_counts = {}
    after_counts = {}

    for img_path in images:
        w, h = read_image_size(img_path)

        label_path = lbl_dir / f"{img_path.stem}.txt"
        objects = load_yolo_segments(label_path, w, h)

        grouped = {}

        for cls_id, pts in objects:
            grouped.setdefault(cls_id, []).append(pts)
            before_counts[cls_id] = before_counts.get(cls_id, 0) + 1

        out_lines = []

        for cls_id in sorted(grouped.keys()):
            polygons = grouped[cls_id]
            merged = merge_polygons(polygons)
            line = pts_to_yolo_line(cls_id, merged, w, h)

            if line is not None:
                out_lines.append(line)
                after_counts[cls_id] = after_counts.get(cls_id, 0) + 1

            if len(polygons) > 1:
                report_rows.append({
                    "split": split,
                    "image": img_path.name,
                    "class_id": cls_id,
                    "before_instances": len(polygons),
                    "after_instances": 1,
                })

        label_path.write_text(
            "\n".join(out_lines) + ("\n" if out_lines else ""),
            encoding="utf-8",
        )

    return {
        "split": split,
        "images": len(images),
        "before_counts": before_counts,
        "after_counts": after_counts,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--backup-root", type=str, required=True)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    backup_root = Path(args.backup_root)

    if not dataset_root.exists():
        raise FileNotFoundError(f"[ERROR] dataset-root 없음: {dataset_root}")

    print("============================================")
    print("[STEP 1] standard 전체 백업")
    print("SRC   :", dataset_root)
    print("BACKUP:", backup_root)
    print("============================================")
    backup_dataset(dataset_root, backup_root)

    report_rows = []
    summaries = []

    print("")
    print("============================================")
    print("[STEP 2] labels 직접 수정")
    print("============================================")

    for split in ["train", "val"]:
        summaries.append(process_split(split, dataset_root, report_rows))

    report_csv = dataset_root / "fix_label_report.csv"

    with report_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["split", "image", "class_id", "before_instances", "after_instances"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)

    print("")
    print("============================================")
    print("[DONE] standard labels 직접 수정 완료")
    print("DATASET:", dataset_root)
    print("BACKUP :", backup_root)
    print("REPORT :", report_csv)
    print("============================================")

    for s in summaries:
        print(f"\n[{s['split']}] images={s['images']}")
        print("before:", dict(sorted(s["before_counts"].items())))
        print("after :", dict(sorted(s["after_counts"].items())))

    print("")
    print(f"merged duplicate rows: {len(report_rows)}")


if __name__ == "__main__":
    main()
