#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_id_crop_dataset.py

목적:
- 기존 YOLO segmentation dataset에서 class0~3 라벨 전체 영역을 기준으로 crop 이미지를 만든다.
- crop 후 polygon 좌표를 crop 이미지 기준으로 다시 변환한다.
- 작은 ID 블록이 이미지 안에서 더 크게 보이도록 만든다.
- 학습 데이터와 라벨 좌표를 동시에 변환하므로 image-label 좌표 불일치를 만들지 않는다.

입력 구조:
src-root/
  images/train/*.png
  images/val/*.png
  labels/train/*.txt
  labels/val/*.txt
  data.yaml

출력 구조:
out-root/
  images/train/*.png
  images/val/*.png
  labels/train/*.txt
  labels/val/*.txt
  data.yaml
"""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def list_images(img_dir: Path):
    """이미지 폴더에서 이미지 파일 목록을 정렬해서 반환한다."""
    if not img_dir.exists():
        return []
    return sorted([p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS])


def read_image(path: Path):
    """OpenCV로 이미지를 읽는다."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"[ERROR] 이미지 읽기 실패: {path}")
    return img


def load_yolo_segments(label_path: Path, w: int, h: int):
    """
    YOLO segmentation txt를 pixel polygon으로 읽는다.

    반환:
        [{"cls": int, "pts": np.ndarray(N,2)}, ...]
    """
    objects = []

    if not label_path.exists():
        return objects

    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
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

        objects.append({"cls": cls_id, "pts": pts})

    return objects


def polygon_area(pts: np.ndarray):
    """polygon 면적 계산."""
    if pts is None or len(pts) < 3:
        return 0.0

    x = pts[:, 0]
    y = pts[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)


def clip_pts_to_crop(pts: np.ndarray, x1: int, y1: int, x2: int, y2: int):
    """
    crop 좌표계로 polygon 좌표를 변환한다.
    단순 clip 방식이다.
    """
    new_pts = pts.copy()
    new_pts[:, 0] = np.clip(new_pts[:, 0], x1, x2 - 1) - x1
    new_pts[:, 1] = np.clip(new_pts[:, 1], y1, y2 - 1) - y1
    return new_pts


def pts_to_yolo_line(cls_id: int, pts: np.ndarray, w: int, h: int):
    """pixel polygon을 YOLO normalized segmentation line으로 변환한다."""
    if pts is None or len(pts) < 3:
        return None

    if polygon_area(pts) < 1.0:
        return None

    pts = pts.astype(np.float32).copy()
    pts[:, 0] = np.clip(pts[:, 0] / max(w, 1), 0.0, 1.0)
    pts[:, 1] = np.clip(pts[:, 1] / max(h, 1), 0.0, 1.0)

    vals = [str(int(cls_id))]

    for x, y in pts:
        vals.append(f"{float(x):.6f}")
        vals.append(f"{float(y):.6f}")

    return " ".join(vals)


def compute_crop_box(objects, img_w: int, img_h: int, pad_ratio: float, min_size: int):
    """
    전체 class0~3 polygon을 포함하는 crop box 계산.
    """
    all_pts = []

    for obj in objects:
        pts = obj["pts"]
        if pts is not None and len(pts) >= 3:
            all_pts.append(pts)

    if not all_pts:
        return None

    pts_all = np.concatenate(all_pts, axis=0)

    xmin = float(np.min(pts_all[:, 0]))
    ymin = float(np.min(pts_all[:, 1]))
    xmax = float(np.max(pts_all[:, 0]))
    ymax = float(np.max(pts_all[:, 1]))

    bw = max(1.0, xmax - xmin)
    bh = max(1.0, ymax - ymin)

    pad = max(bw, bh) * pad_ratio

    cx = (xmin + xmax) * 0.5
    cy = (ymin + ymax) * 0.5

    crop_w = max(bw + 2.0 * pad, float(min_size))
    crop_h = max(bh + 2.0 * pad, float(min_size))

    # 정사각 crop으로 맞춘다.
    side = max(crop_w, crop_h)

    x1 = int(round(cx - side * 0.5))
    y1 = int(round(cy - side * 0.5))
    x2 = int(round(cx + side * 0.5))
    y2 = int(round(cy + side * 0.5))

    # 이미지 범위 밖으로 나가면 이동시킨다.
    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > img_w:
        shift = x2 - img_w
        x1 -= shift
        x2 = img_w
    if y2 > img_h:
        shift = y2 - img_h
        y1 -= shift
        y2 = img_h

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_w, x2)
    y2 = min(img_h, y2)

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def write_data_yaml(src_root: Path, out_root: Path):
    """원본 data.yaml의 names block은 유지하고 path만 out_root로 바꾼다."""
    src_yaml = src_root / "data.yaml"
    names_block = None

    if src_yaml.exists():
        lines = src_yaml.read_text(encoding="utf-8", errors="ignore").splitlines()
        for i, line in enumerate(lines):
            if line.strip().startswith("names:"):
                names_block = "\n".join(lines[i:])
                break

    if names_block is None:
        names_block = """names:
  0: square
  1: rect1
  2: rect2
  3: rect3"""

    text = f"""path: {out_root}
train: images/train
val: images/val

{names_block}
"""
    (out_root / "data.yaml").write_text(text, encoding="utf-8")


def process_split(split: str, src_root: Path, out_root: Path, args):
    """train/val split crop 생성."""
    src_img_dir = src_root / "images" / split
    src_lbl_dir = src_root / "labels" / split

    out_img_dir = out_root / "images" / split
    out_lbl_dir = out_root / "labels" / split

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(src_img_dir)

    made = 0
    skipped_no_label = 0
    skipped_no_object = 0
    empty_after_crop = 0

    for img_path in images:
        img = read_image(img_path)
        h, w = img.shape[:2]

        label_path = src_lbl_dir / f"{img_path.stem}.txt"

        if not label_path.exists():
            skipped_no_label += 1
            continue

        objects = load_yolo_segments(label_path, w, h)

        if not objects:
            skipped_no_object += 1
            continue

        crop_box = compute_crop_box(
            objects=objects,
            img_w=w,
            img_h=h,
            pad_ratio=args.pad_ratio,
            min_size=args.min_crop_size,
        )

        if crop_box is None:
            skipped_no_object += 1
            continue

        x1, y1, x2, y2 = crop_box

        crop = img[y1:y2, x1:x2].copy()
        ch, cw = crop.shape[:2]

        out_lines = []

        for obj in objects:
            cls_id = obj["cls"]
            pts = obj["pts"]

            new_pts = clip_pts_to_crop(pts, x1, y1, x2, y2)

            line = pts_to_yolo_line(cls_id, new_pts, cw, ch)

            if line is not None:
                out_lines.append(line)

        dst_img = out_img_dir / img_path.name
        dst_lbl = out_lbl_dir / f"{img_path.stem}.txt"

        cv2.imwrite(str(dst_img), crop)

        if out_lines:
            dst_lbl.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
        else:
            empty_after_crop += 1
            dst_lbl.write_text("", encoding="utf-8")

        made += 1

    return {
        "split": split,
        "src_images": len(images),
        "made": made,
        "skipped_no_label": skipped_no_label,
        "skipped_no_object": skipped_no_object,
        "empty_after_crop": empty_after_crop,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--src-root", type=str, required=True)
    parser.add_argument("--out-root", type=str, required=True)
    parser.add_argument("--pad-ratio", type=float, default=0.40)
    parser.add_argument("--min-crop-size", type=int, default=256)

    args = parser.parse_args()

    src_root = Path(args.src_root)
    out_root = Path(args.out_root)

    if not src_root.exists():
        raise FileNotFoundError(f"[ERROR] src-root 없음: {src_root}")

    if out_root.exists():
        shutil.rmtree(out_root)

    out_root.mkdir(parents=True, exist_ok=True)

    summaries = []
    for split in ["train", "val"]:
        summaries.append(process_split(split, src_root, out_root, args))

    write_data_yaml(src_root, out_root)

    print("============================================")
    print("[DONE] ID crop dataset 생성 완료")
    print("SRC :", src_root)
    print("OUT :", out_root)
    print("YAML:", out_root / "data.yaml")
    print("============================================")

    for s in summaries:
        print(
            f"{s['split']}: "
            f"src_images={s['src_images']}, "
            f"made={s['made']}, "
            f"skipped_no_label={s['skipped_no_label']}, "
            f"skipped_no_object={s['skipped_no_object']}, "
            f"empty_after_crop={s['empty_after_crop']}"
        )


if __name__ == "__main__":
    main()
