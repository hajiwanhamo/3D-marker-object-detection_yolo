#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_dataset_D4square_D3rect.py

목적:
- 기존 가상 YOLO segmentation dataset 2개를 조합한다.
- class0(square)은 D4 dataset의 이미지/라벨 성향을 사용한다.
- class1~3(rect)은 D3 dataset의 이미지/라벨 성향을 사용한다.

생성 방식:
1. D3 이미지를 기본 이미지로 사용
2. D4 label의 class0 polygon 영역만 mask로 만든다
3. 해당 class0 영역 픽셀을 D4 이미지에서 가져와 D3 이미지 위에 덮어쓴다
4. label은 class0은 D4 label에서, class1~3은 D3 label에서 가져온다
"""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def find_images(image_dir: Path):
    """이미지 폴더에서 stem 기준으로 이미지 경로를 수집한다."""
    out = {}
    if not image_dir.exists():
        return out

    for p in sorted(image_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            out[p.stem] = p

    return out


def read_label_lines(label_path: Path):
    """YOLO segmentation label txt를 줄 단위로 읽는다."""
    if not label_path.exists():
        return []

    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    return [line.strip() for line in text.splitlines() if line.strip()]


def split_label_by_class(lines):
    """라벨 줄을 class 번호별로 분리한다."""
    by_class = {0: [], 1: [], 2: [], 3: []}

    for line in lines:
        parts = line.split()
        if len(parts) < 7:
            continue

        try:
            cls = int(float(parts[0]))
        except ValueError:
            continue

        if cls in by_class:
            by_class[cls].append(line)

    return by_class


def polygon_mask_from_class0(label_lines, width: int, height: int, dilate_iter: int):
    """
    D4 class0 polygon 라벨로부터 square 영역 mask를 만든다.
    이 mask 영역만 D4 이미지에서 가져와 D3 이미지에 덮어쓴다.
    """
    mask = np.zeros((height, width), dtype=np.uint8)

    for line in label_lines:
        parts = line.split()
        if len(parts) < 7:
            continue

        cls = int(float(parts[0]))
        if cls != 0:
            continue

        coords = [float(x) for x in parts[1:]]

        # dataset label에는 confidence가 없어야 정상이다.
        # 혹시 마지막 conf가 섞여 홀수 개가 되면 마지막 값 제거.
        if len(coords) % 2 == 1:
            coords = coords[:-1]

        if len(coords) < 6:
            continue

        pts = np.array(coords, dtype=np.float32).reshape(-1, 2)
        pts[:, 0] *= float(width)
        pts[:, 1] *= float(height)

        pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)

        pts_i = pts.astype(np.int32)
        cv2.fillPoly(mask, [pts_i], 255)

    if dilate_iter > 0:
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=int(dilate_iter))

    return mask


def draw_check_overlay(image, out_label_lines, out_path: Path):
    """생성된 dataset 확인용 overlay 이미지 저장."""
    h, w = image.shape[:2]
    vis = image.copy()

    colors = {
        0: (0, 255, 255),  # class0 square
        1: (0, 0, 255),    # class1
        2: (0, 255, 0),    # class2
        3: (255, 0, 0),    # class3
    }

    for line in out_label_lines:
        parts = line.split()
        if len(parts) < 7:
            continue

        cls = int(float(parts[0]))
        coords = [float(x) for x in parts[1:]]

        if len(coords) % 2 == 1:
            coords = coords[:-1]

        if len(coords) < 6:
            continue

        pts = np.array(coords, dtype=np.float32).reshape(-1, 2)
        pts[:, 0] *= float(w)
        pts[:, 1] *= float(h)
        pts = pts.astype(np.int32)

        color = colors.get(cls, (255, 255, 255))
        cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=2)

        x, y, bw, bh = cv2.boundingRect(pts)
        cv2.putText(
            vis,
            f"class{cls}",
            (x, max(y - 5, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)


def process_split(d3_root: Path, d4_root: Path, out_root: Path, check_dir: Path, split: str, args):
    """train 또는 val split 하나 생성."""
    d3_images = find_images(d3_root / "images" / split)
    d4_images = find_images(d4_root / "images" / split)

    common_stems = sorted(set(d3_images.keys()) & set(d4_images.keys()))

    if not common_stems:
        raise RuntimeError(f"{split}: D3/D4 공통 이미지 없음")

    out_img_dir = out_root / "images" / split
    out_lbl_dir = out_root / "labels" / split

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    check_split_dir = check_dir / split
    check_split_dir.mkdir(parents=True, exist_ok=True)

    created = 0
    skipped = 0

    for idx, stem in enumerate(common_stems):
        d3_img_path = d3_images[stem]
        d4_img_path = d4_images[stem]

        d3_label_path = d3_root / "labels" / split / f"{stem}.txt"
        d4_label_path = d4_root / "labels" / split / f"{stem}.txt"

        img_d3 = cv2.imread(str(d3_img_path), cv2.IMREAD_COLOR)
        img_d4 = cv2.imread(str(d4_img_path), cv2.IMREAD_COLOR)

        if img_d3 is None or img_d4 is None:
            skipped += 1
            continue

        h, w = img_d3.shape[:2]

        # 크기가 다르면 D4 이미지를 D3 이미지 크기에 맞춘다.
        if img_d4.shape[:2] != img_d3.shape[:2]:
            img_d4 = cv2.resize(img_d4, (w, h), interpolation=cv2.INTER_NEAREST)

        d3_lines = read_label_lines(d3_label_path)
        d4_lines = read_label_lines(d4_label_path)

        d3_by_cls = split_label_by_class(d3_lines)
        d4_by_cls = split_label_by_class(d4_lines)

        # class0은 D4 라벨 사용
        class0_lines = d4_by_cls[0]

        # class1~3은 D3 라벨 사용
        rect_lines = d3_by_cls[1] + d3_by_cls[2] + d3_by_cls[3]

        out_label_lines = class0_lines + rect_lines

        # D3 이미지를 기본으로 하고, D4 class0 영역만 D4 이미지로 덮어쓴다.
        out_img = img_d3.copy()

        class0_mask = polygon_mask_from_class0(
            label_lines=class0_lines,
            width=w,
            height=h,
            dilate_iter=args.square_mask_dilate,
        )

        out_img[class0_mask > 0] = img_d4[class0_mask > 0]

        out_img_path = out_img_dir / d3_img_path.name
        out_lbl_path = out_lbl_dir / f"{stem}.txt"

        cv2.imwrite(str(out_img_path), out_img)
        out_lbl_path.write_text("\n".join(out_label_lines) + ("\n" if out_label_lines else ""), encoding="utf-8")

        # 확인용 overlay는 일부만 저장
        if idx < args.check_max_per_split:
            draw_check_overlay(
                image=out_img,
                out_label_lines=out_label_lines,
                out_path=check_split_dir / f"{stem}_check.jpg",
            )

        created += 1

    print("=" * 80)
    print(f"[SPLIT] {split}")
    print("common:", len(common_stems))
    print("created:", created)
    print("skipped:", skipped)
    print("out images:", out_img_dir)
    print("out labels:", out_lbl_dir)


def write_data_yaml(out_root: Path):
    """YOLO data.yaml 저장."""
    text = f"""path: {out_root}
train: images/train
val: images/val

names:
  0: class0
  1: class1
  2: class2
  3: class3
"""
    (out_root / "data.yaml").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--d3_root", required=True)
    parser.add_argument("--d4_root", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--check_dir", required=True)

    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--square_mask_dilate", type=int, default=1)
    parser.add_argument("--check_max_per_split", type=int, default=80)

    args = parser.parse_args()

    d3_root = Path(args.d3_root)
    d4_root = Path(args.d4_root)
    out_root = Path(args.out_root)
    check_dir = Path(args.check_dir)

    if args.clean:
        if out_root.exists():
            shutil.rmtree(out_root)
        if check_dir.exists():
            shutil.rmtree(check_dir)

    for p in [d3_root, d4_root]:
        if not (p / "images" / "train").exists():
            raise FileNotFoundError(f"images/train 없음: {p}")
        if not (p / "labels" / "train").exists():
            raise FileNotFoundError(f"labels/train 없음: {p}")
        if not (p / "images" / "val").exists():
            raise FileNotFoundError(f"images/val 없음: {p}")
        if not (p / "labels" / "val").exists():
            raise FileNotFoundError(f"labels/val 없음: {p}")

    process_split(d3_root, d4_root, out_root, check_dir, "train", args)
    process_split(d3_root, d4_root, out_root, check_dir, "val", args)

    write_data_yaml(out_root)

    print("")
    print("[DONE]")
    print("out_root:", out_root)
    print("data_yaml:", out_root / "data.yaml")
    print("check_dir:", check_dir)


if __name__ == "__main__":
    main()
