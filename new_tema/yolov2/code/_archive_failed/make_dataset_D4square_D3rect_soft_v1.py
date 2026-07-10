#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_dataset_D4square_D3rect_soft_v1.py

목적:
- 기존 D4square_D3rect가 실해역에서 아무것도 못 잡는 문제 대응.
- 원인 가정: D4 square 픽셀을 D3 이미지에 100% 덮어쓴 것이 너무 강한 domain gap을 만듦.
- 수정: D4 square 영역을 D3 이미지에 약하게 blend한다.

구성:
- 이미지 기본: D3 damage_v1 이미지
- class0 square 영역: D4 이미지를 alpha 비율로 약하게 blend
- label: class0은 D4, class1~3은 D3
"""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def find_images(image_dir: Path):
    out = {}
    for p in sorted(image_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            out[p.stem] = p
    return out


def read_label_lines(label_path: Path):
    if not label_path.exists():
        return []

    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    return [line.strip() for line in text.splitlines() if line.strip()]


def split_label_by_class(lines):
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


def make_class0_mask(label_lines, width, height):
    mask = np.zeros((height, width), dtype=np.uint8)

    for line in label_lines:
        parts = line.split()
        if len(parts) < 7:
            continue

        cls = int(float(parts[0]))
        if cls != 0:
            continue

        coords = [float(x) for x in parts[1:]]

        # dataset label에는 conf가 없어야 하지만, 홀수면 마지막 값 제거
        if len(coords) % 2 == 1:
            coords = coords[:-1]

        if len(coords) < 6:
            continue

        pts = np.array(coords, dtype=np.float32).reshape(-1, 2)
        pts[:, 0] *= float(width)
        pts[:, 1] *= float(height)
        pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)

        cv2.fillPoly(mask, [pts.astype(np.int32)], 255)

    return mask


def process_split(d3_root: Path, d4_root: Path, out_root: Path, check_dir: Path, split: str, alpha: float, check_max: int):
    d3_images = find_images(d3_root / "images" / split)
    d4_images = find_images(d4_root / "images" / split)

    common = sorted(set(d3_images.keys()) & set(d4_images.keys()))
    if not common:
        raise RuntimeError(f"{split}: D3/D4 공통 이미지 없음")

    out_img_dir = out_root / "images" / split
    out_lbl_dir = out_root / "labels" / split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    check_split_dir = check_dir / split
    check_split_dir.mkdir(parents=True, exist_ok=True)

    created = 0
    skipped = 0

    for idx, stem in enumerate(common):
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

        if img_d4.shape[:2] != img_d3.shape[:2]:
            img_d4 = cv2.resize(img_d4, (w, h), interpolation=cv2.INTER_NEAREST)

        d3_lines = read_label_lines(d3_label_path)
        d4_lines = read_label_lines(d4_label_path)

        d3_by_cls = split_label_by_class(d3_lines)
        d4_by_cls = split_label_by_class(d4_lines)

        # class0은 D4 label
        class0_lines = d4_by_cls[0]

        # class1~3은 D3 label
        rect_lines = d3_by_cls[1] + d3_by_cls[2] + d3_by_cls[3]
        out_lines = class0_lines + rect_lines

        # 이미지: D3 기반, class0 영역만 D4를 약하게 blend
        mask = make_class0_mask(class0_lines, w, h)

        out_img = img_d3.copy().astype(np.float32)
        d4_float = img_d4.astype(np.float32)

        m = mask > 0
        out_img[m] = (1.0 - alpha) * out_img[m] + alpha * d4_float[m]
        out_img = np.clip(out_img, 0, 255).astype(np.uint8)

        out_img_path = out_img_dir / d3_img_path.name
        out_lbl_path = out_lbl_dir / f"{stem}.txt"

        cv2.imwrite(str(out_img_path), out_img)
        out_lbl_path.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")

        if idx < check_max:
            check_img = out_img.copy()
            colors = {
                0: (0, 255, 255),
                1: (0, 0, 255),
                2: (0, 255, 0),
                3: (255, 0, 0),
            }

            for line in out_lines:
                parts = line.split()
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
                cv2.polylines(check_img, [pts], True, color, 2)

                x, y, bw, bh = cv2.boundingRect(pts)
                cv2.putText(
                    check_img,
                    f"class{cls}",
                    (x, max(y - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA,
                )

            cv2.imwrite(str(check_split_dir / f"{stem}_check.jpg"), check_img)

        created += 1

    print("=" * 80)
    print("[SPLIT]", split)
    print("common:", len(common))
    print("created:", created)
    print("skipped:", skipped)


def write_data_yaml(out_root: Path):
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
    parser.add_argument("--alpha", type=float, default=0.30)
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

    for split in ["train", "val"]:
        process_split(
            d3_root=d3_root,
            d4_root=d4_root,
            out_root=out_root,
            check_dir=check_dir,
            split=split,
            alpha=float(args.alpha),
            check_max=int(args.check_max_per_split),
        )

    write_data_yaml(out_root)

    print("")
    print("[DONE]")
    print("out_root:", out_root)
    print("data_yaml:", out_root / "data.yaml")
    print("check_dir:", check_dir)
    print("alpha:", args.alpha)


if __name__ == "__main__":
    main()
