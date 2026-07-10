#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import shutil
import random
import cv2
import numpy as np

ROOT = Path("/Users/hajiwan/Desktop/object_detection/new_tema/yolov2/dataset/dataset11")

BASE = ROOT / "damage_manual_poly_D3_real_empirical_v2_spatial_train160_val40"
OUT = ROOT / "damage_manual_poly_D3_real_empirical_v2_c0c2_boost_train200_val40"

EXTRA_N = 40
SEED = 42
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

BOOST_CLASSES = {0, 2}

def imread(p):
    data = np.fromfile(str(p), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

def imwrite(p, img):
    p.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(p.suffix.lower(), img)
    if not ok:
        raise RuntimeError(p)
    buf.tofile(str(p))

def list_imgs(p):
    return sorted([x for x in p.iterdir() if x.suffix.lower() in IMG_EXTS])

def class_mask(label_path, w, h):
    mask = np.zeros((h, w), np.uint8)

    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 7:
            continue

        cls = int(float(parts[0]))
        if cls not in BOOST_CLASSES:
            continue

        xy = np.array([float(v) for v in parts[1:]], dtype=np.float32).reshape(-1, 2)
        xy[:, 0] *= w
        xy[:, 1] *= h
        pts = np.round(xy).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [pts], 255)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask

def boost_c0c2(img, mask):
    out = img.copy()

    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    m = mask > 0

    # class0/class2 영역만 약하게 강화
    hsv[:, :, 1][m] = np.clip(hsv[:, :, 1][m] * 1.35, 0, 255)
    hsv[:, :, 2][m] = np.clip(hsv[:, :, 2][m] * 1.25 + 8, 0, 255)

    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # class0/class2 경계만 약하게 선명화
    blur = cv2.GaussianBlur(out, (0, 0), 1.0)
    sharp = cv2.addWeighted(out, 1.45, blur, -0.45, 0)
    out[m] = sharp[m]

    return out

def copy_pair(img, src_lbl_dir, out_img_dir, out_lbl_dir, prefix=""):
    lbl = src_lbl_dir / f"{img.stem}.txt"
    if not lbl.exists():
        raise FileNotFoundError(lbl)

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(img, out_img_dir / f"{prefix}{img.name}")
    shutil.copy2(lbl, out_lbl_dir / f"{prefix}{lbl.name}")

def main():
    rng = random.Random(SEED)

    if OUT.exists():
        shutil.rmtree(OUT)

    for d in ["images/train", "labels/train", "images/val", "labels/val", "qc"]:
        (OUT / d).mkdir(parents=True, exist_ok=True)

    # 기존 train/val 그대로 복사
    for img in list_imgs(BASE / "images/train"):
        copy_pair(img, BASE / "labels/train", OUT / "images/train", OUT / "labels/train")

    for img in list_imgs(BASE / "images/val"):
        copy_pair(img, BASE / "labels/val", OUT / "images/val", OUT / "labels/val")

    # class0/class2 강화 추가 40장
    train_imgs = list_imgs(BASE / "images/train")
    selected = sorted(rng.sample(train_imgs, EXTRA_N), key=lambda p: p.name)

    for i, img_path in enumerate(selected):
        lbl = BASE / "labels/train" / f"{img_path.stem}.txt"
        img = imread(img_path)
        h, w = img.shape[:2]

        m = class_mask(lbl, w, h)
        boosted = boost_c0c2(img, m)

        out_name = f"c0c2_boost_{i:03d}_{img_path.name}"
        out_lbl = f"c0c2_boost_{i:03d}_{lbl.name}"

        imwrite(OUT / "images/train" / out_name, boosted)
        shutil.copy2(lbl, OUT / "labels/train" / out_lbl)

    (OUT / "data.yaml").write_text(f"""path: {OUT}
train: images/train
val: images/val

names:
  0: class0
  1: class1
  2: class2
  3: class3
""", encoding="utf-8")

    print("[DONE]", OUT)
    print("train images:", len(list_imgs(OUT / "images/train")))
    print("train labels:", len(list((OUT / "labels/train").glob("*.txt"))))
    print("val images:", len(list_imgs(OUT / "images/val")))
    print("val labels:", len(list((OUT / "labels/val").glob("*.txt"))))

if __name__ == "__main__":
    main()
