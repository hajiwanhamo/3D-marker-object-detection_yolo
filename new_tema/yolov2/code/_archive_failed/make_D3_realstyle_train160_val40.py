#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import shutil, random, csv
import cv2
import numpy as np

ROOT = Path("/Users/hajiwan/Desktop/object_detection/new_tema/yolov2/dataset/dataset11")
D3_TRAIN = ROOT / "damage_manual_poly_D3_exact"
D3_VAL = ROOT / "_tmp_val_D3_aug"
OUT = ROOT / "damage_manual_poly_D3_realstyle_train160_val40"

TRAIN_N = 160
VAL_N = 40
SEED = 42

TARGET_NONZERO = 0.08
TARGET_SAT = 20.0
VALUE_GAIN = 1.8
VALUE_BIAS = 3.0

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

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

def read_label_mask(label_path, w, h):
    mask = np.zeros((h, w), np.uint8)
    if not label_path.exists():
        return mask
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        vals = np.array([float(x) for x in parts[1:]], dtype=np.float32).reshape(-1, 2)
        vals[:, 0] *= w
        vals[:, 1] *= h
        pts = np.round(vals).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [pts], 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.dilate(mask, kernel, iterations=1)

def realstyle(img, label_mask, rng):
    h, w = img.shape[:2]
    total = h * w

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)

    # real처럼 저채도화
    s = hsv[:, :, 1]
    scale_s = TARGET_SAT / max(float(s.mean()), 1.0)
    hsv[:, :, 1] = np.clip(s * scale_s, 0, 60)

    # real처럼 밝기/대비 증가
    v = np.clip(gray * VALUE_GAIN + VALUE_BIAS, 0, 255)
    noise = rng.normal(0, 2.0, size=v.shape)
    v = np.clip(v + noise, 0, 255)
    hsv[:, :, 2] = v

    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # nonzero ratio를 real에 가깝게 낮춤
    g = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    nonzero = g > 0
    protect = label_mask > 0

    cur = int(nonzero.sum())
    target = int(TARGET_NONZERO * total)

    if cur > target:
        removable = np.argwhere(nonzero & (~protect))
        remove_n = min(len(removable), cur - target)
        if remove_n > 0:
            idx = rng.choice(len(removable), size=remove_n, replace=False)
            yyxx = removable[idx]
            out[yyxx[:, 0], yyxx[:, 1]] = 0

    return out

def copy_transform(img, src_lbl_dir, out_img_dir, out_lbl_dir, prefix, rng):
    lbl = src_lbl_dir / f"{img.stem}.txt"
    if not lbl.exists():
        raise FileNotFoundError(lbl)

    raw = imread(img)
    h, w = raw.shape[:2]
    mask = read_label_mask(lbl, w, h)
    styled = realstyle(raw, mask, rng)

    out_img = out_img_dir / f"{prefix}{img.name}"
    out_lbl = out_lbl_dir / f"{prefix}{lbl.name}"

    imwrite(out_img, styled)
    shutil.copy2(lbl, out_lbl)

def stats(img_dir, name):
    rows = []
    for p in list_imgs(img_dir):
        img = imread(p)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        edges = cv2.Canny(gray, 50, 150)
        total = gray.shape[0] * gray.shape[1]
        rows.append([
            name, p.name,
            float(gray.mean()),
            float(gray.std()),
            float((gray > 0).sum() / total),
            float((gray < 20).sum() / total),
            float((edges > 0).sum() / total),
            float(hsv[:, :, 1].mean()),
            float(hsv[:, :, 2].mean()),
        ])
    return rows

def main():
    rng = np.random.default_rng(SEED)
    py_rng = random.Random(SEED)

    if OUT.exists():
        shutil.rmtree(OUT)

    for d in ["images/train", "labels/train", "images/val", "labels/val"]:
        (OUT / d).mkdir(parents=True, exist_ok=True)

    train_imgs = list_imgs(D3_TRAIN / "images/train")
    val_imgs = list_imgs(D3_VAL / "images/train")

    if len(train_imgs) < TRAIN_N:
        raise RuntimeError(f"train 부족: {len(train_imgs)} < {TRAIN_N}")
    if len(val_imgs) < VAL_N:
        raise RuntimeError(f"val 부족: {len(val_imgs)} < {VAL_N}")

    for img in sorted(py_rng.sample(train_imgs, TRAIN_N), key=lambda p: p.name):
        copy_transform(img, D3_TRAIN / "labels/train", OUT / "images/train", OUT / "labels/train", "", rng)

    for img in sorted(py_rng.sample(val_imgs, VAL_N), key=lambda p: p.name):
        copy_transform(img, D3_VAL / "labels/train", OUT / "images/val", OUT / "labels/val", "", rng)

    (OUT / "data.yaml").write_text(f"""path: {OUT}
train: images/train
val: images/val

names:
  0: class0
  1: class1
  2: class2
  3: class3
""", encoding="utf-8")

    stat_rows = stats(OUT / "images/train", "D3_realstyle_train")
    stat_csv = OUT / "realstyle_stats.csv"
    with stat_csv.open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["domain","file","gray_mean","gray_std","nonzero_ratio","dark_ratio","edge_density","saturation_mean","value_mean"])
        wr.writerows(stat_rows)

    arr = np.array([[r[2],r[3],r[4],r[5],r[6],r[7],r[8]] for r in stat_rows], dtype=float)
    print("[DONE]", OUT)
    print("train images:", len(list_imgs(OUT / "images/train")))
    print("train labels:", len(list((OUT / "labels/train").glob("*.txt"))))
    print("val images:", len(list_imgs(OUT / "images/val")))
    print("val labels:", len(list((OUT / "labels/val").glob("*.txt"))))
    print("gray_mean, gray_std, nonzero, dark, edge, sat, value")
    print(arr.mean(axis=0))

if __name__ == "__main__":
    main()
