#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import shutil, csv
import cv2
import numpy as np

ROOT = Path("/Users/hajiwan/Desktop/object_detection/new_tema/yolov2/dataset/dataset11")
SRC = ROOT / "damage_manual_poly_D3_realstyle_auto_train160_val40"
OUT = ROOT / "damage_manual_poly_D3_realstyle_v4_edge_train160_val40"

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

TARGET_NONZERO = 0.080
EDGE_BRIGHTNESS = 55
VALUE_ADD = 3
EDGE_DILATE = 1

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

def label_mask(label_path, w, h):
    mask = np.zeros((h, w), np.uint8)
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        xy = np.array([float(v) for v in parts[1:]], dtype=np.float32).reshape(-1, 2)
        xy[:, 0] *= w
        xy[:, 1] *= h
        pts = np.round(xy).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [pts], 255)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask

def transform(img, mask, rng):
    out = img.copy()

    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + VALUE_ADD, 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 90)

    if EDGE_DILATE > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=EDGE_DILATE)

    edge_px = edges > 0
    out[edge_px] = np.maximum(out[edge_px], EDGE_BRIGHTNESS)

    h, w = gray.shape
    total = h * w
    target_n = int(TARGET_NONZERO * total)

    gray2 = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    nonzero = gray2 > 0

    protect = (mask > 0) | edge_px

    cur_n = int(nonzero.sum())
    if cur_n > target_n:
        removable = np.argwhere(nonzero & (~protect))
        remove_n = min(len(removable), cur_n - target_n)
        if remove_n > 0:
            idx = rng.choice(len(removable), size=remove_n, replace=False)
            yyxx = removable[idx]
            out[yyxx[:, 0], yyxx[:, 1]] = 0

    return out

def stats(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    edges = cv2.Canny(gray, 50, 150)
    total = gray.shape[0] * gray.shape[1]
    return {
        "gray_mean": float(gray.mean()),
        "gray_std": float(gray.std()),
        "nonzero_ratio": float((gray > 0).sum() / total),
        "dark_ratio": float((gray < 20).sum() / total),
        "edge_density": float((edges > 0).sum() / total),
        "saturation_mean": float(hsv[:, :, 1].mean()),
        "value_mean": float(hsv[:, :, 2].mean()),
    }

def process_split(split, rng):
    rows = []
    for img_path in list_imgs(SRC / "images" / split):
        lbl = SRC / "labels" / split / f"{img_path.stem}.txt"
        img = imread(img_path)
        h, w = img.shape[:2]
        m = label_mask(lbl, w, h)
        out = transform(img, m, rng)

        imwrite(OUT / "images" / split / img_path.name, out)
        shutil.copy2(lbl, OUT / "labels" / split / lbl.name)

        r = stats(out)
        r["split"] = split
        r["file"] = img_path.name
        rows.append(r)
    return rows

def main():
    rng = np.random.default_rng(42)

    if OUT.exists():
        shutil.rmtree(OUT)

    for d in ["images/train", "labels/train", "images/val", "labels/val"]:
        (OUT / d).mkdir(parents=True, exist_ok=True)

    rows = []
    rows += process_split("train", rng)
    rows += process_split("val", rng)

    shutil.copy2(SRC / "data.yaml", OUT / "data.yaml")

    # data.yaml path 수정
    yaml_path = OUT / "data.yaml"
    s = yaml_path.read_text()
    lines = []
    for line in s.splitlines():
        if line.startswith("path:"):
            lines.append(f"path: {OUT}")
        else:
            lines.append(line)
    yaml_path.write_text("\n".join(lines) + "\n")

    with (OUT / "v4_stats.csv").open("w", newline="", encoding="utf-8") as f:
        keys = ["split","file","gray_mean","gray_std","nonzero_ratio","dark_ratio","edge_density","saturation_mean","value_mean"]
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)

    train_rows = [r for r in rows if r["split"] == "train"]
    keys = ["gray_mean","gray_std","nonzero_ratio","dark_ratio","edge_density","saturation_mean","value_mean"]

    print("[DONE]", OUT)
    print("train images:", len(list_imgs(OUT / "images/train")))
    print("train labels:", len(list((OUT / "labels/train").glob("*.txt"))))
    print("val images:", len(list_imgs(OUT / "images/val")))
    print("val labels:", len(list((OUT / "labels/val").glob("*.txt"))))
    print("[TRAIN STATS]")
    for k in keys:
        vals = np.array([r[k] for r in train_rows], dtype=float)
        print(f"{k}: mean={vals.mean():.4f}, std={vals.std():.4f}")

if __name__ == "__main__":
    main()
