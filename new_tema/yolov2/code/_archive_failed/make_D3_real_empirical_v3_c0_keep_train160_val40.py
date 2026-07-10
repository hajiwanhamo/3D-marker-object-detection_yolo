#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import shutil, random, csv
import cv2
import numpy as np

ROOT = Path("/Users/hajiwan/Desktop/object_detection/new_tema/yolov2/dataset/dataset11")
D3_TRAIN = ROOT / "damage_manual_poly_D3_exact"
D3_VAL = ROOT / "_tmp_val_D3_aug"
REAL_DIR = Path("/Users/hajiwan/Desktop/object_detection/new_tema/yolov2/realdata/range_sweep_down_10sets/01_down/images_color")
OUT = ROOT / "damage_manual_poly_D3_real_empirical_v3_c0_keep_train160_val40"

TRAIN_N = 160
VAL_N = 40
SEED = 42
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

# class0/class2 내부 포인트 최소 보존율
C0_KEEP_RATIO = 0.65
C2_KEEP_RATIO = 0.0

def imread(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

def imwrite(path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix.lower(), img)
    if not ok:
        raise RuntimeError(path)
    buf.tofile(str(path))

def list_imgs(path):
    return sorted([p for p in path.iterdir() if p.suffix.lower() in IMG_EXTS]) if path.exists() else []

def image_stats(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    edges = cv2.Canny(gray, 50, 150)
    total = gray.shape[0] * gray.shape[1]
    nz = gray > 0
    return {
        "gray_mean": float(gray.mean()),
        "gray_std": float(gray.std()),
        "nonzero_ratio": float(nz.sum() / total),
        "dark_ratio": float((gray < 20).sum() / total),
        "edge_density": float((edges > 0).sum() / total),
        "saturation_mean": float(hsv[:, :, 1].mean()),
        "value_mean": float(hsv[:, :, 2].mean()),
        "nonzero_saturation_mean": float(hsv[:, :, 1][nz].mean()) if nz.any() else 0.0,
        "nonzero_value_mean": float(hsv[:, :, 2][nz].mean()) if nz.any() else 0.0,
    }

def mean_stats(rows):
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0].keys()}

def collect_real_distribution():
    hsv_pixels, ratios, rows = [], [], []
    for p in list_imgs(REAL_DIR):
        img = imread(p)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        nz = gray > 0
        ratios.append(float(nz.sum() / gray.size))
        rows.append(image_stats(img))
        if nz.any():
            hsv_pixels.append(hsv[nz])
    return np.concatenate(hsv_pixels, axis=0).astype(np.uint8), np.array(ratios, dtype=np.float32), mean_stats(rows)

def read_class_masks(label_path, w, h):
    masks = {0: np.zeros((h, w), np.uint8), 1: np.zeros((h, w), np.uint8), 2: np.zeros((h, w), np.uint8), 3: np.zeros((h, w), np.uint8)}

    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        cls = int(float(parts[0]))
        if cls not in masks:
            continue
        xy = np.array([float(v) for v in parts[1:]], dtype=np.float32).reshape(-1, 2)
        xy[:, 0] *= w
        xy[:, 1] *= h
        pts = np.round(xy).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(masks[cls], [pts], 255)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for c in masks:
        masks[c] = cv2.dilate(masks[c], kernel, iterations=1)

    return masks

def select_top_pixels(candidates, score_map, n):
    if n <= 0 or len(candidates) == 0:
        return np.empty((0, 2), dtype=np.int64)
    n = min(n, len(candidates))
    yy = candidates[:, 0]
    xx = candidates[:, 1]
    vals = score_map[yy, xx]
    idx = np.argpartition(vals, -n)[-n:]
    return candidates[idx]

def transform(img, class_masks, hsv_pool, real_ratio_values, rng):
    h, w = img.shape[:2]
    total = h * w

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv_src = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    src_nz = gray > 0
    if not src_nz.any():
        return np.zeros_like(img)

    target_ratio = float(rng.choice(real_ratio_values))
    target_n = int(target_ratio * total)

    gray_blur = cv2.GaussianBlur(gray, (0, 0), 1.3)
    gray_norm = gray_blur / max(float(gray_blur.max()), 1.0)

    edge = (cv2.Canny(gray.astype(np.uint8), 30, 90) > 0).astype(np.float32)
    edge = cv2.GaussianBlur(edge, (0, 0), 1.0)

    smooth_noise = rng.random((h, w)).astype(np.float32)
    smooth_noise = cv2.GaussianBlur(smooth_noise, (0, 0), 5.0)

    label_all = np.zeros((h, w), np.float32)
    for c in [0, 1, 2, 3]:
        label_all += (class_masks[c] > 0).astype(np.float32)
    label_all = cv2.GaussianBlur(label_all, (0, 0), 1.5)

    importance = 0.50 * gray_norm + 0.25 * smooth_noise + 0.15 * edge + 0.45 * label_all

    selected_mask = np.zeros((h, w), np.uint8)

    # class0 보존
    c0_candidates = np.argwhere(src_nz & (class_masks[0] > 0))
    c0_n = int(len(c0_candidates) * C0_KEEP_RATIO)
    c0_sel = select_top_pixels(c0_candidates, importance, c0_n)
    selected_mask[c0_sel[:, 0], c0_sel[:, 1]] = 255

    # class2 보존
    c2_candidates = np.argwhere(src_nz & (class_masks[2] > 0))
    c2_n = int(len(c2_candidates) * C2_KEEP_RATIO)
    c2_sel = select_top_pixels(c2_candidates, importance, c2_n)
    selected_mask[c2_sel[:, 0], c2_sel[:, 1]] = 255

    # 남은 nonzero는 전체 구조 기준으로 선택
    cur_n = int((selected_mask > 0).sum())
    remain_n = max(1, target_n - cur_n)

    all_candidates = np.argwhere(src_nz & ~(selected_mask > 0))
    rest_sel = select_top_pixels(all_candidates, importance, remain_n)
    selected_mask[rest_sel[:, 0], rest_sel[:, 1]] = 255

    selected = np.argwhere(selected_mask > 0)

    real_h = hsv_pool[:, 0].astype(np.float32)
    real_s = hsv_pool[:, 1].astype(np.float32)
    real_v = hsv_pool[:, 2].astype(np.float32)

    s_mean, s_std = float(real_s.mean()), float(real_s.std())
    v_mean, v_std = float(real_v.mean()), float(real_v.std())

    src_vals = gray[selected[:, 0], selected[:, 1]]
    z = (src_vals - float(src_vals.mean())) / (float(src_vals.std()) + 1e-6)

    out_hsv = np.zeros_like(hsv_src)
    n = len(selected)

    h_sample = real_h[rng.integers(0, len(real_h), size=n)]
    s_new = np.clip(s_mean + z * s_std * 0.15 + rng.normal(0, 3, size=n), 0, 255)
    v_new = np.clip(v_mean + z * v_std * 0.35 + rng.normal(0, 4, size=n), 0, 255)

    out_hsv[selected[:, 0], selected[:, 1], 0] = h_sample.astype(np.uint8)
    out_hsv[selected[:, 0], selected[:, 1], 1] = s_new.astype(np.uint8)
    out_hsv[selected[:, 0], selected[:, 1], 2] = v_new.astype(np.uint8)

    return cv2.cvtColor(out_hsv, cv2.COLOR_HSV2BGR)

def copy_transform(img_path, src_lbl_dir, out_img_dir, out_lbl_dir, hsv_pool, real_ratio_values, rng):
    lbl = src_lbl_dir / f"{img_path.stem}.txt"
    img = imread(img_path)
    h, w = img.shape[:2]
    masks = read_class_masks(lbl, w, h)
    out = transform(img, masks, hsv_pool, real_ratio_values, rng)
    imwrite(out_img_dir / img_path.name, out)
    shutil.copy2(lbl, out_lbl_dir / lbl.name)

def contact_sheet(img_dir, out_path, max_n=40):
    imgs = list_imgs(img_dir)[:max_n]
    thumbs = []
    for p in imgs:
        img = imread(p)
        t = cv2.resize(img, (160, 160))
        cv2.putText(t, p.stem[:18], (4, 154), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)
        thumbs.append(t)
    cols = 5
    rows = int(np.ceil(len(thumbs) / cols))
    sheet = np.zeros((rows * 160, cols * 160, 3), np.uint8)
    for i, t in enumerate(thumbs):
        y = (i // cols) * 160
        x = (i % cols) * 160
        sheet[y:y+160, x:x+160] = t
    imwrite(out_path, sheet)

def main():
    py_rng = random.Random(SEED)
    rng = np.random.default_rng(SEED)

    hsv_pool, real_ratio_values, real_stat = collect_real_distribution()

    train_candidates = list_imgs(D3_TRAIN / "images/train")
    val_candidates = list_imgs(D3_VAL / "images/train")

    if OUT.exists():
        shutil.rmtree(OUT)
    for d in ["images/train", "labels/train", "images/val", "labels/val", "qc"]:
        (OUT / d).mkdir(parents=True, exist_ok=True)

    selected_train = sorted(py_rng.sample(train_candidates, TRAIN_N), key=lambda p: p.name)
    selected_val = sorted(py_rng.sample(val_candidates, VAL_N), key=lambda p: p.name)

    for img in selected_train:
        copy_transform(img, D3_TRAIN / "labels/train", OUT / "images/train", OUT / "labels/train", hsv_pool, real_ratio_values, rng)

    for img in selected_val:
        copy_transform(img, D3_VAL / "labels/train", OUT / "images/val", OUT / "labels/val", hsv_pool, real_ratio_values, rng)

    (OUT / "data.yaml").write_text(f"""path: {OUT}
train: images/train
val: images/val

names:
  0: class0
  1: class1
  2: class2
  3: class3
""", encoding="utf-8")

    rows = [image_stats(imread(p)) for p in list_imgs(OUT / "images/train")]
    syn_stat = mean_stats(rows)

    with (OUT / "empirical_v3_stats.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["metric", "synthetic", "real"])
        for k in syn_stat:
            wr.writerow([k, f"{syn_stat[k]:.6f}", f"{real_stat[k]:.6f}"])

    contact_sheet(OUT / "images/train", OUT / "qc/train_contact_sheet.png")

    print("[DONE]", OUT)
    print("train images:", len(list_imgs(OUT / "images/train")))
    print("train labels:", len(list((OUT / "labels/train").glob("*.txt"))))
    print("val images:", len(list_imgs(OUT / "images/val")))
    print("val labels:", len(list((OUT / "labels/val").glob("*.txt"))))
    print("[REAL vs SYNTHETIC]")
    for k in syn_stat:
        print(f"{k}: synthetic={syn_stat[k]:.4f} / real={real_stat[k]:.4f}")
    print("qc:", OUT / "qc/train_contact_sheet.png")

if __name__ == "__main__":
    main()
