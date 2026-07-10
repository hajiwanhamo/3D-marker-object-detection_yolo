#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import shutil
import random
import csv
import cv2
import numpy as np

ROOT = Path("/Users/hajiwan/Desktop/object_detection/new_tema/yolov2/dataset/dataset11")

D3_TRAIN = ROOT / "damage_manual_poly_D3_exact"
D3_VAL = ROOT / "_tmp_val_D3_aug"

REAL_DIR = Path("/Users/hajiwan/Desktop/object_detection/new_tema/yolov2/realdata/range_sweep_down_10sets/01_down/images_color")

OUT = ROOT / "damage_manual_poly_D3_real_empirical_v2_spatial_train160_val40"

TRAIN_N = 160
VAL_N = 40
SEED = 42

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def imread(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite(path: Path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix.lower(), img)
    if not ok:
        raise RuntimeError(f"이미지 저장 실패: {path}")
    buf.tofile(str(path))


def list_imgs(path: Path):
    if not path.exists():
        return []
    return sorted([p for p in path.iterdir() if p.suffix.lower() in IMG_EXTS])


def image_stats(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    edges = cv2.Canny(gray, 50, 150)

    total = gray.shape[0] * gray.shape[1]
    nz = gray > 0

    if nz.any():
        nz_sat = float(hsv[:, :, 1][nz].mean())
        nz_val = float(hsv[:, :, 2][nz].mean())
    else:
        nz_sat = 0.0
        nz_val = 0.0

    return {
        "gray_mean": float(gray.mean()),
        "gray_std": float(gray.std()),
        "nonzero_ratio": float(nz.sum() / total),
        "dark_ratio": float((gray < 20).sum() / total),
        "edge_density": float((edges > 0).sum() / total),
        "saturation_mean": float(hsv[:, :, 1].mean()),
        "value_mean": float(hsv[:, :, 2].mean()),
        "nonzero_saturation_mean": nz_sat,
        "nonzero_value_mean": nz_val,
    }


def mean_stats(rows):
    keys = list(rows[0].keys())
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


def collect_real_distribution():
    hsv_pixels = []
    ratios = []
    stats_rows = []

    for p in list_imgs(REAL_DIR):
        img = imread(p)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        nz = gray > 0
        ratios.append(float(nz.sum() / gray.size))
        stats_rows.append(image_stats(img))

        if nz.any():
            hsv_pixels.append(hsv[nz])

    if not hsv_pixels:
        raise RuntimeError("real nonzero pixel이 없음")

    hsv_pool = np.concatenate(hsv_pixels, axis=0).astype(np.uint8)
    real_ratio_values = np.array(ratios, dtype=np.float32)
    real_stat = mean_stats(stats_rows)

    return hsv_pool, real_ratio_values, real_stat


def read_label_mask(label_path: Path, w: int, h: int):
    mask = np.zeros((h, w), dtype=np.uint8)

    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) < 7:
            continue

        coords = np.array([float(v) for v in parts[1:]], dtype=np.float32)
        if len(coords) < 6 or len(coords) % 2 != 0:
            continue

        pts = coords.reshape(-1, 2)
        pts[:, 0] *= w
        pts[:, 1] *= h
        pts = np.round(pts).astype(np.int32).reshape(-1, 1, 2)

        cv2.fillPoly(mask, [pts], 255)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


def transform_empirical(img, label_mask, hsv_pool, real_ratio_values, rng):
    """
    real nonzero 분포를 쓰되, 픽셀을 독립 랜덤으로 뿌리지 않는다.
    D3의 공간 구조를 유지하면서 sparse하게 선택한다.
    """
    h, w = img.shape[:2]
    total = h * w

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv_src = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    src_nz = gray > 0
    if not src_nz.any():
        return np.zeros_like(img)

    # real 이미지 중 하나의 nonzero ratio를 직접 사용
    target_ratio = float(rng.choice(real_ratio_values))
    target_n = int(target_ratio * total)

    candidates = np.argwhere(src_nz)
    target_n = max(1, min(target_n, len(candidates)))

    # 공간적으로 연속된 점을 남기기 위한 importance map
    gray_blur = cv2.GaussianBlur(gray, (0, 0), 1.3)
    gray_norm = gray_blur / max(float(gray_blur.max()), 1.0)

    edge = (cv2.Canny(gray.astype(np.uint8), 30, 90) > 0).astype(np.float32)
    edge = cv2.GaussianBlur(edge, (0, 0), 1.0)

    label = (label_mask > 0).astype(np.float32)
    label = cv2.GaussianBlur(label, (0, 0), 1.5)

    smooth_noise = rng.random((h, w)).astype(np.float32)
    smooth_noise = cv2.GaussianBlur(smooth_noise, (0, 0), 5.0)

    importance = (
        0.50 * gray_norm +
        0.25 * smooth_noise +
        0.15 * edge +
        0.45 * label
    )

    yy = candidates[:, 0]
    xx = candidates[:, 1]
    vals = importance[yy, xx]

    # top-k 선택: 랜덤 픽셀 분산 방지
    if target_n < len(vals):
        idx = np.argpartition(vals, -target_n)[-target_n:]
    else:
        idx = np.arange(len(vals))

    selected = candidates[idx]

    # 너무 점상으로 흩어지지 않게 작은 open 적용
    sel_mask = np.zeros((h, w), dtype=np.uint8)
    sel_mask[selected[:, 0], selected[:, 1]] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    sel_mask = cv2.morphologyEx(sel_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    selected = np.argwhere(sel_mask > 0)
    if len(selected) == 0:
        selected = candidates[idx]

    # real nonzero HSV 통계 사용
    real_h = hsv_pool[:, 0].astype(np.float32)
    real_s = hsv_pool[:, 1].astype(np.float32)
    real_v = hsv_pool[:, 2].astype(np.float32)

    s_mean, s_std = float(real_s.mean()), float(real_s.std())
    v_mean, v_std = float(real_v.mean()), float(real_v.std())

    src_vals = gray[selected[:, 0], selected[:, 1]]
    src_mean = float(src_vals.mean())
    src_std = float(src_vals.std()) + 1e-6
    z = (src_vals - src_mean) / src_std

    out_hsv = np.zeros_like(hsv_src)

    n = len(selected)
    h_sample = real_h[rng.integers(0, len(real_h), size=n)]

    # V/S를 독립 랜덤으로 뿌리지 않고 source 구조에 맞춰 매핑
    s_new = np.clip(s_mean + z * s_std * 0.15 + rng.normal(0, 3, size=n), 0, 255)
    v_new = np.clip(v_mean + z * v_std * 0.35 + rng.normal(0, 4, size=n), 0, 255)

    out_hsv[selected[:, 0], selected[:, 1], 0] = h_sample.astype(np.uint8)
    out_hsv[selected[:, 0], selected[:, 1], 1] = s_new.astype(np.uint8)
    out_hsv[selected[:, 0], selected[:, 1], 2] = v_new.astype(np.uint8)

    out = cv2.cvtColor(out_hsv, cv2.COLOR_HSV2BGR)
    return out


def copy_transform(img_path, src_lbl_dir, out_img_dir, out_lbl_dir, hsv_pool, real_ratio_values, rng):
    lbl = src_lbl_dir / f"{img_path.stem}.txt"
    if not lbl.exists():
        raise FileNotFoundError(f"라벨 없음: {lbl}")

    img = imread(img_path)
    if img is None:
        raise RuntimeError(f"이미지 읽기 실패: {img_path}")

    h, w = img.shape[:2]
    mask = read_label_mask(lbl, w, h)

    out = transform_empirical(img, mask, hsv_pool, real_ratio_values, rng)

    imwrite(out_img_dir / img_path.name, out)
    shutil.copy2(lbl, out_lbl_dir / lbl.name)


def contact_sheet(img_dir: Path, out_path: Path, max_n=40):
    imgs = list_imgs(img_dir)[:max_n]
    thumbs = []

    for p in imgs:
        img = imread(p)
        if img is None:
            continue
        t = cv2.resize(img, (160, 160))
        cv2.putText(t, p.stem[:18], (4, 154), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)
        thumbs.append(t)

    if not thumbs:
        return

    cols = 5
    rows = int(np.ceil(len(thumbs) / cols))
    sheet = np.zeros((rows * 160, cols * 160, 3), dtype=np.uint8)

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

    if len(train_candidates) < TRAIN_N:
        raise RuntimeError(f"train 후보 부족: {len(train_candidates)} < {TRAIN_N}")
    if len(val_candidates) < VAL_N:
        raise RuntimeError(f"val 후보 부족: {len(val_candidates)} < {VAL_N}")

    if OUT.exists():
        shutil.rmtree(OUT)

    for d in ["images/train", "labels/train", "images/val", "labels/val", "qc"]:
        (OUT / d).mkdir(parents=True, exist_ok=True)

    selected_train = sorted(py_rng.sample(train_candidates, TRAIN_N), key=lambda p: p.name)
    selected_val = sorted(py_rng.sample(val_candidates, VAL_N), key=lambda p: p.name)

    for img in selected_train:
        copy_transform(
            img,
            D3_TRAIN / "labels/train",
            OUT / "images/train",
            OUT / "labels/train",
            hsv_pool,
            real_ratio_values,
            rng,
        )

    for img in selected_val:
        copy_transform(
            img,
            D3_VAL / "labels/train",
            OUT / "images/val",
            OUT / "labels/val",
            hsv_pool,
            real_ratio_values,
            rng,
        )

    (OUT / "data.yaml").write_text(f"""path: {OUT}
train: images/train
val: images/val

names:
  0: class0
  1: class1
  2: class2
  3: class3
""", encoding="utf-8")

    train_stats = []
    for p in list_imgs(OUT / "images/train"):
        img = imread(p)
        train_stats.append(image_stats(img))

    syn_stat = mean_stats(train_stats)

    with (OUT / "empirical_stats.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["metric", "synthetic", "real"])
        for k in syn_stat:
            wr.writerow([k, f"{syn_stat[k]:.6f}", f"{real_stat[k]:.6f}"])

    contact_sheet(OUT / "images/train", OUT / "qc/train_contact_sheet.png")

    print("============================================")
    print("[DONE]", OUT)
    print("train images:", len(list_imgs(OUT / "images/train")))
    print("train labels:", len(list((OUT / "labels/train").glob("*.txt"))))
    print("val images:", len(list_imgs(OUT / "images/val")))
    print("val labels:", len(list((OUT / "labels/val").glob("*.txt"))))
    print("============================================")
    print("[REAL vs SYNTHETIC]")
    for k in syn_stat:
        print(f"{k}: synthetic={syn_stat[k]:.4f} / real={real_stat[k]:.4f}")
    print("============================================")
    print("stats:", OUT / "empirical_stats.csv")
    print("qc:", OUT / "qc/train_contact_sheet.png")


if __name__ == "__main__":
    main()
