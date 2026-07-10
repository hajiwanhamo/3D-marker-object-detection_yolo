#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import shutil
import random
import csv
import cv2
import numpy as np

# ============================================================
# 목적:
# 기존 empirical_v2_spatial 데이터의 라벨/샘플 구성은 그대로 유지하고,
# 출력 이미지의 global statistics만 실해역 이미지에 더 가깝게 보정한다.
#
# 하지 않는 것:
# - class0/class2 보존 강제 없음
# - fine-tuning 없음
# - 라벨 변경 없음
# - 샘플 제거 없음
# ============================================================

ROOT = Path("/Users/hajiwan/Desktop/object_detection/new_tema/yolov2/dataset/dataset11")

BASE = ROOT / "damage_manual_poly_D3_real_empirical_v2_spatial_train160_val40"
OUT = ROOT / "damage_manual_poly_D3_real_empirical_v2_statsmatch_train160_val40"

REAL_DIR = Path("/Users/hajiwan/Desktop/object_detection/new_tema/yolov2/realdata/range_sweep_down_10sets/01_down/images_color")

SEED = 42
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

# grid search는 전체 160장이 아니라 일부 샘플로 먼저 최적 설정 탐색
SEARCH_N = 40

# real target에 대한 중요도
WEIGHTS = {
    "nonzero_ratio": 3.0,
    "gray_mean": 1.2,
    "gray_std": 2.0,
    "edge_density": 4.0,
    "saturation_mean": 1.5,
    "value_mean": 1.5,
}

def imread(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"이미지 읽기 실패: {path}")
    return img

def imwrite(path: Path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix.lower(), img)
    if not ok:
        raise RuntimeError(f"이미지 저장 실패: {path}")
    buf.tofile(str(path))

def list_imgs(path: Path):
    return sorted([p for p in path.iterdir() if p.suffix.lower() in IMG_EXTS])

def image_stats(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    edge = cv2.Canny(gray, 50, 150)

    total = gray.size
    nz = gray > 0

    return {
        "gray_mean": float(gray.mean()),
        "gray_std": float(gray.std()),
        "nonzero_ratio": float(nz.sum() / total),
        "dark_ratio": float((gray < 20).sum() / total),
        "edge_density": float((edge > 0).sum() / total),
        "saturation_mean": float(hsv[:, :, 1].mean()),
        "value_mean": float(hsv[:, :, 2].mean()),
        "nonzero_saturation_mean": float(hsv[:, :, 1][nz].mean()) if nz.any() else 0.0,
        "nonzero_value_mean": float(hsv[:, :, 2][nz].mean()) if nz.any() else 0.0,
    }

def mean_stats(rows):
    keys = rows[0].keys()
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}

def collect_real_stats_and_pool():
    rows = []
    hsv_pool = []
    gray_pool = []

    for p in list_imgs(REAL_DIR):
        img = imread(p)
        rows.append(image_stats(img))

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        nz = gray > 0

        if nz.any():
            hsv_pool.append(hsv[nz])
            gray_pool.append(gray[nz])

    if len(hsv_pool) == 0:
        raise RuntimeError("real 이미지에서 nonzero pixel을 찾지 못함")

    real_stat = mean_stats(rows)
    hsv_pool = np.concatenate(hsv_pool, axis=0).astype(np.uint8)
    gray_pool = np.concatenate(gray_pool, axis=0).astype(np.uint8)

    return real_stat, hsv_pool, gray_pool

def score_stats(syn, real):
    score = 0.0

    for k, w in WEIGHTS.items():
        denom = abs(real[k]) + 1e-6
        score += w * abs(syn[k] - real[k]) / denom

    return float(score)

def select_mask_by_score(gray, target_ratio, sigma, dilate_k):
    h, w = gray.shape
    total = h * w
    target_n = int(round(total * target_ratio))

    src_nz = gray > 0
    if not src_nz.any():
        return np.zeros_like(gray, dtype=np.uint8)

    # 기존 nonzero 구조를 부드럽게 확산시켜 candidate 생성
    score = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), sigma)

    base = src_nz.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_k, dilate_k))
    candidate = cv2.dilate(base, kernel, iterations=1) > 0

    yyxx = np.argwhere(candidate)
    if len(yyxx) == 0:
        return np.zeros_like(gray, dtype=np.uint8)

    target_n = min(target_n, len(yyxx))

    vals = score[yyxx[:, 0], yyxx[:, 1]]

    # score가 높은 pixel부터 target_n개 선택
    idx = np.argpartition(vals, -target_n)[-target_n:]
    selected = yyxx[idx]

    mask = np.zeros_like(gray, dtype=np.uint8)
    mask[selected[:, 0], selected[:, 1]] = 255

    return mask

def assign_real_hsv_by_rank(mask, score_img, hsv_pool, gray_pool, rng):
    h, w = mask.shape
    selected = np.argwhere(mask > 0)

    out_hsv = np.zeros((h, w, 3), dtype=np.uint8)
    n = len(selected)

    if n == 0:
        return out_hsv

    # 선택된 pixel을 score 순서로 정렬
    coord_score = score_img[selected[:, 0], selected[:, 1]]
    coord_order = np.argsort(coord_score)
    selected_sorted = selected[coord_order]

    # real nonzero HSV를 샘플링하되, gray 순서로 정렬해서 공간적으로 부드럽게 배치
    sample_idx = rng.integers(0, len(hsv_pool), size=n)
    sample_hsv = hsv_pool[sample_idx]
    sample_gray = gray_pool[sample_idx]

    hsv_order = np.argsort(sample_gray)
    sample_sorted = sample_hsv[hsv_order]

    out_hsv[selected_sorted[:, 0], selected_sorted[:, 1]] = sample_sorted

    return out_hsv

def transform_image(img, cfg, real_stat, hsv_pool, gray_pool, rng):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    target_ratio = real_stat["nonzero_ratio"] * cfg["ratio_scale"]

    score_img = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), cfg["sigma"])
    mask = select_mask_by_score(
        gray=gray,
        target_ratio=target_ratio,
        sigma=cfg["sigma"],
        dilate_k=cfg["dilate_k"],
    )

    out_hsv = assign_real_hsv_by_rank(mask, score_img, hsv_pool, gray_pool, rng)
    out = cv2.cvtColor(out_hsv, cv2.COLOR_HSV2BGR)

    # 내부 salt edge를 줄이기 위한 약한 blur
    # outside는 다시 0으로 고정해서 nonzero_ratio가 변하지 않게 한다.
    if cfg["final_blur"] > 0:
        blur = cv2.GaussianBlur(out, (0, 0), cfg["final_blur"])
        out[mask > 0] = blur[mask > 0]
        out[mask == 0] = 0

    return out

def evaluate_config(imgs, cfg, real_stat, hsv_pool, gray_pool):
    rng = np.random.default_rng(SEED)
    rows = []

    for p in imgs:
        img = imread(p)
        out = transform_image(img, cfg, real_stat, hsv_pool, gray_pool, rng)
        rows.append(image_stats(out))

    syn = mean_stats(rows)
    return score_stats(syn, real_stat), syn

def make_contact_sheet(img_dir, out_path, max_n=40):
    imgs = list_imgs(img_dir)[:max_n]
    if len(imgs) == 0:
        return

    thumbs = []
    for p in imgs:
        img = imread(p)
        thumb = cv2.resize(img, (160, 160))
        cv2.putText(
            thumb,
            p.stem[:18],
            (4, 154),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        thumbs.append(thumb)

    cols = 5
    rows = int(np.ceil(len(thumbs) / cols))
    sheet = np.zeros((rows * 160, cols * 160, 3), dtype=np.uint8)

    for i, t in enumerate(thumbs):
        y = (i // cols) * 160
        x = (i % cols) * 160
        sheet[y:y + 160, x:x + 160] = t

    imwrite(out_path, sheet)

def copy_labels(src_label_dir, dst_label_dir):
    dst_label_dir.mkdir(parents=True, exist_ok=True)
    for p in sorted(src_label_dir.glob("*.txt")):
        shutil.copy2(p, dst_label_dir / p.name)

def transform_split(split, cfg, real_stat, hsv_pool, gray_pool):
    rng = np.random.default_rng(SEED + (0 if split == "train" else 1000))

    src_img_dir = BASE / "images" / split
    src_lbl_dir = BASE / "labels" / split

    dst_img_dir = OUT / "images" / split
    dst_lbl_dir = OUT / "labels" / split

    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for img_path in list_imgs(src_img_dir):
        img = imread(img_path)
        out = transform_image(img, cfg, real_stat, hsv_pool, gray_pool, rng)

        imwrite(dst_img_dir / img_path.name, out)
        shutil.copy2(src_lbl_dir / f"{img_path.stem}.txt", dst_lbl_dir / f"{img_path.stem}.txt")

        rows.append(image_stats(out))

    return mean_stats(rows)

def main():
    random.seed(SEED)
    np.random.seed(SEED)

    if not BASE.exists():
        raise FileNotFoundError(f"BASE 없음: {BASE}")

    real_stat, hsv_pool, gray_pool = collect_real_stats_and_pool()

    train_imgs = list_imgs(BASE / "images/train")
    search_imgs = train_imgs[:SEARCH_N]

    configs = []

    for sigma in [1.2, 1.6, 2.0, 2.4]:
        for dilate_k in [3, 5, 7]:
            for final_blur in [0.0, 0.5, 0.8]:
                for ratio_scale in [0.98, 1.00, 1.02]:
                    configs.append({
                        "sigma": sigma,
                        "dilate_k": dilate_k,
                        "final_blur": final_blur,
                        "ratio_scale": ratio_scale,
                    })

    ranked = []

    print("[SEARCH] configs:", len(configs), "images:", len(search_imgs))

    for i, cfg in enumerate(configs, 1):
        sc, syn = evaluate_config(search_imgs, cfg, real_stat, hsv_pool, gray_pool)
        ranked.append((sc, cfg, syn))
        print(f"[{i:03d}/{len(configs)}] score={sc:.6f} cfg={cfg} edge={syn['edge_density']:.4f} nz={syn['nonzero_ratio']:.4f} gray_std={syn['gray_std']:.4f}")

    ranked.sort(key=lambda x: x[0])
    best_score, best_cfg, best_search_stat = ranked[0]

    print("\n[BEST CONFIG]")
    print("score:", best_score)
    print(best_cfg)

    if OUT.exists():
        shutil.rmtree(OUT)

    for d in ["images/train", "labels/train", "images/val", "labels/val", "qc"]:
        (OUT / d).mkdir(parents=True, exist_ok=True)

    train_stat = transform_split("train", best_cfg, real_stat, hsv_pool, gray_pool)
    val_stat = transform_split("val", best_cfg, real_stat, hsv_pool, gray_pool)

    (OUT / "data.yaml").write_text(f"""path: {OUT}
train: images/train
val: images/val

names:
  0: class0
  1: class1
  2: class2
  3: class3
""", encoding="utf-8")

    with (OUT / "statsmatch_stats.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["metric", "train_synthetic", "val_synthetic", "real"])
        for k in train_stat.keys():
            wr.writerow([k, f"{train_stat[k]:.6f}", f"{val_stat[k]:.6f}", f"{real_stat[k]:.6f}"])

    make_contact_sheet(OUT / "images/train", OUT / "qc/train_contact_sheet.png")
    make_contact_sheet(OUT / "images/val", OUT / "qc/val_contact_sheet.png")

    print("\n[DONE]", OUT)
    print("train images:", len(list_imgs(OUT / "images/train")))
    print("train labels:", len(list((OUT / "labels/train").glob("*.txt"))))
    print("val images:", len(list_imgs(OUT / "images/val")))
    print("val labels:", len(list((OUT / "labels/val").glob("*.txt"))))

    print("\n[TRAIN SYNTHETIC vs REAL]")
    for k in train_stat:
        print(f"{k}: synthetic={train_stat[k]:.4f} / real={real_stat[k]:.4f}")

    print("\n[QC]")
    print(OUT / "qc/train_contact_sheet.png")
    print(OUT / "statsmatch_stats.csv")

if __name__ == "__main__":
    main()
