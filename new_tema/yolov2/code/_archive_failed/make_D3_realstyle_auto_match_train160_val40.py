#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_D3_realstyle_auto_match_train160_val40.py

목적:
- pure D3 synthetic 이미지를 실해역(real_01_down) 통계에 최대한 가깝게 변환
- 라벨은 기존 YOLO segmentation label 그대로 유지
- train 160 / val 40 dataset 생성
- 학습 전 판단용 stats csv와 QC contact sheet 생성

주의:
- real 라벨은 사용하지 않음
- 후처리 class 재부여 없음
- 학습은 이 코드에서 하지 않음
"""

from pathlib import Path
import shutil
import random
import csv
import itertools
import cv2
import numpy as np


ROOT = Path("/Users/hajiwan/Desktop/object_detection/new_tema/yolov2/dataset/dataset11")

# pure D3 후보
D3_TRAIN = ROOT / "damage_manual_poly_D3_exact"
D3_VAL = ROOT / "_tmp_val_D3_aug"

# output
OUT = ROOT / "damage_manual_poly_D3_realstyle_auto_train160_val40"

TRAIN_N = 160
VAL_N = 40
SEED = 42

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

# real_01_down 통계 기준
TARGET = {
    "gray_mean": 8.7316,
    "gray_std": 32.5194,
    "nonzero_ratio": 0.0795,
    "edge_density": 0.0253,
    "saturation_mean": 19.9141,
    "value_mean": 18.1359,
}

# 학습 진행 가능 기준
RANGE_OK = {
    "nonzero_ratio": (0.070, 0.100),
    "value_mean": (14.0, 22.0),
    "saturation_mean": (12.0, 25.0),
    "edge_density": (0.018, 0.040),
    "gray_std": (25.0, 40.0),
}


def imread_unicode(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix.lower(), img)
    if not ok:
        raise RuntimeError(f"이미지 저장 실패: {path}")
    buf.tofile(str(path))


def list_images(img_dir: Path):
    if not img_dir.exists():
        return []
    return sorted([p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS])


def read_yolo_label_mask(label_path: Path, width: int, height: int, context_kernel: int):
    """
    YOLO segmentation label을 읽어서 ID 주변 보호 mask 생성
    """
    mask = np.zeros((height, width), dtype=np.uint8)

    if not label_path.exists():
        raise FileNotFoundError(f"label 없음: {label_path}")

    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) < 7:
            continue

        coords = np.array([float(v) for v in parts[1:]], dtype=np.float32)
        if len(coords) < 6 or len(coords) % 2 != 0:
            continue

        pts = coords.reshape(-1, 2)
        pts[:, 0] *= width
        pts[:, 1] *= height
        pts = np.round(pts).astype(np.int32).reshape(-1, 1, 2)

        cv2.fillPoly(mask, [pts], 255)

    if context_kernel > 1:
        k = context_kernel if context_kernel % 2 == 1 else context_kernel + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


def transform_realstyle(img, protect_mask, cfg, rng):
    """
    D3 이미지를 real_01_down 스타일에 가깝게 변환
    """
    h, w = img.shape[:2]
    total = h * w

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)

    # 채도 조절
    cur_sat = max(float(hsv[:, :, 1].mean()), 1.0)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (cfg["sat_scale"] / cur_sat), 0, cfg["sat_clip"])

    # 밝기/대비 조절
    v = gray * cfg["gain"] + cfg["bias"]
    v += rng.normal(0, cfg["noise_std"], size=v.shape)
    hsv[:, :, 2] = np.clip(v, 0, 255)

    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # edge 강화
    blur = cv2.GaussianBlur(out, (0, 0), cfg["blur_sigma"])
    out = cv2.addWeighted(out, cfg["sharp_alpha"], blur, 1.0 - cfg["sharp_alpha"], 0)

    # nonzero ratio 조절
    g = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    nonzero = g > 0
    protect = protect_mask > 0

    cur_n = int(nonzero.sum())
    target_n = int(cfg["nonzero_target"] * total)

    if cur_n > target_n:
        removable = np.argwhere(nonzero & (~protect))
        remove_n = min(len(removable), cur_n - target_n)

        if remove_n > 0:
            idx = rng.choice(len(removable), size=remove_n, replace=False)
            yyxx = removable[idx]
            out[yyxx[:, 0], yyxx[:, 1]] = 0

    return out


def image_stats(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    edges = cv2.Canny(gray, 50, 150)

    total = gray.shape[0] * gray.shape[1]

    return {
        "gray_mean": float(gray.mean()),
        "gray_std": float(gray.std()),
        "nonzero_ratio": float((gray > 0).sum() / total),
        "edge_density": float((edges > 0).sum() / total),
        "saturation_mean": float(hsv[:, :, 1].mean()),
        "value_mean": float(hsv[:, :, 2].mean()),
    }


def mean_stats(rows):
    return {k: float(np.mean([r[k] for r in rows])) for k in TARGET.keys()}


def pass_ranges(st):
    result = {}
    for k, (lo, hi) in RANGE_OK.items():
        result[k] = lo <= st[k] <= hi
    return result


def score_stats(st):
    """
    real target과의 상대 오차 + 기준 범위 미달 penalty
    """
    score = 0.0

    weights = {
        "nonzero_ratio": 3.0,
        "edge_density": 3.0,
        "saturation_mean": 2.0,
        "value_mean": 2.0,
        "gray_std": 2.0,
        "gray_mean": 1.0,
    }

    for k, target in TARGET.items():
        score += weights[k] * abs(st[k] - target) / max(target, 1e-6)

    for k, ok in pass_ranges(st).items():
        if not ok:
            score += 5.0

    return score


def make_cfgs():
    """
    너무 넓게 탐색하면 오래 걸리므로 핵심 파라미터만 탐색
    """
    cfgs = []

    for nonzero_target, sat_scale, sat_clip, gain, bias, noise_std, sharp_alpha, context_kernel in itertools.product(
        [0.075, 0.080, 0.085, 0.090],
        [30.0, 45.0, 60.0, 80.0],
        [80.0, 120.0, 180.0],
        [3.4, 4.2, 5.0],
        [6.0, 9.0, 12.0],
        [3.0, 5.0],
        [1.4, 1.7, 2.0],
        [3, 5, 9],
    ):
        cfgs.append({
            "nonzero_target": nonzero_target,
            "sat_scale": sat_scale,
            "sat_clip": sat_clip,
            "gain": gain,
            "bias": bias,
            "noise_std": noise_std,
            "sharp_alpha": sharp_alpha,
            "blur_sigma": 1.0,
            "context_kernel": context_kernel,
        })

    return cfgs


def evaluate_cfg(cfg, sample_images):
    rng = np.random.default_rng(SEED)
    rows = []

    for img_path in sample_images:
        label_path = D3_TRAIN / "labels" / "train" / f"{img_path.stem}.txt"

        img = imread_unicode(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]
        protect = read_yolo_label_mask(label_path, w, h, cfg["context_kernel"])
        out = transform_realstyle(img, protect, cfg, rng)
        rows.append(image_stats(out))

    return mean_stats(rows)


def copy_transform_pair(img_path, src_label_dir, out_img_dir, out_label_dir, cfg, rng):
    label_path = src_label_dir / f"{img_path.stem}.txt"
    if not label_path.exists():
        raise FileNotFoundError(f"label 없음: {label_path}")

    img = imread_unicode(img_path)
    if img is None:
        raise RuntimeError(f"image 읽기 실패: {img_path}")

    h, w = img.shape[:2]
    protect = read_yolo_label_mask(label_path, w, h, cfg["context_kernel"])

    out = transform_realstyle(img, protect, cfg, rng)

    imwrite_unicode(out_img_dir / img_path.name, out)
    shutil.copy2(label_path, out_label_dir / label_path.name)


def make_contact_sheet(img_dir: Path, out_path: Path, max_images: int = 40):
    imgs = list_images(img_dir)[:max_images]
    thumbs = []

    for p in imgs:
        img = imread_unicode(p)
        if img is None:
            continue

        thumb = cv2.resize(img, (160, 160))
        cv2.putText(
            thumb,
            p.stem[:18],
            (4, 155),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        thumbs.append(thumb)

    if not thumbs:
        return

    cols = 5
    rows = int(np.ceil(len(thumbs) / cols))
    sheet = np.zeros((rows * 160, cols * 160, 3), dtype=np.uint8)

    for i, t in enumerate(thumbs):
        y = (i // cols) * 160
        x = (i % cols) * 160
        sheet[y:y+160, x:x+160] = t

    imwrite_unicode(out_path, sheet)


def main():
    py_rng = random.Random(SEED)

    train_candidates = list_images(D3_TRAIN / "images" / "train")
    val_candidates = list_images(D3_VAL / "images" / "train")

    if len(train_candidates) < TRAIN_N:
        raise RuntimeError(f"train 후보 부족: {len(train_candidates)} < {TRAIN_N}")
    if len(val_candidates) < VAL_N:
        raise RuntimeError(f"val 후보 부족: {len(val_candidates)} < {VAL_N}")

    # 빠른 탐색용 sample
    sample_images = sorted(py_rng.sample(train_candidates, min(24, len(train_candidates))), key=lambda p: p.name)

    best = None

    cfgs = make_cfgs()
    print(f"[INFO] search configs: {len(cfgs)}")
    print(f"[INFO] sample images: {len(sample_images)}")

    for i, cfg in enumerate(cfgs, 1):
        st = evaluate_cfg(cfg, sample_images)
        sc = score_stats(st)

        if best is None or sc < best["score"]:
            best = {
                "score": sc,
                "cfg": cfg,
                "stats": st,
            }

        if i % 500 == 0:
            print(f"[SEARCH] {i}/{len(cfgs)} best_score={best['score']:.4f}")

    best_cfg = best["cfg"]

    if OUT.exists():
        shutil.rmtree(OUT)

    for d in ["images/train", "labels/train", "images/val", "labels/val", "qc"]:
        (OUT / d).mkdir(parents=True, exist_ok=True)

    selected_train = sorted(py_rng.sample(train_candidates, TRAIN_N), key=lambda p: p.name)
    selected_val = sorted(py_rng.sample(val_candidates, VAL_N), key=lambda p: p.name)

    rng = np.random.default_rng(SEED)

    for img in selected_train:
        copy_transform_pair(
            img,
            D3_TRAIN / "labels" / "train",
            OUT / "images" / "train",
            OUT / "labels" / "train",
            best_cfg,
            rng,
        )

    for img in selected_val:
        copy_transform_pair(
            img,
            D3_VAL / "labels" / "train",
            OUT / "images" / "val",
            OUT / "labels" / "val",
            best_cfg,
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

    # 최종 train stats
    final_rows = []
    for p in list_images(OUT / "images" / "train"):
        img = imread_unicode(p)
        final_rows.append(image_stats(img))

    final_stats = mean_stats(final_rows)
    range_check = pass_ranges(final_stats)

    # stats 저장
    with (OUT / "auto_match_stats.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "synthetic_mean", "real_target", "ok_range", "pass"])
        for k in TARGET.keys():
            writer.writerow([
                k,
                f"{final_stats[k]:.6f}",
                f"{TARGET[k]:.6f}",
                RANGE_OK.get(k, ""),
                range_check.get(k, ""),
            ])

    with (OUT / "best_config.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "value"])
        for k, v in best_cfg.items():
            writer.writerow([k, v])

    make_contact_sheet(
        OUT / "images" / "train",
        OUT / "qc" / "train_contact_sheet.png",
        max_images=40,
    )

    print("============================================")
    print("[DONE]", OUT)
    print("train images:", len(list_images(OUT / "images" / "train")))
    print("train labels:", len(list((OUT / "labels" / "train").glob("*.txt"))))
    print("val images:", len(list_images(OUT / "images" / "val")))
    print("val labels:", len(list((OUT / "labels" / "val").glob("*.txt"))))
    print("============================================")
    print("[BEST CFG]")
    for k, v in best_cfg.items():
        print(f"{k}: {v}")
    print("============================================")
    print("[FINAL STATS]")
    for k in TARGET.keys():
        mark = "OK" if range_check.get(k, False) else "NG"
        print(f"{k}: {final_stats[k]:.4f} / target {TARGET[k]:.4f} [{mark}]")
    print("============================================")
    print("stats:", OUT / "auto_match_stats.csv")
    print("config:", OUT / "best_config.csv")
    print("qc:", OUT / "qc" / "train_contact_sheet.png")


if __name__ == "__main__":
    main()
