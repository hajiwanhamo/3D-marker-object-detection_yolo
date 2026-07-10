#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import shutil, random, csv
import cv2
import numpy as np

ROOT = Path("/Users/hajiwan/Desktop/object_detection/new_tema/yolov2/dataset/dataset11")

# pure D3 후보
D3_TRAIN = ROOT / "damage_manual_poly_D3_exact"
D3_VAL = ROOT / "_tmp_val_D3_aug"

# 출력 dataset
OUT = ROOT / "damage_manual_poly_D3_realstyle_v2_context_train160_val40"

TRAIN_N = 160
VAL_N = 40
SEED = 42
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

# 실해역 통계 기준
TARGET_NONZERO = 0.08
TARGET_SAT = 20.0
VALUE_GAIN = 1.8
VALUE_BIAS = 3.0
NOISE_STD = 2.0

# 핵심 변경점: ID 주변 context 보호 범위
CONTEXT_KERNEL = 41


def imread(p: Path):
    data = np.fromfile(str(p), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite(p: Path, img):
    p.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(p.suffix.lower(), img)
    if not ok:
        raise RuntimeError(f"이미지 저장 실패: {p}")
    buf.tofile(str(p))


def list_imgs(p: Path):
    return sorted([x for x in p.iterdir() if x.suffix.lower() in IMG_EXTS]) if p.exists() else []


def read_label_mask(label_path: Path, w: int, h: int):
    # YOLO segmentation label을 mask로 변환
    mask = np.zeros((h, w), np.uint8)

    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 7:
            continue

        coords = np.array([float(x) for x in parts[1:]], dtype=np.float32).reshape(-1, 2)
        coords[:, 0] *= w
        coords[:, 1] *= h

        pts = np.round(coords).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [pts], 255)

    return mask


def make_context_mask(label_mask: np.ndarray):
    # 기존 v1은 label 내부만 보호
    # v2는 ID 주변 context까지 보호해서 rect 위치관계가 사라지지 않게 함
    k = CONTEXT_KERNEL if CONTEXT_KERNEL % 2 == 1 else CONTEXT_KERNEL + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    context = cv2.dilate(label_mask, kernel, iterations=1)
    return context


def realstyle_v2(img: np.ndarray, label_mask: np.ndarray, rng):
    h, w = img.shape[:2]
    total = h * w

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)

    # 1) 실해역처럼 저채도화
    s = hsv[:, :, 1]
    sat_scale = TARGET_SAT / max(float(s.mean()), 1.0)
    hsv[:, :, 1] = np.clip(s * sat_scale, 0, 60)

    # 2) 실해역처럼 밝기/대비 증가
    v = np.clip(gray * VALUE_GAIN + VALUE_BIAS, 0, 255)
    v += rng.normal(0, NOISE_STD, size=v.shape)
    hsv[:, :, 2] = np.clip(v, 0, 255)

    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # 3) edge 약간 강화
    blur = cv2.GaussianBlur(out, (0, 0), 1.0)
    out = cv2.addWeighted(out, 1.35, blur, -0.35, 0)

    # 4) 실해역처럼 sparse하게 만들되, ID 주변 context는 보호
    context_mask = make_context_mask(label_mask)
    g = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)

    nonzero = g > 0
    protect = context_mask > 0

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


def copy_transform(img_path: Path, src_label_dir: Path, out_img_dir: Path, out_label_dir: Path, rng):
    label_path = src_label_dir / f"{img_path.stem}.txt"
    if not label_path.exists():
        raise FileNotFoundError(label_path)

    img = imread(img_path)
    h, w = img.shape[:2]

    label_mask = read_label_mask(label_path, w, h)
    styled = realstyle_v2(img, label_mask, rng)

    out_img = out_img_dir / img_path.name
    out_lbl = out_label_dir / label_path.name

    imwrite(out_img, styled)
    shutil.copy2(label_path, out_lbl)


def calc_stats(img_dir: Path):
    rows = []

    for p in list_imgs(img_dir):
        img = imread(p)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        edges = cv2.Canny(gray, 50, 150)

        total = gray.shape[0] * gray.shape[1]

        rows.append({
            "file": p.name,
            "gray_mean": float(gray.mean()),
            "gray_std": float(gray.std()),
            "nonzero_ratio": float((gray > 0).sum() / total),
            "dark_ratio": float((gray < 20).sum() / total),
            "edge_density": float((edges > 0).sum() / total),
            "saturation_mean": float(hsv[:, :, 1].mean()),
            "value_mean": float(hsv[:, :, 2].mean()),
        })

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
        raise RuntimeError(f"train 후보 부족: {len(train_imgs)} < {TRAIN_N}")
    if len(val_imgs) < VAL_N:
        raise RuntimeError(f"val 후보 부족: {len(val_imgs)} < {VAL_N}")

    selected_train = sorted(py_rng.sample(train_imgs, TRAIN_N), key=lambda p: p.name)
    selected_val = sorted(py_rng.sample(val_imgs, VAL_N), key=lambda p: p.name)

    for img in selected_train:
        copy_transform(img, D3_TRAIN / "labels/train", OUT / "images/train", OUT / "labels/train", rng)

    for img in selected_val:
        copy_transform(img, D3_VAL / "labels/train", OUT / "images/val", OUT / "labels/val", rng)

    (OUT / "data.yaml").write_text(f"""path: {OUT}
train: images/train
val: images/val

names:
  0: class0
  1: class1
  2: class2
  3: class3
""", encoding="utf-8")

    rows = calc_stats(OUT / "images/train")
    stat_path = OUT / "realstyle_v2_stats.csv"

    with stat_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("[DONE]", OUT)
    print("train images:", len(list_imgs(OUT / "images/train")))
    print("train labels:", len(list((OUT / "labels/train").glob("*.txt"))))
    print("val images:", len(list_imgs(OUT / "images/val")))
    print("val labels:", len(list((OUT / "labels/val").glob("*.txt"))))

    keys = ["gray_mean", "gray_std", "nonzero_ratio", "dark_ratio", "edge_density", "saturation_mean", "value_mean"]
    print("[STATS]")
    for k in keys:
        vals = np.array([r[k] for r in rows], dtype=float)
        print(f"{k}: mean={vals.mean():.4f}, std={vals.std():.4f}")


if __name__ == "__main__":
    main()
