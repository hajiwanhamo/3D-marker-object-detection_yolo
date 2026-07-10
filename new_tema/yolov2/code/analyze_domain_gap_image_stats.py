#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
D3 synthetic 이미지와 실해역 real 이미지의 표현 차이를 수치로 비교한다.
분석 항목:
- brightness mean/std
- contrast
- nonzero pixel ratio
- dark ratio
- edge density
- saturation mean
"""

from pathlib import Path
import csv
import cv2
import numpy as np

D3_DIR = Path("/Users/hajiwan/Desktop/object_detection/new_tema/yolov2/dataset/dataset11/damage_manual_poly_D3_exact_train160_val40/images/train")
REAL_DIR = Path("/Users/hajiwan/Desktop/object_detection/new_tema/yolov2/realdata/range_sweep_down_10sets/01_down/images_color")
OUT_DIR = Path("/Users/hajiwan/Desktop/object_detection/new_tema/yolov2/result/domain_gap_analysis")
OUT_CSV = OUT_DIR / "image_domain_stats_D3_vs_real.csv"

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def imread(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def image_stats(path: Path, domain: str):
    img = imread(path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h, w = gray.shape[:2]
    total = h * w

    nonzero = gray > 0
    dark = gray < 20

    edges = cv2.Canny(gray, 50, 150)

    return {
        "domain": domain,
        "file": path.name,
        "width": w,
        "height": h,
        "gray_mean": float(gray.mean()),
        "gray_std": float(gray.std()),
        "gray_p05": float(np.percentile(gray, 5)),
        "gray_p50": float(np.percentile(gray, 50)),
        "gray_p95": float(np.percentile(gray, 95)),
        "nonzero_ratio": float(nonzero.sum() / total),
        "dark_ratio": float(dark.sum() / total),
        "edge_density": float((edges > 0).sum() / total),
        "saturation_mean": float(hsv[:, :, 1].mean()),
        "value_mean": float(hsv[:, :, 2].mean()),
    }


def list_images(d: Path):
    return sorted([p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS])


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    for p in list_images(D3_DIR):
        r = image_stats(p, "D3_train")
        if r:
            rows.append(r)

    for p in list_images(REAL_DIR):
        r = image_stats(p, "real_01_down")
        if r:
            rows.append(r)

    if not rows:
        raise RuntimeError("분석할 이미지가 없음")

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("[DONE]", OUT_CSV)

    # domain별 평균 출력
    for domain in ["D3_train", "real_01_down"]:
        sub = [r for r in rows if r["domain"] == domain]
        print("==================================================")
        print(domain, "count:", len(sub))
        for key in ["gray_mean", "gray_std", "nonzero_ratio", "dark_ratio", "edge_density", "saturation_mean", "value_mean"]:
            vals = np.array([r[key] for r in sub], dtype=float)
            print(f"{key}: mean={vals.mean():.4f}, std={vals.std():.4f}")


if __name__ == "__main__":
    main()
