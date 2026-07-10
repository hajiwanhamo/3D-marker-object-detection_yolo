#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import csv
import cv2
import numpy as np

PRED_DIR = Path("/Users/hajiwan/Desktop/object_detection/new_tema/yolov2/result/yolo11/val/yolo11n_D3_realstyle_train160_val40_epoch60_01_down_conf055/01_down")
SRC_IMG_DIR = Path("/Users/hajiwan/Desktop/object_detection/new_tema/yolov2/realdata/range_sweep_down_10sets/01_down/images_color")
OUT_DIR = Path("/Users/hajiwan/Desktop/object_detection/new_tema/yolov2/result/domain_gap_analysis/class_position_check_realstyle")

IMG_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".webp"]
COLORS = {
    0: (0, 255, 255),
    1: (0, 200, 0),
    2: (255, 0, 0),
    3: (0, 0, 255),
}

def imread(p):
    data = np.fromfile(str(p), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

def imwrite(p, img):
    p.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(p.suffix.lower(), img)
    if not ok:
        raise RuntimeError(p)
    buf.tofile(str(p))

def find_image(stem):
    for ext in IMG_EXTS:
        p = SRC_IMG_DIR / f"{stem}{ext}"
        if p.exists():
            return p
    return None

def poly_center_area(norm_xy, w, h):
    pts = norm_xy.copy()
    pts[:, 0] *= w
    pts[:, 1] *= h
    pts = pts.astype(np.float32)

    area = abs(cv2.contourArea(pts))
    if area <= 0:
        return float(pts[:,0].mean()), float(pts[:,1].mean()), 0.0

    m = cv2.moments(pts)
    if abs(m["m00"]) < 1e-6:
        cx, cy = pts[:,0].mean(), pts[:,1].mean()
    else:
        cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
    return float(cx), float(cy), float(area)

def parse_txt(txt_path, w, h):
    objs = []
    for line in txt_path.read_text().splitlines():
        p = line.strip().split()
        if len(p) < 8:
            continue

        cls = int(float(p[0]))

        # save_conf=True이면 마지막 값이 confidence
        vals = [float(x) for x in p[1:]]
        if len(vals) % 2 == 1:
            conf = vals[-1]
            coords = vals[:-1]
        else:
            conf = -1.0
            coords = vals

        if len(coords) < 6:
            continue

        xy = np.array(coords, dtype=np.float32).reshape(-1, 2)
        cx, cy, area = poly_center_area(xy, w, h)

        objs.append({
            "cls": cls,
            "cx": cx,
            "cy": cy,
            "cx_norm": cx / w,
            "cy_norm": cy / h,
            "area": area,
            "area_norm": area / (w * h),
            "conf": conf,
            "points": xy,
        })
    return objs

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    summary_rows = []

    label_dir = PRED_DIR / "labels"
    txts = sorted(label_dir.glob("*.txt"))

    for txt in txts:
        stem = txt.stem
        img_path = find_image(stem)
        if img_path is None:
            print("[MISS IMG]", stem)
            continue

        img = imread(img_path)
        h, w = img.shape[:2]
        objs = parse_txt(txt, w, h)

        counts = {c: 0 for c in range(4)}
        for o in objs:
            if o["cls"] in counts:
                counts[o["cls"]] += 1

        missing = [f"class{c}" for c in range(4) if counts[c] == 0]
        duplicate = [f"class{c}" for c in range(4) if counts[c] > 1]

        summary_rows.append({
            "image": stem,
            "class0_count": counts[0],
            "class1_count": counts[1],
            "class2_count": counts[2],
            "class3_count": counts[3],
            "missing": ",".join(missing),
            "duplicate": ",".join(duplicate),
        })

        vis = img.copy()

        # 중심 위치 확인용 overlay
        for o in objs:
            cls = o["cls"]
            color = COLORS.get(cls, (255, 255, 255))
            pts = o["points"].copy()
            pts[:, 0] *= w
            pts[:, 1] *= h
            pts_i = pts.astype(np.int32).reshape(-1, 1, 2)

            cx, cy = int(o["cx"]), int(o["cy"])
            cv2.polylines(vis, [pts_i], True, color, 2)
            cv2.circle(vis, (cx, cy), 5, color, -1)
            cv2.putText(
                vis,
                f"class{cls} {o['conf']:.2f}",
                (cx + 6, cy - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

            rows.append({
                "image": stem,
                "class": cls,
                "cx": f"{o['cx']:.2f}",
                "cy": f"{o['cy']:.2f}",
                "cx_norm": f"{o['cx_norm']:.6f}",
                "cy_norm": f"{o['cy_norm']:.6f}",
                "area": f"{o['area']:.2f}",
                "area_norm": f"{o['area_norm']:.8f}",
                "conf": f"{o['conf']:.6f}",
            })

        imwrite(OUT_DIR / "overlay" / f"{stem}_position.png", vis)

    with (OUT_DIR / "pred_objects_position.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image","class","cx","cy","cx_norm","cy_norm","area","area_norm","conf"])
        writer.writeheader()
        writer.writerows(rows)

    with (OUT_DIR / "pred_image_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image","class0_count","class1_count","class2_count","class3_count","missing","duplicate"])
        writer.writeheader()
        writer.writerows(summary_rows)

    print("[DONE]", OUT_DIR)
    print("objects csv:", OUT_DIR / "pred_objects_position.csv")
    print("summary csv:", OUT_DIR / "pred_image_summary.csv")
    print("overlay:", OUT_DIR / "overlay")

    print("\n[SUMMARY]")
    for r in summary_rows:
        print(r["image"], "missing=", r["missing"], "duplicate=", r["duplicate"])

if __name__ == "__main__":
    main()
