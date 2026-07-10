#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_damage_manual_poly_D3style.py

목적:
- 사용자가 직접 만든 수동 polygon 라벨 dataset을 기준으로
  D3(damage_v1) 정도의 손상/노이즈 YOLO segmentation dataset을 생성한다.

입력:
- standard_manual_poly_labeled_only/
  images/train, images/val
  labels/train, labels/val

출력:
- damage_manual_poly_D3style/
  images/train, images/val
  labels/train, labels/val
  qc/train, qc/val
  data.yaml
  damage_metrics.csv

처리 방식:
- train: 원본 + 손상 증강 생성
- val: 기본적으로 원본만 복사
- 손상 방식:
  1. label polygon -> mask
  2. erosion으로 ID 영역 약간 축소
  3. boundary bite로 가장자리 일부 제거
  4. largest component 유지
  5. 남은 면적 비율이 keep_ratio_min ~ keep_ratio_max 안에 들어오면 채택
  6. 손상된 mask 기준으로 YOLO segmentation label 재생성
  7. 이미지에서도 제거된 mask 영역을 black 처리

주의:
- 원래 D3 생성 파라미터가 완전히 복구된 것은 아니므로
  이름은 D3style로 둔다.
- 하지만 D3와 같은 핵심 원리인 mask 손상 + label 동시 갱신 방식이다.
"""

import argparse
import csv
import shutil
from pathlib import Path

import cv2
import numpy as np


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
VALID_CLASSES = [0, 1, 2, 3]


def imread_unicode(path: Path):
    """한글/일본어/특수문자 경로 대응 이미지 읽기"""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, img):
    """한글/일본어/특수문자 경로 대응 이미지 저장"""
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix.lower(), img)
    if not ok:
        raise RuntimeError(f"이미지 저장 실패: {path}")
    buf.tofile(str(path))


def list_images(img_dir: Path):
    """이미지 목록 반환"""
    if not img_dir.exists():
        return []
    return sorted([p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS])


def polygon_area(pts: np.ndarray):
    """polygon 면적 계산"""
    if pts is None or len(pts) < 3:
        return 0.0
    return float(abs(cv2.contourArea(pts.astype(np.float32))))


def read_yolo_segments(label_path: Path, width: int, height: int):
    """YOLO segmentation txt를 class별 polygon list로 읽기"""
    objects = []

    if not label_path.exists():
        return objects

    lines = label_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 7:
            continue

        try:
            cls_id = int(float(parts[0]))
            coords = np.array([float(x) for x in parts[1:]], dtype=np.float32)
        except ValueError:
            continue

        if cls_id not in VALID_CLASSES:
            continue

        if len(coords) < 6 or len(coords) % 2 != 0:
            continue

        pts = coords.reshape(-1, 2)
        pts[:, 0] *= width
        pts[:, 1] *= height
        pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)

        if polygon_area(pts) < 1.0:
            continue

        objects.append({"cls": cls_id, "pts": pts.astype(np.float32)})

    return objects


def polygon_to_mask(pts: np.ndarray, width: int, height: int):
    """polygon을 binary mask로 변환"""
    mask = np.zeros((height, width), dtype=np.uint8)
    arr = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [arr], 255)
    return mask


def keep_largest_component(mask: np.ndarray):
    """mask에서 largest connected component만 유지"""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8),
        connectivity=8,
    )

    if num_labels <= 1:
        return np.zeros_like(mask)

    areas = stats[1:, cv2.CC_STAT_AREA]
    best_idx = 1 + int(np.argmax(areas))

    return ((labels == best_idx).astype(np.uint8) * 255)


def boundary_points(mask: np.ndarray):
    """mask boundary point 추출"""
    cnts, _ = cv2.findContours(
        (mask > 0).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )

    if not cnts:
        return None

    cnt = max(cnts, key=cv2.contourArea)
    pts = cnt.reshape(-1, 2)

    if len(pts) == 0:
        return None

    return pts


def damage_mask(mask: np.ndarray, rng: np.random.Generator, args):
    """
    D3-style mask damage 생성.
    - erosion
    - boundary bite
    - largest component
    - keep ratio 범위 만족
    """
    orig = (mask > 0).astype(np.uint8) * 255
    orig_area = int((orig > 0).sum())

    if orig_area < args.min_mask_pixels:
        return None, 0.0, "orig_too_small"

    for _ in range(args.max_trials):
        work = orig.copy()

        # 1) 약한 erosion
        erode_iter = int(rng.integers(args.erode_iter_min, args.erode_iter_max + 1))
        if erode_iter > 0:
            k = int(rng.integers(args.erode_kernel_min, args.erode_kernel_max + 1))
            k = max(3, k if k % 2 == 1 else k + 1)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            work = cv2.erode(work, kernel, iterations=erode_iter)

        # 2) boundary bite
        bpts = boundary_points(work)
        if bpts is None or len(bpts) == 0:
            continue

        h, w = work.shape[:2]
        bite_count = int(rng.integers(args.bite_count_min, args.bite_count_max + 1))

        for _b in range(bite_count):
            p = bpts[int(rng.integers(0, len(bpts)))]
            px, py = int(p[0]), int(p[1])

            radius = int(rng.integers(args.bite_radius_min, args.bite_radius_max + 1))
            ox = int(rng.integers(-args.bite_offset, args.bite_offset + 1))
            oy = int(rng.integers(-args.bite_offset, args.bite_offset + 1))

            cx = int(np.clip(px + ox, 0, w - 1))
            cy = int(np.clip(py + oy, 0, h - 1))

            cv2.circle(work, (cx, cy), radius, 0, thickness=-1)

        # 3) largest component
        work = keep_largest_component(work)
        area = int((work > 0).sum())

        if area <= 0:
            continue

        ratio = area / max(orig_area, 1)

        if ratio < args.keep_ratio_min or ratio > args.keep_ratio_max:
            continue

        # 4) 너무 찢어진 경계 완화
        if args.close_kernel > 0:
            k2 = int(args.close_kernel)
            k2 = max(3, k2 if k2 % 2 == 1 else k2 + 1)
            kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k2, k2))
            work = cv2.morphologyEx(work, cv2.MORPH_CLOSE, kernel2, iterations=1)
            work = keep_largest_component(work)

            area2 = int((work > 0).sum())
            ratio2 = area2 / max(orig_area, 1)

            if ratio2 < args.keep_ratio_min or ratio2 > args.keep_ratio_max:
                continue

            return work, ratio2, "ok"

        return work, ratio, "ok"

    return None, 0.0, "failed_trials"


def mask_to_yolo_lines(cls_id: int, mask: np.ndarray, width: int, height: int, args):
    """binary mask를 YOLO segmentation line으로 변환"""
    lines = []

    mask = (mask > 0).astype(np.uint8) * 255

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8),
        connectivity=8,
    )

    if num_labels <= 1:
        return lines

    comp_ids = list(range(1, num_labels))
    comp_ids.sort(key=lambda idx: int(stats[idx, cv2.CC_STAT_AREA]), reverse=True)
    comp_ids = comp_ids[: args.max_components_per_class]

    for comp_id in comp_ids:
        area = float(stats[comp_id, cv2.CC_STAT_AREA])
        if area < args.min_contour_area:
            continue

        comp = ((labels == comp_id).astype(np.uint8) * 255)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not cnts:
            continue

        cnt = max(cnts, key=cv2.contourArea)

        if cv2.contourArea(cnt) < args.min_contour_area:
            continue

        eps = float(args.approx_eps_ratio) * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps, True)

        # point 수 제한
        loop = 0
        while len(approx) > args.max_polygon_points and loop < 20:
            eps *= 1.25
            approx = cv2.approxPolyDP(cnt, eps, True)
            loop += 1

        pts = approx.reshape(-1, 2).astype(np.float32)

        if len(pts) < 3:
            continue

        if polygon_area(pts) < args.min_contour_area:
            continue

        pts[:, 0] = np.clip(pts[:, 0] / max(width, 1), 0.0, 1.0)
        pts[:, 1] = np.clip(pts[:, 1] / max(height, 1), 0.0, 1.0)

        values = [str(int(cls_id))]
        for x, y in pts:
            values.append(f"{float(x):.6f}")
            values.append(f"{float(y):.6f}")

        lines.append(" ".join(values))

    return lines


def write_original_sample(img_path: Path, label_path: Path, out_img: Path, out_lbl: Path):
    """원본 이미지/라벨 복사"""
    out_img.parent.mkdir(parents=True, exist_ok=True)
    out_lbl.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(img_path, out_img)
    shutil.copy2(label_path, out_lbl)


def make_damaged_sample(img_path: Path, label_path: Path, out_img: Path, out_lbl: Path, rng, args):
    """이미지 1장에 대해 손상 이미지와 손상 label 생성"""
    img = imread_unicode(img_path)
    if img is None:
        return None, [{"reason": "image_read_fail"}]

    h, w = img.shape[:2]
    objects = read_yolo_segments(label_path, w, h)

    if len(objects) == 0:
        return None, [{"reason": "no_objects"}]

    out = img.copy()
    label_lines = []
    rows = []

    for obj in objects:
        cls_id = int(obj["cls"])
        orig_mask = polygon_to_mask(obj["pts"], w, h)
        damaged, ratio, status = damage_mask(orig_mask, rng, args)

        orig_area = int((orig_mask > 0).sum())

        if damaged is None:
            rows.append({
                "class": cls_id,
                "orig_area": orig_area,
                "damaged_area": 0,
                "keep_ratio": 0.0,
                "status": status,
            })
            return None, rows

        damaged_area = int((damaged > 0).sum())

        # 이미지에서도 제거된 영역 black 처리
        remove_region = (orig_mask > 0) & ~(damaged > 0)
        out[remove_region] = 0

        # 손상 mask 기준 label 재생성
        lines = mask_to_yolo_lines(cls_id, damaged, w, h, args)
        if not lines:
            rows.append({
                "class": cls_id,
                "orig_area": orig_area,
                "damaged_area": damaged_area,
                "keep_ratio": ratio,
                "status": "label_empty_after_damage",
            })
            return None, rows

        label_lines.extend(lines)

        rows.append({
            "class": cls_id,
            "orig_area": orig_area,
            "damaged_area": damaged_area,
            "keep_ratio": ratio,
            "status": "ok",
        })

    out_img.parent.mkdir(parents=True, exist_ok=True)
    out_lbl.parent.mkdir(parents=True, exist_ok=True)

    imwrite_unicode(out_img, out)
    out_lbl.write_text("\n".join(label_lines) + "\n", encoding="utf-8")

    return out, rows


def draw_qc(img_path: Path, label_path: Path, qc_path: Path):
    """생성 label 확인용 QC overlay"""
    img = imread_unicode(img_path)
    if img is None:
        return

    h, w = img.shape[:2]
    objects = read_yolo_segments(label_path, w, h)

    colors = {
        0: (0, 255, 255),
        1: (0, 200, 0),
        2: (255, 0, 0),
        3: (0, 0, 255),
    }

    vis = img.copy()

    for obj in objects:
        cls_id = int(obj["cls"])
        pts = obj["pts"].astype(np.int32).reshape(-1, 1, 2)
        color = colors.get(cls_id, (255, 255, 255))
        cv2.polylines(vis, [pts], True, color, 2)
        x, y = pts.reshape(-1, 2)[0]
        cv2.putText(
            vis,
            f"class{cls_id}",
            (int(x), max(0, int(y) - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    imwrite_unicode(qc_path, vis)


def write_data_yaml(out_root: Path):
    """YOLO data.yaml 생성"""
    text = f"""path: {out_root}
train: images/train
val: images/val

names:
  0: class0
  1: class1
  2: class2
  3: class3
"""
    (out_root / "data.yaml").write_text(text, encoding="utf-8")


def process_split(split: str, src_root: Path, out_root: Path, rng, args):
    """train/val split 처리"""
    src_img_dir = src_root / "images" / split
    src_lbl_dir = src_root / "labels" / split

    out_img_dir = out_root / "images" / split
    out_lbl_dir = out_root / "labels" / split
    qc_dir = out_root / "qc" / split

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(src_img_dir)

    made = 0
    skipped = 0
    metric_rows = []

    for img_path in images:
        label_path = src_lbl_dir / f"{img_path.stem}.txt"

        if not label_path.exists():
            skipped += 1
            metric_rows.append({
                "split": split,
                "stem": img_path.stem,
                "variant": "",
                "class": "",
                "orig_area": "",
                "damaged_area": "",
                "keep_ratio": "",
                "status": "missing_label",
            })
            continue

        # 원본 샘플 복사
        if args.include_original:
            out_img = out_img_dir / img_path.name
            out_lbl = out_lbl_dir / f"{img_path.stem}.txt"
            write_original_sample(img_path, label_path, out_img, out_lbl)
            if made < args.qc_limit:
                draw_qc(out_img, out_lbl, qc_dir / f"{img_path.stem}_orig_qc.png")
            made += 1

        # val은 기본적으로 원본만 사용
        if split == "val" and not args.damage_val:
            continue

        # 손상 증강
        aug_mult = args.aug_mult if split == "train" else args.val_aug_mult

        for k in range(int(aug_mult)):
            out_name = f"{img_path.stem}_d3style_{k:03d}{img_path.suffix}"
            out_img = out_img_dir / out_name
            out_lbl = out_lbl_dir / f"{Path(out_name).stem}.txt"

            damaged_img, rows = make_damaged_sample(
                img_path=img_path,
                label_path=label_path,
                out_img=out_img,
                out_lbl=out_lbl,
                rng=rng,
                args=args,
            )

            if damaged_img is None:
                skipped += 1
                for r in rows:
                    metric_rows.append({
                        "split": split,
                        "stem": img_path.stem,
                        "variant": k,
                        "class": r.get("class", ""),
                        "orig_area": r.get("orig_area", ""),
                        "damaged_area": r.get("damaged_area", ""),
                        "keep_ratio": r.get("keep_ratio", ""),
                        "status": r.get("status", "failed"),
                    })
                continue

            for r in rows:
                metric_rows.append({
                    "split": split,
                    "stem": img_path.stem,
                    "variant": k,
                    "class": r["class"],
                    "orig_area": r["orig_area"],
                    "damaged_area": r["damaged_area"],
                    "keep_ratio": f"{float(r['keep_ratio']):.6f}",
                    "status": r["status"],
                })

            if made < args.qc_limit:
                draw_qc(out_img, out_lbl, qc_dir / f"{Path(out_name).stem}_qc.png")

            made += 1

    return {
        "split": split,
        "src_images": len(images),
        "made": made,
        "skipped": skipped,
        "metrics": metric_rows,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--src-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--seed", type=int, default=42)

    # 생성 배율
    parser.add_argument("--aug-mult", type=int, default=9, help="train 이미지 1장당 손상 샘플 수")
    parser.add_argument("--val-aug-mult", type=int, default=0)
    parser.add_argument("--include-original", action="store_true")
    parser.add_argument("--damage-val", action="store_true")

    # D3-style 손상 정도
    parser.add_argument("--keep-ratio-min", type=float, default=0.50)
    parser.add_argument("--keep-ratio-max", type=float, default=0.82)

    parser.add_argument("--erode-iter-min", type=int, default=0)
    parser.add_argument("--erode-iter-max", type=int, default=1)
    parser.add_argument("--erode-kernel-min", type=int, default=3)
    parser.add_argument("--erode-kernel-max", type=int, default=5)

    parser.add_argument("--bite-count-min", type=int, default=2)
    parser.add_argument("--bite-count-max", type=int, default=5)
    parser.add_argument("--bite-radius-min", type=int, default=4)
    parser.add_argument("--bite-radius-max", type=int, default=14)
    parser.add_argument("--bite-offset", type=int, default=8)

    parser.add_argument("--close-kernel", type=int, default=3)
    parser.add_argument("--max-trials", type=int, default=80)
    parser.add_argument("--min-mask-pixels", type=int, default=20)

    # polygon 변환
    parser.add_argument("--min-contour-area", type=float, default=3.0)
    parser.add_argument("--approx-eps-ratio", type=float, default=0.006)
    parser.add_argument("--max-polygon-points", type=int, default=80)
    parser.add_argument("--max-components-per-class", type=int, default=1)

    # QC
    parser.add_argument("--qc-limit", type=int, default=120)

    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    src_root = Path(args.src_root)
    out_root = Path(args.out_root)

    if not src_root.exists():
        raise FileNotFoundError(f"src-root 없음: {src_root}")

    if out_root.exists():
        if args.overwrite:
            shutil.rmtree(out_root)
        else:
            raise FileExistsError(f"out-root 이미 존재함. 삭제하려면 --overwrite 사용: {out_root}")

    out_root.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    summaries = []
    all_metrics = []

    for split in ["train", "val"]:
        result = process_split(split, src_root, out_root, rng, args)
        summaries.append(result)
        all_metrics.extend(result["metrics"])

    write_data_yaml(out_root)

    metric_path = out_root / "damage_metrics.csv"
    with metric_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "split",
            "stem",
            "variant",
            "class",
            "orig_area",
            "damaged_area",
            "keep_ratio",
            "status",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_metrics)

    print("========== DONE ==========")
    print(f"SRC:  {src_root}")
    print(f"OUT:  {out_root}")
    print(f"YAML: {out_root / 'data.yaml'}")
    print(f"QC:   {out_root / 'qc'}")
    print(f"CSV:  {metric_path}")
    for s in summaries:
        print(
            f"{s['split']}: "
            f"src_images={s['src_images']}, "
            f"made={s['made']}, "
            f"skipped={s['skipped']}"
        )
    print("==========================")

if __name__ == "__main__":
    main()
