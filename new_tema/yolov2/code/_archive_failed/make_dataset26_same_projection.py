import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# make_dataset26_same_projection.py
#
# 목적:
#   실해역 이미지 생성 방식과 같은 "projection image" 형식을 유지한 상태로
#   가상 YOLO segmentation dataset을 새로 구축한다.
#
# 핵심:
#   - 이미지 자체는 기존 clean projection image를 그대로 복사한다.
#   - 노이즈, 가림, region-density 삭제/보상, intensity jitter를 절대 적용하지 않는다.
#   - labels_source의 *_top_id_uv.npy + *_meta.json을 이용해 point를 pixel 좌표로 변환한다.
#   - 기존 detect label bbox를 이용해 uv point에 class를 부여한다.
#   - class별 projected point mask에서 YOLO segmentation polygon label을 생성한다.
#
# 출력:
#   dataset26/
#     images/train, images/val
#     labels/train, labels/val
#     qc/train, qc/val
#     data.yaml
# ============================================================

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
VALID_CLASSES = [0, 1, 2, 3]


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def remove_if_exists(path: Path):
    if path.exists():
        shutil.rmtree(path)


def imread_unicode(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, img):
    ensure_dir(path.parent)
    ok, buf = cv2.imencode(path.suffix, img)
    if not ok:
        raise RuntimeError(f"이미지 저장 실패: {path}")
    buf.tofile(str(path))


def collect_images(image_dir: Path):
    if not image_dir.exists():
        raise FileNotFoundError(f"이미지 폴더 없음: {image_dir}")

    return sorted([
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ])


def read_detect_labels(label_path: Path):
    """
    기존 clean YOLO detect label 읽기.
    형식:
      class x_center y_center width height
    """
    labels = []

    if not label_path.exists():
        return labels

    for line_idx, line in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 5:
            continue

        try:
            cls_id = int(float(parts[0]))
            x = float(parts[1])
            y = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
        except Exception:
            continue

        if cls_id not in VALID_CLASSES:
            continue

        labels.append({
            "line_idx": line_idx,
            "class_id": cls_id,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "area": max(w * h, 1e-12),
            "raw": line,
        })

    return labels


def yolo_to_xyxy(label, W: int, H: int):
    x, y, w, h = label["x"], label["y"], label["w"], label["h"]

    x1 = int(round((x - w / 2.0) * W))
    y1 = int(round((y - h / 2.0) * H))
    x2 = int(round((x + w / 2.0) * W))
    y2 = int(round((y + h / 2.0) * H))

    x1 = max(0, min(W - 1, x1))
    y1 = max(0, min(H - 1, y1))
    x2 = max(0, min(W - 1, x2))
    y2 = max(0, min(H - 1, y2))

    if x2 <= x1:
        x2 = min(W - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(H - 1, y1 + 1)

    return x1, y1, x2, y2


def first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None


def find_source_files(source_dir: Path, stem: str):
    uv_candidates = [
        source_dir / f"{stem}_top_id_uv.npy",
        source_dir / f"{stem}_marker_top_id_uv.npy",
        source_dir / f"{stem}_uv.npy",
    ]

    meta_candidates = [
        source_dir / f"{stem}_meta.json",
        source_dir / f"{stem}_marker_meta.json",
    ]

    return {
        "uv": first_existing(uv_candidates),
        "meta": first_existing(meta_candidates),
    }


def uv_to_pixel(uv: np.ndarray, meta: dict, W: int, H: int):
    """
    기존 projection meta 기준으로 uv 좌표를 pixel 좌표로 변환.
    make_top_id_projection 계열 meta에 있는 pixel_size_u_m, pixel_size_v_m이 있으면 우선 사용.
    """
    uv = np.asarray(uv, dtype=np.float64)
    u = uv[:, 0]
    v = uv[:, 1]

    u_min = float(meta.get("u_min", np.min(u)))
    u_max = float(meta.get("u_max", np.max(u)))
    v_min = float(meta.get("v_min", np.min(v)))
    v_max = float(meta.get("v_max", np.max(v)))

    if "pixel_size_u_m" in meta and "pixel_size_v_m" in meta:
        ps_u = float(meta["pixel_size_u_m"])
        ps_v = float(meta["pixel_size_v_m"])

        px = np.rint((u - u_min) / max(ps_u, 1e-12)).astype(np.int32)
        py = np.rint((v_max - v) / max(ps_v, 1e-12)).astype(np.int32)
    else:
        px = np.rint((u - u_min) / max(u_max - u_min, 1e-12) * (W - 1)).astype(np.int32)
        py = np.rint((v_max - v) / max(v_max - v_min, 1e-12) * (H - 1)).astype(np.int32)

    px = np.clip(px, 0, W - 1)
    py = np.clip(py, 0, H - 1)

    return px, py


def assign_class_by_detect_bbox(px, py, labels, W: int, H: int):
    """
    class npy가 없는 경우 기존 detect bbox 안에 들어가는 uv point에 class 부여.
    겹치는 경우 면적이 작은 bbox를 우선한다.
    """
    cls = np.full(len(px), -1, dtype=np.int32)
    best_area = np.full(len(px), np.inf, dtype=np.float64)

    for label in labels:
        cid = int(label["class_id"])
        if cid not in VALID_CLASSES:
            continue

        x1, y1, x2, y2 = yolo_to_xyxy(label, W, H)
        inside = (px >= x1) & (px <= x2) & (py >= y1) & (py <= y2)

        area = float((x2 - x1 + 1) * (y2 - y1 + 1))
        update = inside & (area < best_area)

        cls[update] = cid
        best_area[update] = area

    return cls


def render_point_mask(px, py, W: int, H: int, radius: int):
    """
    projected point를 mask로 변환.
    이미지 자체는 건드리지 않고 label 생성용 mask만 만든다.
    """
    mask = np.zeros((H, W), dtype=np.uint8)

    if len(px) == 0:
        return mask

    r = max(0, int(radius))

    for x, y in zip(px, py):
        x = int(np.clip(x, 0, W - 1))
        y = int(np.clip(y, 0, H - 1))

        if r <= 0:
            mask[y, x] = 255
        else:
            cv2.circle(mask, (x, y), r, 255, -1)

    return mask


def build_class_masks(px, py, cls, W: int, H: int, point_radius: int):
    """
    noise/occlusion/삭제 없이 class별 projected point 전체를 mask로 만든다.
    """
    class_masks = {cid: np.zeros((H, W), dtype=np.uint8) for cid in VALID_CLASSES}

    for cid in VALID_CLASSES:
        idx = np.where(cls == cid)[0]
        if len(idx) == 0:
            continue

        class_masks[cid] = render_point_mask(
            px[idx],
            py[idx],
            W,
            H,
            point_radius,
        )

    return class_masks


def mask_to_polygons(mask, W: int, H: int, min_area: float, approx_eps_ratio: float,
                     max_polygon_points: int, close_kernel: int, dilate_iter: int,
                     max_components: int):
    """
    class mask를 YOLO segmentation polygon으로 변환.
    close/dilate는 label polygon 안정화용이며 이미지에는 적용하지 않는다.
    """
    work = mask.copy()

    if close_kernel > 1:
        k = int(close_kernel)
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        work = cv2.morphologyEx(work, cv2.MORPH_CLOSE, kernel)

    if dilate_iter > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        work = cv2.dilate(work, kernel, iterations=int(dilate_iter))

    contours, _ = cv2.findContours(work, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    items = []
    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area < min_area or len(cnt) < 3:
            continue

        arc = cv2.arcLength(cnt, True)
        eps = max(0.5, float(approx_eps_ratio) * arc)
        approx = cv2.approxPolyDP(cnt, eps, True)

        if len(approx) < 3:
            continue

        pts = approx.reshape(-1, 2).astype(np.float64)

        if len(pts) > max_polygon_points:
            sel = np.linspace(0, len(pts) - 1, max_polygon_points).astype(np.int32)
            pts = pts[sel]

        poly = []
        for x, y in pts:
            nx = float(np.clip(x / max(W - 1, 1), 0.0, 1.0))
            ny = float(np.clip(y / max(H - 1, 1), 0.0, 1.0))
            poly.extend([nx, ny])

        if len(poly) >= 6:
            items.append((area, poly))

    # 너무 작은 조각이 많으면 면적 큰 순서로 제한
    items = sorted(items, key=lambda x: x[0], reverse=True)[:max_components]
    return [poly for _, poly in items]


def write_seg_label(label_path: Path, class_masks, W: int, H: int, args):
    lines = []
    poly_counts = {}

    for cid in VALID_CLASSES:
        polys = mask_to_polygons(
            class_masks[cid],
            W,
            H,
            min_area=args.min_contour_area,
            approx_eps_ratio=args.approx_eps_ratio,
            max_polygon_points=args.max_polygon_points,
            close_kernel=args.label_close_kernel,
            dilate_iter=args.label_dilate_iter,
            max_components=args.max_components_per_class,
        )

        poly_counts[cid] = len(polys)

        for poly in polys:
            values = [str(cid)] + [f"{v:.6f}" for v in poly]
            lines.append(" ".join(values))

    ensure_dir(label_path.parent)
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    return len(lines), poly_counts


def draw_qc(img, class_masks, out_path: Path):
    """
    QC 이미지 저장.
    원본 projection image 위에 class mask만 색으로 overlay.
    """
    base = img.copy()
    overlay = img.copy()

    colors = {
        0: (0, 255, 255),   # square
        1: (0, 180, 0),     # rect1
        2: (255, 0, 0),     # rect2
        3: (0, 0, 255),     # rect3
    }

    for cid in VALID_CLASSES:
        mask = class_masks[cid]
        overlay[mask > 0] = colors[cid]

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < 5:
                continue
            cv2.drawContours(base, [cnt], -1, colors[cid], 1)

    qc = cv2.addWeighted(base, 0.65, overlay, 0.35, 0)
    imwrite_unicode(out_path, qc)


def write_data_yaml(out_root: Path):
    text = f"""path: {out_root.as_posix()}
train: images/train
val: images/val

names:
  0: square
  1: rect1
  2: rect2
  3: rect3
"""
    (out_root / "data.yaml").write_text(text, encoding="utf-8")


def process_split(split: str, src_dataset_root: Path, src_source_root: Path,
                  out_root: Path, args):
    image_dir = src_dataset_root / "images" / split
    detect_label_dir = src_dataset_root / "labels" / split
    source_dir = src_source_root / split

    out_img_dir = out_root / "images" / split
    out_lab_dir = out_root / "labels" / split
    out_qc_dir = out_root / "qc" / split

    ensure_dir(out_img_dir)
    ensure_dir(out_lab_dir)
    ensure_dir(out_qc_dir)

    images = collect_images(image_dir)

    stats = {
        "images": 0,
        "ok": 0,
        "missing_source": 0,
        "missing_label": 0,
        "empty_seg": 0,
        "bad_image": 0,
    }

    for img_path in images:
        stats["images"] += 1
        stem = img_path.stem

        img = imread_unicode(img_path)
        if img is None:
            stats["bad_image"] += 1
            continue

        H, W = img.shape[:2]

        src = find_source_files(source_dir, stem)
        if src["uv"] is None or src["meta"] is None:
            stats["missing_source"] += 1
            continue

        detect_label_path = detect_label_dir / f"{stem}.txt"
        detect_labels = read_detect_labels(detect_label_path)

        if len(detect_labels) == 0:
            stats["missing_label"] += 1
            continue

        uv = np.load(str(src["uv"]))
        uv = np.asarray(uv).reshape(-1, 2)

        meta = json.loads(src["meta"].read_text(encoding="utf-8"))

        px, py = uv_to_pixel(uv, meta, W, H)
        cls = assign_class_by_detect_bbox(px, py, detect_labels, W, H)

        class_masks = build_class_masks(
            px,
            py,
            cls,
            W,
            H,
            point_radius=args.point_radius,
        )

        out_label_path = out_lab_dir / f"{stem}.txt"
        n_lines, poly_counts = write_seg_label(out_label_path, class_masks, W, H, args)

        if n_lines == 0:
            stats["empty_seg"] += 1

        # 이미지 자체는 그대로 복사
        out_img_path = out_img_dir / img_path.name
        shutil.copy2(img_path, out_img_path)

        # QC 저장
        if args.qc_all or stats["ok"] < args.qc_limit:
            draw_qc(img, class_masks, out_qc_dir / f"{stem}_qc.png")

        stats["ok"] += 1

    print(f"\n===== {split} =====")
    for k, v in stats.items():
        print(f"{k}: {v}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--src_dataset_root", type=str, required=True)
    parser.add_argument("--src_source_root", type=str, required=True)
    parser.add_argument("--out_dataset_root", type=str, required=True)

    parser.add_argument("--point_radius", type=int, default=2)

    parser.add_argument("--min_contour_area", type=float, default=8.0)
    parser.add_argument("--approx_eps_ratio", type=float, default=0.006)
    parser.add_argument("--max_polygon_points", type=int, default=80)
    parser.add_argument("--max_components_per_class", type=int, default=3)

    parser.add_argument("--label_close_kernel", type=int, default=3)
    parser.add_argument("--label_dilate_iter", type=int, default=0)

    parser.add_argument("--qc_limit", type=int, default=80)
    parser.add_argument("--qc_all", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    src_dataset_root = Path(args.src_dataset_root)
    src_source_root = Path(args.src_source_root)
    out_root = Path(args.out_dataset_root)

    if out_root.exists():
        if args.overwrite:
            remove_if_exists(out_root)
        else:
            raise FileExistsError(f"이미 존재함: {out_root}  --overwrite 사용 필요")

    ensure_dir(out_root)

    print("===== make dataset26 same projection =====")
    print(f"src_dataset_root: {src_dataset_root}")
    print(f"src_source_root:  {src_source_root}")
    print(f"out_dataset_root: {out_root}")
    print("image modification: NONE")
    print("noise/occlusion/region-density: NONE")

    for split in ["train", "val"]:
        process_split(split, src_dataset_root, src_source_root, out_root, args)

    write_data_yaml(out_root)

    print("\n[OK] dataset26 생성 완료")
    print(f"yaml: {out_root / 'data.yaml'}")
    print(f"qc:   {out_root / 'qc'}")


if __name__ == "__main__":
    main()
