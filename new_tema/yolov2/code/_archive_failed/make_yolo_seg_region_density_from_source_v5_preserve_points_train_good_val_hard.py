import argparse
import csv
import json
import random
import shutil
import copy
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# make_yolo_seg_region_density_from_source.py
#
# 목적:
#   기존 clean YOLO dataset + labels_source(3D->2D 투영 source)를 사용해서
#   region-density 노이즈 이미지와 YOLO segmentation label을 동시에 생성한다.
#
# 핵심:
#   - 이미지 threshold / detect bbox contour 변환 사용하지 않음
#   - source uv/meta + class 정보 또는 원본 detect label 기반으로 class별 point를 구성
#   - 노이즈 적용 후 남은 class별 point mask로 segmentation polygon 생성
#
# 입력:
#   src_dataset_root/
#     images/train, images/val
#     labels/train, labels/val      # 기존 4-class detect label
#   src_labels_source_root/
#     train, val
#       *_top_id_uv.npy
#       *_meta.json
#       optional: *_class.npy / *_rule_id.npy / *_id.npy 등
#
# 출력:
#   out_dataset_root/
#     images/train, images/val      # 노이즈 이미지
#     labels/train, labels/val      # YOLO segmentation polygon label
#     data.yaml
#   out_source_root/
#     train, val
#       *_seg_source.npz            # 이후 검증용 source 기록
# ============================================================

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
VALID_CLASSES = [0, 1, 2, 3]


# ============================================================
# 기본 IO
# ============================================================
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def remove_if_exists(path: Path):
    if path.exists():
        shutil.rmtree(str(path))


def imread_unicode(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, img):
    ensure_dir(path.parent)
    ok, buf = cv2.imencode(path.suffix, img)
    if not ok:
        raise RuntimeError(f"이미지 인코딩 실패: {path}")
    buf.tofile(str(path))


def collect_images(image_dir: Path):
    if not image_dir.exists():
        raise FileNotFoundError(f"이미지 폴더 없음: {image_dir}")
    return sorted([p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


# ============================================================
# YOLO detect label 읽기
# ============================================================
def read_detect_labels(label_path: Path):
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

        cls_id = int(float(parts[0]))
        x = float(parts[1])
        y = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])

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


# ============================================================
# labels_source 파일 찾기
# ============================================================
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

    # class array는 프로젝트 코드마다 이름이 다를 수 있어 여러 후보를 지원한다.
    class_candidates = [
        source_dir / f"{stem}_top_id_class.npy",
        source_dir / f"{stem}_top_id_label.npy",
        source_dir / f"{stem}_top_id_rule_id.npy",
        source_dir / f"{stem}_class.npy",
        source_dir / f"{stem}_label.npy",
        source_dir / f"{stem}_rule_id.npy",
        source_dir / f"{stem}_id.npy",
        source_dir / f"{stem}_marker_top_id_class.npy",
        source_dir / f"{stem}_marker_top_id_label.npy",
        source_dir / f"{stem}_marker_top_id_rule_id.npy",
        source_dir / f"{stem}_marker_class.npy",
        source_dir / f"{stem}_marker_rule_id.npy",
        source_dir / f"{stem}_marker_id.npy",
    ]

    return {
        "uv": first_existing(uv_candidates),
        "meta": first_existing(meta_candidates),
        "class": first_existing(class_candidates),
    }


# ============================================================
# uv -> pixel 변환
# ============================================================
def uv_to_pixel(uv: np.ndarray, meta: dict, W: int, H: int):
    uv = np.asarray(uv, dtype=np.float64)
    u = uv[:, 0]
    v = uv[:, 1]

    u_min = float(meta.get("u_min", np.min(u)))
    u_max = float(meta.get("u_max", np.max(u)))
    v_min = float(meta.get("v_min", np.min(v)))
    v_max = float(meta.get("v_max", np.max(v)))

    # make_top_id_projection 계열 meta에 들어있는 공식 변환식 우선 사용
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


# ============================================================
# class array 로드 및 remap
# ============================================================
def load_class_array(class_path: Path | None, n_points: int):
    if class_path is None:
        return None, "none"

    arr = np.load(str(class_path))
    arr = np.asarray(arr).reshape(-1)

    if len(arr) != n_points:
        return None, f"length_mismatch:{class_path.name}:{len(arr)}!={n_points}"

    arr = arr.astype(np.int32)
    valid = arr[arr >= 0]
    uniq = sorted(set(valid.tolist()))

    # 이미 0~3이면 그대로 사용
    if set(uniq).issubset({0, 1, 2, 3}):
        return arr, f"npy:{class_path.name}:0to3"

    # 1~4 라벨이면 0~3으로 변환
    if set(uniq).issubset({1, 2, 3, 4}):
        out = arr.copy()
        m = out >= 1
        out[m] = out[m] - 1
        return out, f"npy:{class_path.name}:1to4_remap_to_0to3"

    # 0~4가 섞인 경우는 프로젝트마다 의미가 다를 수 있으므로 그대로 단정하지 않음
    return None, f"unsupported_values:{class_path.name}:{uniq}"


# ============================================================
# class npy가 없을 때: 원본 detect label bbox로 uv point에 class 부여
# 이미지 threshold가 아니라 원본 라벨 bbox와 투영 point만 사용한다.
# ============================================================
def assign_class_by_detect_bbox(px, py, labels, W: int, H: int):
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


# ============================================================
# region density noise field
# ============================================================
def make_region_field(h: int, w: int, rng: random.Random, sigma: float,
                      grid_min: int, grid_max: int, blur_ratio: float,
                      local_min: float, local_max: float):
    if h <= 1 or w <= 1:
        return np.ones((h, w), dtype=np.float32)

    grid_h = rng.randint(grid_min, grid_max)
    grid_w = rng.randint(grid_min, grid_max)

    np_rng = np.random.default_rng(rng.randint(0, 2**31 - 1))
    raw = np_rng.normal(loc=0.0, scale=sigma, size=(grid_h, grid_w)).astype(np.float32)
    field = np.exp(raw).astype(np.float32)
    field = cv2.resize(field, (w, h), interpolation=cv2.INTER_CUBIC)

    blur_k = max(3, int(round(min(h, w) * blur_ratio)))
    if blur_k % 2 == 0:
        blur_k += 1
    blur_k = min(blur_k, 31)

    if blur_k >= 3:
        field = cv2.GaussianBlur(field, (blur_k, blur_k), 0)

    mean_val = float(np.mean(field))
    if mean_val > 1e-8:
        field = field / mean_val

    field = np.clip(field, local_min, local_max)
    return field.astype(np.float32)


def make_square_keep_prob(h: int, w: int, rng: random.Random, args):
    if h <= 1 or w <= 1:
        return np.ones((h, w), dtype=np.float32)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    cx = (w - 1) * (0.5 + rng.uniform(-args.square_center_jitter, args.square_center_jitter))
    cy = (h - 1) * (0.5 + rng.uniform(-args.square_center_jitter, args.square_center_jitter))

    sx = max(1.0, w * rng.uniform(args.square_center_sigma_min, args.square_center_sigma_max))
    sy = max(1.0, h * rng.uniform(args.square_center_sigma_min, args.square_center_sigma_max))

    gaussian = np.exp(-(((xx - cx) ** 2) / (2 * sx * sx) + ((yy - cy) ** 2) / (2 * sy * sy))).astype(np.float32)

    center_keep = rng.uniform(args.square_center_keep_min, args.square_center_keep_max)
    outer_keep = rng.uniform(args.square_outer_keep_min, args.square_outer_keep_max)

    field = outer_keep + (center_keep - outer_keep) * gaussian

    local = make_region_field(
        h=h,
        w=w,
        rng=rng,
        sigma=args.square_region_sigma,
        grid_min=args.square_region_grid_min,
        grid_max=args.square_region_grid_max,
        blur_ratio=args.region_blur_ratio,
        local_min=args.local_density_min,
        local_max=args.local_density_max,
    )

    field = field * local
    return np.clip(field, args.keep_prob_min, args.keep_prob_max).astype(np.float32)


def make_rect_keep_prob(h: int, w: int, rng: random.Random, args):
    if h <= 1 or w <= 1:
        return np.ones((h, w), dtype=np.float32)

    base_keep = rng.uniform(args.rect_keep_min, args.rect_keep_max)

    local = make_region_field(
        h=h,
        w=w,
        rng=rng,
        sigma=args.rect_region_sigma,
        grid_min=args.rect_region_grid_min,
        grid_max=args.rect_region_grid_max,
        blur_ratio=args.region_blur_ratio,
        local_min=args.local_density_min,
        local_max=args.local_density_max,
    )

    field = base_keep * local

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    patch_count = rng.randint(args.rect_low_patch_min, args.rect_low_patch_max)

    for _ in range(patch_count):
        pcx = rng.uniform(0.15, 0.85) * max(1, w - 1)
        pcy = rng.uniform(0.15, 0.85) * max(1, h - 1)

        psx = max(1.0, w * rng.uniform(args.rect_patch_sigma_min, args.rect_patch_sigma_max))
        psy = max(1.0, h * rng.uniform(args.rect_patch_sigma_min, args.rect_patch_sigma_max))

        patch = np.exp(-(((xx - pcx) ** 2) / (2 * psx * psx) + ((yy - pcy) ** 2) / (2 * psy * psy))).astype(np.float32)
        reduce_strength = rng.uniform(args.rect_patch_reduce_min, args.rect_patch_reduce_max)
        field = field * (1.0 - reduce_strength * patch)

    return np.clip(field, args.keep_prob_min, args.keep_prob_max).astype(np.float32)


def keep_points_for_class(px_c, py_c, cls_id: int, rng: random.Random, args):
    if len(px_c) == 0:
        return np.zeros(0, dtype=bool), 0.0

    x1 = int(np.min(px_c))
    x2 = int(np.max(px_c))
    y1 = int(np.min(py_c))
    y2 = int(np.max(py_c))

    w = max(1, x2 - x1 + 1)
    h = max(1, y2 - y1 + 1)

    if cls_id == 0:
        field = make_square_keep_prob(h, w, rng, args)
    else:
        field = make_rect_keep_prob(h, w, rng, args)

    lx = np.clip(px_c - x1, 0, w - 1)
    ly = np.clip(py_c - y1, 0, h - 1)
    prob = field[ly, lx]

    np_rng = np.random.default_rng(rng.randint(0, 2**31 - 1))
    keep = np_rng.random(len(prob)).astype(np.float32) < prob

    return keep, float(np.mean(prob))


# ============================================================
# point mask 렌더링
# ============================================================
def render_point_mask(px, py, W: int, H: int, radius: int):
    mask = np.zeros((H, W), dtype=np.uint8)
    if len(px) == 0:
        return mask

    r = max(0, int(radius))
    for x, y in zip(px, py):
        if r <= 0:
            mask[int(y), int(x)] = 255
        else:
            cv2.circle(mask, (int(x), int(y)), r, 255, -1)
    return mask




def sample_compensation_points(px_src, py_src, keep_mask, add_count: int, rng: random.Random, args):
    """
    누락 영역은 비워두되, 전체 visible point 수가 너무 줄지 않도록
    살아남은 포인트 주변에 보상 포인트를 추가한다.
    - 삭제된 위치를 되살리는 것이 아님
    - 살아남은 영역 안쪽으로 포인트를 중복/지터링하여 밀도만 보정
    """
    if add_count <= 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)

    kept_indices = np.where(keep_mask)[0]
    if len(kept_indices) == 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)

    np_rng = np.random.default_rng(rng.randint(0, 2**31 - 1))
    src_indices = np_rng.choice(kept_indices, size=add_count, replace=True)

    jitter = int(max(0, args.compensate_jitter_px))
    if jitter > 0:
        dx = np_rng.integers(-jitter, jitter + 1, size=add_count, endpoint=False)
        dy = np_rng.integers(-jitter, jitter + 1, size=add_count, endpoint=False)
    else:
        dx = np.zeros(add_count, dtype=np.int32)
        dy = np.zeros(add_count, dtype=np.int32)

    add_x = px_src[src_indices].astype(np.int32) + dx.astype(np.int32)
    add_y = py_src[src_indices].astype(np.int32) + dy.astype(np.int32)

    return add_x, add_y, src_indices.astype(np.int32)


def draw_compensation_points(out, img_original, px_target, py_target, px_source, py_source, source_indices, radius: int):
    """
    보상 포인트를 원본 이미지의 색상으로 다시 그린다.
    원본 threshold 기반이 아니라 source point 좌표와 원본 픽셀 색만 사용한다.
    """
    H, W = out.shape[:2]
    r = max(0, int(radius))

    for tx, ty, si in zip(px_target, py_target, source_indices):
        tx = int(np.clip(tx, 0, W - 1))
        ty = int(np.clip(ty, 0, H - 1))
        sx = int(np.clip(px_source[si], 0, W - 1))
        sy = int(np.clip(py_source[si], 0, H - 1))
        color = img_original[sy, sx].tolist()

        if r <= 0:
            out[ty, tx] = color
        else:
            cv2.circle(out, (tx, ty), r, color, -1)

def build_noisy_image_and_masks(img, px, py, cls, rng: random.Random, args):
    H, W = img.shape[:2]
    out = img.copy()

    valid = np.isin(cls, VALID_CLASSES)
    px = px[valid].astype(np.int32)
    py = py[valid].astype(np.int32)
    cls = cls[valid].astype(np.int32)

    keep_all = np.zeros(len(px), dtype=bool)
    class_masks = {cid: np.zeros((H, W), dtype=np.uint8) for cid in VALID_CLASSES}
    stats = {}

    # 보상 포인트 기록용
    comp_px_all = []
    comp_py_all = []
    comp_cls_all = []
    comp_src_all = []

    for cid in VALID_CLASSES:
        idx = np.where(cls == cid)[0]
        if len(idx) == 0:
            stats[cid] = {
                "points": 0,
                "kept": 0,
                "compensated": 0,
                "final_points": 0,
                "keep_ratio": 0.0,
                "final_ratio": 0.0,
                "mean_prob": 0.0,
            }
            continue

        keep_c, mean_prob = keep_points_for_class(px[idx], py[idx], cid, rng, args)
        keep_all[idx] = keep_c

        kept_count = int(np.count_nonzero(keep_c))
        original_count = int(len(idx))

        # 삭제는 유지하되, 전체 point 수가 너무 줄지 않도록 살아남은 위치 주변에 보상 포인트를 추가한다.
        compensate_count = 0
        if args.preserve_point_count:
            missing_count = max(0, original_count - kept_count)
            compensate_count = int(round(missing_count * float(args.compensate_ratio)))

            add_x, add_y, local_src = sample_compensation_points(
                px_src=px[idx],
                py_src=py[idx],
                keep_mask=keep_c,
                add_count=compensate_count,
                rng=rng,
                args=args,
            )

            if len(add_x) > 0:
                add_x = np.clip(add_x, 0, W - 1).astype(np.int32)
                add_y = np.clip(add_y, 0, H - 1).astype(np.int32)
                global_src = idx[local_src]

                comp_px_all.append(add_x)
                comp_py_all.append(add_y)
                comp_cls_all.append(np.full(len(add_x), cid, dtype=np.int32))
                comp_src_all.append(global_src.astype(np.int32))
                compensate_count = int(len(add_x))
            else:
                compensate_count = 0

        # class mask는 최종 visible point 기준으로 생성한다.
        kept_idx = idx[keep_c]
        px_final = px[kept_idx]
        py_final = py[kept_idx]

        if comp_px_all and compensate_count > 0:
            # 방금 class에서 추가된 보상 포인트만 붙인다.
            px_final = np.concatenate([px_final, comp_px_all[-1]])
            py_final = np.concatenate([py_final, comp_py_all[-1]])

        class_masks[cid] = render_point_mask(px_final, py_final, W, H, args.point_radius)

        final_count = int(len(px_final))
        stats[cid] = {
            "points": original_count,
            "kept": kept_count,
            "compensated": compensate_count,
            "final_points": final_count,
            "keep_ratio": float(kept_count / max(original_count, 1)),
            "final_ratio": float(final_count / max(original_count, 1)),
            "mean_prob": mean_prob,
        }

    # 제거된 원본 포인트 영역은 먼저 검정색으로 지운다.
    remove_mask = render_point_mask(px[~keep_all], py[~keep_all], W, H, args.point_radius)
    out[remove_mask > 0] = (0, 0, 0)

    # 그 다음 보상 포인트를 살아남은 영역 주변에 다시 그린다.
    if args.preserve_point_count and len(comp_px_all) > 0:
        comp_px = np.concatenate(comp_px_all).astype(np.int32)
        comp_py = np.concatenate(comp_py_all).astype(np.int32)
        comp_src = np.concatenate(comp_src_all).astype(np.int32)
        draw_compensation_points(
            out=out,
            img_original=img,
            px_target=comp_px,
            py_target=comp_py,
            px_source=px,
            py_source=py,
            source_indices=comp_src,
            radius=args.point_radius,
        )

    # 남은 포인트에 intensity jitter 선택 적용
    if args.intensity_jitter:
        # 보상 포인트까지 포함된 최종 mask 기준으로 intensity 조정
        final_visible_mask = np.zeros((H, W), dtype=np.uint8)
        for cid in VALID_CLASSES:
            final_visible_mask[class_masks[cid] > 0] = 255
        if np.count_nonzero(final_visible_mask) > 0:
            factor = rng.uniform(args.intensity_min, args.intensity_max)
            tmp = out.astype(np.float32)
            tmp[final_visible_mask > 0] *= factor
            out = np.clip(tmp, 0, 255).astype(np.uint8)

    return out, class_masks, px, py, cls, keep_all, stats


# ============================================================
# YOLO segmentation polygon 생성
# ============================================================
def mask_to_polygons(mask, W: int, H: int, args):
    work = mask.copy()

    if args.label_close_kernel > 1:
        k = int(args.label_close_kernel)
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        work = cv2.morphologyEx(work, cv2.MORPH_CLOSE, kernel)

    if args.label_dilate_iter > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        work = cv2.dilate(work, kernel, iterations=int(args.label_dilate_iter))

    contours, _ = cv2.findContours(work, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polys = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < args.min_contour_area or len(cnt) < 3:
            continue

        arc = cv2.arcLength(cnt, True)
        eps = max(0.5, float(args.approx_eps_ratio) * arc)
        approx = cv2.approxPolyDP(cnt, eps, True)

        if len(approx) < 3:
            continue

        pts = approx.reshape(-1, 2).astype(np.float64)

        if len(pts) > args.max_polygon_points:
            sel = np.linspace(0, len(pts) - 1, args.max_polygon_points).astype(np.int32)
            pts = pts[sel]

        poly = []
        for x, y in pts:
            nx = float(np.clip(x / max(W - 1, 1), 0.0, 1.0))
            ny = float(np.clip(y / max(H - 1, 1), 0.0, 1.0))
            poly.extend([nx, ny])

        if len(poly) >= 6:
            polys.append(poly)

    # 너무 많은 작은 조각이 생기면 면적 큰 순서로 제한한다.
    if len(polys) > args.max_components_per_class:
        def poly_area(poly):
            pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
            pts[:, 0] *= W
            pts[:, 1] *= H
            return abs(cv2.contourArea(pts.astype(np.float32)))
        polys = sorted(polys, key=poly_area, reverse=True)[:args.max_components_per_class]

    return polys


def write_seg_label(label_path: Path, class_masks, W: int, H: int, args):
    lines = []
    poly_counts = {}

    for cid in VALID_CLASSES:
        polys = mask_to_polygons(class_masks[cid], W, H, args)
        poly_counts[cid] = len(polys)
        for poly in polys:
            values = [str(cid)] + [f"{v:.6f}" for v in poly]
            lines.append(" ".join(values))

    ensure_dir(label_path.parent)
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines), poly_counts


# ============================================================
# 확인 이미지
# ============================================================
def draw_seg_check(img, class_masks, out_path: Path):
    base = img.copy()
    overlay = img.copy()

    colors = {
        0: (0, 255, 255),   # square: yellow
        1: (0, 180, 0),     # rect1: green
        2: (255, 0, 0),     # rect2: blue
        3: (0, 0, 255),     # rect3: red
    }

    names = {
        0: "class0_square",
        1: "class1_rect",
        2: "class2_rect",
        3: "class3_rect",
    }

    for cid in VALID_CLASSES:
        mask = class_masks[cid]
        color = colors[cid]
        overlay[mask > 0] = color

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            continue

        all_pts = np.vstack([c.reshape(-1, 2) for c in contours if len(c) >= 1])
        if len(all_pts) == 0:
            continue

        cx = int(np.mean(all_pts[:, 0]))
        cy = int(np.mean(all_pts[:, 1]))

        cv2.putText(
            base,
            names[cid],
            (cx + 4, cy - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

        cv2.drawContours(base, contours, -1, color, 1)

    vis = cv2.addWeighted(overlay, 0.45, base, 0.55, 0)
    imwrite_unicode(out_path, vis)


# ============================================================
# data.yaml
# ============================================================
def write_data_yaml(out_root: Path):
    text = f"""path: {out_root.resolve().as_posix()}
train: images/train
val: images/val

names:
  0: square
  1: rect1
  2: rect2
  3: rect3
"""
    (out_root / "data.yaml").write_text(text, encoding="utf-8")


# ============================================================
# main processing
# ============================================================
def process_one_image(img_path: Path, label_path: Path, source_dir: Path, out_stem: str, split: str, idx: int, variant_idx: int, args):
    img = imread_unicode(img_path)
    if img is None:
        raise RuntimeError(f"이미지 읽기 실패: {img_path}")

    H, W = img.shape[:2]

    labels = read_detect_labels(label_path)
    if len(labels) == 0:
        raise RuntimeError(f"detect label 없음: {label_path}")

    src = find_source_files(source_dir, img_path.stem)
    if src["uv"] is None:
        raise RuntimeError(f"top_id_uv.npy 없음: {source_dir} / stem={img_path.stem}")
    if src["meta"] is None:
        raise RuntimeError(f"meta.json 없음: {source_dir} / stem={img_path.stem}")

    uv = np.load(str(src["uv"])).astype(np.float64)
    if uv.ndim != 2 or uv.shape[1] < 2:
        raise RuntimeError(f"uv 형식 오류: {src['uv']} shape={uv.shape}")
    uv = uv[:, :2]

    meta = json.loads(src["meta"].read_text(encoding="utf-8"))
    px, py = uv_to_pixel(uv, meta, W, H)

    cls, cls_source = load_class_array(src["class"], len(uv))

    if cls is None:
        if args.require_class_npy:
            raise RuntimeError(f"class npy 사용 실패: {cls_source}")
        cls = assign_class_by_detect_bbox(px, py, labels, W, H)
        cls_source = f"detect_bbox_fallback_after:{cls_source}"

    rng_seed = int(args.seed + idx * 1009 + variant_idx * 9176 + (0 if split == "train" else 100000))
    rng = random.Random(rng_seed)

    noisy_img, class_masks, px_used, py_used, cls_used, keep_all, stats = build_noisy_image_and_masks(img, px, py, cls, rng, args)

    return {
        "img": img,
        "noisy_img": noisy_img,
        "class_masks": class_masks,
        "px": px_used,
        "py": py_used,
        "cls": cls_used,
        "keep": keep_all,
        "uv": uv,
        "stats": stats,
        "cls_source": cls_source,
        "source_files": src,
        "seed": rng_seed,
        "W": W,
        "H": H,
    }


def process_split(split: str, args, summary_rows):
    src_dataset_root = Path(args.src_dataset_root)
    src_labels_source_root = Path(args.src_labels_source_root)
    out_dataset_root = Path(args.out_dataset_root)
    out_source_root = Path(args.out_source_root) if args.out_source_root else None
    check_root = Path(args.check_dir)

    src_img_dir = src_dataset_root / "images" / split
    src_lbl_dir = src_dataset_root / "labels" / split
    src_source_dir = src_labels_source_root / split

    out_img_dir = out_dataset_root / "images" / split
    out_lbl_dir = out_dataset_root / "labels" / split
    out_source_dir = out_source_root / split if out_source_root else None
    check_dir = check_root / split

    ensure_dir(out_img_dir)
    ensure_dir(out_lbl_dir)
    ensure_dir(check_dir)
    if out_source_dir is not None:
        ensure_dir(out_source_dir)

    images = collect_images(src_img_dir)

    created = 0
    failed = 0
    check_saved = 0

    print(f"\n========== {split.upper()} ==========")
    print(f"images: {len(images)}")

    for idx, img_path in enumerate(images):
        stem = img_path.stem
        label_path = src_lbl_dir / f"{stem}.txt"

        try:
            for variant_idx in range(args.variants_per_image):
                if args.keep_original_name and args.variants_per_image == 1:
                    out_stem = stem
                else:
                    out_stem = f"{stem}_segdens_{variant_idx + 1:02d}"

                result = process_one_image(
                    img_path=img_path,
                    label_path=label_path,
                    source_dir=src_source_dir,
                    out_stem=out_stem,
                    split=split,
                    idx=idx,
                    variant_idx=variant_idx,
                    args=args,
                )

                out_img_path = out_img_dir / f"{out_stem}.png"
                out_lbl_path = out_lbl_dir / f"{out_stem}.txt"

                imwrite_unicode(out_img_path, result["noisy_img"])
                obj_count, poly_counts = write_seg_label(out_lbl_path, result["class_masks"], result["W"], result["H"], args)

                if obj_count == 0:
                    raise RuntimeError("segmentation polygon 0개 생성")

                if out_source_dir is not None:
                    npz_path = out_source_dir / f"{out_stem}_seg_source.npz"
                    np.savez_compressed(
                        str(npz_path),
                        px=result["px"].astype(np.int32),
                        py=result["py"].astype(np.int32),
                        cls=result["cls"].astype(np.int32),
                        keep=result["keep"].astype(np.bool_),
                        seed=np.array([result["seed"]], dtype=np.int64),
                    )

                    # 원본 meta/uv도 추적용으로 복사한다.
                    src_uv = result["source_files"]["uv"]
                    src_meta = result["source_files"]["meta"]
                    if src_uv is not None:
                        shutil.copy2(str(src_uv), str(out_source_dir / f"{out_stem}_top_id_uv.npy"))
                    if src_meta is not None:
                        shutil.copy2(str(src_meta), str(out_source_dir / f"{out_stem}_meta.json"))

                if check_saved < args.check_max_per_split:
                    check_path = check_dir / f"{out_stem}_seg_check.jpg"
                    draw_seg_check(result["noisy_img"], result["class_masks"], check_path)
                    check_saved += 1

                row = {
                    "split": split,
                    "src_stem": stem,
                    "out_stem": out_stem,
                    "status": "ok",
                    "class_source": result["cls_source"],
                    "seg_objects": obj_count,
                    "poly_c0": poly_counts.get(0, 0),
                    "poly_c1": poly_counts.get(1, 0),
                    "poly_c2": poly_counts.get(2, 0),
                    "poly_c3": poly_counts.get(3, 0),
                }

                for cid in VALID_CLASSES:
                    st = result["stats"].get(cid, {})
                    row[f"c{cid}_points"] = st.get("points", 0)
                    row[f"c{cid}_kept"] = st.get("kept", 0)
                    row[f"c{cid}_keep_ratio"] = st.get("keep_ratio", 0.0)
                    row[f"c{cid}_mean_prob"] = st.get("mean_prob", 0.0)

                summary_rows.append(row)
                created += 1

            print(f"[OK] {split} {idx + 1}/{len(images)} {stem}")

        except Exception as e:
            failed += 1
            summary_rows.append({
                "split": split,
                "src_stem": stem,
                "out_stem": "",
                "status": f"fail: {e}",
            })
            print(f"[FAIL] {split} {idx + 1}/{len(images)} {stem}: {e}")

    print(f"[{split}] created={created}, failed={failed}, check_saved={check_saved}")
    return created, failed



# ============================================================
# train/val 재분할 + split별 증강 배율 처리
# ============================================================
def collect_split_items(src_dataset_root: Path, src_labels_source_root: Path, split: str):
    """
    기존 src_dataset_root/images/{split}, labels/{split}, labels_source/{split}를 묶어서
    하나의 source item 목록으로 만든다.
    이후 train_ratio 옵션을 쓰면 기존 train/val을 합친 뒤 새로 섞어서 다시 나눈다.
    """
    src_img_dir = src_dataset_root / "images" / split
    src_lbl_dir = src_dataset_root / "labels" / split
    src_source_dir = src_labels_source_root / split

    images = collect_images(src_img_dir)
    items = []
    for img_path in images:
        stem = img_path.stem
        items.append({
            "source_split": split,
            "stem": stem,
            "img_path": img_path,
            "label_path": src_lbl_dir / f"{stem}.txt",
            "source_dir": src_source_dir,
        })
    return items


def make_resplit_items(args):
    """
    기존 train/val 전체를 합쳐서 seed 기준으로 섞고, args.train_ratio에 따라 새 train/val로 나눈다.
    labels_source는 각 원본 split의 폴더를 그대로 참조하므로 source 추적이 깨지지 않는다.
    """
    src_dataset_root = Path(args.src_dataset_root)
    src_labels_source_root = Path(args.src_labels_source_root)

    items = []
    for source_split in ["train", "val"]:
        items.extend(collect_split_items(src_dataset_root, src_labels_source_root, source_split))

    if len(items) == 0:
        raise RuntimeError("재분할할 source image가 없습니다.")

    rng = random.Random(args.seed)
    rng.shuffle(items)

    ratio = float(args.train_ratio)
    if not (0.0 < ratio < 1.0):
        raise ValueError(f"--train_ratio는 0~1 사이여야 합니다: {ratio}")

    n_total = len(items)
    n_train = int(round(n_total * ratio))
    n_train = max(1, min(n_total - 1, n_train))

    train_items = items[:n_train]
    val_items = items[n_train:]

    return train_items, val_items




# ============================================================
# train=학습용 안정 데이터 / val=어려운 검증 데이터 분리 지원
# ============================================================
def parse_class_list(text: str):
    """쉼표로 입력된 class id 문자열을 int list로 변환한다. 예: "1,2,3"."""
    out = []
    if text is None:
        return out
    for t in str(text).split(','):
        t = t.strip()
        if not t:
            continue
        try:
            v = int(t)
            if v in VALID_CLASSES:
                out.append(v)
        except ValueError:
            pass
    return sorted(set(out))


def analyze_source_item_for_split(item, args):
    """
    원본 source 하나의 class별 point 수를 계산한다.
    목적:
      - train에는 class 구성이 안정적인 source를 우선 배치
      - val에는 point 수가 적거나 특정 rect class가 약한 hard source를 우선 배치
    주의:
      - 이 함수는 이미지를 변형하지 않고 원본 uv/class 정보만 읽는다.
      - class npy가 없으면 기존 코드와 동일하게 detect bbox fallback을 사용한다.
    """
    img_path = Path(item['img_path'])
    label_path = Path(item['label_path'])
    source_dir = Path(item['source_dir'])

    img = imread_unicode(img_path)
    if img is None:
        raise RuntimeError(f"이미지 읽기 실패: {img_path}")
    H, W = img.shape[:2]

    labels = read_detect_labels(label_path)
    src = find_source_files(source_dir, img_path.stem)
    if src['uv'] is None or src['meta'] is None:
        raise RuntimeError(f"source 파일 부족: {source_dir} / {img_path.stem}")

    uv = np.load(str(src['uv'])).astype(np.float64)
    uv = uv[:, :2]
    meta = json.loads(src['meta'].read_text(encoding='utf-8'))
    px, py = uv_to_pixel(uv, meta, W, H)

    cls, cls_source = load_class_array(src['class'], len(uv))
    if cls is None:
        if args.require_class_npy:
            raise RuntimeError(f"class npy 사용 실패: {cls_source}")
        cls = assign_class_by_detect_bbox(px, py, labels, W, H)
        cls_source = f"detect_bbox_fallback_after:{cls_source}"

    counts = {cid: int(np.count_nonzero(cls == cid)) for cid in VALID_CLASSES}
    total = int(sum(counts.values()))
    vals = np.array([counts[cid] for cid in VALID_CLASSES], dtype=np.float64)
    valid_vals = vals[vals > 0]

    missing = [cid for cid in VALID_CLASSES if counts[cid] <= 0]
    min_count = int(np.min(valid_vals)) if len(valid_vals) else 0
    max_count = int(np.max(valid_vals)) if len(valid_vals) else 0
    balance = float(min_count / max(max_count, 1))

    target_classes = parse_class_list(args.hard_val_target_classes)
    if not target_classes:
        target_classes = VALID_CLASSES
    target_counts = np.array([counts[cid] for cid in target_classes], dtype=np.float64)
    target_min = int(np.min(target_counts)) if len(target_counts) else min_count
    target_mean = float(np.mean(target_counts)) if len(target_counts) else 0.0

    # hard 점수: 특정 class point가 적고, class 간 불균형이 크고, 총 point가 적을수록 높다.
    # 절대값 스케일 차이를 줄이기 위해 비율 중심으로 계산한다.
    target_weak = 1.0 - float(target_min / max(target_mean, 1.0))
    imbalance = 1.0 - balance
    low_total = 1.0 / max(np.log10(total + 10.0), 1.0)
    missing_penalty = 1.5 * len(missing)
    hard_score = float(2.0 * target_weak + 1.0 * imbalance + 0.25 * low_total + missing_penalty)

    return {
        'stem': item['stem'],
        'source_split': item['source_split'],
        'counts': counts,
        'total_points': total,
        'min_count': min_count,
        'max_count': max_count,
        'balance': balance,
        'target_min': target_min,
        'target_mean': target_mean,
        'missing': missing,
        'hard_score': hard_score,
        'class_source': cls_source,
    }


def make_resplit_items_val_hard(args):
    """
    기존 train/val 전체 source를 합친 뒤 hard score 기준으로 분리한다.
    - val: hard_score가 높은 데이터, 즉 노이즈/손상/특정 class 취약성이 큰 데이터
    - train: 나머지 상대적으로 안정적인 데이터
    """
    src_dataset_root = Path(args.src_dataset_root)
    src_labels_source_root = Path(args.src_labels_source_root)

    items = []
    for source_split in ['train', 'val']:
        items.extend(collect_split_items(src_dataset_root, src_labels_source_root, source_split))

    if len(items) == 0:
        raise RuntimeError('재분할할 source image가 없습니다.')

    rng = random.Random(args.seed)
    rng.shuffle(items)  # 동점일 때 순서 고정

    diagnostics = []
    usable = []
    rejected = []
    for item in items:
        try:
            d = analyze_source_item_for_split(item, args)
            item = dict(item)
            item['_diag'] = d
            if (not args.allow_missing_class_in_split) and len(d['missing']) > 0:
                rejected.append(item)
            else:
                usable.append(item)
            diagnostics.append((item, d, 'usable' if item in usable else 'rejected_missing_class'))
        except Exception as e:
            item = dict(item)
            item['_diag_error'] = str(e)
            rejected.append(item)
            diagnostics.append((item, None, f'rejected_error:{e}'))

    if len(usable) < 2:
        raise RuntimeError(f'사용 가능한 source가 너무 적습니다: usable={len(usable)}, rejected={len(rejected)}')

    ratio = float(args.train_ratio)
    if not (0.0 < ratio < 1.0):
        raise ValueError(f"--train_ratio는 0~1 사이여야 합니다: {ratio}")

    n_total = len(usable)
    n_train = int(round(n_total * ratio))
    n_train = max(1, min(n_total - 1, n_train))
    n_val = n_total - n_train

    # hard score 높은 순서가 val 후보
    usable_sorted = sorted(usable, key=lambda x: x['_diag']['hard_score'], reverse=True)
    val_items = usable_sorted[:n_val]
    train_items = usable_sorted[n_val:]

    # train 쪽은 source 순서가 치우치지 않게 다시 섞는다.
    rng.shuffle(train_items)

    return train_items, val_items, diagnostics


def apply_noise_profile(args, split: str):
    """
    split별 노이즈 강도를 다르게 적용하기 위한 args 복사본을 만든다.
    원본 args는 건드리지 않는다.

    profile:
      source : 명령어로 받은 값을 그대로 사용
      stable : train용. class 구조가 잘 보존되도록 약한 손상만 적용
      hard   : val용. 손상/노이즈가 큰 어려운 평가 샘플 생성
    """
    prof = args.train_noise_profile if split == 'train' else args.val_noise_profile
    a = copy.copy(args)

    if prof == 'source':
        return a

    if prof == 'stable':
        # train은 모델이 기본 구조를 제대로 배우도록 rect를 많이 남기고, label fragment를 줄인다.
        a.rect_keep_min = 0.85
        a.rect_keep_max = 0.98
        a.rect_region_sigma = 0.25
        a.rect_low_patch_min = 0
        a.rect_low_patch_max = 0
        a.square_center_keep_min = 0.80
        a.square_center_keep_max = 0.98
        a.square_outer_keep_min = 0.0
        a.square_outer_keep_max = 0.03
        a.square_region_sigma = 0.20
        a.label_close_kernel = max(int(a.label_close_kernel), 5)
        a.label_dilate_iter = max(int(a.label_dilate_iter), 1)
        a.min_contour_area = max(float(a.min_contour_area), 10.0)
        a.max_components_per_class = 1
        return a

    if prof == 'hard':
        # val은 실제로 어려운 상황을 평가하도록 더 강한 density 변화/손상을 적용한다.
        # 단, label 중복으로 square가 폭증하지 않도록 component는 1개로 제한한다.
        a.rect_keep_min = 0.35
        a.rect_keep_max = 0.70
        a.rect_region_sigma = 0.85
        a.rect_low_patch_min = 1
        a.rect_low_patch_max = 3
        a.rect_patch_reduce_min = 0.35
        a.rect_patch_reduce_max = 0.75
        a.square_center_keep_min = 0.55
        a.square_center_keep_max = 0.90
        a.square_outer_keep_min = 0.0
        a.square_outer_keep_max = 0.03
        a.square_region_sigma = 0.45
        a.label_close_kernel = max(int(a.label_close_kernel), 3)
        a.label_dilate_iter = max(int(a.label_dilate_iter), 0)
        a.min_contour_area = max(float(a.min_contour_area), 10.0)
        a.max_components_per_class = 1
        return a

    raise ValueError(f'알 수 없는 noise profile: {prof}')


def save_split_plan(diagnostics, train_items, val_items, out_path: Path):
    ensure_dir(out_path.parent)
    train_keys = {(it['source_split'], it['stem']) for it in train_items}
    val_keys = {(it['source_split'], it['stem']) for it in val_items}

    rows = []
    for item, d, status in diagnostics:
        key = (item.get('source_split', ''), item.get('stem', ''))
        out_split = 'train' if key in train_keys else ('val' if key in val_keys else '')
        row = {
            'source_split': item.get('source_split', ''),
            'stem': item.get('stem', ''),
            'out_split': out_split,
            'status': status,
        }
        if d is not None:
            row.update({
                'hard_score': d['hard_score'],
                'total_points': d['total_points'],
                'min_count': d['min_count'],
                'max_count': d['max_count'],
                'balance': d['balance'],
                'target_min': d['target_min'],
                'target_mean': d['target_mean'],
                'missing': ','.join(map(str, d['missing'])),
                'class_source': d['class_source'],
            })
            for cid in VALID_CLASSES:
                row[f'c{cid}_points'] = d['counts'][cid]
        else:
            row['error'] = item.get('_diag_error', '')
        rows.append(row)

    fields = sorted(set().union(*[r.keys() for r in rows])) if rows else []
    with open(out_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

def resolve_split_aug_mult(args, split: str):
    """
    train_ratio 모드에서 split별 생성 배율을 결정한다.
    - --aug_mult: train 증강 배율
    - --val_aug_mult: val 증강 배율
    - 둘 다 없으면 기존 --variants_per_image를 사용한다.
    """
    if args.aug_mult is None:
        train_mult = int(args.variants_per_image)
    else:
        train_mult = int(args.aug_mult)

    if args.val_aug_mult is None:
        # train_ratio를 쓰는 경우 val은 기본적으로 1배가 안전하다.
        val_mult = 1 if args.train_ratio is not None else train_mult
    else:
        val_mult = int(args.val_aug_mult)

    if split == "train":
        return max(1, train_mult)
    return max(1, val_mult)


def process_item_list(out_split: str, items, variants_per_item: int, args, summary_rows):
    """
    재분할된 item 목록을 out_split(train/val)에 저장한다.
    핵심: source_split과 out_split이 달라도 labels_source는 원래 source_split 폴더를 사용한다.
    """
    out_dataset_root = Path(args.out_dataset_root)
    out_source_root = Path(args.out_source_root) if args.out_source_root else None
    check_root = Path(args.check_dir)

    out_img_dir = out_dataset_root / "images" / out_split
    out_lbl_dir = out_dataset_root / "labels" / out_split
    out_source_dir = out_source_root / out_split if out_source_root else None
    check_dir = check_root / out_split

    ensure_dir(out_img_dir)
    ensure_dir(out_lbl_dir)
    ensure_dir(check_dir)
    if out_source_dir is not None:
        ensure_dir(out_source_dir)

    created = 0
    failed = 0
    check_saved = 0

    print(f"\n========== {out_split.upper()} / RESPLIT ==========")
    print(f"source images: {len(items)}")
    print(f"variants_per_item: {variants_per_item}")

    for idx, item in enumerate(items):
        img_path = Path(item["img_path"])
        label_path = Path(item["label_path"])
        source_dir = Path(item["source_dir"])
        stem = item["stem"]
        source_split = item["source_split"]

        try:
            split_args = apply_noise_profile(args, out_split)
            for variant_idx in range(variants_per_item):
                # source split을 prefix로 붙여 train/val에 같은 stem이 있을 때도 파일명 충돌을 막는다.
                if args.keep_original_name and variants_per_item == 1:
                    out_stem = f"{source_split}_{stem}"
                else:
                    out_stem = f"{source_split}_{stem}_segdens_{variant_idx + 1:02d}"

                result = process_one_image(
                    img_path=img_path,
                    label_path=label_path,
                    source_dir=source_dir,
                    out_stem=out_stem,
                    split=out_split,
                    idx=idx,
                    variant_idx=variant_idx,
                    args=split_args,
                )

                out_img_path = out_img_dir / f"{out_stem}.png"
                out_lbl_path = out_lbl_dir / f"{out_stem}.txt"

                imwrite_unicode(out_img_path, result["noisy_img"])
                obj_count, poly_counts = write_seg_label(out_lbl_path, result["class_masks"], result["W"], result["H"], split_args)

                if obj_count == 0:
                    raise RuntimeError("segmentation polygon 0개 생성")

                if out_source_dir is not None:
                    npz_path = out_source_dir / f"{out_stem}_seg_source.npz"
                    np.savez_compressed(
                        str(npz_path),
                        px=result["px"].astype(np.int32),
                        py=result["py"].astype(np.int32),
                        cls=result["cls"].astype(np.int32),
                        keep=result["keep"].astype(np.bool_),
                        seed=np.array([result["seed"]], dtype=np.int64),
                    )

                    src_uv = result["source_files"]["uv"]
                    src_meta = result["source_files"]["meta"]
                    if src_uv is not None:
                        shutil.copy2(str(src_uv), str(out_source_dir / f"{out_stem}_top_id_uv.npy"))
                    if src_meta is not None:
                        shutil.copy2(str(src_meta), str(out_source_dir / f"{out_stem}_meta.json"))

                if check_saved < args.check_max_per_split:
                    check_path = check_dir / f"{out_stem}_seg_check.jpg"
                    draw_seg_check(result["noisy_img"], result["class_masks"], check_path)
                    check_saved += 1

                row = {
                    "split": out_split,
                    "source_split": source_split,
                    "src_stem": stem,
                    "out_stem": out_stem,
                    "status": "ok",
                    "noise_profile": split_args.train_noise_profile if out_split == "train" else split_args.val_noise_profile,
                    "class_source": result["cls_source"],
                    "seg_objects": obj_count,
                    "poly_c0": poly_counts.get(0, 0),
                    "poly_c1": poly_counts.get(1, 0),
                    "poly_c2": poly_counts.get(2, 0),
                    "poly_c3": poly_counts.get(3, 0),
                }

                for cid in VALID_CLASSES:
                    st = result["stats"].get(cid, {})
                    row[f"c{cid}_points"] = st.get("points", 0)
                    row[f"c{cid}_kept"] = st.get("kept", 0)
                    row[f"c{cid}_keep_ratio"] = st.get("keep_ratio", 0.0)
                    row[f"c{cid}_mean_prob"] = st.get("mean_prob", 0.0)

                summary_rows.append(row)
                created += 1

            print(f"[OK] {out_split} {idx + 1}/{len(items)} {source_split}/{stem}")

        except Exception as e:
            failed += 1
            summary_rows.append({
                "split": out_split,
                "source_split": source_split,
                "src_stem": stem,
                "out_stem": "",
                "status": f"fail: {e}",
            })
            print(f"[FAIL] {out_split} {idx + 1}/{len(items)} {source_split}/{stem}: {e}")

    print(f"[{out_split}] created={created}, failed={failed}, check_saved={check_saved}")
    return created, failed

def save_summary(rows, out_path: Path):
    ensure_dir(out_path.parent)
    if not rows:
        return
    fields = sorted(set().union(*[r.keys() for r in rows]))
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--src_dataset_root", required=True, help="원본 clean YOLO dataset root")
    parser.add_argument("--src_labels_source_root", required=True, help="원본 labels_source root")
    parser.add_argument("--out_dataset_root", required=True, help="출력 YOLO segmentation dataset root")
    parser.add_argument("--out_source_root", default="", help="출력 source 기록 root")
    parser.add_argument("--check_dir", required=True, help="파일별 라벨 확인 이미지 저장 root")

    parser.add_argument("--variants_per_image", type=int, default=1, help="기존 방식용: train/val 모두 동일하게 생성할 variant 수")
    parser.add_argument("--train_ratio", type=float, default=None, help="기존 train/val을 합친 뒤 새 train 비율로 재분할. 예: 0.8")
    parser.add_argument("--aug_mult", type=int, default=None, help="train 증강 배율. --train_ratio 사용 시 train split에 적용")
    parser.add_argument("--val_aug_mult", type=int, default=None, help="val 증강 배율. --train_ratio 사용 시 기본값은 1")
    parser.add_argument("--split_mode", choices=["random", "val_hard"], default="random", help="random: 기존 랜덤 재분할, val_hard: 어려운 source를 val에 우선 배치")
    parser.add_argument("--hard_val_target_classes", default="1,2,3", help="val_hard에서 hard score를 계산할 대상 class. 기본값은 rect 계열 1,2,3")
    parser.add_argument("--allow_missing_class_in_split", action="store_true", help="class가 누락된 source도 train/val 분할에 포함")
    parser.add_argument("--train_noise_profile", choices=["source", "stable"], default="source", help="train 생성 노이즈 profile. stable은 학습용으로 구조 보존")
    parser.add_argument("--val_noise_profile", choices=["source", "hard"], default="source", help="val 생성 노이즈 profile. hard는 어려운 검증용 손상 강화")
    parser.add_argument("--keep_original_name", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check_max_per_split", type=int, default=40)
    parser.add_argument("--clean", action="store_true")

    # class npy가 반드시 있어야 한다고 강제할지 여부
    parser.add_argument("--require_class_npy", action="store_true", help="class npy가 없으면 bbox fallback 사용하지 않고 실패 처리")

    # point/mask rendering
    parser.add_argument("--point_radius", type=int, default=2)
    parser.add_argument("--preserve_point_count", action="store_true", help="삭제된 포인트 수만큼 살아남은 영역 주변에 보상 포인트를 추가해 전체 point 수를 유지")
    parser.add_argument("--compensate_ratio", type=float, default=1.0, help="삭제된 포인트 중 보상할 비율. 1.0이면 class별 원래 point 수에 가깝게 유지")
    parser.add_argument("--compensate_jitter_px", type=int, default=1, help="보상 포인트를 살아남은 source point 주변 몇 px 범위에 흩뿌릴지")

    # square noise
    parser.add_argument("--square_center_keep_min", type=float, default=0.75)
    parser.add_argument("--square_center_keep_max", type=float, default=0.98)
    parser.add_argument("--square_outer_keep_min", type=float, default=0.0)
    parser.add_argument("--square_outer_keep_max", type=float, default=0.05)
    parser.add_argument("--square_center_sigma_min", type=float, default=0.10)
    parser.add_argument("--square_center_sigma_max", type=float, default=0.20)
    parser.add_argument("--square_center_jitter", type=float, default=0.10)
    parser.add_argument("--square_region_sigma", type=float, default=0.35)
    parser.add_argument("--square_region_grid_min", type=int, default=3)
    parser.add_argument("--square_region_grid_max", type=int, default=5)

    # rect noise
    parser.add_argument("--rect_keep_min", type=float, default=0.50)
    parser.add_argument("--rect_keep_max", type=float, default=0.82)
    parser.add_argument("--rect_region_sigma", type=float, default=0.70)
    parser.add_argument("--rect_region_grid_min", type=int, default=3)
    parser.add_argument("--rect_region_grid_max", type=int, default=6)
    parser.add_argument("--rect_low_patch_min", type=int, default=1)
    parser.add_argument("--rect_low_patch_max", type=int, default=3)
    parser.add_argument("--rect_patch_sigma_min", type=float, default=0.12)
    parser.add_argument("--rect_patch_sigma_max", type=float, default=0.28)
    parser.add_argument("--rect_patch_reduce_min", type=float, default=0.25)
    parser.add_argument("--rect_patch_reduce_max", type=float, default=0.65)

    # common density
    parser.add_argument("--region_blur_ratio", type=float, default=0.08)
    parser.add_argument("--local_density_min", type=float, default=0.35)
    parser.add_argument("--local_density_max", type=float, default=1.85)
    parser.add_argument("--keep_prob_min", type=float, default=0.03)
    parser.add_argument("--keep_prob_max", type=float, default=0.98)

    parser.add_argument("--intensity_jitter", action="store_true")
    parser.add_argument("--intensity_min", type=float, default=0.85)
    parser.add_argument("--intensity_max", type=float, default=1.15)

    # segmentation label options
    parser.add_argument("--label_close_kernel", type=int, default=3)
    parser.add_argument("--label_dilate_iter", type=int, default=0)
    parser.add_argument("--approx_eps_ratio", type=float, default=0.006)
    parser.add_argument("--min_contour_area", type=float, default=3.0)
    parser.add_argument("--max_polygon_points", type=int, default=80)
    parser.add_argument("--max_components_per_class", type=int, default=3)

    args = parser.parse_args()

    out_dataset_root = Path(args.out_dataset_root)
    out_source_root = Path(args.out_source_root) if args.out_source_root else None
    check_dir = Path(args.check_dir)

    if args.clean:
        remove_if_exists(out_dataset_root)
        remove_if_exists(check_dir)
        if out_source_root is not None:
            remove_if_exists(out_source_root)

    ensure_dir(out_dataset_root)
    ensure_dir(check_dir)
    if out_source_root is not None:
        ensure_dir(out_source_root)

    print("========== CONFIG ==========")
    print(f"src_dataset_root:        {args.src_dataset_root}")
    print(f"src_labels_source_root:  {args.src_labels_source_root}")
    print(f"out_dataset_root:        {args.out_dataset_root}")
    print(f"out_source_root:         {args.out_source_root}")
    print(f"check_dir:               {args.check_dir}")
    print("image threshold used:    False")
    print("detect bbox to polygon:  False")
    print("noise and seg label:     same keep mask")
    print(f"preserve_point_count:    {args.preserve_point_count}")
    print(f"train_ratio:             {args.train_ratio}")
    print(f"aug_mult:                {args.aug_mult}")
    print(f"val_aug_mult:            {args.val_aug_mult}")
    print(f"variants_per_image:      {args.variants_per_image}")
    print(f"split_mode:              {args.split_mode}")
    print(f"train_noise_profile:     {args.train_noise_profile}")
    print(f"val_noise_profile:       {args.val_noise_profile}")
    print("============================")

    rows = []
    total_created = 0
    total_failed = 0

    if args.train_ratio is not None:
        if args.split_mode == "val_hard":
            train_items, val_items, diagnostics = make_resplit_items_val_hard(args)
            save_split_plan(diagnostics, train_items, val_items, check_dir / "split_plan.csv")
        else:
            train_items, val_items = make_resplit_items(args)
        train_aug = resolve_split_aug_mult(args, "train")
        val_aug = resolve_split_aug_mult(args, "val")

        print("\n========== RESPLIT INFO ==========")
        print(f"total source images: {len(train_items) + len(val_items)}")
        print(f"train source images: {len(train_items)}")
        print(f"val source images:   {len(val_items)}")
        print(f"train aug mult:      {train_aug}")
        print(f"val aug mult:        {val_aug}")
        print(f"expected train imgs: {len(train_items) * train_aug}")
        print(f"expected val imgs:   {len(val_items) * val_aug}")
        print("==================================")

        created, failed = process_item_list("train", train_items, train_aug, args, rows)
        total_created += created
        total_failed += failed

        created, failed = process_item_list("val", val_items, val_aug, args, rows)
        total_created += created
        total_failed += failed
    else:
        # 기존 원본 동작 유지: src_dataset_root/images/train, images/val을 그대로 사용하고
        # --variants_per_image를 train/val 모두에 동일하게 적용한다.
        if args.aug_mult is not None:
            args.variants_per_image = max(1, int(args.aug_mult))
        for split in ["train", "val"]:
            created, failed = process_split(split, args, rows)
            total_created += created
            total_failed += failed

    write_data_yaml(out_dataset_root)
    save_summary(rows, check_dir / "seg_region_density_generation_summary.csv")

    print("\n========== TOTAL RESULT ==========")
    print(f"created: {total_created}")
    print(f"failed:  {total_failed}")
    print(f"dataset: {out_dataset_root}")
    print(f"yaml:    {out_dataset_root / 'data.yaml'}")
    print(f"summary: {check_dir / 'seg_region_density_generation_summary.csv'}")
    print("==================================")


if __name__ == "__main__":
    main()
