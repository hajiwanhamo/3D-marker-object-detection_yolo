#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
manual_view_range_sweep_updown_10sets.py

목적:
- 사용자가 저장한 *_manual_view.json의 normal/u_axis/v_axis/lookat을 그대로 사용한다.
- 기존 make_top_id_projection.py의 adaptive layer 선택으로 기준 중심/범위를 먼저 계산한다.
- 그 기준 범위에서
  01~05_up   : 하단(base_lower)은 고정하고 상단만 확장
  06~10_down : 상단(base_upper)은 고정하고 하단만 확장
- 각 폴더에 기존 형식 그대로 저장한다.
  *_top_id.xyz
  *_top_id_uv.npy
  *_marker_all_uv.npy
  *_binary.png
  *_color.png
  *_meta.json
  images_color/*.png
"""

import argparse
import importlib.util
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ORIGINAL_PATH = Path("/Users/hajiwan/Desktop/object_detection/new_tema/pointnet/make_top_id_projection.py")


SETS = [
    ("01_down", "down", 0.010),
    ("02_down", "down", 0.020),
    ("03_down", "down", 0.030),
    ("04_down", "down", 0.040),
    ("05_down", "down", 0.050),
    ("06_down", "down", 0.060),
    ("07_down", "down", 0.070),
    ("08_down", "down", 0.080),
    ("09_down", "down", 0.090),
    ("10_down", "down", 0.100),
]


def unit(v):
    """벡터 정규화."""
    v = np.asarray(v, dtype=np.float64)
    return v / (np.linalg.norm(v) + 1e-12)


def load_original_module():
    """기존 make_top_id_projection.py 로드."""
    spec = importlib.util.spec_from_file_location("make_top_id_projection_original", ORIGINAL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"원본 코드 로드 실패: {ORIGINAL_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scan_xyz_files(in_path: Path):
    """단일 xyz 또는 폴더 입력 처리."""
    if in_path.is_file():
        return [in_path]

    files = sorted(in_path.glob("*_marker.xyz"))
    if not files:
        files = sorted(in_path.glob("*.xyz"))

    if not files:
        raise RuntimeError(f"xyz 파일 없음: {in_path}")

    return files


def load_manual_view(view_dir: Path, stem: str, P: np.ndarray):
    """
    manual_view.json에서 수동 저장된 projection 축 로드.
    lookat이 없거나 비정상이면 meta.json의 plane_center를 사용하고,
    그것도 없으면 P.mean을 사용한다.
    """
    view_path = view_dir / f"{stem}_manual_view.json"
    meta_path = view_dir / f"{stem}_meta.json"

    if not view_path.exists():
        raise FileNotFoundError(f"manual_view 없음: {view_path}")

    d = json.loads(view_path.read_text(encoding="utf-8"))

    normal = unit(np.asarray(d["normal"], dtype=np.float64))
    u_axis = unit(np.asarray(d["u_axis"], dtype=np.float64))
    v_axis = unit(np.asarray(d["v_axis"], dtype=np.float64))

    # 수치 오차만 보정. 방향 자체는 저장된 manual view를 유지.
    u_axis = unit(u_axis - np.dot(u_axis, normal) * normal)
    v_axis = unit(v_axis - np.dot(v_axis, normal) * normal)

    if np.dot(np.cross(u_axis, v_axis), normal) < 0:
        v_axis = -v_axis

    lookat = d.get("lookat", None)

    if lookat is None and meta_path.exists():
        md = json.loads(meta_path.read_text(encoding="utf-8"))
        lookat = md.get("plane_center", md.get("marker_origin", None))

    if lookat is None:
        center = P.mean(axis=0)
    else:
        center = np.asarray(lookat, dtype=np.float64)
        if center.shape[0] < 3 or not np.all(np.isfinite(center[:3])):
            center = P.mean(axis=0)
        else:
            center = center[:3]

    return {
        "view_path": str(view_path),
        "normal": normal,
        "u_axis": u_axis,
        "v_axis": v_axis,
        "center": center,
    }


def make_base_args(args):
    """기준 adaptive layer 계산용 인자."""
    return SimpleNamespace(
        height_mode="adaptive",
        top_candidate_q=float(args.base_top_candidate_q),
        height_band_m=float(args.base_height_band_m),

        adaptive_search_q=float(args.adaptive_search_q),
        adaptive_upper_q=float(args.adaptive_upper_q),
        adaptive_bins=int(args.adaptive_bins),
        adaptive_min_points=int(args.adaptive_min_points),
        adaptive_min_ratio=float(args.adaptive_min_ratio),
        adaptive_neighbor_ratio=float(args.adaptive_neighbor_ratio),
        adaptive_band_scale=float(args.adaptive_band_scale),
        adaptive_min_band_m=float(args.adaptive_min_band_m),
        adaptive_max_band_m=float(args.base_adaptive_max_band_m),
        adaptive_max_selected_ratio=float(args.adaptive_max_selected_ratio),

        image_size=int(args.image_size),
        point_radius=int(args.point_radius),
        view_margin=float(args.view_margin),

        color_q_min=float(args.color_q_min),
        color_q_max=float(args.color_q_max),
        invert_jet=bool(args.invert_jet),
    )


def select_expanded_layer(module, h, base_args, mode: str, expand_m: float):
    """
    기존 adaptive 방식으로 base layer를 잡은 뒤,
    상단 또는 하단 한쪽만 확장한다.
    """
    base_mask, base_center, base_band, base_info = module.select_top_id_layer(h, base_args)

    base_center = float(base_center)
    base_band = float(base_band)
    base_lower = base_center - base_band
    base_upper = base_center + base_band

    expand_m = float(expand_m)

    if mode == "up":
        # 하단 고정 + 상단 확장
        new_lower = base_lower
        new_upper = base_upper + expand_m
        rule = "lower_fixed_upper_expanded"
    elif mode == "down":
        # 상단 고정 + 하단 확장
        new_lower = base_lower - expand_m
        new_upper = base_upper
        rule = "upper_fixed_lower_expanded"
    else:
        raise ValueError(f"지원하지 않는 mode: {mode}")

    mask = (h >= new_lower) & (h <= new_upper)
    selected = h[mask]

    if selected.size == 0:
        # 완전히 실패하면 기준 layer로 fallback
        mask = base_mask
        selected = h[mask]
        new_lower = base_lower
        new_upper = base_upper
        expand_m = 0.0
        rule = rule + "_fallback_base"

    new_center = 0.5 * (new_lower + new_upper)
    new_band = 0.5 * (new_upper - new_lower)

    layer_info = {
        "height_mode": "manual_view_updown_expand",
        "selection_rule": rule,
        "expand_mode": mode,
        "expand_m": float(expand_m),

        "base_id_height_center": float(base_center),
        "base_height_band_m_used": float(base_band),
        "base_lower": float(base_lower),
        "base_upper": float(base_upper),

        "id_height_center": float(new_center),
        "height_band_m": float(new_band),
        "height_band_m_used": float(new_band),
        "new_lower": float(new_lower),
        "new_upper": float(new_upper),

        "selected_height_min": float(selected.min()) if selected.size > 0 else None,
        "selected_height_max": float(selected.max()) if selected.size > 0 else None,
        "selected_points": int(np.count_nonzero(mask)),
        "base_layer_info": base_info,
    }

    return mask, float(new_center), float(new_band), layer_info


def process_one(module, xyz_path: Path, view_dir: Path, out_dir: Path, mode: str, expand_m: float, args):
    """파일 하나를 manual view 기준으로 projection 저장."""
    P = module.load_xyz(xyz_path)
    stem = xyz_path.stem

    view = load_manual_view(view_dir=view_dir, stem=stem, P=P)

    center = view["center"]
    normal = view["normal"]
    u_axis = view["u_axis"]
    v_axis = view["v_axis"]

    base_args = make_base_args(args)

    # manual view normal 기준 높이 계산
    h = module.signed_height(P, center, normal)

    # 기준 layer에서 상단/하단 한쪽만 확장
    id_mask, id_height_center, height_band_used, layer_info = select_expanded_layer(
        module=module,
        h=h,
        base_args=base_args,
        mode=mode,
        expand_m=expand_m,
    )

    P_top = P[id_mask]

    # manual view u/v 기준 투영
    uv_all = module.project_to_uv(P, center, u_axis, v_axis)
    uv_top = uv_all[id_mask]

    # 기존 projection과 동일하게 marker 전체 기준 bounds
    bounds = module.compute_view_bounds(uv_all, args.view_margin)

    px, py, valid = module.uv_to_pixel(
        uv=uv_top,
        bounds=bounds,
        image_size=args.image_size,
    )

    binary_img = module.rasterize_binary(
        px=px,
        py=py,
        valid=valid,
        image_size=args.image_size,
        point_radius=args.point_radius,
    )

    if len(P_top) > 0:
        top_values = P_top[:, 2]
        all_values = P[:, 2]
        cmin = float(np.quantile(all_values, args.color_q_min))
        cmax = float(np.quantile(all_values, args.color_q_max))
        rgb = module.jet_color_with_range(
            top_values,
            cmin,
            cmax,
            invert=bool(args.invert_jet),
        )
    else:
        cmin = 0.0
        cmax = 1.0
        rgb = np.zeros((0, 3), dtype=np.uint8)

    color_img = module.rasterize_color(
        px=px,
        py=py,
        valid=valid,
        rgb=rgb,
        image_size=args.image_size,
        point_radius=args.point_radius,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    images_color_dir = out_dir / "images_color"
    images_color_dir.mkdir(parents=True, exist_ok=True)

    out_top_xyz = out_dir / f"{stem}_top_id.xyz"
    out_top_uv = out_dir / f"{stem}_top_id_uv.npy"
    out_all_uv = out_dir / f"{stem}_marker_all_uv.npy"
    out_binary = out_dir / f"{stem}_binary.png"
    out_color = out_dir / f"{stem}_color.png"
    out_meta = out_dir / f"{stem}_meta.json"

    module.save_xyz(out_top_xyz, P_top)
    np.save(str(out_top_uv), uv_top.astype(np.float32))
    np.save(str(out_all_uv), uv_all.astype(np.float32))

    module.save_png_gray(out_binary, binary_img)
    module.save_png_rgb(out_color, color_img)
    shutil.copy2(out_color, images_color_dir / out_color.name)

    xmin, xmax, ymin, ymax = bounds

    meta = {
        "source_file": str(xyz_path),
        "projection_mode": "manual_view_updown_10sets",
        "manual_view_source": view["view_path"],
        "capture_image": False,

        "expand_mode": str(mode),
        "expand_m": float(expand_m),

        "image_size": int(args.image_size),
        "point_radius": int(args.point_radius),
        "view_margin": float(args.view_margin),

        "base_top_candidate_q": float(args.base_top_candidate_q),
        "base_height_band_m": float(args.base_height_band_m),
        "base_adaptive_max_band_m": float(args.base_adaptive_max_band_m),

        "plane_center": center.tolist(),
        "plane_normal": normal.tolist(),
        "marker_origin": center.tolist(),
        "image_u_axis_3d": u_axis.tolist(),
        "image_v_axis_3d": v_axis.tolist(),

        "u_min": float(xmin),
        "u_max": float(xmax),
        "v_min": float(ymin),
        "v_max": float(ymax),
        "pixel_size_u_m": float((xmax - xmin) / max(args.image_size - 1, 1)),
        "pixel_size_v_m": float((ymax - ymin) / max(args.image_size - 1, 1)),

        "color_mode": "raw_z",
        "color_ref": "marker_all_raw_z",
        "color_min": float(cmin),
        "color_max": float(cmax),

        "debug": {
            "total_points": int(len(P)),
            "top_id_points": int(len(P_top)),
            "id_height_center": float(id_height_center),
            "height_band_m_used": float(height_band_used),
            "selected_height_min": float(h[id_mask].min()) if int(np.sum(id_mask)) > 0 else None,
            "selected_height_max": float(h[id_mask].max()) if int(np.sum(id_mask)) > 0 else None,
            "height_min": float(h.min()),
            "height_max": float(h.max()),
            "layer_selection": layer_info,
        },
    }

    module.save_meta_json(out_meta, meta)

    print(
        f"[OK] {out_dir.name} | {xyz_path.name} | "
        f"mode={mode} | expand={expand_m:.3f} | "
        f"top_id_points={len(P_top)} | band_used={height_band_used:.6f}"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--in_path", required=True)
    parser.add_argument("--view_dir", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--overwrite", action="store_true")

    # 기준 범위 계산값
    parser.add_argument("--base_top_candidate_q", type=float, default=0.85)
    parser.add_argument("--base_height_band_m", type=float, default=0.030)
    parser.add_argument("--base_adaptive_max_band_m", type=float, default=0.080)

    # adaptive 기본값
    parser.add_argument("--adaptive_search_q", type=float, default=0.60)
    parser.add_argument("--adaptive_upper_q", type=float, default=0.995)
    parser.add_argument("--adaptive_bins", type=int, default=64)
    parser.add_argument("--adaptive_min_points", type=int, default=20)
    parser.add_argument("--adaptive_min_ratio", type=float, default=0.003)
    parser.add_argument("--adaptive_neighbor_ratio", type=float, default=0.35)
    parser.add_argument("--adaptive_band_scale", type=float, default=2.5)
    parser.add_argument("--adaptive_min_band_m", type=float, default=0.010)
    parser.add_argument("--adaptive_max_selected_ratio", type=float, default=0.35)

    # 이미지 생성값
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--point_radius", type=int, default=2)
    parser.add_argument("--view_margin", type=float, default=1.2)

    # 컬러맵값
    parser.add_argument("--color_q_min", type=float, default=0.02)
    parser.add_argument("--color_q_max", type=float, default=0.98)
    parser.add_argument("--invert_jet", action="store_true")

    args = parser.parse_args()

    module = load_original_module()

    in_path = Path(args.in_path)
    view_dir = Path(args.view_dir)
    out_root = Path(args.out_root)

    files = scan_xyz_files(in_path)

    if args.overwrite and out_root.exists():
        shutil.rmtree(out_root)

    out_root.mkdir(parents=True, exist_ok=True)

    index_text = [
        "manual_view_range_sweep_updown_10sets",
        "",
        "기준 범위:",
        f"base_top_candidate_q={args.base_top_candidate_q}",
        f"base_height_band_m={args.base_height_band_m}",
        f"base_adaptive_max_band_m={args.base_adaptive_max_band_m}",
        "",
        "01_up   : lower fixed, upper +0.010m",
        "02_up   : lower fixed, upper +0.020m",
        "03_up   : lower fixed, upper +0.030m",
        "04_up   : lower fixed, upper +0.040m",
        "05_up   : lower fixed, upper +0.050m",
        "06_down : upper fixed, lower -0.010m",
        "07_down : upper fixed, lower -0.020m",
        "08_down : upper fixed, lower -0.030m",
        "09_down : upper fixed, lower -0.040m",
        "10_down : upper fixed, lower -0.050m",
        "",
        f"input={in_path}",
        f"view_dir={view_dir}",
    ]
    (out_root / "index.txt").write_text("\n".join(index_text), encoding="utf-8")

    print("=" * 80)
    print("[MANUAL VIEW RANGE SWEEP UP/DOWN 10SETS]")
    print("input files:", len(files))
    print("view_dir:", view_dir)
    print("out_root:", out_root)
    print("=" * 80)

    for folder_name, mode, expand_m in SETS:
        out_dir = out_root / folder_name
        print("")
        print("=" * 80)
        print(f"[SET] {folder_name} | mode={mode} | expand={expand_m:.3f}m")
        print("=" * 80)

        ok = 0
        fail = 0

        for fp in files:
            try:
                process_one(
                    module=module,
                    xyz_path=fp,
                    view_dir=view_dir,
                    out_dir=out_dir,
                    mode=mode,
                    expand_m=expand_m,
                    args=args,
                )
                ok += 1
            except Exception as e:
                fail += 1
                print(f"[FAIL] {folder_name} | {fp.name} | {e}")

        print(f"[SET DONE] {folder_name} | ok={ok} | fail={fail}")

    print("")
    print("[DONE]", out_root)


if __name__ == "__main__":
    main()
