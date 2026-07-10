#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
manual_view_range_sweep_same_cut.py

목적:
- Open3D 수동 view를 다시 잡지 않는다.
- 이미 저장된 *_manual_view.json의 normal/u_axis/v_axis/lookat을 그대로 사용한다.
- 기존 make_top_id_projection.py의 select_top_id_layer()를 그대로 사용한다.
- height_band_m / adaptive_max_band_m 범위만 바꿔서 실해역 projection을 여러 세트 생성한다.
- 각 세트마다 images_color 폴더를 자동 생성한다.

출력 구조:
out_root/
  real_flat_hband003_max008/
    *_color.png
    *_binary.png
    *_top_id.xyz
    *_top_id_uv.npy
    *_marker_all_uv.npy
    *_meta.json
    images_color/
      *_color.png
  real_flat_hband004_max009/
  ...
"""

import argparse
import importlib.util
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ORIGINAL_PATH = Path("/Users/hajiwan/Desktop/object_detection/new_tema/pointnet/make_top_id_projection.py")


# 앞서 사용한 범위별 실해역 projection 설정
SWEEP_PRESET = [
    ("hband003_max008", 0.03, 0.080),
    ("hband004_max009", 0.04, 0.090),
    ("hband005_max010", 0.05, 0.100),
    ("hband006_max012", 0.06, 0.120),
]


def load_original_module():
    """기존 make_top_id_projection.py를 그대로 불러온다."""
    spec = importlib.util.spec_from_file_location("make_top_id_projection_original", ORIGINAL_PATH)

    if spec is None or spec.loader is None:
        raise RuntimeError(f"원본 코드 로드 실패: {ORIGINAL_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def unit(v):
    """벡터 정규화."""
    v = np.asarray(v, dtype=np.float64)
    return v / (np.linalg.norm(v) + 1e-12)


def load_manual_view(view_path: Path):
    """
    수동 view 정보 로드.
    manual_view_projection_same_cut.py에서 저장한 *_manual_view.json을 사용한다.
    """
    if not view_path.exists():
        raise FileNotFoundError(f"manual view json 없음: {view_path}")

    d = json.loads(view_path.read_text(encoding="utf-8"))

    normal = unit(np.asarray(d["normal"], dtype=np.float64))
    u_axis = unit(np.asarray(d["u_axis"], dtype=np.float64))
    v_axis = unit(np.asarray(d["v_axis"], dtype=np.float64))
    lookat = np.asarray(d["lookat"], dtype=np.float64)

    # 중요:
    # 사용자가 Open3D에서 맞춘 상태를 그대로 재사용해야 하므로
    # 저장된 u_axis / v_axis를 cross()로 다시 만들지 않는다.
    # 여기서는 길이 정규화만 한다.
    return {
        "normal": normal,
        "u_axis": u_axis,
        "v_axis": v_axis,
        "lookat": lookat,
        "raw": d,
    }


def compute_square_bounds(uv_all, view_margin):
    """
    왜곡 방지용 bounds 계산.
    u/v를 각각 따로 512에 맞추지 않고, 동일한 meter-per-pixel 스케일로 저장한다.
    즉 pixel_size_u_m == pixel_size_v_m 이 되도록 정사각형 view box를 만든다.
    """
    uv_all = np.asarray(uv_all, dtype=np.float64)

    umin = float(np.min(uv_all[:, 0]))
    umax = float(np.max(uv_all[:, 0]))
    vmin = float(np.min(uv_all[:, 1]))
    vmax = float(np.max(uv_all[:, 1]))

    uc = 0.5 * (umin + umax)
    vc = 0.5 * (vmin + vmax)

    span_u = umax - umin
    span_v = vmax - vmin
    span = max(span_u, span_v, 1e-9) * float(view_margin)

    return (
        uc - span / 2.0,
        uc + span / 2.0,
        vc - span / 2.0,
        vc + span / 2.0,
    )


def make_args(base_args, height_band_m: float, adaptive_max_band_m: float):
    """
    기존 make_top_id_projection.py의 select_top_id_layer()가 요구하는 인자 세트.
    여기서 바꾸는 것은 height_band_m / adaptive_max_band_m뿐이다.
    """
    return SimpleNamespace(
        # 기존 top_id 선택 방식
        height_mode="adaptive",
        top_candidate_q=0.85,
        height_band_m=float(height_band_m),

        # adaptive 선택 파라미터
        adaptive_search_q=float(base_args.adaptive_search_q),
        adaptive_upper_q=float(base_args.adaptive_upper_q),
        adaptive_bins=int(base_args.adaptive_bins),
        adaptive_min_points=int(base_args.adaptive_min_points),
        adaptive_min_ratio=float(base_args.adaptive_min_ratio),
        adaptive_neighbor_ratio=float(base_args.adaptive_neighbor_ratio),
        adaptive_band_scale=float(base_args.adaptive_band_scale),
        adaptive_min_band_m=float(base_args.adaptive_min_band_m),
        adaptive_max_band_m=float(adaptive_max_band_m),
        adaptive_max_selected_ratio=float(base_args.adaptive_max_selected_ratio),

        # 이미지 저장 파라미터
        image_size=int(base_args.image_size),
        point_radius=int(base_args.point_radius),
        view_margin=float(base_args.view_margin),

        # 컬러맵 파라미터
        color_q_min=float(base_args.color_q_min),
        color_q_max=float(base_args.color_q_max),
        invert_jet=bool(base_args.invert_jet),
    )


def process_one(module, xyz_path: Path, view_dir: Path, out_dir: Path, args):
    """
    단일 marker xyz를 저장된 manual view 기준으로 projection한다.
    자르는 방식은 기존 select_top_id_layer() 그대로 사용한다.
    """
    P = module.load_xyz(xyz_path)
    stem = xyz_path.stem

    view_path = view_dir / f"{stem}_manual_view.json"
    view = load_manual_view(view_path)

    plane_center = view["lookat"]
    normal = view["normal"]
    u_axis = view["u_axis"]
    v_axis = view["v_axis"]

    # 핵심: 기존 코드와 같은 signed_height + select_top_id_layer 사용
    h = module.signed_height(P, plane_center, normal)
    id_mask, id_height_center, height_band_used, layer_info = module.select_top_id_layer(h, args)

    P_top = P[id_mask]

    # 핵심: 수동 view의 u/v축으로 전체 marker와 top_id를 같은 기준으로 투영
    uv_all = module.project_to_uv(P, plane_center, u_axis, v_axis)
    uv_top = uv_all[id_mask]

    bounds = compute_square_bounds(uv_all, args.view_margin)

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
        "projection_mode": "manual_view_range_sweep_same_cut",
        "capture_image": False,
        "note": "저장된 manual view 방향을 사용하고, top_id 선택은 기존 make_top_id_projection.py의 select_top_id_layer를 그대로 사용함.",

        "image_size": int(args.image_size),
        "point_radius": int(args.point_radius),
        "view_margin": float(args.view_margin),

        "height_mode": str(args.height_mode),
        "top_candidate_q": float(args.top_candidate_q),
        "height_band_m": float(args.height_band_m),

        "adaptive_search_q": float(args.adaptive_search_q),
        "adaptive_upper_q": float(args.adaptive_upper_q),
        "adaptive_bins": int(args.adaptive_bins),
        "adaptive_min_points": int(args.adaptive_min_points),
        "adaptive_min_ratio": float(args.adaptive_min_ratio),
        "adaptive_neighbor_ratio": float(args.adaptive_neighbor_ratio),
        "adaptive_band_scale": float(args.adaptive_band_scale),
        "adaptive_min_band_m": float(args.adaptive_min_band_m),
        "adaptive_max_band_m": float(args.adaptive_max_band_m),
        "adaptive_max_selected_ratio": float(args.adaptive_max_selected_ratio),

        "plane_center": plane_center.tolist(),
        "plane_normal": normal.tolist(),
        "marker_origin": plane_center.tolist(),

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
        "invert_jet": bool(args.invert_jet),
        "color_min": float(cmin),
        "color_max": float(cmax),
        "color_q_min": float(args.color_q_min),
        "color_q_max": float(args.color_q_max),

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
            "manual_view_source": str(view_path),
        },
    }

    module.save_meta_json(out_meta, meta)

    print(
        f"[OK] {xyz_path.name} | "
        f"top_id_points={len(P_top)} | "
        f"band_used={height_band_used:.6f} | "
        f"height_band_m={args.height_band_m:.3f} | "
        f"adaptive_max_band_m={args.adaptive_max_band_m:.3f}"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--in_path", required=True, help="실해역 marker_points 폴더")
    parser.add_argument("--view_dir", required=True, help="*_manual_view.json이 저장된 폴더")
    parser.add_argument("--out_root", required=True, help="범위별 real_flat_* 폴더를 만들 root")
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--point_radius", type=int, default=2)
    parser.add_argument("--view_margin", type=float, default=1.2)

    # 기존 adaptive 기본값 유지
    parser.add_argument("--adaptive_search_q", type=float, default=0.60)
    parser.add_argument("--adaptive_upper_q", type=float, default=0.995)
    parser.add_argument("--adaptive_bins", type=int, default=64)
    parser.add_argument("--adaptive_min_points", type=int, default=20)
    parser.add_argument("--adaptive_min_ratio", type=float, default=0.003)
    parser.add_argument("--adaptive_neighbor_ratio", type=float, default=0.35)
    parser.add_argument("--adaptive_band_scale", type=float, default=2.5)
    parser.add_argument("--adaptive_min_band_m", type=float, default=0.010)
    parser.add_argument("--adaptive_max_selected_ratio", type=float, default=0.35)

    parser.add_argument("--color_q_min", type=float, default=0.02)
    parser.add_argument("--color_q_max", type=float, default=0.98)
    parser.add_argument("--invert_jet", action="store_true")

    args = parser.parse_args()

    module = load_original_module()

    in_path = Path(args.in_path)
    view_dir = Path(args.view_dir)
    out_root = Path(args.out_root)

    files = sorted(in_path.glob("*_marker.xyz"))

    if not files:
        raise RuntimeError(f"입력 marker xyz 없음: {in_path}")

    out_root.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("[MANUAL VIEW RANGE SWEEP]")
    print("input:", in_path)
    print("view_dir:", view_dir)
    print("out_root:", out_root)
    print("files:", len(files))
    print("=" * 80)

    for name, hband, hmax in SWEEP_PRESET:
        out_dir = out_root / f"real_flat_{name}"

        if out_dir.exists():
            if args.overwrite:
                shutil.rmtree(out_dir)
            else:
                raise FileExistsError(f"이미 존재함: {out_dir}  --overwrite 사용")

        out_dir.mkdir(parents=True, exist_ok=True)

        run_args = make_args(args, height_band_m=hband, adaptive_max_band_m=hmax)

        print("")
        print("=" * 80)
        print(f"MAKE REAL PROJECTION: {name}")
        print(f"height_band_m={hband} adaptive_max_band_m={hmax}")
        print("out:", out_dir)
        print("=" * 80)

        ok_count = 0

        for fp in files:
            process_one(
                module=module,
                xyz_path=fp,
                view_dir=view_dir,
                out_dir=out_dir,
                args=run_args,
            )
            ok_count += 1

        print(f"[DONE] {name}: ok={ok_count}/{len(files)}")
        print(f"images_color: {out_dir / 'images_color'}")

    print("")
    print("[ALL DONE]")
    print("out_root:", out_root)


if __name__ == "__main__":
    main()
