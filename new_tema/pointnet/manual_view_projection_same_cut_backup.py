#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
manual_view_projection_same_cut.py

핵심:
- 사용자가 Open3D에서 직접 수직 투영 방향을 맞춘다.
- 화면 캡처가 아니다.
- 기존 make_top_id_projection.py의 top_id 자르기 메커니즘을 그대로 사용한다.
- 바꾸는 것은 자동 평면 normal이 아니라 사용자가 맞춘 view normal을 쓰는 것뿐이다.

저장 형식:
- *_top_id.xyz
- *_top_id_uv.npy
- *_marker_all_uv.npy
- *_binary.png
- *_color.png
- *_meta.json
- images_color/*.png
"""

import argparse
import importlib.util
import json
import select
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d


ORIGINAL_PATH = Path("/Users/hajiwan/Desktop/object_detection/new_tema/pointnet/make_top_id_projection.py")


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


def load_xyz(path: Path):
    """xyz 파일 로드."""
    P = np.loadtxt(str(path), dtype=np.float64)

    if P.ndim == 1:
        P = P.reshape(1, -1)

    if P.shape[1] < 3:
        raise RuntimeError(f"xyz 파일 형식 오류: {path}")

    return P[:, :3].astype(np.float64)


def make_display_colors(P):
    """Open3D 표시용 색상."""
    z = P[:, 2].astype(np.float64)
    zmin = float(np.quantile(z, 0.02))
    zmax = float(np.quantile(z, 0.98))

    if zmax <= zmin:
        zmax = zmin + 1e-6

    t = np.clip((z - zmin) / (zmax - zmin), 0.0, 1.0)

    colors = np.zeros((len(P), 3), dtype=np.float64)
    colors[:, 0] = t
    colors[:, 1] = 1.0 - np.abs(t - 0.5) * 2.0
    colors[:, 2] = 1.0 - t

    return np.clip(colors, 0.0, 1.0)


def get_manual_view(P, args):
    """
    Open3D 창에서 사용자가 view를 맞춘 뒤,
    터미널에서 Enter를 누른 순간의 front/up/lookat을 저장한다.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(P)
    pcd.colors = o3d.utility.Vector3dVector(make_display_colors(P))

    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name="Manual View - view 맞춘 뒤 터미널에서 Enter",
        width=1200,
        height=900,
    )

    vis.add_geometry(pcd)

    opt = vis.get_render_option()
    opt.point_size = float(args.display_point_size)
    opt.background_color = np.array([0.0, 0.0, 0.0])

    last_front = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    last_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    last_lookat = P.mean(axis=0)

    print("=" * 80)
    print("[MANUAL VIEW SETTING]")
    print("1. Open3D 창에서 마커를 원하는 수직 방향으로 맞춘다.")
    print("2. Open3D 창은 닫지 않는다.")
    print("3. 터미널을 클릭하고 Enter를 누른다.")
    print("4. Enter 순간의 view 방향만 저장한다.")
    print("5. 화면 캡처가 아니라 기존 방식으로 다시 투영한다.")
    print("=" * 80)

    confirmed = False

    while True:
        alive = vis.poll_events()
        vis.update_renderer()

        try:
            vc = vis.get_view_control()
            front = np.asarray(vc.get_front(), dtype=np.float64)
            up = np.asarray(vc.get_up(), dtype=np.float64)
            lookat = np.asarray(vc.get_lookat(), dtype=np.float64)

            if np.all(np.isfinite(front)) and np.linalg.norm(front) > 1e-9:
                last_front = front.copy()

            if np.all(np.isfinite(up)) and np.linalg.norm(up) > 1e-9:
                last_up = up.copy()

            if np.all(np.isfinite(lookat)):
                last_lookat = lookat.copy()

        except Exception:
            pass

        ready, _, _ = select.select([sys.stdin], [], [], 0.01)
        if ready:
            _ = sys.stdin.readline()
            confirmed = True
            break

        if not alive:
            break

        time.sleep(0.01)

    vis.destroy_window()

    if not confirmed:
        raise RuntimeError("Enter를 누르지 않고 창이 닫혔음. 저장하지 않음.")

    return last_front, last_up, last_lookat


def build_axes_from_view(front, up, normal_from_view):
    """
    Open3D view에서 투영축 생성.

    Open3D front는 보통 카메라가 바라보는 방향이다.
    사용자가 마커를 위에서 내려다보는 상태라면,
    마커의 위쪽 normal은 대체로 -front가 된다.
    그래서 기본값은 back(-front)이다.
    """
    front = unit(front)

    if normal_from_view == "front":
        normal = front
    else:
        normal = -front

    # 이미지 위쪽 방향
    up = np.asarray(up, dtype=np.float64)
    up = up - np.dot(up, normal) * normal
    v_axis = unit(up)

    # 이미지 오른쪽 방향
    u_axis = unit(np.cross(front, v_axis))

    # u/v/normal이 서로 직교하도록 한 번 더 보정
    u_axis = unit(u_axis - np.dot(u_axis, normal) * normal)
    v_axis = unit(np.cross(normal, u_axis))

    return normal, u_axis, v_axis


def process_same_cut(module, src_file, out_dir, front, up, lookat, args):
    """
    기존 make_top_id_projection.py와 같은 방식으로 저장한다.
    단, 평면 normal/u/v축만 수동 view 기준으로 교체한다.
    """
    P = load_xyz(src_file)

    normal, u_axis, v_axis = build_axes_from_view(
        front=front,
        up=up,
        normal_from_view=args.normal_from_view,
    )

    # center는 height offset에만 영향. 선택 범위 자체는 기존 select_top_id_layer가 결정.
    plane_center = np.asarray(lookat, dtype=np.float64)
    if not np.all(np.isfinite(plane_center)):
        plane_center = P.mean(axis=0)

    # 핵심: 기존 코드의 signed_height + select_top_id_layer 그대로 사용
    h = module.signed_height(P, plane_center, normal)
    id_mask, id_height_center, height_band_used, layer_info = module.select_top_id_layer(h, args)

    P_top = P[id_mask]

    # 기존 방식과 동일하게 전체 marker 기준 uv_all, 선택된 top_id 기준 uv_top 생성
    uv_all = module.project_to_uv(P, plane_center, u_axis, v_axis)
    uv_top = uv_all[id_mask]

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

    stem = src_file.stem

    out_top_xyz = out_dir / f"{stem}_top_id.xyz"
    out_top_uv = out_dir / f"{stem}_top_id_uv.npy"
    out_all_uv = out_dir / f"{stem}_marker_all_uv.npy"
    out_binary = out_dir / f"{stem}_binary.png"
    out_color = out_dir / f"{stem}_color.png"
    out_meta = out_dir / f"{stem}_meta.json"
    out_view = out_dir / f"{stem}_manual_view.json"

    module.save_xyz(out_top_xyz, P_top)
    np.save(str(out_top_uv), uv_top.astype(np.float32))
    np.save(str(out_all_uv), uv_all.astype(np.float32))

    module.save_png_gray(out_binary, binary_img)
    module.save_png_rgb(out_color, color_img)

    shutil.copy2(out_color, images_color_dir / out_color.name)

    xmin, xmax, ymin, ymax = bounds

    meta = {
        "source_file": str(src_file),
        "projection_mode": "manual_view_same_cut",
        "capture_image": False,
        "note": "수동 view 방향만 사용하고, top_id 선택은 기존 make_top_id_projection.py의 select_top_id_layer를 그대로 사용함.",

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

        "manual_normal_from_view": str(args.normal_from_view),
        "manual_front_3d": unit(front).tolist(),
        "manual_up_3d": unit(up).tolist(),
        "manual_lookat_3d": np.asarray(lookat, dtype=np.float64).tolist(),

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
        },
    }

    module.save_meta_json(out_meta, meta)

    view_info = {
        "front": unit(front).tolist(),
        "up": unit(up).tolist(),
        "lookat": np.asarray(lookat, dtype=np.float64).tolist(),
        "normal_from_view": str(args.normal_from_view),
        "normal": normal.tolist(),
        "u_axis": u_axis.tolist(),
        "v_axis": v_axis.tolist(),
    }

    out_view.write_text(json.dumps(view_info, indent=2, ensure_ascii=False), encoding="utf-8")

    print("")
    print("[SAVED]")
    print("color:", out_color)
    print("images_color:", images_color_dir / out_color.name)
    print("meta:", out_meta)
    print("view:", out_view)
    print("top_id_points:", int(len(P_top)))
    print("height_band_used:", float(height_band_used))
    print("normal_from_view:", args.normal_from_view)
    print("")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--file", required=True)
    parser.add_argument("--out", required=True)

    parser.add_argument(
        "--normal_from_view",
        choices=["back", "front"],
        default="back",
        help="back=-Open3D front, front=Open3D front. ID가 안 잡히면 front로 한 번만 반대로 실행.",
    )

    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--point_radius", type=int, default=2)
    parser.add_argument("--view_margin", type=float, default=1.2)

    # 기존 make_top_id_projection.py와 같은 top_id 선택 인자
    parser.add_argument("--height_mode", choices=["fixed", "adaptive"], default="adaptive")
    parser.add_argument("--top_candidate_q", type=float, default=0.85)
    parser.add_argument("--height_band_m", type=float, default=0.03)

    parser.add_argument("--adaptive_search_q", type=float, default=0.60)
    parser.add_argument("--adaptive_upper_q", type=float, default=0.995)
    parser.add_argument("--adaptive_bins", type=int, default=64)
    parser.add_argument("--adaptive_min_points", type=int, default=20)
    parser.add_argument("--adaptive_min_ratio", type=float, default=0.003)
    parser.add_argument("--adaptive_neighbor_ratio", type=float, default=0.35)
    parser.add_argument("--adaptive_band_scale", type=float, default=2.5)
    parser.add_argument("--adaptive_min_band_m", type=float, default=0.010)
    parser.add_argument("--adaptive_max_band_m", type=float, default=0.060)
    parser.add_argument("--adaptive_max_selected_ratio", type=float, default=0.35)

    parser.add_argument("--color_q_min", type=float, default=0.02)
    parser.add_argument("--color_q_max", type=float, default=0.98)
    parser.add_argument("--invert_jet", action="store_true")

    parser.add_argument("--display_point_size", type=float, default=3.0)

    args = parser.parse_args()

    module = load_original_module()

    src_file = Path(args.file)
    out_dir = Path(args.out)

    P = load_xyz(src_file)

    front, up, lookat = get_manual_view(P, args)

    process_same_cut(
        module=module,
        src_file=src_file,
        out_dir=out_dir,
        front=front,
        up=up,
        lookat=lookat,
        args=args,
    )


if __name__ == "__main__":
    main()
