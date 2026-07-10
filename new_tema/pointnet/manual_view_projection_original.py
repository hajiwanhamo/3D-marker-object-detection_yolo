#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
manual_view_projection_original.py

목적:
- 사용자가 Open3D 화면에서 직접 수직 투영 방향을 맞춤
- 화면 캡처가 아니라, 현재 view의 front/up 방향만 읽음
- 기존 make_top_id_projection.py와 동일하게 top_id height layer를 자름
- 기존과 같은 형식으로 저장:
  *_top_id.xyz
  *_top_id_uv.npy
  *_marker_all_uv.npy
  *_binary.png
  *_color.png
  *_meta.json
  images_color/*.png

사용 방식:
- Open3D 창에서 view를 맞춤
- 창을 닫지 말고 터미널에서 Enter 입력
- 그 순간의 view 방향으로 기존 방식 투영 저장
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
    """기존 make_top_id_projection.py를 모듈로 로드한다."""
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
        raise RuntimeError(f"xyz 파일 열 개수 부족: {path}")

    return P[:, :3].astype(np.float64)


def make_display_colors(P):
    """Open3D 표시용 색상. raw z 기준 단순 정규화."""
    z = P[:, 2].astype(np.float64)
    z_min = float(np.quantile(z, 0.02))
    z_max = float(np.quantile(z, 0.98))

    if z_max <= z_min:
        z_max = z_min + 1e-6

    t = np.clip((z - z_min) / (z_max - z_min), 0.0, 1.0)

    colors = np.zeros((len(P), 3), dtype=np.float64)
    colors[:, 0] = t
    colors[:, 1] = 1.0 - np.abs(t - 0.5) * 2.0
    colors[:, 2] = 1.0 - t

    return np.clip(colors, 0.0, 1.0)


def wait_manual_view(P, args):
    """
    Open3D 창에서 사용자가 view를 맞춘 뒤,
    터미널 Enter 입력 시점의 front/up/lookat을 가져온다.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(P)
    pcd.colors = o3d.utility.Vector3dVector(make_display_colors(P))

    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name="Manual projection view - view 맞춘 뒤 터미널에서 Enter",
        width=1200,
        height=900,
    )

    vis.add_geometry(pcd)

    opt = vis.get_render_option()
    opt.point_size = float(args.display_point_size)
    opt.background_color = np.array([0.0, 0.0, 0.0])

    last_view = {
        "front": np.array([0.0, 0.0, -1.0], dtype=np.float64),
        "up": np.array([0.0, 1.0, 0.0], dtype=np.float64),
        "lookat": P.mean(axis=0),
    }

    print("=" * 80)
    print("[MANUAL VIEW]")
    print("1. Open3D 창에서 마우스로 마커를 수직으로 보이게 맞춰라.")
    print("2. Open3D 창은 닫지 마라.")
    print("3. view를 맞춘 뒤 이 터미널에서 Enter를 눌러라.")
    print("4. Enter를 누른 순간의 front/up 방향으로 기존 top_id 방식 저장.")
    print("5. 화면 캡처가 아니다.")
    print("=" * 80)

    saved = False

    while True:
        alive = vis.poll_events()
        vis.update_renderer()

        try:
            vc = vis.get_view_control()

            front = np.asarray(vc.get_front(), dtype=np.float64)
            up = np.asarray(vc.get_up(), dtype=np.float64)
            lookat = np.asarray(vc.get_lookat(), dtype=np.float64)

            if np.all(np.isfinite(front)) and np.linalg.norm(front) > 1e-9:
                last_view["front"] = front.copy()

            if np.all(np.isfinite(up)) and np.linalg.norm(up) > 1e-9:
                last_view["up"] = up.copy()

            if np.all(np.isfinite(lookat)):
                last_view["lookat"] = lookat.copy()

        except Exception:
            pass

        # 터미널에서 Enter 입력 감지
        ready, _, _ = select.select([sys.stdin], [], [], 0.01)
        if ready:
            _ = sys.stdin.readline()
            saved = True
            break

        if not alive:
            break

        time.sleep(0.01)

    vis.destroy_window()

    if not saved:
        raise RuntimeError("Enter 입력 없이 Open3D 창이 닫혔음. 저장하지 않음.")

    return last_view["front"], last_view["up"], last_view["lookat"]


def build_axes_from_view(front, up):
    """
    Open3D view의 front/up으로 투영 좌표계 생성.
    front: 사용자가 맞춘 수직 방향
    up: 이미지의 위쪽 방향
    right: 이미지의 오른쪽 방향
    """
    normal = unit(front)

    up = np.asarray(up, dtype=np.float64)
    up = up - np.dot(up, normal) * normal
    v_axis = unit(up)

    u_axis = unit(np.cross(normal, v_axis))

    return normal, u_axis, v_axis


def choose_normal_sign(module, P, center, normal, args):
    """
    normal과 -normal 중 기존 select_top_id_layer가 더 그럴듯한 top_id 층을 선택하는 방향을 고른다.
    ID 상단층은 보통 전체 marker_points보다 작은 비율이므로, 너무 큰 선택은 피한다.
    """
    candidates = []

    for sign_name, n in [("front", normal), ("back", -normal)]:
        h = module.signed_height(P, center, n)

        try:
            id_mask, id_height_center, height_band_used, layer_info = module.select_top_id_layer(h, args)
        except Exception as e:
            candidates.append(
                {
                    "sign": sign_name,
                    "normal": n,
                    "ok": False,
                    "reason": str(e),
                    "count": 0,
                    "ratio": 1.0,
                }
            )
            continue

        count = int(np.sum(id_mask))
        ratio = float(count / max(len(P), 1))

        # 너무 적거나 너무 많으면 감점
        valid = count >= int(args.manual_min_selected_points) and ratio <= float(args.manual_max_selected_ratio)

        score = 0.0
        if not valid:
            score -= 1000.0

        # ID 상단은 하단 상판보다 점 수가 작을 가능성이 높음
        score -= ratio * 100.0
        score += min(count, 500) * 0.001

        candidates.append(
            {
                "sign": sign_name,
                "normal": n,
                "ok": True,
                "id_mask": id_mask,
                "h": h,
                "id_height_center": float(id_height_center),
                "height_band_used": float(height_band_used),
                "layer_info": layer_info,
                "count": count,
                "ratio": ratio,
                "valid": valid,
                "score": float(score),
            }
        )

    good = [c for c in candidates if c.get("ok", False)]

    if not good:
        raise RuntimeError(f"normal sign 선택 실패: {candidates}")

    if args.manual_sign == "front":
        chosen = [c for c in good if c["sign"] == "front"][0]
    elif args.manual_sign == "back":
        chosen = [c for c in good if c["sign"] == "back"][0]
    else:
        chosen = max(good, key=lambda x: x["score"])

    return chosen, candidates


def process_manual_projection(module, src_file, out_dir, front, up, lookat, args):
    """
    수동 view 방향을 normal/u/v로 사용하되,
    기존 make_top_id_projection.py의 top_id height layer selection과 저장 형식을 유지한다.
    """
    P = load_xyz(src_file)

    normal, u_axis, v_axis = build_axes_from_view(front, up)

    center = np.asarray(lookat, dtype=np.float64)
    if not np.all(np.isfinite(center)):
        center = P.mean(axis=0)

    chosen, sign_candidates = choose_normal_sign(
        module=module,
        P=P,
        center=center,
        normal=normal,
        args=args,
    )

    normal = chosen["normal"]
    h = chosen["h"]
    id_mask = chosen["id_mask"]

    P_top = P[id_mask]

    # 기존 방식처럼 전체 marker와 top_id를 같은 u/v 기준으로 투영
    uv_all = module.project_to_uv(P, center, u_axis, v_axis)
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

    module.save_xyz(out_top_xyz, P_top)
    np.save(str(out_top_uv), uv_top.astype(np.float32))
    np.save(str(out_all_uv), uv_all.astype(np.float32))

    module.save_png_gray(out_binary, binary_img)
    module.save_png_rgb(out_color, color_img)

    shutil.copy2(out_color, images_color_dir / out_color.name)

    xmin, xmax, ymin, ymax = bounds

    pixel_size_u = float((xmax - xmin) / max(args.image_size - 1, 1))
    pixel_size_v = float((ymax - ymin) / max(args.image_size - 1, 1))

    meta = {
        "source_file": str(src_file),
        "projection_mode": "manual_view_original_mechanism",
        "capture_image": False,
        "note": "Open3D 화면 캡처가 아니라, 사용자가 맞춘 view의 front/up 방향만 사용하고 기존 top_id height layer selection으로 저장함.",

        "image_size": int(args.image_size),
        "point_radius": int(args.point_radius),
        "view_margin": float(args.view_margin),

        "height_mode": str(args.height_mode),
        "top_candidate_q": float(args.top_candidate_q),
        "height_band_m": float(args.height_band_m),

        "manual_sign_mode": str(args.manual_sign),
        "chosen_sign": str(chosen["sign"]),
        "manual_min_selected_points": int(args.manual_min_selected_points),
        "manual_max_selected_ratio": float(args.manual_max_selected_ratio),

        "id_height_center": float(chosen["id_height_center"]),
        "height_band_m_used": float(chosen["height_band_used"]),
        "top_id_points": int(len(P_top)),
        "total_points": int(len(P)),

        "manual_lookat_3d": center.tolist(),
        "manual_front_3d": unit(front).tolist(),
        "manual_up_3d": unit(up).tolist(),

        "plane_center": center.tolist(),
        "plane_normal": normal.tolist(),
        "marker_origin": center.tolist(),

        "image_u_axis_3d": u_axis.tolist(),
        "image_v_axis_3d": v_axis.tolist(),

        "u_min": float(xmin),
        "u_max": float(xmax),
        "v_min": float(ymin),
        "v_max": float(ymax),
        "pixel_size_u_m": pixel_size_u,
        "pixel_size_v_m": pixel_size_v,

        "color_mode": "raw_z",
        "color_ref": "marker_all_raw_z",
        "color_min": float(cmin),
        "color_max": float(cmax),
        "color_q_min": float(args.color_q_min),
        "color_q_max": float(args.color_q_max),

        "debug": {
            "sign_candidates": [
                {
                    "sign": c.get("sign"),
                    "ok": c.get("ok"),
                    "count": c.get("count"),
                    "ratio": c.get("ratio"),
                    "valid": c.get("valid"),
                    "score": c.get("score"),
                    "reason": c.get("reason"),
                }
                for c in sign_candidates
            ],
            "layer_selection": chosen["layer_info"],
        },
    }

    module.save_meta_json(out_meta, meta)

    print("")
    print("[SAVED]")
    print("color:", out_color)
    print("images_color:", images_color_dir / out_color.name)
    print("top_id_points:", len(P_top))
    print("chosen_sign:", chosen["sign"])
    print("height_band_used:", chosen["height_band_used"])
    print("")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--file", required=True)
    parser.add_argument("--out", required=True)

    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--point_radius", type=int, default=2)
    parser.add_argument("--view_margin", type=float, default=1.2)

    parser.add_argument("--height_mode", choices=["fixed", "adaptive"], default="adaptive")
    parser.add_argument("--top_candidate_q", type=float, default=0.85)
    parser.add_argument("--height_band_m", type=float, default=0.03)

    parser.add_argument("--adaptive_search_q", type=float, default=0.60)
    parser.add_argument("--adaptive_upper_q", type=float, default=0.995)
    parser.add_argument("--adaptive_bins", type=int, default=64)
    parser.add_argument("--adaptive_min_points", type=int, default=20)
    parser.add_argument("--adaptive_min_ratio", type=float, default=0.003)
    parser.add_argument("--adaptive_neighbor_ratio", type=float, default=0.35)
    parser.add_argument("--adaptive_band_scale", type=float, default=3.0)
    parser.add_argument("--adaptive_min_band_m", type=float, default=0.010)
    parser.add_argument("--adaptive_max_band_m", type=float, default=0.080)
    parser.add_argument("--adaptive_max_selected_ratio", type=float, default=0.35)

    parser.add_argument("--color_q_min", type=float, default=0.02)
    parser.add_argument("--color_q_max", type=float, default=0.98)
    parser.add_argument("--invert_jet", action="store_true")

    parser.add_argument("--manual_sign", choices=["auto", "front", "back"], default="auto")
    parser.add_argument("--manual_min_selected_points", type=int, default=10)
    parser.add_argument("--manual_max_selected_ratio", type=float, default=0.50)

    parser.add_argument("--display_point_size", type=float, default=3.0)

    args = parser.parse_args()

    module = load_original_module()

    src_file = Path(args.file)
    out_dir = Path(args.out)

    P = load_xyz(src_file)

    front, up, lookat = wait_manual_view(P, args)

    process_manual_projection(
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
