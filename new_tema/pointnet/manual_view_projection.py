#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
manual_view_projection.py

기능:
- Open3D PointCloud 화면에서 사용자가 직접 마우스로 view를 맞춤
- S 키를 누르면 현재 화면의 카메라 front/up 방향을 읽음
- 화면 캡처가 아니라, 그 방향으로 3D point를 다시 직교투영해서 PNG 생성
- *_color.png, *_binary.png, *_top_id_uv.npy, *_marker_all_uv.npy, *_meta.json 저장
- images_color 폴더에도 color image 자동 저장
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import open3d as o3d

try:
    import cv2
except Exception:
    cv2 = None

try:
    from PIL import Image
except Exception:
    Image = None


def unit(v):
    """벡터 정규화."""
    v = np.asarray(v, dtype=np.float64)
    return v / (np.linalg.norm(v) + 1e-12)


def load_xyz(path: Path):
    """xyz 파일 로드. 3열 이상이면 앞의 x,y,z만 사용."""
    P = np.loadtxt(str(path), dtype=np.float64)

    if P.ndim == 1:
        P = P.reshape(1, -1)

    if P.shape[1] < 3:
        raise RuntimeError(f"xyz 파일 열 개수 부족: {path}")

    return P[:, :3].astype(np.float64)


def save_xyz(path: Path, P: np.ndarray):
    """선택된 point를 xyz로 저장."""
    np.savetxt(str(path), P, fmt="%.8f")


def jet_color(values):
    """값을 jet 스타일 RGB 색상으로 변환."""
    values = np.asarray(values, dtype=np.float64)

    if len(values) == 0:
        return np.zeros((0, 3), dtype=np.uint8)

    vmin = float(np.quantile(values, 0.02))
    vmax = float(np.quantile(values, 0.98))

    if vmax <= vmin:
        vmax = vmin + 1e-6

    t = np.clip((values - vmin) / (vmax - vmin), 0.0, 1.0)

    r = np.clip(1.5 - np.abs(4.0 * t - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * t - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * t - 1.0), 0.0, 1.0)

    rgb = np.stack([r, g, b], axis=1)
    return (rgb * 255).astype(np.uint8)


def save_png_rgb(path: Path, img: np.ndarray):
    """RGB PNG 저장."""
    path.parent.mkdir(parents=True, exist_ok=True)

    if cv2 is not None:
        cv2.imwrite(str(path), img[:, :, ::-1])
    elif Image is not None:
        Image.fromarray(img).save(str(path))
    else:
        raise RuntimeError("PNG 저장 실패: cv2 또는 PIL이 필요함")


def save_png_gray(path: Path, img: np.ndarray):
    """gray PNG 저장."""
    path.parent.mkdir(parents=True, exist_ok=True)

    if cv2 is not None:
        cv2.imwrite(str(path), img)
    elif Image is not None:
        Image.fromarray(img).save(str(path))
    else:
        raise RuntimeError("PNG 저장 실패: cv2 또는 PIL이 필요함")


def rasterize_rgb(px, py, rgb, image_size, radius):
    """point를 RGB 이미지로 rasterize."""
    img = np.zeros((image_size, image_size, 3), dtype=np.uint8)

    for x, y, c in zip(px, py, rgb):
        x = int(x)
        y = int(y)

        if x < 0 or x >= image_size or y < 0 or y >= image_size:
            continue

        if cv2 is not None:
            cv2.circle(img, (x, y), int(radius), c.tolist(), -1)
        else:
            r = int(radius)
            x0 = max(0, x - r)
            x1 = min(image_size, x + r + 1)
            y0 = max(0, y - r)
            y1 = min(image_size, y + r + 1)
            img[y0:y1, x0:x1] = c

    return img


def rasterize_gray(px, py, image_size, radius):
    """point를 binary 이미지로 rasterize."""
    img = np.zeros((image_size, image_size), dtype=np.uint8)

    for x, y in zip(px, py):
        x = int(x)
        y = int(y)

        if x < 0 or x >= image_size or y < 0 or y >= image_size:
            continue

        if cv2 is not None:
            cv2.circle(img, (x, y), int(radius), 255, -1)
        else:
            r = int(radius)
            x0 = max(0, x - r)
            x1 = min(image_size, x + r + 1)
            y0 = max(0, y - r)
            y1 = min(image_size, y + r + 1)
            img[y0:y1, x0:x1] = 255

    return img


def select_points_by_mode(P, depth, args):
    """
    투영에 사용할 point 선택.
    기본값 all: 자동으로 자르지 않고 전체 marker_points를 투영한다.
    """
    if args.use_points == "all":
        return np.ones(len(P), dtype=bool)

    band = float(args.depth_band_m)

    if args.depth_side == "high":
        return depth >= float(depth.max() - band)

    return depth <= float(depth.min() + band)


def project_and_save(P, src_file, out_dir, front, up, lookat, args):
    """
    현재 Open3D view 방향으로 3D point를 직교투영하고 저장한다.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    images_color_dir = out_dir / "images_color"
    images_color_dir.mkdir(parents=True, exist_ok=True)

    # Open3D camera 방향 사용
    front = unit(front)

    # up이 front와 정확히 직교하도록 보정
    up = np.asarray(up, dtype=np.float64)
    up = up - np.dot(up, front) * front
    up = unit(up)

    # 화면 오른쪽 방향
    right = unit(np.cross(front, up))

    # origin은 사용자가 보고 있는 lookat을 우선 사용
    origin = np.asarray(lookat, dtype=np.float64)

    if not np.all(np.isfinite(origin)):
        origin = P.mean(axis=0)

    # 3D point를 사용자가 맞춘 화면 좌표계로 투영
    X = P - origin[None, :]
    u = X @ right
    v = X @ up
    depth = X @ front

    uv_all = np.stack([u, v], axis=1)

    mask = select_points_by_mode(P, depth, args)

    if int(mask.sum()) < 3:
        raise RuntimeError("선택된 point가 너무 적음. --use_points all로 실행 필요")

    P_sel = P[mask]
    uv_sel = uv_all[mask]

    # 왜곡 방지: u/v를 서로 다른 비율로 늘리지 않고 정사각형 스케일 사용
    umin, vmin = uv_all.min(axis=0)
    umax, vmax = uv_all.max(axis=0)

    uc = 0.5 * (umin + umax)
    vc = 0.5 * (vmin + vmax)

    span_u = float(umax - umin)
    span_v = float(vmax - vmin)
    span = max(span_u, span_v, 1e-6) * float(args.view_margin)

    u_min = uc - span / 2.0
    u_max = uc + span / 2.0
    v_min = vc - span / 2.0
    v_max = vc + span / 2.0

    image_size = int(args.image_size)

    px = np.round((uv_sel[:, 0] - u_min) / span * (image_size - 1)).astype(np.int32)
    py = np.round((v_max - uv_sel[:, 1]) / span * (image_size - 1)).astype(np.int32)

    valid = (px >= 0) & (px < image_size) & (py >= 0) & (py < image_size)

    px = px[valid]
    py = py[valid]
    P_valid = P_sel[valid]
    uv_valid = uv_sel[valid]

    rgb = jet_color(P_valid[:, 2])

    color_img = rasterize_rgb(px, py, rgb, image_size, int(args.point_radius))
    binary_img = rasterize_gray(px, py, image_size, int(args.point_radius))

    stem = src_file.stem

    out_top_xyz = out_dir / f"{stem}_top_id.xyz"
    out_top_uv = out_dir / f"{stem}_top_id_uv.npy"
    out_all_uv = out_dir / f"{stem}_marker_all_uv.npy"
    out_color = out_dir / f"{stem}_color.png"
    out_binary = out_dir / f"{stem}_binary.png"
    out_meta = out_dir / f"{stem}_meta.json"

    save_xyz(out_top_xyz, P_valid)
    np.save(str(out_top_uv), uv_valid.astype(np.float32))
    np.save(str(out_all_uv), uv_all.astype(np.float32))

    save_png_rgb(out_color, color_img)
    save_png_gray(out_binary, binary_img)

    shutil.copy2(out_color, images_color_dir / out_color.name)

    meta = {
        "source_file": str(src_file),
        "projection_mode": "manual_open3d_view",
        "capture_image": False,
        "note": "Open3D 화면 캡처가 아니라, 사용자가 맞춘 camera front/up 방향으로 3D point를 직교투영함.",

        "use_points": str(args.use_points),
        "depth_side": str(args.depth_side),
        "depth_band_m": float(args.depth_band_m),

        "image_size": int(args.image_size),
        "point_radius": int(args.point_radius),
        "view_margin": float(args.view_margin),

        "manual_origin_3d": origin.tolist(),
        "manual_front_3d": front.tolist(),
        "manual_up_3d": up.tolist(),
        "manual_right_3d": right.tolist(),

        "image_u_axis_3d": right.tolist(),
        "image_v_axis_3d": up.tolist(),
        "view_direction_3d": front.tolist(),

        "u_min": float(u_min),
        "u_max": float(u_max),
        "v_min": float(v_min),
        "v_max": float(v_max),
        "pixel_size_u_m": float(span / max(image_size - 1, 1)),
        "pixel_size_v_m": float(span / max(image_size - 1, 1)),

        "total_points": int(len(P)),
        "selected_points": int(len(P_valid)),
    }

    out_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print("")
    print("[SAVED]")
    print("color:", out_color)
    print("images_color:", images_color_dir / out_color.name)
    print("selected_points:", len(P_valid))
    print("")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--file", required=True, help="수동 투영할 *_marker.xyz 파일")
    parser.add_argument("--out", required=True, help="출력 폴더")

    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--point_radius", type=int, default=2)
    parser.add_argument("--view_margin", type=float, default=1.2)

    parser.add_argument(
        "--use_points",
        choices=["all", "depth_band"],
        default="all",
        help="all=전체 marker_points 투영, depth_band=현재 view 방향에서 가까운/먼 층만 투영",
    )

    parser.add_argument(
        "--depth_side",
        choices=["high", "low"],
        default="high",
        help="depth_band 사용 시 high/low 방향 선택",
    )

    parser.add_argument(
        "--depth_band_m",
        type=float,
        default=0.04,
        help="depth_band 사용 시 선택 두께",
    )

    parser.add_argument("--display_point_size", type=float, default=3.0)

    args = parser.parse_args()

    src_file = Path(args.file)
    out_dir = Path(args.out)

    P = load_xyz(src_file)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(P)

    colors = jet_color(P[:, 2]).astype(np.float64) / 255.0
    pcd.colors = o3d.utility.Vector3dVector(colors)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(
        window_name="Manual View Projection - 마우스로 view 맞춘 뒤 S 저장, Q 종료",
        width=1200,
        height=900,
    )

    vis.add_geometry(pcd)

    opt = vis.get_render_option()
    opt.point_size = float(args.display_point_size)
    opt.background_color = np.array([0.0, 0.0, 0.0])

    state = {"saved": False}

    def save_callback(vis_obj):
        vc = vis_obj.get_view_control()

        front = np.asarray(vc.get_front(), dtype=np.float64)
        up = np.asarray(vc.get_up(), dtype=np.float64)
        lookat = np.asarray(vc.get_lookat(), dtype=np.float64)

        project_and_save(
            P=P,
            src_file=src_file,
            out_dir=out_dir,
            front=front,
            up=up,
            lookat=lookat,
            args=args,
        )

        state["saved"] = True
        return False

    def quit_callback(vis_obj):
        vis_obj.close()
        return False

    vis.register_key_callback(ord("S"), save_callback)
    vis.register_key_callback(ord("Q"), quit_callback)

    print("=" * 80)
    print("[MANUAL VIEW PROJECTION]")
    print("파일:", src_file)
    print("출력:", out_dir)
    print("")
    print("마우스로 PointCloud를 원하는 수직 방향으로 맞춘 뒤 S 키를 누르면 저장됨.")
    print("Q 키를 누르면 종료.")
    print("주의: 화면 캡처가 아니라, 현재 view의 front/up 방향으로 3D 좌표를 다시 투영함.")
    print("=" * 80)

    vis.run()
    vis.destroy_window()

    if not state["saved"]:
        print("[INFO] 저장하지 않고 종료됨. S 키를 눌러야 projection이 저장됨.")


if __name__ == "__main__":
    main()
