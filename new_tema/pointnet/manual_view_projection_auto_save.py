#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
manual_view_projection_auto_save.py

- Open3D 창에서 사용자가 직접 point cloud view를 맞춤
- S 키 사용 안 함
- 창을 닫는 순간 마지막 view 방향으로 3D point를 직교투영해서 저장
- 화면 캡처 아님
"""

import argparse
import importlib.util
import time
from pathlib import Path

import numpy as np
import open3d as o3d


BASE_PATH = Path("/Users/hajiwan/Desktop/object_detection/new_tema/pointnet/manual_view_projection.py")


def load_base_module():
    """기존 manual_view_projection.py의 저장/투영 함수들을 재사용한다."""
    spec = importlib.util.spec_from_file_location("manual_view_projection_base", BASE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"기존 manual_view_projection.py 로드 실패: {BASE_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--file", required=True)
    parser.add_argument("--out", required=True)

    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--point_radius", type=int, default=2)
    parser.add_argument("--view_margin", type=float, default=1.2)

    parser.add_argument(
        "--use_points",
        choices=["all", "depth_band"],
        default="all",
    )

    parser.add_argument(
        "--depth_side",
        choices=["high", "low"],
        default="high",
    )

    parser.add_argument("--depth_band_m", type=float, default=0.04)
    parser.add_argument("--display_point_size", type=float, default=3.0)

    args = parser.parse_args()

    base = load_base_module()

    src_file = Path(args.file)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "images_color").mkdir(parents=True, exist_ok=True)

    P = base.load_xyz(src_file)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(P)

    colors = base.jet_color(P[:, 2]).astype(np.float64) / 255.0
    pcd.colors = o3d.utility.Vector3dVector(colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name="Manual View Projection Auto Save - view 맞춘 뒤 창 닫으면 저장",
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
    print("[MANUAL VIEW AUTO SAVE]")
    print("파일:", src_file)
    print("출력:", out_dir)
    print("")
    print("1. Open3D 창에서 마우스로 원하는 수직 방향을 맞춤")
    print("2. 창을 닫으면 마지막 view 방향으로 자동 저장")
    print("3. S 키 필요 없음")
    print("4. 화면 캡처가 아니라 현재 view 방향으로 3D 좌표를 직교투영함")
    print("=" * 80)

    while True:
        alive = vis.poll_events()
        vis.update_renderer()

        try:
            vc = vis.get_view_control()
            front = np.asarray(vc.get_front(), dtype=np.float64)
            up = np.asarray(vc.get_up(), dtype=np.float64)
            lookat = np.asarray(vc.get_lookat(), dtype=np.float64)

            if np.all(np.isfinite(front)) and np.linalg.norm(front) > 1e-9:
                last_view["front"] = front

            if np.all(np.isfinite(up)) and np.linalg.norm(up) > 1e-9:
                last_view["up"] = up

            if np.all(np.isfinite(lookat)):
                last_view["lookat"] = lookat

        except Exception:
            pass

        if not alive:
            break

        time.sleep(0.01)

    vis.destroy_window()

    base.project_and_save(
        P=P,
        src_file=src_file,
        out_dir=out_dir,
        front=last_view["front"],
        up=last_view["up"],
        lookat=last_view["lookat"],
        args=args,
    )


if __name__ == "__main__":
    main()
