from pathlib import Path
import argparse
import sys

import numpy as np


POSE_ROOT = Path("/Users/hajiwan/Desktop/object_detection/new_tema/pose")
RECON_OUT = POSE_ROOT / "reconstruction_3d" / "output"


def find_latest_ply_dir():
    runs = sorted(RECON_OUT.glob("orientation_3d_reconstruction_*"))

    if not runs:
        raise FileNotFoundError(f"[ERROR] 3D 복원 결과 폴더 없음: {RECON_OUT}")

    latest = runs[-1]
    ply_dir = latest / "ply"

    if not ply_dir.exists():
        raise FileNotFoundError(f"[ERROR] PLY 폴더 없음: {ply_dir}")

    return latest, ply_dir


def load_open3d():
    try:
        import open3d as o3d
        return o3d
    except ImportError:
        print("[ERROR] open3d가 설치되어 있지 않음")
        print("설치 명령어:")
        print("pip install open3d")
        sys.exit(1)


def visualize_one(ply_path, point_size=3.0):
    o3d = load_open3d()

    pcd = o3d.io.read_point_cloud(str(ply_path))

    if pcd.is_empty():
        print(f"[WARN] 빈 PLY: {ply_path}")
        return

    pts = np.asarray(pcd.points)
    bbox = pcd.get_axis_aligned_bounding_box()
    extent = np.linalg.norm(bbox.get_extent())
    frame_size = max(extent * 0.25, 0.05)

    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=frame_size,
        origin=bbox.get_center()
    )

    print(f"[OPEN] {ply_path.name}")
    print(f"points: {len(pts)}")
    print("색상 의미: CENTER=흰색, N=빨강, E=초록, S=파랑, W=노랑")

    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name=ply_path.name,
        width=1200,
        height=900
    )

    vis.add_geometry(pcd)
    vis.add_geometry(coord)

    opt = vis.get_render_option()
    opt.point_size = point_size
    opt.background_color = np.asarray([0.02, 0.02, 0.02])

    vis.run()
    vis.destroy_window()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=0, help="열 PLY index")
    parser.add_argument("--name", type=str, default="", help="파일명 일부로 선택")
    parser.add_argument("--all", action="store_true", help="전체 PLY를 순서대로 열기")
    parser.add_argument("--point_size", type=float, default=3.0)
    args = parser.parse_args()

    latest, ply_dir = find_latest_ply_dir()
    ply_files = sorted(ply_dir.glob("*.ply"))

    if not ply_files:
        raise FileNotFoundError(f"[ERROR] PLY 파일 없음: {ply_dir}")

    print(f"[LATEST] {latest}")
    print(f"[PLY_DIR] {ply_dir}")
    print(f"[PLY COUNT] {len(ply_files)}")

    if args.name:
        selected = [p for p in ply_files if args.name in p.name]

        if not selected:
            raise FileNotFoundError(f"[ERROR] name='{args.name}'에 해당하는 PLY 없음")

        for p in selected:
            visualize_one(p, args.point_size)
        return

    if args.all:
        for p in ply_files:
            visualize_one(p, args.point_size)
        return

    if args.index < 0 or args.index >= len(ply_files):
        raise IndexError(f"[ERROR] index 범위 초과: 0 ~ {len(ply_files)-1}")

    visualize_one(ply_files[args.index], args.point_size)


if __name__ == "__main__":
    main()
