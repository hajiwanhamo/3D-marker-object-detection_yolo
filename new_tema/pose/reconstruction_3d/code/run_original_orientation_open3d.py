from pathlib import Path
from datetime import datetime
import csv
import math
import argparse
import numpy as np
import open3d as o3d


# ============================================================
# 원본 xyz에 3D 방향추정 결과 적용 후 Open3D 즉시 시각화
# ------------------------------------------------------------
# 입력:
# - reconstruction_3d/output/orientation_3d_reconstruction_*/orientation_3d_reconstruction_results.csv
# - 원본 *.xyz
#
# 출력:
# - reconstruction_3d/output/original_open3d_run_*/ply/*.ply
#
# 실행하면:
# 1. 원본 xyz 자동 탐색
# 2. CENTER/N/E/S/W 방향점 추가한 PLY 생성
# 3. Open3D로 바로 시각화
# ============================================================


POSE_ROOT = Path("/Users/hajiwan/Desktop/object_detection/new_tema/pose")
OBJECT_ROOT = Path("/Users/hajiwan/Desktop/object_detection")
RECON_OUTPUT = POSE_ROOT / "reconstruction_3d" / "output"

SPHERE_POINTS = 400
SPHERE_RADIUS_RATIO = 0.012


def find_latest_direction_csv():
    runs = sorted(RECON_OUTPUT.glob("orientation_3d_reconstruction_*"))

    if not runs:
        raise FileNotFoundError(f"[ERROR] 기존 3D 방향추정 결과 없음: {RECON_OUTPUT}")

    latest = runs[-1]
    csv_path = latest / "orientation_3d_reconstruction_results.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"[ERROR] CSV 없음: {csv_path}")

    return csv_path


def image_name_to_base(image_name):
    stem = Path(image_name).stem

    if stem.endswith("_color"):
        stem = stem[:-len("_color")]

    return stem


def find_original_xyz(base, root):
    candidates = []

    for p in root.rglob("*.xyz"):
        s = str(p)

        if base not in p.name:
            continue

        if "_top_id.xyz" in p.name:
            continue

        if "_selected.xyz" in p.name:
            continue

        if "direction" in p.name:
            continue

        if "reconstruction_3d" in s:
            continue

        candidates.append(p)

    def score(p):
        s = str(p).lower()
        v = 0

        if "complete_denoise" in s:
            v -= 20
        if "realdata" in s:
            v -= 10
        if "raw" in s:
            v -= 5
        if "flat" in s:
            v += 20
        if "selected" in s:
            v += 50
        if "top_id" in s:
            v += 100

        return v

    candidates = sorted(candidates, key=score)

    if not candidates:
        return None

    return candidates[0]


def load_xyz(path):
    xyz = np.loadtxt(path).astype(np.float64)

    if xyz.ndim == 1:
        xyz = xyz.reshape(1, 3)

    return xyz[:, :3]


def parse_direction(row):
    keys = {
        "CENTER": ("center_x", "center_y", "center_z"),
        "N": ("north_x", "north_y", "north_z"),
        "E": ("east_x", "east_y", "east_z"),
        "S": ("south_x", "south_y", "south_z"),
        "W": ("west_x", "west_y", "west_z"),
    }

    out = {}

    for name, ks in keys.items():
        vals = []

        for k in ks:
            v = row.get(k, "")

            if v is None or v == "":
                return None

            vals.append(float(v))

        out[name] = np.asarray(vals, dtype=np.float64)

    return out


def make_sphere(center, radius, n):
    pts = []
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))

    for i in range(n):
        z = 1.0 - 2.0 * i / max(n - 1, 1)
        r = math.sqrt(max(0.0, 1.0 - z * z))
        theta = golden_angle * i

        x = math.cos(theta) * r
        y = math.sin(theta) * r

        pts.append(center + radius * np.array([x, y, z], dtype=np.float64))

    return np.asarray(pts, dtype=np.float64)


def write_ply(path, original_xyz, direction_points):
    vertices = []
    colors = []

    # 원본 point cloud: 회색
    for p in original_xyz:
        vertices.append(p)
        colors.append((150, 150, 150))

    mn = original_xyz.min(axis=0)
    mx = original_xyz.max(axis=0)
    diag = float(np.linalg.norm(mx - mn))
    radius = max(diag * SPHERE_RADIUS_RATIO, 1e-5)

    color_map = {
        "CENTER": (255, 255, 255),
        "N": (255, 0, 0),
        "E": (0, 255, 0),
        "S": (0, 0, 255),
        "W": (255, 255, 0),
    }

    for key in ["CENTER", "N", "E", "S", "W"]:
        sphere = make_sphere(direction_points[key], radius, SPHERE_POINTS)

        for p in sphere:
            vertices.append(p)
            colors.append(color_map[key])

    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for p, c in zip(vertices, colors):
            f.write(f"{p[0]:.8f} {p[1]:.8f} {p[2]:.8f} {c[0]} {c[1]} {c[2]}\n")


def visualize_ply(ply_path):
    pcd = o3d.io.read_point_cloud(str(ply_path))

    if pcd.is_empty():
        print(f"[WARN] 빈 PLY: {ply_path}")
        return

    bbox = pcd.get_axis_aligned_bounding_box()
    extent = np.linalg.norm(bbox.get_extent())

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=max(extent * 0.15, 0.05),
        origin=bbox.get_center()
    )

    print(f"[OPEN3D] {ply_path.name}")
    print("색상: CENTER=흰색, N=빨강, E=초록, S=파랑, W=노랑")

    o3d.visualization.draw_geometries(
        [pcd, frame],
        window_name=ply_path.name,
        width=1200,
        height=900
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_root", type=str, default=str(OBJECT_ROOT))
    parser.add_argument("--name", type=str, default="", help="특정 파일명 일부만 시각화")
    parser.add_argument("--all", action="store_true", help="전체 결과 순서대로 시각화")
    args = parser.parse_args()

    original_root = Path(args.original_root)
    csv_path = find_latest_direction_csv()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = RECON_OUTPUT / f"original_open3d_run_{ts}"
    ply_dir = out_dir / "ply"
    ply_dir.mkdir(parents=True, exist_ok=False)

    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8")))

    made = []

    print(f"[CSV] {csv_path}")
    print(f"[ORIGINAL_ROOT] {original_root}")
    print(f"[OUT_DIR] {out_dir}")

    for row in rows:
        if row.get("reconstruct_status", "") != "OK":
            continue

        image_name = row["image_name"]
        base = image_name_to_base(image_name)

        if args.name and args.name not in base:
            continue

        direction = parse_direction(row)

        if direction is None:
            print(f"[SKIP] 방향 좌표 없음: {base}")
            continue

        original_xyz_path = find_original_xyz(base, original_root)

        if original_xyz_path is None:
            print(f"[MISS] 원본 xyz 없음: {base}")
            continue

        original_xyz = load_xyz(original_xyz_path)

        ply_path = ply_dir / f"{base}_original_orientation_3d.ply"
        write_ply(ply_path, original_xyz, direction)

        made.append(ply_path)

        print(f"[OK] {base}")
        print(f"     original = {original_xyz_path}")
        print(f"     ply      = {ply_path}")

    print(f"[DONE] made={len(made)}")

    if not made:
        print("[STOP] 시각화할 PLY 없음")
        return

    if args.all:
        for p in made:
            visualize_ply(p)
    else:
        visualize_ply(made[0])

    print(f"[DONE] output_dir = {out_dir}")


if __name__ == "__main__":
    main()
