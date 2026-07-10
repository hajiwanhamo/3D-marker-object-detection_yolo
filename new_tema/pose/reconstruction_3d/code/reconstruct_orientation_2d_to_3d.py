from pathlib import Path
from datetime import datetime
import csv
import json
import math

import numpy as np


# ============================================================
# 2D 방향추정 결과를 3D point cloud 좌표로 복원하는 코드
# ------------------------------------------------------------
# 입력:
# - estimation/output/.../orientation_results_conf040_template2d_imgcenter.csv
# - realdata/01_down/*_top_id_uv.npy
# - realdata/01_down/*_top_id.xyz
# - realdata/01_down/*_meta.json
#
# 출력:
# - 3D CENTER/N/E/S/W CSV
# - 각 데이터별 PLY 시각화 파일
# - 각 데이터별 3D 방향점 xyz 파일
#
# 핵심:
# - 방향추정은 다시 하지 않음
# - 이미 계산된 2D CENTER/N/E/S/W를 3D로 복원만 함
# ============================================================


POSE_ROOT = Path("/Users/hajiwan/Desktop/object_detection/new_tema/pose")
NEWTEMA_ROOT = Path("/Users/hajiwan/Desktop/object_detection/new_tema")

ESTIMATION_DIR = POSE_ROOT / "estimation"
REALDATA_DIR = NEWTEMA_ROOT / "yolov2/realdata/range_sweep_down_10sets/01_down"

OUTPUT_ROOT = POSE_ROOT / "reconstruction_3d" / "output"

# PLY에서 방향점 구 크기
SPHERE_RADIUS_RATIO = 0.018
SPHERE_POINTS = 180


def find_latest_2d_result_csv():
    """
    estimation/output 안에서 가장 최신 imgcenter 결과 CSV를 찾는다.
    """
    candidates = sorted(
        ESTIMATION_DIR.glob(
            "output/orientation_output_01_down_conf040_template2d_imgcenter_*/orientation_results_conf040_template2d_imgcenter.csv"
        )
    )

    if not candidates:
        raise FileNotFoundError(
            "[ERROR] imgcenter 2D 결과 CSV를 찾지 못함: "
            f"{ESTIMATION_DIR}/output"
        )

    return candidates[-1]


def image_name_to_real_base(image_name):
    """
    예:
    1_19_2026_162815_complete_denoise_marker_color.jpg
    -> 1_19_2026_162815_complete_denoise_marker
    """
    stem = Path(image_name).stem

    if stem.endswith("_color"):
        stem = stem[:-len("_color")]

    return stem


def load_meta(real_base):
    """
    실해역 투영 meta json 로드.
    """
    p = REALDATA_DIR / f"{real_base}_meta.json"

    if not p.exists():
        raise FileNotFoundError(f"[ERROR] meta json 없음: {p}")

    with p.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    return meta, p


def parse_image_size(meta):
    """
    meta에서 image size를 읽는다.
    image_size가 int이면 정사각 이미지로 처리한다.
    """
    if "image_size" in meta:
        s = meta["image_size"]

        if isinstance(s, int):
            return s, s

        if isinstance(s, (list, tuple)) and len(s) == 2:
            return int(s[0]), int(s[1])

    if "W" in meta and "H" in meta:
        return int(meta["W"]), int(meta["H"])

    if "width" in meta and "height" in meta:
        return int(meta["width"]), int(meta["height"])

    # 현재 데이터는 512 기준
    return 512, 512


def parse_uv_bounds(meta, real_base):
    """
    meta에서 uv bounds를 읽는다.

    지원 키:
    - u_min, u_max, v_min, v_max
    - xmin, xmax, ymin, ymax
    - view_bounds dict
    - bounds dict

    meta에 없으면 marker_all_uv.npy로 AABB + margin을 재계산한다.
    """
    direct_sets = [
        ("u_min", "u_max", "v_min", "v_max"),
        ("xmin", "xmax", "ymin", "ymax"),
    ]

    for keys in direct_sets:
        if all(k in meta for k in keys):
            a, b, c, d = keys
            return float(meta[a]), float(meta[b]), float(meta[c]), float(meta[d])

    for parent_key in ["view_bounds", "bounds", "uv_bounds"]:
        if parent_key in meta and isinstance(meta[parent_key], dict):
            b = meta[parent_key]

            if all(k in b for k in ["u_min", "u_max", "v_min", "v_max"]):
                return float(b["u_min"]), float(b["u_max"]), float(b["v_min"]), float(b["v_max"])

            if all(k in b for k in ["xmin", "xmax", "ymin", "ymax"]):
                return float(b["xmin"]), float(b["xmax"]), float(b["ymin"]), float(b["ymax"])

    # fallback: marker_all_uv의 AABB와 view_margin으로 bounds 재계산
    marker_uv_path = REALDATA_DIR / f"{real_base}_marker_all_uv.npy"

    if not marker_uv_path.exists():
        raise FileNotFoundError(
            f"[ERROR] uv bounds를 meta에서 못 읽었고 marker_all_uv도 없음: {marker_uv_path}"
        )

    uv_all = np.load(marker_uv_path).astype(np.float64)
    mn = uv_all.min(axis=0)
    mx = uv_all.max(axis=0)

    center = 0.5 * (mn + mx)
    half = 0.5 * (mx - mn)

    margin = float(meta.get("view_margin", 1.2))
    half = half * margin

    u_min = center[0] - half[0]
    u_max = center[0] + half[0]
    v_min = center[1] - half[1]
    v_max = center[1] + half[1]

    return float(u_min), float(u_max), float(v_min), float(v_max)


def pixel_to_uv(px, py, W, H, bounds):
    """
    2D pixel 좌표를 투영 uv 좌표로 역변환한다.

    projection:
      px = (u - u_min) / (u_max - u_min) * (W - 1)
      py = (v_max - v) / (v_max - v_min) * (H - 1)
    """
    u_min, u_max, v_min, v_max = bounds

    u = (px / max(W - 1, 1)) * (u_max - u_min) + u_min
    v = v_max - (py / max(H - 1, 1)) * (v_max - v_min)

    return np.array([u, v], dtype=np.float64)


def fit_uv_to_xyz_affine(top_uv, top_xyz):
    """
    top_id_uv와 top_id.xyz의 1:1 대응을 이용해
    uv -> xyz affine mapping을 추정한다.

    xyz = [u, v, 1] @ B
    """
    if len(top_uv) != len(top_xyz):
        raise ValueError(f"[ERROR] top_uv/top_xyz 길이 불일치: {len(top_uv)} vs {len(top_xyz)}")

    A = np.c_[top_uv, np.ones(len(top_uv))]
    B, residuals, rank, s = np.linalg.lstsq(A, top_xyz, rcond=None)

    pred = A @ B
    err = np.linalg.norm(pred - top_xyz, axis=1)

    return {
        "B": B,
        "fit_error_mean": float(err.mean()),
        "fit_error_median": float(np.median(err)),
        "fit_error_p95": float(np.percentile(err, 95)),
        "fit_error_max": float(err.max()),
        "rank": int(rank),
    }


def uv_to_xyz(uv, affine):
    """
    uv 좌표 하나를 xyz로 복원한다.
    """
    A = np.array([uv[0], uv[1], 1.0], dtype=np.float64)
    return A @ affine["B"]


def nearest_uv_distance(uv, top_uv):
    """
    복원 대상 uv가 top_id_uv 영역과 얼마나 가까운지 확인.
    """
    d = np.linalg.norm(top_uv - uv.reshape(1, 2), axis=1)
    return float(d.min())


def make_sphere_points(center, radius, n=180):
    """
    PLY 시각화용 작은 구 point 생성.
    """
    pts = []

    golden_angle = math.pi * (3.0 - math.sqrt(5.0))

    for i in range(n):
        y = 1.0 - (i / max(n - 1, 1)) * 2.0
        r = math.sqrt(max(0.0, 1.0 - y * y))
        theta = golden_angle * i

        x = math.cos(theta) * r
        z = math.sin(theta) * r

        pts.append(center + radius * np.array([x, y, z], dtype=np.float64))

    return np.asarray(pts, dtype=np.float64)


def write_ply(path, base_xyz, direction_points):
    """
    top_id point cloud + CENTER/N/E/S/W 구를 PLY로 저장.
    """
    base_xyz = np.asarray(base_xyz, dtype=np.float64)

    mn = base_xyz.min(axis=0)
    mx = base_xyz.max(axis=0)
    diag = float(np.linalg.norm(mx - mn))
    radius = max(diag * SPHERE_RADIUS_RATIO, 1e-4)

    vertices = []
    colors = []

    # 원본 top_id point cloud: 회색
    for p in base_xyz:
        vertices.append(p)
        colors.append((150, 150, 150))

    color_map = {
        "CENTER": (255, 255, 255),
        "N": (255, 0, 0),
        "E": (0, 255, 0),
        "S": (0, 0, 255),
        "W": (255, 255, 0),
    }

    # 방향점: 색상 구
    for key, xyz in direction_points.items():
        sphere = make_sphere_points(np.asarray(xyz, dtype=np.float64), radius, SPHERE_POINTS)
        color = color_map[key]

        for p in sphere:
            vertices.append(p)
            colors.append(color)

    vertices = np.asarray(vertices, dtype=np.float64)

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


def write_direction_xyz(path, direction_points):
    """
    CENTER/N/E/S/W만 xyz 텍스트로 저장.
    """
    with path.open("w", encoding="utf-8") as f:
        for key in ["CENTER", "N", "E", "S", "W"]:
            p = direction_points[key]
            f.write(f"{key} {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n")


def parse_float(row, key):
    v = row.get(key, "")

    if v is None or v == "":
        return None

    return float(v)


def main():
    result_csv = find_latest_2d_result_csv()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_ROOT / f"orientation_3d_reconstruction_{ts}"
    ply_dir = out_dir / "ply"
    xyz_dir = out_dir / "direction_xyz"

    out_dir.mkdir(parents=True, exist_ok=False)
    ply_dir.mkdir(parents=True, exist_ok=False)
    xyz_dir.mkdir(parents=True, exist_ok=False)

    print(f"[INFO] 2D result CSV = {result_csv}")
    print(f"[INFO] REALDATA_DIR  = {REALDATA_DIR}")
    print(f"[INFO] OUTPUT_DIR    = {out_dir}")

    rows = list(csv.DictReader(result_csv.open("r", encoding="utf-8")))

    summary_csv = out_dir / "orientation_3d_reconstruction_results.csv"

    fieldnames = [
        "image_name",
        "status_2d",
        "reconstruct_status",
        "reason",
        "center_x", "center_y", "center_z",
        "north_x", "north_y", "north_z",
        "east_x", "east_y", "east_z",
        "south_x", "south_y", "south_z",
        "west_x", "west_y", "west_z",
        "uv_center_u", "uv_center_v",
        "uv_n_u", "uv_n_v",
        "uv_e_u", "uv_e_v",
        "uv_s_u", "uv_s_v",
        "uv_w_u", "uv_w_v",
        "nearest_uv_center",
        "nearest_uv_n",
        "nearest_uv_e",
        "nearest_uv_s",
        "nearest_uv_w",
        "affine_fit_mean",
        "affine_fit_p95",
        "ply_path",
        "xyz_path",
    ]

    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            image_name = row["image_name"]
            real_base = image_name_to_real_base(image_name)

            out_row = {
                "image_name": image_name,
                "status_2d": row.get("status", ""),
                "reconstruct_status": "",
                "reason": "",
                "ply_path": "",
                "xyz_path": "",
            }

            try:
                # 실패한 2D 방향추정 결과는 3D 복원하지 않음
                required = [
                    "fixed_center_x", "fixed_center_y",
                    "north_x", "north_y",
                    "east_x", "east_y",
                    "south_x", "south_y",
                    "west_x", "west_y",
                ]

                if any(parse_float(row, k) is None for k in required):
                    out_row["reconstruct_status"] = "SKIP_NO_2D_DIRECTION"
                    out_row["reason"] = "2D CENTER/N/E/S/W 좌표 없음"
                    writer.writerow(out_row)
                    continue

                meta, meta_path = load_meta(real_base)
                W, H = parse_image_size(meta)
                bounds = parse_uv_bounds(meta, real_base)

                top_uv_path = REALDATA_DIR / f"{real_base}_top_id_uv.npy"
                top_xyz_path = REALDATA_DIR / f"{real_base}_top_id.xyz"

                if not top_uv_path.exists():
                    raise FileNotFoundError(f"top_id_uv 없음: {top_uv_path}")

                if not top_xyz_path.exists():
                    raise FileNotFoundError(f"top_id.xyz 없음: {top_xyz_path}")

                top_uv = np.load(top_uv_path).astype(np.float64)
                top_xyz = np.loadtxt(top_xyz_path).astype(np.float64)

                if top_xyz.ndim == 1:
                    top_xyz = top_xyz.reshape(1, 3)

                affine = fit_uv_to_xyz_affine(top_uv, top_xyz)

                points_2d = {
                    "CENTER": (parse_float(row, "fixed_center_x"), parse_float(row, "fixed_center_y")),
                    "N": (parse_float(row, "north_x"), parse_float(row, "north_y")),
                    "E": (parse_float(row, "east_x"), parse_float(row, "east_y")),
                    "S": (parse_float(row, "south_x"), parse_float(row, "south_y")),
                    "W": (parse_float(row, "west_x"), parse_float(row, "west_y")),
                }

                points_uv = {}
                points_xyz = {}
                nearest_dist = {}

                for key, (px, py) in points_2d.items():
                    uv = pixel_to_uv(px, py, W, H, bounds)
                    xyz = uv_to_xyz(uv, affine)

                    points_uv[key] = uv
                    points_xyz[key] = xyz
                    nearest_dist[key] = nearest_uv_distance(uv, top_uv)

                ply_path = ply_dir / f"{real_base}_orientation_3d.ply"
                xyz_path = xyz_dir / f"{real_base}_direction_points.xyz"

                write_ply(ply_path, top_xyz, points_xyz)
                write_direction_xyz(xyz_path, points_xyz)

                out_row.update({
                    "reconstruct_status": "OK",
                    "reason": "",
                    "center_x": points_xyz["CENTER"][0],
                    "center_y": points_xyz["CENTER"][1],
                    "center_z": points_xyz["CENTER"][2],
                    "north_x": points_xyz["N"][0],
                    "north_y": points_xyz["N"][1],
                    "north_z": points_xyz["N"][2],
                    "east_x": points_xyz["E"][0],
                    "east_y": points_xyz["E"][1],
                    "east_z": points_xyz["E"][2],
                    "south_x": points_xyz["S"][0],
                    "south_y": points_xyz["S"][1],
                    "south_z": points_xyz["S"][2],
                    "west_x": points_xyz["W"][0],
                    "west_y": points_xyz["W"][1],
                    "west_z": points_xyz["W"][2],
                    "uv_center_u": points_uv["CENTER"][0],
                    "uv_center_v": points_uv["CENTER"][1],
                    "uv_n_u": points_uv["N"][0],
                    "uv_n_v": points_uv["N"][1],
                    "uv_e_u": points_uv["E"][0],
                    "uv_e_v": points_uv["E"][1],
                    "uv_s_u": points_uv["S"][0],
                    "uv_s_v": points_uv["S"][1],
                    "uv_w_u": points_uv["W"][0],
                    "uv_w_v": points_uv["W"][1],
                    "nearest_uv_center": nearest_dist["CENTER"],
                    "nearest_uv_n": nearest_dist["N"],
                    "nearest_uv_e": nearest_dist["E"],
                    "nearest_uv_s": nearest_dist["S"],
                    "nearest_uv_w": nearest_dist["W"],
                    "affine_fit_mean": affine["fit_error_mean"],
                    "affine_fit_p95": affine["fit_error_p95"],
                    "ply_path": str(ply_path),
                    "xyz_path": str(xyz_path),
                })

                writer.writerow(out_row)

            except Exception as e:
                out_row["reconstruct_status"] = "ERROR"
                out_row["reason"] = str(e)
                writer.writerow(out_row)
                print(f"[ERROR] {image_name}: {e}")

    print("[DONE] 3D reconstruction 완료")
    print(f"[DONE] OUTPUT_DIR: {out_dir}")
    print(f"[DONE] PLY_DIR: {ply_dir}")
    print(f"[DONE] XYZ_DIR: {xyz_dir}")
    print(f"[DONE] CSV: {summary_csv}")


if __name__ == "__main__":
    main()
