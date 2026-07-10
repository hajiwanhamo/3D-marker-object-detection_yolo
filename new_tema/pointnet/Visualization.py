# import os, numpy as np, open3d as o3d

# POINT_SIZE = 5.0
# SEPARATION = 0.0
# AXIS_SCALE_RATIO = 0.15

# def visualize_xyz(file_path):
#     try:
#         raw = np.loadtxt(file_path).astype(np.float64)
#         pts = raw[:, :3]
#         pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))

#         # 색상 보존: XYZRGB(.xyz에 6열 이상)인 경우만 적용
#         if raw.shape[1] >= 6:
#             rgb = raw[:, 3:6]
#             if rgb.max() > 1.0:  # 0~255인 경우 정규화
#                 rgb = rgb / 255.0
#             pc.colors = o3d.utility.Vector3dVector(rgb)

#         if SEPARATION:
#             pc = pc.voxel_down_sample(float(SEPARATION))

#         center = np.asarray(pc.points).mean(axis=0)
#         minb, maxb = np.asarray(pc.points).min(axis=0), np.asarray(pc.points).max(axis=0)
#         extent = np.linalg.norm(maxb - minb)
#         axis_len = max(extent * AXIS_SCALE_RATIO, 1e-3)

#         origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_len)

#         # 콘솔에 좌표 텍스트 출력
#         print(f"[파일] {os.path.basename(file_path)}")
#         print(f"Origin = (0.000000, 0.000000, 0.000000)")
#         print(f"Center = ({center[0]:.6f}, {center[1]:.6f}, {center[2]:.6f})")

#         # 좌표 텍스트 파일로도 저장
#         with open(file_path + ".coords.txt", "w", encoding="utf-8") as f:
#             f.write(f"Origin = (0.000000, 0.000000, 0.000000)\n")
#             f.write(f"Center = ({center[0]:.6f}, {center[1]:.6f}, {center[2]:.6f})\n")

#         # 시각화(색상 유지)
#         vis = o3d.visualization.Visualizer()
#         title = f"시각화: {os.path.basename(file_path)} | Center=({center[0]:.3f},{center[1]:.3f},{center[2]:.3f})"
#         vis.create_window(title)
#         vis.add_geometry(pc)
#         vis.add_geometry(origin_frame)
#         vis.get_render_option().point_size = float(POINT_SIZE)

#         ctr = vis.get_view_control()
#         ctr.set_lookat(center.tolist()); ctr.set_up([0,0,1]); ctr.set_front([0,-1,0]); ctr.set_zoom(0.8)

#         vis.run(); vis.destroy_window()

#     except Exception as e:
#         print(f"[오류] {file_path} 시각화 실패: {e}")

# # output_folder = r"C:\Users\yncit\OneDrive\Desktop\jiwan\3D-Marker-Pose-Estimation\CAD_to_PointCloud_conversion\dataset\joint\train\points"
# # for file_name in os.listdir(output_folder):
# #     if file_name.lower().endswith(".xyz"):
# #         visualize_xyz(os.path.join(output_folder, file_name))
#     output_folder = r"C:\Users\yncit\OneDrive\Desktop\jiwan\3D-Marker-Pose-Estimation\CAD_to_PointCloud_conversion\dataset\output"
#     for file_name in os.listdir(output_folder):
#         if file_name.lower().endswith(".xyz"):
#             visualize_xyz(os.path.join(output_folder, file_name))
##--------------------------------------------------------------##
# # - mask.npy 있으면: 배경(0) 점들로 plane fit
# # - mask.npy 없으면: z 하위 q% 점들로 plane fit
# # - (핵심) 색상은 z가 아니라 "plane까지의 signed distance"로 칠함(마커/배경 동일 기준)
# # - jet 컬러맵은 전체 점의 dist 범위를 공유(따로 정규화 X)

# import os, argparse
# import numpy as np
# import open3d as o3d

# POINT_SIZE = 3.0
# AXIS_SCALE_RATIO = 0.15

# def jet01(z: np.ndarray) -> np.ndarray:
#     z = np.asarray(z, dtype=np.float64)
#     if z.size == 0:
#         return np.zeros((0,3), dtype=np.float64)
#     rng = np.ptp(z) + 1e-12
#     z = (z - z.min()) / rng
#     r = np.clip(1.5 - np.abs(4*z - 3), 0, 1)
#     g = np.clip(1.5 - np.abs(4*z - 2), 0, 1)
#     b = np.clip(1.5 - np.abs(4*z - 1), 0, 1)
#     return np.stack([r,g,b], 1)

# def fit_plane_o3d(P: np.ndarray, dist_th=0.01, iters=2000):
#     pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P.astype(np.float64)))
#     plane, inliers = pcd.segment_plane(distance_threshold=float(dist_th), ransac_n=3, num_iterations=int(iters))
#     a,b,c,d = plane
#     n = np.array([a,b,c], dtype=np.float64)
#     nn = np.linalg.norm(n) + 1e-12
#     n = n / nn
#     d = d / nn
#     return n, d, inliers

# def signed_dist(P: np.ndarray, n: np.ndarray, d: float) -> np.ndarray:
#     # ax+by+cz+d = 0, with ||n||=1
#     return (P @ n + d)

# def to_frame(P: np.ndarray):
#     center = P.mean(0)
#     minb, maxb = P.min(0), P.max(0)
#     extent = np.linalg.norm(maxb - minb)
#     axis_len = max(extent * AXIS_SCALE_RATIO, 1e-3)
#     return center, o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_len)

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--xyz", required=True)
#     ap.add_argument("--mask", default="")
#     ap.add_argument("--plane_th", type=float, default=0.01)
#     ap.add_argument("--plane_iters", type=int, default=2000)
#     ap.add_argument("--fallback_low_q", type=float, default=10.0, help="mask 없을 때 z하위 q%로 plane fit")
#     ap.add_argument("--clip_q_lo", type=float, default=1.0, help="dist 컬러 클리핑 하위 분위")
#     ap.add_argument("--clip_q_hi", type=float, default=99.0, help="dist 컬러 클리핑 상위 분위")
#     args = ap.parse_args()

#     raw = np.loadtxt(args.xyz, dtype=np.float64)
#     P = raw[:, :3].astype(np.float64)

#     # plane fit용 점 선택
#     fitP = None
#     if args.mask and os.path.isfile(args.mask):
#         M = np.load(args.mask).ravel().astype(np.int64)
#         if len(M) == len(P) and np.any(M == 0):
#             fitP = P[M == 0]
#         else:
#             fitP = None

#     if fitP is None:
#         z = P[:,2]
#         q = float(args.fallback_low_q)
#         zcut = np.percentile(z, q)
#         fitP = P[z <= zcut]

#     # plane fit
#     n, d, _ = fit_plane_o3d(fitP, dist_th=args.plane_th, iters=args.plane_iters)

#     # 전체 점에 대해 plane-distance
#     dist = signed_dist(P, n, d)

#     # 컬러는 전체 dist 범위를 공유 (분위수로 클리핑해서 outlier 영향 제거)
#     lo = np.percentile(dist, float(args.clip_q_lo))
#     hi = np.percentile(dist, float(args.clip_q_hi))
#     dist_clip = np.clip(dist, lo, hi)
#     colors = jet01(dist_clip)

#     pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
#     pcd.colors = o3d.utility.Vector3dVector(colors)

#     center, frame = to_frame(P)
#     plane_center = center - (dist.mean()) * n  # 대충 시각화용
#     plane_mesh = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1e-6)
#     plane_mesh.translate(plane_center - plane_mesh.get_center())
#     plane_mesh.paint_uniform_color([0.2,0.2,0.2])

#     print(f"[file] {args.xyz}")
#     print(f"[plane] n={n.tolist()} d={float(d):.6f}")
#     print(f"[dist] lo={float(lo):.6f} hi={float(hi):.6f}")

#     vis = o3d.visualization.Visualizer()
#     vis.create_window(f"plane-distance: {os.path.basename(args.xyz)}")
#     vis.add_geometry(pcd)
#     vis.add_geometry(frame)
#     opt = vis.get_render_option()
#     opt.point_size = float(POINT_SIZE)
#     opt.background_color = np.array([1,1,1], dtype=np.float64)

#     ctr = vis.get_view_control()
#     ctr.set_lookat(center.tolist()); ctr.set_up([0,0,1]); ctr.set_front([0,-1,0]); ctr.set_zoom(0.8)

#     vis.run()
#     vis.destroy_window()

# if __name__ == "__main__":
#     main()
# ##--------------------------------------------------------------##
import os, argparse, glob
import numpy as np
import open3d as o3d

POINT_SIZE = 3.0
AXIS_SCALE_RATIO = 0.15

def jet01(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    if z.size == 0:
        return np.zeros((0,3), dtype=np.float64)
    rng = np.ptp(z) + 1e-12
    z = (z - z.min()) / rng
    r = np.clip(1.5 - np.abs(4*z - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4*z - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4*z - 1), 0, 1)
    return np.stack([r,g,b], 1)

def fit_plane_o3d(P: np.ndarray, dist_th=0.01, iters=2000):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P.astype(np.float64)))
    plane, inliers = pcd.segment_plane(distance_threshold=float(dist_th), ransac_n=3, num_iterations=int(iters))
    a,b,c,d = plane
    n = np.array([a,b,c], dtype=np.float64)
    nn = np.linalg.norm(n) + 1e-12
    n = n / nn
    d = d / nn
    return n, d, inliers

def signed_dist(P: np.ndarray, n: np.ndarray, d: float) -> np.ndarray:
    return (P @ n + d)

def to_frame(P: np.ndarray):
    center = P.mean(0)
    minb, maxb = P.min(0), P.max(0)
    extent = np.linalg.norm(maxb - minb)
    axis_len = max(extent * AXIS_SCALE_RATIO, 1e-3)
    return center, o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_len)

def pick_fit_points(P: np.ndarray, mask_path: str, fallback_low_q: float):
    if mask_path and os.path.isfile(mask_path):
        M = np.load(mask_path).ravel().astype(np.int64)
        if len(M) == len(P) and np.any(M == 0):
            return P[M == 0], mask_path
    z = P[:,2]
    zcut = np.percentile(z, float(fallback_low_q))
    return P[z <= zcut], ""

def visualize_one(xyz_path: str, mask_arg: str, args, vis, first=False):
    raw = np.loadtxt(xyz_path, dtype=np.float64)
    P = raw[:, :3].astype(np.float64)

    mask_path = mask_arg.strip()
    if (not mask_path) or (mask_path.lower() in ("none", "null")):
        auto = os.path.splitext(xyz_path)[0] + ".mask.npy"
        if os.path.isfile(auto):
            mask_path = auto

    fitP, used_mask = pick_fit_points(P, mask_path, args.fallback_low_q)

    n, d, _ = fit_plane_o3d(fitP, dist_th=args.plane_th, iters=args.plane_iters)
    dist = signed_dist(P, n, d)

    lo = np.percentile(dist, float(args.clip_q_lo))
    hi = np.percentile(dist, float(args.clip_q_hi))
    dist_clip = np.clip(dist, lo, hi)
    colors = jet01(dist_clip)

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    pcd.colors = o3d.utility.Vector3dVector(colors)

    center, frame = to_frame(P)

    vis.clear_geometries()
    vis.add_geometry(pcd)
    vis.add_geometry(frame)

    if first:
        opt = vis.get_render_option()
        opt.point_size = float(POINT_SIZE)
        opt.background_color = np.array([1,1,1], dtype=np.float64)

    ctr = vis.get_view_control()
    ctr.set_lookat(center.tolist()); ctr.set_up([0,0,1]); ctr.set_front([0,-1,0]); ctr.set_zoom(0.8)

    print(f"\n[file] {xyz_path}")
    if used_mask:
        print(f"[mask] {used_mask}  (fitP=mask==0)")
    else:
        print(f"[mask] (none)  (fitP=z <= {args.fallback_low_q}%)")
    print(f"[plane] n={n.tolist()} d={float(d):.6f}")
    print(f"[dist] lo={float(lo):.6f} hi={float(hi):.6f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xyz", default="")
    ap.add_argument("--dir", default="", help="폴더 지정 시 *.xyz를 N/P로 넘기며 확인")
    ap.add_argument("--mask", default="")
    ap.add_argument("--plane_th", type=float, default=0.01)
    ap.add_argument("--plane_iters", type=int, default=2000)
    ap.add_argument("--fallback_low_q", type=float, default=10.0)
    ap.add_argument("--clip_q_lo", type=float, default=1.0)
    ap.add_argument("--clip_q_hi", type=float, default=99.0)
    args = ap.parse_args()

    files = []
    if args.dir.strip():
        d = args.dir
        files = sorted(glob.glob(os.path.join(d, "*.xyz")))
        if not files:
            raise SystemExit(f"[error] no *.xyz in dir: {d}")
    else:
        if not args.xyz.strip():
            raise SystemExit("[error] provide --xyz or --dir")
        if not os.path.isfile(args.xyz):
            raise SystemExit(f"[error] xyz not found: {args.xyz}")
        files = [args.xyz]

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("plane-distance viewer")
    idx = 0

    def show(i, first=False):
        visualize_one(files[i], args.mask, args, vis, first=first)
        vis.poll_events(); vis.update_renderer()
        # Open3D VisualizerWithKeyCallback에는 get_window() 없음 → 제목 변경 제거

    def on_next(v):
        nonlocal idx
        idx = (idx + 1) % len(files)
        show(idx, first=False)
        return False

    def on_prev(v):
        nonlocal idx
        idx = (idx - 1 + len(files)) % len(files)
        show(idx, first=False)
        return False

    def on_quit(v):
        v.close()
        return False

    vis.register_key_callback(ord('N'), on_next)
    vis.register_key_callback(ord('P'), on_prev)
    vis.register_key_callback(ord('Q'), on_quit)

    show(idx, first=True)
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    main()
