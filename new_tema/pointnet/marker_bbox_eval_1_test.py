# # marker_bbox_eval_1_test.py
# # legacy eval 복원 + radius_m 직접 지정 버전
# # 핵심:
# # - width_mm / height_mm / margin_mm 제거
# # - --radius_m 으로 후보 반경 직접 지정
# # - 기존 prob > th + z top_q + radius filter + OBB 생성 흐름 유지
# # - marker_points 저장, idx 저장, Open3D 시각화 유지

# import argparse
# from pathlib import Path

# import numpy as np
# import torch
# import open3d as o3d

# try:
#     from marker_train_1_test import Model, DEV
# except Exception:
#     from marker_train_1 import Model, DEV


# POINT_SIZE = 5.0
# AXIS_SCALE_RATIO = 0.15


# def _jet01(z):
#     z = np.asarray(z, dtype=np.float64)
#     rng = np.ptp(z) + 1e-12
#     z = (z - z.min()) / rng
#     r = np.clip(1.5 - np.abs(4 * z - 3), 0, 1)
#     g = np.clip(1.5 - np.abs(4 * z - 2), 0, 1)
#     b = np.clip(1.5 - np.abs(4 * z - 1), 0, 1)
#     return np.stack([r, g, b], axis=1)


# def _normalize_xyz(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
#     C = X.mean(axis=0, keepdims=True)
#     Xc = X - C
#     s = np.linalg.norm(Xc, axis=1).max()
#     s = float(s) if np.isfinite(s) and s > eps else 1.0
#     return Xc / s


# def load_xyz(path: Path):
#     X = np.loadtxt(str(path), dtype=np.float64)
#     if X.ndim == 1:
#         X = X.reshape(-1, 3)
#     return X


# def scan_xyz_files(root: Path):
#     pts_dir = root / "points"
#     data_dir = pts_dir if pts_dir.is_dir() else root
#     files = sorted(data_dir.glob("*.xyz"))

#     if not files:
#         raise SystemExit(f"[FAIL] .xyz 없음: {data_dir}")

#     return files


# def make_pcd(raw_xyz):
#     pts = raw_xyz[:, :3].astype(np.float64)
#     pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))

#     if raw_xyz.shape[1] >= 6:
#         rgb = raw_xyz[:, 3:6].astype(np.float64)
#         if rgb.max() > 1.0:
#             rgb /= 255.0
#         pcd.colors = o3d.utility.Vector3dVector(rgb)
#     else:
#         pcd.colors = o3d.utility.Vector3dVector(_jet01(pts[:, 2]))

#     return pcd


# def to_frame_for_points(pts):
#     center = pts.mean(axis=0)
#     minb = pts.min(axis=0)
#     maxb = pts.max(axis=0)
#     extent = np.linalg.norm(maxb - minb)
#     axis_len = max(extent * AXIS_SCALE_RATIO, 1e-3)

#     frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_len)
#     return center, frame


# def sample_indices(n, npts):
#     if n <= 0:
#         raise RuntimeError("point 개수가 0입니다.")

#     if n >= npts:
#         return np.random.choice(n, npts, replace=False)

#     rep = (npts + n - 1) // n
#     extra = np.random.choice(n, rep * n - n, replace=True)
#     return np.concatenate([np.arange(n), extra])[:npts]


# def infer_prob_once(net, P3, normalize=True):
#     Pin = P3.astype(np.float32, copy=False)

#     if normalize:
#         Pin = _normalize_xyz(Pin).astype(np.float32, copy=False)

#     x = torch.from_numpy(Pin).unsqueeze(0).to(DEV)

#     with torch.inference_mode():
#         logit = net(x)
#         prob = torch.sigmoid(logit).squeeze().detach().cpu().numpy()

#     return prob


# def infer_prob(net, P, npts, normalize=True):
#     N = len(P)
#     idx = sample_indices(N, npts)

#     prob_s = infer_prob_once(
#         net,
#         P[idx].astype(np.float32),
#         normalize=normalize,
#     )

#     prob = np.zeros((N,), dtype=np.float64)
#     np.maximum.at(prob, idx, prob_s.astype(np.float64))

#     return prob, idx


# def create_box_from_points(Pkeep):
#     if Pkeep is None or Pkeep.shape[0] < 4:
#         return None

#     try:
#         box = o3d.geometry.OrientedBoundingBox.create_from_points(
#             o3d.utility.Vector3dVector(Pkeep.astype(np.float64))
#         )
#         box.color = (0.1, 0.8, 0.1)
#         return box

#     except Exception:
#         try:
#             box = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
#                 o3d.utility.Vector3dVector(Pkeep.astype(np.float64))
#             )
#             box.color = (0.1, 0.8, 0.1)
#             return box

#         except Exception:
#             return None


# def get_box_extent(box):
#     if box is None:
#         return None

#     if isinstance(box, o3d.geometry.OrientedBoundingBox):
#         return np.asarray(box.extent, dtype=np.float64)

#     if isinstance(box, o3d.geometry.AxisAlignedBoundingBox):
#         return np.asarray(box.get_extent(), dtype=np.float64)

#     return None


# def save_marker_points(out_dir: Path, stem: str, raw_xyz: np.ndarray, keep_idx: np.ndarray):
#     out_dir.mkdir(parents=True, exist_ok=True)

#     kept = raw_xyz[keep_idx]
#     out_xyz = out_dir / f"{stem}_marker.xyz"

#     np.savetxt(str(out_xyz), kept[:, :3], fmt="%.6f")
#     np.save(str(out_dir / f"{stem}_marker_idx.npy"), keep_idx.astype(np.int64))

#     return out_xyz


# def expand_keep_idx_aabb_xyz(P: np.ndarray, keep_idx: np.ndarray, expand_mm: float):
#     if keep_idx is None or keep_idx.size == 0:
#         return keep_idx, None

#     expand_m = float(expand_mm) / 1000.0

#     if expand_m <= 0:
#         return keep_idx, None

#     Pk = P[keep_idx]

#     mn = Pk.min(axis=0)
#     mx = Pk.max(axis=0)

#     mn2 = mn - expand_m
#     mx2 = mx + expand_m

#     inside = (
#         (P[:, 0] >= mn2[0]) & (P[:, 0] <= mx2[0]) &
#         (P[:, 1] >= mn2[1]) & (P[:, 1] <= mx2[1]) &
#         (P[:, 2] >= mn2[2]) & (P[:, 2] <= mx2[2])
#     )

#     keep2 = np.flatnonzero(inside)

#     info = {
#         "expand_m": float(expand_m),
#         "aabb_min": mn.tolist(),
#         "aabb_max": mx.tolist(),
#         "aabb_min_exp": mn2.tolist(),
#         "aabb_max_exp": mx2.tolist(),
#         "keep_before": int(keep_idx.size),
#         "keep_after": int(keep2.size),
#     }

#     return keep2, info


# def compute_bbox_legacy_radius(
#     P,
#     prob,
#     th,
#     q_floor,
#     top_q,
#     radius_m,
#     min_pts,
# ):
#     """
#     bbox 생성 기준:

#     1. z 하위 q_floor를 바닥 기준으로 설정
#     2. dz = z - z_floor 계산
#     3. dz 상위 top_q 기준과 prob > th를 동시에 만족하는 점 선택
#     4. 선택점의 XY 중앙값 계산
#     5. --radius_m 안쪽 점만 keep
#     6. keep 점들로 OBB 생성
#     """
#     z = P[:, 2]

#     z_floor = float(np.quantile(z, q_floor))
#     dz = z - z_floor
#     dz_thr = float(np.quantile(dz, top_q))

#     sel = (prob > th) & (dz >= dz_thr)
#     sel_idx = np.flatnonzero(sel)

#     if sel_idx.size < min_pts:
#         return None, None, {
#             "mode": "legacy_radius",
#             "stage": "pre",
#             "sel": int(sel_idx.size),
#             "keep": 0,
#             "z_floor": z_floor,
#             "dz_thr": dz_thr,
#             "prob_gt_th": int((prob > th).sum()),
#             "radius_m": float(radius_m),
#         }

#     Psel = P[sel_idx]
#     center_xy = np.median(Psel[:, :2], axis=0)

#     dxy = np.linalg.norm(Psel[:, :2] - center_xy[None, :], axis=1)
#     keep_idx = sel_idx[dxy <= float(radius_m)]

#     if keep_idx.size < min_pts:
#         return None, None, {
#             "mode": "legacy_radius",
#             "stage": "radius",
#             "sel": int(sel_idx.size),
#             "keep": int(keep_idx.size),
#             "radius_m": float(radius_m),
#             "z_floor": z_floor,
#             "dz_thr": dz_thr,
#             "prob_gt_th": int((prob > th).sum()),
#         }

#     Pkeep = P[keep_idx]
#     box = create_box_from_points(Pkeep)

#     if box is None:
#         return None, None, {
#             "mode": "legacy_radius",
#             "stage": "box_fail",
#             "sel": int(sel_idx.size),
#             "keep": int(keep_idx.size),
#             "radius_m": float(radius_m),
#             "z_floor": z_floor,
#             "dz_thr": dz_thr,
#             "prob_gt_th": int((prob > th).sum()),
#         }

#     extent = get_box_extent(box)

#     meta = {
#         "mode": "legacy_radius",
#         "stage": "ok",
#         "sel": int(sel_idx.size),
#         "keep": int(keep_idx.size),
#         "radius_m": float(radius_m),
#         "z_floor": z_floor,
#         "dz_thr": dz_thr,
#         "prob_gt_th": int((prob > th).sum()),
#         "box_extent": extent.tolist() if extent is not None else None,
#     }

#     return box, keep_idx, meta


# def main(args):
#     np.random.seed(args.seed)

#     out_dir = Path(args.out)
#     out_dir.mkdir(parents=True, exist_ok=True)

#     marker_dir = out_dir / "marker_points"

#     ckpt = torch.load(args.ckpt, map_location=DEV)
#     ckpt_meta = ckpt.get("meta", {}) if isinstance(ckpt, dict) else {}

#     model_k = int(ckpt_meta.get("k", 16))

#     try:
#         net = Model(k=model_k).to(DEV).eval()
#     except TypeError:
#         net = Model().to(DEV).eval()

#     state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
#     net.load_state_dict(state)

#     ckpt_norm = ckpt_meta.get("normalize", None)

#     if args.force_no_norm:
#         use_norm = False
#     elif ckpt_norm is None:
#         use_norm = True
#     else:
#         use_norm = bool(ckpt_norm)

#     print(f"[INFO] loaded ckpt: {args.ckpt}  device={DEV.type}")
#     print(f"[INFO] ckpt_meta_normalize={ckpt_norm}  eval_normalize={use_norm}")

#     files = scan_xyz_files(Path(args.data))

#     print(f"[INFO] files={len(files)}  data={args.data}")
#     print(f"[INFO] legacy eval with direct radius_m")
#     print(f"[INFO] npts={args.npts}  th={args.th}  q_floor={args.q_floor}  top_q={args.top_q}")
#     print(f"[INFO] radius_m={args.radius_m}  min_pts={args.min_pts}")
#     print(f"[INFO] save_marker={int(args.save_marker)}  marker_dir={marker_dir}")
#     print(f"[INFO] save_expand_mm={args.save_expand_mm}")

#     cache = []
#     ok = 0
#     saved = 0

#     for fp in files:
#         raw = load_xyz(fp)
#         P = raw[:, :3].astype(np.float64)

#         prob, sampled_idx = infer_prob(
#             net,
#             P,
#             npts=args.npts,
#             normalize=use_norm,
#         )

#         box, keep_idx, meta = compute_bbox_legacy_radius(
#             P=P,
#             prob=prob,
#             th=args.th,
#             q_floor=args.q_floor,
#             top_q=args.top_q,
#             radius_m=args.radius_m,
#             min_pts=args.min_pts,
#         )

#         expand_info = None

#         if box is not None:
#             ok += 1

#             if args.save_marker and keep_idx is not None:
#                 keep_idx_save, expand_info = expand_keep_idx_aabb_xyz(
#                     P,
#                     keep_idx,
#                     args.save_expand_mm,
#                 )
#                 save_marker_points(marker_dir, fp.stem, raw, keep_idx_save)
#                 saved += 1

#         if expand_info is not None:
#             meta = dict(meta)
#             meta["save_expand"] = expand_info

#         sampled_prob = prob[sampled_idx]

#         print(
#             f"[GEN] {fp.name}  "
#             f"bbox={'OK' if box is not None else 'NONE'}  "
#             f"sampled={int(sampled_idx.size)}  "
#             f"prob_min={float(sampled_prob.min()):.4f}  "
#             f"prob_mean={float(sampled_prob.mean()):.4f}  "
#             f"prob_max={float(sampled_prob.max()):.4f}  "
#             f"meta={meta}"
#         )

#         cache.append((fp, box))

#     print(
#         f"[DONE] bbox OK: {ok}/{len(files)}  "
#         f"marker_saved: {saved}/{len(files)} -> {marker_dir if args.save_marker else '(off)'}"
#     )

#     if args.no_vis:
#         print("[INFO] visualization skipped by --no_vis")
#         return

#     vis = o3d.visualization.VisualizerWithKeyCallback()
#     vis.create_window("marker viewer (legacy radius_m)")

#     opt = vis.get_render_option()
#     opt.point_size = float(POINT_SIZE)
#     opt.background_color = np.array([1, 1, 1], dtype=np.float64)

#     i = 0

#     def show(k):
#         fp, box = cache[k]

#         raw = load_xyz(fp)
#         pcd = make_pcd(raw)
#         pts = np.asarray(pcd.points)

#         center, frame = to_frame_for_points(pts)

#         vis.clear_geometries()
#         vis.add_geometry(pcd)
#         vis.add_geometry(frame)

#         if box is not None:
#             vis.add_geometry(box)

#         ctr = vis.get_view_control()
#         ctr.set_lookat(center.tolist())
#         ctr.set_up([0, 0, 1])
#         ctr.set_front([0, -1, 0])
#         ctr.set_zoom(0.8)

#         vis.poll_events()
#         vis.update_renderer()

#         print(f"[VIEW] {k + 1}/{len(cache)}: {fp.name}")

#     def on_next(v):
#         nonlocal i
#         i = (i + 1) % len(cache)
#         show(i)
#         return False

#     def on_prev(v):
#         nonlocal i
#         i = (i - 1 + len(cache)) % len(cache)
#         show(i)
#         return False

#     def on_quit(v):
#         v.close()
#         return False

#     vis.register_key_callback(ord("N"), on_next)
#     vis.register_key_callback(ord("P"), on_prev)
#     vis.register_key_callback(ord("Q"), on_quit)

#     show(i)
#     vis.run()
#     vis.destroy_window()


# if __name__ == "__main__":
#     ap = argparse.ArgumentParser()

#     ap.add_argument("--data", required=True)
#     ap.add_argument("--ckpt", required=True)
#     ap.add_argument("--out", required=True)

#     ap.add_argument("--npts", type=int, default=16384)

#     ap.add_argument("--th", type=float, default=0.5)
#     ap.add_argument("--q_floor", type=float, default=0.05)
#     ap.add_argument("--top_q", type=float, default=0.95)

#     ap.add_argument("--radius_m", type=float, default=1.1)

#     ap.add_argument("--min_pts", type=int, default=150)

#     ap.add_argument("--save_marker", action="store_true")
#     ap.add_argument("--save_expand_mm", type=float, default=0.0)

#     ap.add_argument("--force_no_norm", action="store_true")
#     ap.add_argument("--no_vis", action="store_true")
#     ap.add_argument("--seed", type=int, default=0)

#     args = ap.parse_args()
#     main(args)
    
    
#     #실행명령어
#     #python -u .\marker_bbox_eval_1_test.py --data "C:\Users\gkwld\Desktop\3D-Marker-Pose-Estimation\CAD_to_PointCloud_conversion\dataset\joint\val" --ckpt "C:\Users\gkwld\Desktop\3D-Marker-Pose-Estimation\CAD_to_PointCloud_conversion\pointnet2\result\exp_seg_train1_posw30\best.pth" --out "C:\Users\gkwld\Desktop\3D-Marker-Pose-Estimation\CAD_to_PointCloud_conversion\pointnet2\result\restore_eval_joint_posw30_radius080" --npts 16384 --th 0.45 --q_floor 0.05 --top_q 0.95 --radius_m 0.8 --min_pts 150 --save_marker --save_expand_mm 0   


# marker_bbox_eval_1_test.py
# 수정본
#
# 목적:
# - train1 모델 예측 결과로 마커 후보영역 검출
# - bbox 생성 전에 선택 포인트 주변 범위를 확장하여 마커 전체를 더 안정적으로 포함
# - 기본 bbox는 AABB 방식으로 생성하여 불필요한 기울어짐 방지
#
# 핵심 변경:
# - --box_expand_mm 추가
# - --box_mode aabb/obb 추가
# - bbox 생성에 사용하는 포인트를 keep_idx → expanded_keep_idx로 변경
#
# 색상:
# - 원본 포인트: z값 기반 색상
# - bbox: 초록색

import argparse
from pathlib import Path

import numpy as np
import torch
import open3d as o3d

try:
    from marker_train_1_test import Model, DEV
except Exception:
    from marker_train_1 import Model, DEV


POINT_SIZE = 5.0
AXIS_SCALE_RATIO = 0.15


def _jet01(z):
    z = np.asarray(z, dtype=np.float64)
    rng = np.ptp(z) + 1e-12
    z = (z - z.min()) / rng
    r = np.clip(1.5 - np.abs(4 * z - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * z - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * z - 1), 0, 1)
    return np.stack([r, g, b], axis=1)


def _normalize_xyz(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    C = X.mean(axis=0, keepdims=True)
    Xc = X - C
    s = np.linalg.norm(Xc, axis=1).max()
    s = float(s) if np.isfinite(s) and s > eps else 1.0
    return Xc / s


def load_xyz(path: Path):
    X = np.loadtxt(str(path), dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 3)
    return X


def scan_xyz_files(root: Path):
    pts_dir = root / "points"
    data_dir = pts_dir if pts_dir.is_dir() else root
    files = sorted(data_dir.glob("*.xyz"))

    if not files:
        raise SystemExit(f"[FAIL] .xyz 없음: {data_dir}")

    return files


def make_pcd(raw_xyz):
    pts = raw_xyz[:, :3].astype(np.float64)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))

    if raw_xyz.shape[1] >= 6:
        rgb = raw_xyz[:, 3:6].astype(np.float64)
        if rgb.max() > 1.0:
            rgb /= 255.0
        pcd.colors = o3d.utility.Vector3dVector(rgb)
    else:
        pcd.colors = o3d.utility.Vector3dVector(_jet01(pts[:, 2]))

    return pcd


def to_frame_for_points(pts):
    center = pts.mean(axis=0)
    minb = pts.min(axis=0)
    maxb = pts.max(axis=0)
    extent = np.linalg.norm(maxb - minb)
    axis_len = max(extent * AXIS_SCALE_RATIO, 1e-3)

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_len)
    return center, frame


def sample_indices(n, npts):
    if n <= 0:
        raise RuntimeError("point 개수가 0입니다.")

    if n >= npts:
        return np.random.choice(n, npts, replace=False)

    rep = (npts + n - 1) // n
    extra = np.random.choice(n, rep * n - n, replace=True)
    return np.concatenate([np.arange(n), extra])[:npts]


def infer_prob_once(net, P3, normalize=True):
    Pin = P3.astype(np.float32, copy=False)

    if normalize:
        Pin = _normalize_xyz(Pin).astype(np.float32, copy=False)

    x = torch.from_numpy(Pin).unsqueeze(0).to(DEV)

    with torch.inference_mode():
        logit = net(x)
        prob = torch.sigmoid(logit).squeeze().detach().cpu().numpy()

    return prob


def infer_prob(net, P, npts, normalize=True):
    N = len(P)
    idx = sample_indices(N, npts)

    prob_s = infer_prob_once(
        net,
        P[idx].astype(np.float32),
        normalize=normalize,
    )

    prob = np.zeros((N,), dtype=np.float64)
    np.maximum.at(prob, idx, prob_s.astype(np.float64))

    return prob, idx


def create_box_from_points(Pkeep, box_mode="aabb"):
    if Pkeep is None or Pkeep.shape[0] < 4:
        return None

    try:
        if box_mode == "aabb":
            box = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
                o3d.utility.Vector3dVector(Pkeep.astype(np.float64))
            )
        else:
            box = o3d.geometry.OrientedBoundingBox.create_from_points(
                o3d.utility.Vector3dVector(Pkeep.astype(np.float64))
            )

        box.color = (0.1, 0.8, 0.1)
        return box

    except Exception:
        try:
            box = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
                o3d.utility.Vector3dVector(Pkeep.astype(np.float64))
            )
            box.color = (0.1, 0.8, 0.1)
            return box
        except Exception:
            return None


def save_marker_points(out_dir: Path, stem: str, raw_xyz: np.ndarray, keep_idx: np.ndarray):
    out_dir.mkdir(parents=True, exist_ok=True)

    kept = raw_xyz[keep_idx]
    out_xyz = out_dir / f"{stem}_marker.xyz"

    np.savetxt(str(out_xyz), kept[:, :3], fmt="%.6f")
    np.save(str(out_dir / f"{stem}_marker_idx.npy"), keep_idx.astype(np.int64))

    return out_xyz


def expand_keep_idx_by_range(P: np.ndarray, keep_idx: np.ndarray, expand_mm: float):
    """
    선택된 포인트 전체 범위를 기준으로 주변 포인트를 추가 포함.

    중심점 기준 확장이 아니라,
    선택 포인트의 x/y/z 전체 범위에 margin을 더하는 방식.

    expand_mm:
    - 100이면 선택 포인트 범위의 앞/뒤/좌/우/상/하로 100mm 확장
    """
    if keep_idx is None or keep_idx.size == 0:
        return keep_idx, {
            "expand_mm": float(expand_mm),
            "keep_before": 0,
            "keep_after": 0,
        }

    expand_m = float(expand_mm) / 1000.0

    if expand_m <= 0:
        return keep_idx, {
            "expand_mm": float(expand_mm),
            "keep_before": int(keep_idx.size),
            "keep_after": int(keep_idx.size),
        }

    Pk = P[keep_idx]

    mn = Pk.min(axis=0)
    mx = Pk.max(axis=0)

    mn2 = mn - expand_m
    mx2 = mx + expand_m

    inside = (
        (P[:, 0] >= mn2[0]) & (P[:, 0] <= mx2[0]) &
        (P[:, 1] >= mn2[1]) & (P[:, 1] <= mx2[1]) &
        (P[:, 2] >= mn2[2]) & (P[:, 2] <= mx2[2])
    )

    keep2 = np.flatnonzero(inside)

    info = {
        "expand_mm": float(expand_mm),
        "keep_before": int(keep_idx.size),
        "keep_after": int(keep2.size),
    }

    return keep2.astype(np.int64), info


def compute_bbox_legacy_radius(
    P,
    prob,
    th,
    q_floor,
    top_q,
    radius_m,
    min_pts,
    box_expand_mm,
    box_mode,
):
    """
    bbox 생성 기준:

    1. prob > th 포인트 선택
    2. z 상단 후보 선택
    3. 선택 포인트의 XY 중앙값 기준 radius_m 안쪽 포인트 선택
    4. 선택 포인트 전체 x/y/z 범위를 box_expand_mm만큼 확장
    5. 확장된 포인트로 bbox 생성

    핵심:
    - 박스는 선택된 일부 포인트만 보지 않고,
      그 주변 실제 포인트를 추가로 포함한 뒤 생성함.
    """
    z = P[:, 2]

    z_floor = float(np.quantile(z, q_floor))
    dz = z - z_floor
    dz_thr = float(np.quantile(dz, top_q))

    sel = (prob > th) & (dz >= dz_thr)
    sel_idx = np.flatnonzero(sel)

    if sel_idx.size < min_pts:
        return None, None, {
            "stage": "pre",
            "sel": int(sel_idx.size),
            "keep_seed": 0,
            "keep_box": 0,
            "prob_gt_th": int((prob > th).sum()),
            "radius_m": float(radius_m),
            "box_expand_mm": float(box_expand_mm),
            "box_mode": str(box_mode),
        }

    Psel = P[sel_idx]
    center_xy = np.median(Psel[:, :2], axis=0)

    dxy = np.linalg.norm(Psel[:, :2] - center_xy[None, :], axis=1)
    keep_seed_idx = sel_idx[dxy <= float(radius_m)]

    if keep_seed_idx.size < min_pts:
        return None, None, {
            "stage": "radius",
            "sel": int(sel_idx.size),
            "keep_seed": int(keep_seed_idx.size),
            "keep_box": 0,
            "prob_gt_th": int((prob > th).sum()),
            "radius_m": float(radius_m),
            "box_expand_mm": float(box_expand_mm),
            "box_mode": str(box_mode),
        }

    keep_box_idx, expand_info = expand_keep_idx_by_range(
        P=P,
        keep_idx=keep_seed_idx,
        expand_mm=box_expand_mm,
    )

    if keep_box_idx.size < min_pts:
        return None, None, {
            "stage": "expand",
            "sel": int(sel_idx.size),
            "keep_seed": int(keep_seed_idx.size),
            "keep_box": int(keep_box_idx.size),
            "prob_gt_th": int((prob > th).sum()),
            "radius_m": float(radius_m),
            "box_expand_mm": float(box_expand_mm),
            "box_mode": str(box_mode),
        }

    Pbox = P[keep_box_idx]
    box = create_box_from_points(Pbox, box_mode=box_mode)

    if box is None:
        return None, None, {
            "stage": "box_fail",
            "sel": int(sel_idx.size),
            "keep_seed": int(keep_seed_idx.size),
            "keep_box": int(keep_box_idx.size),
            "prob_gt_th": int((prob > th).sum()),
            "radius_m": float(radius_m),
            "box_expand_mm": float(box_expand_mm),
            "box_mode": str(box_mode),
        }

    meta = {
        "stage": "ok",
        "sel": int(sel_idx.size),
        "keep_seed": int(keep_seed_idx.size),
        "keep_box": int(keep_box_idx.size),
        "prob_gt_th": int((prob > th).sum()),
        "radius_m": float(radius_m),
        "box_expand_mm": float(box_expand_mm),
        "box_mode": str(box_mode),
        "expand_info": expand_info,
    }

    return box, keep_box_idx, meta


def load_model_from_ckpt(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location=DEV)
    ckpt_meta = ckpt.get("meta", {}) if isinstance(ckpt, dict) else {}

    model_k = int(ckpt_meta.get("k", 16))
    floor_q = float(ckpt_meta.get("floor_q", 0.05))

    try:
        net = Model(k=model_k, floor_q=floor_q).to(DEV).eval()
    except TypeError:
        try:
            net = Model(k=model_k).to(DEV).eval()
        except TypeError:
            net = Model().to(DEV).eval()

    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    net.load_state_dict(state)

    ckpt_norm = ckpt_meta.get("normalize", None)

    return net, ckpt_meta, ckpt_norm


def main(args):
    np.random.seed(args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    marker_dir = out_dir / "marker_points"

    net, ckpt_meta, ckpt_norm = load_model_from_ckpt(args.ckpt)

    if args.force_no_norm:
        use_norm = False
    elif ckpt_norm is None:
        use_norm = True
    else:
        use_norm = bool(ckpt_norm)

    print(f"[INFO] loaded ckpt: {args.ckpt}  device={DEV.type}")
    print(f"[INFO] ckpt_meta_normalize={ckpt_norm}  eval_normalize={use_norm}")
    print(f"[INFO] ckpt_feature={ckpt_meta.get('feature', None)}")
    print(f"[INFO] ckpt_floor_q={ckpt_meta.get('floor_q', None)}")

    files = scan_xyz_files(Path(args.data))

    print(f"[INFO] files={len(files)}  data={args.data}")
    print(f"[INFO] npts={args.npts}  th={args.th}  q_floor={args.q_floor}  top_q={args.top_q}")
    print(f"[INFO] radius_m={args.radius_m}  min_pts={args.min_pts}")
    print(f"[INFO] box_expand_mm={args.box_expand_mm}  box_mode={args.box_mode}")
    print(f"[INFO] save_marker={int(args.save_marker)}  marker_dir={marker_dir}")
    print(f"[INFO] save_expand_mm={args.save_expand_mm}")

    cache = []
    ok = 0
    saved = 0

    for fp in files:
        raw = load_xyz(fp)
        P = raw[:, :3].astype(np.float64)

        prob, sampled_idx = infer_prob(
            net,
            P,
            npts=args.npts,
            normalize=use_norm,
        )

        box, keep_idx, meta = compute_bbox_legacy_radius(
            P=P,
            prob=prob,
            th=args.th,
            q_floor=args.q_floor,
            top_q=args.top_q,
            radius_m=args.radius_m,
            min_pts=args.min_pts,
            box_expand_mm=args.box_expand_mm,
            box_mode=args.box_mode,
        )

        save_info = None

        if box is not None:
            ok += 1

            if args.save_marker and keep_idx is not None:
                keep_idx_save, save_info = expand_keep_idx_by_range(
                    P=P,
                    keep_idx=keep_idx,
                    expand_mm=args.save_expand_mm,
                )

                save_marker_points(marker_dir, fp.stem, raw, keep_idx_save)
                saved += 1

        if save_info is not None:
            meta = dict(meta)
            meta["save_expand"] = save_info

        sampled_prob = prob[sampled_idx]

        print(
            f"[GEN] {fp.name}  "
            f"bbox={'OK' if box is not None else 'NONE'}  "
            f"sampled={int(sampled_idx.size)}  "
            f"prob_min={float(sampled_prob.min()):.4f}  "
            f"prob_mean={float(sampled_prob.mean()):.4f}  "
            f"prob_max={float(sampled_prob.max()):.4f}  "
            f"meta={meta}"
        )

        cache.append((fp, box))

    print(
        f"[DONE] bbox OK: {ok}/{len(files)}  "
        f"marker_saved: {saved}/{len(files)} -> {marker_dir if args.save_marker else '(off)'}"
    )

    if args.no_vis:
        print("[INFO] visualization skipped by --no_vis")
        return

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("marker viewer")

    opt = vis.get_render_option()
    opt.point_size = float(POINT_SIZE)
    opt.background_color = np.array([1, 1, 1], dtype=np.float64)

    i = 0

    def show(k):
        fp, box = cache[k]

        raw = load_xyz(fp)
        pcd = make_pcd(raw)
        pts = np.asarray(pcd.points)

        center, frame = to_frame_for_points(pts)

        vis.clear_geometries()
        vis.add_geometry(pcd)
        vis.add_geometry(frame)

        if box is not None:
            vis.add_geometry(box)

        ctr = vis.get_view_control()
        ctr.set_lookat(center.tolist())
        ctr.set_up([0, 0, 1])
        ctr.set_front([0, -1, 0])
        ctr.set_zoom(0.8)

        vis.poll_events()
        vis.update_renderer()

        print(f"[VIEW] {k + 1}/{len(cache)}: {fp.name}")

    def on_next(v):
        nonlocal i
        i = (i + 1) % len(cache)
        show(i)
        return False

    def on_prev(v):
        nonlocal i
        i = (i - 1 + len(cache)) % len(cache)
        show(i)
        return False

    def on_quit(v):
        v.close()
        return False

    vis.register_key_callback(ord("N"), on_next)
    vis.register_key_callback(ord("P"), on_prev)
    vis.register_key_callback(ord("Q"), on_quit)

    show(i)
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)

    ap.add_argument("--npts", type=int, default=16384)

    ap.add_argument("--th", type=float, default=0.5)
    ap.add_argument("--q_floor", type=float, default=0.05)
    ap.add_argument("--top_q", type=float, default=0.95)

    ap.add_argument("--radius_m", type=float, default=0.8)
    ap.add_argument("--min_pts", type=int, default=150)

    # 새 옵션
    ap.add_argument(
        "--box_expand_mm",
        type=float,
        default=150.0,
        help="bbox 생성 전에 선택 포인트 범위를 이 값만큼 확장",
    )

    ap.add_argument(
        "--box_mode",
        choices=["aabb", "obb"],
        default="aabb",
        help="aabb=기울지 않는 박스, obb=회전 박스",
    )

    ap.add_argument("--save_marker", action="store_true")
    ap.add_argument("--save_expand_mm", type=float, default=0.0)

    ap.add_argument("--force_no_norm", action="store_true")
    ap.add_argument("--no_vis", action="store_true")
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()
    main(args)
    # python -u .\marker_bbox_eval_1_test.py --data "C:\Users\gkwld\Desktop\3D-Marker-Pose-Estimation\CAD_to_PointCloud_conversion\dataset\real\val\" --ckpt "C:\Users\gkwld\Desktop\3D-Marker-Pose-Estimation\CAD_to_PointCloud_conversion\pointnet2\result\exp_seg_train1_newmask_height_auto\best.pth" --out "C:\Users\gkwld\Desktop\3D-Marker-Pose-Estimation\CAD_to_PointCloud_conversion\pointnet2\result\eval1_real_newmask_height_auto_boxexpand150" --npts 16384 --th 0.45 --q_floor 0.05 --top_q 0.95 --radius_m 0.8 --min_pts 150 --box_expand_mm 100 --box_mode aabb --save_marker --save_expand_mm 0