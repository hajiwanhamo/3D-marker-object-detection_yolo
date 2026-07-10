# # make_top_id_projection.py
# # 목적:
# # - eval1로 저장된 *_marker.xyz에서 내부 ID 최상단부 포인트만 추출
# # - 각 파일별 마커 평면을 추정하고 local 평면 좌표계로 직교 투영
# # - YOLO 입력용 binary image 저장
# # - 실제 소나 시각화 느낌 확인용 raw_z 기반 jet color image 저장
# # - 2D 결과를 다시 3D 방향 벡터로 변환하기 위한 meta.json 저장
# #
# # 유지 기능:
# # - *_top_id.xyz
# # - *_top_id_uv.npy
# # - *_marker_all_uv.npy
# # - *_binary.png
# # - *_color.png
# # - *_meta.json
# #
# # 핵심 수정:
# # - height_mode fixed/adaptive 지원
# # - adaptive 방식은 파일별 평면 기준 높이 분포를 보고 내부 ID 최상단부 높이층을 자동 선택
# # - raw_z jet 색상 기준은 marker crop 전체 raw_z 범위로 고정
# # - 회전 정렬 없음

# import argparse
# import json
# from pathlib import Path

# import numpy as np

# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt


# # ============================================================
# # IO
# # ============================================================
# def load_xyz(path: Path) -> np.ndarray:
#     """xyz 파일 로드. 앞 3열만 좌표로 사용."""
#     P = np.loadtxt(str(path), dtype=np.float64)

#     if P.ndim == 1:
#         P = P.reshape(-1, 3)

#     if P.shape[1] < 3:
#         raise RuntimeError(f"xyz 형식 오류: {path}")

#     return P[:, :3].astype(np.float64, copy=False)


# def save_xyz(path: Path, P: np.ndarray):
#     """xyz 저장."""
#     path.parent.mkdir(parents=True, exist_ok=True)
#     np.savetxt(str(path), P.astype(np.float64), fmt="%.6f")


# def scan_xyz_files(in_path: Path):
#     """단일 xyz 파일 또는 폴더에서 xyz 파일 목록 생성."""
#     if in_path.is_file():
#         return [in_path]

#     if not in_path.is_dir():
#         raise SystemExit(f"[FAIL] 입력 경로 없음: {in_path}")

#     files = sorted(in_path.glob("*.xyz"))

#     if not files:
#         raise SystemExit(f"[FAIL] xyz 파일 없음: {in_path}")

#     return files


# # ============================================================
# # Jet color
# # ============================================================
# def jet_color_with_range(values: np.ndarray, vmin: float, vmax: float, invert: bool = False) -> np.ndarray:
#     """
#     raw_z 값을 jet 유사 컬러맵으로 변환.
#     색상 범위는 marker crop 전체 raw_z 기준으로 넣어야 함.

#     invert=False:
#         낮은 값=파랑, 높은 값=빨강
#     invert=True:
#         낮은 값=빨강, 높은 값=파랑
#     """
#     v = np.asarray(values, dtype=np.float64)

#     if len(v) == 0:
#         return np.zeros((0, 3), dtype=np.uint8)

#     rng = max(float(vmax) - float(vmin), 1e-12)
#     z = (v - float(vmin)) / rng
#     z = np.clip(z, 0.0, 1.0)

#     if invert:
#         z = 1.0 - z

#     r = np.clip(1.5 - np.abs(4.0 * z - 3.0), 0.0, 1.0)
#     g = np.clip(1.5 - np.abs(4.0 * z - 2.0), 0.0, 1.0)
#     b = np.clip(1.5 - np.abs(4.0 * z - 1.0), 0.0, 1.0)

#     rgb = np.stack([r, g, b], axis=1)
#     return (rgb * 255.0).astype(np.uint8)


# # ============================================================
# # Plane / local projection
# # ============================================================
# def fit_plane_pca(P: np.ndarray):
#     """
#     PCA 기반 평면 추정.
#     가장 분산이 작은 방향을 평면 normal로 사용.
#     """
#     center = P.mean(axis=0)
#     X = P - center

#     cov = (X.T @ X) / max(len(X), 1)
#     eigvals, eigvecs = np.linalg.eigh(cov)

#     normal = eigvecs[:, np.argmin(eigvals)]
#     normal = normal / (np.linalg.norm(normal) + 1e-12)

#     if normal[2] < 0:
#         normal = -normal

#     return center, normal


# def signed_height(P: np.ndarray, plane_center: np.ndarray, normal: np.ndarray):
#     """평면 기준 signed height 계산."""
#     return (P - plane_center[None, :]) @ normal


# def make_local_basis_from_global(normal: np.ndarray):
#     """
#     전역 x축을 현재 평면 위로 투영해 u축으로 사용.
#     자동 회전 정렬을 하지 않기 때문에 crop 외곽/노이즈에 끌려가지 않음.
#     """
#     normal = normal / (np.linalg.norm(normal) + 1e-12)

#     ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
#     u = ref - np.dot(ref, normal) * normal

#     if np.linalg.norm(u) < 1e-8:
#         ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
#         u = ref - np.dot(ref, normal) * normal

#     u = u / (np.linalg.norm(u) + 1e-12)

#     v = np.cross(normal, u)
#     v = v / (np.linalg.norm(v) + 1e-12)

#     return u, v


# def make_local_basis_pca(P_all: np.ndarray, normal: np.ndarray):
#     """
#     선택 옵션용 PCA 기반 local 축.
#     기본값은 global을 권장.
#     """
#     center = P_all.mean(axis=0)
#     X = P_all - center

#     X_proj = X - ((X @ normal)[:, None] * normal[None, :])
#     cov = (X_proj.T @ X_proj) / max(len(X_proj), 1)
#     eigvals, eigvecs = np.linalg.eigh(cov)

#     order = np.argsort(eigvals)[::-1]
#     u = eigvecs[:, order[0]]
#     u = u - np.dot(u, normal) * normal

#     if np.linalg.norm(u) < 1e-8:
#         return make_local_basis_from_global(normal)

#     u = u / (np.linalg.norm(u) + 1e-12)

#     v = np.cross(normal, u)
#     v = v / (np.linalg.norm(v) + 1e-12)

#     return u, v


# def project_to_uv(P: np.ndarray, origin: np.ndarray, u: np.ndarray, v: np.ndarray):
#     """3D point를 local 평면 좌표 u/v로 직교 투영."""
#     X = P - origin[None, :]
#     uv = np.stack([X @ u, X @ v], axis=1)
#     return uv.astype(np.float64)


# # ============================================================
# # Height layer selection
# # ============================================================
# def select_top_id_layer_fixed(h: np.ndarray, args):
#     """
#     기존 fixed 방식.
#     모든 파일에 동일하게 top_candidate_q와 height_band_m 적용.
#     """
#     top_thr = float(np.quantile(h, float(args.top_candidate_q)))
#     high_mask = h >= top_thr
#     high_h = h[high_mask]

#     if len(high_h) < 3:
#         center = top_thr
#         band = float(args.height_band_m)
#         mask = np.zeros((len(h),), dtype=bool)
#         method_status = "fixed_too_few_high_candidates"
#     else:
#         center = float(np.median(high_h))
#         band = float(args.height_band_m)
#         mask = np.abs(h - center) <= band
#         method_status = "fixed_ok"

#     info = {
#         "height_mode": "fixed",
#         "status": method_status,
#         "top_candidate_q": float(args.top_candidate_q),
#         "top_candidate_threshold": float(top_thr),
#         "id_height_center": float(center),
#         "height_band_m": float(band),
#         "selected_height_min": float(h[mask].min()) if np.any(mask) else None,
#         "selected_height_max": float(h[mask].max()) if np.any(mask) else None,
#         "selected_points": int(np.sum(mask)),
#     }

#     return mask, center, band, info


# def select_top_id_layer_adaptive(h: np.ndarray, args):
#     """
#     파일별 adaptive height layer selection.

#     원리:
#     1. 평면 기준 높이 h의 높은 쪽 영역만 탐색
#     2. 히스토그램으로 높이층을 나눔
#     3. 너무 적은 포인트만 있는 최상단 노이즈 bin은 제외
#     4. 높은 쪽에서 충분히 밀집된 bin을 ID 최상단 후보층으로 선택
#     5. 선택된 층의 median과 분산으로 파일별 band 자동 결정
#     6. 실패하면 fixed 방식으로 fallback
#     """
#     h = np.asarray(h, dtype=np.float64)
#     n = len(h)

#     if n < 10:
#         return select_top_id_layer_fixed(h, args)

#     # 검색 범위: 너무 낮은 지면/상판층은 제외하고 상단 영역 중심으로 탐색
#     q_low = float(args.adaptive_search_q)
#     q_high = float(args.adaptive_upper_q)

#     h_min = float(np.quantile(h, q_low))
#     h_max = float(np.quantile(h, q_high))

#     if not np.isfinite(h_min) or not np.isfinite(h_max) or h_max <= h_min:
#         return select_top_id_layer_fixed(h, args)

#     bins = int(max(args.adaptive_bins, 8))
#     hist, edges = np.histogram(h, bins=bins, range=(h_min, h_max))

#     min_pts_abs = int(args.adaptive_min_points)
#     min_pts_ratio = int(np.ceil(float(args.adaptive_min_ratio) * n))
#     min_pts = max(min_pts_abs, min_pts_ratio)

#     # 높은 쪽 bin부터 내려오면서 충분한 포인트가 있는 층 선택
#     chosen_bin = None
#     for i in range(len(hist) - 1, -1, -1):
#         if hist[i] >= min_pts:
#             chosen_bin = i
#             break

#     if chosen_bin is None:
#         fixed_mask, center, band, fixed_info = select_top_id_layer_fixed(h, args)
#         fixed_info["adaptive_status"] = "fallback_no_dense_bin"
#         fixed_info["adaptive_min_points"] = int(min_pts)
#         return fixed_mask, center, band, fixed_info

#     # 선택 bin 주변의 인접 bin까지 포함해서 층의 높이 분포 추정
#     main_count = max(int(hist[chosen_bin]), 1)
#     neighbor_ratio = float(args.adaptive_neighbor_ratio)

#     left = chosen_bin
#     while left - 1 >= 0 and hist[left - 1] >= main_count * neighbor_ratio:
#         left -= 1

#     right = chosen_bin
#     while right + 1 < len(hist) and hist[right + 1] >= main_count * neighbor_ratio:
#         right += 1

#     layer_lo = float(edges[left])
#     layer_hi = float(edges[right + 1])

#     layer_mask_for_stats = (h >= layer_lo) & (h <= layer_hi)
#     layer_h = h[layer_mask_for_stats]

#     if len(layer_h) < min_pts:
#         fixed_mask, center, band, fixed_info = select_top_id_layer_fixed(h, args)
#         fixed_info["adaptive_status"] = "fallback_too_few_layer_points"
#         fixed_info["adaptive_layer_points"] = int(len(layer_h))
#         return fixed_mask, center, band, fixed_info

#     center = float(np.median(layer_h))

#     # 층 두께 자동 계산: MAD 기반
#     mad = float(np.median(np.abs(layer_h - center)))
#     robust_sigma = 1.4826 * mad

#     band_auto = max(
#         float(args.adaptive_min_band_m),
#         float(args.adaptive_band_scale) * robust_sigma,
#     )
#     band_auto = min(band_auto, float(args.adaptive_max_band_m))

#     # 너무 얇은 층일 때도 최소 band 보장
#     band = float(band_auto)

#     mask = np.abs(h - center) <= band

#     selected_count = int(np.sum(mask))
#     selected_ratio = selected_count / max(n, 1)

#     # 선택 결과가 너무 적으면 band를 조금 넓혀 재선택
#     if selected_count < min_pts:
#         band = min(float(args.adaptive_max_band_m), max(band * 1.5, float(args.height_band_m)))
#         mask = np.abs(h - center) <= band
#         selected_count = int(np.sum(mask))
#         selected_ratio = selected_count / max(n, 1)

#     # 여전히 너무 적으면 fixed fallback
#     if selected_count < min_pts:
#         fixed_mask, fixed_center, fixed_band, fixed_info = select_top_id_layer_fixed(h, args)
#         fixed_info["adaptive_status"] = "fallback_selected_too_few"
#         fixed_info["adaptive_selected_points"] = int(selected_count)
#         return fixed_mask, fixed_center, fixed_band, fixed_info

#     # 너무 많이 선택되면 상판 포함 가능성이 있으므로 band 축소 시도
#     max_ratio = float(args.adaptive_max_selected_ratio)
#     if selected_ratio > max_ratio:
#         band_shrink = max(float(args.adaptive_min_band_m), band * 0.7)
#         mask_shrink = np.abs(h - center) <= band_shrink
#         if np.sum(mask_shrink) >= min_pts:
#             band = band_shrink
#             mask = mask_shrink
#             selected_count = int(np.sum(mask))
#             selected_ratio = selected_count / max(n, 1)

#     selected_h = h[mask]

#     info = {
#         "height_mode": "adaptive",
#         "status": "adaptive_ok",
#         "adaptive_search_q": float(args.adaptive_search_q),
#         "adaptive_upper_q": float(args.adaptive_upper_q),
#         "adaptive_bins": int(bins),
#         "adaptive_min_points": int(min_pts),
#         "adaptive_chosen_bin": int(chosen_bin),
#         "adaptive_bin_count": int(hist[chosen_bin]),
#         "adaptive_layer_bin_left": int(left),
#         "adaptive_layer_bin_right": int(right),
#         "adaptive_layer_lo": float(layer_lo),
#         "adaptive_layer_hi": float(layer_hi),
#         "adaptive_layer_points": int(len(layer_h)),
#         "adaptive_mad": float(mad),
#         "adaptive_robust_sigma": float(robust_sigma),
#         "id_height_center": float(center),
#         "height_band_m": float(band),
#         "selected_height_min": float(selected_h.min()) if len(selected_h) > 0 else None,
#         "selected_height_max": float(selected_h.max()) if len(selected_h) > 0 else None,
#         "selected_points": int(selected_count),
#         "selected_ratio": float(selected_ratio),
#     }

#     return mask, center, band, info


# def select_top_id_layer(h: np.ndarray, args):
#     """height_mode에 따라 fixed/adaptive 선택."""
#     if args.height_mode == "fixed":
#         return select_top_id_layer_fixed(h, args)

#     return select_top_id_layer_adaptive(h, args)


# # ============================================================
# # Top ID extraction
# # ============================================================
# def extract_top_id_points(P: np.ndarray, args):
#     """
#     내부 ID 최상단부 추출.

#     절차:
#     1. plane_fit_q 기준으로 평면 추정용 포인트 선택
#     2. 평면 normal 계산
#     3. local 평면 u/v 좌표계 생성
#     4. 평면 기준 높이 h 계산
#     5. fixed 또는 adaptive 방식으로 내부 ID 최상단부 높이층 선택
#     """
#     if len(P) < 10:
#         raise RuntimeError("포인트 수가 너무 적습니다.")

#     z = P[:, 2]

#     # 평면 추정에 사용할 포인트 선택
#     z_cut = np.quantile(z, float(args.plane_fit_q))
#     plane_fit_points = P[z <= z_cut]

#     if len(plane_fit_points) < 10:
#         plane_fit_points = P

#     plane_center, normal = fit_plane_pca(plane_fit_points)

#     # local 좌표 원점은 marker crop 전체 중심
#     marker_origin = P.mean(axis=0)

#     # local 축 생성
#     if args.basis_mode == "global":
#         u_axis, v_axis = make_local_basis_from_global(normal)
#     else:
#         u_axis, v_axis = make_local_basis_pca(P, normal)

#     # marker 전체 투영
#     uv_all = project_to_uv(P, marker_origin, u_axis, v_axis)

#     # 평면 기준 높이 계산
#     h = signed_height(P, plane_center, normal)

#     # 내부 ID 최상단부 선택
#     id_mask, id_height_center, height_band_used, layer_info = select_top_id_layer(h, args)

#     P_top = P[id_mask]
#     uv_top = uv_all[id_mask]

#     selected_h = h[id_mask]

#     debug = {
#         "total_points": int(len(P)),
#         "plane_fit_points": int(len(plane_fit_points)),
#         "basis_mode": str(args.basis_mode),
#         "height_mode": str(args.height_mode),
#         "top_candidate_q": float(args.top_candidate_q),
#         "height_band_m_input": float(args.height_band_m),
#         "height_band_m_used": float(height_band_used),
#         "id_height_center": float(id_height_center),
#         "top_id_points": int(len(P_top)),
#         "selected_height_min": float(selected_h.min()) if len(selected_h) > 0 else None,
#         "selected_height_max": float(selected_h.max()) if len(selected_h) > 0 else None,
#         "height_min": float(h.min()),
#         "height_mean": float(h.mean()),
#         "height_max": float(h.max()),
#         "height_q50": float(np.quantile(h, 0.50)),
#         "height_q70": float(np.quantile(h, 0.70)),
#         "height_q80": float(np.quantile(h, 0.80)),
#         "height_q85": float(np.quantile(h, 0.85)),
#         "height_q90": float(np.quantile(h, 0.90)),
#         "height_q95": float(np.quantile(h, 0.95)),
#         "height_q99": float(np.quantile(h, 0.99)),
#         "layer_selection": layer_info,
#     }

#     return {
#         "P_top": P_top,
#         "uv_top": uv_top,
#         "uv_all": uv_all,
#         "height_all": h,
#         "id_mask": id_mask,
#         "plane_center": plane_center,
#         "normal": normal,
#         "marker_origin": marker_origin,
#         "u_axis": u_axis,
#         "v_axis": v_axis,
#         "debug": debug,
#     }


# # ============================================================
# # Image rasterization
# # ============================================================
# def compute_view_bounds(uv_all: np.ndarray, margin_ratio: float):
#     """
#     marker 전체 기준으로 이미지 범위 계산.
#     내부 ID 기준으로 범위를 잡지 않음.
#     """
#     if len(uv_all) == 0:
#         return -1.0, 1.0, -1.0, 1.0

#     mn = uv_all.min(axis=0)
#     mx = uv_all.max(axis=0)

#     center = 0.5 * (mn + mx)
#     half = 0.5 * (mx - mn)

#     half[0] = max(float(half[0]), 0.05)
#     half[1] = max(float(half[1]), 0.05)

#     half = half * max(float(margin_ratio), 1.0)

#     xmin = center[0] - half[0]
#     xmax = center[0] + half[0]
#     ymin = center[1] - half[1]
#     ymax = center[1] + half[1]

#     return xmin, xmax, ymin, ymax


# def uv_to_pixel(uv: np.ndarray, bounds, image_size: int):
#     """uv 좌표를 image pixel 좌표로 변환."""
#     xmin, xmax, ymin, ymax = bounds
#     W = int(image_size)
#     H = int(image_size)

#     if len(uv) == 0:
#         return (
#             np.empty((0,), dtype=np.int32),
#             np.empty((0,), dtype=np.int32),
#             np.empty((0,), dtype=bool),
#         )

#     x = uv[:, 0]
#     y = uv[:, 1]

#     px = np.round((x - xmin) / max(xmax - xmin, 1e-12) * (W - 1)).astype(np.int32)
#     py = np.round((ymax - y) / max(ymax - ymin, 1e-12) * (H - 1)).astype(np.int32)

#     valid = (px >= 0) & (px < W) & (py >= 0) & (py < H)

#     return px, py, valid


# def disk_offsets(radius: int):
#     """포인트 두께용 원형 offset."""
#     radius = int(max(radius, 0))

#     if radius == 0:
#         return [(0, 0)]

#     offs = []
#     r2 = radius * radius

#     for dy in range(-radius, radius + 1):
#         for dx in range(-radius, radius + 1):
#             if dx * dx + dy * dy <= r2:
#                 offs.append((dx, dy))

#     return offs


# def rasterize_binary(px, py, valid, image_size: int, point_radius: int):
#     """YOLO 입력용 binary image 생성."""
#     H = int(image_size)
#     W = int(image_size)

#     img = np.zeros((H, W), dtype=np.uint8)
#     offs = disk_offsets(point_radius)

#     for x, y, ok in zip(px, py, valid):
#         if not ok:
#             continue

#         for dx, dy in offs:
#             xx = x + dx
#             yy = y + dy

#             if 0 <= xx < W and 0 <= yy < H:
#                 img[yy, xx] = 255

#     return img


# def rasterize_color(px, py, valid, rgb, image_size: int, point_radius: int):
#     """raw_z jet color image 생성."""
#     H = int(image_size)
#     W = int(image_size)

#     img = np.zeros((H, W, 3), dtype=np.uint8)
#     offs = disk_offsets(point_radius)

#     for x, y, ok, c in zip(px, py, valid, rgb):
#         if not ok:
#             continue

#         for dx, dy in offs:
#             xx = x + dx
#             yy = y + dy

#             if 0 <= xx < W and 0 <= yy < H:
#                 img[yy, xx, :] = c

#     return img


# # ============================================================
# # Save
# # ============================================================
# def save_png_gray(path: Path, img: np.ndarray):
#     """grayscale png 저장."""
#     path.parent.mkdir(parents=True, exist_ok=True)
#     plt.imsave(str(path), img, cmap="gray", vmin=0, vmax=255)


# def save_png_rgb(path: Path, img: np.ndarray):
#     """rgb png 저장."""
#     path.parent.mkdir(parents=True, exist_ok=True)
#     plt.imsave(str(path), img)


# def save_meta_json(path: Path, meta: dict):
#     """json 저장."""
#     path.parent.mkdir(parents=True, exist_ok=True)
#     with open(path, "w", encoding="utf-8") as f:
#         json.dump(meta, f, indent=2, ensure_ascii=False)


# # ============================================================
# # Process
# # ============================================================
# def process_one(fp: Path, out_dir: Path, args):
#     P = load_xyz(fp)
#     result = extract_top_id_points(P, args)

#     P_top = result["P_top"]
#     uv_top = result["uv_top"]
#     uv_all = result["uv_all"]

#     plane_center = result["plane_center"]
#     normal = result["normal"]
#     marker_origin = result["marker_origin"]
#     u_axis = result["u_axis"]
#     v_axis = result["v_axis"]
#     debug = result["debug"]

#     # 이미지 범위는 marker 전체 기준
#     bounds = compute_view_bounds(uv_all, args.view_margin)

#     px, py, valid = uv_to_pixel(
#         uv=uv_top,
#         bounds=bounds,
#         image_size=args.image_size,
#     )

#     # binary image
#     binary_img = rasterize_binary(
#         px=px,
#         py=py,
#         valid=valid,
#         image_size=args.image_size,
#         point_radius=args.point_radius,
#     )

#     # raw_z jet color image
#     if len(P_top) > 0:
#         # 표시 대상: 내부 ID 최상단부
#         top_values = P_top[:, 2]

#         # 색상 범위 기준: marker crop 전체 raw_z
#         all_values = P[:, 2]

#         cmin = float(np.quantile(all_values, args.color_q_min))
#         cmax = float(np.quantile(all_values, args.color_q_max))

#         rgb = jet_color_with_range(
#             top_values,
#             cmin,
#             cmax,
#             invert=bool(args.invert_jet),
#         )
#     else:
#         cmin = 0.0
#         cmax = 1.0
#         rgb = np.zeros((0, 3), dtype=np.uint8)

#     color_img = rasterize_color(
#         px=px,
#         py=py,
#         valid=valid,
#         rgb=rgb,
#         image_size=args.image_size,
#         point_radius=args.point_radius,
#     )

#     xmin, xmax, ymin, ymax = bounds

#     pixel_size_u = float((xmax - xmin) / max(args.image_size - 1, 1))
#     pixel_size_v = float((ymax - ymin) / max(args.image_size - 1, 1))

#     meta = {
#         "source_file": str(fp),

#         "image_size": int(args.image_size),
#         "point_radius": int(args.point_radius),
#         "view_margin": float(args.view_margin),

#         "plane_fit_q": float(args.plane_fit_q),
#         "height_mode": str(args.height_mode),
#         "top_candidate_q": float(args.top_candidate_q),
#         "height_band_m": float(args.height_band_m),
#         "basis_mode": str(args.basis_mode),

#         "align_2d": False,
#         "align_method": "none",

#         "color_mode": "raw_z",
#         "color_ref": "marker_all_raw_z",
#         "invert_jet": bool(args.invert_jet),
#         "color_min": float(cmin),
#         "color_max": float(cmax),
#         "color_q_min": float(args.color_q_min),
#         "color_q_max": float(args.color_q_max),

#         # 3D 평면 정보
#         "plane_center": plane_center.tolist(),
#         "plane_normal": normal.tolist(),
#         "marker_origin": marker_origin.tolist(),

#         # 이미지 좌표계의 3D 축
#         "image_u_axis_3d": u_axis.tolist(),
#         "image_v_axis_3d": v_axis.tolist(),

#         # 이미지 좌표 범위
#         "u_min": float(xmin),
#         "u_max": float(xmax),
#         "v_min": float(ymin),
#         "v_max": float(ymax),
#         "pixel_size_u_m": pixel_size_u,
#         "pixel_size_v_m": pixel_size_v,

#         # 2D -> 3D 역변환 공식
#         "back_projection_note": {
#             "pixel_to_uv": "u = u_min + px * pixel_size_u_m, v = v_max - py * pixel_size_v_m",
#             "point_on_plane_3d": "marker_origin + u * image_u_axis_3d + v * image_v_axis_3d",
#             "direction_3d": "du * image_u_axis_3d + dv * image_v_axis_3d",
#         },

#         "debug": debug,
#     }

#     stem = fp.stem

#     out_top_xyz = out_dir / f"{stem}_top_id.xyz"
#     out_top_uv = out_dir / f"{stem}_top_id_uv.npy"
#     out_all_uv = out_dir / f"{stem}_marker_all_uv.npy"
#     out_binary = out_dir / f"{stem}_binary.png"
#     out_color = out_dir / f"{stem}_color.png"
#     out_meta = out_dir / f"{stem}_meta.json"

#     save_xyz(out_top_xyz, P_top)
#     np.save(str(out_top_uv), uv_top.astype(np.float32))
#     np.save(str(out_all_uv), uv_all.astype(np.float32))

#     save_png_gray(out_binary, binary_img)
#     save_png_rgb(out_color, color_img)
#     save_meta_json(out_meta, meta)

#     print(
#         f"[OK] {fp.name} | "
#         f"total={debug['total_points']} | "
#         f"top_id_points={debug['top_id_points']} | "
#         f"id_height={debug['id_height_center']:.6f} | "
#         f"band_used={debug['height_band_m_used']:.6f} | "
#         f"height_mode={args.height_mode} | "
#         f"basis={args.basis_mode} | "
#         f"color_raw_z_range=({cmin:.4f}, {cmax:.4f})"
#     )


# # ============================================================
# # Main
# # ============================================================
# def main():
#     ap = argparse.ArgumentParser()

#     ap.add_argument("--in_path", required=True, help="*_marker.xyz 파일 또는 marker_points 폴더")
#     ap.add_argument("--out", required=True, help="출력 폴더")

#     # 평면 / 최상단부 추출
#     ap.add_argument("--plane_fit_q", type=float, default=0.80)

#     ap.add_argument(
#         "--height_mode",
#         choices=["fixed", "adaptive"],
#         default="adaptive",
#         help="fixed=기존 방식, adaptive=파일별 높이층 자동 선택. 기본 adaptive",
#     )

#     # fixed fallback 또는 fixed mode용
#     ap.add_argument("--top_candidate_q", type=float, default=0.85)
#     ap.add_argument("--height_band_m", type=float, default=0.03)

#     # adaptive mode용
#     ap.add_argument("--adaptive_search_q", type=float, default=0.60)
#     ap.add_argument("--adaptive_upper_q", type=float, default=0.995)
#     ap.add_argument("--adaptive_bins", type=int, default=64)
#     ap.add_argument("--adaptive_min_points", type=int, default=20)
#     ap.add_argument("--adaptive_min_ratio", type=float, default=0.003)
#     ap.add_argument("--adaptive_neighbor_ratio", type=float, default=0.35)
#     ap.add_argument("--adaptive_band_scale", type=float, default=2.5)
#     ap.add_argument("--adaptive_min_band_m", type=float, default=0.010)
#     ap.add_argument("--adaptive_max_band_m", type=float, default=0.060)
#     ap.add_argument("--adaptive_max_selected_ratio", type=float, default=0.35)

#     # local basis
#     ap.add_argument(
#         "--basis_mode",
#         choices=["global", "pca"],
#         default="global",
#         help="global=전역축을 평면에 투영, pca=marker crop 분포 기준. 기본 global",
#     )

#     # 이미지 생성
#     ap.add_argument("--image_size", type=int, default=512)
#     ap.add_argument("--point_radius", type=int, default=2)
#     ap.add_argument("--view_margin", type=float, default=1.2)

#     # raw_z jet color range
#     ap.add_argument("--color_q_min", type=float, default=0.02)
#     ap.add_argument("--color_q_max", type=float, default=0.98)
#     ap.add_argument(
#         "--invert_jet",
#         action="store_true",
#         help="jet 색상 방향 반전. 기본은 높은 z=빨강",
#     )

#     args = ap.parse_args()

#     in_path = Path(args.in_path)
#     out_dir = Path(args.out)
#     out_dir.mkdir(parents=True, exist_ok=True)

#     files = scan_xyz_files(in_path)

#     print(f"[INFO] input files={len(files)}")
#     print(f"[INFO] out={out_dir}")
#     print(f"[INFO] plane_fit_q={args.plane_fit_q}")
#     print(f"[INFO] height_mode={args.height_mode}")
#     print(f"[INFO] top_candidate_q={args.top_candidate_q}")
#     print(f"[INFO] height_band_m={args.height_band_m}")
#     print(f"[INFO] basis_mode={args.basis_mode}")
#     print(f"[INFO] image_size={args.image_size}")
#     print(f"[INFO] point_radius={args.point_radius}")
#     print(f"[INFO] view_margin={args.view_margin}")
#     print("[INFO] align_2d=disabled")
#     print("[INFO] color_mode=raw_z")
#     print("[INFO] color_ref=marker_all_raw_z")
#     print(f"[INFO] invert_jet={args.invert_jet}")

#     ok_count = 0

#     for fp in files:
#         try:
#             process_one(fp, out_dir, args)
#             ok_count += 1
#         except Exception as e:
#             print(f"[FAIL] {fp.name}: {e}")

#     print(f"[DONE] ok={ok_count}/{len(files)}")


# if __name__ == "__main__":
#     main()
# python -u .\make_top_id_projection.py --in_path "C:\Users\gkwld\Desktop\3D-Marker-Pose-Estimation\CAD_to_PointCloud_conversion\pointnet2\result\eval1_real_newmask_height_auto_boxexpand100\marker_points" --out "C:\Users\gkwld\Desktop\3D-Marker-Pose-Estimation\CAD_to_PointCloud_conversion\pointnet2\result\eval1_real_newmask_height_auto_boxexpand100\top_id_projection_adaptive_wider_id" --plane_fit_q 0.80 --height_mode adaptive --top_candidate_q 0.85 --height_band_m 0.03 --basis_mode global --image_size 512 --point_radius 2 --view_margin 1.2 --color_q_min 0.02 --color_q_max 0.98 --adaptive_min_band_m 0.015 --adaptive_max_band_m 0.080 --adaptive_band_scale 3.0


# make_top_id_projection.py
# 목적:
# - *_marker.xyz에서 내부 ID 최상단부 포인트만 추출
# - 각 데이터별 마커 기준 평면을 더 정확히 추정
# - 해당 평면의 수직 관찰자 시점으로 2D 평면화
# - YOLO 입력용 binary image 저장
# - raw_z 기반 jet color image 저장
# - 2D -> 3D 복원용 meta.json 저장
#
# 유지 기능:
# - *_top_id.xyz
# - *_top_id_uv.npy
# - *_marker_all_uv.npy
# - *_binary.png
# - *_color.png
# - *_meta.json
#
# 핵심 수정:
# - 기존 1회 평면 추정만 사용하지 않음
# - 1차 평면 추정 후 내부 ID보다 낮은 마커 상판 후보층을 찾음
# - 그 상판 후보층으로 2차 평면을 다시 추정
# - 최종 투영은 2차 평면 기준으로 수행
# - 회전 정렬은 하지 않음
# - raw_z jet 색상 기준은 marker crop 전체 raw_z로 고정

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# IO
# ============================================================
def load_xyz(path: Path) -> np.ndarray:
    """xyz 파일 로드. 앞 3열만 좌표로 사용."""
    P = np.loadtxt(str(path), dtype=np.float64)

    if P.ndim == 1:
        P = P.reshape(-1, 3)

    if P.shape[1] < 3:
        raise RuntimeError(f"xyz 형식 오류: {path}")

    return P[:, :3].astype(np.float64, copy=False)


def save_xyz(path: Path, P: np.ndarray):
    """xyz 저장."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(str(path), P.astype(np.float64), fmt="%.6f")


def scan_xyz_files(in_path: Path):
    """단일 xyz 파일 또는 폴더에서 xyz 파일 목록 생성."""
    if in_path.is_file():
        return [in_path]

    if not in_path.is_dir():
        raise SystemExit(f"[FAIL] 입력 경로 없음: {in_path}")

    files = sorted(in_path.glob("*.xyz"))

    if not files:
        raise SystemExit(f"[FAIL] xyz 파일 없음: {in_path}")

    return files


# ============================================================
# Color
# ============================================================
def jet_color_with_range(values: np.ndarray, vmin: float, vmax: float, invert: bool = False) -> np.ndarray:
    """raw_z 값을 jet 유사 컬러맵으로 변환."""
    v = np.asarray(values, dtype=np.float64)

    if len(v) == 0:
        return np.zeros((0, 3), dtype=np.uint8)

    rng = max(float(vmax) - float(vmin), 1e-12)
    z = (v - float(vmin)) / rng
    z = np.clip(z, 0.0, 1.0)

    if invert:
        z = 1.0 - z

    r = np.clip(1.5 - np.abs(4.0 * z - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * z - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * z - 1.0), 0.0, 1.0)

    rgb = np.stack([r, g, b], axis=1)
    return (rgb * 255.0).astype(np.uint8)


# ============================================================
# Plane / Projection
# ============================================================
def fit_plane_pca(P: np.ndarray):
    """PCA 기반 평면 추정."""
    center = P.mean(axis=0)
    X = P - center

    cov = (X.T @ X) / max(len(X), 1)
    eigvals, eigvecs = np.linalg.eigh(cov)

    normal = eigvecs[:, np.argmin(eigvals)]
    normal = normal / (np.linalg.norm(normal) + 1e-12)

    if normal[2] < 0:
        normal = -normal

    return center, normal


def align_normal_direction(normal: np.ndarray, ref_normal: np.ndarray):
    """2차 평면 normal 방향이 1차 평면 normal과 반대가 되지 않게 정렬."""
    if np.dot(normal, ref_normal) < 0:
        normal = -normal
    return normal / (np.linalg.norm(normal) + 1e-12)


def signed_height(P: np.ndarray, plane_center: np.ndarray, normal: np.ndarray):
    """평면 기준 signed height 계산."""
    return (P - plane_center[None, :]) @ normal


def make_local_basis_from_global(normal: np.ndarray):
    """
    관찰 방향 = 평면 normal.
    u축 = 전역 x축을 평면 위로 투영.
    v축 = normal과 u축의 외적.
    """
    normal = normal / (np.linalg.norm(normal) + 1e-12)

    ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    u = ref - np.dot(ref, normal) * normal

    if np.linalg.norm(u) < 1e-8:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        u = ref - np.dot(ref, normal) * normal

    u = u / (np.linalg.norm(u) + 1e-12)

    v = np.cross(normal, u)
    v = v / (np.linalg.norm(v) + 1e-12)

    return u, v


def make_local_basis_pca(P_all: np.ndarray, normal: np.ndarray):
    """선택 옵션용 PCA 기반 local 축."""
    center = P_all.mean(axis=0)
    X = P_all - center

    X_proj = X - ((X @ normal)[:, None] * normal[None, :])
    cov = (X_proj.T @ X_proj) / max(len(X_proj), 1)
    eigvals, eigvecs = np.linalg.eigh(cov)

    order = np.argsort(eigvals)[::-1]
    u = eigvecs[:, order[0]]
    u = u - np.dot(u, normal) * normal

    if np.linalg.norm(u) < 1e-8:
        return make_local_basis_from_global(normal)

    u = u / (np.linalg.norm(u) + 1e-12)

    v = np.cross(normal, u)
    v = v / (np.linalg.norm(v) + 1e-12)

    return u, v


def project_to_uv(P: np.ndarray, origin: np.ndarray, u: np.ndarray, v: np.ndarray):
    """3D point를 local 평면 좌표 u/v로 직교투영."""
    X = P - origin[None, :]
    uv = np.stack([X @ u, X @ v], axis=1)
    return uv.astype(np.float64)


# ============================================================
# Height layer selection
# ============================================================
def select_top_id_layer_fixed(h: np.ndarray, args):
    """기존 fixed 방식."""
    top_thr = float(np.quantile(h, float(args.top_candidate_q)))
    high_mask = h >= top_thr
    high_h = h[high_mask]

    if len(high_h) < 3:
        center = top_thr
        band = float(args.height_band_m)
        mask = np.zeros((len(h),), dtype=bool)
        status = "fixed_too_few_high_candidates"
    else:
        center = float(np.median(high_h))
        band = float(args.height_band_m)
        mask = np.abs(h - center) <= band
        status = "fixed_ok"

    selected = h[mask]

    info = {
        "height_mode": "fixed",
        "status": status,
        "top_candidate_q": float(args.top_candidate_q),
        "top_candidate_threshold": float(top_thr),
        "id_height_center": float(center),
        "height_band_m": float(band),
        "selected_height_min": float(selected.min()) if len(selected) > 0 else None,
        "selected_height_max": float(selected.max()) if len(selected) > 0 else None,
        "selected_points": int(np.sum(mask)),
    }

    return mask, center, band, info


def select_top_id_layer_adaptive(h: np.ndarray, args):
    """
    파일별 adaptive height layer selection.
    높이 분포에서 높은 쪽의 밀집층을 찾아 내부 ID 최상단부로 선택.
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)

    if n < 10:
        return select_top_id_layer_fixed(h, args)

    q_low = float(args.adaptive_search_q)
    q_high = float(args.adaptive_upper_q)

    h_min = float(np.quantile(h, q_low))
    h_max = float(np.quantile(h, q_high))

    if not np.isfinite(h_min) or not np.isfinite(h_max) or h_max <= h_min:
        return select_top_id_layer_fixed(h, args)

    bins = int(max(args.adaptive_bins, 8))
    hist, edges = np.histogram(h, bins=bins, range=(h_min, h_max))

    min_pts_abs = int(args.adaptive_min_points)
    min_pts_ratio = int(np.ceil(float(args.adaptive_min_ratio) * n))
    min_pts = max(min_pts_abs, min_pts_ratio)

    chosen_bin = None

    # 높은 쪽에서부터 충분히 점이 모인 층 선택
    for i in range(len(hist) - 1, -1, -1):
        if hist[i] >= min_pts:
            chosen_bin = i
            break

    if chosen_bin is None:
        mask, center, band, info = select_top_id_layer_fixed(h, args)
        info["adaptive_status"] = "fallback_no_dense_bin"
        return mask, center, band, info

    main_count = max(int(hist[chosen_bin]), 1)
    neighbor_ratio = float(args.adaptive_neighbor_ratio)

    left = chosen_bin
    while left - 1 >= 0 and hist[left - 1] >= main_count * neighbor_ratio:
        left -= 1

    right = chosen_bin
    while right + 1 < len(hist) and hist[right + 1] >= main_count * neighbor_ratio:
        right += 1

    layer_lo = float(edges[left])
    layer_hi = float(edges[right + 1])

    layer_mask_for_stats = (h >= layer_lo) & (h <= layer_hi)
    layer_h = h[layer_mask_for_stats]

    if len(layer_h) < min_pts:
        mask, center, band, info = select_top_id_layer_fixed(h, args)
        info["adaptive_status"] = "fallback_too_few_layer_points"
        return mask, center, band, info

    center = float(np.median(layer_h))

    mad = float(np.median(np.abs(layer_h - center)))
    robust_sigma = 1.4826 * mad

    band = max(
        float(args.adaptive_min_band_m),
        float(args.adaptive_band_scale) * robust_sigma,
    )
    band = min(band, float(args.adaptive_max_band_m))

    mask = np.abs(h - center) <= band

    selected_count = int(np.sum(mask))
    selected_ratio = selected_count / max(n, 1)

    if selected_count < min_pts:
        band = min(float(args.adaptive_max_band_m), max(band * 1.5, float(args.height_band_m)))
        mask = np.abs(h - center) <= band
        selected_count = int(np.sum(mask))
        selected_ratio = selected_count / max(n, 1)

    if selected_count < min_pts:
        fixed_mask, fixed_center, fixed_band, fixed_info = select_top_id_layer_fixed(h, args)
        fixed_info["adaptive_status"] = "fallback_selected_too_few"
        fixed_info["adaptive_selected_points"] = int(selected_count)
        return fixed_mask, fixed_center, fixed_band, fixed_info

    max_ratio = float(args.adaptive_max_selected_ratio)

    if selected_ratio > max_ratio:
        band_shrink = max(float(args.adaptive_min_band_m), band * 0.7)
        mask_shrink = np.abs(h - center) <= band_shrink

        if np.sum(mask_shrink) >= min_pts:
            band = band_shrink
            mask = mask_shrink
            selected_count = int(np.sum(mask))
            selected_ratio = selected_count / max(n, 1)

    selected_h = h[mask]

    info = {
        "height_mode": "adaptive",
        "status": "adaptive_ok",
        "adaptive_search_q": float(args.adaptive_search_q),
        "adaptive_upper_q": float(args.adaptive_upper_q),
        "adaptive_bins": int(bins),
        "adaptive_min_points": int(min_pts),
        "adaptive_chosen_bin": int(chosen_bin),
        "adaptive_bin_count": int(hist[chosen_bin]),
        "adaptive_layer_bin_left": int(left),
        "adaptive_layer_bin_right": int(right),
        "adaptive_layer_lo": float(layer_lo),
        "adaptive_layer_hi": float(layer_hi),
        "adaptive_layer_points": int(len(layer_h)),
        "adaptive_mad": float(mad),
        "adaptive_robust_sigma": float(robust_sigma),
        "id_height_center": float(center),
        "height_band_m": float(band),
        "selected_height_min": float(selected_h.min()) if len(selected_h) > 0 else None,
        "selected_height_max": float(selected_h.max()) if len(selected_h) > 0 else None,
        "selected_points": int(selected_count),
        "selected_ratio": float(selected_ratio),
    }

    return mask, center, band, info


def select_top_id_layer(h: np.ndarray, args):
    """height_mode에 따라 fixed/adaptive 선택."""
    if args.height_mode == "fixed":
        return select_top_id_layer_fixed(h, args)

    return select_top_id_layer_adaptive(h, args)


# ============================================================
# 2-stage plane refinement
# ============================================================
def select_plate_support_layer(P: np.ndarray, h: np.ndarray, id_center: float, args):
    """
    2차 평면 추정용 마커 상판 후보층 선택.

    원리:
    - 내부 ID 최상단부보다 낮은 높이 영역에서
    - 높은 쪽부터 충분히 밀집된 층을 찾음
    - 그 층을 마커 상판 후보로 보고 2차 평면 추정에 사용
    """
    n = len(h)

    upper = float(id_center) - float(args.plate_gap_min_m)
    lower = float(np.quantile(h, float(args.plate_search_q_low)))

    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        return np.zeros((n,), dtype=bool), {
            "plate_status": "invalid_range",
            "plate_lower": lower,
            "plate_upper": upper,
            "plate_points": 0,
        }

    candidate_mask = (h >= lower) & (h <= upper)
    candidate_h = h[candidate_mask]

    min_pts = max(
        int(args.plate_min_points),
        int(np.ceil(float(args.plate_min_ratio) * n)),
    )

    if len(candidate_h) < min_pts:
        return np.zeros((n,), dtype=bool), {
            "plate_status": "too_few_candidates",
            "plate_lower": lower,
            "plate_upper": upper,
            "plate_candidate_points": int(len(candidate_h)),
            "plate_points": 0,
        }

    bins = int(max(args.plate_bins, 8))
    hist, edges = np.histogram(candidate_h, bins=bins, range=(lower, upper))

    chosen_bin = None

    # 내부 ID 바로 아래의 높은 밀집층을 상판 후보로 선택
    for i in range(len(hist) - 1, -1, -1):
        if hist[i] >= min_pts:
            chosen_bin = i
            break

    if chosen_bin is None:
        return np.zeros((n,), dtype=bool), {
            "plate_status": "no_dense_plate_bin",
            "plate_lower": lower,
            "plate_upper": upper,
            "plate_candidate_points": int(len(candidate_h)),
            "plate_points": 0,
        }

    main_count = max(int(hist[chosen_bin]), 1)
    neighbor_ratio = float(args.plate_neighbor_ratio)

    left = chosen_bin
    while left - 1 >= 0 and hist[left - 1] >= main_count * neighbor_ratio:
        left -= 1

    right = chosen_bin
    while right + 1 < len(hist) and hist[right + 1] >= main_count * neighbor_ratio:
        right += 1

    plate_lo = float(edges[left])
    plate_hi = float(edges[right + 1])

    plate_mask = (h >= plate_lo) & (h <= plate_hi)

    if np.sum(plate_mask) < min_pts:
        return np.zeros((n,), dtype=bool), {
            "plate_status": "too_few_final_plate_points",
            "plate_lower": lower,
            "plate_upper": upper,
            "plate_lo": plate_lo,
            "plate_hi": plate_hi,
            "plate_points": int(np.sum(plate_mask)),
        }

    return plate_mask, {
        "plate_status": "ok",
        "plate_lower": lower,
        "plate_upper": upper,
        "plate_lo": plate_lo,
        "plate_hi": plate_hi,
        "plate_bins": int(bins),
        "plate_chosen_bin": int(chosen_bin),
        "plate_bin_count": int(hist[chosen_bin]),
        "plate_left_bin": int(left),
        "plate_right_bin": int(right),
        "plate_points": int(np.sum(plate_mask)),
        "plate_min_points": int(min_pts),
    }


def estimate_final_plane(P: np.ndarray, args):
    """
    각 데이터별 최종 마커 평면 추정.

    1. raw z 하위 plane_fit_q로 1차 평면 추정
    2. 1차 평면 기준 높이로 내부 ID 최상단부 대략 선택
    3. 내부 ID보다 낮은 마커 상판 후보층 선택
    4. 상판 후보층으로 2차 평면 추정
    5. 실패 시 1차 평면 사용
    """
    z = P[:, 2]

    z_cut = np.quantile(z, float(args.plane_fit_q))
    plane_fit_points = P[z <= z_cut]

    if len(plane_fit_points) < 10:
        plane_fit_points = P

    plane_center_1, normal_1 = fit_plane_pca(plane_fit_points)
    h1 = signed_height(P, plane_center_1, normal_1)

    # 1차 높이 기준으로 ID 중심만 대략 추정
    _, id_center_1, _, layer_info_1 = select_top_id_layer(h1, args)

    use_refined = False
    plate_mask = np.zeros((len(P),), dtype=bool)
    plate_info = {
        "plate_status": "disabled",
        "plate_points": 0,
    }

    plane_center_final = plane_center_1
    normal_final = normal_1

    if not args.no_plane_refine:
        plate_mask, plate_info = select_plate_support_layer(P, h1, id_center_1, args)

        if np.sum(plate_mask) >= max(10, int(args.plate_min_points)):
            plane_center_2, normal_2 = fit_plane_pca(P[plate_mask])
            normal_2 = align_normal_direction(normal_2, normal_1)

            # 2차 평면이 1차 평면과 너무 다르면 잘못 잡힌 것으로 보고 fallback
            angle_cos = float(np.clip(np.dot(normal_1, normal_2), -1.0, 1.0))
            angle_deg = float(np.degrees(np.arccos(angle_cos)))

            plate_info["plane_refine_angle_deg"] = angle_deg

            if angle_deg <= float(args.max_refine_angle_deg):
                plane_center_final = plane_center_2
                normal_final = normal_2
                use_refined = True
            else:
                plate_info["plate_status"] = "rejected_large_angle"
        else:
            plate_info["plane_refine_angle_deg"] = None

    final_info = {
        "plane_refine_used": bool(use_refined),
        "coarse_plane_points": int(len(plane_fit_points)),
        "coarse_plane_center": plane_center_1.tolist(),
        "coarse_plane_normal": normal_1.tolist(),
        "coarse_id_layer": layer_info_1,
        "plate_support": plate_info,
    }

    return plane_center_final, normal_final, h1, plate_mask, final_info


# ============================================================
# Top ID extraction
# ============================================================
def extract_top_id_points(P: np.ndarray, args):
    """
    내부 ID 최상단부 추출.

    1. 각 파일별 최종 마커 평면 추정
    2. 최종 평면 기준으로 h 재계산
    3. fixed/adaptive 방식으로 내부 ID 최상단부 선택
    4. 최종 평면의 normal 방향에서 관찰하는 local u/v 좌표로 투영
    """
    if len(P) < 10:
        raise RuntimeError("포인트 수가 너무 적습니다.")

    plane_center, normal, h_coarse, plate_mask, plane_debug = estimate_final_plane(P, args)

    # 최종 평면 기준 높이
    h = signed_height(P, plane_center, normal)

    # 내부 ID 최상단부 선택
    id_mask, id_height_center, height_band_used, layer_info = select_top_id_layer(h, args)

    P_top = P[id_mask]

    # local 좌표 원점은 실제 평면 위의 점인 plane_center 사용
    marker_origin = plane_center.copy()

    if args.basis_mode == "global":
        u_axis, v_axis = make_local_basis_from_global(normal)
    else:
        u_axis, v_axis = make_local_basis_pca(P, normal)

    uv_all = project_to_uv(P, marker_origin, u_axis, v_axis)
    uv_top = uv_all[id_mask]

    selected_h = h[id_mask]

    debug = {
        "total_points": int(len(P)),
        "basis_mode": str(args.basis_mode),
        "height_mode": str(args.height_mode),
        "top_candidate_q": float(args.top_candidate_q),
        "height_band_m_input": float(args.height_band_m),
        "height_band_m_used": float(height_band_used),
        "id_height_center": float(id_height_center),
        "top_id_points": int(len(P_top)),
        "selected_height_min": float(selected_h.min()) if len(selected_h) > 0 else None,
        "selected_height_max": float(selected_h.max()) if len(selected_h) > 0 else None,
        "height_min": float(h.min()),
        "height_mean": float(h.mean()),
        "height_max": float(h.max()),
        "height_q50": float(np.quantile(h, 0.50)),
        "height_q70": float(np.quantile(h, 0.70)),
        "height_q80": float(np.quantile(h, 0.80)),
        "height_q85": float(np.quantile(h, 0.85)),
        "height_q90": float(np.quantile(h, 0.90)),
        "height_q95": float(np.quantile(h, 0.95)),
        "height_q99": float(np.quantile(h, 0.99)),
        "layer_selection": layer_info,
        "plane_estimation": plane_debug,
    }

    return {
        "P_top": P_top,
        "uv_top": uv_top,
        "uv_all": uv_all,
        "height_all": h,
        "id_mask": id_mask,
        "plate_mask": plate_mask,
        "plane_center": plane_center,
        "normal": normal,
        "marker_origin": marker_origin,
        "u_axis": u_axis,
        "v_axis": v_axis,
        "debug": debug,
    }


# ============================================================
# Image rasterization
# ============================================================
def compute_view_bounds(uv_all: np.ndarray, margin_ratio: float):
    """marker 전체 기준으로 이미지 범위 계산."""
    if len(uv_all) == 0:
        return -1.0, 1.0, -1.0, 1.0

    mn = uv_all.min(axis=0)
    mx = uv_all.max(axis=0)

    center = 0.5 * (mn + mx)
    half = 0.5 * (mx - mn)

    half[0] = max(float(half[0]), 0.05)
    half[1] = max(float(half[1]), 0.05)

    half = half * max(float(margin_ratio), 1.0)

    xmin = center[0] - half[0]
    xmax = center[0] + half[0]
    ymin = center[1] - half[1]
    ymax = center[1] + half[1]

    return xmin, xmax, ymin, ymax


def uv_to_pixel(uv: np.ndarray, bounds, image_size: int):
    """uv 좌표를 image pixel 좌표로 변환."""
    xmin, xmax, ymin, ymax = bounds
    W = int(image_size)
    H = int(image_size)

    if len(uv) == 0:
        return (
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=bool),
        )

    x = uv[:, 0]
    y = uv[:, 1]

    px = np.round((x - xmin) / max(xmax - xmin, 1e-12) * (W - 1)).astype(np.int32)
    py = np.round((ymax - y) / max(ymax - ymin, 1e-12) * (H - 1)).astype(np.int32)

    valid = (px >= 0) & (px < W) & (py >= 0) & (py < H)

    return px, py, valid


def disk_offsets(radius: int):
    """포인트 두께용 원형 offset."""
    radius = int(max(radius, 0))

    if radius == 0:
        return [(0, 0)]

    offs = []
    r2 = radius * radius

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy <= r2:
                offs.append((dx, dy))

    return offs


def rasterize_binary(px, py, valid, image_size: int, point_radius: int):
    """YOLO 입력용 binary image 생성."""
    H = int(image_size)
    W = int(image_size)

    img = np.zeros((H, W), dtype=np.uint8)
    offs = disk_offsets(point_radius)

    for x, y, ok in zip(px, py, valid):
        if not ok:
            continue

        for dx, dy in offs:
            xx = x + dx
            yy = y + dy

            if 0 <= xx < W and 0 <= yy < H:
                img[yy, xx] = 255

    return img


def rasterize_color(px, py, valid, rgb, image_size: int, point_radius: int):
    """raw_z jet color image 생성."""
    H = int(image_size)
    W = int(image_size)

    img = np.zeros((H, W, 3), dtype=np.uint8)
    offs = disk_offsets(point_radius)

    for x, y, ok, c in zip(px, py, valid, rgb):
        if not ok:
            continue

        for dx, dy in offs:
            xx = x + dx
            yy = y + dy

            if 0 <= xx < W and 0 <= yy < H:
                img[yy, xx, :] = c

    return img


# ============================================================
# Save
# ============================================================
def save_png_gray(path: Path, img: np.ndarray):
    """grayscale png 저장."""
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(str(path), img, cmap="gray", vmin=0, vmax=255)


def save_png_rgb(path: Path, img: np.ndarray):
    """rgb png 저장."""
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(str(path), img)


def save_meta_json(path: Path, meta: dict):
    """json 저장."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


# ============================================================
# Process
# ============================================================
def process_one(fp: Path, out_dir: Path, args):
    P = load_xyz(fp)
    result = extract_top_id_points(P, args)

    P_top = result["P_top"]
    uv_top = result["uv_top"]
    uv_all = result["uv_all"]

    plane_center = result["plane_center"]
    normal = result["normal"]
    marker_origin = result["marker_origin"]
    u_axis = result["u_axis"]
    v_axis = result["v_axis"]
    debug = result["debug"]

    bounds = compute_view_bounds(uv_all, args.view_margin)

    px, py, valid = uv_to_pixel(
        uv=uv_top,
        bounds=bounds,
        image_size=args.image_size,
    )

    binary_img = rasterize_binary(
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

        rgb = jet_color_with_range(
            top_values,
            cmin,
            cmax,
            invert=bool(args.invert_jet),
        )
    else:
        cmin = 0.0
        cmax = 1.0
        rgb = np.zeros((0, 3), dtype=np.uint8)

    color_img = rasterize_color(
        px=px,
        py=py,
        valid=valid,
        rgb=rgb,
        image_size=args.image_size,
        point_radius=args.point_radius,
    )

    xmin, xmax, ymin, ymax = bounds

    pixel_size_u = float((xmax - xmin) / max(args.image_size - 1, 1))
    pixel_size_v = float((ymax - ymin) / max(args.image_size - 1, 1))

    meta = {
        "source_file": str(fp),

        "image_size": int(args.image_size),
        "point_radius": int(args.point_radius),
        "view_margin": float(args.view_margin),

        "plane_fit_q": float(args.plane_fit_q),
        "height_mode": str(args.height_mode),
        "top_candidate_q": float(args.top_candidate_q),
        "height_band_m": float(args.height_band_m),
        "basis_mode": str(args.basis_mode),

        "plane_refine_enabled": not bool(args.no_plane_refine),
        "align_2d": False,
        "align_method": "none",

        "color_mode": "raw_z",
        "color_ref": "marker_all_raw_z",
        "invert_jet": bool(args.invert_jet),
        "color_min": float(cmin),
        "color_max": float(cmax),
        "color_q_min": float(args.color_q_min),
        "color_q_max": float(args.color_q_max),

        # 3D 평면 정보
        "plane_center": plane_center.tolist(),
        "plane_normal": normal.tolist(),
        "marker_origin": marker_origin.tolist(),

        # 이미지 좌표계의 3D 축
        "image_u_axis_3d": u_axis.tolist(),
        "image_v_axis_3d": v_axis.tolist(),

        # 이미지 좌표 범위
        "u_min": float(xmin),
        "u_max": float(xmax),
        "v_min": float(ymin),
        "v_max": float(ymax),
        "pixel_size_u_m": pixel_size_u,
        "pixel_size_v_m": pixel_size_v,

        # 2D -> 3D 역변환 공식
        "back_projection_note": {
            "pixel_to_uv": "u = u_min + px * pixel_size_u_m, v = v_max - py * pixel_size_v_m",
            "point_on_plane_3d": "marker_origin + u * image_u_axis_3d + v * image_v_axis_3d",
            "direction_3d": "du * image_u_axis_3d + dv * image_v_axis_3d",
        },

        "debug": debug,
    }

    stem = fp.stem

    out_top_xyz = out_dir / f"{stem}_top_id.xyz"
    out_top_uv = out_dir / f"{stem}_top_id_uv.npy"
    out_all_uv = out_dir / f"{stem}_marker_all_uv.npy"
    out_binary = out_dir / f"{stem}_binary.png"
    out_color = out_dir / f"{stem}_color.png"
    out_meta = out_dir / f"{stem}_meta.json"

    save_xyz(out_top_xyz, P_top)
    np.save(str(out_top_uv), uv_top.astype(np.float32))
    np.save(str(out_all_uv), uv_all.astype(np.float32))

    save_png_gray(out_binary, binary_img)
    save_png_rgb(out_color, color_img)
    save_meta_json(out_meta, meta)

    plane_used = debug["plane_estimation"]["plane_refine_used"]

    print(
        f"[OK] {fp.name} | "
        f"total={debug['total_points']} | "
        f"top_id_points={debug['top_id_points']} | "
        f"id_height={debug['id_height_center']:.6f} | "
        f"band_used={debug['height_band_m_used']:.6f} | "
        f"height_mode={args.height_mode} | "
        f"plane_refine={plane_used} | "
        f"basis={args.basis_mode} | "
        f"color_raw_z_range=({cmin:.4f}, {cmax:.4f})"
    )


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--in_path", required=True, help="*_marker.xyz 파일 또는 marker_points 폴더")
    ap.add_argument("--out", required=True, help="출력 폴더")

    # 평면 / 최상단부 추출
    ap.add_argument("--plane_fit_q", type=float, default=0.80)

    ap.add_argument(
        "--no_plane_refine",
        action="store_true",
        help="2차 평면 재추정 비활성화",
    )

    ap.add_argument("--plate_gap_min_m", type=float, default=0.005)
    ap.add_argument("--plate_search_q_low", type=float, default=0.20)
    ap.add_argument("--plate_bins", type=int, default=64)
    ap.add_argument("--plate_min_points", type=int, default=30)
    ap.add_argument("--plate_min_ratio", type=float, default=0.005)
    ap.add_argument("--plate_neighbor_ratio", type=float, default=0.35)
    ap.add_argument("--max_refine_angle_deg", type=float, default=25.0)

    ap.add_argument(
        "--height_mode",
        choices=["fixed", "adaptive"],
        default="adaptive",
        help="fixed=기존 방식, adaptive=파일별 높이층 자동 선택. 기본 adaptive",
    )

    ap.add_argument("--top_candidate_q", type=float, default=0.85)
    ap.add_argument("--height_band_m", type=float, default=0.03)

    # adaptive mode
    ap.add_argument("--adaptive_search_q", type=float, default=0.60)
    ap.add_argument("--adaptive_upper_q", type=float, default=0.995)
    ap.add_argument("--adaptive_bins", type=int, default=64)
    ap.add_argument("--adaptive_min_points", type=int, default=20)
    ap.add_argument("--adaptive_min_ratio", type=float, default=0.003)
    ap.add_argument("--adaptive_neighbor_ratio", type=float, default=0.35)
    ap.add_argument("--adaptive_band_scale", type=float, default=2.5)
    ap.add_argument("--adaptive_min_band_m", type=float, default=0.010)
    ap.add_argument("--adaptive_max_band_m", type=float, default=0.060)
    ap.add_argument("--adaptive_max_selected_ratio", type=float, default=0.35)

    # local basis
    ap.add_argument(
        "--basis_mode",
        choices=["global", "pca"],
        default="global",
        help="global=전역축을 평면에 투영, pca=marker crop 분포 기준. 기본 global",
    )

    # image
    ap.add_argument("--image_size", type=int, default=512)
    ap.add_argument("--point_radius", type=int, default=2)
    ap.add_argument("--view_margin", type=float, default=1.2)

    # color
    ap.add_argument("--color_q_min", type=float, default=0.02)
    ap.add_argument("--color_q_max", type=float, default=0.98)
    ap.add_argument(
        "--invert_jet",
        action="store_true",
        help="jet 색상 방향 반전. 기본은 높은 z=빨강",
    )

    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = scan_xyz_files(in_path)

    print(f"[INFO] input files={len(files)}")
    print(f"[INFO] out={out_dir}")
    print(f"[INFO] plane_fit_q={args.plane_fit_q}")
    print(f"[INFO] plane_refine={not args.no_plane_refine}")
    print(f"[INFO] height_mode={args.height_mode}")
    print(f"[INFO] top_candidate_q={args.top_candidate_q}")
    print(f"[INFO] height_band_m={args.height_band_m}")
    print(f"[INFO] basis_mode={args.basis_mode}")
    print(f"[INFO] image_size={args.image_size}")
    print(f"[INFO] point_radius={args.point_radius}")
    print(f"[INFO] view_margin={args.view_margin}")
    print("[INFO] align_2d=disabled")
    print("[INFO] color_mode=raw_z")
    print("[INFO] color_ref=marker_all_raw_z")
    print(f"[INFO] invert_jet={args.invert_jet}")

    ok_count = 0

    for fp in files:
        try:
            process_one(fp, out_dir, args)
            ok_count += 1
        except Exception as e:
            print(f"[FAIL] {fp.name}: {e}")

    print(f"[DONE] ok={ok_count}/{len(files)}")


if __name__ == "__main__":
    main()