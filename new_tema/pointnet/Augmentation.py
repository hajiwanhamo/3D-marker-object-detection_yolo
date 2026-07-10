# import os, csv, math, random
# from pathlib import Path
# import numpy as np
# import open3d as o3d

# # -------------------- 경로 --------------------
# ROOT = r"C:\Users\yncit\OneDrive\Desktop\jiwan\3D-Marker-Pose-Estimation\CAD_to_PointCloud_conversion\dataset"
# SRC  = fr"{ROOT}\output"              # 마커만 들어있는 xyz
# OUT  = fr"{ROOT}\joint"
# TRN  = fr"{OUT}\train\points"
# VAL  = fr"{OUT}\val\points"

# # -------------------- 고정 파라미터 --------------------
# SEED = 42
# K = 100
# MINPTS = 100_000
# TRAIN_RATIO = 0.8

# # 회전: yaw만 허용 (기울기 제거)
# YAW = (0, 360)

# # z 방향 변형 제거
# Z_SHIFT_RANGE = (0.0, 0.0)
# Z_JITTER_STD  = 0.0

# # xy 평행이동만 허용
# TX_RANGE = (-10, 10)
# TY_RANGE = (-10, 10)

# # -------------------- 배경(완전 평면) --------------------
# BG_ENABLE = True
# BG_COUNT = 800_000
# BG_XY_MARGIN = 10.0

# BG_THICK_STD  = 0.0
# BG_THICK_CLIP = 0.0

# # -------------------- ✅ 옆면 제거(핵심) --------------------
# # "상단면 2장"만 남김: (몸체 상단면 + ID 상단면)
# # - 퍼센타일로 두 레이어 중심 z를 잡고
# # - 각 레이어 주변 band 두께로 슬랩 필터
# KEEP_TWO_TOP_LAYERS = True
# Z_BODY_Q = 0.70      # 몸체 상단면(대략)
# Z_ID_Q   = 0.98      # ID 상단면(대략)
# Z_BAND_BODY = 0.012  # 몸체 상단면 두께 (unit_sphere 기준)
# Z_BAND_ID   = 0.012  # ID 상단면 두께
# MIN_KEEP_BODY = 500  # 몸체 상단면 최소
# MIN_KEEP_ID   = 50   # ID 상단면 최소

# random.seed(SEED)
# np.random.seed(SEED)

# # -------------------- 유틸 --------------------
# def center_scale(P):
#     C = P.mean(0)
#     S = max(np.linalg.norm(P - C, axis=1).max(), 1e-12)
#     return C, S

# def norm_by(C, S, P):
#     return (P - C) / S

# def Rz(a):
#     r = math.radians(a)
#     return np.array([[math.cos(r),-math.sin(r),0],
#                      [math.sin(r), math.cos(r),0],
#                      [0,0,1]])

# def ensure_min(P, n):
#     if len(P) >= n: 
#         return P
#     idx = np.random.choice(len(P), n, replace=True)
#     return P[idx]

# # -------------------- 배경 생성 (초창기 코드 그대로) --------------------
# def make_background(A):
#     x, y, z = A[:,0], A[:,1], A[:,2]
#     xmin, xmax = x.min(), x.max()
#     ymin, ymax = y.min(), y.max()

#     cx, cy = (xmin+xmax)/2, (ymin+ymax)/2
#     w = (xmax-xmin)*(1+BG_XY_MARGIN)
#     h = (ymax-ymin)*(1+BG_XY_MARGIN)

#     bx = np.random.uniform(cx-w/2, cx+w/2, BG_COUNT)
#     by = np.random.uniform(cy-h/2, cy+h/2, BG_COUNT)

#     # ✅ 필터링 "전" 최저 z에 배경을 깔아야 겹침 없음
#     z0 = z.min()
#     bz = np.full_like(bx, z0)

#     return np.stack([bx,by,bz],1).astype(np.float32), z0

# # -------------------- ✅ 옆면 제거: 상단면 2장만 남기기 --------------------
# def keep_top_two_layers(P):
#     z = P[:,2]
#     z_body = float(np.quantile(z, Z_BODY_Q))
#     z_id   = float(np.quantile(z, Z_ID_Q))

#     body = P[np.abs(z - z_body) <= float(Z_BAND_BODY)]
#     top  = P[np.abs(z - z_id)   <= float(Z_BAND_ID)]

#     # 최소 포인트 보장(부족하면 가장 가까운 점으로 채움)
#     if len(body) < MIN_KEEP_BODY:
#         order = np.argsort(np.abs(z - z_body))
#         body = P[order[:MIN_KEEP_BODY]]
#     if len(top) < MIN_KEEP_ID:
#         order = np.argsort(np.abs(z - z_id))
#         top = P[order[:MIN_KEEP_ID]]

#     return np.vstack([body, top]).astype(np.float32)

# # -------------------- 증강 1회 --------------------
# def augment_one(xyz):
#     P = np.loadtxt(xyz).astype(np.float32)

#     C,S = center_scale(P)
#     P = norm_by(C,S,P)

#     yaw = random.uniform(*YAW)
#     P = P @ Rz(yaw).T

#     # xy 이동만
#     P[:,0]+=random.uniform(*TX_RANGE)
#     P[:,1]+=random.uniform(*TY_RANGE)

#     # ✅ 배경은 "옆면 제거 전"의 P로 생성(겹침 방지)
#     bg, z0 = make_background(P)

#     # ✅ 옆면 제거(상단면 2장만 남김)
#     if KEEP_TWO_TOP_LAYERS:
#         P = keep_top_two_layers(P)

#     # 마커 최소점 확보
#     P = ensure_min(P,MINPTS)

#     # 배경 합치기 + 마스크
#     pts = np.vstack([P,bg])
#     mask = np.concatenate([
#         np.ones(len(P),np.uint8),
#         np.zeros(len(bg),np.uint8)
#     ])

#     return pts, mask

# # -------------------- 메인 --------------------
# def main():
#     for p in [TRN,VAL]:
#         Path(p).mkdir(parents=True,exist_ok=True)

#     wtr = csv.writer(open(Path(OUT)/"train"/"labels.csv","w",newline="",encoding="utf-8"))
#     wva = csv.writer(open(Path(OUT)/"val"/"labels.csv","w",newline="",encoding="utf-8"))
#     wtr.writerow(["filename"])
#     wva.writerow(["filename"])

#     idx=0
#     for f in sorted(os.listdir(SRC)):
#         if not f.endswith(".xyz"): 
#             continue
#         for _ in range(K):
#             pts,mask = augment_one(os.path.join(SRC,f))
#             split = "train" if random.random()<TRAIN_RATIO else "val"
#             out = TRN if split=="train" else VAL
#             name=f"aug_{idx:06d}.xyz"
#             np.savetxt(os.path.join(out,name),pts,fmt="%.6f")
#             np.save(os.path.join(out,name.replace(".xyz",".mask.npy")),mask)
#             (wtr if split=="train" else wva).writerow([name])
#             idx+=1
#     print("DONE:",OUT)

# if __name__=="__main__":
#     main()


# Augmentation.py
# train1용 가상데이터 생성 코드
#
# 목적:
# - dataset/output/*.xyz 마커 원본 point cloud를 이용해
#   dataset/joint/train/points, dataset/joint/val/points 생성
# - train1용 .mask.npy 생성
#
# mask 규칙:
#   1 = 마커 포인트
#   0 = 배경 포인트
#
# 중요:
# - train1용 mask.npy는 여기서 생성
# - label_make.py는 train2/internal ID용이므로 train1 학습 전에 실행하지 말 것
#
# 배경 범위 조절:
# - BG_XY_MARGIN_DEFAULT 또는 실행 옵션 --bg_margin
# - BG_COUNT_DEFAULT 또는 실행 옵션 --bg_count

import argparse
import csv
import math
import random
import shutil
from pathlib import Path

import numpy as np


# ============================================================
# 기본 경로
# ============================================================
ROOT_DEFAULT = r"C:\Users\gkwld\Desktop\3D-Marker-Pose-Estimation\CAD_to_PointCloud_conversion\dataset"

# ============================================================
# 기본 파라미터
# ============================================================
SEED_DEFAULT = 42

# 원본 marker xyz 하나당 생성할 증강 개수
K_DEFAULT = 100

# train / val 비율
TRAIN_RATIO_DEFAULT = 0.8

# 최종 마커 포인트 수
# 이 포인트들은 전부 mask=1
MARKER_KEEP_POINTS_DEFAULT = 100_000

# 배경 포인트 수
# 이 포인트들은 전부 mask=0
BG_COUNT_DEFAULT = 400_000

# 배경 XY 범위
# 최종 배경 폭 = 마커 XY 폭 * (1 + BG_XY_MARGIN)
# 예: 4.0이면 마커 폭의 약 5배 영역
BG_XY_MARGIN_DEFAULT = 4.0

# 배경 z 노이즈
# 처음에는 0.0 권장. 이후 실해역 지면 요철 반영 시 조금씩 증가.
BG_Z_NOISE_STD_DEFAULT = 0.0

# 배경 경사
# 처음에는 0.0 권장.
BG_SLOPE_X_DEFAULT = 0.0
BG_SLOPE_Y_DEFAULT = 0.0

# XY 평행이동 범위
TX_RANGE_DEFAULT = (-10.0, 10.0)
TY_RANGE_DEFAULT = (-10.0, 10.0)

# yaw 회전 범위
YAW_RANGE_DEFAULT = (0.0, 360.0)

# 마커에서 상단부 두 층만 남길지 여부
# 실해역에서 윗면 위주로 찍히는 조건을 반영
KEEP_TWO_TOP_LAYERS_DEFAULT = True

# 몸체 상단면 / 내부 ID 상단면 추정 분위수
Z_BODY_Q_DEFAULT = 0.70
Z_ID_Q_DEFAULT = 0.98

# 상단면으로 인정할 z band 두께
# unit sphere 정규화 이후 기준
Z_BAND_BODY_DEFAULT = 0.012
Z_BAND_ID_DEFAULT = 0.012

# 최소 상단부 포인트 수
MIN_KEEP_BODY_DEFAULT = 500
MIN_KEEP_ID_DEFAULT = 50


# ============================================================
# 유틸
# ============================================================
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def clear_dataset(out_root: Path):
    """
    기존 joint/train, joint/val 내부 파일 제거.
    오래된 mask.npy가 섞이는 것을 방지하기 위함.
    """
    for split in ["train", "val"]:
        split_dir = out_root / split
        points_dir = split_dir / "points"

        if points_dir.exists():
            shutil.rmtree(points_dir)

        ensure_dir(points_dir)

        labels_csv = split_dir / "labels.csv"
        if labels_csv.exists():
            labels_csv.unlink()

    stats_csv = out_root / "augmentation_stats.csv"
    if stats_csv.exists():
        stats_csv.unlink()


def center_scale(P: np.ndarray):
    center = P.mean(axis=0)
    scale = np.linalg.norm(P - center, axis=1).max()
    scale = float(scale) if np.isfinite(scale) and scale > 1e-12 else 1.0
    return center, scale


def normalize_by_center_scale(P: np.ndarray):
    center, scale = center_scale(P)
    return ((P - center) / scale).astype(np.float32)


def rotation_z(angle_deg: float):
    r = math.radians(float(angle_deg))
    c = math.cos(r)
    s = math.sin(r)
    return np.array(
        [
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def ensure_min_points(P: np.ndarray, target_count: int, rng: np.random.Generator):
    """
    포인트 수가 부족하면 중복 샘플링.
    """
    if len(P) <= 0:
        raise RuntimeError("마커 포인트가 비어 있습니다.")

    if len(P) >= target_count:
        idx = rng.choice(len(P), target_count, replace=False)
        return P[idx].astype(np.float32)

    idx = rng.choice(len(P), target_count, replace=True)
    return P[idx].astype(np.float32)


# ============================================================
# 마커 상단부 추출
# ============================================================
def keep_top_two_layers(P: np.ndarray, args):
    """
    마커에서 몸체 상단면 + 내부 ID 상단면을 남김.
    이 함수로 남은 마커 포인트는 전부 mask=1로 저장됨.
    """
    z = P[:, 2]

    z_body = float(np.quantile(z, args.z_body_q))
    z_id = float(np.quantile(z, args.z_id_q))

    body_mask = np.abs(z - z_body) <= float(args.z_band_body)
    id_mask = np.abs(z - z_id) <= float(args.z_band_id)

    body = P[body_mask]
    top = P[id_mask]

    if len(body) < args.min_keep_body:
        order = np.argsort(np.abs(z - z_body))
        body = P[order[:args.min_keep_body]]

    if len(top) < args.min_keep_id:
        order = np.argsort(np.abs(z - z_id))
        top = P[order[:args.min_keep_id]]

    marker = np.vstack([body, top]).astype(np.float32)

    return marker


# ============================================================
# 배경 생성
# ============================================================
def make_background(marker_full: np.ndarray, args, rng: np.random.Generator):
    """
    배경 평면 생성.

    배경 범위 조절 위치:
    - args.bg_margin
    - args.bg_count

    계산식:
    - bg_width  = marker_width  * (1 + bg_margin)
    - bg_height = marker_height * (1 + bg_margin)

    bg_margin이 커질수록 마커 주변 배경 범위가 넓어짐.
    bg_count가 커질수록 배경 포인트가 많아짐.
    """
    x = marker_full[:, 0]
    y = marker_full[:, 1]
    z = marker_full[:, 2]

    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())

    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)

    marker_w = max(xmax - xmin, 1e-6)
    marker_h = max(ymax - ymin, 1e-6)

    bg_w = marker_w * (1.0 + float(args.bg_margin))
    bg_h = marker_h * (1.0 + float(args.bg_margin))

    bx = rng.uniform(cx - bg_w * 0.5, cx + bg_w * 0.5, size=(args.bg_count,))
    by = rng.uniform(cy - bg_h * 0.5, cy + bg_h * 0.5, size=(args.bg_count,))

    # 배경은 마커 전체의 최저 z에 접촉하도록 생성
    z0 = float(z.min())

    bz = np.full((args.bg_count,), z0, dtype=np.float32)

    # 선택적 경사
    if abs(args.bg_slope_x) > 0 or abs(args.bg_slope_y) > 0:
        bz = bz + (
            float(args.bg_slope_x) * (bx - cx)
            + float(args.bg_slope_y) * (by - cy)
        ).astype(np.float32)

    # 선택적 지면 노이즈
    if args.bg_z_noise_std > 0:
        bz = bz + rng.normal(
            loc=0.0,
            scale=float(args.bg_z_noise_std),
            size=(args.bg_count,),
        ).astype(np.float32)

    bg = np.stack([bx, by, bz], axis=1).astype(np.float32)

    debug = {
        "bg_center_x": cx,
        "bg_center_y": cy,
        "marker_w": marker_w,
        "marker_h": marker_h,
        "bg_w": bg_w,
        "bg_h": bg_h,
        "bg_z": z0,
        "bg_count": int(args.bg_count),
        "bg_margin": float(args.bg_margin),
    }

    return bg, debug


# ============================================================
# 증강 1개 생성
# ============================================================
def augment_one(src_xyz: Path, args, rng: np.random.Generator):
    raw = np.loadtxt(str(src_xyz), dtype=np.float32)

    if raw.ndim == 1:
        raw = raw.reshape(-1, 3)

    P = raw[:, :3].astype(np.float32)

    if len(P) <= 0:
        raise RuntimeError(f"비어 있는 xyz: {src_xyz}")

    # 1. 원본 마커 정규화
    P = normalize_by_center_scale(P)

    # 2. yaw 회전
    yaw = rng.uniform(float(args.yaw_min), float(args.yaw_max))
    P = P @ rotation_z(yaw).T

    # 3. XY 평행이동
    tx = rng.uniform(float(args.tx_min), float(args.tx_max))
    ty = rng.uniform(float(args.ty_min), float(args.ty_max))
    P[:, 0] += tx
    P[:, 1] += ty

    # 4. 배경 생성은 상단부 제거 전 전체 마커 기준
    #    그래야 배경이 실제 마커 하단 기준에 놓임.
    bg, bg_debug = make_background(P, args, rng)

    # 5. train1용 마커 포인트 결정
    if args.keep_two_top_layers:
        marker = keep_top_two_layers(P, args)
    else:
        marker = P.copy()

    # 6. 마커 포인트 수 고정
    marker = ensure_min_points(marker, args.marker_keep_points, rng)

    # 7. 최종 point cloud + mask 생성
    if args.bg_count > 0:
        pts = np.vstack([marker, bg]).astype(np.float32)
        mask = np.concatenate(
            [
                np.ones((len(marker),), dtype=np.uint8),
                np.zeros((len(bg),), dtype=np.uint8),
            ],
            axis=0,
        )
    else:
        pts = marker.astype(np.float32)
        mask = np.ones((len(marker),), dtype=np.uint8)

    debug = {
        "source": src_xyz.name,
        "yaw": float(yaw),
        "tx": float(tx),
        "ty": float(ty),
        "marker_points": int(len(marker)),
        "background_points": int(len(bg)),
        "total_points": int(len(pts)),
        "mask_ratio": float(mask.mean()),
        **bg_debug,
    }

    return pts, mask, debug


# ============================================================
# labels.csv / stats.csv
# ============================================================
def open_writer(path: Path, header):
    ensure_dir(path.parent)
    f = open(path, "w", newline="", encoding="utf-8")
    w = csv.writer(f)
    w.writerow(header)
    return f, w


# ============================================================
# main
# ============================================================
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--root", default=ROOT_DEFAULT)
    ap.add_argument("--src", default="")
    ap.add_argument("--out", default="")

    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    ap.add_argument("--k", type=int, default=K_DEFAULT)
    ap.add_argument("--train_ratio", type=float, default=TRAIN_RATIO_DEFAULT)

    ap.add_argument("--marker_keep_points", type=int, default=MARKER_KEEP_POINTS_DEFAULT)

    # 배경 조절 핵심 옵션
    ap.add_argument("--bg_count", type=int, default=BG_COUNT_DEFAULT)
    ap.add_argument("--bg_margin", type=float, default=BG_XY_MARGIN_DEFAULT)

    # 배경 지형 변화 옵션
    ap.add_argument("--bg_z_noise_std", type=float, default=BG_Z_NOISE_STD_DEFAULT)
    ap.add_argument("--bg_slope_x", type=float, default=BG_SLOPE_X_DEFAULT)
    ap.add_argument("--bg_slope_y", type=float, default=BG_SLOPE_Y_DEFAULT)

    # 회전 / 이동
    ap.add_argument("--yaw_min", type=float, default=YAW_RANGE_DEFAULT[0])
    ap.add_argument("--yaw_max", type=float, default=YAW_RANGE_DEFAULT[1])
    ap.add_argument("--tx_min", type=float, default=TX_RANGE_DEFAULT[0])
    ap.add_argument("--tx_max", type=float, default=TX_RANGE_DEFAULT[1])
    ap.add_argument("--ty_min", type=float, default=TY_RANGE_DEFAULT[0])
    ap.add_argument("--ty_max", type=float, default=TY_RANGE_DEFAULT[1])

    # 상단부 추출
    ap.add_argument("--keep_two_top_layers", action="store_true", default=KEEP_TWO_TOP_LAYERS_DEFAULT)
    ap.add_argument("--no_keep_two_top_layers", action="store_false", dest="keep_two_top_layers")

    ap.add_argument("--z_body_q", type=float, default=Z_BODY_Q_DEFAULT)
    ap.add_argument("--z_id_q", type=float, default=Z_ID_Q_DEFAULT)
    ap.add_argument("--z_band_body", type=float, default=Z_BAND_BODY_DEFAULT)
    ap.add_argument("--z_band_id", type=float, default=Z_BAND_ID_DEFAULT)
    ap.add_argument("--min_keep_body", type=int, default=MIN_KEEP_BODY_DEFAULT)
    ap.add_argument("--min_keep_id", type=int, default=MIN_KEEP_ID_DEFAULT)

    # 기존 joint 데이터 제거
    ap.add_argument("--clear", action="store_true")

    args = ap.parse_args()

    root = Path(args.root)
    src = Path(args.src) if args.src else root / "output"
    out = Path(args.out) if args.out else root / "joint"

    train_points = out / "train" / "points"
    val_points = out / "val" / "points"

    if not src.is_dir():
        raise SystemExit(f"[FAIL] SRC 폴더 없음: {src}")

    if args.bg_count < 0:
        raise SystemExit("[FAIL] --bg_count는 0 이상이어야 합니다.")

    if args.marker_keep_points <= 0:
        raise SystemExit("[FAIL] --marker_keep_points는 1 이상이어야 합니다.")

    if not (0.0 < args.train_ratio < 1.0):
        raise SystemExit("[FAIL] --train_ratio는 0~1 사이여야 합니다.")

    if args.clear:
        clear_dataset(out)
    else:
        ensure_dir(train_points)
        ensure_dir(val_points)

    rng = np.random.default_rng(int(args.seed))
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))

    xyz_files = sorted(src.glob("*.xyz"))

    if not xyz_files:
        raise SystemExit(f"[FAIL] SRC에 .xyz 없음: {src}")

    train_csv_f, train_writer = open_writer(out / "train" / "labels.csv", ["filename"])
    val_csv_f, val_writer = open_writer(out / "val" / "labels.csv", ["filename"])

    stats_f, stats_writer = open_writer(
        out / "augmentation_stats.csv",
        [
            "filename",
            "split",
            "source",
            "total_points",
            "marker_points",
            "background_points",
            "mask_ratio",
            "bg_margin",
            "bg_count",
            "bg_w",
            "bg_h",
            "bg_z",
            "yaw",
            "tx",
            "ty",
        ],
    )

    idx = 0
    try:
        for src_xyz in xyz_files:
            for _ in range(int(args.k)):
                pts, mask, debug = augment_one(src_xyz, args, rng)

                split = "train" if rng.random() < float(args.train_ratio) else "val"
                points_dir = train_points if split == "train" else val_points
                writer = train_writer if split == "train" else val_writer

                name = f"aug_{idx:06d}.xyz"

                xyz_out = points_dir / name
                mask_out = points_dir / name.replace(".xyz", ".mask.npy")

                np.savetxt(str(xyz_out), pts, fmt="%.6f")
                np.save(str(mask_out), mask.astype(np.uint8))

                writer.writerow([name])

                stats_writer.writerow(
                    [
                        name,
                        split,
                        debug["source"],
                        debug["total_points"],
                        debug["marker_points"],
                        debug["background_points"],
                        f"{debug['mask_ratio']:.8f}",
                        f"{debug['bg_margin']:.6f}",
                        debug["bg_count"],
                        f"{debug['bg_w']:.6f}",
                        f"{debug['bg_h']:.6f}",
                        f"{debug['bg_z']:.6f}",
                        f"{debug['yaw']:.6f}",
                        f"{debug['tx']:.6f}",
                        f"{debug['ty']:.6f}",
                    ]
                )

                if idx % 10 == 0:
                    print(
                        f"[GEN] {name} split={split} "
                        f"marker={debug['marker_points']} "
                        f"bg={debug['background_points']} "
                        f"mask_ratio={debug['mask_ratio']:.4f} "
                        f"bg_margin={debug['bg_margin']}"
                    )

                idx += 1

    finally:
        train_csv_f.close()
        val_csv_f.close()
        stats_f.close()

    print("")
    print(f"[DONE] total_generated={idx}")
    print(f"[OUT] {out}")
    print("")
    print("[중요]")
    print("train1 학습 전에는 label_make.py를 실행하지 마세요.")
    print("train1용 mask.npy는 이 Augmentation.py에서 생성된 것을 사용하세요.")


if __name__ == "__main__":
    main()