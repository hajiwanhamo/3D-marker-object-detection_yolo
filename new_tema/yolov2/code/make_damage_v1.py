#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_damage_v1.py

목적:
- standard 데이터셋 기반으로
  "마커 손상 정도가 심한" YOLO segmentation 학습 데이터 생성
- 모든 클래스(class0~3)에 동일 원칙 적용
- 손상된 visible shape에 맞춰 segmentation label도 같이 갱신
- val은 원본 유지, train만 원본+손상 증강 생성

입력 구조:
src-root/
  images/train
  images/val
  labels/train
  labels/val
  data.yaml

출력 구조:
out-root/
  images/train
  images/val
  labels/train
  labels/val
  data.yaml
"""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def list_images(img_dir: Path):
    """이미지 파일 목록 반환"""
    if not img_dir.exists():
        return []
    return sorted([p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS])


def read_image(path: Path):
    """이미지 읽기"""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"[ERROR] 이미지 읽기 실패: {path}")
    return img


def polygon_area(pts: np.ndarray):
    """polygon 면적 계산"""
    if pts is None or len(pts) < 3:
        return 0.0
    x = pts[:, 0]
    y = pts[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)


def load_yolo_segments(label_path: Path, w: int, h: int):
    """YOLO segmentation txt -> pixel polygon"""
    objects = []

    if not label_path.exists():
        return objects

    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        try:
            vals = [float(x) for x in parts]
        except ValueError:
            continue

        if len(vals) < 7:
            continue

        cls_id = int(vals[0])
        coords = vals[1:]
        if len(coords) % 2 != 0:
            continue

        pts = np.array(coords, dtype=np.float32).reshape(-1, 2)
        pts[:, 0] *= w
        pts[:, 1] *= h

        if polygon_area(pts) <= 1.0:
            continue

        objects.append({"cls": cls_id, "pts": pts})

    return objects


def pts_to_yolo_line(cls_id: int, pts: np.ndarray, w: int, h: int):
    """pixel polygon -> YOLO segmentation line"""
    if pts is None or len(pts) < 3:
        return None

    pts = pts.astype(np.float32).copy()
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)

    if polygon_area(pts) <= 1.0:
        return None

    pts[:, 0] = np.clip(pts[:, 0] / max(w, 1), 0.0, 1.0)
    pts[:, 1] = np.clip(pts[:, 1] / max(h, 1), 0.0, 1.0)

    vals = [str(int(cls_id))]
    for x, y in pts:
        vals.append(f"{float(x):.6f}")
        vals.append(f"{float(y):.6f}")
    return " ".join(vals)


def polygon_to_mask(pts: np.ndarray, w: int, h: int):
    """polygon -> binary mask"""
    mask = np.zeros((h, w), dtype=np.uint8)
    arr = np.round(pts).astype(np.int32)
    cv2.fillPoly(mask, [arr], 255)
    return mask


def mask_to_polygon(mask: np.ndarray, min_area: float):
    """
    binary mask -> single polygon
    가장 큰 connected component만 사용
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)

    if num_labels <= 1:
        return None

    # background 제외 largest component 선택
    areas = stats[1:, cv2.CC_STAT_AREA]
    best_idx = 1 + int(np.argmax(areas))
    comp = (labels == best_idx).astype(np.uint8) * 255

    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < min_area:
        return None

    # polygon 단순화
    eps = 0.003 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, eps, True)

    pts = approx.reshape(-1, 2).astype(np.float32)
    if len(pts) < 3:
        return None
    if polygon_area(pts) < min_area:
        return None

    return pts


def boundary_points(mask: np.ndarray):
    """mask 경계점 추출"""
    cnts, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    pts = cnt.reshape(-1, 2)
    if len(pts) == 0:
        return None
    return pts


def keep_largest_component(mask: np.ndarray):
    """largest component만 유지"""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(mask)
    areas = stats[1:, cv2.CC_STAT_AREA]
    best_idx = 1 + int(np.argmax(areas))
    return ((labels == best_idx).astype(np.uint8) * 255)


def damage_single_mask(mask: np.ndarray, rng: np.random.Generator, args):
    """
    객체 mask를 심하게 손상시킨다.
    - boundary bite
    - erosion
    - 연결성 유지
    - 최소/최대 area ratio 만족하는 결과만 채택
    """
    orig = (mask > 0).astype(np.uint8) * 255
    orig_area = int((orig > 0).sum())
    if orig_area < args.min_mask_pixels:
        return None

    for _ in range(args.max_trials):
        work = orig.copy()

        # 1) 약한 erosion
        erode_iter = int(rng.integers(args.erode_iter_min, args.erode_iter_max + 1))
        if erode_iter > 0:
            k = int(rng.integers(args.erode_kernel_min, args.erode_kernel_max + 1))
            k = max(3, k if k % 2 == 1 else k + 1)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            work = cv2.erode(work, kernel, iterations=erode_iter)

        # 2) 경계 일부를 크게 갉아먹는 bite
        bpts = boundary_points(work)
        if bpts is None or len(bpts) == 0:
            continue

        bite_count = int(rng.integers(args.bite_count_min, args.bite_count_max + 1))

        h, w = work.shape[:2]
        for _b in range(bite_count):
            p = bpts[int(rng.integers(0, len(bpts)))]
            px, py = int(p[0]), int(p[1])

            radius = int(rng.integers(args.bite_radius_min, args.bite_radius_max + 1))

            # 경계 바깥쪽/안쪽 모두 조금 포함되도록 offset
            ox = int(rng.integers(-args.bite_offset, args.bite_offset + 1))
            oy = int(rng.integers(-args.bite_offset, args.bite_offset + 1))

            cx = int(np.clip(px + ox, 0, w - 1))
            cy = int(np.clip(py + oy, 0, h - 1))

            cv2.circle(work, (cx, cy), radius, 0, thickness=-1)

        # 3) largest component 유지
        work = keep_largest_component(work)

        area = int((work > 0).sum())
        if area <= 0:
            continue

        ratio = area / max(orig_area, 1)

        # 손상 정도 조건
        if ratio < args.keep_ratio_min or ratio > args.keep_ratio_max:
            continue

        # 너무 가는 조각 방지용 closing
        k2 = int(rng.integers(args.close_kernel_min, args.close_kernel_max + 1))
        k2 = max(3, k2 if k2 % 2 == 1 else k2 + 1)
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k2, k2))
        work2 = cv2.morphologyEx(work, cv2.MORPH_CLOSE, kernel2, iterations=1)
        work2 = keep_largest_component(work2)

        area2 = int((work2 > 0).sum())
        ratio2 = area2 / max(orig_area, 1)
        if area2 <= 0:
            continue
        if ratio2 < args.keep_ratio_min or ratio2 > args.keep_ratio_max:
            continue

        return work2

    return None


def write_label(path: Path, objects, w: int, h: int):
    """YOLO seg label 저장"""
    lines = []
    for obj in objects:
        line = pts_to_yolo_line(obj["cls"], obj["pts"], w, h)
        if line is not None:
            lines.append(line)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def copy_original(img_path: Path, lbl_path: Path, out_img: Path, out_lbl: Path):
    """원본 복사"""
    shutil.copy2(img_path, out_img)
    shutil.copy2(lbl_path, out_lbl)
    return True


def make_damaged_sample(img_path: Path, lbl_path: Path, out_img: Path, out_lbl: Path, rng: np.random.Generator, args):
    """
    손상 이미지 + 손상 라벨 생성
    """
    img = read_image(img_path)
    h, w = img.shape[:2]
    objects = load_yolo_segments(lbl_path, w, h)

    if len(objects) == 0:
        return False, "no_objects"

    aug_img = img.copy()
    new_objects = []

    for obj in objects:
        cls_id = int(obj["cls"])
        pts = obj["pts"]

        orig_mask = polygon_to_mask(pts, w, h)
        damaged_mask = damage_single_mask(orig_mask, rng, args)

        if damaged_mask is None:
            return False, "damage_fail"

        # 이미지에서 제거된 부분은 검게 만든다
        removed = ((orig_mask > 0) & (damaged_mask == 0))
        aug_img[removed] = 0

        # 약간의 랜덤 intensity 스케일
        remain = (damaged_mask > 0)
        if remain.any():
            alpha = float(rng.uniform(args.intensity_min, args.intensity_max))
            region = aug_img[remain].astype(np.float32) * alpha
            aug_img[remain] = np.clip(region, 0, 255).astype(np.uint8)

        new_pts = mask_to_polygon(damaged_mask, args.min_polygon_area)
        if new_pts is None:
            return False, "bad_polygon"

        new_objects.append({"cls": cls_id, "pts": new_pts})

    # 전체 이미지에 약한 사진 노이즈
    if args.global_noise_std > 0:
        noise = rng.normal(0.0, args.global_noise_std, size=aug_img.shape).astype(np.float32)
        aug_img = np.clip(aug_img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    cv2.imwrite(str(out_img), aug_img)
    n = write_label(out_lbl, new_objects, w, h)

    if n == 0:
        return False, "empty_label"

    return True, "ok"


def write_data_yaml(src_root: Path, out_root: Path):
    """data.yaml 생성"""
    src_yaml = src_root / "data.yaml"
    names_block = None

    if src_yaml.exists():
        lines = src_yaml.read_text(encoding="utf-8", errors="ignore").splitlines()
        for i, line in enumerate(lines):
            if line.strip().startswith("names:"):
                names_block = "\n".join(lines[i:])
                break

    if names_block is None:
        names_block = """names:
  0: class0
  1: class1
  2: class2
  3: class3"""

    text = f"""path: {out_root}
train: images/train
val: images/val

{names_block}
"""
    (out_root / "data.yaml").write_text(text, encoding="utf-8")


def process_split(split: str, src_root: Path, out_root: Path, rng: np.random.Generator, args):
    """train/val 처리"""
    src_img_dir = src_root / "images" / split
    src_lbl_dir = src_root / "labels" / split

    out_img_dir = out_root / "images" / split
    out_lbl_dir = out_root / "labels" / split

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(src_img_dir)

    made = 0
    skipped_missing_label = 0
    failed_damage = 0

    for img_path in images:
        lbl_path = src_lbl_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            skipped_missing_label += 1
            continue

        if split == "val":
            out_img = out_img_dir / img_path.name
            out_lbl = out_lbl_dir / f"{img_path.stem}.txt"
            copy_original(img_path, lbl_path, out_img, out_lbl)
            made += 1
            continue

        # train: 원본 1개 유지
        out_img = out_img_dir / f"{img_path.stem}_orig{img_path.suffix}"
        out_lbl = out_lbl_dir / f"{img_path.stem}_orig.txt"
        copy_original(img_path, lbl_path, out_img, out_lbl)
        made += 1

        # train: 손상 증강 N개 생성
        for k in range(args.aug_per_image):
            out_img = out_img_dir / f"{img_path.stem}_damage{k:02d}{img_path.suffix}"
            out_lbl = out_lbl_dir / f"{img_path.stem}_damage{k:02d}.txt"

            ok, reason = make_damaged_sample(img_path, lbl_path, out_img, out_lbl, rng, args)
            if ok:
                made += 1
            else:
                failed_damage += 1

    return {
        "split": split,
        "src_images": len(images),
        "made": made,
        "skipped_missing_label": skipped_missing_label,
        "failed_damage": failed_damage,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--src-root", type=str, required=True)
    parser.add_argument("--out-root", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)

    # train 증강 배수
    parser.add_argument("--aug-per-image", type=int, default=2)

    # 손상 강도 관련
    parser.add_argument("--keep-ratio-min", type=float, default=0.35)
    parser.add_argument("--keep-ratio-max", type=float, default=0.70)

    parser.add_argument("--bite-count-min", type=int, default=3)
    parser.add_argument("--bite-count-max", type=int, default=7)

    parser.add_argument("--bite-radius-min", type=int, default=10)
    parser.add_argument("--bite-radius-max", type=int, default=28)
    parser.add_argument("--bite-offset", type=int, default=8)

    parser.add_argument("--erode-iter-min", type=int, default=0)
    parser.add_argument("--erode-iter-max", type=int, default=2)
    parser.add_argument("--erode-kernel-min", type=int, default=3)
    parser.add_argument("--erode-kernel-max", type=int, default=7)

    parser.add_argument("--close-kernel-min", type=int, default=3)
    parser.add_argument("--close-kernel-max", type=int, default=7)

    # 밝기/노이즈
    parser.add_argument("--intensity-min", type=float, default=0.75)
    parser.add_argument("--intensity-max", type=float, default=1.00)
    parser.add_argument("--global-noise-std", type=float, default=2.0)

    # 안전장치
    parser.add_argument("--min-mask-pixels", type=int, default=60)
    parser.add_argument("--min-polygon-area", type=float, default=12.0)
    parser.add_argument("--max-trials", type=int, default=20)

    args = parser.parse_args()

    src_root = Path(args.src_root)
    out_root = Path(args.out_root)

    if not src_root.exists():
        raise FileNotFoundError(f"[ERROR] src-root 없음: {src_root}")

    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    summaries = []
    for split in ["train", "val"]:
        summaries.append(process_split(split, src_root, out_root, rng, args))

    write_data_yaml(src_root, out_root)

    print("============================================")
    print("[DONE] damage_v1 dataset 생성 완료")
    print("SRC :", src_root)
    print("OUT :", out_root)
    print("YAML:", out_root / "data.yaml")
    print("============================================")
    for s in summaries:
        print(
            f"{s['split']}: "
            f"src_images={s['src_images']}, "
            f"made={s['made']}, "
            f"skipped_missing_label={s['skipped_missing_label']}, "
            f"failed_damage={s['failed_damage']}"
        )


if __name__ == "__main__":
    main()
