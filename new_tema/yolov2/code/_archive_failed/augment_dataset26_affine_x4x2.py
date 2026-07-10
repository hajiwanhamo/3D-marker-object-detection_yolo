import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
VALID_CLASSES = [0, 1, 2, 3]


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def remove_if_exists(path: Path):
    if path.exists():
        shutil.rmtree(path)


def imread_unicode(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, img):
    ensure_dir(path.parent)
    ok, buf = cv2.imencode(path.suffix, img)
    if not ok:
        raise RuntimeError(f"이미지 저장 실패: {path}")
    buf.tofile(str(path))


def collect_images(image_dir: Path):
    return sorted([
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ])


def read_yolo_seg_label(label_path: Path):
    """
    YOLO segmentation label 읽기.
    형식:
      class x1 y1 x2 y2 ... xn yn
    """
    items = []

    if not label_path.exists():
        return items

    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 7:
            continue

        try:
            cls_id = int(float(parts[0]))
            coords = list(map(float, parts[1:]))
        except Exception:
            continue

        if cls_id not in VALID_CLASSES:
            continue

        if len(coords) % 2 != 0:
            continue

        pts = np.array(coords, dtype=np.float32).reshape(-1, 2)

        # 좌표 범위 검사
        if np.any(pts < 0.0) or np.any(pts > 1.0):
            continue

        if len(pts) >= 3:
            items.append((cls_id, pts))

    return items


def labels_to_class_masks(labels, W, H):
    """
    YOLO polygon label을 class별 mask로 복원.
    """
    masks = {cid: np.zeros((H, W), dtype=np.uint8) for cid in VALID_CLASSES}

    for cls_id, pts_norm in labels:
        pts = pts_norm.copy()
        pts[:, 0] = pts[:, 0] * (W - 1)
        pts[:, 1] = pts[:, 1] * (H - 1)
        pts = np.round(pts).astype(np.int32)

        if len(pts) >= 3:
            cv2.fillPoly(masks[cls_id], [pts], 255)

    return masks


def mask_to_polygons(mask, W, H, min_area, approx_eps_ratio, max_polygon_points, max_components):
    """
    변환된 mask에서 YOLO segmentation polygon 재추출.
    """
    work = mask.copy()

    # 작은 구멍/끊김 정리. 이미지에는 적용하지 않고 label 추출에만 사용.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    work = cv2.morphologyEx(work, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(work, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polys = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or len(cnt) < 3:
            continue

        arc = cv2.arcLength(cnt, True)
        eps = max(0.5, approx_eps_ratio * arc)
        approx = cv2.approxPolyDP(cnt, eps, True)

        if len(approx) < 3:
            continue

        pts = approx.reshape(-1, 2).astype(np.float32)

        if len(pts) > max_polygon_points:
            idx = np.linspace(0, len(pts) - 1, max_polygon_points).astype(np.int32)
            pts = pts[idx]

        poly = []
        for x, y in pts:
            xn = float(np.clip(x / max(W - 1, 1), 0.0, 1.0))
            yn = float(np.clip(y / max(H - 1, 1), 0.0, 1.0))
            poly.extend([xn, yn])

        if len(poly) >= 6:
            polys.append((area, poly))

    polys = sorted(polys, key=lambda x: x[0], reverse=True)[:max_components]
    return [p for _, p in polys]


def write_yolo_seg_label(label_path: Path, class_masks, W, H, args):
    lines = []

    for cid in VALID_CLASSES:
        polys = mask_to_polygons(
            class_masks[cid],
            W,
            H,
            min_area=args.min_contour_area,
            approx_eps_ratio=args.approx_eps_ratio,
            max_polygon_points=args.max_polygon_points,
            max_components=args.max_components_per_class,
        )

        for poly in polys:
            values = [str(cid)] + [f"{v:.6f}" for v in poly]
            lines.append(" ".join(values))

    ensure_dir(label_path.parent)
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    return len(lines)


def make_affine_matrix(W, H, angle_deg, scale, tx_ratio, ty_ratio):
    """
    이미지 중심 기준 회전/스케일 + 평행이동.
    flip은 사용하지 않음.
    """
    center = (W / 2.0, H / 2.0)
    M = cv2.getRotationMatrix2D(center, angle_deg, scale)

    M[0, 2] += tx_ratio * W
    M[1, 2] += ty_ratio * H

    return M


def warp_image_and_masks(img, masks, M):
    H, W = img.shape[:2]

    # 이미지 바깥 영역은 검정색. 실해역 projection 배경과 가장 무난하게 맞음.
    img_warp = cv2.warpAffine(
        img,
        M,
        (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    mask_warp = {}
    for cid, mask in masks.items():
        m = cv2.warpAffine(
            mask,
            M,
            (W, H),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        mask_warp[cid] = m

    return img_warp, mask_warp


def draw_qc(img, masks, out_path: Path):
    overlay = img.copy()
    base = img.copy()

    colors = {
        0: (0, 255, 255),  # square
        1: (0, 180, 0),    # rect1
        2: (255, 0, 0),    # rect2
        3: (0, 0, 255),    # rect3
    }

    for cid in VALID_CLASSES:
        mask = masks[cid]
        overlay[mask > 0] = colors[cid]

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) >= 5:
                cv2.drawContours(base, [cnt], -1, colors[cid], 1)

    qc = cv2.addWeighted(base, 0.65, overlay, 0.35, 0)
    imwrite_unicode(out_path, qc)


def get_aug_params(split, aug_idx):
    """
    split별 고정 증강 파라미터.
    랜덤이 아니라 재현 가능하게 고정한다.

    train x4:
      a0 = 원본
      a1/a2/a3 = 약한 회전/이동/스케일

    val x2:
      a0 = 원본
      a1 = 아주 약한 변환
    """
    if split == "train":
        params = [
            (0.0, 1.00, 0.000, 0.000),   # 원본
            (-5.0, 0.98, -0.025, 0.018),
            (4.0, 1.03, 0.020, -0.020),
            (7.0, 0.96, 0.015, 0.025),
        ]
    else:
        params = [
            (0.0, 1.00, 0.000, 0.000),   # 원본
            (3.0, 1.02, 0.015, -0.015),
        ]

    return params[aug_idx]


def process_split(split, src_root: Path, out_root: Path, mult: int, args):
    src_img_dir = src_root / "images" / split
    src_lab_dir = src_root / "labels" / split

    out_img_dir = out_root / "images" / split
    out_lab_dir = out_root / "labels" / split
    out_qc_dir = out_root / "qc" / split

    ensure_dir(out_img_dir)
    ensure_dir(out_lab_dir)
    ensure_dir(out_qc_dir)

    images = collect_images(src_img_dir)

    stats = {
        "images": len(images),
        "written": 0,
        "missing_label": 0,
        "empty_label_after_aug": 0,
        "bad_image": 0,
    }

    for img_path in images:
        stem = img_path.stem
        label_path = src_lab_dir / f"{stem}.txt"

        img = imread_unicode(img_path)
        if img is None:
            stats["bad_image"] += 1
            continue

        H, W = img.shape[:2]

        labels = read_yolo_seg_label(label_path)
        if len(labels) == 0:
            stats["missing_label"] += 1
            continue

        masks = labels_to_class_masks(labels, W, H)

        for aug_idx in range(mult):
            angle, scale, tx, ty = get_aug_params(split, aug_idx)

            out_stem = f"{stem}_a{aug_idx}"

            if aug_idx == 0:
                img_aug = img.copy()
                masks_aug = {cid: m.copy() for cid, m in masks.items()}
            else:
                M = make_affine_matrix(W, H, angle, scale, tx, ty)
                img_aug, masks_aug = warp_image_and_masks(img, masks, M)

            out_img_path = out_img_dir / f"{out_stem}{img_path.suffix}"
            out_lab_path = out_lab_dir / f"{out_stem}.txt"

            n_lines = write_yolo_seg_label(out_lab_path, masks_aug, W, H, args)

            if n_lines == 0:
                stats["empty_label_after_aug"] += 1
                # 빈 라벨이면 학습에 부적절하므로 이미지도 저장하지 않음
                if out_lab_path.exists():
                    out_lab_path.unlink()
                continue

            imwrite_unicode(out_img_path, img_aug)

            if args.qc_all or stats["written"] < args.qc_limit:
                draw_qc(img_aug, masks_aug, out_qc_dir / f"{out_stem}_qc.png")

            stats["written"] += 1

    print(f"\n===== {split} =====")
    for k, v in stats.items():
        print(f"{k}: {v}")


def write_data_yaml(out_root: Path):
    text = f"""path: {out_root.as_posix()}
train: images/train
val: images/val

names:
  0: square
  1: rect1
  2: rect2
  3: rect3
"""
    (out_root / "data.yaml").write_text(text, encoding="utf-8")


def check_dataset(out_root: Path):
    print("\n===== dataset check =====")

    for split in ["train", "val"]:
        img_dir = out_root / "images" / split
        lab_dir = out_root / "labels" / split

        imgs = sorted([
            p for p in img_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ])
        labs = sorted(lab_dir.glob("*.txt"))

        img_stems = {p.stem for p in imgs}
        lab_stems = {p.stem for p in labs}

        class_count = {cid: 0 for cid in VALID_CLASSES}
        bad_lines = 0

        for lf in labs:
            for line in lf.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) < 7:
                    bad_lines += 1
                    continue

                try:
                    cls_id = int(float(parts[0]))
                    coords = list(map(float, parts[1:]))
                except Exception:
                    bad_lines += 1
                    continue

                if cls_id not in class_count:
                    bad_lines += 1
                    continue

                if len(coords) % 2 != 0:
                    bad_lines += 1
                    continue

                if any(v < 0.0 or v > 1.0 for v in coords):
                    bad_lines += 1
                    continue

                class_count[cls_id] += 1

        print(f"\n[{split}]")
        print(f"images: {len(imgs)}")
        print(f"labels: {len(labs)}")
        print(f"missing labels: {len(img_stems - lab_stems)}")
        print(f"missing images: {len(lab_stems - img_stems)}")
        print(f"class count: {class_count}")
        print(f"bad lines: {bad_lines}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--src", type=str, required=True, help="source dataset26 root")
    parser.add_argument("--out", type=str, required=True, help="augmented output dataset root")

    parser.add_argument("--train_mult", type=int, default=4)
    parser.add_argument("--val_mult", type=int, default=2)

    parser.add_argument("--min_contour_area", type=float, default=8.0)
    parser.add_argument("--approx_eps_ratio", type=float, default=0.006)
    parser.add_argument("--max_polygon_points", type=int, default=80)
    parser.add_argument("--max_components_per_class", type=int, default=3)

    parser.add_argument("--qc_limit", type=int, default=100)
    parser.add_argument("--qc_all", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    src_root = Path(args.src)
    out_root = Path(args.out)

    if not src_root.exists():
        raise FileNotFoundError(f"src 없음: {src_root}")

    if out_root.exists():
        if args.overwrite:
            remove_if_exists(out_root)
        else:
            raise FileExistsError(f"이미 존재함: {out_root}  --overwrite 사용 필요")

    ensure_dir(out_root)

    print("===== augment dataset26 affine x4/x2 =====")
    print(f"src: {src_root}")
    print(f"out: {out_root}")
    print(f"train_mult: {args.train_mult}")
    print(f"val_mult: {args.val_mult}")
    print("flip: NONE")
    print("noise/occlusion/blur/color jitter: NONE")
    print("label method: mask warp + polygon re-extraction")

    process_split("train", src_root, out_root, args.train_mult, args)
    process_split("val", src_root, out_root, args.val_mult, args)

    write_data_yaml(out_root)
    check_dataset(out_root)

    print("\n[OK] augmented dataset created")
    print(f"data.yaml: {out_root / 'data.yaml'}")
    print(f"qc: {out_root / 'qc'}")


if __name__ == "__main__":
    main()
