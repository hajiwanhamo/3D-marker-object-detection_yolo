#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_class_identity_context_dataset.py

목적:
- YOLO 4클래스 class confusion 완화용 학습 데이터 생성
- 후처리 없음
- class별 개별 변형 없음
- square/rect1/rect2/rect3의 상대 배치를 유지
- train에는 전체 이미지 단위 동일 변형만 적용
- val은 원본을 그대로 복사

입력:
src-root/
  images/train
  images/val
  labels/train
  labels/val
  data.yaml

출력:
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
    """이미지 목록을 정렬해서 반환한다."""
    if not img_dir.exists():
        return []
    return sorted([p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS])


def read_image(path: Path):
    """OpenCV로 이미지를 읽는다."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"[ERROR] 이미지 읽기 실패: {path}")
    return img


def polygon_area(pts: np.ndarray):
    """polygon 면적 계산."""
    if pts is None or len(pts) < 3:
        return 0.0
    x = pts[:, 0]
    y = pts[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)


def load_yolo_segments(label_path: Path, w: int, h: int):
    """
    YOLO segmentation label을 pixel polygon으로 읽는다.

    반환:
    [
      {"cls": int, "pts": np.ndarray(N,2)}
    ]
    """
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
    """pixel polygon을 YOLO normalized segmentation line으로 변환한다."""
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


def has_all_classes(objects, num_classes: int):
    """이미지 안에 class0~num_classes-1이 모두 존재하는지 확인한다."""
    found = {int(o["cls"]) for o in objects}
    return all(c in found for c in range(num_classes))


def make_affine_matrix(w: int, h: int, rng: np.random.Generator, args):
    """
    이미지 전체에 동일하게 적용할 affine matrix 생성.
    class별 개별 변형은 하지 않는다.
    """
    cx = w * 0.5
    cy = h * 0.5

    angle = float(rng.uniform(-args.max_degrees, args.max_degrees))
    scale = float(rng.uniform(1.0 - args.max_scale, 1.0 + args.max_scale))

    tx = float(rng.uniform(-args.max_translate, args.max_translate) * w)
    ty = float(rng.uniform(-args.max_translate, args.max_translate) * h)

    M = cv2.getRotationMatrix2D((cx, cy), angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty

    return M


def transform_points(pts: np.ndarray, M: np.ndarray):
    """affine matrix를 polygon 좌표에 적용한다."""
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    homo = np.concatenate([pts.astype(np.float32), ones], axis=1)
    out = homo @ M.T
    return out.astype(np.float32)


def apply_global_photo(img: np.ndarray, rng: np.random.Generator, args):
    """
    전체 이미지에 동일한 밝기/대비 변화 적용.
    class별 밝기 변형은 하지 않는다.
    """
    alpha = float(rng.uniform(1.0 - args.max_contrast, 1.0 + args.max_contrast))
    beta = float(rng.uniform(-args.max_brightness, args.max_brightness))

    out = img.astype(np.float32) * alpha + beta

    if args.noise_std > 0:
        noise = rng.normal(0.0, args.noise_std, size=out.shape).astype(np.float32)
        out += noise

    out = np.clip(out, 0, 255).astype(np.uint8)

    return out


def write_label(path: Path, objects, w: int, h: int):
    """YOLO segmentation txt 저장."""
    lines = []

    for obj in objects:
        line = pts_to_yolo_line(obj["cls"], obj["pts"], w, h)
        if line is not None:
            lines.append(line)

    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def copy_original(img_path: Path, label_path: Path, out_img: Path, out_lbl: Path, num_classes: int, require_all4: bool):
    """원본 이미지와 원본 라벨을 그대로 복사한다."""
    img = read_image(img_path)
    h, w = img.shape[:2]
    objects = load_yolo_segments(label_path, w, h)

    if require_all4 and not has_all_classes(objects, num_classes):
        return False, "skip_not_all4"

    shutil.copy2(img_path, out_img)
    shutil.copy2(label_path, out_lbl)
    return True, "ok"


def make_augmented(img_path: Path, label_path: Path, out_img: Path, out_lbl: Path, rng: np.random.Generator, args):
    """
    전체 이미지 단위 동일 변형으로 증강 이미지 생성.
    label polygon에도 동일한 affine transform 적용.
    """
    img = read_image(img_path)
    h, w = img.shape[:2]

    objects = load_yolo_segments(label_path, w, h)

    if args.require_all4 and not has_all_classes(objects, args.num_classes):
        return False, "skip_not_all4"

    M = make_affine_matrix(w, h, rng, args)

    aug_img = cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    aug_img = apply_global_photo(aug_img, rng, args)

    aug_objects = []
    for obj in objects:
        new_pts = transform_points(obj["pts"], M)
        new_pts[:, 0] = np.clip(new_pts[:, 0], 0, w - 1)
        new_pts[:, 1] = np.clip(new_pts[:, 1], 0, h - 1)

        if polygon_area(new_pts) <= args.min_area:
            continue

        aug_objects.append({"cls": obj["cls"], "pts": new_pts})

    if args.require_all4 and not has_all_classes(aug_objects, args.num_classes):
        return False, "skip_aug_lost_class"

    cv2.imwrite(str(out_img), aug_img)
    n_lines = write_label(out_lbl, aug_objects, w, h)

    if n_lines == 0:
        return False, "empty_label"

    return True, "ok"


def write_data_yaml(src_root: Path, out_root: Path):
    """원본 data.yaml의 names block은 유지하고 path만 out_root로 바꾼다."""
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
  0: square
  1: rect1
  2: rect2
  3: rect3"""

    text = f"""path: {out_root}
train: images/train
val: images/val

{names_block}
"""
    (out_root / "data.yaml").write_text(text, encoding="utf-8")


def process_split(split: str, src_root: Path, out_root: Path, rng: np.random.Generator, args):
    """train/val 처리."""
    src_img_dir = src_root / "images" / split
    src_lbl_dir = src_root / "labels" / split

    out_img_dir = out_root / "images" / split
    out_lbl_dir = out_root / "labels" / split

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(src_img_dir)

    made = 0
    skipped_missing_label = 0
    skipped_not_all4 = 0
    skipped_aug_lost_class = 0
    empty_label = 0

    for img_path in images:
        label_path = src_lbl_dir / f"{img_path.stem}.txt"

        if not label_path.exists():
            skipped_missing_label += 1
            continue

        # val은 원본 그대로만 복사한다.
        if split == "val":
            out_img = out_img_dir / img_path.name
            out_lbl = out_lbl_dir / f"{img_path.stem}.txt"

            ok, reason = copy_original(
                img_path=img_path,
                label_path=label_path,
                out_img=out_img,
                out_lbl=out_lbl,
                num_classes=args.num_classes,
                require_all4=args.require_all4,
            )

            if ok:
                made += 1
            elif reason == "skip_not_all4":
                skipped_not_all4 += 1
            continue

        # train은 원본 + 전체 이미지 단위 동일 변형
        out_img = out_img_dir / f"{img_path.stem}_orig{img_path.suffix}"
        out_lbl = out_lbl_dir / f"{img_path.stem}_orig.txt"

        ok, reason = copy_original(
            img_path=img_path,
            label_path=label_path,
            out_img=out_img,
            out_lbl=out_lbl,
            num_classes=args.num_classes,
            require_all4=args.require_all4,
        )

        if ok:
            made += 1
        elif reason == "skip_not_all4":
            skipped_not_all4 += 1
            continue

        for k in range(args.train_aug):
            out_img = out_img_dir / f"{img_path.stem}_ctx{k:02d}{img_path.suffix}"
            out_lbl = out_lbl_dir / f"{img_path.stem}_ctx{k:02d}.txt"

            ok, reason = make_augmented(
                img_path=img_path,
                label_path=label_path,
                out_img=out_img,
                out_lbl=out_lbl,
                rng=rng,
                args=args,
            )

            if ok:
                made += 1
            elif reason == "skip_not_all4":
                skipped_not_all4 += 1
            elif reason == "skip_aug_lost_class":
                skipped_aug_lost_class += 1
            elif reason == "empty_label":
                empty_label += 1

    return {
        "split": split,
        "src_images": len(images),
        "made": made,
        "skipped_missing_label": skipped_missing_label,
        "skipped_not_all4": skipped_not_all4,
        "skipped_aug_lost_class": skipped_aug_lost_class,
        "empty_label": empty_label,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--src-root", type=str, required=True)
    parser.add_argument("--out-root", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--require-all4", action="store_true")

    parser.add_argument("--train-aug", type=int, default=2)
    parser.add_argument("--max-degrees", type=float, default=8.0)
    parser.add_argument("--max-scale", type=float, default=0.08)
    parser.add_argument("--max-translate", type=float, default=0.03)

    parser.add_argument("--max-contrast", type=float, default=0.12)
    parser.add_argument("--max-brightness", type=float, default=12.0)
    parser.add_argument("--noise-std", type=float, default=1.5)

    parser.add_argument("--min-area", type=float, default=8.0)

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
    print("[DONE] class identity context dataset 생성 완료")
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
            f"skipped_not_all4={s['skipped_not_all4']}, "
            f"skipped_aug_lost_class={s['skipped_aug_lost_class']}, "
            f"empty_label={s['empty_label']}"
        )


if __name__ == "__main__":
    main()
