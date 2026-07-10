import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


# ============================================================
# 기본 파일/라벨 유틸
# ============================================================

def list_images(img_dir: Path):
    """이미지 폴더에서 이미지 파일 목록을 정렬해서 반환한다."""
    if not img_dir.exists():
        return []
    return sorted([p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS])


def read_image(path: Path):
    """OpenCV로 이미지를 BGR 이미지로 읽는다."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"[ERROR] 이미지 읽기 실패: {path}")
    return img


def load_yolo_segments(label_path: Path, w: int, h: int):
    """
    YOLO segmentation label을 읽어 pixel polygon으로 변환한다.

    반환:
        [(class_id, pts_abs), ...]
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

        objects.append((cls_id, pts))

    return objects


def polygon_to_mask(pts: np.ndarray, h: int, w: int):
    """polygon을 binary mask로 변환한다."""
    mask = np.zeros((h, w), dtype=np.uint8)

    if pts is None or len(pts) < 3:
        return mask.astype(bool)

    poly = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [poly], 255)

    return mask > 0


def pts_to_yolo_line(cls_id: int, pts: np.ndarray, w: int, h: int):
    """
    pixel polygon을 YOLO normalized segmentation line으로 변환한다.
    """
    if pts is None or len(pts) < 3:
        return None

    pts = pts.astype(np.float32).copy()
    pts[:, 0] = np.clip(pts[:, 0] / max(w, 1), 0.0, 1.0)
    pts[:, 1] = np.clip(pts[:, 1] / max(h, 1), 0.0, 1.0)

    vals = [str(int(cls_id))]

    for x, y in pts:
        vals.append(f"{float(x):.6f}")
        vals.append(f"{float(y):.6f}")

    return " ".join(vals)


# ============================================================
# 노이즈 생성 유틸
# ============================================================

def lowfreq_field(h: int, w: int, rng: np.random.Generator, blur: int):
    """
    부드러운 위치별 밀도 변화를 만들기 위한 low-frequency field.
    """
    if blur % 2 == 0:
        blur += 1

    field = rng.random((h, w)).astype(np.float32)
    field = cv2.GaussianBlur(field, (blur, blur), 0)

    field -= field.min()
    denom = field.max() - field.min()

    if denom > 1e-6:
        field /= denom
    else:
        field[:] = 1.0

    return field


def make_square_core_polygon(
    pts: np.ndarray,
    rng: np.random.Generator,
    core_min: float,
    core_max: float,
    offset_ratio: float,
    jitter_ratio: float,
):
    """
    square 중심부 polygon을 생성한다.

    목적:
    - square는 실해역에서 전체 사각형보다 중심부만 남는 경우가 많음.
    - 따라서 이미지와 라벨 모두 중심부 기준으로 맞춘다.
    """
    center = pts.mean(axis=0, keepdims=True)

    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)

    bw = max(1.0, x_max - x_min)
    bh = max(1.0, y_max - y_min)

    offset = np.array(
        [
            rng.uniform(-offset_ratio, offset_ratio) * bw,
            rng.uniform(-offset_ratio, offset_ratio) * bh,
        ],
        dtype=np.float32,
    ).reshape(1, 2)

    scale = rng.uniform(core_min, core_max)
    core = center + offset + (pts - center) * scale

    # 항상 같은 정사각형 중심부가 남지 않도록 작은 흔들림 추가
    jitter = rng.normal(0.0, jitter_ratio, size=core.shape).astype(np.float32)
    jitter[:, 0] *= bw
    jitter[:, 1] *= bh

    core = core + jitter

    return core.astype(np.float32)


def apply_square_core(
    out: np.ndarray,
    obj_mask: np.ndarray,
    core_mask: np.ndarray,
    rng: np.random.Generator,
    variant: str,
):
    """
    square 영역 처리.

    - core 외부는 제거
    - core 내부는 대부분 유지
    - 약한 밀도/밝기 변화만 적용
    """
    out[obj_mask & (~core_mask)] = 0

    ys, xs = np.where(core_mask)
    if len(xs) == 0:
        return out

    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()

    crop = out[y1:y2 + 1, x1:x2 + 1].copy()
    cmask = core_mask[y1:y2 + 1, x1:x2 + 1]

    ch, cw = cmask.shape

    if variant == "base":
        keep_base = rng.uniform(0.92, 0.99)
        intensity = rng.uniform(0.85, 1.00)
    elif variant == "mild":
        keep_base = rng.uniform(0.84, 0.96)
        intensity = rng.uniform(0.76, 1.00)
    else:
        keep_base = rng.uniform(0.76, 0.92)
        intensity = rng.uniform(0.68, 0.96)

    field = lowfreq_field(ch, cw, rng, blur=11)
    keep_prob = np.clip(keep_base * (0.90 + 0.15 * field), 0.65, 0.99)

    keep = cmask & (rng.random((ch, cw)) < keep_prob)
    drop = cmask & (~keep)

    crop[drop] = 0

    crop_f = crop.astype(np.float32)
    crop_f[keep] *= intensity

    noise = rng.normal(0.0, 2.0, size=crop.shape).astype(np.float32)
    crop_f[keep] += noise[keep]

    crop = np.clip(crop_f, 0, 255).astype(np.uint8)
    out[y1:y2 + 1, x1:x2 + 1] = crop

    return out


def apply_rect_mild_density(
    out: np.ndarray,
    obj_mask: np.ndarray,
    rng: np.random.Generator,
    variant: str,
):
    """
    rect1/2/3 영역 처리.

    핵심 원칙:
    - rect는 조각내지 않는다.
    - 끝부분 절단 없음.
    - 중간 끊김 없음.
    - decoy 없음.
    - 라벨은 원래 polygon 유지.
    - 이미지에는 약한 밀도 저하와 약한 내부 구멍만 적용.
    """
    ys, xs = np.where(obj_mask)
    if len(xs) == 0:
        return out

    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()

    crop = out[y1:y2 + 1, x1:x2 + 1].copy()
    cmask = obj_mask[y1:y2 + 1, x1:x2 + 1]

    ch, cw = cmask.shape

    if variant == "base":
        # base는 rect를 거의 그대로 둔다.
        keep_base = rng.uniform(0.96, 1.00)
        intensity = rng.uniform(0.90, 1.05)
        hole_count = 0
    elif variant == "mild":
        # mild는 약한 점밀도 변화만 준다.
        keep_base = rng.uniform(0.86, 0.97)
        intensity = rng.uniform(0.78, 1.00)
        hole_count = int(rng.integers(0, 2))
    else:
        # mild_sparse도 rect 구조는 유지하되 조금 더 sparse하게 만든다.
        keep_base = rng.uniform(0.76, 0.92)
        intensity = rng.uniform(0.70, 0.96)
        hole_count = int(rng.integers(0, 3))

    field = lowfreq_field(ch, cw, rng, blur=21)

    # 너무 잘게 찢기지 않도록 keep 확률을 높게 유지한다.
    keep_prob = np.clip(keep_base * (0.88 + 0.18 * field), 0.68, 1.00)
    keep = cmask & (rng.random((ch, cw)) < keep_prob)

    # 내부에 작은 hole만 만든다. 끝부분 절단/중간 절단은 절대 하지 않는다.
    cys, cxs = np.where(cmask)
    if len(cxs) > 0 and hole_count > 0:
        min_side = max(1, min(ch, cw))

        for _ in range(hole_count):
            idx = int(rng.integers(0, len(cxs)))
            cx = int(cxs[idx])
            cy = int(cys[idx])

            rx = max(1, int(min_side * rng.uniform(0.025, 0.055)))
            ry = max(1, int(min_side * rng.uniform(0.025, 0.055)))

            hole = np.zeros((ch, cw), dtype=np.uint8)
            cv2.ellipse(
                hole,
                center=(cx, cy),
                axes=(rx, ry),
                angle=float(rng.uniform(0.0, 180.0)),
                startAngle=0,
                endAngle=360,
                color=255,
                thickness=-1,
            )

            keep[(hole > 0) & cmask] = False

    # 과도한 삭제 방지: rect가 최소 70% 이상은 남게 한다.
    min_ratio = 0.90 if variant == "base" else (0.78 if variant == "mild" else 0.68)

    if keep.sum() < cmask.sum() * min_ratio:
        restore_prob = min(1.0, min_ratio)
        restore = cmask & (rng.random((ch, cw)) < restore_prob)
        keep |= restore

    drop = cmask & (~keep)
    crop[drop] = 0

    crop_f = crop.astype(np.float32)
    crop_f[keep] *= intensity

    noise = rng.normal(0.0, 2.0, size=crop.shape).astype(np.float32)
    crop_f[keep] += noise[keep]

    crop = np.clip(crop_f, 0, 255).astype(np.uint8)
    out[y1:y2 + 1, x1:x2 + 1] = crop

    return out


def process_variant(img: np.ndarray, objects, rng: np.random.Generator, variant: str, args):
    """
    base / mild / mild_sparse 중 하나의 이미지를 생성한다.

    라벨 규칙:
    - square: 중심부 polygon으로 새 라벨 생성
    - rect1/2/3: 원래 polygon 라벨 유지
    """
    h, w = img.shape[:2]
    out = img.copy()
    new_lines = []

    # 전체 이미지 밝기 변동은 약하게만 적용한다.
    if variant == "base":
        global_scale = rng.uniform(0.96, 1.04)
    elif variant == "mild":
        global_scale = rng.uniform(0.88, 1.04)
    else:
        global_scale = rng.uniform(0.80, 1.00)

    out = np.clip(out.astype(np.float32) * global_scale, 0, 255).astype(np.uint8)

    for cls_id, pts in objects:
        obj_mask = polygon_to_mask(pts, h, w)

        if obj_mask.sum() < args.min_label_area:
            continue

        if cls_id == args.square_class:
            if variant == "base":
                core_pts = make_square_core_polygon(
                    pts=pts,
                    rng=rng,
                    core_min=0.48,
                    core_max=0.68,
                    offset_ratio=0.04,
                    jitter_ratio=0.018,
                )
            elif variant == "mild":
                core_pts = make_square_core_polygon(
                    pts=pts,
                    rng=rng,
                    core_min=0.40,
                    core_max=0.62,
                    offset_ratio=0.05,
                    jitter_ratio=0.022,
                )
            else:
                core_pts = make_square_core_polygon(
                    pts=pts,
                    rng=rng,
                    core_min=0.34,
                    core_max=0.56,
                    offset_ratio=0.06,
                    jitter_ratio=0.026,
                )

            core_mask = polygon_to_mask(core_pts, h, w) & obj_mask

            out = apply_square_core(
                out=out,
                obj_mask=obj_mask,
                core_mask=core_mask,
                rng=rng,
                variant=variant,
            )

            # square는 이미지에서 실제 남긴 중심부 기준으로 라벨 생성
            line = pts_to_yolo_line(cls_id, core_pts, w, h)

        else:
            out = apply_rect_mild_density(
                out=out,
                obj_mask=obj_mask,
                rng=rng,
                variant=variant,
            )

            # rect는 조각난 visible contour를 쓰지 않고 원래 polygon 유지
            line = pts_to_yolo_line(cls_id, pts, w, h)

        if line is not None:
            new_lines.append(line)

    return out, new_lines


def write_data_yaml(src_root: Path, out_root: Path):
    """원본 data.yaml의 names block을 유지하고 path만 새 out_root로 바꾼다."""
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
    """train 또는 val split을 생성한다."""
    src_img_dir = src_root / "images" / split
    src_lbl_dir = src_root / "labels" / split

    out_img_dir = out_root / "images" / split
    out_lbl_dir = out_root / "labels" / split

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(src_img_dir)

    variants = ["base", "mild", "mild_sparse"]

    made = 0
    missing_labels = 0
    empty_labels = 0

    for img_path in images:
        img = read_image(img_path)
        h, w = img.shape[:2]

        label_path = src_lbl_dir / f"{img_path.stem}.txt"
        objects = load_yolo_segments(label_path, w, h)

        if not label_path.exists():
            missing_labels += 1

        for variant in variants:
            dst_img = out_img_dir / f"{img_path.stem}_{variant}{img_path.suffix}"
            dst_lbl = out_lbl_dir / f"{img_path.stem}_{variant}.txt"

            out_img, out_lines = process_variant(
                img=img,
                objects=objects,
                rng=rng,
                variant=variant,
                args=args,
            )

            cv2.imwrite(str(dst_img), out_img)

            if out_lines:
                dst_lbl.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
            else:
                empty_labels += 1
                dst_lbl.write_text("", encoding="utf-8")

            made += 1

    return {
        "split": split,
        "src_images": len(images),
        "made_images": made,
        "missing_labels": missing_labels,
        "empty_labels": empty_labels,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--src-root", type=str, required=True)
    parser.add_argument("--out-root", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--square-class", type=int, default=0)
    parser.add_argument("--min-label-area", type=float, default=8.0)

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
    print("[DONE] conservative noise_v2 생성 완료")
    print("SRC :", src_root)
    print("OUT :", out_root)
    print("YAML:", out_root / "data.yaml")
    print("============================================")

    for s in summaries:
        print(
            f"{s['split']}: "
            f"src_images={s['src_images']}, "
            f"made_images={s['made_images']}, "
            f"missing_labels={s['missing_labels']}, "
            f"empty_labels={s['empty_labels']}"
        )


if __name__ == "__main__":
    main()
