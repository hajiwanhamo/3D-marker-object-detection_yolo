import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np


# ------------------------------------------------------------
# 이미지 확장자 목록
# ------------------------------------------------------------
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def read_image(path: Path):
    """이미지를 BGR 형식으로 읽는다."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"이미지를 읽을 수 없습니다: {path}")
    return img


def add_pixel_noise(
    img: np.ndarray,
    rng: np.random.Generator,
    gauss_std_max: float = 8.0,
    speckle_prob: float = 0.0003,
    blur_prob: float = 0.25,
    brightness_range: float = 0.12,
):
    """
    geometry를 바꾸지 않는 noise만 추가한다.
    따라서 YOLO segmentation label 좌표는 그대로 사용할 수 있다.

    적용 내용:
    1. 밝기/대비 변화
    2. 약한 Gaussian noise
    3. 배경 speckle point
    4. 약한 blur
    """
    out = img.astype(np.float32)

    # --------------------------------------------------------
    # 1. 밝기/대비 변화
    # --------------------------------------------------------
    alpha = 1.0 + rng.uniform(-brightness_range, brightness_range)  # contrast
    beta = rng.uniform(-10.0, 10.0)  # brightness
    out = out * alpha + beta

    # --------------------------------------------------------
    # 2. Gaussian noise
    # --------------------------------------------------------
    std = rng.uniform(0.0, gauss_std_max)
    noise = rng.normal(0.0, std, size=out.shape).astype(np.float32)
    out = out + noise

    # --------------------------------------------------------
    # 3. 배경 speckle point 추가
    #    검은 배경에 약한 점 노이즈를 추가한다.
    #    단, 너무 강하면 라벨 없는 객체처럼 보일 수 있으므로 낮게 유지한다.
    # --------------------------------------------------------
    h, w = out.shape[:2]
    n_speckle = int(h * w * speckle_prob)

    if n_speckle > 0:
        ys = rng.integers(0, h, size=n_speckle)
        xs = rng.integers(0, w, size=n_speckle)

        # 약한 회색/컬러 점
        vals = rng.integers(40, 180, size=(n_speckle, 3))
        out[ys, xs] = vals

    # --------------------------------------------------------
    # 4. 약한 blur
    # --------------------------------------------------------
    if rng.random() < blur_prob:
        out = cv2.GaussianBlur(out, (3, 3), 0)

    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def list_images(img_dir: Path):
    """폴더 내부 이미지 파일 목록을 정렬해서 반환한다."""
    if not img_dir.exists():
        return []
    return sorted([p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS])


def copy_label(src_label: Path, dst_label: Path):
    """YOLO label txt를 복사한다."""
    dst_label.parent.mkdir(parents=True, exist_ok=True)

    if not src_label.exists():
        # 라벨이 없는 경우 빈 txt 생성
        dst_label.write_text("", encoding="utf-8")
        return False

    shutil.copy2(src_label, dst_label)
    return True


def process_split(
    split: str,
    src_root: Path,
    out_root: Path,
    rng: np.random.Generator,
    include_original: bool,
    noisy_val: bool,
    gauss_std_max: float,
    speckle_prob: float,
    blur_prob: float,
    brightness_range: float,
):
    """
    train / val split을 처리한다.

    include_original=False:
        원본 개수와 동일한 noise-only 데이터셋 생성

    include_original=True:
        원본 + noise를 함께 저장하여 데이터 개수 2배 증가
    """
    src_img_dir = src_root / "images" / split
    src_lbl_dir = src_root / "labels" / split

    out_img_dir = out_root / "images" / split
    out_lbl_dir = out_root / "labels" / split

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(src_img_dir)

    made_images = 0
    missing_labels = 0

    for img_path in images:
        label_path = src_lbl_dir / f"{img_path.stem}.txt"

        # ----------------------------------------------------
        # val은 기본적으로 원본을 그대로 복사한다.
        # train만 noise를 주는 것이 원인 분리에 더 좋다.
        # ----------------------------------------------------
        apply_noise = True
        if split == "val" and not noisy_val:
            apply_noise = False

        # ----------------------------------------------------
        # 원본도 같이 포함하는 옵션
        # ----------------------------------------------------
        if include_original:
            dst_img = out_img_dir / img_path.name
            dst_lbl = out_lbl_dir / f"{img_path.stem}.txt"

            shutil.copy2(img_path, dst_img)
            ok = copy_label(label_path, dst_lbl)

            made_images += 1
            if not ok:
                missing_labels += 1

            noise_stem = f"{img_path.stem}_noise"
            dst_noise_img = out_img_dir / f"{noise_stem}{img_path.suffix}"
            dst_noise_lbl = out_lbl_dir / f"{noise_stem}.txt"
        else:
            # 원본 개수와 동일하게 유지
            dst_noise_img = out_img_dir / img_path.name
            dst_noise_lbl = out_lbl_dir / f"{img_path.stem}.txt"

        if apply_noise:
            img = read_image(img_path)
            noisy = add_pixel_noise(
                img,
                rng=rng,
                gauss_std_max=gauss_std_max,
                speckle_prob=speckle_prob,
                blur_prob=blur_prob,
                brightness_range=brightness_range,
            )
            cv2.imwrite(str(dst_noise_img), noisy)
        else:
            shutil.copy2(img_path, dst_noise_img)

        ok = copy_label(label_path, dst_noise_lbl)

        made_images += 1
        if not ok:
            missing_labels += 1

    return {
        "split": split,
        "src_images": len(images),
        "made_images": made_images,
        "missing_labels": missing_labels,
    }


def write_data_yaml(out_root: Path):
    """Ultralytics YOLO용 data.yaml 생성."""
    text = f"""path: {out_root}
train: images/train
val: images/val

names:
  0: class0
  1: class1
  2: class2
  3: class3
"""
    (out_root / "data.yaml").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--src-root",
        type=str,
        default="/Users/hajiwan/Desktop/object_detection/new_tema/yolov2/dataset/dataset26/standard",
        help="원본 standard YOLO dataset root",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default="/Users/hajiwan/Desktop/object_detection/new_tema/yolov2/dataset/dataset26/standard_noise_only_x1",
        help="생성할 noise-only YOLO dataset root",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="재현 가능한 noise 생성을 위한 seed",
    )
    parser.add_argument(
        "--include-original",
        action="store_true",
        help="원본 이미지도 함께 포함한다. 사용하면 train 개수가 증가함.",
    )
    parser.add_argument(
        "--noisy-val",
        action="store_true",
        help="val에도 noise를 적용한다. 기본값은 val 원본 유지.",
    )
    parser.add_argument(
        "--gauss-std-max",
        type=float,
        default=8.0,
        help="Gaussian noise 최대 표준편차",
    )
    parser.add_argument(
        "--speckle-prob",
        type=float,
        default=0.0003,
        help="배경 speckle point 비율",
    )
    parser.add_argument(
        "--blur-prob",
        type=float,
        default=0.25,
        help="Gaussian blur 적용 확률",
    )
    parser.add_argument(
        "--brightness-range",
        type=float,
        default=0.12,
        help="밝기/대비 변화 범위",
    )

    args = parser.parse_args()

    src_root = Path(args.src_root)
    out_root = Path(args.out_root)

    if not src_root.exists():
        raise FileNotFoundError(f"src-root 없음: {src_root}")

    # 기존 출력 폴더 삭제 후 새로 생성
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)

    summaries = []
    for split in ["train", "val"]:
        summary = process_split(
            split=split,
            src_root=src_root,
            out_root=out_root,
            rng=rng,
            include_original=args.include_original,
            noisy_val=args.noisy_val,
            gauss_std_max=args.gauss_std_max,
            speckle_prob=args.speckle_prob,
            blur_prob=args.blur_prob,
            brightness_range=args.brightness_range,
        )
        summaries.append(summary)

    write_data_yaml(out_root)

    print("============================================")
    print("[DONE] noise-only dataset 생성 완료")
    print("SRC :", src_root)
    print("OUT :", out_root)
    print("DATA:", out_root / "data.yaml")
    print("============================================")

    for s in summaries:
        print(
            f"{s['split']}: src_images={s['src_images']}, "
            f"made_images={s['made_images']}, "
            f"missing_labels={s['missing_labels']}"
        )


if __name__ == "__main__":
    main()
