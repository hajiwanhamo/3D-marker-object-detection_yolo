from pathlib import Path
import shutil
import cv2
import numpy as np

ROOT = Path("/Users/hajiwan/Desktop/object_detection/new_tema/yolov2/realdata/range_sweep_10sets_keepupper")
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def is_color_image(path: Path) -> bool:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if img is None:
        return False

    if img.ndim == 2:
        return False

    if img.shape[2] < 3:
        return False

    bgr = img[:, :, :3].astype(np.int16)
    b = bgr[:, :, 0]
    g = bgr[:, :, 1]
    r = bgr[:, :, 2]

    diff_bg = np.mean(np.abs(b - g))
    diff_gr = np.mean(np.abs(g - r))
    diff_br = np.mean(np.abs(b - r))

    return max(diff_bg, diff_gr, diff_br) > 2.0


def main():
    if not ROOT.exists():
        raise FileNotFoundError(f"ROOT 없음: {ROOT}")

    dirs = sorted([p for p in ROOT.iterdir() if p.is_dir() and p.name.startswith("real_flat_")])

    for d in dirs:
        out = d / "images_color"

        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True, exist_ok=True)

        image_files = [
            p for p in d.rglob("*")
            if p.is_file()
            and p.suffix.lower() in IMG_EXTS
            and "images_color" not in p.parts
        ]

        copied = 0
        skipped = 0

        for p in sorted(image_files):
            if is_color_image(p):
                dst = out / p.name
                if dst.exists():
                    dst = out / f"{p.parent.name}_{p.name}"
                shutil.copy2(p, dst)
                copied += 1
            else:
                skipped += 1

        print("=" * 70)
        print(f"DIR: {d}")
        print(f"found images: {len(image_files)}")
        print(f"color copied: {copied}")
        print(f"gray skipped: {skipped}")
        print(f"images_color: {out}")

    print("\n[OK] keep-upper 컬러 이미지 정리 완료")


if __name__ == "__main__":
    main()
