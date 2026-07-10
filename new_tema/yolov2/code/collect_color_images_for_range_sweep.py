from pathlib import Path
import shutil
import cv2
import numpy as np

ROOTS = [
    Path("/Users/hajiwan/Desktop/object_detection/new_tema/yolov2/realdata/range_sweep_center_fixed"),
    Path("/Users/hajiwan/Desktop/object_detection/new_tema/yolov2/realdata/range_sweep_keepupper_v2"),
]

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def is_color_image(path: Path) -> bool:
    """
    컬러 projection만 True.
    흑백 이미지는 B/G/R 채널 차이가 거의 없으므로 제외.
    """
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if img is None:
        return False

    if img.ndim == 2:
        return False

    if len(img.shape) < 3 or img.shape[2] < 3:
        return False

    bgr = img[:, :, :3].astype(np.int16)
    b = bgr[:, :, 0]
    g = bgr[:, :, 1]
    r = bgr[:, :, 2]

    diff_bg = float(np.mean(np.abs(b - g)))
    diff_gr = float(np.mean(np.abs(g - r)))
    diff_br = float(np.mean(np.abs(b - r)))

    return max(diff_bg, diff_gr, diff_br) > 2.0


def collect_one_root(root: Path):
    if not root.exists():
        print(f"[SKIP] ROOT 없음: {root}")
        return

    dirs = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("real_flat_")])

    if not dirs:
        print(f"[SKIP] real_flat_* 폴더 없음: {root}")
        return

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
        skipped_gray = 0
        skipped_unreadable = 0

        for p in sorted(image_files):
            img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if img is None:
                skipped_unreadable += 1
                continue

            if is_color_image(p):
                dst = out / p.name

                # 이름 충돌 방지
                if dst.exists():
                    dst = out / f"{p.parent.name}_{p.name}"

                shutil.copy2(p, dst)
                copied += 1
            else:
                skipped_gray += 1

        print("=" * 90)
        print(f"ROOT: {root}")
        print(f"DIR: {d}")
        print(f"found image files: {len(image_files)}")
        print(f"color copied: {copied}")
        print(f"gray skipped: {skipped_gray}")
        print(f"unreadable skipped: {skipped_unreadable}")
        print(f"images_color: {out}")


def main():
    for root in ROOTS:
        collect_one_root(root)

    print("\n[OK] images_color 폴더 생성 완료")


if __name__ == "__main__":
    main()
