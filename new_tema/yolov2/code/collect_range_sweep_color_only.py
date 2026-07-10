from pathlib import Path
import shutil
import cv2
import numpy as np

ROOT = Path("/Users/hajiwan/Desktop/object_detection/new_tema/yolov2/realdata/range_sweep_10sets")
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def is_color_image(path: Path) -> bool:
    """
    컬러 projection만 True.
    흑백 이미지는 R/G/B 채널 차이가 거의 없으므로 제외.
    """
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if img is None:
        return False

    # 1채널이면 흑백
    if img.ndim == 2:
        return False

    # 3채널 미만이면 제외
    if img.shape[2] < 3:
        return False

    bgr = img[:, :, :3].astype(np.int16)
    b = bgr[:, :, 0]
    g = bgr[:, :, 1]
    r = bgr[:, :, 2]

    diff_bg = float(np.mean(np.abs(b - g)))
    diff_gr = float(np.mean(np.abs(g - r)))
    diff_br = float(np.mean(np.abs(b - r)))

    # Jet/raw_z 컬러맵이면 채널 차이가 존재함
    return max(diff_bg, diff_gr, diff_br) > 2.0


def main():
    if not ROOT.exists():
        raise FileNotFoundError(f"ROOT 없음: {ROOT}")

    dirs = sorted([p for p in ROOT.iterdir() if p.is_dir() and p.name.startswith("real_flat_")])

    if not dirs:
        raise RuntimeError(f"real_flat_* 폴더 없음: {ROOT}")

    print(f"[INFO] target dirs: {len(dirs)}")

    for d in dirs:
        out = d / "images_color"

        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True, exist_ok=True)

        # images_color 내부는 제외하고, 해당 projection 폴더 전체에서 이미지 검색
        image_files = []
        for p in d.rglob("*"):
            if not p.is_file():
                continue
            if "images_color" in p.parts:
                continue
            if p.suffix.lower() in IMG_EXTS:
                image_files.append(p)

        copied = 0
        skipped_gray = 0
        skipped_unreadable = 0

        for p in sorted(image_files):
            img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if img is None:
                skipped_unreadable += 1
                continue

            if is_color_image(p):
                # 같은 이름 충돌 방지
                dst_name = p.name
                dst = out / dst_name
                if dst.exists():
                    dst = out / f"{p.parent.name}_{p.name}"
                shutil.copy2(p, dst)
                copied += 1
            else:
                skipped_gray += 1

        print("=" * 80)
        print(f"DIR: {d}")
        print(f"found image files: {len(image_files)}")
        print(f"color copied: {copied}")
        print(f"gray skipped: {skipped_gray}")
        print(f"unreadable skipped: {skipped_unreadable}")
        print(f"images_color: {out}")

    print("\n[OK] images_color 생성 완료")


if __name__ == "__main__":
    main()
