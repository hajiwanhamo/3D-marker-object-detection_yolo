from pathlib import Path
import shutil
import cv2
import numpy as np

ROOT = Path("/Users/hajiwan/Desktop/object_detection/new_tema/yolov2/realdata")

TARGET_DIRS = [
    ROOT / "real_flat_hband003_max008",
    ROOT / "real_flat_hband004_max009",
    ROOT / "real_flat_hband005_max010",
    ROOT / "real_flat_hband006_max012",
]

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def is_color_image(path: Path) -> bool:
    """
    흑백 이미지는 R=G=B 값이 거의 동일함.
    컬러 projection은 채널 간 차이가 존재함.
    """
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if img is None:
        return False

    # 1채널이면 흑백
    if img.ndim == 2:
        return False

    # alpha 채널이 있으면 BGR만 사용
    if img.shape[2] >= 3:
        bgr = img[:, :, :3].astype(np.int16)
    else:
        return False

    b = bgr[:, :, 0]
    g = bgr[:, :, 1]
    r = bgr[:, :, 2]

    # 채널 차이가 거의 없으면 흑백으로 판단
    diff_bg = np.mean(np.abs(b - g))
    diff_gr = np.mean(np.abs(g - r))
    diff_br = np.mean(np.abs(b - r))

    return max(diff_bg, diff_gr, diff_br) > 2.0


def main():
    for d in TARGET_DIRS:
        if not d.exists():
            print(f"[SKIP] 폴더 없음: {d}")
            continue

        out = d / "images_color"

        # 기존 images_color를 새로 구성
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True, exist_ok=True)

        copied = 0
        skipped_gray = 0
        skipped_other = 0

        # 루트 바로 아래 이미지 파일만 검사
        for p in sorted(d.iterdir()):
            if not p.is_file() or p.suffix.lower() not in IMG_EXTS:
                continue

            if is_color_image(p):
                shutil.copy2(p, out / p.name)
                copied += 1
            else:
                skipped_gray += 1

        print("=" * 60)
        print(f"DIR: {d}")
        print(f"color copied: {copied}")
        print(f"gray skipped: {skipped_gray}")
        print(f"images_color: {out}")

    print("\n[OK] 컬러 이미지만 images_color에 저장 완료")


if __name__ == "__main__":
    main()
