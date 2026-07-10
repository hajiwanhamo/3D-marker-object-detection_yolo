from pathlib import Path
import argparse
import json
import shutil

import cv2
import numpy as np


# ============================================================
# 가상데이터 YOLO 학습용 이미지 생성 코드 (좌우반전 보정용)
#
# 목적:
#   eval1 결과 폴더(2D_image) 안의 uv/meta 파일을 읽어
#   실해역 방향과 맞는 YOLO 학습용 color 이미지를 다시 생성한다.
#
# 핵심:
#   - 기존 가상데이터가 실해역 대비 좌우반전 상태였으므로
#   - X축 flip을 적용해서 다시 생성한다.
#
# 입력 예:
#   aug_000000_marker_meta.json
#   aug_000000_marker_marker_all_uv.npy
#   aug_000000_marker_top_id_uv.npy
#
# 우선순위:
#   1) *_marker_marker_all_uv.npy
#   2) *_marker_all_uv.npy
#   3) *_marker_top_id_uv.npy
#
# 출력:
#   aug_000000.png
#
# 주의:
#   이 코드는 "이미지 생성"만 담당한다.
#   이후 라벨 생성 코드도 동일한 flip_x 기준으로 다시 생성해야
#   이미지와 라벨이 일치한다.
# ============================================================


def load_json(json_path: Path) -> dict:
    """json 읽기"""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_base_stem_from_meta(meta_path: Path) -> str:
    """
    aug_000000_marker_meta.json -> aug_000000
    """
    suffix = "_marker_meta.json"
    name = meta_path.name

    if not name.endswith(suffix):
        raise RuntimeError(f"meta 파일 이름 형식이 예상과 다릅니다: {meta_path}")

    return name[:-len(suffix)]


def find_uv_file(src_dir: Path, base_stem: str) -> Path | None:
    """
    사용할 uv 파일 우선순위 탐색
    """
    candidates = [
        src_dir / f"{base_stem}_marker_marker_all_uv.npy",
        src_dir / f"{base_stem}_marker_all_uv.npy",
        src_dir / f"{base_stem}_marker_top_id_uv.npy",
    ]

    for p in candidates:
        if p.exists():
            return p

    return None


def load_uv_array(uv_path: Path) -> np.ndarray:
    """
    uv npy 로드
    최소 2열(u, v)은 필요
    """
    arr = np.load(str(uv_path), allow_pickle=True)
    arr = np.asarray(arr)

    if arr.ndim == 3:
        arr = arr.reshape(-1, arr.shape[-1])

    if arr.ndim != 2 or arr.shape[1] < 2:
        raise RuntimeError(f"uv 배열 형식 오류: {uv_path}, shape={arr.shape}")

    arr = arr.astype(np.float32)

    valid = np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1])
    arr = arr[valid]

    if len(arr) == 0:
        raise RuntimeError(f"유효한 uv 포인트가 없음: {uv_path}")

    return arr


def choose_scalar_for_colormap(uv_arr: np.ndarray) -> np.ndarray:
    """
    색상용 scalar 선택
    - 3열 이상이면 3열 사용
    - 없으면 v 사용
    """
    if uv_arr.shape[1] >= 3:
        scalar = uv_arr[:, 2]
    else:
        scalar = uv_arr[:, 1]

    scalar = np.asarray(scalar, dtype=np.float32)

    valid = np.isfinite(scalar)
    if np.sum(valid) == 0:
        scalar = uv_arr[:, 1].astype(np.float32)

    return scalar


def normalize_to_uint8(values: np.ndarray) -> np.ndarray:
    """
    float 배열을 0~255 uint8로 정규화
    """
    values = np.asarray(values, dtype=np.float32)

    valid = np.isfinite(values)
    if np.sum(valid) == 0:
        return np.zeros_like(values, dtype=np.uint8)

    v = values[valid]
    vmin = float(np.min(v))
    vmax = float(np.max(v))

    if abs(vmax - vmin) < 1e-8:
        return np.full_like(values, 220, dtype=np.uint8)

    norm = (values - vmin) / (vmax - vmin)
    norm = np.clip(norm, 0.0, 1.0)

    return (norm * 255.0).astype(np.uint8)


def uv_to_pixel_flipx(uv_arr: np.ndarray, meta: dict, image_size: int) -> tuple[np.ndarray, np.ndarray]:
    """
    uv -> pixel 변환
    핵심 수정:
      X축 flip 적용
    """
    u = uv_arr[:, 0]
    v = uv_arr[:, 1]

    # meta에 있는 값 사용
    u_min = float(meta["u_min"])
    v_max = float(meta["v_max"])
    pixel_size_u_m = float(meta["pixel_size_u_m"])
    pixel_size_v_m = float(meta["pixel_size_v_m"])

    # 기본 정방향
    px = (u - u_min) / pixel_size_u_m
    py = (v_max - v) / pixel_size_v_m

    # -------------------------------
    # 핵심: 좌우반전 보정
    # -------------------------------
    px = (image_size - 1) - px

    px = np.round(px).astype(np.int32)
    py = np.round(py).astype(np.int32)

    px = np.clip(px, 0, image_size - 1)
    py = np.clip(py, 0, image_size - 1)

    return px, py


def render_color_image(px: np.ndarray, py: np.ndarray, scalar_u8: np.ndarray,
                       image_size: int, point_radius: int, blur_ksize: int) -> np.ndarray:
    """
    color 이미지 렌더링
    """
    gray = np.zeros((image_size, image_size), dtype=np.uint8)

    # 같은 위치에 여러 포인트가 오면 최대값 유지
    for x, y, s in zip(px, py, scalar_u8):
        if s > gray[y, x]:
            gray[y, x] = s

    # 점이 너무 성기지 않도록 확장
    kernel_size = max(1, point_radius * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    gray = cv2.dilate(gray, kernel, iterations=1)

    # 너무 각지지 않게 blur
    if blur_ksize % 2 == 0:
        blur_ksize += 1
    blur_ksize = max(3, blur_ksize)
    gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    # jet 컬러맵 적용
    color = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    # 배경은 검정으로
    color[gray == 0] = (0, 0, 0)

    return color


def draw_preview_text(image: np.ndarray, base_stem: str, uv_name: str) -> np.ndarray:
    """
    확인용 미리보기 텍스트 추가
    """
    vis = image.copy()

    texts = [
        f"{base_stem}",
        f"source uv: {uv_name}",
        "flip_x applied: YES",
    ]

    y = 22
    for t in texts:
        cv2.putText(
            vis,
            t,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        cv2.putText(
            vis,
            t,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )
        y += 22

    return vis


def process_one(meta_path: Path, src_dir: Path, out_dir: Path, preview_dir: Path | None,
                image_size_default: int, point_radius: int, blur_ksize: int):
    """
    파일 1개 처리
    """
    meta = load_json(meta_path)
    base_stem = find_base_stem_from_meta(meta_path)

    uv_path = find_uv_file(src_dir, base_stem)
    if uv_path is None:
        raise RuntimeError(f"uv 파일을 찾지 못했습니다: {base_stem}")

    uv_arr = load_uv_array(uv_path)

    image_size = int(meta.get("image_size", image_size_default))

    px, py = uv_to_pixel_flipx(uv_arr, meta, image_size)

    scalar = choose_scalar_for_colormap(uv_arr)
    scalar_u8 = normalize_to_uint8(scalar)

    color_img = render_color_image(
        px=px,
        py=py,
        scalar_u8=scalar_u8,
        image_size=image_size,
        point_radius=point_radius,
        blur_ksize=blur_ksize
    )

    out_img_path = out_dir / f"{base_stem}.png"
    cv2.imwrite(str(out_img_path), color_img)

    if preview_dir is not None:
        preview = draw_preview_text(color_img, base_stem, uv_path.name)
        cv2.imwrite(str(preview_dir / f"{base_stem}_preview.png"), preview)

    return base_stem, uv_path.name, out_img_path


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--src_dir",
        type=str,
        required=True,
        help="가상데이터 2D_image 폴더 경로"
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="생성된 이미지를 저장할 폴더"
    )

    parser.add_argument(
        "--image_size_default",
        type=int,
        default=512,
        help="meta에 image_size가 없을 때 사용할 기본 크기"
    )

    parser.add_argument(
        "--point_radius",
        type=int,
        default=2,
        help="포인트 확장 반경"
    )

    parser.add_argument(
        "--blur_ksize",
        type=int,
        default=5,
        help="Gaussian blur kernel size"
    )

    parser.add_argument(
        "--save_preview",
        action="store_true",
        help="확인용 preview 이미지도 같이 저장"
    )

    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    out_dir = Path(args.out_dir)

    if not src_dir.exists():
        raise FileNotFoundError(f"src_dir 없음: {src_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    preview_dir = None
    if args.save_preview:
        preview_dir = out_dir / "preview"
        preview_dir.mkdir(parents=True, exist_ok=True)

    meta_files = sorted(src_dir.glob("*_marker_meta.json"))

    if len(meta_files) == 0:
        raise RuntimeError(f"meta 파일을 찾지 못했습니다: {src_dir}")

    print("========== CONFIG ==========")
    print(f"src_dir:            {src_dir}")
    print(f"out_dir:            {out_dir}")
    print(f"image_size_default: {args.image_size_default}")
    print(f"point_radius:       {args.point_radius}")
    print(f"blur_ksize:         {args.blur_ksize}")
    print(f"save_preview:       {args.save_preview}")
    print("중요: virtual image에 X축 flip 적용")
    print("============================")

    success = 0
    failed = 0

    for idx, meta_path in enumerate(meta_files):
        try:
            base_stem, uv_name, out_img_path = process_one(
                meta_path=meta_path,
                src_dir=src_dir,
                out_dir=out_dir,
                preview_dir=preview_dir,
                image_size_default=args.image_size_default,
                point_radius=args.point_radius,
                blur_ksize=args.blur_ksize
            )

            success += 1
            print(f"[OK] {idx + 1}/{len(meta_files)} {base_stem} | uv={uv_name} | out={out_img_path.name}")

        except Exception as e:
            failed += 1
            print(f"[FAIL] {idx + 1}/{len(meta_files)} {meta_path.name}: {e}")

    print("\n========== DONE ==========")
    print(f"success: {success}")
    print(f"failed:  {failed}")
    print("==========================")

    if failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()