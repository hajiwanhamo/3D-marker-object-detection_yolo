from pathlib import Path
import argparse
import csv

import cv2


# ============================================================
# clean 2-class YOLO 결과의 square 후보 crop 확인 코드
#
# 목적:
#   YOLO 2-class predict 결과에서 class 0(square_id) 후보만 crop하여 확인한다.
#
# 입력:
#   real_images/*.png
#   result/predict_real_yolo11n_detect_2class/labels/*.txt
#
# 출력:
#   out_dir/overview/*.png
#   out_dir/square_crops/*.png
#   out_dir/square_candidate_summary.csv
#
# 주의:
#   - 학습 데이터 수정 없음
#   - 라벨 수정 없음
#   - YOLO 결과 수정 없음
#   - 분석용 이미지와 CSV만 생성
# ============================================================

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def collect_images(image_dir: Path):
    """실해역 이미지 목록 수집"""
    if not image_dir.exists():
        raise FileNotFoundError(f"image_dir 없음: {image_dir}")

    images = []

    for path in image_dir.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            images.append(path)

    return sorted(images)


def read_yolo_txt(txt_path: Path):
    """
    YOLO predict txt 읽기

    지원 형식:
      class x_center y_center width height
      class x_center y_center width height confidence
    """
    preds = []

    if not txt_path.exists():
        return preds

    with open(txt_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            parts = line.strip().split()

            if len(parts) not in [5, 6]:
                continue

            class_id = int(float(parts[0]))

            preds.append({
                "line_idx": line_idx,
                "class_id": class_id,
                "x": float(parts[1]),
                "y": float(parts[2]),
                "w": float(parts[3]),
                "h": float(parts[4]),
                "conf": float(parts[5]) if len(parts) == 6 else 1.0,
            })

    return preds


def box_area(pred):
    """정규화 bbox 면적"""
    return max(0.0, pred["w"]) * max(0.0, pred["h"])


def box_aspect(pred):
    """bbox 장축/단축 비율"""
    w = max(pred["w"], 1e-8)
    h = max(pred["h"], 1e-8)

    return max(w / h, h / w)


def norm_to_xyxy(pred, image_w: int, image_h: int):
    """정규화 bbox를 pixel 좌표로 변환"""
    x1 = int(round((pred["x"] - pred["w"] / 2.0) * image_w))
    y1 = int(round((pred["y"] - pred["h"] / 2.0) * image_h))
    x2 = int(round((pred["x"] + pred["w"] / 2.0) * image_w))
    y2 = int(round((pred["y"] + pred["h"] / 2.0) * image_h))

    x1 = max(0, min(image_w - 1, x1))
    y1 = max(0, min(image_h - 1, y1))
    x2 = max(0, min(image_w - 1, x2))
    y2 = max(0, min(image_h - 1, y2))

    return x1, y1, x2, y2


def expand_xyxy(x1, y1, x2, y2, image_w: int, image_h: int, margin_px: int):
    """crop 확인용 bbox margin 확장"""
    xx1 = max(0, x1 - margin_px)
    yy1 = max(0, y1 - margin_px)
    xx2 = min(image_w - 1, x2 + margin_px)
    yy2 = min(image_h - 1, y2 + margin_px)

    return xx1, yy1, xx2, yy2


def draw_overview(image, square_preds, rect_preds, image_w: int, image_h: int):
    """
    전체 이미지 위에 square/rect 후보 표시
    blue: square 후보
    yellow: rect 후보
    """
    vis = image.copy()

    # rect 후보는 얇게 표시
    for pred in rect_preds:
        x1, y1, x2, y2 = norm_to_xyxy(pred, image_w, image_h)

        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 1)
        cv2.putText(
            vis,
            f"R:{pred['conf']:.2f}",
            (x1, max(15, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # square 후보는 굵게 표시
    for idx, pred in enumerate(square_preds):
        x1, y1, x2, y2 = norm_to_xyxy(pred, image_w, image_h)

        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 3)
        cv2.putText(
            vis,
            f"S{idx}: conf={pred['conf']:.2f}, area={box_area(pred):.3f}, asp={box_aspect(pred):.2f}",
            (x1, min(image_h - 5, y2 + 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

    return vis


def draw_crop(crop, pred):
    """crop 이미지 위에 후보 정보 텍스트 표시"""
    out = crop.copy()

    text_lines = [
        f"class 0 square candidate",
        f"conf: {pred['conf']:.4f}",
        f"area: {box_area(pred):.6f}",
        f"aspect: {box_aspect(pred):.4f}",
        f"x,y,w,h: {pred['x']:.3f}, {pred['y']:.3f}, {pred['w']:.3f}, {pred['h']:.3f}",
    ]

    y = 18

    for text in text_lines:
        cv2.putText(
            out,
            text,
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            text,
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        y += 18

    return out


def save_csv(path: Path, rows):
    """CSV 저장"""
    if not rows:
        return

    fieldnames = sorted(set().union(*[row.keys() for row in rows]))

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image_dir",
        type=str,
        default="../real_images",
        help="실해역 이미지 폴더",
    )

    parser.add_argument(
        "--pred_label_dir",
        type=str,
        required=True,
        help="YOLO 2-class predict labels 폴더",
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="square 후보 crop 결과 저장 폴더",
    )

    parser.add_argument(
        "--margin_px",
        type=int,
        default=25,
        help="crop 주변 margin pixel",
    )

    parser.add_argument(
        "--min_conf",
        type=float,
        default=0.0,
        help="저장할 square 후보 최소 confidence",
    )

    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    pred_label_dir = Path(args.pred_label_dir)
    out_dir = Path(args.out_dir)

    overview_dir = out_dir / "overview"
    crop_dir = out_dir / "square_crops"

    overview_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)

    images = collect_images(image_dir)

    print("========== CONFIG ==========")
    print(f"image_dir:      {image_dir.resolve()}")
    print(f"pred_label_dir: {pred_label_dir.resolve()}")
    print(f"out_dir:        {out_dir.resolve()}")
    print(f"image_count:    {len(images)}")
    print(f"margin_px:      {args.margin_px}")
    print(f"min_conf:       {args.min_conf}")
    print("============================")

    rows = []

    for image_idx, image_path in enumerate(images):
        stem = image_path.stem
        pred_txt = pred_label_dir / f"{stem}.txt"

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

        if image is None:
            print(f"[FAIL] {stem}: 이미지 읽기 실패")
            rows.append({
                "stem": stem,
                "status": "image_read_failed",
            })
            continue

        image_h, image_w = image.shape[:2]
        preds = read_yolo_txt(pred_txt)

        square_preds = [
            p for p in preds
            if p["class_id"] == 0 and p["conf"] >= args.min_conf
        ]

        rect_preds = [
            p for p in preds
            if p["class_id"] == 1
        ]

        square_preds = sorted(square_preds, key=lambda p: p["conf"], reverse=True)
        rect_preds = sorted(rect_preds, key=lambda p: p["conf"], reverse=True)

        overview = draw_overview(image, square_preds, rect_preds, image_w, image_h)
        cv2.imwrite(str(overview_dir / f"{stem}_overview.png"), overview)

        if len(square_preds) == 0:
            print(f"[{image_idx + 1}/{len(images)}] {stem}: square 후보 없음")
            rows.append({
                "stem": stem,
                "status": "no_square_candidate",
                "square_count": 0,
                "rect_count": len(rect_preds),
            })
            continue

        for square_idx, pred in enumerate(square_preds):
            x1, y1, x2, y2 = norm_to_xyxy(pred, image_w, image_h)
            xx1, yy1, xx2, yy2 = expand_xyxy(
                x1, y1, x2, y2,
                image_w=image_w,
                image_h=image_h,
                margin_px=args.margin_px,
            )

            crop = image[yy1:yy2 + 1, xx1:xx2 + 1].copy()

            if crop.size == 0:
                continue

            crop_vis = draw_crop(crop, pred)

            crop_name = (
                f"{stem}_S{square_idx:02d}"
                f"_conf{pred['conf']:.3f}"
                f"_area{box_area(pred):.4f}"
                f"_asp{box_aspect(pred):.2f}.png"
            )

            cv2.imwrite(str(crop_dir / crop_name), crop_vis)

            rows.append({
                "stem": stem,
                "status": "ok",
                "square_index": square_idx,
                "square_count": len(square_preds),
                "rect_count": len(rect_preds),
                "conf": pred["conf"],
                "area": box_area(pred),
                "aspect": box_aspect(pred),
                "x": pred["x"],
                "y": pred["y"],
                "w": pred["w"],
                "h": pred["h"],
                "crop_file": crop_name,
            })

        print(
            f"[{image_idx + 1}/{len(images)}] {stem}: "
            f"square={len(square_preds)}, rect={len(rect_preds)}"
        )

    summary_path = out_dir / "square_candidate_summary.csv"
    save_csv(summary_path, rows)

    print("\n========== DONE ==========")
    print(f"overview: {overview_dir}")
    print(f"crops:    {crop_dir}")
    print(f"summary:  {summary_path}")
    print("==========================")


if __name__ == "__main__":
    main()