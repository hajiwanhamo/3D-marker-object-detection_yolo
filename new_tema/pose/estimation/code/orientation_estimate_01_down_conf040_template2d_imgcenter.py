from pathlib import Path
from datetime import datetime
from collections import Counter
import csv

import cv2
import numpy as np

# 기존 GT-template2D 방향추정 코드 재사용
# class 조합 선택 / square 추정 / 방향 벡터 계산은 그대로 사용한다.
import orientation_estimate_01_down_conf040_template2d as base


# ============================================================
# 목적
# ------------------------------------------------------------
# CENTER 원인 분석 결과를 검증하기 위한 확인용 코드.
#
# 기존 template2d 코드:
#   CENTER = template fitting으로 계산된 중심
#
# 이번 imgcenter 코드:
#   방향 벡터 = 기존 template2d 결과 그대로 유지
#   CENTER   = image center (0.5, 0.5)로 고정
#
# 주의:
# - square_center - image_center로 방향을 다시 계산하지 않는다.
# - 기존 template2d가 계산한 north_vec 방향만 사용한다.
# - npy/xyz는 사용하지 않는다.
# ============================================================


POSE_ROOT = Path("/Users/hajiwan/Desktop/object_detection/new_tema/pose")


def make_output_dirs():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = POSE_ROOT / f"orientation_output_01_down_conf040_template2d_imgcenter_{ts}"

    image_dir = out_dir / "images_orientation"
    debug_dir = out_dir / "images_debug_selection"

    if out_dir.exists():
        raise FileExistsError(f"[ERROR] 출력 폴더 이미 존재: {out_dir}")

    image_dir.mkdir(parents=True, exist_ok=False)
    debug_dir.mkdir(parents=True, exist_ok=False)

    return out_dir, image_dir, debug_dir


def format_float(v):
    if v == "" or v is None:
        return ""
    try:
        return f"{float(v):.6f}"
    except Exception:
        return str(v)


def list_to_str(values):
    return "|".join(str(v) for v in values)


def apply_image_center(result):
    """
    기존 template2d 결과에서 CENTER만 image center로 교체한다.

    중요:
    - north_vec는 기존 result["north_vec"]를 그대로 유지한다.
    - 즉 방향은 기존 template2d 방향과 동일하다.
    """
    if "marker_center" not in result or "north_vec" not in result:
        return result

    new_result = dict(result)
    new_result["old_marker_center"] = np.asarray(result["marker_center"], dtype=np.float64)
    new_result["marker_center"] = np.array([0.5, 0.5], dtype=np.float64)
    new_result["north_vec"] = np.asarray(result["north_vec"], dtype=np.float64)
    new_result["center_source"] = "IMAGE_CENTER_FIXED"
    return new_result


def main():
    image_dir, label_dir = base.find_conf040_paths()
    out_dir, image_out_dir, debug_out_dir = make_output_dirs()

    print(f"[INFO] image_dir  = {image_dir}")
    print(f"[INFO] label_dir  = {label_dir}")
    print(f"[INFO] OUTPUT_DIR = {out_dir}")

    # 기존 GT-template2D와 동일한 template 사용
    gt_info = base.build_gt_template()
    template = gt_info["template"]

    print("[INFO] GT template samples:", gt_info["num_samples"])
    print("[INFO] GT residual mean:", gt_info["gt_residual_mean"])
    print("[INFO] GT residual p95 :", gt_info["gt_residual_p95"])
    print("[INFO] CENTER fixed to image center: normalized (0.5, 0.5)")

    txt_paths = sorted(label_dir.glob("*.txt"))

    if not txt_paths:
        raise FileNotFoundError(f"[ERROR] conf040 label txt 없음: {label_dir}")

    result_csv = out_dir / "orientation_results_conf040_template2d_imgcenter.csv"

    fieldnames = [
        "image_name",
        "status",
        "reason",
        "detected_classes",
        "used_classes",
        "class0_count",
        "class1_count",
        "class2_count",
        "class3_count",
        "square_estimated",
        "old_center_x",
        "old_center_y",
        "fixed_center_x",
        "fixed_center_y",
        "square_x",
        "square_y",
        "north_x",
        "north_y",
        "east_x",
        "east_y",
        "south_x",
        "south_y",
        "west_x",
        "west_y",
        "angle_deg",
        "residual_norm",
        "residual_mean",
        "confidence",
        "ranked_count",
        "center_source",
    ]

    status_counter = Counter()

    with result_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for txt_path in txt_paths:
            stem = txt_path.stem
            image_path = base.find_image_path(image_dir, stem)

            if image_path is None:
                print(f"[WARN] 이미지 없음: {stem}")
                continue

            image = cv2.imread(str(image_path))

            if image is None:
                print(f"[WARN] 이미지 읽기 실패: {image_path}")
                continue

            h, w = image.shape[:2]

            detections = base.read_yolo_seg_label(txt_path)
            grouped = base.group_by_class(detections)

            original_result = base.estimate_orientation_from_selection(grouped, template)
            result = apply_image_center(original_result)

            status_counter[result.get("status", "UNKNOWN")] += 1

            counts = {
                0: len(grouped.get(0, [])),
                1: len(grouped.get(1, [])),
                2: len(grouped.get(2, [])),
                3: len(grouped.get(3, [])),
            }

            detected_classes = original_result.get(
                "detected_classes",
                sorted([c for c in grouped.keys() if c in [0, 1, 2, 3]])
            )
            used_classes = original_result.get("used_classes", [])

            row = {
                "image_name": image_path.name,
                "status": result.get("status", ""),
                "reason": result.get("reason", ""),
                "detected_classes": list_to_str(detected_classes),
                "used_classes": list_to_str(used_classes),
                "class0_count": counts[0],
                "class1_count": counts[1],
                "class2_count": counts[2],
                "class3_count": counts[3],
                "square_estimated": result.get("square_estimated", ""),
                "old_center_x": "",
                "old_center_y": "",
                "fixed_center_x": "",
                "fixed_center_y": "",
                "square_x": "",
                "square_y": "",
                "north_x": "",
                "north_y": "",
                "east_x": "",
                "east_y": "",
                "south_x": "",
                "south_y": "",
                "west_x": "",
                "west_y": "",
                "angle_deg": "",
                "residual_norm": format_float(result.get("residual_norm", "")),
                "residual_mean": format_float(result.get("residual_mean", "")),
                "confidence": format_float(result.get("confidence", "")),
                "ranked_count": result.get("ranked_count", ""),
                "center_source": result.get("center_source", ""),
            }

            if result.get("old_marker_center", None) is not None:
                old = result["old_marker_center"]
                row["old_center_x"] = format_float(old[0] * w)
                row["old_center_y"] = format_float(old[1] * h)

            # image center
            row["fixed_center_x"] = format_float(0.5 * w)
            row["fixed_center_y"] = format_float(0.5 * h)

            if "square_center" in result:
                square = result["square_center"]
                row["square_x"] = format_float(square[0] * w)
                row["square_y"] = format_float(square[1] * h)

            if "marker_center" in result and "north_vec" in result:
                dir_points = base.make_direction_points(result, image.shape)

                if dir_points is not None:
                    drawn = base.draw_orientation(image, result, dir_points, debug=False)
                    debug_drawn = base.draw_orientation(image, result, dir_points, debug=True)

                    cv2.imwrite(str(image_out_dir / f"{stem}_orientation.jpg"), drawn)
                    cv2.imwrite(str(debug_out_dir / f"{stem}_debug.jpg"), debug_drawn)

                    row.update({
                        "north_x": format_float(dir_points["N"][0]),
                        "north_y": format_float(dir_points["N"][1]),
                        "east_x": format_float(dir_points["E"][0]),
                        "east_y": format_float(dir_points["E"][1]),
                        "south_x": format_float(dir_points["S"][0]),
                        "south_y": format_float(dir_points["S"][1]),
                        "west_x": format_float(dir_points["W"][0]),
                        "west_y": format_float(dir_points["W"][1]),
                        "angle_deg": format_float(dir_points["angle_deg"]),
                    })
                else:
                    fail_drawn = base.draw_fail_image(image, {
                        "status": "FAIL_BAD_DIRECTION_POINTS",
                        "reason": "make_direction_points returned None",
                        "detected_classes": detected_classes,
                    })
                    cv2.imwrite(str(image_out_dir / f"{stem}_orientation.jpg"), fail_drawn)
                    cv2.imwrite(str(debug_out_dir / f"{stem}_debug.jpg"), fail_drawn)

            else:
                fail_drawn = base.draw_fail_image(image, result)
                cv2.imwrite(str(image_out_dir / f"{stem}_orientation.jpg"), fail_drawn)
                cv2.imwrite(str(debug_out_dir / f"{stem}_debug.jpg"), fail_drawn)

            writer.writerow(row)

    print("[DONE] image-center 고정 template2d orientation 완료")
    print(f"[DONE] OUTPUT_DIR: {out_dir}")
    print(f"[DONE] images_orientation: {image_out_dir}")
    print(f"[DONE] images_debug_selection: {debug_out_dir}")
    print(f"[DONE] CSV: {result_csv}")
    print("[DONE] status summary:")

    for k, v in sorted(status_counter.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
