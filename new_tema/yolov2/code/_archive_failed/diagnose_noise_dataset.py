import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


CLASS_NAMES = {
    0: "square",
    1: "rect1",
    2: "rect2",
    3: "rect3",
}


def list_images(img_dir: Path):
    """이미지 폴더 내부 이미지 목록을 정렬해서 반환한다."""
    if not img_dir.exists():
        return []
    return sorted([p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS])


def read_image(path: Path):
    """이미지를 BGR로 읽는다."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"[ERROR] 이미지를 읽을 수 없습니다: {path}")
    return img


def load_yolo_segments(label_path: Path, w: int, h: int):
    """
    YOLO segmentation label txt를 읽는다.

    반환:
    [
        {
            "cls": int,
            "pts": np.ndarray(N,2)
        },
        ...
    ]
    """
    objects = []

    if not label_path.exists():
        return objects

    lines = label_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts = line.split()

        try:
            vals = [float(x) for x in parts]
        except ValueError:
            continue

        # class + 최소 3개 점
        if len(vals) < 7:
            continue

        cls_id = int(vals[0])
        coords = vals[1:]

        if len(coords) % 2 != 0:
            continue

        pts = np.array(coords, dtype=np.float32).reshape(-1, 2)
        pts[:, 0] *= w
        pts[:, 1] *= h

        objects.append({
            "cls": cls_id,
            "pts": pts,
        })

    return objects


def polygon_mask(pts: np.ndarray, h: int, w: int):
    """polygon을 binary mask로 변환한다."""
    mask = np.zeros((h, w), dtype=np.uint8)

    if pts is None or len(pts) < 3:
        return mask.astype(bool)

    poly = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [poly], 255)

    return mask > 0


def get_foreground_mask(img: np.ndarray, fg_th: int):
    """
    실제 남아있는 point/ID 픽셀을 foreground로 판단한다.

    검은 배경 기준:
    gray > fg_th 이면 visible point로 간주한다.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fg = gray > fg_th
    return fg, gray


def connected_component_stats(visible_mask: np.ndarray):
    """
    visible 영역이 얼마나 조각나 있는지 계산한다.

    반환:
    - component_count: 작은 조각 포함 connected component 수
    - largest_ratio: 전체 visible pixel 중 가장 큰 component 비율
    """
    u8 = visible_mask.astype(np.uint8)

    if u8.sum() == 0:
        return 0, 0.0

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(u8, connectivity=8)

    areas = []
    for lab in range(1, num_labels):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area >= 2:
            areas.append(area)

    if not areas:
        return 0, 0.0

    total = float(sum(areas))
    largest = float(max(areas))
    return len(areas), largest / max(total, 1.0)


def draw_overlay(img: np.ndarray, objects, fg_th: int):
    """
    이미지 위에 라벨 polygon과 foreground를 같이 표시한다.

    표시 의미:
    - 라벨 polygon 외곽선: class별 색상
    - 실제 visible foreground: 반투명 흰색 강조
    """
    vis = img.copy()
    h, w = img.shape[:2]

    fg, _ = get_foreground_mask(img, fg_th=fg_th)

    # 실제 남은 픽셀을 약하게 밝게 표시
    fg_overlay = vis.copy()
    fg_overlay[fg] = np.clip(fg_overlay[fg].astype(np.int16) + 55, 0, 255).astype(np.uint8)
    vis = cv2.addWeighted(fg_overlay, 0.45, vis, 0.55, 0)

    colors = {
        0: (0, 255, 255),   # square
        1: (0, 255, 0),     # rect1
        2: (255, 0, 255),   # rect2
        3: (255, 255, 0),   # rect3
    }

    for obj in objects:
        cls_id = obj["cls"]
        pts = obj["pts"]
        color = colors.get(cls_id, (255, 255, 255))

        poly = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [poly], isClosed=True, color=color, thickness=2)

        x, y = np.round(pts.mean(axis=0)).astype(int)
        x = int(np.clip(x, 0, w - 1))
        y = int(np.clip(y, 0, h - 1))

        cv2.putText(
            vis,
            f"{cls_id}:{CLASS_NAMES.get(cls_id, 'cls')}",
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    return vis


def make_contact_sheet(items, out_path: Path, tile_w: int = 320, tile_h: int = 320, cols: int = 4):
    """여러 overlay 이미지를 하나의 contact sheet로 저장한다."""
    if not items:
        return

    rows = int(np.ceil(len(items) / cols))
    sheet = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)

    for idx, (title, img) in enumerate(items):
        r = idx // cols
        c = idx % cols

        resized = cv2.resize(img, (tile_w, tile_h), interpolation=cv2.INTER_AREA)

        y1 = r * tile_h
        x1 = c * tile_w
        sheet[y1:y1 + tile_h, x1:x1 + tile_w] = resized

        cv2.putText(
            sheet,
            title[:38],
            (x1 + 8, y1 + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet)


def percentile(values, q):
    """빈 리스트 대응 percentile."""
    if not values:
        return 0.0
    return float(np.percentile(np.array(values, dtype=np.float32), q))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset-root", type=str, required=True, help="진단할 YOLO dataset root")
    parser.add_argument("--out-dir", type=str, required=True, help="진단 결과 저장 폴더")
    parser.add_argument("--fg-th", type=int, default=8, help="foreground threshold")
    parser.add_argument("--low-ratio-th", type=float, default=0.25, help="visible_ratio 낮음 기준")
    parser.add_argument("--max-samples", type=int, default=32, help="contact sheet에 넣을 최대 샘플 수")

    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    detail_rows = []
    overlay_items = []
    low_items = []

    for split in ["train", "val"]:
        img_dir = dataset_root / "images" / split
        lbl_dir = dataset_root / "labels" / split

        images = list_images(img_dir)

        for img_path in images:
            img = read_image(img_path)
            h, w = img.shape[:2]

            label_path = lbl_dir / f"{img_path.stem}.txt"
            objects = load_yolo_segments(label_path, w, h)

            fg, gray = get_foreground_mask(img, fg_th=args.fg_th)

            if len(overlay_items) < args.max_samples:
                overlay = draw_overlay(img, objects, fg_th=args.fg_th)
                overlay_items.append((f"{split}/{img_path.name}", overlay))

            for obj_idx, obj in enumerate(objects):
                cls_id = obj["cls"]
                pts = obj["pts"]

                mask = polygon_mask(pts, h, w)
                label_area = int(mask.sum())

                visible_mask = mask & fg
                visible_area = int(visible_mask.sum())

                visible_ratio = visible_area / max(label_area, 1)

                if visible_area > 0:
                    mean_intensity = float(gray[visible_mask].mean())
                else:
                    mean_intensity = 0.0

                comp_count, largest_ratio = connected_component_stats(visible_mask)

                x_min, y_min = pts.min(axis=0)
                x_max, y_max = pts.max(axis=0)

                row = {
                    "split": split,
                    "image": img_path.name,
                    "object_index": obj_idx,
                    "class_id": cls_id,
                    "class_name": CLASS_NAMES.get(cls_id, f"class{cls_id}"),
                    "label_area_px": label_area,
                    "visible_area_px": visible_area,
                    "visible_ratio": visible_ratio,
                    "mean_intensity": mean_intensity,
                    "component_count": comp_count,
                    "largest_component_ratio": largest_ratio,
                    "bbox_w_px": float(x_max - x_min),
                    "bbox_h_px": float(y_max - y_min),
                }
                detail_rows.append(row)

                if visible_ratio < args.low_ratio_th and len(low_items) < args.max_samples:
                    overlay = draw_overlay(img, objects, fg_th=args.fg_th)
                    title = f"{split}/{img_path.name} cls{cls_id} vr={visible_ratio:.2f}"
                    low_items.append((title, overlay))

    # --------------------------------------------------------
    # 상세 CSV 저장
    # --------------------------------------------------------
    detail_csv = out_dir / "visibility_detail.csv"

    if detail_rows:
        fieldnames = list(detail_rows[0].keys())

        with detail_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(detail_rows)

    # --------------------------------------------------------
    # class별 summary 저장
    # --------------------------------------------------------
    summary_rows = []

    for split in ["train", "val"]:
        for cls_id in sorted(CLASS_NAMES.keys()):
            rows = [
                r for r in detail_rows
                if r["split"] == split and int(r["class_id"]) == cls_id
            ]

            ratios = [float(r["visible_ratio"]) for r in rows]
            comps = [float(r["component_count"]) for r in rows]
            largest = [float(r["largest_component_ratio"]) for r in rows]
            areas = [float(r["visible_area_px"]) for r in rows]

            summary_rows.append({
                "split": split,
                "class_id": cls_id,
                "class_name": CLASS_NAMES.get(cls_id, f"class{cls_id}"),
                "objects": len(rows),
                "visible_ratio_mean": float(np.mean(ratios)) if ratios else 0.0,
                "visible_ratio_median": float(np.median(ratios)) if ratios else 0.0,
                "visible_ratio_p10": percentile(ratios, 10),
                "visible_ratio_p90": percentile(ratios, 90),
                "visible_area_mean": float(np.mean(areas)) if areas else 0.0,
                "component_count_mean": float(np.mean(comps)) if comps else 0.0,
                "largest_component_ratio_mean": float(np.mean(largest)) if largest else 0.0,
                "low_visible_count": sum(1 for v in ratios if v < args.low_ratio_th),
            })

    summary_csv = out_dir / "visibility_summary.csv"

    if summary_rows:
        fieldnames = list(summary_rows[0].keys())

        with summary_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)

    # --------------------------------------------------------
    # overlay contact sheet 저장
    # --------------------------------------------------------
    make_contact_sheet(
        overlay_items,
        out_dir / "overlay_samples.jpg",
        tile_w=360,
        tile_h=360,
        cols=4,
    )

    make_contact_sheet(
        low_items,
        out_dir / "low_visible_samples.jpg",
        tile_w=360,
        tile_h=360,
        cols=4,
    )

    # --------------------------------------------------------
    # 콘솔 요약 출력
    # --------------------------------------------------------
    print("============================================")
    print("[DONE] noise dataset 진단 완료")
    print("DATASET:", dataset_root)
    print("OUT    :", out_dir)
    print("DETAIL :", detail_csv)
    print("SUMMARY:", summary_csv)
    print("OVERLAY:", out_dir / "overlay_samples.jpg")
    print("LOW    :", out_dir / "low_visible_samples.jpg")
    print("============================================")

    print("")
    print("[CLASS SUMMARY]")
    for row in summary_rows:
        print(
            f"{row['split']} cls{row['class_id']} {row['class_name']}: "
            f"objects={row['objects']}, "
            f"visible_ratio_mean={row['visible_ratio_mean']:.3f}, "
            f"median={row['visible_ratio_median']:.3f}, "
            f"p10={row['visible_ratio_p10']:.3f}, "
            f"low_count={row['low_visible_count']}, "
            f"components_mean={row['component_count_mean']:.2f}, "
            f"largest_comp_mean={row['largest_component_ratio_mean']:.3f}"
        )


if __name__ == "__main__":
    main()
