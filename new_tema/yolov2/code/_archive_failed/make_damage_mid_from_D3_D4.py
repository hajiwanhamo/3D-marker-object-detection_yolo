#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_damage_mid_from_D3_D4.py

목적:
- 원본 D3/D4 생성 파라미터가 남아 있지 않은 상태에서
  현재 존재하는 D3(damage_v1)와 D4(damage_D4) 결과물 자체를 기준으로
  중간 상태 YOLO segmentation dataset을 재현한다.

재현 방식:
1. 이미지: D3 이미지와 D4 이미지를 50:50 blend
2. 라벨: class별 D3/D4 polygon을 mask로 변환
3. class별 목표 면적 = (D3 mask 면적 + D4 mask 면적) / 2
4. D3/D4 mask의 intersection을 우선 보존하고, union 영역에서 필요한 만큼 채움
5. 생성된 중간 mask를 YOLO segmentation polygon label로 다시 저장

주의:
- 이 코드는 원본 생성 파라미터의 중간값이 아니다.
- 현재 남아 있는 D3/D4 dataset 결과물 기준의 중간 상태 재현이다.
"""

import argparse
import csv
import shutil
from pathlib import Path

import cv2
import numpy as np


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
VALID_CLASSES = [0, 1, 2, 3]


def imread_unicode(path: Path):
    """한글/일본어/특수문자 경로 대응 이미지 읽기"""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, img):
    """한글/일본어/특수문자 경로 대응 이미지 저장"""
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    if ext not in IMG_EXTS:
        ext = ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise RuntimeError(f"이미지 인코딩 실패: {path}")
    buf.tofile(str(path))


def list_images(img_dir: Path):
    """stem 기준 이미지 path dict 생성"""
    if not img_dir.exists():
        return {}
    out = {}
    for p in sorted(img_dir.iterdir()):
        if p.suffix.lower() in IMG_EXTS:
            out[p.stem] = p
    return out


def polygon_area_px(pts: np.ndarray):
    """pixel polygon 면적"""
    if pts is None or len(pts) < 3:
        return 0.0
    return float(abs(cv2.contourArea(pts.astype(np.float32))))


def read_yolo_seg_as_masks(label_path: Path, width: int, height: int):
    """YOLO segmentation label을 class별 binary mask로 변환"""
    masks = {cid: np.zeros((height, width), dtype=np.uint8) for cid in VALID_CLASSES}

    if not label_path.exists():
        return masks

    text = label_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    for line in text:
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 7:
            continue

        try:
            cls_id = int(float(parts[0]))
            coords = np.array([float(x) for x in parts[1:]], dtype=np.float32)
        except ValueError:
            continue

        if cls_id not in VALID_CLASSES:
            continue

        if len(coords) < 6 or len(coords) % 2 != 0:
            continue

        pts = coords.reshape(-1, 2)
        pts[:, 0] = np.clip(pts[:, 0] * width, 0, width - 1)
        pts[:, 1] = np.clip(pts[:, 1] * height, 0, height - 1)

        if polygon_area_px(pts) < 1.0:
            continue

        pts_i = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(masks[cls_id], [pts_i], 255)

    return masks


def select_top_pixels(mask_bool: np.ndarray, score: np.ndarray, n: int):
    """
    score가 높은 pixel n개 선택.
    tie는 y, x 순서로 고정해서 결과가 항상 동일하게 나오게 한다.
    """
    y, x = np.nonzero(mask_bool)
    if n <= 0 or len(y) == 0:
        return np.zeros_like(mask_bool, dtype=bool)

    if len(y) <= n:
        out = np.zeros_like(mask_bool, dtype=bool)
        out[y, x] = True
        return out

    s = score[y, x]
    order = np.lexsort((x, y, -s))  # -score가 primary key → score 큰 순서
    y_sel = y[order[:n]]
    x_sel = x[order[:n]]

    out = np.zeros_like(mask_bool, dtype=bool)
    out[y_sel, x_sel] = True
    return out


def keep_exact_area(mask_bool: np.ndarray, target_area: int):
    """
    mask 내부에서 중심부 pixel 우선으로 target_area만 남긴다.
    """
    target_area = int(target_area)
    if target_area <= 0:
        return np.zeros_like(mask_bool, dtype=bool)

    area = int(mask_bool.sum())
    if area <= target_area:
        return mask_bool.copy()

    mask_u8 = (mask_bool.astype(np.uint8) * 255)
    dist = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 3)
    return select_top_pixels(mask_bool, dist, target_area)


def make_mid_mask(mask3: np.ndarray, mask4: np.ndarray):
    """
    D3/D4 mask의 중간 상태 생성.

    - 목표 면적은 D3/D4 면적 평균
    - 공통 영역(intersection)을 먼저 유지
    - 부족한 면적은 union-intersection 영역에서 intersection에 가까운 pixel부터 채움
    - intersection이 없으면 union 중심부부터 채움
    """
    a = mask3 > 0
    b = mask4 > 0

    area3 = int(a.sum())
    area4 = int(b.sum())
    target = int(round((area3 + area4) / 2.0))

    if target <= 0:
        return np.zeros_like(mask3, dtype=np.uint8), area3, area4, 0

    union = a | b
    if int(union.sum()) == 0:
        return np.zeros_like(mask3, dtype=np.uint8), area3, area4, 0

    inter = a & b
    out = inter.copy()

    # 이론상 inter 면적은 평균 면적보다 작거나 같지만, 안전 처리
    if int(out.sum()) > target:
        out = keep_exact_area(out, target)
        return (out.astype(np.uint8) * 255), area3, area4, int(out.sum())

    need = target - int(out.sum())
    cand = union & (~out)

    if need >= int(cand.sum()):
        out = union.copy()
        return (out.astype(np.uint8) * 255), area3, area4, int(out.sum())

    if int(out.sum()) > 0:
        # intersection에 가까운 uncertain 영역부터 채움
        base = np.where(out, 0, 255).astype(np.uint8)
        dist_to_inter = cv2.distanceTransform(base, cv2.DIST_L2, 3)
        score = -dist_to_inter
    else:
        # 공통 영역이 없으면 union의 중심부부터 채움
        union_u8 = union.astype(np.uint8) * 255
        score = cv2.distanceTransform(union_u8, cv2.DIST_L2, 3)

    add = select_top_pixels(cand, score, need)
    out[add] = True

    return (out.astype(np.uint8) * 255), area3, area4, int(out.sum())


def mask_to_yolo_lines(cls_id: int, mask: np.ndarray, width: int, height: int, args):
    """binary mask를 YOLO segmentation polygon line으로 변환"""
    lines = []

    if args.label_close_kernel > 0:
        k = int(args.label_close_kernel)
        k = k if k % 2 == 1 else k + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    num, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    if num <= 1:
        return lines

    comp_ids = list(range(1, num))
    comp_ids.sort(key=lambda i: int(stats[i, cv2.CC_STAT_AREA]), reverse=True)
    comp_ids = comp_ids[: args.max_components_per_class]

    for comp_id in comp_ids:
        area = float(stats[comp_id, cv2.CC_STAT_AREA])
        if area < args.min_contour_area:
            continue

        comp = ((labels == comp_id).astype(np.uint8) * 255)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue

        cnt = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(cnt) < args.min_contour_area:
            continue

        eps = float(args.approx_eps_ratio) * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps, True)

        # polygon point 수가 너무 많으면 eps를 조금씩 키워서 단순화
        loop = 0
        while len(approx) > args.max_polygon_points and loop < 20:
            eps *= 1.25
            approx = cv2.approxPolyDP(cnt, eps, True)
            loop += 1

        pts = approx.reshape(-1, 2).astype(np.float32)
        if len(pts) < 3:
            continue

        if polygon_area_px(pts) < args.min_contour_area:
            continue

        pts[:, 0] = np.clip(pts[:, 0] / max(width, 1), 0.0, 1.0)
        pts[:, 1] = np.clip(pts[:, 1] / max(height, 1), 0.0, 1.0)

        values = [str(cls_id)]
        for x, y in pts:
            values.append(f"{float(x):.6f}")
            values.append(f"{float(y):.6f}")

        lines.append(" ".join(values))

    return lines


def write_data_yaml_from_d3(d3_root: Path, out_root: Path):
    """D3 data.yaml의 names block을 유지하면서 path만 새 output으로 변경"""
    src_yaml = d3_root / "data.yaml"
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

    text = f"""path: {out_root.resolve().as_posix()}
train: images/train
val: images/val

{names_block}
"""
    (out_root / "data.yaml").write_text(text, encoding="utf-8")


def draw_qc(img, masks, out_path: Path):
    """생성 label 확인용 overlay 이미지 저장"""
    vis = img.copy()
    overlay = img.copy()

    colors = {
        0: (0, 255, 255),
        1: (0, 180, 0),
        2: (255, 0, 0),
        3: (0, 0, 255),
    }

    for cid in VALID_CLASSES:
        mask = masks[cid]
        overlay[mask > 0] = colors[cid]
        cnts, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, colors[cid], 1)

    out = cv2.addWeighted(overlay, 0.35, vis, 0.65, 0)
    imwrite_unicode(out_path, out)


def process_split(split: str, d3_root: Path, d4_root: Path, out_root: Path, args):
    """train 또는 val split 처리"""
    d3_img_dir = d3_root / "images" / split
    d4_img_dir = d4_root / "images" / split
    d3_lbl_dir = d3_root / "labels" / split
    d4_lbl_dir = d4_root / "labels" / split

    out_img_dir = out_root / "images" / split
    out_lbl_dir = out_root / "labels" / split
    qc_dir = out_root / "qc" / split

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)

    d3_imgs = list_images(d3_img_dir)
    d4_imgs = list_images(d4_img_dir)

    common_stems = sorted(set(d3_imgs.keys()) & set(d4_imgs.keys()))
    if not common_stems:
        raise RuntimeError(f"[ERROR] {split}: D3/D4 공통 이미지 stem 없음")

    missing_d3 = sorted(set(d4_imgs.keys()) - set(d3_imgs.keys()))
    missing_d4 = sorted(set(d3_imgs.keys()) - set(d4_imgs.keys()))

    rows = []
    made = 0

    for idx, stem in enumerate(common_stems):
        img3_path = d3_imgs[stem]
        img4_path = d4_imgs[stem]

        img3 = imread_unicode(img3_path)
        img4 = imread_unicode(img4_path)

        if img3 is None:
            raise RuntimeError(f"[ERROR] D3 이미지 읽기 실패: {img3_path}")
        if img4 is None:
            raise RuntimeError(f"[ERROR] D4 이미지 읽기 실패: {img4_path}")

        h, w = img3.shape[:2]
        if img4.shape[:2] != (h, w):
            img4 = cv2.resize(img4, (w, h), interpolation=cv2.INTER_LINEAR)

        # 이미지 중간 재현
        mid_img = cv2.addWeighted(img3, 0.5, img4, 0.5, 0.0)

        d3_label = d3_lbl_dir / f"{stem}.txt"
        d4_label = d4_lbl_dir / f"{stem}.txt"

        masks3 = read_yolo_seg_as_masks(d3_label, w, h)
        masks4 = read_yolo_seg_as_masks(d4_label, w, h)

        mid_masks = {}
        label_lines = []

        for cid in VALID_CLASSES:
            mid_mask, area3, area4, area_mid = make_mid_mask(masks3[cid], masks4[cid])
            mid_masks[cid] = mid_mask

            lines = mask_to_yolo_lines(cid, mid_mask, w, h, args)
            label_lines.extend(lines)

            rows.append({
                "split": split,
                "stem": stem,
                "class": cid,
                "area_d3": area3,
                "area_d4": area4,
                "area_mid": area_mid,
                "target_area": int(round((area3 + area4) / 2.0)),
                "label_polygons": len(lines),
            })

        out_img = out_img_dir / img3_path.name
        out_lbl = out_lbl_dir / f"{stem}.txt"

        imwrite_unicode(out_img, mid_img)
        out_lbl.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")

        if args.qc_all or idx < args.qc_limit:
            draw_qc(mid_img, mid_masks, qc_dir / f"{stem}_qc.png")

        made += 1

    return {
        "split": split,
        "made": made,
        "missing_d3": len(missing_d3),
        "missing_d4": len(missing_d4),
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--d3-root", required=True, help="D3 dataset root = damage_v1")
    parser.add_argument("--d4-root", required=True, help="D4 dataset root = damage_D4")
    parser.add_argument("--out-root", required=True, help="output dataset root")

    parser.add_argument("--min_contour_area", type=float, default=3.0)
    parser.add_argument("--approx_eps_ratio", type=float, default=0.006)
    parser.add_argument("--max_polygon_points", type=int, default=80)
    parser.add_argument("--max_components_per_class", type=int, default=3)
    parser.add_argument("--label_close_kernel", type=int, default=0)

    parser.add_argument("--qc-limit", type=int, default=80)
    parser.add_argument("--qc-all", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    d3_root = Path(args.d3_root)
    d4_root = Path(args.d4_root)
    out_root = Path(args.out_root)

    if not d3_root.exists():
        raise FileNotFoundError(f"[ERROR] D3 root 없음: {d3_root}")
    if not d4_root.exists():
        raise FileNotFoundError(f"[ERROR] D4 root 없음: {d4_root}")

    if out_root.exists():
        if args.overwrite:
            shutil.rmtree(out_root)
        else:
            raise FileExistsError(f"[ERROR] out-root 이미 존재함. 삭제하려면 --overwrite 사용: {out_root}")

    out_root.mkdir(parents=True, exist_ok=True)

    all_rows = []
    summaries = []

    for split in ["train", "val"]:
        result = process_split(split, d3_root, d4_root, out_root, args)
        summaries.append(result)
        all_rows.extend(result["rows"])

    write_data_yaml_from_d3(d3_root, out_root)

    report_path = out_root / "damage_mid_D3_D4_reproduce_metrics.csv"
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "stem",
                "class",
                "area_d3",
                "area_d4",
                "target_area",
                "area_mid",
                "label_polygons",
            ],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    print("========== DONE ==========")
    print(f"D3 root: {d3_root}")
    print(f"D4 root: {d4_root}")
    print(f"OUT:     {out_root}")
    for s in summaries:
        print(f"{s['split']}: made={s['made']} missing_d3={s['missing_d3']} missing_d4={s['missing_d4']}")
    print(f"data.yaml: {out_root / 'data.yaml'}")
    print(f"metrics:   {report_path}")
    print(f"qc:        {out_root / 'qc'}")
    print("==========================")

if __name__ == "__main__":
    main()
