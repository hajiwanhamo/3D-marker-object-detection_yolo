import argparse
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import torch
import open3d as o3d

from marker_train_2_test import Model, DEV
from cheack5 import preprocess_xyz


def load_xyz(path: Path) -> np.ndarray:
    x = np.loadtxt(str(path), dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    return x[:, :3].astype(np.float32, copy=False)


def sample_indices(n: int, npts: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)

    if n <= 0:
        raise RuntimeError("empty point cloud")

    if n >= npts:
        return rng.choice(n, npts, replace=False).astype(np.int64)

    extra = rng.choice(n, npts - n, replace=True).astype(np.int64)
    return np.concatenate([np.arange(n, dtype=np.int64), extra], axis=0)


def make_colors(labels: np.ndarray) -> np.ndarray:
    colors = np.full((len(labels), 3), 0.55, dtype=np.float64)

    cmap = {
        -1: np.array([0.55, 0.55, 0.55]),  # removed / uncertain
        0: np.array([1.00, 0.00, 0.00]),  # class0 square ID
        1: np.array([0.00, 0.80, 0.00]),  # class1
        2: np.array([0.00, 0.20, 1.00]),  # class2
        3: np.array([1.00, 0.80, 0.00]),  # class3
    }

    for k, col in cmap.items():
        colors[labels == k] = col

    return colors


def make_pcd(xyz: np.ndarray, labels: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(make_colors(labels))
    return pcd


@torch.no_grad()
def predict_rule4(
    net,
    xyz_raw: np.ndarray,
    npts: int,
    preprocess_mode: str,
    seed: int,
    votes: int,
):
    xyz_in = preprocess_xyz(xyz_raw, preprocess_mode)

    prob_sum = np.zeros((len(xyz_in), 4), dtype=np.float64)
    count = np.zeros((len(xyz_in),), dtype=np.float64)

    for v in range(max(1, votes)):
        sample_idx = sample_indices(len(xyz_in), npts, seed + v * 1009)
        x = torch.from_numpy(xyz_in[sample_idx].astype(np.float32)).unsqueeze(0).to(DEV)

        logits = net(x)
        prob_sample = torch.softmax(logits, dim=-1).squeeze(0).detach().cpu().numpy()

        np.add.at(prob_sum, sample_idx, prob_sample)
        np.add.at(count, sample_idx, 1.0)

    count[count == 0] = 1.0
    prob = prob_sum / count[:, None]

    raw_pred = prob.argmax(axis=1).astype(np.int64)

    return raw_pred, prob


def dbscan_components(xyz: np.ndarray, eps: float, min_points: int) -> np.ndarray:
    if len(xyz) == 0:
        return np.zeros((0,), dtype=np.int64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))

    labels = np.asarray(
        pcd.cluster_dbscan(
            eps=float(eps),
            min_points=int(min_points),
            print_progress=False,
        ),
        dtype=np.int64,
    )

    return labels


def get_thresholds(args) -> List[float]:
    return [
        float(args.class0_th),
        float(args.class1_th),
        float(args.class2_th),
        float(args.class3_th),
    ]


def get_min_pts(args) -> List[int]:
    return [
        int(args.min_pts_class0),
        int(args.min_pts_class1),
        int(args.min_pts_class2),
        int(args.min_pts_class3),
    ]


def build_components_for_class(
    class_id: int,
    xyz_raw: np.ndarray,
    xyz_local: np.ndarray,
    prob: np.ndarray,
    th: float,
    db_eps: float,
    min_pts: int,
    cluster_space: str,
):
    p = prob[:, class_id]
    cand_idx = np.where(p >= th)[0]

    if len(cand_idx) == 0:
        return []

    if cluster_space == "local":
        cluster_xyz = xyz_local[cand_idx]
    else:
        cluster_xyz = xyz_raw[cand_idx]

    labels_db = dbscan_components(cluster_xyz, eps=db_eps, min_points=min_pts)

    comps = []

    for comp_id in sorted(set(labels_db.tolist())):
        if comp_id < 0:
            continue

        local_ids = np.where(labels_db == comp_id)[0]
        global_ids = cand_idx[local_ids]

        if len(global_ids) < min_pts:
            continue

        pc = prob[global_ids, class_id]
        mean_prob = float(pc.mean())
        max_prob = float(pc.max())
        p95_prob = float(np.percentile(pc, 95))

        center = xyz_raw[global_ids].mean(axis=0)
        mn = xyz_raw[global_ids].min(axis=0)
        mx = xyz_raw[global_ids].max(axis=0)
        size = mx - mn

        # count가 너무 큰 영역이 무조건 이기지 않도록 sqrt 사용
        score = float(mean_prob * np.sqrt(len(global_ids)))

        comps.append({
            "class": int(class_id),
            "idx": global_ids,
            "count": int(len(global_ids)),
            "mean_prob": mean_prob,
            "max_prob": max_prob,
            "p95_prob": p95_prob,
            "score": score,
            "center": center,
            "size": size,
        })

    comps.sort(key=lambda d: d["score"], reverse=True)
    return comps


def probability_map_postprocess(
    xyz_raw: np.ndarray,
    prob: np.ndarray,
    preprocess_mode: str,
    args,
):
    xyz_local = preprocess_xyz(xyz_raw, preprocess_mode)

    thresholds = get_thresholds(args)
    min_pts = get_min_pts(args)

    all_components = []
    class_components = {}

    for c in range(4):
        comps = build_components_for_class(
            class_id=c,
            xyz_raw=xyz_raw,
            xyz_local=xyz_local,
            prob=prob,
            th=thresholds[c],
            db_eps=args.db_eps,
            min_pts=min_pts[c],
            cluster_space=args.cluster_space,
        )

        class_components[c] = comps

        if args.keep_one_per_class:
            if len(comps) > 0:
                all_components.append(comps[0])
        else:
            all_components.extend(comps[: max(1, args.max_components_per_class)])

    post = np.full((len(xyz_raw),), -1, dtype=np.int64)
    assign_score = np.full((len(xyz_raw),), -np.inf, dtype=np.float64)

    # component 단위 class 유지
    # overlap이 생기면 component score가 높은 class가 우선
    for comp in sorted(all_components, key=lambda d: d["score"], reverse=True):
        idx = comp["idx"]
        c = comp["class"]
        score = comp["score"]

        update = score > assign_score[idx]
        target_idx = idx[update]

        post[target_idx] = c
        assign_score[target_idx] = score

    return post, all_components, class_components


def print_result(
    name: str,
    raw_pred: np.ndarray,
    post_pred: np.ndarray,
    prob: np.ndarray,
    components: List[Dict[str, Any]],
    class_components: Dict[int, List[Dict[str, Any]]],
    args,
):
    thresholds = get_thresholds(args)

    print("")
    print("=" * 120)
    print(f"[VIEW] {name}")
    print(f"[PREPROCESS] {args.preprocess_mode}")
    print(f"[MODEL] {args.ckpt}")
    print("[POSTPROCESS] class-wise probability map + class-wise DBSCAN")
    print(
        f"[THRESHOLD] class0={thresholds[0]:.3f}, class1={thresholds[1]:.3f}, "
        f"class2={thresholds[2]:.3f}, class3={thresholds[3]:.3f}"
    )
    print(
        f"[DBSCAN] space={args.cluster_space}, db_eps={args.db_eps}, "
        f"min_pts: c0={args.min_pts_class0}, c1={args.min_pts_class1}, "
        f"c2={args.min_pts_class2}, c3={args.min_pts_class3}"
    )

    print("")
    print("[RAW ARGMAX COUNT]")
    for c in range(4):
        print(f"raw class{c}: {int((raw_pred == c).sum())}")

    print("")
    print("[POST COUNT]")
    for c in [-1, 0, 1, 2, 3]:
        print(f"post class{c}: {int((post_pred == c).sum())}")

    print("")
    print("[PROBABILITY SUMMARY]")
    for c in range(4):
        p = prob[:, c]
        print(
            f"class{c}: mean={float(p.mean()):.6f}, "
            f"p95={float(np.percentile(p, 95)):.6f}, "
            f"max={float(p.max()):.6f}, "
            f"over_th={int((p >= thresholds[c]).sum())}"
        )

    print("")
    print("[CLASS COMPONENT CANDIDATES]")
    for c in range(4):
        comps = class_components.get(c, [])
        print(f"class{c}: candidates={len(comps)}")
        for j, comp in enumerate(comps[:5]):
            center = comp["center"]
            size = comp["size"]
            print(
                f"  cand{j}: count={comp['count']} "
                f"mean_prob={comp['mean_prob']:.6f} "
                f"max_prob={comp['max_prob']:.6f} "
                f"score={comp['score']:.6f} "
                f"center=({center[0]:.4f},{center[1]:.4f},{center[2]:.4f}) "
                f"size=({size[0]:.4f},{size[1]:.4f},{size[2]:.4f})"
            )

    print("")
    print("[SELECTED COMPONENTS]")
    for i, comp in enumerate(components):
        center = comp["center"]
        size = comp["size"]
        print(
            f"comp{i}: class={comp['class']} count={comp['count']} "
            f"mean_prob={comp['mean_prob']:.6f} "
            f"score={comp['score']:.6f} "
            f"center=({center[0]:.4f},{center[1]:.4f},{center[2]:.4f}) "
            f"size=({size[0]:.4f},{size[1]:.4f},{size[2]:.4f})"
        )

    print("")
    print("[VIEW MODE] left=raw argmax prediction, right=class-wise probability-map postprocess")
    print("[COLOR] class0(square)=red, class1=green, class2=blue, class3=yellow, removed=-1 gray")
    print("[KEY] N=next, P=prev, Q=quit")
    print("=" * 120)


def save_outputs(out_dir: Path, base: str, raw_pred: np.ndarray, prob: np.ndarray, post_pred: np.ndarray):
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(str(out_dir / f"{base}_raw_pred.npy"), raw_pred.astype(np.int64))
    np.save(str(out_dir / f"{base}_prob.npy"), prob.astype(np.float32))
    np.save(str(out_dir / f"{base}_post_pred.npy"), post_pred.astype(np.int64))


def build_scene(xyz_path: Path, net, args, file_i: int):
    xyz = load_xyz(xyz_path)

    raw_pred, prob = predict_rule4(
        net=net,
        xyz_raw=xyz,
        npts=args.npts,
        preprocess_mode=args.preprocess_mode,
        seed=args.seed + file_i,
        votes=args.votes,
    )

    post_pred, components, class_components = probability_map_postprocess(
        xyz_raw=xyz,
        prob=prob,
        preprocess_mode=args.preprocess_mode,
        args=args,
    )

    if args.out:
        save_outputs(
            out_dir=Path(args.out),
            base=xyz_path.stem,
            raw_pred=raw_pred,
            prob=prob,
            post_pred=post_pred,
        )

    extent = np.linalg.norm(xyz.max(axis=0) - xyz.min(axis=0))
    offset = np.array([extent * 1.35, 0.0, 0.0], dtype=np.float32)

    raw_pcd = make_pcd(xyz - offset, raw_pred)
    post_pcd = make_pcd(xyz + offset, post_pred)

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=max(extent * 0.15, 1e-3))

    return {
        "name": xyz_path.name,
        "pcds": [raw_pcd, post_pcd],
        "frame": frame,
        "center": xyz.mean(axis=0),
        "raw_pred": raw_pred,
        "post_pred": post_pred,
        "prob": prob,
        "components": components,
        "class_components": class_components,
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--marker_root", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="")

    ap.add_argument("--base", default="")
    ap.add_argument("--npts", type=int, default=16384)
    ap.add_argument("--preprocess_mode", type=str, default="auto", choices=["auto", "raw", "normalize", "marker_local"])
    ap.add_argument("--votes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--class0_th", type=float, default=0.45)
    ap.add_argument("--class1_th", type=float, default=0.30)
    ap.add_argument("--class2_th", type=float, default=0.25)
    ap.add_argument("--class3_th", type=float, default=0.45)

    ap.add_argument("--cluster_space", type=str, default="raw", choices=["raw", "local"])
    ap.add_argument("--db_eps", type=float, default=0.05)

    ap.add_argument("--min_pts_class0", type=int, default=3)
    ap.add_argument("--min_pts_class1", type=int, default=3)
    ap.add_argument("--min_pts_class2", type=int, default=3)
    ap.add_argument("--min_pts_class3", type=int, default=5)

    ap.add_argument("--keep_one_per_class", action="store_true")
    ap.add_argument("--max_components_per_class", type=int, default=1)

    ap.add_argument("--point_size", type=float, default=7.0)
    ap.add_argument("--zoom", type=float, default=0.8)

    args = ap.parse_args()

    marker_root = Path(args.marker_root)

    ckpt = torch.load(args.ckpt, map_location=DEV)
    ckpt_args = ckpt.get("args", {})

    if args.preprocess_mode == "auto":
        args.preprocess_mode = ckpt_args.get("preprocess_mode", "marker_local")

    print("=" * 120)
    print("[EVAL2 RULE4 REAL]")
    print(f"[LOAD CKPT] {args.ckpt}")
    print(f"[CKPT preprocess_mode] {ckpt_args.get('preprocess_mode', 'NONE')}")
    print(f"[USE preprocess_mode] {args.preprocess_mode}")
    print("[IMPORTANT] preprocess_xyz is imported from cheack5.py")
    print("[REAL MODE] no GT / no rule_id required")
    print("[POSTPROCESS] class-wise probability map")
    print("=" * 120)

    net = Model(num_classes=4).to(DEV)
    net.load_state_dict(ckpt["model"])
    net.eval()

    if args.base:
        base = args.base
        if base.endswith("_marker"):
            xyz_files = [marker_root / f"{base}.xyz"]
        else:
            xyz_files = [marker_root / f"{base}_marker.xyz"]
    else:
        xyz_files = sorted(marker_root.glob("*_marker.xyz"))

    if not xyz_files:
        raise SystemExit(f"[FAIL] marker xyz 없음: {marker_root}")

    cache = []

    for i, xyz_path in enumerate(xyz_files):
        if not xyz_path.exists():
            print(f"[SKIP] missing: {xyz_path}")
            continue

        try:
            cache.append(build_scene(xyz_path, net, args, i))
        except Exception as e:
            print(f"[SKIP] {xyz_path.name}: {e}")

    if not cache:
        raise SystemExit("[FAIL] 시각화 가능한 파일 없음")

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("real rule4 internal ID eval - probability map")

    opt = vis.get_render_option()
    opt.point_size = float(args.point_size)
    opt.background_color = np.array([1.0, 1.0, 1.0], dtype=np.float64)

    cur = 0

    def show(i):
        vis.clear_geometries()

        for g in cache[i]["pcds"]:
            vis.add_geometry(g)

        vis.add_geometry(cache[i]["frame"])

        ctr = vis.get_view_control()
        ctr.set_lookat(cache[i]["center"].tolist())
        ctr.set_front([0, -1, 0])
        ctr.set_up([0, 0, 1])
        ctr.set_zoom(float(args.zoom))

        vis.poll_events()
        vis.update_renderer()

        print_result(
            cache[i]["name"],
            cache[i]["raw_pred"],
            cache[i]["post_pred"],
            cache[i]["prob"],
            cache[i]["components"],
            cache[i]["class_components"],
            args,
        )

    def next_file(v):
        nonlocal cur
        cur = (cur + 1) % len(cache)
        show(cur)
        return False

    def prev_file(v):
        nonlocal cur
        cur = (cur - 1 + len(cache)) % len(cache)
        show(cur)
        return False

    def quit_view(v):
        v.close()
        return False

    vis.register_key_callback(ord("N"), next_file)
    vis.register_key_callback(ord("P"), prev_file)
    vis.register_key_callback(ord("Q"), quit_view)

    show(cur)
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()