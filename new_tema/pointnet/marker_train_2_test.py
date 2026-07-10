import argparse
import random
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def square_distance(src, dst):
    return torch.sum((src[:, :, None, :] - dst[:, None, :, :]) ** 2, dim=-1)


def index_points(points, idx):
    device = points.device
    B = points.shape[0]

    if idx.dim() == 2:
        batch_indices = torch.arange(B, dtype=torch.long, device=device).view(B, 1)
        return points[batch_indices, idx]

    if idx.dim() == 3:
        batch_indices = torch.arange(B, dtype=torch.long, device=device).view(B, 1, 1)
        return points[batch_indices, idx]

    raise RuntimeError(f"invalid idx dim: {idx.dim()}")


def farthest_point_sample(xyz, npoint):
    B, N, _ = xyz.shape
    npoint = min(int(npoint), N)

    centroids = torch.zeros(B, npoint, dtype=torch.long, device=xyz.device)
    distance = torch.ones(B, N, device=xyz.device) * 1e10
    farthest = torch.zeros(B, dtype=torch.long, device=xyz.device)
    batch_indices = torch.arange(B, dtype=torch.long, device=xyz.device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=-1)[1]

    return centroids


def knn_point(k, xyz, new_xyz):
    dist = square_distance(new_xyz, xyz)
    k = min(int(k), xyz.shape[1])
    _, group_idx = torch.topk(dist, k=k, dim=-1, largest=False, sorted=False)
    return group_idx


class SA(nn.Module):
    def __init__(self, in_ch, mlp, npoint, k):
        super().__init__()
        self.npoint = int(npoint)
        self.k = int(k)

        layers = []
        last_ch = int(in_ch) + 3
        for out_ch in mlp:
            layers.append(nn.Conv2d(last_ch, out_ch, 1, bias=False))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            last_ch = out_ch
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz, points=None):
        B, N, _ = xyz.shape

        if self.npoint <= 1:
            new_xyz = xyz.mean(dim=1, keepdim=True)
            group_idx = torch.arange(N, device=xyz.device).view(1, 1, N).repeat(B, 1, 1)
        else:
            fps_idx = farthest_point_sample(xyz, min(self.npoint, N))
            new_xyz = index_points(xyz, fps_idx)
            group_idx = knn_point(self.k, xyz, new_xyz)

        grouped_xyz = index_points(xyz, group_idx)
        grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)

        if points is not None:
            grouped_points = index_points(points, group_idx)
            new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
        else:
            new_points = grouped_xyz_norm

        new_points = new_points.permute(0, 3, 2, 1).contiguous()
        new_points = self.mlp(new_points)
        new_points = torch.max(new_points, dim=2)[0]
        new_points = new_points.permute(0, 2, 1).contiguous()

        return new_xyz, new_points


class FP(nn.Module):
    def __init__(self, in_ch, mlp):
        super().__init__()

        layers = []
        last_ch = int(in_ch)
        for out_ch in mlp:
            layers.append(nn.Conv1d(last_ch, out_ch, 1, bias=False))
            layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            last_ch = out_ch
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz1, xyz2, points1, points2):
        B, N, _ = xyz1.shape
        S = xyz2.shape[1]

        if S == 1:
            interpolated = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = torch.topk(dists, k=3, dim=-1, largest=False, sorted=False)
            dists = torch.clamp(dists, min=1e-10)
            inv = 1.0 / dists
            norm = torch.sum(inv, dim=2, keepdim=True)
            weight = inv / norm
            interpolated = torch.sum(index_points(points2, idx) * weight.unsqueeze(-1), dim=2)

        if points1 is not None:
            new_points = torch.cat([points1, interpolated], dim=-1)
        else:
            new_points = interpolated

        new_points = new_points.permute(0, 2, 1).contiguous()
        new_points = self.mlp(new_points)
        new_points = new_points.permute(0, 2, 1).contiguous()

        return new_points


class Model(nn.Module):
    def __init__(self, num_classes=4, k=32, dropout=0.0):
        super().__init__()
        self.num_classes = int(num_classes)

        self.sa1 = SA(0, [64, 64, 128], 512, k)
        self.sa2 = SA(128, [128, 128, 256], 128, k)
        self.sa3 = SA(256, [256, 512, 1024], 1, k)

        self.fp2 = FP(256 + 1024, [256, 256])
        self.fp1 = FP(128 + 256, [256, 128])
        self.fp0 = FP(128 + 3, [128, 128, 128])

        self.c1 = nn.Conv1d(128, 128, 1, bias=False)
        self.b1 = nn.BatchNorm1d(128)
        self.dp = nn.Dropout(float(dropout))
        self.c2 = nn.Conv1d(128, self.num_classes, 1)

    def forward(self, xyz):
        l0_xyz = xyz

        l1_xyz, l1_pts = self.sa1(l0_xyz, None)
        l2_xyz, l2_pts = self.sa2(l1_xyz, l1_pts)
        l3_xyz, l3_pts = self.sa3(l2_xyz, l2_pts)

        f2 = self.fp2(l2_xyz, l3_xyz, l2_pts, l3_pts)
        f1 = self.fp1(l1_xyz, l2_xyz, l1_pts, f2)
        f0 = self.fp0(l0_xyz, l1_xyz, l0_xyz, f1)

        y = f0.permute(0, 2, 1).contiguous()
        y = F.relu(self.b1(self.c1(y)), inplace=True)
        y = self.dp(y)
        y = self.c2(y).permute(0, 2, 1).contiguous()
        return y


def load_xyz(path: Path) -> np.ndarray:
    x = np.loadtxt(str(path), dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    return x[:, :3].astype(np.float32, copy=False)


def normalize_xyz(x: np.ndarray) -> np.ndarray:
    c = x.mean(axis=0, keepdims=True)
    y = x - c
    s = np.linalg.norm(y, axis=1).max()
    if not np.isfinite(s) or s < 1e-8:
        s = 1.0
    return (y / s).astype(np.float32, copy=False)


def marker_local_align_xyz(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    c = x.mean(axis=0, keepdims=True)
    y = x - c

    xy = y[:, :2]
    cov = np.cov(xy.T)

    if not np.all(np.isfinite(cov)):
        return normalize_xyz(x)

    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    r2 = eigvecs[:, order]

    if np.linalg.det(r2) < 0:
        r2[:, 1] *= -1.0

    local_xy = xy @ r2

    skew_x = np.mean(local_xy[:, 0] ** 3)
    skew_y = np.mean(local_xy[:, 1] ** 3)

    if skew_x < 0:
        local_xy[:, 0] *= -1.0

    if skew_y < 0:
        local_xy[:, 1] *= -1.0

    local = np.zeros_like(y, dtype=np.float32)
    local[:, :2] = local_xy.astype(np.float32)
    local[:, 2] = y[:, 2]

    s = np.linalg.norm(local, axis=1).max()
    if not np.isfinite(s) or s < 1e-8:
        s = 1.0

    return (local / s).astype(np.float32, copy=False)


def preprocess_xyz(x: np.ndarray, mode: str) -> np.ndarray:
    if mode == "raw":
        return x.astype(np.float32, copy=False)
    if mode == "normalize":
        return normalize_xyz(x)
    if mode == "marker_local":
        return marker_local_align_xyz(x)
    raise ValueError(f"unknown preprocess_mode: {mode}")


def parse_roots(s: str) -> List[Path]:
    roots = []
    for part in s.split(";"):
        part = part.strip().strip('"')
        if part:
            roots.append(Path(part))
    if not roots:
        raise ValueError("label_roots is empty")
    return roots


def base_from_marker_stem(stem: str) -> str:
    if stem.endswith("_marker"):
        return stem[:-7]
    return stem


def find_idx_file(marker_dir: Path, stem: str) -> Optional[Path]:
    p1 = marker_dir / f"{stem}_idx.npy"
    if p1.exists():
        return p1

    p2 = marker_dir / f"{stem}.idx.npy"
    if p2.exists():
        return p2

    return None


def find_rule_file(roots: List[Path], base: str) -> Optional[Path]:
    for root in roots:
        p = root / f"{base}.rule_id.npy"
        if p.exists():
            return p
    return None


def full_context_indices(n: int, npts: int) -> np.ndarray:
    if n <= 0:
        raise RuntimeError("empty point cloud")

    if n >= npts:
        return np.random.choice(n, npts, replace=False).astype(np.int64)

    extra = np.random.choice(n, npts - n, replace=True).astype(np.int64)
    return np.concatenate([np.arange(n, dtype=np.int64), extra], axis=0)


class RuleIDDataset(Dataset):
    def __init__(
        self,
        marker_root: str,
        label_roots: str,
        selected_files: Optional[List[Path]] = None,
        npts: int = 16384,
        preprocess_mode: str = "marker_local",
        cache_mode: str = "ram",
        min_valid_pts: int = 8,
        min_class0_pts: int = 1,
    ):
        self.marker_root = Path(marker_root)
        self.label_roots = parse_roots(label_roots)
        self.npts = int(npts)
        self.preprocess_mode = str(preprocess_mode)
        self.cache_mode = str(cache_mode).lower()
        self.min_valid_pts = int(min_valid_pts)
        self.min_class0_pts = int(min_class0_pts)

        if not self.marker_root.is_dir():
            raise FileNotFoundError(f"marker_root not found: {self.marker_root}")

        files = selected_files if selected_files is not None else sorted(self.marker_root.glob("*_marker.xyz"))

        raw_items = []
        for xyz_path in files:
            stem = xyz_path.stem
            base = base_from_marker_stem(stem)

            idx_path = find_idx_file(self.marker_root, stem)
            if idx_path is None:
                continue

            rule_path = find_rule_file(self.label_roots, base)
            if rule_path is None:
                continue

            raw_items.append({
                "xyz": xyz_path,
                "idx": idx_path,
                "rule": rule_path,
                "stem": stem,
                "base": base,
            })

        if not raw_items:
            raise RuntimeError(f"dataset empty: marker_root={self.marker_root}")

        self.items: List[Dict[str, Any]] = []
        tmp_cache: List[Dict[str, np.ndarray]] = []
        dropped = 0

        for item in raw_items:
            d = self._load_item(item)
            valid_n = int((d["label"] != -1).sum())
            class0_n = int((d["label"] == 0).sum())

            if valid_n < self.min_valid_pts or class0_n < self.min_class0_pts:
                dropped += 1
                continue

            self.items.append(item)
            tmp_cache.append(d)

        if not self.items:
            raise RuntimeError("all samples dropped")

        if self.cache_mode == "ram":
            self.cache = tmp_cache
        else:
            self.cache = [None] * len(self.items)

        counts = np.zeros((4,), dtype=np.int64)
        ignore = 0

        for i in range(len(self.items)):
            d = self.cache[i] if self.cache_mode == "ram" else self._load_item(self.items[i])
            y = d["label"]
            for c in range(4):
                counts[c] += int((y == c).sum())
            ignore += int((y == -1).sum())

        print(
            f"[RuleIDDataset] samples={len(self.items)} dropped={dropped} "
            f"marker_root={self.marker_root} preprocess_mode={self.preprocess_mode} "
            f"sampling=full_context cache_mode={self.cache_mode}"
        )
        print(
            f"[LabelCount] class0_square={counts[0]} class1={counts[1]} "
            f"class2={counts[2]} class3={counts[3]} ignore={ignore}"
        )

    def __len__(self):
        return len(self.items)

    def _load_item(self, item: Dict[str, Any]) -> Dict[str, np.ndarray]:
        xyz = load_xyz(item["xyz"])
        idx = np.load(str(item["idx"]), allow_pickle=False).astype(np.int64).reshape(-1)
        rule_full = np.load(str(item["rule"]), allow_pickle=False).astype(np.int64).reshape(-1)

        if len(xyz) != len(idx):
            raise RuntimeError(f"xyz/idx length mismatch: {item['xyz'].name}")

        if idx.size > 0 and (idx.min() < 0 or idx.max() >= len(rule_full)):
            raise RuntimeError(f"idx range error: {item['idx'].name}")

        label = rule_full[idx]

        valid = (label == -1) | ((label >= 0) & (label <= 3))
        if not np.all(valid):
            bad = np.unique(label[~valid])
            raise RuntimeError(f"invalid rule label in {item['rule'].name}: {bad}")

        xyz = np.nan_to_num(xyz.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
        xyz_in = preprocess_xyz(xyz, self.preprocess_mode)

        return {
            "xyz": xyz_in.astype(np.float32, copy=False),
            "label": label.astype(np.int64, copy=False),
        }

    def __getitem__(self, idx: int):
        d = self.cache[idx] if self.cache_mode == "ram" else self._load_item(self.items[idx])
        choice = full_context_indices(len(d["label"]), self.npts)

        xyz = d["xyz"][choice]
        y = d["label"][choice]

        return (
            torch.from_numpy(xyz.astype(np.float32, copy=False)),
            torch.from_numpy(y.astype(np.int64, copy=False)),
        )


def split_files(marker_root: str, val_ratio: float, seed: int):
    files = sorted(Path(marker_root).glob("*_marker.xyz"))

    if not files:
        raise RuntimeError(f"no *_marker.xyz files: {marker_root}")

    if len(files) <= 1:
        return files, files

    rng = np.random.default_rng(seed)
    order = np.arange(len(files))
    rng.shuffle(order)

    n_val = int(round(len(files) * float(val_ratio)))
    n_val = max(1, min(n_val, len(files) - 1))

    val_idx = set(order[:n_val].tolist())
    train_files = [f for i, f in enumerate(files) if i not in val_idx]
    val_files = [f for i, f in enumerate(files) if i in val_idx]

    return train_files, val_files


def make_loader(ds, batch_size: int, shuffle: bool, workers: int):
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": workers,
        "drop_last": False,
        "pin_memory": DEV.type == "cuda",
    }

    if workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2

    return DataLoader(ds, **kwargs)


def amp_on(use_amp: bool) -> bool:
    return bool(use_amp) and DEV.type == "cuda"


def autocast_ctx(use_amp: bool):
    return torch.cuda.amp.autocast(enabled=amp_on(use_amp))


def make_scaler(use_amp: bool):
    return torch.cuda.amp.GradScaler(enabled=amp_on(use_amp))


def update_confusion(conf: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray):
    for t, p in zip(y_true, y_pred):
        if t == -1:
            continue
        if 0 <= int(t) < 4 and 0 <= int(p) < 4:
            conf[int(t), int(p)] += 1


def metrics_from_conf(conf: np.ndarray):
    total = int(conf.sum())
    acc = float(np.trace(conf)) / float(max(total, 1))

    precision = []
    recall = []
    f1 = []

    for c in range(4):
        tp = float(conf[c, c])
        fp = float(conf[:, c].sum() - conf[c, c])
        fn = float(conf[c, :].sum() - conf[c, c])

        p = tp / max(tp + fp, 1e-8)
        r = tp / max(tp + fn, 1e-8)
        f = 2.0 * p * r / max(p + r, 1e-8)

        precision.append(p)
        recall.append(r)
        f1.append(f)

    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_f1": float(np.mean(f1)),
    }


def forward_loss_pred(net, batch, crit, use_amp: bool):
    xyz, y = batch
    xyz = xyz.to(DEV, non_blocking=True)
    y = y.to(DEV, non_blocking=True)

    with autocast_ctx(use_amp):
        logits = net(xyz)
        loss = crit(logits.reshape(-1, 4), y.reshape(-1))

    pred = logits.reshape(-1, 4).argmax(dim=1)

    return loss, y.reshape(-1).detach().cpu().numpy(), pred.detach().cpu().numpy()


def train_epoch(net, loader, opt, crit, scaler, use_amp: bool):
    net.train()

    total_loss = 0.0
    total_count = 0
    conf = np.zeros((4, 4), dtype=np.int64)

    for batch in loader:
        opt.zero_grad(set_to_none=True)

        loss, y_true, y_pred = forward_loss_pred(net, batch, crit, use_amp)

        if amp_on(use_amp):
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()

        valid_count = int((y_true != -1).sum())
        total_loss += float(loss.item()) * max(valid_count, 1)
        total_count += max(valid_count, 1)
        update_confusion(conf, y_true, y_pred)

    m = metrics_from_conf(conf)
    avg_loss = total_loss / max(total_count, 1)

    return avg_loss, m


@torch.no_grad()
def eval_epoch(net, loader, crit, use_amp: bool):
    net.eval()

    total_loss = 0.0
    total_count = 0
    conf = np.zeros((4, 4), dtype=np.int64)

    for batch in loader:
        loss, y_true, y_pred = forward_loss_pred(net, batch, crit, use_amp)

        valid_count = int((y_true != -1).sum())
        total_loss += float(loss.item()) * max(valid_count, 1)
        total_count += max(valid_count, 1)
        update_confusion(conf, y_true, y_pred)

    m = metrics_from_conf(conf)
    avg_loss = total_loss / max(total_count, 1)

    return avg_loss, m, total_count, conf


def select_score(metric_name: str, m: Dict[str, Any]):
    if metric_name == "class0_f1":
        return float(m["f1"][0])
    if metric_name == "macro_f1":
        return float(m["macro_f1"])
    if metric_name == "acc":
        return float(m["acc"])
    raise ValueError(f"unknown best_metric: {metric_name}")


def main(a):
    set_seed(a.seed)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    train_files, val_files = split_files(a.marker_root, a.val_ratio, a.seed)

    train_ds = RuleIDDataset(
        marker_root=a.marker_root,
        label_roots=a.label_roots,
        selected_files=train_files,
        npts=a.npts,
        preprocess_mode=a.preprocess_mode,
        cache_mode=a.cache_mode,
        min_valid_pts=a.min_valid_pts,
        min_class0_pts=a.min_class0_pts,
    )

    val_ds = RuleIDDataset(
        marker_root=a.marker_root,
        label_roots=a.label_roots,
        selected_files=val_files,
        npts=a.npts,
        preprocess_mode=a.preprocess_mode,
        cache_mode=a.cache_mode,
        min_valid_pts=a.min_valid_pts,
        min_class0_pts=a.min_class0_pts,
    )

    train_loader = make_loader(train_ds, a.batch, True, a.workers)
    val_loader = make_loader(val_ds, a.batch, False, a.workers)

    net = Model(num_classes=4, dropout=a.dropout).to(DEV)

    weights = torch.tensor(
        [
            float(a.class0_weight),
            float(a.class1_weight),
            float(a.class2_weight),
            float(a.class3_weight),
        ],
        dtype=torch.float32,
        device=DEV,
    )

    print("[ClassMeaning] class0=square_ID class1=clockwise_1 class2=clockwise_2 class3=clockwise_3 ignore=-1")
    print(f"[Preprocess] mode={a.preprocess_mode}")
    print("[Sampling] full_context")
    print(f"[Dropout] {a.dropout}")
    print(f"[BestMetric] {a.best_metric}")
    print(
        f"[LossWeight] class0={weights[0].item()} class1={weights[1].item()} "
        f"class2={weights[2].item()} class3={weights[3].item()}"
    )

    crit = nn.CrossEntropyLoss(weight=weights, ignore_index=-1)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr, weight_decay=a.wd)
    scaler = make_scaler(a.amp)

    best_score = -1.0
    best_path = out / "best_id_seg_rule4_fullcontext.pth"
    last_path = out / "last_id_seg_rule4_fullcontext.pth"

    for epoch in range(1, a.epochs + 1):
        train_loss, train_m = train_epoch(net, train_loader, opt, crit, scaler, a.amp)

        do_eval = (epoch == 1) or (epoch == a.epochs) or (a.eval_every > 0 and epoch % a.eval_every == 0)

        if do_eval:
            val_loss, val_m, val_count, val_conf = eval_epoch(net, val_loader, crit, a.amp)
            score = select_score(a.best_metric, val_m)

            print(
                f"[{epoch}] "
                f"train={train_loss:.6f} train_acc={train_m['acc']:.6f} train_macro_f1={train_m['macro_f1']:.6f} "
                f"val={val_loss:.6f} val_acc={val_m['acc']:.6f} "
                f"c0_f1={val_m['f1'][0]:.6f} c1_f1={val_m['f1'][1]:.6f} "
                f"c2_f1={val_m['f1'][2]:.6f} c3_f1={val_m['f1'][3]:.6f} "
                f"macro_f1={val_m['macro_f1']:.6f} val_count={val_count}"
            )
        else:
            val_loss = float("inf")
            val_m = {"acc": 0.0, "precision": [0.0] * 4, "recall": [0.0] * 4, "f1": [0.0] * 4, "macro_f1": 0.0}
            val_conf = np.zeros((4, 4), dtype=np.int64)
            val_count = 0
            score = -1.0
            print(
                f"[{epoch}] train={train_loss:.6f} "
                f"train_acc={train_m['acc']:.6f} train_macro_f1={train_m['macro_f1']:.6f} val=skip"
            )

        ckpt = {
            "model": net.state_dict(),
            "args": vars(a),
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_acc": float(train_m["acc"]),
            "train_macro_f1": float(train_m["macro_f1"]),
            "val_loss": float(val_loss),
            "val_acc": float(val_m["acc"]),
            "class0_precision": float(val_m["precision"][0]),
            "class0_recall": float(val_m["recall"][0]),
            "class0_f1": float(val_m["f1"][0]),
            "macro_f1": float(val_m["macro_f1"]),
            "val_confusion": val_conf.tolist(),
            "best_metric": a.best_metric,
            "class_rule": "class0=square_ID, class1~3=clockwise_ID, ignore=-1",
            "label_rule": "label_crop = rule_id[marker_idx]",
            "sampling": "full_context",
            "preprocess_mode": a.preprocess_mode,
            "dropout": float(a.dropout),
        }

        torch.save(ckpt, last_path)

        if do_eval and val_count > 0 and score > best_score:
            best_score = score
            torch.save(ckpt, best_path)
            print(f"[save] best {a.best_metric}={best_score:.6f} -> {best_path}")

    print(f"[done] best_{a.best_metric}={best_score:.6f} saved={best_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("--marker_root", required=True)
    ap.add_argument("--label_roots", required=True)
    ap.add_argument("--out", required=True)

    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--npts", type=int, default=16384)
    ap.add_argument("--val_ratio", type=float, default=0.2)

    ap.add_argument("--preprocess_mode", type=str, default="marker_local", choices=["raw", "normalize", "marker_local"])

    ap.add_argument("--min_valid_pts", type=int, default=8)
    ap.add_argument("--min_class0_pts", type=int, default=1)

    ap.add_argument("--class0_weight", type=float, default=3.0)
    ap.add_argument("--class1_weight", type=float, default=1.2)
    ap.add_argument("--class2_weight", type=float, default=1.2)
    ap.add_argument("--class3_weight", type=float, default=1.2)

    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--best_metric", type=str, default="macro_f1", choices=["class0_f1", "macro_f1", "acc"])

    ap.add_argument("--cache_mode", type=str, default="ram", choices=["ram", "none"])
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=1)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()
    main(args)