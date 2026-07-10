# # marker_train_1_test.py
# import os
# import csv
# import argparse
# import time
# from pathlib import Path

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader

# DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# if DEV.type == "cuda":
#     torch.backends.cudnn.benchmark = True
#     try:
#         torch.set_float32_matmul_precision("high")
#     except Exception:
#         pass


# # -------------------------- PointNet++ SSG+FP 최소 백본 --------------------------
# @torch.no_grad()
# def fps(xyz, M):
#     B, N, _ = xyz.shape
#     M = int(max(1, min(M, N)))

#     idx = torch.zeros(B, M, dtype=torch.long, device=xyz.device)
#     far = torch.randint(0, N, (B,), device=xyz.device)
#     dist = torch.full((B, N), 1e10, device=xyz.device)
#     b = torch.arange(B, device=xyz.device)

#     for i in range(M):
#         idx[:, i] = far
#         c = xyz[b, far, :].unsqueeze(1)
#         d = ((xyz - c) ** 2).sum(-1)
#         dist = torch.minimum(dist, d)
#         far = torch.argmax(dist, dim=1)

#     return idx


# @torch.no_grad()
# def knn(src, q, k=16):
#     D = torch.cdist(q, src, p=2)
#     _, ind = torch.topk(D, k, dim=-1, largest=False)
#     return ind


# def gather(P, idx):
#     B, N, C = P.shape
#     idx = idx.long()

#     if idx.ndim == 2:
#         B2, M = idx.shape
#         if B2 != B:
#             raise RuntimeError(f"batch mismatch: P={B}, idx={B2}")
#         bidx = torch.arange(B, device=P.device).view(B, 1).expand(B, M)
#         return P[bidx, idx]

#     if idx.ndim == 3:
#         B2, M, K = idx.shape
#         if B2 != B:
#             raise RuntimeError(f"batch mismatch: P={B}, idx={B2}")
#         bidx = torch.arange(B, device=P.device).view(B, 1, 1).expand(B, M, K)
#         return P[bidx, idx]

#     raise RuntimeError("idx ndim must be 2 or 3")


# class SA(nn.Module):
#     def __init__(self, in_ch, mlp, M, k):
#         super().__init__()
#         self.M = int(M)
#         self.k = int(k)

#         ch = in_ch + 3
#         layers = []

#         for c in mlp:
#             layers += [
#                 nn.Conv2d(ch, c, 1, bias=False),
#                 nn.BatchNorm2d(c),
#                 nn.ReLU(True),
#             ]
#             ch = c

#         self.mlp = nn.Sequential(*layers)

#     def forward(self, xyz, feat):
#         idx = fps(xyz, self.M)
#         cen = gather(xyz, idx)
#         nbr = knn(xyz, cen, self.k)

#         gxyz = gather(xyz, nbr) - cen.unsqueeze(2)

#         if feat is None:
#             g = gxyz
#         else:
#             g = torch.cat([gather(feat, nbr), gxyz], dim=-1)

#         x = self.mlp(g.permute(0, 3, 1, 2).contiguous())
#         x = x.max(dim=-1)[0]
#         x = x.permute(0, 2, 1).contiguous()

#         return cen, x


# class FP(nn.Module):
#     def __init__(self, in_ch, mlp):
#         super().__init__()

#         ch = in_ch
#         layers = []

#         for c in mlp:
#             layers += [
#                 nn.Conv1d(ch, c, 1, bias=False),
#                 nn.BatchNorm1d(c),
#                 nn.ReLU(True),
#             ]
#             ch = c

#         self.mlp = nn.Sequential(*layers)

#     def forward(self, xyz_l, xyz_h, f_l, f_h):
#         Nh = xyz_h.shape[1]
#         k = 3 if Nh >= 3 else Nh

#         if k <= 0:
#             raise RuntimeError("FP: high-resolution set has zero points")

#         d = torch.cdist(xyz_l, xyz_h, p=2) + 1e-8
#         w, idx = torch.topk(d, k, dim=-1, largest=False)

#         w = 1.0 / (w + 1e-8)
#         w = w / w.sum(-1, keepdim=True)

#         f = gather(f_h, idx)
#         f = (f * w.unsqueeze(-1)).sum(2)

#         if f_l is not None:
#             f = torch.cat([f, f_l], dim=-1)

#         x = f.permute(0, 2, 1).contiguous()
#         x = self.mlp(x)
#         x = x.permute(0, 2, 1).contiguous()

#         return x


# class Model(nn.Module):
#     """입력 [B,N,3] 또는 [B,3,N] → 출력 [B,N,1] 로짓"""

#     def __init__(self, k=16):
#         super().__init__()

#         self.sa1 = SA(0, [64, 64, 128], 512, k)
#         self.sa2 = SA(128, [128, 128, 256], 128, k)
#         self.sa3 = SA(256, [256, 512, 1024], 1, k)

#         self.fp2 = FP(256 + 1024, [256, 256])
#         self.fp1 = FP(128 + 256, [256, 128])
#         self.fp0 = FP(128 + 3, [128, 128, 128])

#         self.c1 = nn.Conv1d(128, 128, 1, bias=False)
#         self.b1 = nn.BatchNorm1d(128)
#         self.dp = nn.Dropout(0.5)
#         self.c2 = nn.Conv1d(128, 1, 1)

#     def forward(self, x):
#         if x.ndim != 3:
#             raise RuntimeError(f"expected [B,N,3] or [B,3,N], got {tuple(x.shape)}")

#         if x.size(-1) == 3:
#             l0_xyz = x
#         elif x.size(1) == 3:
#             l0_xyz = x.permute(0, 2, 1).contiguous()
#         else:
#             raise RuntimeError(f"last or 2nd dim must be 3, got {tuple(x.shape)}")

#         l0_pts = None

#         l1_xyz, l1_pts = self.sa1(l0_xyz, l0_pts)
#         l2_xyz, l2_pts = self.sa2(l1_xyz, l1_pts)
#         l3_xyz, l3_pts = self.sa3(l2_xyz, l2_pts)

#         f2 = self.fp2(l2_xyz, l3_xyz, l2_pts, l3_pts)
#         f1 = self.fp1(l1_xyz, l2_xyz, l1_pts, f2)
#         f0 = self.fp0(l0_xyz, l1_xyz, l0_xyz, f1)

#         y = f0.permute(0, 2, 1).contiguous()
#         y = self.b1(self.c1(y))
#         y = F.relu(y, True)
#         y = self.dp(y)
#         y = self.c2(y).permute(0, 2, 1).contiguous()

#         return y


# # -------------------------- Dataset --------------------------
# def _normalize_xyz(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
#     C = X.mean(0, keepdims=True)
#     Xc = X - C

#     s = np.linalg.norm(Xc, axis=1).max()
#     s = float(s) if np.isfinite(s) and s > eps else 1.0

#     return Xc / s


# def _load_xyz_mask(xfp: Path, mfp: Path):
#     X = np.loadtxt(str(xfp), dtype=np.float32)

#     if X.ndim == 1:
#         X = X.reshape(-1, 3)

#     X = X[:, :3].astype(np.float32, copy=False)

#     M = np.load(str(mfp)).astype(np.float32)

#     if M.ndim != 1:
#         M = M.reshape(-1)

#     if X.shape[0] != M.shape[0]:
#         raise RuntimeError(
#             f"[데이터오류] point/mask 개수 불일치: {xfp.name}, "
#             f"X={X.shape[0]}, M={M.shape[0]}"
#         )

#     if X.shape[0] <= 0:
#         raise RuntimeError(f"[데이터오류] 포인트가 비어 있음: {xfp}")

#     return X, M


# class SegDataset(Dataset):
#     def __init__(
#         self,
#         root: str,
#         npts: int = 16384,
#         normalize: bool = True,
#         cache_mode: str = "ram",
#     ):
#         base = Path(root)
#         pts_dir = base / "points"
#         lab_csv = base / "labels.csv"

#         if not pts_dir.is_dir():
#             raise SystemExit(f"[검증실패] points 폴더 없음: {pts_dir}")

#         if not lab_csv.exists():
#             raise SystemExit(f"[검증실패] labels.csv 없음: {lab_csv}")

#         self.npts = int(npts)
#         self.normalize = bool(normalize)
#         self.cache_mode = str(cache_mode)

#         items = []

#         with open(lab_csv, "r", encoding="utf-8") as f:
#             rdr = csv.DictReader(f)

#             if "filename" not in (rdr.fieldnames or []):
#                 raise SystemExit("[검증실패] labels.csv에 'filename' 필요")

#             for r in rdr:
#                 stem = Path(r["filename"]).stem
#                 xfp = pts_dir / f"{stem}.xyz"
#                 mfp = pts_dir / f"{stem}.mask.npy"

#                 if xfp.exists() and mfp.exists():
#                     items.append((xfp, mfp))

#         if not items:
#             raise SystemExit("[검증실패] xyz+mask 쌍이 없음")

#         self.items = items
#         self.cache = None
#         self.pos_total = 0
#         self.neg_total = 0
#         self.mask_ratio = 0.0

#         if self.cache_mode == "ram":
#             t0 = time.time()
#             self.cache = []

#             for xfp, mfp in self.items:
#                 X, M = _load_xyz_mask(xfp, mfp)
#                 self.cache.append((X, M))

#                 pos = int((M > 0.5).sum())
#                 total = int(M.size)
#                 self.pos_total += pos
#                 self.neg_total += total - pos

#             total_all = self.pos_total + self.neg_total
#             self.mask_ratio = self.pos_total / max(1, total_all)

#             print(
#                 f"[cache] RAM cache loaded: {len(self.cache)} samples, "
#                 f"{time.time() - t0:.1f}s"
#             )

#         else:
#             self._compute_mask_stats_from_files()

#     def _compute_mask_stats_from_files(self):
#         pos_total = 0
#         neg_total = 0

#         for _, mfp in self.items:
#             M = np.load(str(mfp)).astype(np.float32)

#             if M.ndim != 1:
#                 M = M.reshape(-1)

#             pos = int((M > 0.5).sum())
#             total = int(M.size)

#             pos_total += pos
#             neg_total += total - pos

#         self.pos_total = pos_total
#         self.neg_total = neg_total

#         total_all = self.pos_total + self.neg_total
#         self.mask_ratio = self.pos_total / max(1, total_all)

#     def __len__(self):
#         return len(self.items)

#     def _sample(self, X, M):
#         N = X.shape[0]

#         if N >= self.npts:
#             idx = np.random.choice(N, self.npts, replace=False)
#         else:
#             rep = (self.npts + N - 1) // N
#             extra = np.random.choice(N, rep * N - N, replace=True)
#             idx = np.concatenate([np.arange(N), extra])[:self.npts]

#         Xs = X[idx, :]
#         Ms = M[idx][:, None]

#         if self.normalize:
#             Xs = _normalize_xyz(Xs).astype(np.float32, copy=False)

#         return torch.from_numpy(Xs), torch.from_numpy(Ms)

#     def __getitem__(self, i):
#         if self.cache is not None:
#             X, M = self.cache[i]
#         else:
#             xfp, mfp = self.items[i]
#             X, M = _load_xyz_mask(xfp, mfp)

#         return self._sample(X, M)


# # -------------------------- Train --------------------------
# def preflight(args):
#     base = Path(args.data)

#     if not (base / "points").is_dir():
#         raise SystemExit(f"[검증실패] points 없음: {base / 'points'}")

#     if not (base / "labels.csv").exists():
#         raise SystemExit(f"[검증실패] labels.csv 없음: {base / 'labels.csv'}")

#     net = Model(k=args.k).to(DEV).eval()

#     with torch.inference_mode():
#         dummy = torch.zeros((2, args.npts, 3), dtype=torch.float32, device=DEV)
#         out = net(dummy)

#         if not (
#             isinstance(out, torch.Tensor)
#             and out.ndim == 3
#             and out.size() == torch.Size([2, args.npts, 1])
#         ):
#             raise SystemExit(f"[검증실패] 출력 [B,N,1] 아님: {tuple(out.shape)}")

#     print("[검증통과] 경로/모델 OK")


# def build_dataloader(ds, args):
#     kwargs = dict(
#         dataset=ds,
#         batch_size=args.batch,
#         shuffle=True,
#         num_workers=args.workers,
#         pin_memory=(DEV.type == "cuda"),
#         drop_last=False,
#     )

#     if args.workers > 0:
#         kwargs["persistent_workers"] = True
#         kwargs["prefetch_factor"] = args.prefetch_factor

#     return DataLoader(**kwargs)


# def resolve_pos_weight(args, ds):
#     raw = str(args.pos_weight).strip().lower()

#     auto_value = ds.neg_total / max(1, ds.pos_total)

#     if raw == "auto":
#         value = auto_value

#         if args.max_pos_weight > 0:
#             value = min(value, float(args.max_pos_weight))

#         return float(value), float(auto_value)

#     try:
#         value = float(raw)
#     except ValueError:
#         raise SystemExit("[검증실패] --pos_weight는 숫자 또는 auto만 가능")

#     if value <= 0:
#         raise SystemExit("[검증실패] --pos_weight는 0보다 커야 함")

#     return float(value), float(auto_value)


# def make_ckpt(net, opt, args, epoch, loss, best, pos_weight_value, auto_pos_weight):
#     return {
#         "model": net.state_dict(),
#         "opt": opt.state_dict(),
#         "meta": {
#             "epoch": int(epoch),
#             "loss": float(loss),
#             "best": float(best),
#             "npts": int(args.npts),
#             "task": "seg",
#             "normalize": bool(not args.no_norm),
#             "k": int(args.k),
#             "cache_mode": str(args.cache_mode),
#             "amp": bool(args.amp and DEV.type == "cuda"),
#             "pos_weight": float(pos_weight_value),
#             "auto_pos_weight": float(auto_pos_weight),
#         },
#     }


# def train(args):
#     ds = SegDataset(
#         args.data,
#         npts=args.npts,
#         normalize=(not args.no_norm),
#         cache_mode=args.cache_mode,
#     )

#     pos_weight_value, auto_pos_weight = resolve_pos_weight(args, ds)

#     print(
#         f"[mask] pos={ds.pos_total}  neg={ds.neg_total}  "
#         f"pos_ratio={ds.mask_ratio:.6f}  "
#         f"auto_pos_weight={auto_pos_weight:.3f}  "
#         f"used_pos_weight={pos_weight_value:.3f}"
#     )

#     dl = build_dataloader(ds, args)

#     print(
#         f"samples={len(ds)}  "
#         f"npts={args.npts}  "
#         f"batch={args.batch}  "
#         f"device={DEV.type}  "
#         f"normalize={not args.no_norm}  "
#         f"cache_mode={args.cache_mode}  "
#         f"workers={args.workers}  "
#         f"save_every={args.save_every}  "
#         f"amp={bool(args.amp and DEV.type == 'cuda')}"
#     )

#     net = Model(k=args.k).to(DEV)

#     pos_weight_tensor = torch.tensor([pos_weight_value], dtype=torch.float32, device=DEV)
#     crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

#     opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
#     sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))

#     use_amp = bool(args.amp and DEV.type == "cuda")
#     scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

#     outdir = Path(args.out)
#     outdir.mkdir(parents=True, exist_ok=True)

#     best = float("inf")
#     last_ckpt = None

#     for e in range(1, args.epochs + 1):
#         net.train()

#         t0 = time.time()
#         Lsum = 0.0
#         n = 0

#         for pts, m in dl:
#             pts = pts.to(DEV, non_blocking=True)
#             m = m.to(DEV, non_blocking=True)

#             opt.zero_grad(set_to_none=True)

#             if use_amp:
#                 with torch.cuda.amp.autocast(enabled=True):
#                     logits = net(pts)
#                     loss = crit(logits, m)

#                 scaler.scale(loss).backward()
#                 scaler.step(opt)
#                 scaler.update()

#             else:
#                 logits = net(pts)
#                 loss = crit(logits, m)
#                 loss.backward()
#                 opt.step()

#             bs = pts.size(0)
#             Lsum += float(loss.detach().item()) * bs
#             n += bs

#         sch.step()

#         tr_loss = Lsum / max(1, n)
#         lr_now = sch.get_last_lr()[0]

#         print(
#             f"[{e:03d}] {time.time() - t0:5.1f}s | "
#             f"loss {tr_loss:.5f} | lr {lr_now:.2e} | "
#             f"pos_weight {pos_weight_value:.2f}"
#         )

#         last_ckpt = make_ckpt(
#             net,
#             opt,
#             args,
#             e,
#             tr_loss,
#             best,
#             pos_weight_value,
#             auto_pos_weight,
#         )

#         if args.save_every > 0 and (e % args.save_every == 0):
#             torch.save(last_ckpt, outdir / f"ep{e:03d}.pth")

#         if tr_loss < best:
#             best = tr_loss
#             best_ckpt = make_ckpt(
#                 net,
#                 opt,
#                 args,
#                 e,
#                 tr_loss,
#                 best,
#                 pos_weight_value,
#                 auto_pos_weight,
#             )
#             torch.save(best_ckpt, outdir / "best.pth")

#     if last_ckpt is not None:
#         final_ckpt = make_ckpt(
#             net,
#             opt,
#             args,
#             args.epochs,
#             tr_loss,
#             best,
#             pos_weight_value,
#             auto_pos_weight,
#         )
#         torch.save(final_ckpt, outdir / "last.pth")

#     print(f"[완료] best_loss={best:.5f}")
#     print(f"[저장] best: {outdir / 'best.pth'}")
#     print(f"[저장] last: {outdir / 'last.pth'}")


# if __name__ == "__main__":
#     ap = argparse.ArgumentParser()

#     ap.add_argument("--data", required=True)
#     ap.add_argument("--out", required=True)

#     ap.add_argument("--npts", type=int, default=16384)
#     ap.add_argument("--batch", type=int, default=4)
#     ap.add_argument("--epochs", type=int, default=30)
#     ap.add_argument("--lr", type=float, default=1e-4)
#     ap.add_argument("--wd", type=float, default=3e-4)

#     ap.add_argument("--k", type=int, default=16)

#     ap.add_argument("--no_norm", action="store_true")

#     ap.add_argument(
#         "--cache_mode",
#         choices=["none", "ram"],
#         default="ram",
#     )

#     ap.add_argument(
#         "--workers",
#         type=int,
#         default=0,
#     )

#     ap.add_argument(
#         "--prefetch_factor",
#         type=int,
#         default=2,
#     )

#     ap.add_argument(
#         "--save_every",
#         type=int,
#         default=1,
#     )

#     ap.add_argument(
#         "--amp",
#         action="store_true",
#     )

#     ap.add_argument(
#         "--pos_weight",
#         type=str,
#         default="30",
#         help="마커 포인트 가중치. 숫자 또는 auto 사용 가능",
#     )

#     ap.add_argument(
#         "--max_pos_weight",
#         type=float,
#         default=40.0,
#         help="--pos_weight auto 사용 시 최대값 제한. 0 이하이면 제한 없음",
#     )

#     args = ap.parse_args()

#     if args.workers < 0:
#         raise SystemExit("[검증실패] --workers는 0 이상이어야 함")

#     if args.cache_mode == "ram" and args.workers > 0 and os.name == "nt":
#         print("[주의] Windows에서 RAM cache + workers>0은 메모리 사용량이 커질 수 있음. 느리거나 메모리 문제가 있으면 --workers 0 사용")

#     preflight(args)
#     train(args)

#     #python -u .\marker_train_1_test.py --data "C:\Users\gkwld\Desktop\3D-Marker-Pose-Estimation\CAD_to_PointCloud_conversion\dataset\joint\train" --out "C:\Users\gkwld\Desktop\3D-Marker-Pose-Estimation\CAD_to_PointCloud_conversion\pointnet2\result\exp_seg_train1_posw80" --npts 16384 --batch 4 --epochs 30 --lr 1e-4 --wd 3e-4 --cache_mode ram --workers 0 --save_every 0 --amp --pos_weight 80    


# marker_train_1_test.py
# 기존 코드 주석처리 후, 아래 코드 사용
#
# 목적:
# - 파일명 marker_train_1_test.py 그대로 유지
# - PointNet++ 구조 유지
# - bbox/eval 코드 수정 없음
# - 학습 라벨만 재구성
#
# 라벨 규칙:
#   1  = 마커 포인트
#   0  = 마커 주변 XY 배경
#  -1  = 먼 배경 ignore
#
# 핵심:
# - 먼 배경은 loss 계산에서 제외
# - 마커 주변 지면만 background로 학습
# - 높이 정보는 hard threshold가 아니라 지나치게 높은 near negative 제거용 보조 조건으로만 사용

# marker_train_1_test.py
# 전체 덮어쓰기용
#
# 목적:
# - 기존 파일명 marker_train_1_test.py 그대로 사용
# - PointNet++ 구조 유지
# - 입력 파일은 기존과 동일하게 xyz + mask.npy 사용
# - 모델 내부에서 상대높이 feature 추가
#
# 핵심 변경:
# 기존: XYZ만 사용
# 수정: XYZ + relative_height 사용
#
# relative_height:
# - 각 샘플의 z 하위 분위수(z_floor)를 지면 기준으로 보고
# - height = z - z_floor 를 추가 feature로 사용
#
# 주의:
# - eval 코드에서 기존처럼 XYZ만 넣어도 작동함
# - Model forward 내부에서 height feature를 자동 생성함

import os
import csv
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if DEV.type == "cuda":
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


# ============================================================
# PointNet++ SSG + FP
# ============================================================
@torch.no_grad()
def fps(xyz, M):
    B, N, _ = xyz.shape
    M = int(max(1, min(M, N)))

    idx = torch.zeros(B, M, dtype=torch.long, device=xyz.device)
    far = torch.randint(0, N, (B,), device=xyz.device)
    dist = torch.full((B, N), 1e10, device=xyz.device)
    b = torch.arange(B, device=xyz.device)

    for i in range(M):
        idx[:, i] = far
        c = xyz[b, far, :].unsqueeze(1)
        d = ((xyz - c) ** 2).sum(-1)
        dist = torch.minimum(dist, d)
        far = torch.argmax(dist, dim=1)

    return idx


@torch.no_grad()
def knn(src, q, k=16):
    D = torch.cdist(q, src, p=2)
    _, ind = torch.topk(D, k, dim=-1, largest=False)
    return ind


def gather(P, idx):
    B, N, C = P.shape
    idx = idx.long()

    if idx.ndim == 2:
        B2, M = idx.shape
        if B2 != B:
            raise RuntimeError(f"batch mismatch: P={B}, idx={B2}")
        bidx = torch.arange(B, device=P.device).view(B, 1).expand(B, M)
        return P[bidx, idx]

    if idx.ndim == 3:
        B2, M, K = idx.shape
        if B2 != B:
            raise RuntimeError(f"batch mismatch: P={B}, idx={B2}")
        bidx = torch.arange(B, device=P.device).view(B, 1, 1).expand(B, M, K)
        return P[bidx, idx]

    raise RuntimeError("idx ndim must be 2 or 3")


class SA(nn.Module):
    def __init__(self, in_ch, mlp, M, k):
        super().__init__()
        self.M = int(M)
        self.k = int(k)

        ch = in_ch + 3
        layers = []

        for c in mlp:
            layers += [
                nn.Conv2d(ch, c, 1, bias=False),
                nn.BatchNorm2d(c),
                nn.ReLU(True),
            ]
            ch = c

        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz, feat):
        idx = fps(xyz, self.M)
        cen = gather(xyz, idx)
        nbr = knn(xyz, cen, self.k)

        gxyz = gather(xyz, nbr) - cen.unsqueeze(2)

        if feat is None:
            g = gxyz
        else:
            g = torch.cat([gather(feat, nbr), gxyz], dim=-1)

        x = self.mlp(g.permute(0, 3, 1, 2).contiguous())
        x = x.max(dim=-1)[0]
        x = x.permute(0, 2, 1).contiguous()

        return cen, x


class FP(nn.Module):
    def __init__(self, in_ch, mlp):
        super().__init__()

        ch = in_ch
        layers = []

        for c in mlp:
            layers += [
                nn.Conv1d(ch, c, 1, bias=False),
                nn.BatchNorm1d(c),
                nn.ReLU(True),
            ]
            ch = c

        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz_l, xyz_h, f_l, f_h):
        Nh = xyz_h.shape[1]
        k = 3 if Nh >= 3 else Nh

        if k <= 0:
            raise RuntimeError("FP: high-resolution set has zero points")

        d = torch.cdist(xyz_l, xyz_h, p=2) + 1e-8
        w, idx = torch.topk(d, k, dim=-1, largest=False)

        w = 1.0 / (w + 1e-8)
        w = w / w.sum(-1, keepdim=True)

        f = gather(f_h, idx)
        f = (f * w.unsqueeze(-1)).sum(2)

        if f_l is not None:
            f = torch.cat([f, f_l], dim=-1)

        x = f.permute(0, 2, 1).contiguous()
        x = self.mlp(x)
        x = x.permute(0, 2, 1).contiguous()

        return x


class Model(nn.Module):
    """
    입력:
    - [B,N,3] 또는 [B,3,N]

    내부 feature:
    - x, y, z
    - relative_height = z - z_floor

    출력:
    - [B,N,1] marker logit
    """

    def __init__(self, k=16, floor_q=0.05):
        super().__init__()

        self.floor_q = float(floor_q)

        # l0 feature = xyz(3) + relative_height(1) = 4ch
        self.sa1 = SA(4, [64, 64, 128], 512, k)
        self.sa2 = SA(128, [128, 128, 256], 128, k)
        self.sa3 = SA(256, [256, 512, 1024], 1, k)

        self.fp2 = FP(256 + 1024, [256, 256])
        self.fp1 = FP(128 + 256, [256, 128])
        self.fp0 = FP(128 + 4, [128, 128, 128])

        self.c1 = nn.Conv1d(128, 128, 1, bias=False)
        self.b1 = nn.BatchNorm1d(128)
        self.dp = nn.Dropout(0.5)
        self.c2 = nn.Conv1d(128, 1, 1)

    def _make_l0_feature(self, xyz):
        """
        xyz: [B,N,3]
        return: [B,N,4] = xyz + relative_height
        """
        z = xyz[:, :, 2]

        # torch.quantile이 환경에 따라 느릴 수 있으므로 kthvalue 방식 사용
        B, N = z.shape
        k_floor = int(max(1, min(N, round(N * self.floor_q))))

        z_sorted, _ = torch.sort(z, dim=1)
        z_floor = z_sorted[:, k_floor - 1].view(B, 1)

        rel_h = z - z_floor
        rel_h = rel_h.unsqueeze(-1)

        feat = torch.cat([xyz, rel_h], dim=-1)
        return feat

    def forward(self, x):
        if x.ndim != 3:
            raise RuntimeError(f"expected [B,N,3] or [B,3,N], got {tuple(x.shape)}")

        if x.size(-1) == 3:
            l0_xyz = x
        elif x.size(1) == 3:
            l0_xyz = x.permute(0, 2, 1).contiguous()
        else:
            raise RuntimeError(f"last or 2nd dim must be 3, got {tuple(x.shape)}")

        l0_feat = self._make_l0_feature(l0_xyz)

        l1_xyz, l1_pts = self.sa1(l0_xyz, l0_feat)
        l2_xyz, l2_pts = self.sa2(l1_xyz, l1_pts)
        l3_xyz, l3_pts = self.sa3(l2_xyz, l2_pts)

        f2 = self.fp2(l2_xyz, l3_xyz, l2_pts, l3_pts)
        f1 = self.fp1(l1_xyz, l2_xyz, l1_pts, f2)
        f0 = self.fp0(l0_xyz, l1_xyz, l0_feat, f1)

        y = f0.permute(0, 2, 1).contiguous()
        y = self.b1(self.c1(y))
        y = F.relu(y, True)
        y = self.dp(y)
        y = self.c2(y).permute(0, 2, 1).contiguous()

        return y


# ============================================================
# Dataset
# ============================================================
def _normalize_xyz(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    C = X.mean(0, keepdims=True)
    Xc = X - C

    s = np.linalg.norm(Xc, axis=1).max()
    s = float(s) if np.isfinite(s) and s > eps else 1.0

    return Xc / s


def _load_xyz_mask(xfp: Path, mfp: Path):
    X = np.loadtxt(str(xfp), dtype=np.float32)

    if X.ndim == 1:
        X = X.reshape(-1, 3)

    X = X[:, :3].astype(np.float32, copy=False)

    M = np.load(str(mfp)).astype(np.float32)

    if M.ndim != 1:
        M = M.reshape(-1)

    if X.shape[0] != M.shape[0]:
        raise RuntimeError(
            f"[데이터오류] point/mask 개수 불일치: {xfp.name}, "
            f"X={X.shape[0]}, M={M.shape[0]}"
        )

    if X.shape[0] <= 0:
        raise RuntimeError(f"[데이터오류] 포인트가 비어 있음: {xfp}")

    return X, M


class SegDataset(Dataset):
    def __init__(
        self,
        root: str,
        npts: int = 16384,
        normalize: bool = True,
        cache_mode: str = "ram",
        deterministic: bool = False,
        seed: int = 0,
    ):
        base = Path(root)
        pts_dir = base / "points"
        lab_csv = base / "labels.csv"

        if not pts_dir.is_dir():
            raise SystemExit(f"[검증실패] points 폴더 없음: {pts_dir}")

        if not lab_csv.exists():
            raise SystemExit(f"[검증실패] labels.csv 없음: {lab_csv}")

        self.root = base
        self.npts = int(npts)
        self.normalize = bool(normalize)
        self.cache_mode = str(cache_mode)
        self.deterministic = bool(deterministic)
        self.seed = int(seed)

        items = []

        with open(lab_csv, "r", encoding="utf-8") as f:
            rdr = csv.DictReader(f)

            if "filename" not in (rdr.fieldnames or []):
                raise SystemExit("[검증실패] labels.csv에 'filename' 필요")

            for r in rdr:
                stem = Path(r["filename"]).stem
                xfp = pts_dir / f"{stem}.xyz"
                mfp = pts_dir / f"{stem}.mask.npy"

                if xfp.exists() and mfp.exists():
                    items.append((xfp, mfp))

        if not items:
            raise SystemExit("[검증실패] xyz+mask 쌍이 없음")

        self.items = items
        self.cache = None
        self.pos_total = 0
        self.neg_total = 0
        self.mask_ratio = 0.0

        if self.cache_mode == "ram":
            t0 = time.time()
            self.cache = []

            for xfp, mfp in self.items:
                X, M = _load_xyz_mask(xfp, mfp)
                self.cache.append((X, M))

                pos = int((M > 0.5).sum())
                total = int(M.size)

                self.pos_total += pos
                self.neg_total += total - pos

            total_all = self.pos_total + self.neg_total
            self.mask_ratio = self.pos_total / max(1, total_all)

            print(
                f"[cache] RAM cache loaded: {len(self.cache)} samples, "
                f"{time.time() - t0:.1f}s, root={self.root}"
            )
        else:
            self._compute_mask_stats_from_files()

    def _compute_mask_stats_from_files(self):
        pos_total = 0
        neg_total = 0

        for _, mfp in self.items:
            M = np.load(str(mfp)).astype(np.float32)

            if M.ndim != 1:
                M = M.reshape(-1)

            pos = int((M > 0.5).sum())
            total = int(M.size)

            pos_total += pos
            neg_total += total - pos

        self.pos_total = pos_total
        self.neg_total = neg_total

        total_all = self.pos_total + self.neg_total
        self.mask_ratio = self.pos_total / max(1, total_all)

    def __len__(self):
        return len(self.items)

    def _sample(self, X, M, item_index: int):
        N = X.shape[0]

        if self.deterministic:
            rng = np.random.default_rng(self.seed + int(item_index))
        else:
            rng = np.random.default_rng()

        if N >= self.npts:
            idx = rng.choice(N, self.npts, replace=False)
        else:
            rep = (self.npts + N - 1) // N
            extra = rng.choice(N, rep * N - N, replace=True)
            idx = np.concatenate([np.arange(N), extra])[:self.npts]

        Xs = X[idx, :]
        Ms = M[idx][:, None]

        if self.normalize:
            Xs = _normalize_xyz(Xs).astype(np.float32, copy=False)

        return torch.from_numpy(Xs), torch.from_numpy(Ms)

    def __getitem__(self, i):
        if self.cache is not None:
            X, M = self.cache[i]
        else:
            xfp, mfp = self.items[i]
            X, M = _load_xyz_mask(xfp, mfp)

        return self._sample(X, M, i)


# ============================================================
# Train / Eval
# ============================================================
def preflight(args):
    for base_path in [Path(args.data), Path(args.val_data)]:
        if not (base_path / "points").is_dir():
            raise SystemExit(f"[검증실패] points 없음: {base_path / 'points'}")

        if not (base_path / "labels.csv").exists():
            raise SystemExit(f"[검증실패] labels.csv 없음: {base_path / 'labels.csv'}")

    net = Model(k=args.k, floor_q=args.floor_q).to(DEV).eval()

    with torch.inference_mode():
        dummy = torch.zeros((2, args.npts, 3), dtype=torch.float32, device=DEV)
        out = net(dummy)

        if not (
            isinstance(out, torch.Tensor)
            and out.ndim == 3
            and out.size() == torch.Size([2, args.npts, 1])
        ):
            raise SystemExit(f"[검증실패] 출력 [B,N,1] 아님: {tuple(out.shape)}")

    print("[검증통과] 경로/모델 OK")


def build_dataloader(ds, args, shuffle: bool):
    kwargs = dict(
        dataset=ds,
        batch_size=args.batch,
        shuffle=bool(shuffle),
        num_workers=args.workers,
        pin_memory=(DEV.type == "cuda"),
        drop_last=False,
    )

    if args.workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = args.prefetch_factor

    return DataLoader(**kwargs)


def resolve_pos_weight(args, ds):
    raw = str(args.pos_weight).strip().lower()

    auto_value = ds.neg_total / max(1, ds.pos_total)

    if raw == "auto":
        value = auto_value

        if args.max_pos_weight > 0:
            value = min(value, float(args.max_pos_weight))

        return float(value), float(auto_value)

    try:
        value = float(raw)
    except ValueError:
        raise SystemExit("[검증실패] --pos_weight는 숫자 또는 auto만 가능")

    if value <= 0:
        raise SystemExit("[검증실패] --pos_weight는 0보다 커야 함")

    return float(value), float(auto_value)


def metric_is_better(metric_name, value, best_value):
    if metric_name in ["val_loss", "train_loss"]:
        return value < best_value
    return value > best_value


def initial_best_value(metric_name):
    if metric_name in ["val_loss", "train_loss"]:
        return float("inf")
    return -float("inf")


def make_ckpt(
    net,
    opt,
    args,
    epoch,
    train_loss,
    best_value,
    pos_weight_value,
    auto_pos_weight,
    metrics,
):
    return {
        "model": net.state_dict(),
        "opt": opt.state_dict(),
        "meta": {
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "best_value": float(best_value),
            "best_metric": str(args.best_metric),
            "npts": int(args.npts),
            "task": "seg",
            "normalize": bool(not args.no_norm),
            "k": int(args.k),
            "floor_q": float(args.floor_q),
            "feature": "xyz_plus_relative_height",
            "cache_mode": str(args.cache_mode),
            "amp": bool(args.amp and DEV.type == "cuda"),
            "pos_weight": float(pos_weight_value),
            "auto_pos_weight": float(auto_pos_weight),
            "metric_th": float(args.metric_th),
            "train_data": str(args.data),
            "val_data": str(args.val_data),
            "metrics": dict(metrics),
        },
    }


def append_metrics_csv(csv_path: Path, row: dict):
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "epoch",
        "lr",
        "pos_weight",
        "floor_q",
        "train_loss",
        "val_loss",
        "precision",
        "recall",
        "iou",
        "f1",
        "tp",
        "fp",
        "fn",
        "tn",
        "best_metric",
        "best_value",
        "saved_best",
    ]

    exists = csv_path.exists()

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)

        if not exists:
            wr.writeheader()

        wr.writerow(row)


@torch.no_grad()
def evaluate_epoch(net, dl, crit, metric_th: float):
    net.eval()

    loss_sum = 0.0
    sample_count = 0

    tp = 0
    fp = 0
    fn = 0
    tn = 0

    for pts, m in dl:
        pts = pts.to(DEV, non_blocking=True)
        m = m.to(DEV, non_blocking=True)

        logits = net(pts)
        loss = crit(logits, m)

        bs = pts.size(0)
        loss_sum += float(loss.detach().item()) * bs
        sample_count += bs

        prob = torch.sigmoid(logits)
        pred = prob >= float(metric_th)
        target = m > 0.5

        tp += int((pred & target).sum().item())
        fp += int((pred & (~target)).sum().item())
        fn += int(((~pred) & target).sum().item())
        tn += int(((~pred) & (~target)).sum().item())

    val_loss = loss_sum / max(1, sample_count)

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    iou = tp / max(1, tp + fp + fn)
    f1 = (2.0 * precision * recall) / max(1e-12, precision + recall)

    return {
        "val_loss": float(val_loss),
        "precision": float(precision),
        "recall": float(recall),
        "iou": float(iou),
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def train(args):
    train_ds = SegDataset(
        args.data,
        npts=args.npts,
        normalize=(not args.no_norm),
        cache_mode=args.cache_mode,
        deterministic=False,
        seed=args.seed,
    )

    val_ds = SegDataset(
        args.val_data,
        npts=args.npts,
        normalize=(not args.no_norm),
        cache_mode=args.cache_mode,
        deterministic=True,
        seed=args.seed + 100000,
    )

    pos_weight_value, auto_pos_weight = resolve_pos_weight(args, train_ds)

    print(
        f"[train mask] pos={train_ds.pos_total}  neg={train_ds.neg_total}  "
        f"pos_ratio={train_ds.mask_ratio:.6f}  "
        f"auto_pos_weight={auto_pos_weight:.3f}  "
        f"used_pos_weight={pos_weight_value:.3f}"
    )

    print(
        f"[val mask] pos={val_ds.pos_total}  neg={val_ds.neg_total}  "
        f"pos_ratio={val_ds.mask_ratio:.6f}"
    )

    train_dl = build_dataloader(train_ds, args, shuffle=True)
    val_dl = build_dataloader(val_ds, args, shuffle=False)

    print(
        f"samples={len(train_ds)}  "
        f"val_samples={len(val_ds)}  "
        f"npts={args.npts}  "
        f"batch={args.batch}  "
        f"device={DEV.type}  "
        f"normalize={not args.no_norm}  "
        f"feature=xyz_plus_relative_height  "
        f"floor_q={args.floor_q}  "
        f"cache_mode={args.cache_mode}  "
        f"workers={args.workers}  "
        f"save_every={args.save_every}  "
        f"amp={bool(args.amp and DEV.type == 'cuda')}  "
        f"best_metric={args.best_metric}  "
        f"metric_th={args.metric_th}"
    )

    net = Model(k=args.k, floor_q=args.floor_q).to(DEV)

    pos_weight_tensor = torch.tensor([pos_weight_value], dtype=torch.float32, device=DEV)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))

    use_amp = bool(args.amp and DEV.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    metrics_csv = outdir / "metrics.csv"

    best_value = initial_best_value(args.best_metric)
    last_ckpt = None
    metrics = {}

    for e in range(1, args.epochs + 1):
        net.train()

        t0 = time.time()
        loss_sum = 0.0
        sample_count = 0

        for pts, m in train_dl:
            pts = pts.to(DEV, non_blocking=True)
            m = m.to(DEV, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            if use_amp:
                with torch.cuda.amp.autocast(enabled=True):
                    logits = net(pts)
                    loss = crit(logits, m)

                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

            else:
                logits = net(pts)
                loss = crit(logits, m)
                loss.backward()
                opt.step()

            bs = pts.size(0)
            loss_sum += float(loss.detach().item()) * bs
            sample_count += bs

        sch.step()

        train_loss = loss_sum / max(1, sample_count)
        lr_now = sch.get_last_lr()[0]

        val_metrics = evaluate_epoch(
            net,
            val_dl,
            crit,
            metric_th=args.metric_th,
        )

        metrics = {
            "train_loss": float(train_loss),
            **val_metrics,
        }

        if args.best_metric == "train_loss":
            current_value = float(train_loss)
        elif args.best_metric == "val_loss":
            current_value = float(val_metrics["val_loss"])
        elif args.best_metric == "val_precision":
            current_value = float(val_metrics["precision"])
        elif args.best_metric == "val_recall":
            current_value = float(val_metrics["recall"])
        elif args.best_metric == "val_iou":
            current_value = float(val_metrics["iou"])
        elif args.best_metric == "val_f1":
            current_value = float(val_metrics["f1"])
        else:
            raise SystemExit(f"[검증실패] 지원하지 않는 best_metric: {args.best_metric}")

        saved_best = False

        if metric_is_better(args.best_metric, current_value, best_value):
            best_value = current_value
            best_ckpt = make_ckpt(
                net,
                opt,
                args,
                e,
                train_loss,
                best_value,
                pos_weight_value,
                auto_pos_weight,
                metrics,
            )
            torch.save(best_ckpt, outdir / "best.pth")
            saved_best = True

        last_ckpt = make_ckpt(
            net,
            opt,
            args,
            e,
            train_loss,
            best_value,
            pos_weight_value,
            auto_pos_weight,
            metrics,
        )

        if args.save_every > 0 and (e % args.save_every == 0):
            torch.save(last_ckpt, outdir / f"ep{e:03d}.pth")

        append_metrics_csv(
            metrics_csv,
            {
                "epoch": e,
                "lr": f"{lr_now:.8e}",
                "pos_weight": f"{pos_weight_value:.6f}",
                "floor_q": f"{args.floor_q:.6f}",
                "train_loss": f"{train_loss:.8f}",
                "val_loss": f"{val_metrics['val_loss']:.8f}",
                "precision": f"{val_metrics['precision']:.8f}",
                "recall": f"{val_metrics['recall']:.8f}",
                "iou": f"{val_metrics['iou']:.8f}",
                "f1": f"{val_metrics['f1']:.8f}",
                "tp": int(val_metrics["tp"]),
                "fp": int(val_metrics["fp"]),
                "fn": int(val_metrics["fn"]),
                "tn": int(val_metrics["tn"]),
                "best_metric": args.best_metric,
                "best_value": f"{best_value:.8f}",
                "saved_best": int(saved_best),
            },
        )

        print(
            f"[{e:03d}] {time.time() - t0:5.1f}s | "
            f"train_loss {train_loss:.5f} | "
            f"val_loss {val_metrics['val_loss']:.5f} | "
            f"P {val_metrics['precision']:.4f} | "
            f"R {val_metrics['recall']:.4f} | "
            f"IoU {val_metrics['iou']:.4f} | "
            f"F1 {val_metrics['f1']:.4f} | "
            f"lr {lr_now:.2e} | "
            f"pos_weight {pos_weight_value:.2f} | "
            f"best {args.best_metric}={best_value:.5f}"
            f"{' | saved_best' if saved_best else ''}"
        )

    if last_ckpt is not None:
        final_ckpt = make_ckpt(
            net,
            opt,
            args,
            args.epochs,
            train_loss,
            best_value,
            pos_weight_value,
            auto_pos_weight,
            metrics,
        )
        torch.save(final_ckpt, outdir / "last.pth")

    print(f"[완료] best_metric={args.best_metric}  best_value={best_value:.6f}")
    print(f"[저장] best: {outdir / 'best.pth'}")
    print(f"[저장] last: {outdir / 'last.pth'}")
    print(f"[저장] metrics: {metrics_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("--data", required=True)
    ap.add_argument("--val_data", required=True)
    ap.add_argument("--out", required=True)

    ap.add_argument("--npts", type=int, default=16384)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=3e-4)

    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--floor_q", type=float, default=0.05)

    ap.add_argument("--no_norm", action="store_true")

    ap.add_argument(
        "--cache_mode",
        choices=["none", "ram"],
        default="ram",
    )

    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--prefetch_factor", type=int, default=2)
    ap.add_argument("--save_every", type=int, default=0)
    ap.add_argument("--amp", action="store_true")

    ap.add_argument(
        "--pos_weight",
        type=str,
        default="10",
        help="마커 포인트 가중치. 숫자 또는 auto 사용 가능",
    )

    ap.add_argument(
        "--max_pos_weight",
        type=float,
        default=40.0,
        help="--pos_weight auto 사용 시 최대값 제한. 0 이하이면 제한 없음",
    )

    ap.add_argument("--metric_th", type=float, default=0.5)

    ap.add_argument(
        "--best_metric",
        choices=[
            "train_loss",
            "val_loss",
            "val_precision",
            "val_recall",
            "val_iou",
            "val_f1",
        ],
        default="val_f1",
    )

    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()

    if args.workers < 0:
        raise SystemExit("[검증실패] --workers는 0 이상이어야 함")

    if args.cache_mode == "ram" and args.workers > 0 and os.name == "nt":
        print("[주의] Windows에서 RAM cache + workers>0은 메모리 사용량이 커질 수 있음. --workers 0 권장")

    preflight(args)
    train(args)