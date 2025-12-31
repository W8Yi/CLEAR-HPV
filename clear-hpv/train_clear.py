#!/usr/bin/env python3
"""
Unified trainer for hpv_uni2, k=10, raw h-space, across CLAM checkpoints s_0..s_9.

Per split s:
  • Load CLAM weights: /common/users/wq50/CLAM/results/HPV_CLAM_50_mb_s1/s_{s}_checkpoint.pt
  • Read splits_{s}.csv (TRAIN/TEST slide IDs)
  • Train on TRAIN slides only:
      1) k-means (hard, unweighted)
      2) k-means (hard, attention-weighted)
      3) soft k-means (attention-weighted EM)
      4) vMF (spherical) soft k-means
      5) Student-t mixture (diag) soft EM
      6) Balanced Sinkhorn-k-means (OT)
      7) PACE (global GMM + amortized q_phi, per-slide Dirichlet gamma)
  • Export soft responsibilities r_ic for ALL slides (train+test) per method

Artifacts go under: ./all_models_hpv_uni2_k10/s_{s}/<method_tag>/

Author: Weiyi Qin (consolidated)
"""

from pathlib import Path
import gc, json, re
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
import h5py, joblib
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics import pairwise_distances_argmin

from models.model_clam import CLAM_MB

# ────────────── paths & constants ──────────────
DATASET     = "CPTAC"
FEAT_DIR    = Path("/common/users/wq50/CLAM/features/CPTAC_HNSCC/h5_files")
RESULTS_DIR = Path("/common/users/wq50/CLAM/results/HPV_CLAM_50_mb_s1")   # s_?_checkpoint.pt + splits_?.csv
OUT_BASE    = Path("./all_models_hpv_cptac_k10"); OUT_BASE.mkdir(exist_ok=True)

K              = 10
BATCH_SIZE     = 16_384
RAND_STATE     = 0
MAX_ITER       = 300
MB_INIT_SIZE   = 20_000
MB_N_INIT      = 10
DEVICE         = "cuda:5" if torch.cuda.is_available() else "cpu"

# Soft EM (Euclidean)
TAU_SOFT       = 0.5
EPOCHS_SOFT    = 5
DELTA_TOL      = 1e-4

# vMF (spherical)
TAU_VMF        = 0.07

# t-mixture
NU_T           = 10.0
VAR_FLOOR      = 1e-3

# Sinkhorn OT
EPS_OT         = 0.05
SINK_ITERS     = 100
SINK_TOL       = 1e-4

# PACE (core)
PACE_EPOCHS    = 10
PACE_LR        = 2e-3
PACE_COV_FLOOR = 1e-2
ALPHA0         = 1.0            # Dirichlet prior strength for per-slide theta
PACE_HID       = 512

# ────────────── CLAM: (h, α) ──────────────
@torch.inference_mode()
def h_and_alpha_block(raw_block: np.ndarray, clam: CLAM_MB):
    """
    raw_block: (m,1536) float32
    Returns:
      h: (m,512) float32
      a: (m,)    float32 attention (instance-softmax per block), renorm to mean≈1
    """
    x = torch.from_numpy(raw_block).to(DEVICE)
    A_raw, h = clam.attention_net(x)          # A_raw: (m,2), h: (m,512)
    a = torch.softmax(A_raw[:,0], dim=0).clamp_min(1e-12)
    a = a * (a.numel() / a.sum())
    return h.cpu().numpy().astype(np.float32), a.cpu().numpy().astype(np.float32)

def iter_h5_blocks(h5_path: Path, batch=BATCH_SIZE):
    with h5py.File(h5_path, "r") as f:
        feats = f["features"]; n = feats.shape[0]
        for off in range(0, n, batch):
            yield off, feats[off:off+batch][:].astype(np.float32)

def train_slide_stream(slide_ids, clam):
    for sid in slide_ids:
        h5 = FEAT_DIR / f"{sid}.h5"
        if not h5.exists(): continue
        for _, raw in iter_h5_blocks(h5):
            H, a = h_and_alpha_block(raw, clam)
            yield H, a

def load_clam(weight_path: Path):
    m = CLAM_MB(gate=True, size_arg="small", n_classes=2, embed_dim=1536)
    m.load_state_dict(torch.load(weight_path, map_location=DEVICE), strict=False)
    return m.to(DEVICE).eval()

def get_train_test_ids(split_csv: Path):
    df = pd.read_csv(split_csv, index_col=0)
    train_ids = [s for s in df["train"].dropna().astype(str)]
    test_ids  = [s for s in df["test"].dropna().astype(str)]
    return train_ids, test_ids

# ────────────── Shared: export r_ic shards (generic soft assignment) ──────────────
def _save_shards_from_R(R, slide, ep, off, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{slide}_ep{ep}_b{off}_soft.npy", R.astype(np.float16))

# ────────────── 1) k-means (hard, unweighted) ──────────────
def train_kmeans(train_ids, clam, k=K):
    mbk = MiniBatchKMeans(
        n_clusters=k, batch_size=BATCH_SIZE, init_size=MB_INIT_SIZE,
        n_init=MB_N_INIT, random_state=RAND_STATE, reassignment_ratio=0, verbose=0
    )
    for H, _ in train_slide_stream(train_ids, clam):
        mbk.partial_fit(H)
    # full Lloyd (RAM)
    H_all = [H for H,_ in train_slide_stream(train_ids, clam)]
    X = np.vstack(H_all).astype(np.float32)
    km = KMeans(n_clusters=k, init=mbk.cluster_centers_, n_init=1, max_iter=MAX_ITER,
                copy_x=False, random_state=RAND_STATE, verbose=0).fit(X)
    del H_all, X; gc.collect()
    return km

# ────────────── 2) k-means (hard, attention-weighted) ──────────────
def train_aw_kmeans(train_ids, clam, k=K):
    mbk = MiniBatchKMeans(
        n_clusters=k, batch_size=BATCH_SIZE, init_size=MB_INIT_SIZE,
        n_init=MB_N_INIT, random_state=RAND_STATE, reassignment_ratio=0, verbose=0
    )
    for H, a in train_slide_stream(train_ids, clam):
        mbk.partial_fit(H, sample_weight=a)
    H_all, W_all = [], []
    for H, a in train_slide_stream(train_ids, clam):
        H_all.append(H); W_all.append(a)
    X = np.vstack(H_all).astype(np.float32)
    W = np.concatenate(W_all).astype(np.float32)
    km = KMeans(n_clusters=k, init=mbk.cluster_centers_, n_init=1, max_iter=MAX_ITER,
                copy_x=False, random_state=RAND_STATE, verbose=0).fit(X, sample_weight=W)
    del H_all, W_all, X, W; gc.collect()
    return km

# ────────────── 3) soft k-means (attention-weighted EM) ──────────────
def train_soft_aw_em(train_ids, clam, k=K, tau=TAU_SOFT, epochs=EPOCHS_SOFT, tol=DELTA_TOL):
    rng = np.random.default_rng(RAND_STATE)
    # weighted k-means++ init
    H_res, A_res = [], []
    for H, a in train_slide_stream(train_ids, clam):
        H_res.append(H); A_res.append(a)
    H_res = np.vstack(H_res); A_res = np.concatenate(A_res)
    p0 = A_res / A_res.sum(); idx0 = rng.choice(H_res.shape[0], p=p0)
    centers = [H_res[idx0]]; d2 = ((H_res - centers[0])**2).sum(1)
    for _ in range(1, k):
        q = A_res * d2; q = q / q.sum()
        j = rng.choice(H_res.shape[0], p=q)
        centers.append(H_res[j]); d2 = np.minimum(d2, ((H_res - centers[-1])**2).sum(1))
    centers = np.vstack(centers).astype(np.float32)
    del H_res, A_res, d2; gc.collect()

    prev = centers.copy()
    for ep in range(1, epochs+1):
        print(f"   [softEM] epoch {ep}/{epochs}")
        num = np.zeros_like(centers, dtype=np.float64)
        den = np.zeros((k,), dtype=np.float64)
        for H, a in train_slide_stream(train_ids, clam):
            HH = (H*H).sum(1, keepdims=True)
            CC = (centers*centers).sum(1, keepdims=True).T
            D2 = HH + CC - 2.0 * (H @ centers.T)
            logits = -D2 / tau
            logits -= logits.max(1, keepdims=True)
            R = np.exp(logits); R /= R.sum(1, keepdims=True)
            wR = (a[:,None] * R).astype(np.float64)
            num += wR.T @ H.astype(np.float64); den += wR.sum(0)
            del H, a, HH, CC, D2, logits, R, wR
            gc.collect()
        mask = den > 0
        centers[mask] = (num[mask] / den[mask,None]).astype(np.float32)
        rel = np.linalg.norm(centers - prev) / max(np.linalg.norm(prev), 1e-12)
        print(f"      Δcenters: {rel:.3e}")
        if rel < tol: break
        prev = centers.copy()
    return centers

def export_r_softEM_all_slides(centers: np.ndarray, out_dir: Path, clam: CLAM_MB, tau=TAU_SOFT, ep_tag=1):
    Kc = centers.shape[0]
    C_t = torch.from_numpy(centers).to(DEVICE)
    shard_dir = out_dir / "r_shards"; shard_dir.mkdir(parents=True, exist_ok=True)
    for h5 in sorted(FEAT_DIR.glob("*.h5")):
        slide = h5.stem
        with h5py.File(h5, "r") as f:
            feats = f["features"]; n = feats.shape[0]
            for off in range(0, n, BATCH_SIZE):
                raw = feats[off:off+BATCH_SIZE][:].astype(np.float32)
                H, _ = h_and_alpha_block(raw, clam)
                Ht = torch.from_numpy(H).to(DEVICE)
                D2 = (Ht*Ht).sum(1,True) - 2.0*(Ht@C_t.t()) + (C_t*C_t).sum(1,True).t()
                logits = -D2 / tau; logits = logits - logits.max(1,True).values
                R = torch.softmax(logits,1).cpu().numpy().astype(np.float16)
                _save_shards_from_R(R, slide, ep_tag, off, shard_dir)
                del raw, H, Ht, D2, logits, R
        gc.collect()
    return shard_dir

# ────────────── 4) vMF (spherical) soft k-means ──────────────
def train_vmf(train_ids, clam, k=K, tau=TAU_VMF, epochs=EPOCHS_SOFT):
    rng = np.random.default_rng(RAND_STATE)
    # Init from unit-norm reservoir
    H_res, A_res = [], []
    for H, a in train_slide_stream(train_ids, clam):
        Hn = H / np.maximum(np.linalg.norm(H, axis=1, keepdims=True), 1e-12)
        H_res.append(Hn); A_res.append(a)
    H_res = np.vstack(H_res); A_res = np.concatenate(A_res)
    p0 = A_res / A_res.sum(); idx0 = rng.choice(H_res.shape[0], p=p0)
    MU = [H_res[idx0]]; d2 = ((H_res - MU[0])**2).sum(1)
    for _ in range(1, k):
        q = A_res * d2; q = q / q.sum(); j = rng.choice(H_res.shape[0], p=q)
        MU.append(H_res[j]); d2 = np.minimum(d2, ((H_res - MU[-1])**2).sum(1))
    MU = torch.tensor(np.vstack(MU), dtype=torch.float32, device=DEVICE)
    del H_res, A_res, d2; gc.collect()

    for ep in range(1, epochs+1):
        print(f"   [vMF] epoch {ep}/{epochs}")
        num = torch.zeros_like(MU); den = torch.zeros((k,), dtype=torch.float32, device=DEVICE)
        for H, a in train_slide_stream(train_ids, clam):
            Ht = torch.from_numpy(H).to(DEVICE)
            Hn = torch.nn.functional.normalize(Ht, dim=1)
            logits = (Hn @ MU.T) / tau
            logits = logits - logits.max(1,True).values
            R = torch.softmax(logits, dim=1)              # (m,k)
            w = a[:,None]; w = torch.from_numpy(w).to(DEVICE)
            num += (w*R).T @ Hn; den += (w*R).sum(0)
            del H, a, Ht, Hn, logits, R, w; gc.collect()
        MU = num / den.unsqueeze(1).clamp_min(1e-9)
        MU = torch.nn.functional.normalize(MU, dim=1)
    return MU.detach().cpu().numpy()

def export_r_vmf_all_slides(MU_np: np.ndarray, out_dir: Path, clam: CLAM_MB, tau=TAU_VMF, ep_tag=1):
    MU = torch.from_numpy(MU_np).to(DEVICE)
    shard_dir = out_dir / "r_shards"; shard_dir.mkdir(parents=True, exist_ok=True)
    for h5 in sorted(FEAT_DIR.glob("*.h5")):
        slide = h5.stem
        with h5py.File(h5, "r") as f:
            feats = f["features"]; n = feats.shape[0]
            for off in range(0, n, BATCH_SIZE):
                raw = feats[off:off+BATCH_SIZE][:].astype(np.float32)
                H, _ = h_and_alpha_block(raw, clam)
                Hn = torch.nn.functional.normalize(torch.from_numpy(H).to(DEVICE), dim=1)
                logits = (Hn @ MU.T) / tau
                logits = logits - logits.max(1,True).values
                R = torch.softmax(logits,1).cpu().numpy().astype(np.float16)
                _save_shards_from_R(R, slide, ep_tag, off, shard_dir)
                del raw, H, Hn, logits, R
        gc.collect()
    return shard_dir

# ────────────── 5) Student-t mixture (diag) ──────────────
@torch.inference_mode(False)
def t_logpdf_diag(H, MU, VAR, nu: float):
    m, D = H.shape; Kc = MU.shape[0]
    diff = H.unsqueeze(1) - MU.unsqueeze(0)
    invv = 1.0 / VAR.clamp_min(VAR_FLOOR)
    delta = (diff*diff*invv.unsqueeze(0)).sum(2)
    c1 = torch.lgamma(torch.tensor((nu + D)/2, device=H.device)) - torch.lgamma(torch.tensor(nu/2, device=H.device))
    c2 = -0.5 * (D * torch.log(torch.tensor(nu*np.pi, device=H.device)) + torch.log(VAR.clamp_min(VAR_FLOOR)).sum(1))
    return c1 + c2.unsqueeze(0) - ((nu + D)/2)*torch.log1p(delta/nu), delta  # (m,K), (m,K)

def train_t_mixture(train_ids, clam, k=K, nu=NU_T, epochs=EPOCHS_SOFT):
    rng = np.random.default_rng(RAND_STATE)
    # init MU from reservoir, VAR=I, PI uniform
    H_res = np.vstack([H for H,_ in train_slide_stream(train_ids, clam)])
    idx = rng.choice(H_res.shape[0], size=k, replace=False)
    MU  = torch.tensor(H_res[idx], dtype=torch.float32, device=DEVICE)
    VAR = torch.ones((k, H_res.shape[1]), dtype=torch.float32, device=DEVICE)
    PI  = torch.full((k,), 1.0/k, dtype=torch.float32, device=DEVICE)
    del H_res; gc.collect()

    for ep in range(1, epochs+1):
        print(f"   [tMix] epoch {ep}/{epochs}")
        num_mu = torch.zeros_like(MU, dtype=torch.float64)
        den_mu = torch.zeros((k,), dtype=torch.float64, device=DEVICE)
        den_pi = torch.zeros((k,), dtype=torch.float64, device=DEVICE)

        # pass 1: MU, PI
        for H_np, a_np in train_slide_stream(train_ids, clam):
            H = torch.from_numpy(H_np).to(DEVICE)
            a = torch.from_numpy(a_np).to(DEVICE)
            logp, delta = t_logpdf_diag(H, MU, VAR, nu)
            logits = logp + torch.log(PI+1e-12).unsqueeze(0)
            logits = logits - logits.max(1,True).values
            R = torch.softmax(logits,1)
            u = (nu + H.shape[1]) / (nu + delta + 1e-12)          # latent scales
            w = (a.unsqueeze(1) * R * u).to(torch.float64)
            num_mu += w.t() @ H.to(torch.float64)
            den_mu += w.sum(0)
            den_pi += (a.unsqueeze(1)*R).sum(0).to(torch.float64)
            del H_np, a_np, H, a, logp, delta, logits, R, u, w; gc.collect()
        MU = (num_mu / den_mu.unsqueeze(1).clamp_min(1e-9)).to(torch.float32)
        PI = (den_pi / den_pi.sum().clamp_min(1e-9)).to(torch.float32)

        # pass 2: VAR
        num_var = torch.zeros_like(MU, dtype=torch.float64)
        den_var = torch.zeros((k,), dtype=torch.float64, device=DEVICE)
        for H_np, a_np in train_slide_stream(train_ids, clam):
            H = torch.from_numpy(H_np).to(DEVICE)
            a = torch.from_numpy(a_np).to(DEVICE)
            logp, delta = t_logpdf_diag(H, MU, VAR, nu)
            logits = logp + torch.log(PI+1e-12).unsqueeze(0)
            logits = logits - logits.max(1,True).values
            R = torch.softmax(logits,1)
            u = (nu + H.shape[1]) / (nu + delta + 1e-12)
            diff2 = (H.unsqueeze(1) - MU.unsqueeze(0))**2
            w = (a.unsqueeze(1) * R * u).to(torch.float64)
            num_var += (w.unsqueeze(2) * diff2.to(torch.float64)).sum(0)
            den_var += (a.unsqueeze(1)*R).sum(0).to(torch.float64)
            del H, a, logp, delta, logits, R, u, diff2, w; gc.collect()
        VAR = (num_var / den_var.unsqueeze(1).clamp_min(1e-9)).clamp_min(VAR_FLOOR).to(torch.float32)

    return MU.detach().cpu().numpy(), VAR.detach().cpu().numpy(), PI.detach().cpu().numpy()

def export_r_tmix_all_slides(MU_np, VAR_np, PI_np, out_dir: Path, clam: CLAM_MB, nu=NU_T, ep_tag=1):
    MU = torch.from_numpy(MU_np).to(DEVICE); VAR = torch.from_numpy(VAR_np).to(DEVICE); PI = torch.from_numpy(PI_np).to(DEVICE)
    shard_dir = out_dir / "r_shards"; shard_dir.mkdir(parents=True, exist_ok=True)
    for h5 in sorted(FEAT_DIR.glob("*.h5")):
        slide = h5.stem
        with h5py.File(h5, "r") as f:
            feats = f["features"]; n = feats.shape[0]
            for off in range(0, n, BATCH_SIZE):
                raw = feats[off:off+BATCH_SIZE][:].astype(np.float32)
                H_np, _ = h_and_alpha_block(raw, clam)
                H = torch.from_numpy(H_np).to(DEVICE)
                logp, _ = t_logpdf_diag(H, MU, VAR, nu)
                logits = logp + torch.log(PI+1e-12).unsqueeze(0)
                logits = logits - logits.max(1,True).values
                R = torch.softmax(logits,1).cpu().numpy().astype(np.float16)
                _save_shards_from_R(R, slide, ep_tag, off, shard_dir)
                del raw, H_np, H, logp, logits, R
        gc.collect()
    return shard_dir

# ────────────── 6) Balanced Sinkhorn-k-means ──────────────
def sinkhorn(a, C, eps=EPS_OT, iters=SINK_ITERS, tol=SINK_TOL):
    with torch.no_grad():
        m, Kc = C.shape
        total = a.sum().item()
        b = torch.full((Kc,), total / Kc, dtype=torch.float32, device=C.device)
        Kmat = torch.exp(-C / eps) + 1e-12
        u = torch.ones_like(a); v = torch.ones_like(b)
        for _ in range(iters):
            u_prev = u
            u = a / (Kmat @ v + 1e-12)
            v = b / (Kmat.t() @ u + 1e-12)
            if torch.max(torch.abs(u - u_prev)) < tol:
                break
        return (u.unsqueeze(1) * Kmat) * v.unsqueeze(0)

def train_sinkhorn(train_ids, clam, k=K, epochs=EPOCHS_SOFT):
    rng = np.random.default_rng(RAND_STATE)
    H_res = np.vstack([H for H,_ in train_slide_stream(train_ids, clam)])
    idx = rng.choice(H_res.shape[0], size=k, replace=False)
    MU = torch.tensor(H_res[idx], dtype=torch.float32, device=DEVICE)
    del H_res; gc.collect()

    for ep in range(1, epochs+1):
        print(f"   [Sinkhorn] epoch {ep}/{epochs}")
        num = torch.zeros_like(MU, dtype=torch.float64)
        den = torch.zeros((k,), dtype=torch.float64, device=DEVICE)
        for H_np, a_np in train_slide_stream(train_ids, clam):
            H = torch.from_numpy(H_np).to(DEVICE)
            a = torch.from_numpy(a_np).to(DEVICE)
            C = (H*H).sum(1,True) - 2.0*(H@MU.T) + (MU*MU).sum(1,True).t()
            T = sinkhorn(a, C)
            R = (T / a.unsqueeze(1).clamp_min(1e-12)).clamp_min(0)   # rows ≈1
            T64 = T.to(torch.float64)
            num += (T64.t() @ H.to(torch.float64)); den += T64.sum(0)
            del H_np, a_np, H, a, C, T, T64, R; gc.collect()
        MU = (num / den.unsqueeze(1).clamp_min(1e-9)).to(torch.float32)
    return MU.detach().cpu().numpy()

def export_r_sinkhorn_all_slides(MU_np: np.ndarray, out_dir: Path, clam: CLAM_MB, ep_tag=1):
    MU = torch.from_numpy(MU_np).to(DEVICE)
    shard_dir = out_dir / "r_shards"; shard_dir.mkdir(parents=True, exist_ok=True)
    for h5 in sorted(FEAT_DIR.glob("*.h5")):
        slide = h5.stem
        with h5py.File(h5, "r") as f:
            feats = f["features"]; n = feats.shape[0]
            for off in range(0, n, BATCH_SIZE):
                raw = feats[off:off+BATCH_SIZE][:].astype(np.float32)
                H_np, a_np = h_and_alpha_block(raw, clam)
                H = torch.from_numpy(H_np).to(DEVICE)
                a = torch.from_numpy(a_np).to(DEVICE)
                C = (H*H).sum(1,True) - 2.0*(H@MU.T) + (MU*MU).sum(1,True).t()
                T = sinkhorn(a, C)
                R = (T / a.unsqueeze(1).clamp_min(1e-12)).clamp_min(0).cpu().numpy().astype(np.float16)
                _save_shards_from_R(R, slide, ep_tag, off, shard_dir)
                del raw, H_np, a_np, H, a, C, T, R
        gc.collect()
    return shard_dir

# ────────────── 7) PACE (global GMM + amortized q, per-slide Dirichlet) ──────────────
class Explainer(nn.Module):
    def __init__(self, d=512, k=10, hid=PACE_HID):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d, hid), nn.GELU(),
            nn.Linear(hid, hid//2), nn.GELU(),
            nn.Linear(hid//2, k)
        )
    def forward(self, h): return self.mlp(h)  # logits (m,k)

class ConceptParams(nn.Module):
    def __init__(self, d=512, k=10):
        super().__init__()
        self.eta = nn.Parameter(torch.zeros(k))                 # logits for pi
        self.mu  = nn.Parameter(torch.randn(k, d)*0.01)
        self.logvar = nn.Parameter(torch.zeros(k, d))           # diag covariance
    def forward(self):
        pi = F.softmax(self.eta, dim=0)                         # (k,)
        var = torch.exp(self.logvar).clamp_min(PACE_COV_FLOOR)  # (k,d)
        return pi, self.mu, var

def log_gauss_diag(h, mu, var):
    diff = h.unsqueeze(1) - mu.unsqueeze(0)                     # (m,k,d)
    invv = 1.0 / var.unsqueeze(0)                               # (1,k,d)
    logdet = torch.log(var).sum(1)                              # (k,)
    quad = (diff*diff*invv).sum(2)                              # (m,k)
    d = h.shape[1]
    return -0.5*(quad + logdet.unsqueeze(0) + d*np.log(2*np.pi))# (m,k)

def train_pace(train_ids, clam, k=K, epochs=PACE_EPOCHS, lr=PACE_LR, alpha0=ALPHA0):
    expl = Explainer(512, k).to(DEVICE)
    params = ConceptParams(512, k).to(DEVICE)
    opt = torch.optim.Adam(list(expl.parameters())+list(params.parameters()), lr=lr)

    # per-slide Dirichlet variational params gamma_m (initialize to alpha0)
    gamma = {}  # slide -> (k,) tensor on DEVICE

    for ep in range(1, epochs+1):
        print(f"   [PACE] epoch {ep}/{epochs}")
        total = 0.0
        for sid in train_ids:
            h5 = FEAT_DIR / f"{sid}.h5"
            if not h5.exists(): continue
            if sid not in gamma:
                gamma[sid] = torch.full((k,), float(alpha0), device=DEVICE)
            for off, raw in iter_h5_blocks(h5):
                H_np, a_np = h_and_alpha_block(raw, clam)
                H = torch.from_numpy(H_np).to(DEVICE)
                a = torch.from_numpy(a_np).to(DEVICE)
                pi, mu, var = params()

                # amortized q(z|h) logits
                q_logits = expl(H)                              # (m,k)
                logq = F.log_softmax(q_logits, dim=1)          # (m,k)
                q = logq.exp()

                # generative likelihood + prior
                logp_h_z = log_gauss_diag(H, mu, var)          # (m,k)
                logp_z   = torch.log(pi + 1e-12).unsqueeze(0)  # (1,k)

                # per-slide mixture term via Dirichlet q(theta_m) ~ Dir(gamma_m)
                psi = torch.digamma(gamma[sid])
                psi_sum = torch.digamma(gamma[sid].sum())
                log_theta_exp = (psi - psi_sum).unsqueeze(0)   # (1,k)

                # posterior weight for responsibilities (no closed form update here; gradient-based)
                # ELBO_i = E_q [ log p(h|z) + log p(z) + log theta_k - log q(z|h) ]
                elbo_i = (q * (logp_h_z + logp_z + log_theta_exp - logq)).sum(1)  # (m,)

                # attention-weighted mean loss over patches
                loss = -(a * elbo_i).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(expl.parameters())+list(params.parameters()), 5.0)
                opt.step()
                total += float(loss.item())

                # update gamma_m with attention-weighted expected counts
                with torch.no_grad():
                    Nk = (a.unsqueeze(1) * q).sum(0)           # (k,)
                    gamma[sid] = alpha0 + 0.9*gamma[sid] + 0.1*Nk  # mild momentum to stabilize passes

                del H_np, a_np, H, a, logq, q, q_logits, logp_h_z, logp_z, log_theta_exp, elbo_i
                gc.collect()
        print(f"      loss≈ {total/ max(1,len(train_ids)):.4f}")

    # finalize parameters
    with torch.no_grad():
        pi, mu, var = params()
    return expl.cpu(), (
        pi.detach().cpu().numpy(),
        mu.detach().cpu().numpy(),
        var.detach().cpu().numpy()
    ), {k: v.detach().cpu().numpy() for k, v in gamma.items()}

@torch.inference_mode()
def export_r_pace_all_slides(expl: Explainer, params_pack, gamma_map, out_dir: Path, clam: CLAM_MB):
    pi_np, mu_np, var_np = params_pack
    pi = torch.from_numpy(pi_np).to(DEVICE)
    mu = torch.from_numpy(mu_np).to(DEVICE)
    var = torch.from_numpy(var_np).to(DEVICE)

    shard_dir = out_dir / "r_shards"; shard_dir.mkdir(parents=True, exist_ok=True)
    for h5 in sorted(FEAT_DIR.glob("*.h5")):
        slide = h5.stem
        with h5py.File(h5, "r") as f:
            feats = f["features"]; n = feats.shape[0]
            # get gamma for this slide (posterior if trained; else prior)
            gamma = torch.from_numpy(gamma_map.get(slide, np.full((K,), ALPHA0, np.float32))).to(DEVICE)
            psi = torch.digamma(gamma); psi_sum = torch.digamma(gamma.sum())
            log_theta_exp = (psi - psi_sum).unsqueeze(0)               # (1,k)
            for off in range(0, n, BATCH_SIZE):
                raw = feats[off:off+BATCH_SIZE][:].astype(np.float32)
                H_np, _ = h_and_alpha_block(raw, clam)
                H = torch.from_numpy(H_np).to(DEVICE)
                # amortized logits
                q_logits = expl.to(DEVICE)(H)                           # (m,k)
                # generative terms
                logp_h_z = log_gauss_diag(H, mu, var)                  # (m,k)
                logp_z   = torch.log(pi + 1e-12).unsqueeze(0)
                post_logits = logp_h_z + logp_z + log_theta_exp + q_logits*0.0  # if you want, add small blend of q_logits
                post_logits = post_logits - post_logits.max(1,True).values
                R = torch.softmax(post_logits, dim=1).cpu().numpy().astype(np.float16)
                _save_shards_from_R(R, slide, 1, off, shard_dir)
                del H_np, H, q_logits, logp_h_z, logp_z, post_logits, R
        gc.collect()
    return shard_dir

# ────────────── save helpers ──────────────
def save_kmeans(km, out_dir: Path, tag: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(km, out_dir / f"{DATASET}_k{K}_{tag}.joblib")
    np.save(out_dir / f"{DATASET}_centers_k{K}_{tag}.npy", km.cluster_centers_)
    print(f"   ↳ saved {tag} at {out_dir}")

def save_centers(npy: np.ndarray, out_dir: Path, tag: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{DATASET}_centers_k{K}_{tag}.npy", npy)
    print(f"   ↳ saved centers ({tag}) at {out_dir}")

# ────────────── main loop ──────────────
if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    for split_idx in range(10):
        # if split_idx <= 7:
        #     print(f"[skip] split {split_idx}, already trained")
        #     continue
        print(f"\n=== Split s_{split_idx} (k={K}, RAW h) ===")
        ckpt = RESULTS_DIR / f"s_{split_idx}_checkpoint.pt"
        split_csv = RESULTS_DIR / f"splits_{split_idx}.csv"
        if not (ckpt.exists() and split_csv.exists()):
            print(f"[skip] missing files for split {split_idx}")
            continue

        split_out = OUT_BASE / f"s_{split_idx}"
        # Method-specific subfolders
        p_km_unw   = split_out / "kmeans_rawh"
        p_km_aw    = split_out / "kmeans_awkm_rawh"
        p_softem   = split_out / "soft_awem_rawh"
        p_vmf      = split_out / "vmf_rawh"
        p_tmix     = split_out / "tmixture_rawh"
        p_sink     = split_out / "sinkhorn_rawh"
        p_pace     = split_out / "pace_rawh"
        for p in (p_km_unw, p_km_aw, p_softem, p_vmf, p_tmix, p_sink, p_pace):
            p.mkdir(parents=True, exist_ok=True)

        clam = load_clam(ckpt)
        TRAIN_IDS, TEST_IDS = get_train_test_ids(split_csv)
        print(f"   TRAIN={len(TRAIN_IDS)}  TEST={len(TEST_IDS)}")

        # 1) k-means (hard, unweighted)
        print(" -> k-means (hard, unweighted)")
        km_unw = train_kmeans(TRAIN_IDS, clam, K); save_kmeans(km_unw, p_km_unw, "rawh")

        # 2) k-means (hard, attention-weighted)
        print(" -> k-means (hard, attention-weighted)")
        km_aw  = train_aw_kmeans(TRAIN_IDS, clam, K); save_kmeans(km_aw, p_km_aw, "rawh_awkm")

        # 3) soft k-means (attention-weighted EM)
        print(" -> soft k-means (AW-EM)")
        C_soft = train_soft_aw_em(TRAIN_IDS, clam, K, TAU_SOFT, EPOCHS_SOFT, DELTA_TOL)
        save_centers(C_soft, p_softem, "rawh_awsoft_em")
        print("    exporting r_ic…")
        export_r_softEM_all_slides(C_soft, p_softem, clam, TAU_SOFT, ep_tag=1)

        # # 4) vMF (spherical)
        # print(" -> vMF (spherical)")
        # MU_vmf = train_vmf(TRAIN_IDS, clam, K, TAU_VMF, EPOCHS_SOFT)
        # save_centers(MU_vmf, p_vmf, "rawh_vmf")
        # print("    exporting r_ic…")
        # export_r_vmf_all_slides(MU_vmf, p_vmf, clam, TAU_VMF, ep_tag=1)

        # # 5) Student-t mixture
        # print(" -> Student-t mixture (diag)")
        # MU_t, VAR_t, PI_t = train_t_mixture(TRAIN_IDS, clam, K, NU_T, EPOCHS_SOFT)
        # np.save(p_tmix / f"{DATASET}_centers_k{K}_rawh_tmix_mu.npy", MU_t)
        # np.save(p_tmix / f"{DATASET}_centers_k{K}_rawh_tmix_var.npy", VAR_t)
        # np.save(p_tmix / f"{DATASET}_centers_k{K}_rawh_tmix_pi.npy",  PI_t)
        # print("    exporting r_ic…")
        # export_r_tmix_all_slides(MU_t, VAR_t, PI_t, p_tmix, clam, NU_T, ep_tag=1)

        # # 6) Sinkhorn-k-means (balanced)
        # print(" -> Sinkhorn-k-means (balanced)")
        # MU_sk = train_sinkhorn(TRAIN_IDS, clam, K, EPOCHS_SOFT)
        # save_centers(MU_sk, p_sink, "rawh_sinkhorn")
        # print("    exporting r_ic…")
        # export_r_sinkhorn_all_slides(MU_sk, p_sink, clam, ep_tag=1)

        # # 7) PACE (true-ish core: global GMM + amortized q + per-slide Dirichlet)
        # print(" -> PACE (global GMM + amortized q + per-slide Dirichlet)")
        # expl, params_pack, gamma_map = train_pace(TRAIN_IDS, clam, K, PACE_EPOCHS, PACE_LR, ALPHA0)
        # torch.save(expl.state_dict(), p_pace / f"{DATASET}_pace_explainer_k{K}.pt")
        # np.save(p_pace / f"{DATASET}_pace_pi_k{K}.npy",  params_pack[0])
        # np.save(p_pace / f"{DATASET}_pace_mu_k{K}.npy",  params_pack[1])
        # np.save(p_pace / f"{DATASET}_pace_var_k{K}.npy", params_pack[2])
        # with open(p_pace / f"{DATASET}_pace_gamma_k{K}.json", "w") as f:
        #     json.dump({s: v.tolist() for s, v in gamma_map.items()}, f)
        # print("    exporting r_ic…")
        # export_r_pace_all_slides(expl, params_pack, gamma_map, p_pace, clam)

        # manifest
        manifest = {
            "split": split_idx,
            "checkpoint": str(ckpt),
            "split_csv": str(split_csv),
            "outputs": {
                "kmeans_rawh": {
                    "model": str((p_km_unw / f"{DATASET}_k{K}_rawh.joblib").resolve()),
                    "centers": str((p_km_unw / f"{DATASET}_centers_k{K}_rawh.npy").resolve())
                },
                "kmeans_awkm_rawh": {
                    "model": str((p_km_aw  / f"{DATASET}_k{K}_rawh_awkm.joblib").resolve()),
                    "centers": str((p_km_aw / f"{DATASET}_centers_k{K}_rawh_awkm.npy").resolve())
                },
                "soft_awem_rawh": {
                    "centers": str((p_softem / f"{DATASET}_centers_k{K}_rawh_awsoft_em.npy").resolve()),
                    "r_ic_dir": str((p_softem / "r_shards").resolve())
                },
                # "vmf_rawh": {
                #     "centers": str((p_vmf / f"{DATASET}_centers_k{K}_rawh_vmf.npy").resolve()),
                #     "r_ic_dir": str((p_vmf / "r_shards").resolve())
                # },
                # "tmixture_rawh": {
                #     "mu":  str((p_tmix / f"{DATASET}_centers_k{K}_rawh_tmix_mu.npy").resolve()),
                #     "var": str((p_tmix / f"{DATASET}_centers_k{K}_rawh_tmix_var.npy").resolve()),
                #     "pi":  str((p_tmix / f"{DATASET}_centers_k{K}_rawh_tmix_pi.npy").resolve()),
                #     "r_ic_dir": str((p_tmix / "r_shards").resolve())
                # },
                # "sinkhorn_rawh": {
                #     "centers": str((p_sink / f"{DATASET}_centers_k{K}_rawh_sinkhorn.npy").resolve()),
                #     "r_ic_dir": str((p_sink / "r_shards").resolve())
                # },
                # "pace_rawh": {
                #     "explainer": str((p_pace / f"{DATASET}_pace_explainer_k{K}.pt").resolve()),
                #     "pi":  str((p_pace / f"{DATASET}_pace_pi_k{K}.npy").resolve()),
                #     "mu":  str((p_pace / f"{DATASET}_pace_mu_k{K}.npy").resolve()),
                #     "var": str((p_pace / f"{DATASET}_pace_var_k{K}.npy").resolve()),
                #     "gamma_json": str((p_pace / f"{DATASET}_pace_gamma_k{K}.json").resolve()),
                #     "r_ic_dir": str((p_pace / "r_shards").resolve())
                # }
            }
        }
        with open(split_out / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        print(f" ✓ manifest → {split_out/'manifest.json'}")

        # cleanup
        # del clam, km_unw, km_aw, C_soft, MU_vmf, MU_t, VAR_t, PI_t, MU_sk, expl, params_pack, gamma_map
        gc.collect(); torch.cuda.empty_cache()

    print(f"\n✓ All splits trained. See {OUT_BASE}")