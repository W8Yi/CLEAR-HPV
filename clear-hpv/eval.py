#!/usr/bin/env python3
"""
Unified evaluator for hpv_uni2 (k=10, raw h) across all splits and methods.

This version auto-discovers and evaluates the RENAMED method folders:
  - encoder           -> HARD (KMeans joblib inside)
  - CLEAR-hpv_rawh    -> HARD (KMeans joblib inside)
  - CLEAR-hpv_awh     -> HARD (KMeans joblib inside)
  - Dirichlet         -> SOFT (expects Dirichlet/r_shards/*.npy)

It parses folder names under each split directory (s_0, s_1, ...), selects the
appropriate evaluation pipeline (hard vs soft), and runs identical thresholding
and metrics for all methods.

Source base script: :contentReference[oaicite:0]{index=0}
"""

from pathlib import Path
import re, gc, json
import numpy as np
import pandas as pd
import h5py, torch, joblib

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
)
from sklearn.metrics import pairwise_distances_argmin

# ───────────────────────── configuration ─────────────────────────
DATASET      = "TCGA_HNSCC"
K            = 10
TARGET_REC   = 0.80                 # train recall floor when tuning threshold (unused if objective tuning)
BATCH        = 16_384
DEVICE       = "cuda:2" if torch.cuda.is_available() else "cpu"

# Big training root produced by your unified trainer:
ALL_MODELS_ROOT = Path("/common/users/wq50/CLEAR-HPV/checkpoints")

# CLAM/splits/labels for attention + split IDs:
RESULTS_DIR  = Path("/common/users/wq50/CLAM/results/HPV_CLAM_50_mb_s1")
CSV_LABEL    = Path("/common/users/wq50/CLAM/dataset_csv/HNSCC.csv")
FEAT_DIR     = Path("/common/users/wq50/CLAM/features/HPV_UNI2_features/h5_files")
SPLIT_DIR    = Path("/common/users/wq50/CLAM/splits/HPV_100")

# Which label to evaluate (binary)
# LABEL_COL = "survival"
# POS_NAME  = "Survived"

LABEL_COL = "hpv_status"
POS_NAME  = "HPV+"

# Output directory for this evaluation run
OUT_DIR = Path("./CLEAR_result/eval_TCGA_hpv")
OUT_DIR.mkdir(exist_ok=True, parents=True)

# ───────────────────────── CLAM to get (h, attention) ─────────────────────────
from models.model_clam import CLAM_MB

# Cache CLAM models by their checkpoint path
_CLAM_CACHE = {}  # {Path: torch.nn.Module}

import os, random
SEED = 66

def seed_everything(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # deterministic kernels (may slow a bit)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(SEED)

@torch.inference_mode()
def get_clam_for_weight(ckpt_path: Path):
    """
    Load (or fetch from cache) a CLAM_MB model for the given checkpoint path.
    """
    if ckpt_path not in _CLAM_CACHE:
        if not ckpt_path.exists():
            raise FileNotFoundError(f"CLAM weight not found: {ckpt_path}")
        m = CLAM_MB(gate=True, size_arg="small", n_classes=2, embed_dim=1536)
        state = torch.load(ckpt_path, map_location=DEVICE)
        m.load_state_dict(state, strict=False)
        _CLAM_CACHE[ckpt_path] = m.to(DEVICE).eval()
    return _CLAM_CACHE[ckpt_path]

@torch.inference_mode()
def h_and_att_block(raw_np: np.ndarray, clam):
    """
    raw_np: (m,1536) float32
    Returns:
      h: (m,512) float32
      a: (m,)    float32 attention; instance-softmax within block; mean≈1
    """
    x = torch.from_numpy(raw_np).to(DEVICE)
    A_raw, h = clam.attention_net(x)
    a = torch.softmax(A_raw[:, 0], dim=0).clamp_min(1e-12)
    a = a * (a.numel() / a.sum())
    return h.cpu().numpy().astype(np.float32), a.cpu().numpy().astype(np.float32)

def iter_h5_blocks(h5_path: Path, batch=BATCH):
    with h5py.File(h5_path, "r") as f:
        feats = f["features"]
        n = feats.shape[0]
        for off in range(0, n, batch):
            yield off, feats[off:off + batch][:].astype(np.float32)

# ───────────────────────── split helpers ─────────────────────────
def load_split_ids(split_idx: int):
    csv = RESULTS_DIR / f"splits_{split_idx}.csv"
    if not csv.exists():
        return [], []
    df = pd.read_csv(csv, index_col=0)
    train_ids = [s for s in df["train"].dropna().astype(str)]
    test_ids  = [s for s in df["test"].dropna().astype(str)]
    return train_ids, test_ids

# ───────────────────────── SOFT shard utilities ─────────────────────────
SOFT_RE = re.compile(r"^(?P<slide>.+?)_ep(?P<ep>\d+)_b(?P<off>\d+)_soft\.npy$")

def shards_for_slide(shard_dir: Path, slide: str):
    """Return list[(off:int, path:Path)] for LAST epoch of this slide."""
    by_ep = {}
    for f in shard_dir.glob(f"{slide}_ep*_b*_soft.npy"):
        m = SOFT_RE.match(f.name)
        if not m:
            continue
        ep = int(m["ep"])
        off = int(m["off"])
        by_ep.setdefault(ep, []).append((off, f))
    if not by_ep:
        return []
    last_ep = max(by_ep)
    return sorted(by_ep[last_ep], key=lambda t: t[0])

# ───────────────────────── cluster→label mapping ─────────────────────────
def train_cluster_map_SOFT(shard_dir: Path, train_ids, label_df, clam):
    """
    Attention-weighted masses on TRAIN slides only:
      mass_pos[c] = Σ_{slide∈train,pos} Σ_{i} α_i r_{ic}
      mass_neg[c] = Σ_{slide∈train,neg} Σ_{i} α_i r_{ic}
    pos_mask[c] = (mass_pos[c] >= mass_neg[c])
    """
    Kc = None
    pos_mass = None
    neg_mass = None
    for sid in train_ids:
        shards = shards_for_slide(shard_dir, sid)
        if not shards:
            continue
        for off, path in shards:
            R = np.load(path).astype(np.float32)   # (m,K)
            if Kc is None:
                Kc = R.shape[1]
                pos_mass = np.zeros(Kc, np.float64)
                neg_mass = np.zeros(Kc, np.float64)

            h5 = FEAT_DIR / f"{sid}.h5"
            with h5py.File(h5, "r") as f:
                raw = f["features"][off:off + R.shape[0]].astype(np.float32)

            _, a = h_and_att_block(raw, clam)
            wR = (a[:, None] * R).sum(0)

            if str(label_df.at[sid, LABEL_COL]) == POS_NAME:
                pos_mass += wR
            else:
                neg_mass += wR

            del R, raw, a
            gc.collect()

    if Kc is None:
        raise RuntimeError("No shards found on train")
    return (pos_mass >= neg_mass), Kc

def train_cluster_map_HARD(km, train_ids, label_df, clam):
    """
    Hard assignments + attention weights on TRAIN slides.
    """
    Kc = km.n_clusters
    pos_mass = np.zeros(Kc, np.float64)
    neg_mass = np.zeros(Kc, np.float64)

    for sid in train_ids:
        h5 = FEAT_DIR / f"{sid}.h5"
        if not h5.exists():
            continue
        for _, raw in iter_h5_blocks(h5):
            h, a = h_and_att_block(raw, clam)
            cl = pairwise_distances_argmin(h, km.cluster_centers_)
            counts = np.bincount(cl, weights=a, minlength=Kc)

            if str(label_df.at[sid, LABEL_COL]) == POS_NAME:
                pos_mass += counts
            else:
                neg_mass += counts

            del raw, h, a
            gc.collect()

    return (pos_mass >= neg_mass), Kc

# ───────────────────────── Slide score (FM-style) ─────────────────────────
def slide_score_SOFT(shard_dir: Path, slide: str, pos_mask, clam):
    shards = shards_for_slide(shard_dir, slide)
    if not shards:
        return None

    num = None
    den = 0.0
    for off, path in shards:
        R = np.load(path).astype(np.float32)     # (m,K)

        h5 = FEAT_DIR / f"{slide}.h5"
        with h5py.File(h5, "r") as f:
            raw = f["features"][off:off + R.shape[0]].astype(np.float32)

        _, a = h_and_att_block(raw, clam)
        comp = (a[:, None] * R).sum(0)
        w = float(a.sum())

        num = comp if num is None else num + comp
        den += w

        del R, raw, a
        gc.collect()

    comp_slide = num / max(den, 1e-12)
    return float(comp_slide[pos_mask].sum())

def slide_score_HARD(km, slide: str, pos_mask, clam):
    h5 = FEAT_DIR / f"{slide}.h5"
    if not h5.exists():
        return None

    Kc = km.n_clusters
    num = np.zeros(Kc, np.float64)
    den = 0.0

    for _, raw in iter_h5_blocks(h5):
        h, a = h_and_att_block(raw, clam)
        cl = pairwise_distances_argmin(h, km.cluster_centers_)
        num += np.bincount(cl, weights=a, minlength=Kc)
        den += float(a.sum())

        del raw, h, a
        gc.collect()

    comp_slide = num / max(den, 1e-12)
    return float(comp_slide[pos_mask].sum())

# ───────────────────────── Threshold tuning ─────────────────────────
def tune_threshold(scores_train, y_train, target_recall=0.80):
    """
    Pick θ on TRAIN:
      feasible set S = {θ | recall(θ) ≥ target_recall}
      choose θ ∈ S maximizing estimated F1
    fallback θ = 0.5 if no feasible point
    """
    fpr, tpr, thr = roc_curve(y_train, scores_train)
    ok = tpr >= target_recall
    if ok.any():
        prec_est = 1 - fpr
        f1_est = 2 * (prec_est * tpr) / np.maximum(prec_est + tpr, 1e-12)
        idx = np.argmax(f1_est * ok)
        tuned = thr[idx] if idx < len(thr) else 0.5
        return float(tuned if np.isfinite(tuned) else 0.5)
    return 0.5

def tune_threshold_by_objective(scores, y, objective="acc"):
    s = np.asarray(scores, float)
    y = np.asarray(y, int)

    uniq = np.unique(s)
    if len(uniq) == 1:
        return float(uniq[0])

    thr_cand = np.r_[uniq[0] - 1e-6, (uniq[:-1] + uniq[1:]) / 2.0, uniq[-1] + 1e-6]

    best_thr, best_val = 0.5, -np.inf
    for t in thr_cand:
        yhat = (s >= t).astype(int)
        tp = np.sum((y == 1) & (yhat == 1))
        tn = np.sum((y == 0) & (yhat == 0))
        fp = np.sum((y == 0) & (yhat == 1))
        fn = np.sum((y == 1) & (yhat == 0))

        if objective == "acc":
            val = (tp + tn) / max(len(y), 1)
        elif objective == "bal_acc":
            sens = tp / max(tp + fn, 1)
            spec = tn / max(tn + fp, 1)
            val = 0.5 * (sens + spec)
        else:
            raise ValueError(f"unknown objective {objective}")

        if val > best_val:
            best_val, best_thr = val, float(t)

    return best_thr

def _safe_div(num, den):
    den = float(den)
    return float(num) / den if den > 0 else 0.0

def compute_metrics(y_true, scores, thr):
    """
    Adds Spec and confusion counts.
    """
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    y_hat = (scores >= thr).astype(int)

    TP = int(np.sum((y_true == 1) & (y_hat == 1)))
    TN = int(np.sum((y_true == 0) & (y_hat == 0)))
    FP = int(np.sum((y_true == 0) & (y_hat == 1)))
    FN = int(np.sum((y_true == 1) & (y_hat == 0)))

    P = TP + FN
    N = TN + FP

    acc  = accuracy_score(y_true, y_hat)
    f1   = f1_score(y_true, y_hat, zero_division=0)
    prec = precision_score(y_true, y_hat, zero_division=0)
    rec  = recall_score(y_true, y_hat, zero_division=0)
    spec = _safe_div(TN, TN + FP)

    auroc = roc_auc_score(y_true, scores) if len(np.unique(y_true)) == 2 else np.nan

    return dict(
        ACC=acc, F1=f1, Prec=prec, Rec=rec, Spec=spec, NPV=_safe_div(TN, TN + FN),
        AUROC=auroc, thr=float(thr),
        TP=TP, TN=TN, FP=FP, FN=FN, P=P, N=N
    )

# ───────────────────────── method registry (RENAMED folders) ─────────────────────────
HARD_METHODS = {"encoder", "CLEAR-hpv_rawh", "CLEAR-hpv_awh"}
SOFT_METHODS = {"Dirichlet"}

def parse_method_dirname(name: str):
    """
    Map folder name to evaluation type.
    """
    if name in HARD_METHODS:
        return "hard", name
    if name in SOFT_METHODS:
        return "soft", name
    return None, None

def _find_joblib_in_dir(d: Path):
    """
    Find a joblib file in directory d.
    If multiple exist, prefer ones containing '_k{K}_' or 'kmeans' in the name.
    """
    files = list(d.glob("*.joblib"))
    if not files:
        return None
    if len(files) == 1:
        return files[0]

    prefer = [f for f in files if (f"_k{K}_" in f.name) or ("kmeans" in f.name.lower())]
    return prefer[0] if prefer else files[0]

def method_specs_for_split(split_dir: Path):
    """
    Returns dict {method_name: dict(type='soft'|'hard', resource=Path)}
      - HARD: resource = path to *.joblib inside the method folder
      - SOFT: resource = path to r_shards/ directory
    Auto-discovers by folder name under split_dir.
    """
    specs = {}

    # Only look at immediate child directories under s_i/
    for d in sorted([p for p in split_dir.iterdir() if p.is_dir()]):
        mtype, disp = parse_method_dirname(d.name)
        if mtype is None:
            continue

        if mtype == "hard":
            jm = _find_joblib_in_dir(d)
            if jm and jm.exists():
                specs[disp] = dict(type="hard", resource=jm)

        elif mtype == "soft":
            rdir = d / "r_shards"
            if rdir.exists() and any(rdir.glob("*.npy")):
                specs[disp] = dict(type="soft", resource=rdir)

    return specs

# ───────────────────────── main ─────────────────────────
if __name__ == "__main__":
    # do NOT override seed
    np.random.seed(SEED)

    # labels
    label_df = pd.read_csv(CSV_LABEL).set_index("slide_id")
    if LABEL_COL not in label_df.columns:
        raise KeyError(f"Label column '{LABEL_COL}' not in {CSV_LABEL}")

    rows = []

    # iterate splits and auto-pick models for that split
    for split_dir in sorted(ALL_MODELS_ROOT.glob("s_*")):
        try:
            split_idx = int(split_dir.name.split("_")[1])
        except Exception:
            print(f"[skip] unknown split folder name: {split_dir.name}")
            continue

        # split-specific CLAM weight
        split_ckpt = RESULTS_DIR / f"s_{split_idx}_checkpoint.pt"
        try:
            clam = get_clam_for_weight(split_ckpt)
        except FileNotFoundError as e:
            print(f"[skip] split {split_idx}: {e}")
            continue

        TRAIN_IDS, TEST_IDS = load_split_ids(split_idx)
        if not TRAIN_IDS or not TEST_IDS:
            print(f"[skip] split {split_idx}: missing split CSV ids")
            continue

        specs = method_specs_for_split(split_dir)

        if not specs:
            print(f"[skip] split {split_idx}: no recognized models found in {split_dir}")
            continue

        print(f"\n=== Split {split_idx} ===  methods={list(specs.keys())}")

        for method, info in specs.items():
            try:
                # 1) Build cluster→label map on TRAIN
                if info["type"] == "hard":
                    km = joblib.load(info["resource"])
                    pos_mask, _ = train_cluster_map_HARD(km, TRAIN_IDS, label_df, clam)

                    # 2) TRAIN scores
                    y_tr, s_tr = [], []
                    for sid in TRAIN_IDS:
                        sc = slide_score_HARD(km, sid, pos_mask, clam)
                        if sc is None:
                            continue
                        y_tr.append(int(str(label_df.at[sid, LABEL_COL]) == POS_NAME))
                        s_tr.append(sc)

                else:
                    rdir = info["resource"]
                    pos_mask, _ = train_cluster_map_SOFT(rdir, TRAIN_IDS, label_df, clam)

                    # 2) TRAIN scores
                    y_tr, s_tr = [], []
                    for sid in TRAIN_IDS:
                        sc = slide_score_SOFT(rdir, sid, pos_mask, clam)
                        if sc is None:
                            continue
                        y_tr.append(int(str(label_df.at[sid, LABEL_COL]) == POS_NAME))
                        s_tr.append(sc)

                y_tr = np.array(y_tr, int)
                s_tr = np.array(s_tr, float)
                if len(y_tr) == 0 or len(np.unique(y_tr)) < 2:
                    print(f"[skip] {method} split {split_idx}: insufficient train labels")
                    continue

                # 3) tune threshold on TRAIN (your current choice: objective="acc")
                thr = tune_threshold(s_tr, y_tr, target_recall=TARGET_REC)

                # 4) TEST scores
                if info["type"] == "hard":
                    y_te, s_te = [], []
                    for sid in TEST_IDS:
                        sc = slide_score_HARD(km, sid, pos_mask, clam)
                        if sc is None:
                            continue
                        y_te.append(int(str(label_df.at[sid, LABEL_COL]) == POS_NAME))
                        s_te.append(sc)
                else:
                    y_te, s_te = [], []
                    for sid in TEST_IDS:
                        sc = slide_score_SOFT(rdir, sid, pos_mask, clam)
                        if sc is None:
                            continue
                        y_te.append(int(str(label_df.at[sid, LABEL_COL]) == POS_NAME))
                        s_te.append(sc)

                y_te = np.array(y_te, int)
                s_te = np.array(s_te, float)
                if len(y_te) == 0 or len(np.unique(y_te)) < 2:
                    print(f"[skip] {method} split {split_idx}: insufficient test data")
                    continue

                met = compute_metrics(y_te, s_te, thr)
                rows.append(dict(method=method, split=split_idx, **met))

                print(
                    f"{method:20s}  thr={met['thr']:.3f}  "
                    f"ACC={met['ACC']:.3f}  Rec/Sens={met['Rec']:.3f}  "
                    f"Spec={met['Spec']:.3f}  Prec={met['Prec']:.3f}  "
                    f"AUROC={met['AUROC']:.3f}  TP={met['TP']} TN={met['TN']} FP={met['FP']} FN={met['FN']}"
                )

            except Exception as e:
                print(f"[error] {method} split {split_idx}: {e}")

    # save per-split table (includes TP,TN,FP,FN,P,N,Spec,NPV)
    df = pd.DataFrame(rows)
    per_split_csv = OUT_DIR / "per_split.csv"
    df.to_csv(per_split_csv, index=False)
    print("\nPer-split metrics →", per_split_csv)

    # save summary (mean ± std per method) including Spec and NPV
    if not df.empty:
        summary = (
            df.groupby("method")[["ACC", "F1", "Prec", "Rec", "Spec", "NPV", "AUROC"]]
              .agg(["mean", "std"])
              .sort_index()
        )
        summary_csv = OUT_DIR / "summary.csv"
        summary.to_csv(summary_csv)
        print("Summary (mean±std) →", summary_csv)
        print(summary)
    else:
        print("No results produced. Check folders and splits.")
