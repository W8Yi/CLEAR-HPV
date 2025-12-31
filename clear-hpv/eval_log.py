#!/usr/bin/env python3
"""
Logistic-regression evaluator for hpv_uni2 (k=10, raw h) across all splits.

This version:
- Only uses the two HARD k-means methods on CLAM h-space:
    * kmeans_rawh
    * kmeans_awkm_rawh
- For each method and split:
    1) Uses CLAM to project tiles to h-space and get attention weights.
    2) Assigns tiles to clusters via the fitted KMeans.
    3) Builds slide-level attention-weighted cluster fractions (K-dim features).
    4) Trains a logistic regression on TRAIN slide features.
    5) Gets TRAIN probabilities and tunes a decision threshold θ
       with a recall floor TARGET_REC using ROC-based tuning.
    6) Applies the model and θ to TEST and computes full metrics
       (ACC, F1, Prec, Rec, Spec, NPV, AUROC + TP/TN/FP/FN/P/N).
- Evaluates both survival and HPV endpoints from the same CSV.
"""

from pathlib import Path
import os, re, gc, json, random
import numpy as np
import pandas as pd
import h5py, torch, joblib

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
)
from sklearn.metrics import pairwise_distances_argmin
from sklearn.linear_model import LogisticRegression

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ───────────────────────── configuration ─────────────────────────
DATASET      = "TCGA_HNSCC"
K            = 10
TARGET_REC   = 0.80                 # train recall floor when tuning threshold
BATCH        = 16_384
DEVICE       = "cuda:2" if torch.cuda.is_available() else "cpu"

# Big training root produced by your unified trainer:
ALL_MODELS_ROOT = Path("./") # change to your ALL_MODELS_ROOTc:\Users\weiyi\Downloads\train_others.py

# CLAM/splits/labels for attention + split IDs:
RESULTS_DIR  = Path("/common/users/wq50/CLAM/results/HPV_CLAM_50_mb_s1")
CSV_LABEL    = Path("/common/users/wq50/CLAM/dataset_csv/HNSCC.csv")
FEAT_DIR     = Path("/common/users/wq50/CLAM/features/HPV_UNI2_features/h5_files")
SPLIT_DIR    = Path("/common/users/wq50/CLAM/splits/HPV_100")

# Two tasks: survival and HPV
TASKS = [
    dict(
        name      = "survival",
        label_col = "survival",
        pos_name  = "Survived",
        out_dir   = Path("./uni2_k10_logreg/eval_TCGA_surv_test"),
    ),
    dict(
        name      = "hpv",
        label_col = "hpv_status",
        pos_name  = "HPV+",
        out_dir   = Path("./uni2_k10_logreg/eval_TCGA_hpv_test"),
    ),
]

# label globals (overwritten per task)
LABEL_COL = None
POS_NAME  = None

# ───────────────────────── CLAM to get (h, attention) ─────────────────────────
from models.model_clam import CLAM_MB

_CLAM_CACHE = {}  # {Path: torch.nn.Module}

SEED = 66
def seed_everything(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
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
        feats = f["features"]; n = feats.shape[0]
        for off in range(0, n, batch):
            yield off, feats[off:off+batch][:].astype(np.float32)

# ───────────────────────── split helpers ─────────────────────────
def load_split_ids(split_idx: int):
    csv = RESULTS_DIR / f"splits_{split_idx}.csv"
    if not csv.exists():
        return [], []
    df = pd.read_csv(csv, index_col=0)
    train_ids = [s for s in df["train"].dropna().astype(str)]
    test_ids  = [s for s in df["test"].dropna().astype(str)]
    return train_ids, test_ids

# ───────────────────────── Slide features: cluster fractions ─────────────────────────
def slide_fractions_HARD(km, slide: str, clam):
    """
    Build attention-weighted cluster fractions for a slide:

    For each tile:
      - Project to h-space via CLAM.
      - Compute attention weight a_i.
      - Assign cluster c_i via KMeans on h.
    Then:
      counts[c] = Σ_i (a_i * 1_{c_i = c})
      den = Σ_i a_i
      fractions[c] = counts[c] / den

    Returns:
      frac: (K,) float64 array summing to ~1, or None if no features.
    """
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
        del raw, h, a; gc.collect()

    if den <= 0:
        return None
    return (num / den).astype(np.float32)

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
    s = np.asarray(scores, float); y = np.asarray(y, int)
    uniq = np.unique(s)
    if len(uniq) == 1:
        return float(uniq[0])
    thr_cand = np.r_[uniq[0]-1e-6, (uniq[:-1] + uniq[1:]) / 2.0, uniq[-1]+1e-6]

    best_thr, best_val = 0.5, -np.inf
    for t in thr_cand:
        yhat = (s >= t).astype(int)
        tp = np.sum((y==1)&(yhat==1)); tn = np.sum((y==0)&(yhat==0))
        fp = np.sum((y==0)&(yhat==1)); fn = np.sum((y==1)&(yhat==0))
        if objective == "acc":
            val = (tp + tn) / max(len(y), 1)
        elif objective == "bal_acc":
            sens = tp / max(tp+fn, 1); spec = tn / max(tn+fp, 1)
            val = 0.5*(sens+spec)
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

    P = TP + FN  # positives in truth
    N = TN + FP  # negatives in truth

    acc  = accuracy_score(y_true, y_hat)
    f1   = f1_score(y_true, y_hat, zero_division=0)
    prec = precision_score(y_true, y_hat, zero_division=0)
    rec  = recall_score(y_true, y_hat, zero_division=0)
    spec = _safe_div(TN, TN + FP)

    auroc = roc_auc_score(y_true, scores) if len(np.unique(y_true)) == 2 else np.nan

    return dict(
        ACC=acc, F1=f1, Prec=prec, Rec=rec, Spec=spec, NPV=_safe_div(TN, TN+FN),
        AUROC=auroc, thr=float(thr),
        TP=TP, TN=TN, FP=FP, FN=FN, P=P, N=N
    )

# ───────────────────────── method registry ─────────────────────────
def method_specs_for_split(split_dir: Path):
    """
    Returns a dict {method_name: dict(type='hard', resource=Path)}
      - Only HARD k-means models on raw h:
          * kmeans_rawh
          * kmeans_awkm_rawh
    """
    specs = {}

    # k-means on raw h (unweighted)
    km_unw = next((split_dir/"kmeans_rawh").glob(f"*_k{K}_rawh.joblib"), None)
    if km_unw and km_unw.exists():
        specs["kmeans_rawh"] = dict(type="hard", resource=km_unw)

    # attention-weighted k-means on raw h
    km_aw  = next((split_dir/"kmeans_awkm_rawh").glob(f"*_k{K}_rawh_awkm.joblib"), None)
    if km_aw and km_aw.exists():
        specs["kmeans_awkm_rawh"] = dict(type="hard", resource=km_aw)

    return specs

# ───────────────────────── main ─────────────────────────
if __name__ == "__main__":
    np.random.seed(0)

    # labels (shared across tasks)
    label_df_all = pd.read_csv(CSV_LABEL).set_index("slide_id")

    for task in TASKS:
        LABEL_COL = task["label_col"]
        POS_NAME  = task["pos_name"]
        OUT_DIR   = task["out_dir"]
        OUT_DIR.mkdir(exist_ok=True, parents=True)

        if LABEL_COL not in label_df_all.columns:
            print(f"[skip task {task['name']}] label column '{LABEL_COL}' not in {CSV_LABEL}")
            continue

        label_df = label_df_all.copy()

        print(f"\n================ Task: {task['name']} (label={LABEL_COL}, pos={POS_NAME}) ================")

        rows = []

        # iterate splits and auto-pick models for that split
        for split_dir in sorted(ALL_MODELS_ROOT.glob("s_*")):
            try:
                split_idx = int(split_dir.name.split("_")[1])
            except Exception:
                print(f"[skip] unknown split folder name: {split_dir.name}")
                continue

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
                print(f"[skip] split {split_idx}: no k-means models found in {split_dir}")
                continue

            print(f"\n=== Split {split_idx} ===  methods={list(specs.keys())}")

            for method, info in specs.items():
                try:
                    km = joblib.load(info["resource"])

                    # 1) TRAIN features (cluster fractions) + labels
                    X_tr, y_tr = [], []
                    for sid in TRAIN_IDS:
                        frac = slide_fractions_HARD(km, sid, clam)
                        if frac is None:
                            continue
                        X_tr.append(frac)
                        y_tr.append(int(str(label_df.at[sid, LABEL_COL]) == POS_NAME))

                    if not X_tr or len(set(y_tr)) < 2:
                        print(f"[skip] {method} split {split_idx}: insufficient train data/labels")
                        continue

                    X_tr = np.vstack(X_tr).astype(np.float32)
                    y_tr = np.asarray(y_tr, dtype=int)

                    # 2) Train logistic regression on TRAIN
                    logreg = LogisticRegression(
                        max_iter=1000,
                        solver="lbfgs"
                    )
                    logreg.fit(X_tr, y_tr)

                    # 3) TRAIN scores (probabilities for positive class)
                    s_tr = logreg.predict_proba(X_tr)[:, 1]

                    # 4) tune θ with recall floor
                    thr = tune_threshold(s_tr, y_tr, target_recall=TARGET_REC)
                    # alt: thr = tune_threshold_by_objective(s_tr, y_tr, objective="acc")

                    # 5) TEST features + labels + scores
                    X_te, y_te = [], []
                    for sid in TEST_IDS:
                        frac = slide_fractions_HARD(km, sid, clam)
                        if frac is None:
                            continue
                        X_te.append(frac)
                        y_te.append(int(str(label_df.at[sid, LABEL_COL]) == POS_NAME))

                    if not X_te or len(set(y_te)) < 2:
                        print(f"[skip] {method} split {split_idx}: insufficient test data/labels")
                        continue

                    X_te = np.vstack(X_te).astype(np.float32)
                    y_te = np.asarray(y_te, dtype=int)
                    s_te = logreg.predict_proba(X_te)[:, 1]

                    # 6) metrics on TEST
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
                df.groupby("method")[["ACC","F1","Prec","Rec","Spec","NPV","AUROC"]]
                  .agg(["mean","std"])
                  .sort_index()
            )
            summary_csv = OUT_DIR / "summary.csv"
            summary.to_csv(summary_csv)
            print("Summary (mean±std) →", summary_csv)
            print(summary)
        else:
            print("No results produced for this task. Check folders and splits.")