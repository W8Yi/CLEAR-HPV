#!/usr/bin/env python3
"""
Cross-dataset HARD-only evaluator:
- Source (UNI2): learn cluster→label mask and tune threshold.
- Target (CPTAC): score with fixed mask+threshold.
- Methods: kmeans_rawh, kmeans_awkm_rawh only.

Outputs:
  eval_all_models_hard_cptac/per_split.csv
  eval_all_models_hard_cptac/summary.csv

# New:
Additionally saves per-split target scores/labels and ROC/PR curve points:
  eval_all_models_hard_cptac/curves/{method}/split_{i}_target_scores.csv
  eval_all_models_hard_cptac/curves/{method}/split_{i}_target_roc.csv
  eval_all_models_hard_cptac/curves/{method}/split_{i}_target_pr.csv
"""

from pathlib import Path
import gc, re, json
import numpy as np
import pandas as pd
import h5py, torch, joblib
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve, precision_recall_curve
)
from sklearn.metrics import pairwise_distances_argmin

# ───────── config ─────────
K            = 10
TARGET_REC   = 0.80
BATCH        = 16_384
DEVICE       = "cuda:1" if torch.cuda.is_available() else "cpu"

# models root containing per-split folders s_0, s_1, ...
ALL_MODELS_ROOT = Path("./all_models_hpv_uni2_k10")

# source = UNI2 (training/calibration)
SRC_RESULTS_DIR = Path("/common/users/wq50/CLAM/results/HPV_CLAM_50_mb_s1")
SRC_SPLITS_DIR  = Path("/common/users/wq50/CLAM/splits/HPV_100")
SRC_CSV_LABEL   = Path("/common/users/wq50/CLAM2/dataset_csv/HNSCC.csv")
SRC_FEAT_DIR    = Path("/common/users/wq50/CLAM/features/HPV_UNI2_features/h5_files")

# target = CPTAC (testing)
# TGT_CSV_LABEL   = Path("/common/users/wq50/CLAM/dataset_csv/CPTAC_HNSCC_filtered.csv")
TGT_CSV_LABEL   = Path("/common/users/wq50/CLAM/dataset_csv/CESC_high_conf_HPVs.csv")  # HPV+ vs HPV- only
# TGT_FEAT_DIR    = Path("/common/users/wq50/CLAM/features/CPTAC_HNSCC/h5_files")
TGT_FEAT_DIR    = Path("/common/users/wq50/CLAM/features/HPV_CESC/h5_files")
# TGT_FEAT_DIR    = Path("/common/users/wq50/CLAM/features/HPV_CESC_norm_reinhard/h5_files")
# TGT_FEAT_DIR    = Path("/common/users/wq50/CLAM/features/HPV_CPTAC_norm_reinhard/h5_files")

LABEL_COL = "hpv_status"
POS_NAME  = "HPV+"

OUT_DIR = Path("./uni2_k10_base/eval_CESC2"); OUT_DIR.mkdir(parents=True, exist_ok=True)

# ───────── CLAM backbone to get (h, attention) ─────────
from models.model_clam import CLAM_MB
_CLAM_CACHE = {}

# New: deterministic seeding utilities
import os, random
def seed_everything(seed: int = 1337):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # CUDA determinism (Ampere+)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

@torch.inference_mode()
def get_clam_for_weight(ckpt_path: Path):
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

# ───────── splits and ids ─────────
def load_src_ids(split_idx: int):
    csv = SRC_SPLITS_DIR / f"splits_{split_idx}.csv"
    if not csv.exists(): return []
    df = pd.read_csv(csv, index_col=0)
    ids = [s for s in df["train"].dropna().astype(str)]
    return sorted(ids)  # New: stable order

def load_tgt_ids(tgt_label_df: pd.DataFrame):
    # use all labeled CPTAC slides that have features
    ids = []
    for sid in tgt_label_df.index.astype(str):
        if (TGT_FEAT_DIR / f"{sid}.h5").exists():
            ids.append(sid)
    return sorted(ids)  # New: stable order

# ───────── HARD methods only ─────────
def method_specs_for_split(split_dir: Path):
    specs = {}
    # New: deterministic pick if multiple files match
    cand = sorted((split_dir / "kmeans_rawh").glob(f"*_k{K}_rawh.joblib"))
    if cand:
        specs["kmeans_rawh"] = dict(type="hard", resource=cand[0])
    cand = sorted((split_dir / "kmeans_awkm_rawh").glob(f"*_k{K}_rawh_awkm.joblib"))
    if cand:
        specs["kmeans_awkm_rawh"] = dict(type="hard", resource=cand[0])
    return specs

def train_cluster_map_HARD(km, train_ids, label_df, clam, feat_dir: Path):
    Kc = km.n_clusters
    pos_mass = np.zeros(Kc, np.float64)
    neg_mass = np.zeros(Kc, np.float64)
    for sid in train_ids:
        h5 = feat_dir / f"{sid}.h5"
        if not h5.exists(): continue
        for _, raw in iter_h5_blocks(h5):
            h, a = h_and_att_block(raw, clam)
            cl = pairwise_distances_argmin(h, km.cluster_centers_)
            counts = np.bincount(cl, weights=a, minlength=Kc)
            if str(label_df.at[sid, LABEL_COL]) == POS_NAME:
                pos_mass += counts
            else:
                neg_mass += counts
            del raw, h, a; gc.collect()
    return (pos_mass >= neg_mass), Kc

def slide_score_HARD(km, slide: str, pos_mask, clam, feat_dir: Path):
    h5 = feat_dir / f"{slide}.h5"
    if not h5.exists(): return None
    Kc = km.n_clusters
    num = np.zeros(Kc, np.float64); den = 0.0
    for _, raw in iter_h5_blocks(h5):
        h, a = h_and_att_block(raw, clam)
        cl = pairwise_distances_argmin(h, km.cluster_centers_)
        num += np.bincount(cl, weights=a, minlength=Kc)
        den += float(a.sum())
        del raw, h, a; gc.collect()
    comp_slide = num / max(den, 1e-12)
    return float(comp_slide[pos_mask].sum())

# ───────── thresholding ─────────
def tune_threshold(scores_train, y_train, target_recall=TARGET_REC):
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
        # Degenerate case: all scores identical → any threshold gives same result.
        return float(uniq[0])

    # Candidate thresholds include midpoints + slightly outside range
    thr_cand = np.r_[uniq[0] - 1e-6,
                     (uniq[:-1] + uniq[1:]) / 2.0,
                     uniq[-1] + 1e-6]

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

        elif objective == "f1":
            # Positive-class F1
            prec = tp / max(tp + fp, 1)
            rec  = tp / max(tp + fn, 1)
            if prec + rec == 0:
                val = 0.0
            else:
                val = 2 * (prec * rec) / (prec + rec)

        else:
            raise ValueError(f"unknown objective {objective}")

        if val > best_val:
            best_val = val
            best_thr = float(t)

    return best_thr

def compute_metrics(y_true, scores, thr):
    y_true = np.asarray(y_true, int); scores = np.asarray(scores, float)
    y_hat = (scores >= thr).astype(int)

    tp = int(((y_true == 1) & (y_hat == 1)).sum())
    tn = int(((y_true == 0) & (y_hat == 0)).sum())
    fp = int(((y_true == 0) & (y_hat == 1)).sum())
    fn = int(((y_true == 1) & (y_hat == 0)).sum())

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Sensitivity = Recall
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # Specificity = TNR

    out = dict(
        ACC   = accuracy_score(y_true, y_hat),
        F1    = f1_score(y_true, y_hat, zero_division=0),
        Prec  = precision_score(y_true, y_hat, zero_division=0),
        Rec   = recall_score(y_true, y_hat, zero_division=0),  # alias of Sens
        AUROC = np.nan,
        thr   = float(thr),
        Sens  = float(sens),
        Spec  = float(spec),
        TP    = tp, TN=tn, FP=fp, FN=fn,
    )
    if len(np.unique(y_true)) == 2:
        out["AUROC"] = roc_auc_score(y_true, scores)
    return out

# New: utilities to save scores and curves for later plotting
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True); return p

def save_scores_and_curves(base_dir: Path, method: str, split_idx: int,
                           slide_ids, y_true, scores, prefix="target"):
    """
    Saves:
      - {prefix}_scores.csv: slide_id, y, score
      - {prefix}_roc.csv: fpr, tpr, thr
      - {prefix}_pr.csv: recall, precision, thr
    """
    mdir = ensure_dir(base_dir / "curves" / method)
    # scores
    df_scores = pd.DataFrame({"slide_id": slide_ids, "y": y_true, "score": scores})
    df_scores.to_csv(mdir / f"split_{split_idx}_{prefix}_scores.csv", index=False)

    # ROC
    y = np.asarray(y_true, int); s = np.asarray(scores, float)
    if len(np.unique(y)) == 2:
        fpr, tpr, thr = roc_curve(y, s)
        pd.DataFrame({"fpr": fpr, "tpr": tpr, "thr": thr}).to_csv(
            mdir / f"split_{split_idx}_{prefix}_roc.csv", index=False
        )
        # PR
        prec, rec, thr_pr = precision_recall_curve(y, s)
        # precision_recall_curve returns len(thr_pr) = len(prec)-1 = len(rec)-1
        # Save aligned by trimming last prec/rec to match thr length, or keep full.
        pd.DataFrame({"recall": rec, "precision": prec}).to_csv(
            mdir / f"split_{split_idx}_{prefix}_pr.csv", index=False
        )

# ───────── main ─────────
def main():
    seed_everything(1337)  # New: reproducibility

    # labels
    src_label_df = pd.read_csv(SRC_CSV_LABEL).set_index("slide_id")
    tgt_label_df = pd.read_csv(TGT_CSV_LABEL).set_index("slide_id")
    for df_, name_ in [(src_label_df, "SRC"), (tgt_label_df, "TGT")]:
        if LABEL_COL not in df_.columns:
            raise KeyError(f"{name_} label column '{LABEL_COL}' missing")

    rows = []
    per_split_csv = OUT_DIR / "per_split.csv"

    for split_dir in sorted(ALL_MODELS_ROOT.glob("s_*")):
        try:
            split_idx = int(split_dir.name.split("_")[1])
        except Exception:
            print(f"[skip] unknown split folder name: {split_dir.name}")
            continue

        ckpt = SRC_RESULTS_DIR / f"s_{split_idx}_checkpoint.pt"
        try:
            clam = get_clam_for_weight(ckpt)
        except FileNotFoundError as e:
            print(f"[skip] split {split_idx}: {e}")
            continue

        TRAIN_IDS = load_src_ids(split_idx)
        TEST_IDS  = load_tgt_ids(tgt_label_df)
        if not TRAIN_IDS or not TEST_IDS:
            print(f"[skip] split {split_idx}: missing src or tgt ids")
            continue

        specs = method_specs_for_split(split_dir)
        if not specs:
            print(f"[skip] split {split_idx}: no kmeans models in {split_dir}")
            continue

        print(f"\n=== Split {split_idx} === methods={list(specs.keys())}")

        # New: stable iteration over methods
        for method, info in sorted(specs.items()):
            try:
                km = joblib.load(info["resource"])

                # 1) learn mask on UNI2
                pos_mask, _ = train_cluster_map_HARD(km, TRAIN_IDS, src_label_df, clam, SRC_FEAT_DIR)

                # 2) train scores for threshold on UNI2
                y_tr, s_tr = [], []
                for sid in TRAIN_IDS:
                    sc = slide_score_HARD(km, sid, pos_mask, clam, SRC_FEAT_DIR)
                    if sc is None: continue
                    y_tr.append(int(str(src_label_df.at[sid, LABEL_COL]) == POS_NAME)); s_tr.append(sc)

                uni = np.unique(y_tr)
                if len(uni) < 2:
                    print(f"[skip] {method} split {split_idx}: source labels collapsed")
                    continue

                # Tune threshold on source for maximum ACC (or "bal_acc")
                thr = tune_threshold_by_objective(s_tr, y_tr, objective="f1")

                # 3) test scores on CPTAC
                y_te, s_te, id_te = [], [], []
                for sid in TEST_IDS:
                    sc = slide_score_HARD(km, sid, pos_mask, clam, TGT_FEAT_DIR)
                    if sc is None: continue
                    y_te.append(int(str(tgt_label_df.at[sid, LABEL_COL]) == POS_NAME)); s_te.append(sc); id_te.append(sid)

                if len(y_te) == 0:
                    print(f"[skip] {method} split {split_idx}: no target slides scored")
                    continue

                # Save scores and curves for later plotting
                save_scores_and_curves(OUT_DIR, method, split_idx, id_te, y_te, s_te, prefix="target")

                met = compute_metrics(y_te, s_te, thr)
                rows.append(dict(method=method, split=split_idx, **met))
                print(f"{method:20s} thr={met['thr']:.3f} ACC={met['ACC']:.3f} Rec={met['Rec']:.3f} "
                      f"Prec={met['Prec']:.3f} Sens={met['Sens']:.3f} Spec={met['Spec']:.3f} "
                      f"AUROC={met['AUROC'] if not np.isnan(met['AUROC']) else float('nan')}")
            except Exception as e:
                print(f"[error] {method} split {split_idx}: {e}")

    # save outputs
    df = pd.DataFrame(rows)
    df.to_csv(per_split_csv, index=False)
    print("\nPer-split metrics →", per_split_csv)

    if not df.empty:
        # New: include Sens and Spec in the summary table
        summary = (
            df.groupby("method")[["ACC","F1","Prec","Sens","Spec","AUROC"]]
              .agg(["mean","std"])
              .sort_index()
        )
        summary_csv = OUT_DIR / "summary.csv"
        summary.to_csv(summary_csv)
        print("Summary (mean±std) →", summary_csv)
        print(summary)
    else:
        raise RuntimeError("No results: verify UNI2 splits/weights, UNI2 features, CPTAC labels/features, and per-split kmeans joblibs.")

if __name__ == "__main__":
    main()