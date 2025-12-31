#!/usr/bin/env python3
# Heatmap-area evaluator over 10 splits: splits/HPV_100/splits_0.csv ... splits_9.csv

from pathlib import Path
import numpy as np, pandas as pd, h5py, torch, torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
import gc

# ----- config -----
FEATURE_DIR = Path("/common/users/wq50/CLAM/features/HPV_UNI2_features/h5_files")
LABEL_CSV   = Path("/common/users/wq50/CLAM2/dataset_csv/HNSCC.csv")
SPLITS_DIR  = Path("/common/users/wq50/CLAM/splits/HPV_100")  # files: splits_0.csv ... splits_9.csv
RESULTS_DIR = Path("/common/users/wq50/CLAM/results/HPV_CLAM_50_mb_s1")  # s_{i}_checkpoint.pt
OUT_DIR     = Path("./eval_heatmap_hnscc_tcga"); OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE        = "cuda:0" if torch.cuda.is_available() else "cpu"
BATCH         = 16384
TARGET_REC    = 0.80
POS_NAME      = "HPV+"
POS_ATT_INDEX = 0  # set to 0 if attention columns are [HPV+, HPV-]

# ----- CLAM backbone -----
from models.model_clam import CLAM_MB
_CLAM = {}
@torch.inference_mode()
def load_clam(ckpt: Path):
    if ckpt not in _CLAM:
        m = CLAM_MB(gate=True, size_arg="small", n_classes=2, embed_dim=1536)
        state = torch.load(ckpt, map_location=DEVICE)
        m.load_state_dict(state, strict=False)
        _CLAM[ckpt] = m.to(DEVICE).eval()
    return _CLAM[ckpt]

def iter_blocks(h5_path: Path, batch=BATCH):
    with h5py.File(h5_path, "r") as f:
        feats = f["features"]; n = feats.shape[0]
        for off in range(0, n, batch):
            yield feats[off:off+batch][:].astype(np.float32)

@torch.inference_mode()
def att_logits_block(raw_np, clam):
    x = torch.from_numpy(raw_np).to(DEVICE)
    A_raw, _ = clam.attention_net(x)              # (m, n_classes)
    # a = A_raw[:, 0] if A_raw.size(1)==1 else A_raw[:, POS_ATT_INDEX]
    # instead of POS_ATT_INDEX, use the mean attention over positive class logits:
    a = A_raw.mean(dim=1)
    return a.detach().cpu().numpy().astype(np.float32)

def slide_score_area(clam, sid: str, t: float):
    h5 = FEATURE_DIR / f"{sid}.h5"
    if not h5.exists(): return None
    vals = []
    for raw in iter_blocks(h5):
        vals.append(att_logits_block(raw, clam)); del raw
    if not vals: return None
    a = np.concatenate(vals, 0)
    lo, hi = float(a.min()), float(a.max())
    a01 = (a - lo) / (hi - lo + 1e-12)
    return float((a01 >= t).mean())

# ----- split reader for format: splits_{i}.csv with columns train,val,test -----
def read_split_file(csv_path: Path):
    df = pd.read_csv(csv_path)
    # handle possible unnamed index column
    if df.columns[0].lower().startswith("unnamed"):
        df = df.drop(columns=df.columns[0])
    def col_list(name):
        if name not in df.columns: return []
        # cells may be single IDs; aggregate all rows
        return [s.strip() for s in df[name].dropna().astype(str).tolist() if s.strip()]
    return dict(train=col_list("train"), val=col_list("val"), test=col_list("test"))

# ----- thresholds and metrics -----
def tune_t_by_auc(clam, ids, y):
    grid = np.linspace(0.1, 0.9, 9)
    best_t, best_auc = 0.5, -1.0
    for t in grid:
        scores, yy = [], []
        for sid, yi in zip(ids, y):
            sc = slide_score_area(clam, sid, t)
            if sc is None: continue
            scores.append(sc); yy.append(yi)
        yy = np.array(yy, int)
        if len(np.unique(yy)) < 2: continue
        auc = roc_auc_score(yy, np.array(scores, float))
        if auc > best_auc:
            best_auc, best_t = auc, float(t)
    return best_t, best_auc

def tune_tau(scores, y, target_recall=TARGET_REC):
    y = np.asarray(y, int); s = np.asarray(scores, float)
    if len(np.unique(y))<2: return 0.5
    fpr, tpr, thr = roc_curve(y, s)
    ok = tpr >= target_recall
    if ok.any():
        prec_est = 1 - fpr
        f1_est = 2*(prec_est*tpr)/np.maximum(prec_est+tpr,1e-12)
        idx = np.argmax(f1_est*ok)
        return float(thr[idx] if idx < len(thr) else 0.5)
    return 0.5

def compute_metrics(y_true, scores, thr):
    y_true = np.asarray(y_true, int); scores = np.asarray(scores, float)
    y_hat = (scores >= thr).astype(int)
    return dict(
        ACC   = accuracy_score(y_true, y_hat),
        F1    = f1_score(y_true, y_hat, zero_division=0),
        Prec  = precision_score(y_true, y_hat, zero_division=0),
        Rec   = recall_score(y_true, y_hat, zero_division=0),
        AUROC = roc_auc_score(y_true, scores) if len(np.unique(y_true))==2 else np.nan,
        thr   = float(thr),
    )

# ----- main -----
def main():
    np.random.seed(0)
    lab = pd.read_csv(LABEL_CSV).set_index("slide_id")
    if "hpv_status" not in lab.columns:
        raise KeyError("LABEL_CSV must contain 'hpv_status'")

    rows = []
    for split_idx in range(10):
        split_csv = SPLITS_DIR / f"splits_{split_idx}.csv"
        ckpt = RESULTS_DIR / f"s_{split_idx}_checkpoint.pt"
        if not split_csv.exists():
            print(f"[skip] split {split_idx}: missing {split_csv}")
            continue
        if not ckpt.exists():
            print(f"[skip] split {split_idx}: missing {ckpt}")
            continue

        clam = load_clam(ckpt)
        sp = read_split_file(split_csv)

        train_ids = [s for s in sp["train"] if (FEATURE_DIR / f"{s}.h5").exists()]
        val_ids   = [s for s in sp["val"]   if (FEATURE_DIR / f"{s}.h5").exists()]
        test_ids  = [s for s in sp["test"]  if (FEATURE_DIR / f"{s}.h5").exists()]

        if len(train_ids)==0 or len(test_ids)==0:
            print(f"[skip] split {split_idx}: empty train or test")
            continue

        y_train = [int(str(lab.at[s, "hpv_status"]) == POS_NAME) for s in train_ids]
        y_val   = [int(str(lab.at[s, "hpv_status"]) == POS_NAME) for s in val_ids] if len(val_ids)>0 else []
        y_test  = [int(str(lab.at[s, "hpv_status"]) == POS_NAME) for s in test_ids]

        # 1) tune heatmap threshold t on TRAIN
        t_star, _ = tune_t_by_auc(clam, train_ids, y_train)

        # 2) tune decision tau on VAL or TRAIN
        ids_tau = val_ids if len(val_ids)>0 else train_ids
        y_tau   = y_val   if len(val_ids)>0 else y_train
        s_tau = [slide_score_area(clam, s, t_star) for s in ids_tau]
        tau = tune_tau(s_tau, y_tau, TARGET_REC)

        # 3) evaluate on TEST
        s_test = [slide_score_area(clam, s, t_star) for s in test_ids]
        met = compute_metrics(y_test, s_test, tau)
        rows.append(dict(split=split_idx, t=t_star, tau=tau, **met))
        print(f"[split {split_idx}] t*={t_star:.2f} tau*={tau:.3f} "
              f"ACC={met['ACC']:.3f} Rec={met['Rec']:.3f} Prec={met['Prec']:.3f} AUROC={met['AUROC']:.3f}")
        gc.collect()

    df = pd.DataFrame(rows)
    per_split = OUT_DIR / "per_split.csv"; df.to_csv(per_split, index=False)
    print("Per-split →", per_split)

    if not df.empty:
        summary = df[["ACC","F1","Prec","Rec","AUROC"]].agg(["mean","std"])
        summary_csv = OUT_DIR / "summary.csv"; summary.to_csv(summary_csv)
        print("Summary →", summary_csv); print(summary)

if __name__ == "__main__":
    main()