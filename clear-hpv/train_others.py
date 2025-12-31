#!/usr/bin/env python3
# Unified H-space clustering trainer for CLAM/ABMIL/TransMIL
from pathlib import Path
import argparse, gc, json, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
import h5py, joblib
from sklearn.cluster import MiniBatchKMeans, KMeans

# ────────────────────── models ──────────────────────
# adjust imports to your paths
from models.model_clam import CLAM_MB, ABMIL_SB, TransMIL_SB, ABMIL_CLAM, TransMIL_CLAM

# ────────────────────── CLI ──────────────────────
def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # data/splits
    p.add_argument("--dataset", default="hpv_uni2")
    p.add_argument("--feat-dir", type=Path, default=Path("/common/users/wq50/CLAM/features/HPV_UNI2_features/h5_files"))
    p.add_argument("--results-dir", type=Path, default=Path("/common/users/wq50/CLAM/results/TCGA_HNSCC_AMMIL_CLAM_s1"))
    p.add_argument("--splits", default="0-9")
    # model
    p.add_argument("--model_type", choices=["clam_mb","abmil_sb","transmil_sb", "abmil_clam", "transmil_clam"], default="transmil_clam")
    p.add_argument("--embed_dim", type=int, default=1536)
    p.add_argument("--size_arg", choices=["small","big"], default="small")
    p.add_argument("--d_model", type=int, default=512)      # TransMIL
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--n_heads_tx", type=int, default=4)
    p.add_argument("--ff_dim", type=int, default=1024)
    p.add_argument("--attn_pool_heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.25)
    # attention reduction for multi-branch heads
    p.add_argument("--attn_reduce", choices=["mean","max","cls1","cls0"], default="mean",
                   help="reduce K×N attention to 1×N when K>1")
    # clustering/output
    p.add_argument("--out-root", type=Path, default=Path("./all_models_hpv_uni2_transmil_clam"))
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--methods", default="kmeans,awkm,")
    # batching
    p.add_argument("--batch", type=int, default=16384)
    p.add_argument("--rand-state", type=int, default=0)
    # k-means params
    p.add_argument("--mb-init-size", type=int, default=20000)
    p.add_argument("--mb-n-init", type=int, default=10)
    p.add_argument("--lloyd-max-iter", type=int, default=300)
    # soft/vMF/tmix
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--tau-soft", type=float, default=0.5)
    p.add_argument("--tau-vmf", type=float, default=0.07)
    p.add_argument("--nu-t", type=float, default=10.0)
    p.add_argument("--var-floor", type=float, default=1e-3)
    # Sinkhorn
    p.add_argument("--ot-eps", type=float, default=0.05)
    p.add_argument("--sink-iters", type=int, default=100)
    p.add_argument("--sink-tol", type=float, default=1e-4)
    # PACE
    p.add_argument("--pace-epochs", type=int, default=10)
    p.add_argument("--pace-lr", type=float, default=2e-3)
    p.add_argument("--pace-cov-floor", type=float, default=1e-2)
    p.add_argument("--dirichlet-alpha0", type=float, default=1.0)
    p.add_argument("--pace-hidden", type=int, default=512)
    # misc
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()

# ────────────────────── loaders ──────────────────────
def get_train_test_ids(split_csv: Path):
    df = pd.read_csv(split_csv, index_col=0)
    train_ids = [s for s in df["train"].dropna().astype(str)]
    test_ids  = [s for s in df["test"].dropna().astype(str)]
    return train_ids, test_ids

@torch.inference_mode()
def load_any_model(ckpt_path: Path, args):
    mt = args.model_type
    if mt == "clam_mb":
        m = CLAM_MB(gate=True, size_arg=args.size_arg, n_classes=2, embed_dim=args.embed_dim, dropout=args.dropout)
    elif mt == "abmil_sb":
        m = ABMIL_SB(gate=True, size_arg=args.size_arg, n_classes=2, embed_dim=args.embed_dim, dropout=args.dropout)
    elif mt == "abmil_clam":
        m = ABMIL_CLAM(gate=True, size_arg=args.size_arg, n_classes=2, embed_dim=args.embed_dim, dropout=args.dropout)
    elif mt == "transmil_clam":
        m = TransMIL_CLAM(n_classes=2, embed_dim=args.embed_dim, d_model=args.d_model,
                          n_heads=args.n_heads_tx, n_layers=args.n_layers,
                          dim_feedforward=args.ff_dim, attn_pool_heads=args.attn_pool_heads,
                          dropout=args.dropout)
    else:  # transmil_sb
        m = TransMIL_SB(n_classes=2, embed_dim=args.embed_dim, d_model=args.d_model,
                        n_heads=args.n_heads_tx, n_layers=args.n_layers,
                        dim_feedforward=args.ff_dim, attn_pool_heads=args.attn_pool_heads,
                        dropout=args.dropout)
    sd = torch.load(ckpt_path, map_location=args.device)
    m.load_state_dict(sd, strict=False)
    return m.to(args.device).eval()

# ────────────────────── adapters: H, a ──────────────────────
def reduce_attn(A_kxN: torch.Tensor, mode: str) -> torch.Tensor:
    # returns 1×N
    if mode == "mean":
        return A_kxN.mean(dim=0, keepdim=True)
    if mode == "max":
        return A_kxN.max(dim=0, keepdim=True).values
    if mode == "cls1":
        idx = 1 if A_kxN.size(0) > 1 else 0
        return A_kxN[idx:idx+1]
    if mode == "cls0":
        return A_kxN[0:1]
    return A_kxN.mean(dim=0, keepdim=True)

@torch.inference_mode()
def h_and_alpha_block(raw_block: np.ndarray, model, args):
    """
    Returns:
      H: (m, d_h)  projected instance features (CLAM/ABMIL mid; TransMIL proj)
      a: (m,)      non-negative weights with mean≈1 within the block
    """
    x = torch.from_numpy(raw_block).to(args.device)  # (m, D_in)

    if isinstance(model, CLAM_MB) or isinstance(model, ABMIL_SB) or isinstance(model, ABMIL_CLAM):
        # all expose attention_net: (A, h_mid) with A: (m, K), h_mid: (m, d_h)
        A_mK, H = model.attention_net(x)
        if A_mK.dim() == 1:  # safety
            A_mK = A_mK.unsqueeze(1)
        A_kN = A_mK.transpose(1, 0)  # K×N
        A_1N = reduce_attn(A_kN, args.attn_reduce)
        a = torch.softmax(A_1N, dim=1).squeeze(0).clamp_min(1e-12)
        a = a * (a.numel() / a.sum())
        return H.detach().cpu().numpy().astype(np.float32), a.detach().cpu().numpy().astype(np.float32)

    if isinstance(model, TransMIL_SB) or isinstance(model, TransMIL_CLAM):
        # use projected tokens as H; get A via attention_only
        H = model.proj(x)                                  # (m, d_model)
        A_1N = model.forward(x, attention_only=True)       # 1×N attention over encoded tokens
        a = torch.softmax(A_1N, dim=1).squeeze(0).clamp_min(1e-12)
        a = a * (a.numel() / a.sum())
        return H.detach().cpu().numpy().astype(np.float32), a.detach().cpu().numpy().astype(np.float32)

    raise TypeError("Unsupported model class")

def iter_h5_blocks(h5_path: Path, batch: int):
    with h5py.File(h5_path, "r") as f:
        feats = f["features"]; n = feats.shape[0]
        for off in range(0, n, batch):
            yield off, feats[off:off+batch][:].astype(np.float32)

def train_slide_stream(slide_ids, feat_dir: Path, model, batch: int, args):
    for sid in slide_ids:
        h5 = feat_dir / f"{sid}.h5"
        if not h5.exists(): continue
        for _, raw in iter_h5_blocks(h5, batch):
            H, a = h_and_alpha_block(raw, model, args)
            yield H, a

# ────────────────────── helpers ──────────────────────
def save_kmeans(km, out_dir: Path, dataset: str, k: int, tag: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(km, out_dir / f"{dataset}_k{k}_{tag}.joblib")
    np.save(out_dir / f"{dataset}_centers_k{k}_{tag}.npy", km.cluster_centers_.astype(np.float32))
    print(f"   ↳ saved {tag} @ {out_dir}")

def save_centers(arr: np.ndarray, out_dir: Path, dataset: str, k: int, tag: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{dataset}_centers_k{k}_{tag}.npy", arr.astype(np.float32))
    print(f"   ↳ saved centers ({tag}) @ {out_dir}")

def save_manifest(manifest: dict, path: Path):
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)

# ────────────────────── methods (same as your original) ──────────────────────
# kmeans / awkm / softem / vmf / tmix / sinkhorn / pace
# keep your implementations; only replace calls to CLAM with model + args
# Below shows k-means variants; reuse your other functions unchanged.

def train_kmeans(train_ids, feat_dir, model, k, batch, rand_state, mb_init, mb_n_init, lloyd_max_iter, args):
    mbk = MiniBatchKMeans(n_clusters=k, batch_size=batch, init_size=mb_init,
                          n_init=mb_n_init, random_state=rand_state, reassignment_ratio=0, verbose=0)
    for H, _ in train_slide_stream(train_ids, feat_dir, model, batch, args):
        mbk.partial_fit(H)
    X = np.vstack([H for H,_ in train_slide_stream(train_ids, feat_dir, model, batch, args)]).astype(np.float32)
    km = KMeans(n_clusters=k, init=mbk.cluster_centers_, n_init=1, max_iter=lloyd_max_iter,
                copy_x=False, random_state=rand_state, verbose=0).fit(X)
    del X; gc.collect()
    return km

def train_aw_kmeans(train_ids, feat_dir, model, k, batch, rand_state, mb_init, mb_n_init, lloyd_max_iter, args):
    mbk = MiniBatchKMeans(n_clusters=k, batch_size=batch, init_size=mb_init,
                          n_init=mb_n_init, random_state=rand_state, reassignment_ratio=0, verbose=0)
    for H, a in train_slide_stream(train_ids, feat_dir, model, batch, args):
        mbk.partial_fit(H, sample_weight=a)
    X, W = [], []
    for H, a in train_slide_stream(train_ids, feat_dir, model, batch, args):
        X.append(H); W.append(a)
    X = np.vstack(X).astype(np.float32); W = np.concatenate(W).astype(np.float32)
    km = KMeans(n_clusters=k, init=mbk.cluster_centers_, n_init=1, max_iter=lloyd_max_iter,
                copy_x=False, random_state=rand_state, verbose=0).fit(X, sample_weight=W)
    del X, W; gc.collect()
    return km

# keep your softEM/vMF/tmix/sinkhorn/PACE functions; just make them call
#   for H,a in train_slide_stream(train_ids, feat_dir, model, batch, args):
# and pass `model` instead of `clam`.

# ────────────────────── main ──────────────────────
if __name__ == "__main__":
    args = get_args()
    torch.set_float32_matmul_precision("high")
    np.random.seed(args.rand_state)

    # splits
    if "-" in args.splits:
        a,b = args.splits.split("-"); split_list = list(range(int(a), int(b)+1))
    else:
        split_list = [int(x) for x in args.splits.split(",")]

    method_set = {m.strip() for m in args.methods.split(",") if m.strip()}
    # valid = {"kmeans","awkm","softem","vmf","tmix","sinkhorn","pace"}
    valid = {"kmeans","awkm",}
    if not method_set.issubset(valid):
        raise ValueError(f"Unknown methods: {method_set - valid}")

    args.out_root.mkdir(parents=True, exist_ok=True)

    for split_idx in split_list:
        print(f"\n=== Split s_{split_idx} (K={args.k}) [{args.model_type}] ===")
        ckpt = args.results_dir / f"s_{split_idx}_checkpoint.pt"
        split_csv = args.results_dir / f"splits_{split_idx}.csv"
        if not (ckpt.exists() and split_csv.exists()):
            print(f"[skip] missing files for split {split_idx}")
            continue

        split_out = args.out_root / f"s_{split_idx}"
        # method subfolders
        p_km_unw = split_out / "kmeans_rawh"
        p_km_aw  = split_out / "kmeans_awkm_rawh"
        p_soft   = split_out / "soft_awem_rawh"
        p_vmf    = split_out / "vmf_rawh"
        p_tmix   = split_out / "tmixture_rawh"
        p_sink   = split_out / "sinkhorn_rawh"
        p_pace   = split_out / "pace_rawh"
        for p in (p_km_unw, p_km_aw, p_soft, p_vmf, p_tmix, p_sink, p_pace): p.mkdir(parents=True, exist_ok=True)

        model = load_any_model(ckpt, args)
        TRAIN_IDS, TEST_IDS = get_train_test_ids(split_csv)
        print(f"   TRAIN={len(TRAIN_IDS)}  TEST={len(TEST_IDS)}")

        manifest = {
            "split": split_idx,
            "checkpoint": str(ckpt),
            "split_csv": str(split_csv),
            "model_type": args.model_type,
            "embed_dim": args.embed_dim,
            "size_arg": args.size_arg,
            "outputs": {}
        }

        # 1) k-means (unweighted)
        if "kmeans" in method_set:
            print(" -> k-means (hard, unweighted)")
            km_unw = train_kmeans(TRAIN_IDS, args.feat_dir, model, args.k, args.batch,
                                   args.rand_state, args.mb_init_size, args.mb_n_init,
                                   args.lloyd_max_iter, args)
            save_kmeans(km_unw, p_km_unw, args.dataset, args.k, "rawh")
            manifest["outputs"]["kmeans_rawh"] = {
                "model": str((p_km_unw / f"{args.dataset}_k{args.k}_rawh.joblib").resolve()),
                "centers": str((p_km_unw / f"{args.dataset}_centers_k{args.k}_rawh.npy").resolve())
            }
            del km_unw; gc.collect()

        # 2) k-means (attention-weighted)
        if "awkm" in method_set:
            print(" -> k-means (hard, attention-weighted)")
            km_aw = train_aw_kmeans(TRAIN_IDS, args.feat_dir, model, args.k, args.batch,
                                     args.rand_state, args.mb_init_size, args.mb_n_init,
                                     args.lloyd_max_iter, args)
            save_kmeans(km_aw, p_km_aw, args.dataset, args.k, "rawh_awkm")
            manifest["outputs"]["kmeans_awkm_rawh"] = {
                "model": str((p_km_aw / f"{args.dataset}_k{args.k}_rawh_awkm.joblib").resolve()),
                "centers": str((p_km_aw / f"{args.dataset}_centers_k{args.k}_rawh_awkm.npy").resolve())
            }
            del km_aw; gc.collect()

        # 3–7) reuse your original softem/vmf/tmix/sinkhorn/pace blocks,
        # replacing CLAM references with `model` and passing `args` into helpers.

        # save manifest
        with open(split_out / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        print(f" ✓ manifest → {split_out/'manifest.json'}")

        del model; gc.collect(); torch.cuda.empty_cache()

    print("\n✓ All requested splits trained.")