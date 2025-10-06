#!/usr/bin/env python3
# step3_impute_250915.py
# Impute with VAE, keep observed values fixed, make 99x scatter, save metrics.
# Python 3.9 compatible. Always abs() gene columns.

import os
import re
import math
import argparse
from typing import Optional, List, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ───────────────────── Torch config ─────────────────────
torch.set_default_dtype(torch.float64)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# ───────────────────── Data files ─────────────────────
# Mask is built from zero-fill file (observed = nonzero in cols 2..100)
zero_csv = "all_ctgxyz_99genes_fillzero_fillzero_t012345_noshuffle.csv"
rand_csv = "all_ctgxyz_99genes_fillrand_fillzero_t012345_noshuffle_250915.csv"
avg_csv  = "all_ctgxyz_99genes_fillavg_fillzero_t012345_noshuffle_250915.csv"

TEST_CELLS_CSV = "test_cells.csv"

# ───────────────────── Dirs ─────────────────────
DEFAULT_CKPT_DIR    = "checkpoints"
DEFAULT_IMPUTED_DIR = "ImputedOut_250916"
PLOTS_ROOT_DEFAULT  = Path("Plots/imputation250916")

PLOTS_ROOT = PLOTS_ROOT_DEFAULT  # will be overridden by --plots_root

# ───────────────────── JobID → filling method ─────────────────────
# ───────────────────── JobID → filling method ─────────────────────
ZERO_IDS_250917 = set(range(250917001, 250917013))
AVG_IDS_250917  = set(range(250917013, 250917025))
RAND_IDS_250917 = set(range(250917025, 250917037))

ZERO_IDS_250918 = set(range(250918001, 250918013))
AVG_IDS_250918  = set(range(250918013, 250918025))
RAND_IDS_250918 = set(range(250918025, 250918037))

def filling_method_for(job_id: int) -> str:
    if job_id in (ZERO_IDS_250917 | ZERO_IDS_250918):
        return "zero"
    if job_id in (AVG_IDS_250917 | AVG_IDS_250918):
        return "avg"
    if job_id in (RAND_IDS_250917 | RAND_IDS_250918):
        return "rand"
    raise ValueError(f"Unknown filling method for job_id={job_id}")

ALL_JOB_IDS = sorted(
    ZERO_IDS_250917 | AVG_IDS_250917 | RAND_IDS_250917 |
    ZERO_IDS_250918 | AVG_IDS_250918 | RAND_IDS_250918
)



def preimp_csv_for(method: str) -> str:
    if method == "zero": return zero_csv
    if method == "avg":  return avg_csv
    if method == "rand": return rand_csv
    raise ValueError(method)

# ───────────────────── Utils ─────────────────────
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def parse_ldim_from_ckpt_name(name: str) -> Optional[int]:
    m = re.search(r"_ld(\d+)_", name)
    if m: return int(m.group(1))
    m2 = re.search(r"_ldim(\d+)_", name)
    if m2: return int(m2.group(1))
    return None

def select_ckpt(ckpt_dir: str, job_id: int, ckpt_type: str) -> str:
    """
    ckpt_type: 'valid' -> pick VAE* without 'SMOOTH' or 'TRAIN'
               'smooth'-> pick VAE* with 'SMOOTH' and not 'TRAIN'
               'train' -> pick VAE* with 'TRAIN' (normally unused)
    Only paths starting with 'VAE' are valid.
    """
    paths = []
    for fn in os.listdir(ckpt_dir):
        if not fn.endswith(".ckpt"): continue
        if not fn.startswith("VAE"): continue
        if str(job_id) not in fn:    continue
        paths.append(fn)

    if ckpt_type == "valid":
        cands = [p for p in paths if "SMOOTH" not in p and "TRAIN" not in p]
    elif ckpt_type == "smooth":
        cands = [p for p in paths if "SMOOTH" in p and "TRAIN" not in p]
    elif ckpt_type == "train":
        cands = [p for p in paths if "TRAIN" in p]
    else:
        raise ValueError(f"ckpt_type must be one of valid/smooth/train, got {ckpt_type}")

    if len(cands) != 1:
        msg = ["Expected exactly 1", f"found {len(cands)} for job_id={job_id} ({ckpt_type})."]
        msg += ["Candidates:"] + [f"  {x}" for x in sorted(paths)]
        raise RuntimeError("\n".join(msg))
    ckpt_path = str(Path(ckpt_dir) / cands[0])
    print(f"\n==== Using ckpt: {ckpt_path} ====")
    return ckpt_path

# ───────────────────── Model (inference) ─────────────────────
class Multiply(nn.Module):
    def __init__(self, c: float): super().__init__(); self.c = c
    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.c * x

class VAE(nn.Module):
    def __init__(self, ndense1: int, ndense2: int, ndense3: int,
                 latent_dim: int, input_dim: int = 102, output_dim: int = 99,
                 act_name: str = "gelu", dec_drop_p: float = 0.0):
        super().__init__()
        act = {
            "tanh": nn.Tanh(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
            "relu": nn.ReLU(),
            "mish": nn.Mish(),
            "elu": nn.ELU(),
            "celu": nn.CELU(),
            "softplus": nn.Softplus(),
        }.get(act_name.lower(), nn.GELU())

        self.enc_fc1 = nn.Linear(input_dim, ndense1, dtype=torch.float64)
        self.enc_fc2 = nn.Linear(ndense1, ndense1, dtype=torch.float64)
        self.enc_fc3 = nn.Linear(ndense1, ndense2, dtype=torch.float64)
        self.enc_fc4 = nn.Linear(ndense2, ndense3, dtype=torch.float64)
        self.act = act
        self.mu_layer = nn.Linear(ndense3, latent_dim, dtype=torch.float64)
        self.log_var_layer = nn.Linear(ndense3, latent_dim, dtype=torch.float64)

        dec = []
        dec += [nn.Linear(latent_dim, ndense3, dtype=torch.float64), act, nn.Dropout(p=dec_drop_p)]
        dec += [nn.Linear(ndense3, ndense2, dtype=torch.float64),   act, nn.Dropout(p=dec_drop_p)]
        dec += [nn.Linear(ndense2, ndense1, dtype=torch.float64),   act, nn.Dropout(p=dec_drop_p)]
        dec += [nn.Linear(ndense1, ndense1, dtype=torch.float64),   act]
        dec += [nn.Linear(ndense1, output_dim, dtype=torch.float64)]
        dec += [nn.Sigmoid(), Multiply(3.18)]
        self.decoder = nn.Sequential(*dec)
        self.double()

    @torch.no_grad()
    def encode_mu(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.enc_fc1(x))
        x = self.act(self.enc_fc2(x))
        x = self.act(self.enc_fc3(x))
        x = self.act(self.enc_fc4(x))
        return self.mu_layer(x)

def load_vae_from_ckpt(ckpt_path: str) -> Tuple[VAE, int]:
    sd = torch.load(ckpt_path, map_location="cpu")
    state = sd.get("state_dict", sd)
    hp = sd.get("hyper_parameters", {})
    nd1 = int(hp.get("ndense1", 1024))
    nd2 = int(hp.get("ndense2", 512))
    nd3 = int(hp.get("ndense3", 256))
    ldim = int(hp.get("latent_dim", parse_ldim_from_ckpt_name(os.path.basename(ckpt_path)) or 6))
    in_dim = int(hp.get("input_dim", 102))
    out_dim= int(hp.get("output_dim", 99))
    act   = str(hp.get("act_name", hp.get("actfcn", "gelu")))

    model = VAE(nd1, nd2, nd3, ldim, input_dim=in_dim, output_dim=out_dim, act_name=act, dec_drop_p=0.0).to(DEVICE).eval()
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:   print(f"[warn] Missing keys: {len(missing)} (ok for non-inference parts)")
    if unexpected:print(f"[warn] Unexpected keys: {len(unexpected)}")
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.0
    return model, ldim

# ───────────────────── Core ops ─────────────────────
COL_CID, COL_TIME = 0, 1
G_START, G_END = 2, 2+99
XYZ_START, XYZ_END = 101, 104

def r2_score_vec(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    mean   = np.mean(y_true, axis=0)
    ss_tot = np.sum((y_true - mean) ** 2, axis=0) + 1e-12
    return 1.0 - ss_res / ss_tot

@torch.no_grad()
def predict_decoded(model: VAE, genes_xyz: np.ndarray, batch: int = 4096) -> np.ndarray:
    N = genes_xyz.shape[0]
    out = np.empty((N, 99), dtype=np.float64)
    for s in range(0, N, batch):
        e = min(N, s+batch)
        x = torch.tensor(genes_xyz[s:e], dtype=torch.float64, device=DEVICE)
        mu = model.encode_mu(x)
        dec = model.decoder(mu).cpu().numpy()
        out[s:e] = dec
    return out

def impute_keep_observed(model: VAE,
                         genes_init: np.ndarray,
                         xyz: np.ndarray,
                         observed_mask: np.ndarray,
                         iters: int = 3,
                         nonneg: bool = True) -> np.ndarray:
    genes = genes_init.copy()
    for _ in range(max(1, iters)):
        gx = np.concatenate([genes, xyz], axis=1)  # [N, 102]
        pred = predict_decoded(model, gx)
        if nonneg:
            pred = np.maximum(pred, 0.0)
        genes = np.where(observed_mask, genes, pred)
    return genes

def run_one_job(job_id: int,
                ckpt_type: str,
                ckpt_dir: str,
                imputed_dir: str,
                iters: int = 3,
                recon_batch: int = 4096) -> None:
    method = filling_method_for(job_id)
    src_csv = preimp_csv_for(method)

    # Load method data & zero-fill (for mask)
    df_method = pd.read_csv(src_csv, header=None)
    df_zero   = pd.read_csv(zero_csv, header=None)

    # Extract arrays
    cid = df_method.iloc[:, COL_CID].to_numpy()
    time= df_method.iloc[:, COL_TIME].to_numpy()
    genes_m = df_method.iloc[:, G_START:G_END].to_numpy(dtype=np.float64)
    xyz     = df_method.iloc[:, XYZ_START:XYZ_END].to_numpy(dtype=np.float64)
    genes_zero = df_zero.iloc[:, G_START:G_END].to_numpy(dtype=np.float64)

    # ALWAYS abs() gene columns (both method and zero for mask)
    genes_m    = np.abs(genes_m)
    genes_zero = np.abs(genes_zero)

    # Observed mask: nonzero in zero-fill file (after abs)
    observed_mask = (genes_zero != 0.0)

    # Load model
    ckpt_path = select_ckpt(ckpt_dir, job_id, ckpt_type)
    model, ldim = load_vae_from_ckpt(ckpt_path)

    # Impute (do not change observed)
    genes_imp = impute_keep_observed(model, genes_m, xyz, observed_mask, iters=iters, nonneg=True)

    # Save imputed matrix: [cid, time, 99 genes, xyz]
    ensure_dir(Path(imputed_dir))
    out_csv = Path(imputed_dir) / f"Imputed_{job_id}_{ckpt_type}_ldim{ldim}.csv"
    out_arr = np.concatenate([cid.reshape(-1,1), time.reshape(-1,1), genes_imp, xyz], axis=1)
    pd.DataFrame(out_arr).to_csv(out_csv, header=None, index=False)
    print(f"✅ Saved imputed CSV: {out_csv}")

    # ── Reconstruction scatter + metrics over test cells ──
    if not os.path.exists(TEST_CELLS_CSV):
        print(f"[warn] {TEST_CELLS_CSV} not found, skipping plots/metrics.")
        return
    test_ids = pd.read_csv(TEST_CELLS_CSV, header=None).iloc[:,0].to_numpy()
    isin = np.isin(cid, test_ids)
    if not np.any(isin):
        print("[warn] No overlap with test_cells.csv; skipping plots/metrics.")
        return

    genes_test = genes_imp[isin]
    xyz_test   = xyz[isin]
    preds = predict_decoded(model, np.concatenate([genes_test, xyz_test], axis=1), batch=recon_batch)

    # Metrics
    ae = np.mean(np.abs(genes_test - preds), axis=0)
    se = np.mean((genes_test - preds)**2, axis=0)
    r2 = r2_score_vec(genes_test, preds)

    MAE = float(np.mean(ae))
    MSE = float(np.mean(se))
    meanR2 = float(np.mean(r2))

    # Save metrics csv (overall + per-gene)
    plot_dir = PLOTS_ROOT / method
    ensure_dir(plot_dir)
    metrics_csv = plot_dir / f"recon_metrics_{job_id}_{ckpt_type}.csv"

    metrics_df = pd.DataFrame({
        "metric": ["MAE_overall","MSE_overall","mean_R2_overall"],
        "value":  [MAE, MSE, meanR2]
    })
    per_gene_df = pd.DataFrame({
        "gene_idx": np.arange(1,100),
        "AE": ae,
        "SE": se,
        "R2": r2
    })
    with open(metrics_csv, "w") as f:
        metrics_df.to_csv(f, index=False)
        f.write("\n")
        per_gene_df.to_csv(f, index=False)
    print(f"📊 MAE={MAE:.6f}  MSE={MSE:.6f}  meanR2={meanR2:.6f}")
    print(f"✅ Saved metrics: {metrics_csv}")

    # Scatter grid
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nrows, ncols = 11, 9
    fig, axes = plt.subplots(nrows, ncols, figsize=(24, 28))
    axes = axes.ravel()
    for g in range(99):
        ax = axes[g]
        x = genes_test[:, g]
        y = preds[:, g]
        ax.scatter(x, y, s=12, alpha=0.5)
        lo = float(min(x.min(), y.min())); hi = float(max(x.max(), y.max()))
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1)
        ax.set_title(f"Gene {g+1}  R²={r2[g]:.3f}", fontsize=9)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.grid(True, alpha=0.3)
    for k in range(99, nrows*ncols):
        fig.delaxes(axes[k])
    fig.suptitle(f"VAE recon vs imputed (TEST ONLY)\njob_id={job_id}, ckpt={ckpt_type}, method={method}, ldim={ldim}\nN={genes_test.shape[0]} cells",
                 fontsize=16, y=0.995)
    fig.supxlabel("Imputed value (x)")
    fig.supylabel("VAE reconstruction (y)")
    plt.tight_layout(rect=[0,0,1,0.96])

    out_png = plot_dir / f"recon_scatter_{job_id}_{ckpt_type}.png"
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"🖼️  Saved plot: {out_png}")

# ───────────────────── Batch/sharding ────────────────────

def jobs_for_shard(shard: int, nshards: int) -> List[int]:
    return [jid for i, jid in enumerate(ALL_JOB_IDS) if (i % nshards) == shard]

# ───────────────────── CLI ─────────────────────
def main():
    ap = argparse.ArgumentParser(description="VAE imputation + recon metrics/plots (mask from zero-fill).")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--job_id", type=int, help="Single job id to run")
    mode.add_argument("--shard", type=int, help="Shard index (0-based) for batch mode")
    ap.add_argument("--nshards", type=int, help="Number of shards (batch mode)")
    ap.add_argument("--ckpt_type", type=str, default="valid", choices=["valid","smooth","train"],
                    help="Which checkpoint to use in single-job mode (batch runs valid+smooth)")
    ap.add_argument("--ckpt_dir", type=str, default=DEFAULT_CKPT_DIR)
    ap.add_argument("--imputed_dir", type=str, default=DEFAULT_IMPUTED_DIR)
    ap.add_argument("--iters", type=int, default=3, help="Imputation iterations (update only unobserved)")
    ap.add_argument("--plots_root", type=str, default=str(PLOTS_ROOT_DEFAULT),
                  help="Root folder for plots (will create method subfolders zero/avg/rand)")


    args = ap.parse_args()
    global PLOTS_ROOT
    PLOTS_ROOT = Path(args.plots_root)
    
    if args.job_id is not None:
        run_one_job(
            job_id=args.job_id,
            ckpt_type=args.ckpt_type,
            ckpt_dir=args.ckpt_dir,
            imputed_dir=args.imputed_dir,
            iters=args.iters
        )
        return

    if args.shard is None or args.nshards is None:
        raise SystemExit("In batch mode you must provide both --shard and --nshards.")
    shard_jobs = jobs_for_shard(args.shard, args.nshards)
    print(f"Shard {args.shard}/{args.nshards}: {len(shard_jobs)} jobs -> {shard_jobs}")

    for jid in shard_jobs:
        for ctype in ("valid","smooth"):   # run both; skip TRAIN
            try:
                run_one_job(
                    job_id=jid,
                    ckpt_type=ctype,
                    ckpt_dir=args.ckpt_dir,
                    imputed_dir=args.imputed_dir,
                    iters=args.iters
                )
            except Exception as e:
                print(f"[error] job_id={jid} {ctype}: {e}")

if __name__ == "__main__":
    main()
