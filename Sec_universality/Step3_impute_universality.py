#!/usr/bin/env python3
"""
Step3_impute_universality.py
Impute with VAE checkpoints trained for universality section jobs (250923001–250923128).
- Finds VAE settings automatically from swarm files.
- Runs imputation with mask built from df0 file (negative entries → unobserved).
- Saves outputs into universality/ImputedOut_<jobid>/.
- Prints debug info: first row cols [0:7,23,32] before and after imputation.
"""

import os, re, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ─────────────────── Torch config ───────────────────
torch.set_default_dtype(torch.float64)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────── Files ───────────────────
DATA_DIR = Path("dataset_forallsections")
DF0_FN = DATA_DIR / "all_ctgxyz_99genes_fillrand_fillzero_t012345_noshuffle_250915.csv"

SWARM_FILES = [
    Path("Step2_trainVAE_universality01.swarm"),
    Path("Step2_trainVAE_universality02.swarm")
]

DEFAULT_CKPT_DIR = Path("checkpoints")
DEFAULT_IMPUTED_ROOT = Path("universality")

# ─────────────────── VAE model ───────────────────
class Multiply(nn.Module):
    def __init__(self, c): super().__init__(); self.c = c
    def forward(self, x): return self.c * x

class VAE(nn.Module):
    def __init__(self, N1, N2, N3, ldim, act_name="gelu"):
        super().__init__()
        act = {"tanh": nn.Tanh(), "gelu": nn.GELU()}.get(act_name.lower(), nn.GELU())
        self.encoder = nn.Sequential(
            nn.Linear(102, N1, dtype=torch.float64), act,
            nn.Linear(N1, N1, dtype=torch.float64), act,
            nn.Linear(N1, N2, dtype=torch.float64), act,
            nn.Linear(N2, N3, dtype=torch.float64), act
        )
        self.mu_layer = nn.Linear(N3, ldim, dtype=torch.float64)
        self.log_var_layer = nn.Linear(N3, ldim, dtype=torch.float64)
        dec = [
            nn.Linear(ldim, N3, dtype=torch.float64), act,
            nn.Linear(N3, N2, dtype=torch.float64), act,
            nn.Linear(N2, N1, dtype=torch.float64), act,
            nn.Linear(N1, N1, dtype=torch.float64), act,
            nn.Linear(N1, 99, dtype=torch.float64),
            nn.Sigmoid(), Multiply(3.18)
        ]
        self.decoder = nn.Sequential(*dec)
        self.double()

    @torch.no_grad()
    def encode_mu(self, x):
        h = self.encoder(x)
        return self.mu_layer(h)

# ─────────────────── Utilities ───────────────────
def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def parse_swarm(job_id: int) -> str:
    for swarm in SWARM_FILES:
        if not swarm.exists():
            continue
        with open(swarm) as f:
            for line in f:
                if str(job_id) in line:
                    return line.strip()
    raise ValueError(f"Job {job_id} not found in swarm files.")

def parse_cmdline(line: str) -> dict:
    parts = line.split()
    args = parts[2:]  # skip 'python universality/...py'
    d = {
        "N1": int(args[3]), "N2": int(args[4]), "N3": int(args[5]),
        "ldim": int(args[8]),
        "actfcn": args[9],
        "job_id": int(args[13]),
    }
    return d

def select_ckpt(ckpt_dir: Path, job_id: int) -> Path:
    cands = [fn for fn in os.listdir(ckpt_dir) if fn.endswith(".ckpt") and str(job_id) in fn]
    if not cands:
        raise RuntimeError(f"No ckpt for job_id {job_id}")
    if len(cands) > 1:
        print(f"[warn] Multiple ckpts for job {job_id}, picking first: {cands[0]}")
    return ckpt_dir / cands[0]

# ─────────────────── Imputation ───────────────────
@torch.no_grad()
def predict_decoded(model: VAE, gx: np.ndarray, batch=4096):
    out = np.empty((gx.shape[0], 99))
    for s in range(0, gx.shape[0], batch):
        e = min(gx.shape[0], s+batch)
        x = torch.tensor(gx[s:e], dtype=torch.float64, device=DEVICE)
        mu = model.encode_mu(x)
        dec = model.decoder(mu).cpu().numpy()
        out[s:e] = dec
    return out

def impute_keep_observed(model: VAE, genes, xyz, mask, iters=20, tol=1e-8):
    g = genes.copy()
    prev = g.copy()
    for it in range(iters):
        gx = np.concatenate([g, xyz], axis=1)
        pred = predict_decoded(model, gx)
        pred = np.maximum(pred, 0.0)
        g = np.where(mask, g, pred)

        # Debug: row0 cols [0:7,23,32]
        row = np.concatenate([cid.reshape(-1,1), time.reshape(-1,1), g], axis=1)[0]
        cols = list(row[0:7]) + [row[22], row[31]]
        print(f"[iter {it+1:02d}] {cols}")

        # Early stopping check (only gene cols, 2:101)
        diff = np.max(np.abs(g[:, :99] - prev[:, :99]))
        if diff < tol:
            print(f"Converged at iter {it+1} (max Δ={diff:.2e})")
            break
        prev = g.copy()
    return g




def run_one_job(job_id: int, ckpt_dir=DEFAULT_CKPT_DIR, out_root=DEFAULT_IMPUTED_ROOT):
    # 1. Get command line
    line = parse_swarm(job_id)
    settings = parse_cmdline(line)
    print(f"Job {job_id} settings: {settings}")

    # 2. Load data
    df0 = pd.read_csv(DF0_FN, header=None).values
    global cid, time
    cid, time = df0[:,0], df0[:,1]
    genes, xyz = df0[:,2:101], df0[:,101:104]
    genes = genes.astype(np.float64)
    xyz = xyz.astype(np.float64)

    observed_mask = (genes > 0)
    genes = np.abs(genes)

    # Debug: print row0 before
    print("\n[debug] Before imputation (row 0):")
    row0 = np.concatenate([cid.reshape(-1,1), time.reshape(-1,1), genes], axis=1)[0]
    cols0 = list(row0[0:7]) + [row0[22], row0[31]]
    print("[before] ", cols0)

   

    # 3. Load checkpoint
    ckpt = select_ckpt(ckpt_dir, job_id)
    state = torch.load(ckpt, map_location="cpu")
    model = VAE(settings["N1"], settings["N2"], settings["N3"], settings["ldim"], settings["actfcn"]).to(DEVICE).eval()
    model.load_state_dict(state.get("state_dict", state), strict=False)
    for m in model.modules():
        if isinstance(m, nn.Dropout): m.p = 0.0

    # 4. Impute
    genes_imp = impute_keep_observed(model, genes, xyz, observed_mask, iters=20)

    # 5. Save
    outdir = out_root / f"ImputedOut_{job_id}"
    ensure_dir(outdir)
    out_csv = outdir / f"Imputed_{job_id}_ldim{settings['ldim']}.csv"
    arr = np.concatenate([cid.reshape(-1,1), time.reshape(-1,1), genes_imp, xyz], axis=1)
    pd.DataFrame(arr).to_csv(out_csv, header=None, index=False)
    print(f"✅ Saved {out_csv}")

# ─────────────────── CLI ───────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job_id", type=int, required=True)
    ap.add_argument("--ckpt_dir", type=str, default=str(DEFAULT_CKPT_DIR))
    ap.add_argument("--out_root", type=str, default=str(DEFAULT_IMPUTED_ROOT))
    args = ap.parse_args()
    run_one_job(args.job_id, Path(args.ckpt_dir), Path(args.out_root))

if __name__ == "__main__":
    main()
