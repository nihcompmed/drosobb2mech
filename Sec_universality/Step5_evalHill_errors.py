#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step9_evalHill_errors.py

Compute Hill model prediction errors for each job.

Outputs:
  - Full error matrix per job (per cell × time, 99 relative errors)
"""

import os, argparse
import numpy as np
import pandas as pd
import torch

torch.set_default_dtype(torch.float64)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Hill Model Loader ----
def load_hill_params(csv_path, G=99):
    arr = pd.read_csv(csv_path, header=None).values
    b0 = arr[0, :]
    V  = arr[1:1+G, :]
    K  = arr[1+G:1+2*G, :]
    n  = arr[1+2*G:1+3*G, :]
    gamma = arr[1+3*G, :]
    return b0, V, K, n, gamma

# ---- Hill vector field ----
def hill_field(x, b0, V, K, n, gamma):
    x_safe = torch.clamp(x, min=1e-12)              # (G,)
    n_eff  = torch.clamp(torch.as_tensor(n, device=DEVICE), 1.0, 3.0)
    K_eff  = torch.clamp(torch.as_tensor(K, device=DEVICE), 0.2, 2.5)
    V_t    = torch.as_tensor(V, device=DEVICE)
    b0_t   = torch.as_tensor(b0, device=DEVICE)
    gamma_t= torch.as_tensor(gamma, device=DEVICE)

    # Broadcast x -> (G, G): each row (target i) sees all regulators j
    xj = x_safe.unsqueeze(0).expand_as(V_t)     # (G,G)
    xn = torch.pow(xj, n_eff)                   # (G,G)
    den = torch.pow(K_eff, n_eff) + xn
    h = xn / (den + 1e-12)

    prod = (h * V_t).sum(dim=1) + b0_t          # sum over regulators j
    return prod - gamma_t * x_safe


# ---- Fixed RK4 with substeps=5 ----
def hill_step_rk4_sub5(x0, b0, V, K, n, gamma, dt=1.0, n_sub=5):
    x = torch.as_tensor(x0, dtype=torch.float64, device=DEVICE)
    h = dt / n_sub
    for _ in range(n_sub):
        k1 = hill_field(x, b0, V, K, n, gamma)
        k2 = hill_field(x + 0.5*h*k1, b0, V, K, n, gamma)
        k3 = hill_field(x + 0.5*h*k2, b0, V, K, n, gamma)
        k4 = hill_field(x + h*k3, b0, V, K, n, gamma)
        x = x + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    return x.detach().cpu().numpy()

# ---- Main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job_id", type=int, required=True)
    ap.add_argument("--l1_tag", type=str, required=True)  # 1en4 or 1en5
    args = ap.parse_args()

    job_id = args.job_id
    l1_tag = args.l1_tag

    # Hill params
    hill_fn = f"universality/HillModels_trainonly/Hill_fromVAEonly_sub5_rk4{5}_{job_id}hill_l1{l1_tag}.csv"
    if not os.path.exists(hill_fn):
        raise FileNotFoundError(hill_fn)
    b0, V, K, n, gamma = load_hill_params(hill_fn)

    # Imputed full CSV
    imp_dir = f"universality/ImputedOut/ImputedOut_{job_id}"
    candidates = [f for f in os.listdir(imp_dir) if f.startswith(f"Imputed_{job_id}_ldim")]
    assert candidates, f"No imputed CSV for job {job_id}"
    imp_fn = os.path.join(imp_dir, candidates[0])
    df = pd.read_csv(imp_fn, header=None).values
    cid = df[:, 0].astype(int)
    time = df[:, 1].astype(int)
    genes = df[:, 2:101]

    # Build (cid,t→t+1) pairs
    key2idx = {(int(c), int(t)): i for i, (c, t) in enumerate(zip(cid, time))}
    pairs = [(c, t) for c, t in zip(cid, time) if (c, t+1) in key2idx]

    errs = []
    for c, t in pairs:
        i = key2idx[(c, t)]
        j = key2idx[(c, t+1)]
        g_t, g_t1 = genes[i], genes[j]
        pred = hill_step_rk4_sub5(g_t, b0, V, K, n, gamma, dt=1.0, n_sub=5)
        relerr = np.abs(pred - g_t1) / (np.abs(pred) + np.abs(g_t1) + 1e-8)
        errs.append([c, t, *relerr])

    errs = np.array(errs)

    out_dir = "universality/HillErrors"
    os.makedirs(out_dir, exist_ok=True)
    full_fn = os.path.join(out_dir, f"HillErrors_{job_id}_l1{l1_tag}_full.csv")
    pd.DataFrame(errs).to_csv(full_fn, header=False, index=False)

    print(f"[done] Saved full errors to {full_fn} ({errs.shape[0]} pairs × {errs.shape[1]-2} genes)")

if __name__ == "__main__":
    main()
