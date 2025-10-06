#!/usr/bin/env python3
# Calibrate per-gene decay (gamma) scale, and optionally per-gene production scale, for a Hill-model CSV.
# Output: a new CSV with updated (b0, V, K, n, gamma).

import os
import argparse
import numpy as np
import pandas as pd
import torch

"""
Run like this:
python Step8p3_calibrate_decay_vector_for_csv.py \
   --code_csv YourHillModelParameterMatrix.csv \
   --data_csv data/all_ctgxyz_27genes_fillrand_fillzero_t012345_noshuffle.csv \
   --test_cells data/test_cells.csv \
   --method rk4 --nseg 5 \
   --steps 200 --lr 0.01 --v_prune_thresh 1e-3

"""


torch.set_default_dtype(torch.float64)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-12


# ------------------------------ CSV Loader & Saver ------------------------------
def load_hill_from_csv(path: str):
    """Return (b0, V, K, n, gamma) as float64 tensors on DEVICE, G inferred."""
    M = np.loadtxt(path, delimiter=",")
    def try_parse(A):
        r, c = A.shape
        return c if (r >= 5 and r == 3 * c + 2) else None

    G = try_parse(M)
    if G is None:
        Mt = M.T
        Gt = try_parse(Mt)
        if Gt is None:
            raise RuntimeError(f"{path}: expect (3G+2, G) or (G, 3G+2), got {M.shape}")
        M = Mt
        G = Gt

    b0    = torch.tensor(M[0:1, :].squeeze(0), dtype=torch.float64, device=DEVICE)       # (G,)
    V     = torch.tensor(M[1:1+G, :], dtype=torch.float64, device=DEVICE)                 # (G,G)
    K     = torch.tensor(M[1+G:1+2*G, :], dtype=torch.float64, device=DEVICE)             # (G,G)
    n     = torch.tensor(M[1+2*G:1+3*G, :], dtype=torch.float64, device=DEVICE)           # (G,G)
    gamma = torch.tensor(M[1+3*G:1+3*G+1, :].squeeze(0), dtype=torch.float64, device=DEVICE)  # (G,)

    return b0, V, K, n, gamma, G


def save_hill_to_csv(path_out: str, b0, V, K, n, gamma):
    """Write in (3G+2, G) layout."""
    G = V.shape[0]
    rows = []
    rows.append(b0.detach().cpu().numpy()[None, :])           # 1 x G
    rows.append(V.detach().cpu().numpy())                     # G x G
    rows.append(K.detach().cpu().numpy())                     # G x G
    rows.append(n.detach().cpu().numpy())                     # G x G
    rows.append(gamma.detach().cpu().numpy()[None, :])        # 1 x G
    M = np.vstack(rows)                                       # (3G+2) x G
    np.savetxt(path_out, M, delimiter=",")


# ------------------------------ Data helpers ------------------------------
def load_test_ids(path: str) -> np.ndarray:
    ids = pd.read_csv(path, header=None).iloc[:, 0].to_numpy().astype(np.int64)
    return np.unique(ids)


def collate_pairs_W1(data_np: np.ndarray, G: int, test_ids: np.ndarray):
    """
    Build (X_t -> X_{t+1}) matched pairs for test_ids.
    data columns: [cid, time, g0..g(G-1), x, y, z]
    """
    cids_all = data_np[:, 0].astype(np.int64)
    times_all = data_np[:, 1].astype(np.int64)
    uniq = np.unique(times_all)
    uniq.sort()

    gene_start, gene_end = 2, 2 + G
    Xs, Ys = [], []

    for t0 in uniq:
        t1 = t0 + 1
        if t1 not in uniq:
            continue
        block_t0 = data_np[times_all == t0, :]
        block_t1 = data_np[times_all == t1, :]

        # filter start block to test ids
        mask_t0 = np.isin(block_t0[:, 0].astype(np.int64), test_ids)
        block_t0 = block_t0[mask_t0, :]
        if block_t0.shape[0] == 0:
            continue

        cids_t0 = block_t0[:, 0].astype(np.int64)
        genes_t0 = block_t0[:, gene_start:gene_end]
        # map t1 by cid
        cids_t1 = block_t1[:, 0].astype(np.int64)
        genes_t1 = block_t1[:, gene_start:gene_end]
        next_map = {int(c): genes_t1[i] for i, c in enumerate(cids_t1)}

        for i, cid in enumerate(cids_t0):
            if cid in next_map:
                Xs.append(genes_t0[i])
                Ys.append(next_map[cid])

    if not Xs:
        raise RuntimeError("No matched (t,t+1) pairs found for provided test IDs.")

    X = torch.tensor(np.stack(Xs, axis=0), dtype=torch.float64, device=DEVICE)
    Y = torch.tensor(np.stack(Ys, axis=0), dtype=torch.float64, device=DEVICE)
    return X, Y


# ------------------------------ Hill dynamics (differentiable) ------------------------------
def hill_forward_dxdt(X, b0, V, K, n, gamma):
    """
    X: (B,G)
    b0: (G,)
    V:  (G,G)  rows = targets i, cols = sources j
    K,n: (G,G)
    gamma: (G,)
    return dx/dt: (B,G)
    """
    X = torch.clamp(X, min=EPS)
    B, G = X.shape

    # shape to broadcast per target i and source j
    xj = X[:, None, :].expand(B, G, G)                 # (B,G,G)
    n_eff = torch.clamp(n, min=1e-2, max=10.0)[None]   # (1,G,G)
    K_eff = torch.clamp(K, min=1e-4, max=1e4)[None]    # (1,G,G)

    num = torch.pow(xj, n_eff)
    den = torch.pow(K_eff, n_eff) + num
    h = num / (den + EPS)
    h = torch.nan_to_num(h, nan=0.0, posinf=1.0, neginf=0.0)

    # sum over sources j
    prod_term = (h * V[None, :, :]).sum(dim=2) + b0[None, :]    # (B,G)
    dxdt = prod_term - gamma[None, :] * X
    dxdt = torch.nan_to_num(dxdt, nan=0.0, posinf=1e6, neginf=-1e6)
    return dxdt


def rk4_integrate(X0, steps, b0, V, K, n, gamma):
    """
    One physical time unit split into `steps` micro-steps of dt=1/steps using RK4.
    All tensors must be float64 on the same device.
    """
    dt = 1.0 / steps
    X = X0
    for _ in range(steps):
        k1 = hill_forward_dxdt(X, b0, V, K, n, gamma)
        k2 = hill_forward_dxdt(X + 0.5 * dt * k1, b0, V, K, n, gamma)
        k3 = hill_forward_dxdt(X + 0.5 * dt * k2, b0, V, K, n, gamma)
        k4 = hill_forward_dxdt(X + dt * k3, b0, V, K, n, gamma)
        X = X + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        X = torch.clamp(X, min=EPS)
    return X


# ------------------------------ Main ------------------------------
def main():
    ap = argparse.ArgumentParser(description="Calibrate per-gene decay (and optional production) scales for a Hill CSV.")
    ap.add_argument("--code_csv", required=True, help="Input Hill CSV (3G+2, G layout).")
    ap.add_argument("--data_csv", required=True)
    ap.add_argument("--test_cells", required=True)
    ap.add_argument("--nseg", type=int, default=5, help="RK4 micro-steps per Δt=1")
    ap.add_argument("--steps", type=int, default=200, help="Optimizer steps")
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--v_prune_thresh", type=float, default=1e-3, help="Hard zero |V_ij| below this before calibration")
    ap.add_argument("--lambda_reg", type=float, default=1e-3, help="L2 reg on log-scales")
    ap.add_argument("--learn_prod_scale", action="store_true", help="Also learn per-gene production scale (b0 and V rows)")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # --- Load model & data (on DEVICE, float64) ---
    b0_base, V_base, K, n, gamma_base, G = load_hill_from_csv(args.code_csv)

    # prune small Vij (constant during calibration)
    with torch.no_grad():
        mask_keep = (V_base.abs() >= args.v_prune_thresh)
        V_base = V_base * mask_keep
    print(f"[info] V-prune |V|<{args.v_prune_thresh:g}: pruned {(~mask_keep).sum().item()}/{mask_keep.numel()}, nnz after={int(mask_keep.sum().item())}")

    data_np = pd.read_csv(args.data_csv, header=None).values
    test_ids = load_test_ids(args.test_cells)
    X0_all, Y_true_all = collate_pairs_W1(data_np, G, test_ids)
    print(f"[info] matched pairs: {X0_all.shape[0]}  genes: {G}")

    # --- Learnable log-scales (float64, on DEVICE) ---
    log_s_d = torch.nn.Parameter(torch.zeros(G, dtype=torch.float64, device=DEVICE), requires_grad=True)  # decay scale per gene
    log_s_p = None
    if args.learn_prod_scale:
        log_s_p = torch.nn.Parameter(torch.zeros(G, dtype=torch.float64, device=DEVICE), requires_grad=True)  # production scale per gene

    params = [log_s_d] + ([log_s_p] if log_s_p is not None else [])

    # Adam with foreach disabled (avoids mixed-device grouping issues)
    try:
        opt = torch.optim.Adam(params, lr=args.lr, foreach=False)
    except TypeError:
        opt = torch.optim.Adam(params, lr=args.lr)

    # --- Optimize ---
    for it in range(args.steps):
        opt.zero_grad()

        # current effective scales
        s_d = torch.exp(log_s_d)                                # (G,)
        gamma_eff = gamma_base * s_d                            # (G,)

        if log_s_p is not None:
            s_p = torch.exp(log_s_p)                            # (G,)
            b0_eff = b0_base * s_p                              # (G,)
            V_eff  = V_base * s_p[:, None]                      # (G,G) row-wise scale
        else:
            b0_eff = b0_base
            V_eff  = V_base

        # integrate one Δt=1 with RK4 (differentiable)
        Y_pred = rk4_integrate(X0_all, args.nseg, b0_eff, V_eff, K, n, gamma_eff)

        # MSE loss
        mse = torch.mean((Y_pred - Y_true_all) ** 2)

        # small L2 on log-scales to discourage huge drifts
        reg = args.lambda_reg * (torch.sum(log_s_d**2) + (torch.sum(log_s_p**2) if log_s_p is not None else 0.0))
        loss = mse + reg

        loss.backward()
        opt.step()

        # (optional) print a small heartbeat
        if (it + 1) % max(1, args.steps // 10) == 0 or it == 0:
            with torch.no_grad():
                mae = torch.mean(torch.abs(Y_pred - Y_true_all)).item()
                print(f"[step {it+1:4d}/{args.steps}] loss={loss.item():.6e}  mse={mse.item():.6e}  mae={mae:.6e}")

    # --- Final eval and save ---
    with torch.no_grad():
        s_d = torch.exp(log_s_d)
        gamma_cal = gamma_base * s_d

        if log_s_p is not None:
            s_p = torch.exp(log_s_p)
            b0_cal = b0_base * s_p
            V_cal  = V_base  * s_p[:, None]
            suffix = "_decprodCal.csv"
        else:
            b0_cal = b0_base
            V_cal  = V_base
            suffix = "_decCal.csv"

        # Final metrics
        Y_pred = rk4_integrate(X0_all, args.nseg, b0_cal, V_cal, K, n, gamma_cal)
        mae = torch.mean(torch.abs(Y_pred - Y_true_all)).item()
        mse = torch.mean((Y_pred - Y_true_all) ** 2).item()
        print("\n=== Final metrics on matched pairs ===")
        print(f"MAE={mae:.7f}, MSE={mse:.7f}")

        # Save CSV
        base, ext = os.path.splitext(args.code_csv)
        out_path = base + suffix
        save_hill_to_csv(out_path, b0_cal, V_cal, K, n, gamma_cal)
        print(f"[done] Saved calibrated model -> {out_path}")

        # Also print a one-line RESULT line (handy for grepping)
        print(f"RESULT code_csv={os.path.basename(out_path)} pairs={X0_all.shape[0]} genes={G} MAE={mae:.7f} MSE={mse:.7f}")


if __name__ == "__main__":
    main()
