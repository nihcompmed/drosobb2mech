#!/usr/bin/env python3
# Step8_integrateHill_zerosmallVij_code2_window.py
"""
Evaluate a single Hill model ("Code2") by multi-step integration against truth.

- Loads Code2 (csv/pt/ckpt supported)
- Hard-zeros small |V_ij| < v_prune_thresh
- Integrates windows of size W (e.g., 1: t->t+1, 2: 0->2, 1->3, ...)
- Computes per-gene MAE and overall MAE/MSE for Code2 vs truth

Example:
python Step8_integrateHill_zerosmallVij_code2_window.py \
  --code2 YourHillModelParameterMatrix.csv \
  --data_csv data/all_ctgxyz_27genes_fillrand_fillzero_t012345_noshuffle.csv \
  --test_cells data/test_cells.csv \
  --method dopri5 --rtol 1e-12 --atol 1e-12 --nseg 20 \
  --window 1 --v_prune_thresh 1e-3
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchdiffeq import odeint
from tqdm.auto import tqdm

torch.set_default_dtype(torch.float64)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS    = 1e-12

# ------------------------------ Hill field ------------------------------
class HillField(nn.Module):
    """
    dx_i/dt = α * ( b0_i + Σ_j V_{i,j} * H(x_j; K_{i,j}, n_{i,j}) - γ_i * x_i )
    H(x;K,n) = x^n / (K^n + x^n)
    """
    def __init__(self, G: int):
        super().__init__()
        self.G = G
        self.b0 = nn.Parameter(torch.zeros(G, dtype=torch.float64))
        self.V  = nn.Parameter(torch.zeros(G, G, dtype=torch.float64))
        self.K  = nn.Parameter(torch.ones(G, G,  dtype=torch.float64))
        self.n  = nn.Parameter(torch.ones(G, G,  dtype=torch.float64))
        self.log_gamma = nn.Parameter(torch.zeros(G, dtype=torch.float64))
        self.log_alpha = nn.Parameter(torch.zeros(1, dtype=torch.float64))  # α=1 default

    def forward(self, t, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        B, G = x.shape
        assert G == self.G

        x_safe = torch.clamp(x, min=EPS)
        n_eff  = torch.clamp(self.n, min=1e-2, max=10.0)[None]
        K_eff  = torch.clamp(self.K, min=1e-4, max=1e4)[None]

        xj  = x_safe[:, None, :].expand(B, G, G)
        num = torch.pow(xj, n_eff)
        den = torch.pow(K_eff, n_eff) + num
        h   = num / (den + EPS)
        h   = torch.nan_to_num(h, nan=0.0, posinf=1.0, neginf=0.0)

        prod = (h * self.V[None, :, :]).sum(dim=2) + self.b0

        lg = torch.clamp(self.log_gamma, min=-20, max=20)
        la = torch.clamp(self.log_alpha, min=-20, max=20)
        gamma = torch.exp(lg)
        alpha = torch.exp(la)[0]

        dxdt = alpha * (prod - gamma * x_safe)
        dxdt = torch.nan_to_num(dxdt, nan=0.0, posinf=1e6, neginf=-1e6)
        return dxdt

def _freeze_eval(field: 'HillField') -> 'HillField':
    field.to(DEVICE).eval()
    for p in field.parameters():
        p.requires_grad_(False)
    return field

# ------------------------------ Loaders ------------------------------
def load_field_from_pt(path: str):
    sd = torch.load(path, map_location="cpu")
    if "V" not in sd:
        raise RuntimeError(f"{path}: state dict missing 'V'.")
    G = int(sd["V"].shape[0])
    field = HillField(G)
    field.load_state_dict(sd, strict=False)
    return _freeze_eval(field), G, "pt"

def load_field_from_ckpt(path: str):
    ckpt = torch.load(path, map_location="cpu")
    if "state_dict" not in ckpt:
        raise RuntimeError(f"{path}: not a Lightning checkpoint (no 'state_dict').")
    sfull = ckpt["state_dict"]
    need = ["b0", "V", "K", "n", "log_gamma", "log_alpha"]
    s = {}
    for k in need:
        fk = f"field.{k}"
        if fk in sfull:
            s[k] = sfull[fk].to(torch.float64)
        elif k in sfull:
            s[k] = sfull[k].to(torch.float64)
        else:
            s[k] = torch.zeros(1, dtype=torch.float64) if k == "log_alpha" else None
    if any(v is None for v in s.values()):
        missing = [k for k, v in s.items() if v is None]
        raise RuntimeError(f"{path}: missing keys {missing}")
    G = int(s["V"].shape[0])
    field = HillField(G)
    field.load_state_dict(s, strict=False)
    return _freeze_eval(field), G, "ckpt"

def load_field_from_csv(path: str):
    M = np.loadtxt(path, delimiter=",")
    def try_parse(A):
        r, c = A.shape
        if r >= 5 and c >= 1 and r == 3*c + 2:
            return c
        return None
    G = try_parse(M)
    if G is None:
        Mt = M.T
        Gt = try_parse(Mt)
        if Gt is None:
            raise RuntimeError(f"{path}: expect (3G+2, G) or (G, 3G+2), got {M.shape}")
        M = Mt
        G = Gt

    b0    = M[0:1, :]
    V     = M[1:1+G, :]
    K     = M[1+G:1+2*G, :]
    n     = M[1+2*G:1+3*G, :]
    gamma = M[1+3*G:1+3*G+1, :]

    sd = {
        "b0": torch.tensor(b0.squeeze(0), dtype=torch.float64),
        "V":  torch.tensor(V, dtype=torch.float64),
        "K":  torch.tensor(K, dtype=torch.float64),
        "n":  torch.tensor(n, dtype=torch.float64),
        "log_gamma": torch.tensor(np.log(np.clip(gamma.squeeze(0), 1e-20, None)), dtype=torch.float64),
        "log_alpha": torch.zeros(1, dtype=torch.float64),
    }
    field = HillField(G)
    field.load_state_dict(sd, strict=False)
    return _freeze_eval(field), G, "csv"

def load_hill_model(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pt":
        return load_field_from_pt(path)
    if ext == ".ckpt":
        return load_field_from_ckpt(path)
    if ext == ".csv":
        return load_field_from_csv(path)
    raise ValueError(f"Unknown model file extension for '{path}'")

# ------------------------------ V pruning ------------------------------
def prune_small_V(field: HillField, thresh: float) -> tuple[int,int,int]:
    """
    Hard-zero entries of V with |V_ij| < thresh.
    Returns (num_pruned, num_nonzero_after, total_entries).
    """
    with torch.no_grad():
        V = field.V.data
        total = V.numel()
        mask_small = V.abs() < thresh
        num_pruned = int(mask_small.sum().item())
        V[mask_small] = 0.0
        nnz_after = int((V != 0).sum().item())
    return num_pruned, nnz_after, total

# ------------------------------ Data helpers ------------------------------
def load_test_ids(path: str) -> np.ndarray:
    ids = pd.read_csv(path, header=None).iloc[:, 0].to_numpy().astype(np.int64)
    return np.unique(ids)

def collate_truth_by_cid(block_t: np.ndarray, block_tp: np.ndarray, gene_slice: tuple[int,int]):
    CID_COL = 0
    cids_t    = block_t[:, CID_COL].astype(np.int64)
    cids_tp   = block_tp[:, CID_COL].astype(np.int64)
    genes_tp  = block_tp[:, gene_slice[0]:gene_slice[1]]
    xyz_t     = block_t[:, -3:]
    next_map = {int(c): genes_tp[i] for i, c in enumerate(cids_tp)}

    X_list, Y_list, CID_list, XYZ_list = [], [], [], []
    for i, c in enumerate(cids_t):
        if c in next_map:
            X_list.append(block_t[i, gene_slice[0]:gene_slice[1]])
            Y_list.append(next_map[c])
            CID_list.append(c)
            XYZ_list.append(xyz_t[i])
    if not X_list:
        return None, None, None, None
    return (np.stack(X_list, axis=0),
            np.stack(Y_list, axis=0),
            np.array(CID_list, dtype=np.int64),
            np.stack(XYZ_list, axis=0))

# ------------------------------ Integration ------------------------------
@torch.no_grad()
def integrate_one_step(field: HillField, X0: torch.Tensor, method="dopri5", rtol=1e-12, atol=1e-12, nseg=20) -> torch.Tensor:
    X = torch.clamp(X0, min=EPS)
    for k in range(nseg):
        tseg = torch.tensor([k/nseg, (k+1)/nseg], dtype=torch.float64, device=DEVICE)
        traj = odeint(field, X, tseg, method=method, rtol=rtol, atol=atol)
        X = torch.clamp(traj[-1], min=EPS)
    return X

@torch.no_grad()
def integrate_window(field: HillField, X0: torch.Tensor, W: int, method="dopri5", rtol=1e-12, atol=1e-12, nseg=20) -> torch.Tensor:
    """Integrate W consecutive Δt=1 steps (e.g., W=3: t→t+3), without re-initializing to truth in between."""
    X = X0
    for _ in range(W):
        X = integrate_one_step(field, X, method=method, rtol=rtol, atol=atol, nseg=nseg)
    return X

# ------------------------------ Main ------------------------------
def main():
    ap = argparse.ArgumentParser(description="Evaluate one Hill model by windowed integration (with V pruning).")
    ap.add_argument("--code2", required=True, help="Path to the Hill model to evaluate (.csv/.pt/.ckpt).")
    ap.add_argument("--data_csv", required=True)
    ap.add_argument("--test_cells", required=True)
    ap.add_argument("--method", default="dopri5", choices=["dopri5","rk4","euler","bdf","adams"])
    ap.add_argument("--rtol", type=float, default=1e-12)
    ap.add_argument("--atol", type=float, default=1e-12)
    ap.add_argument("--nseg", type=int, default=20, help="macro-steps within each Δt=1")
    ap.add_argument("--window", type=int, default=1, help="1..5: 1=Δt=1, 2=0→2, 3=0→3, ...")
    ap.add_argument("--max_pairs", type=int, default=None)
    ap.add_argument("--v_prune_thresh", type=float, default=1e-3,
                    help="Hard-zero |V_ij| below this threshold for the model.")
    args = ap.parse_args()

    if not (1 <= args.window <= 5):
        raise ValueError("--window must be in {1,2,3,4,5}")

    # Load model
    field2, G, kind2 = load_hill_model(args.code2)
    print(f"[info] Loaded model: Code2({kind2}, G={G}). Device={DEVICE.type}")

    # Prune small V
    t = float(args.v_prune_thresh)
    npr2, nnz2, tot2 = prune_small_V(field2, t)
    print(f"[info] V-prune |V|<{t:g}: Code2 pruned {npr2}/{tot2}, nnz after={nnz2}")

    # Load data & IDs
    data = pd.read_csv(args.data_csv, header=None).values
    N, D = data.shape
    gene_start, gene_end = 2, 2 + G
    if D < gene_end + 3:
        raise RuntimeError(f"{args.data_csv}: has {D} cols; expected at least {gene_end+3} (cid,time,{G} genes, xyz[3]).")
    cids_all  = data[:, 0].astype(np.int64)
    times_all = data[:, 1].astype(np.int64)
    uniq_times = np.unique(times_all)
    uniq_times.sort()

    W = args.window
    t_windows = [(t0, t0 + W) for t0 in uniq_times if (t0 + W) in uniq_times]
    if not t_windows:
        raise RuntimeError(f"No valid time windows for window={W} using times {uniq_times.tolist()}")
    print(f"[info] Using windows (t0→t1) with W={W}: {t_windows}")

    test_ids = load_test_ids(args.test_cells)
    print(f"[info] Restricting to {test_ids.size} test cell IDs.")

    preds2, truths = [], []
    total_matched = 0

    for (t0, t1) in tqdm(t_windows, desc=f"Integrating windows of size {W}"):
        block_t0 = data[times_all == t0, :]
        block_t1 = data[times_all == t1, :]

        # filter to test IDs on start block
        mask_t0 = np.isin(block_t0[:, 0].astype(np.int64), test_ids)
        block_t0 = block_t0[mask_t0, :]
        if block_t0.shape[0] == 0:
            continue

        X_now, Y_true, _, _ = collate_truth_by_cid(block_t0, block_t1, (gene_start, gene_end))
        if X_now is None or X_now.shape[0] == 0:
            continue

        if args.max_pairs is not None and total_matched + X_now.shape[0] > args.max_pairs:
            take = max(0, args.max_pairs - total_matched)
            if take == 0:
                break
            X_now  = X_now[:take]
            Y_true = Y_true[:take]

        X0_t = torch.tensor(X_now, dtype=torch.float64, device=DEVICE)
        with torch.no_grad():
            Y_pred2 = integrate_window(field2, X0_t, W,
                                       method=args.method, rtol=args.rtol, atol=args.atol, nseg=args.nseg).cpu().numpy()

        preds2.append(Y_pred2)
        truths.append(Y_true)

        total_matched += X_now.shape[0]
        if args.max_pairs is not None and total_matched >= args.max_pairs:
            break

    if not preds2:
        raise RuntimeError("No matched rows found for any window.")

    P2 = np.vstack(preds2)  # code2 predictions
    T  = np.vstack(truths)  # truth

    # ---------- Per-gene MAE ----------
    dbb = np.mean(np.abs(P2 - T), axis=0)  # Code2 vs Truth

    # overall (scalar) MAE & MSE
    overall_mae_code2 = float(np.mean(np.abs(P2 - T)))
    overall_mse_code2 = float(np.mean((P2 - T)**2))

    np.set_printoptions(precision=6, suppress=True, linewidth=200)
    print("\n=== Windowed integration results (Code2 only) ===")
    print(f"Model: {args.code2}")
    print(f"Window size W = {W}  (windows used: {t_windows})")
    print(f"Matched pairs: {P2.shape[0]}   Genes: {P2.shape[1]}")

    print("\nPer-gene MAE (mean(abs(...))) for Code2:")
    print(dbb)

    print("\nOverall metrics (Code2):")
    print(f"MAE={overall_mae_code2:.6g}, MSE={overall_mse_code2:.6g}")

    # Parse-friendly single line:
    print(f"RESULT code2={os.path.basename(args.code2)} W={W} pairs={P2.shape[0]} genes={P2.shape[1]} "
          f"MAE={overall_mae_code2:.6g} MSE={overall_mse_code2:.6g}")

if __name__ == "__main__":
    main()
