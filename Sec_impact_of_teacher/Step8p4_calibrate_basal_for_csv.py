#!/usr/bin/env python3
# calibrate_basal_for_csv.py
import os, argparse, numpy as np
import torch, torch.nn as nn
from torchdiffeq import odeint
import pandas as pd

"""
Run like this:
python Step8p4_calibrate_basal_for_csv.py \
   --code_csv YourHillModelParameterMatrix.csv \
   --data_csv data/all_ctgxyz_27genes_fillrand_fillzero_t012345_noshuffle.csv \
   --test_cells data/test_cells.csv \
   --method rk4 --nseg 5 \
   --steps 200 --lr 0.01 --v_prune_thresh 1e-3

"""


torch.set_default_dtype(torch.float64)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-12

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

def _freeze_eval_except_b0(field: 'HillField') -> 'HillField':
    field.to(DEVICE).requires_grad_(False).eval()
    field.b0.requires_grad_(True)  # only b0 is learnable here
    return field

# ------------------------------ Loaders ------------------------------
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
    return _freeze_eval_except_b0(field), G

# ------------------------------ V pruning ------------------------------
@torch.no_grad()
def prune_small_V_(field: HillField, thresh: float) -> tuple[int,int,int]:
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
def integrate_one_step(field: HillField, X0: torch.Tensor, method="rk4", rtol=1e-12, atol=1e-12, nseg=5) -> torch.Tensor:
    # differentiable integration (no no_grad here)
    X = torch.clamp(X0, min=EPS)
    for k in range(nseg):
        tseg = torch.tensor([k/nseg, (k+1)/nseg], dtype=torch.float64, device=DEVICE)
        traj = odeint(field, X, tseg, method=method, rtol=rtol, atol=atol)
        X = torch.clamp(traj[-1], min=EPS)
    return X

# ------------------------------ Save CSV ------------------------------
def save_field_to_csv(path_out: str, field: HillField):
    G = field.G
    with torch.no_grad():
        b0 = field.b0.detach().cpu().numpy()[None, :]
        V  = field.V.detach().cpu().numpy()
        K  = field.K.detach().cpu().numpy()
        n  = field.n.detach().cpu().numpy()
        gamma = torch.exp(field.log_gamma).detach().cpu().numpy()[None, :]
        M = np.concatenate([b0, V, K, n, gamma], axis=0)
    np.savetxt(path_out, M, delimiter=",", fmt="%.10g")

# ------------------------------ Main ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code_csv", required=True)
    ap.add_argument("--data_csv", required=True)
    ap.add_argument("--test_cells", required=True)
    ap.add_argument("--method", default="rk4", choices=["rk4","dopri5","euler","bdf","adams"])
    ap.add_argument("--rtol", type=float, default=1e-12)
    ap.add_argument("--atol", type=float, default=1e-12)
    ap.add_argument("--nseg", type=int, default=5)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--v_prune_thresh", type=float, default=1e-3)
    ap.add_argument("--lambda_reg", type=float, default=1e-3, help="L2 on b0")
    args = ap.parse_args()

    # Load model
    field, G = load_field_from_csv(args.code_csv)
    field = field.to(DEVICE)  # ensure on one device
    # prune small V in-place
    npr, nnz, tot = prune_small_V_(field, float(args.v_prune_thresh))
    print(f"[info] V-prune |V|<{args.v_prune_thresh:g}: pruned {npr}/{tot}, nnz after={nnz}")

    # sanity: only b0 requires grad
    for name, p in field.named_parameters():
        if name != "b0":
            p.requires_grad_(False)
    field.b0.requires_grad_(True)

    # Load data / match pairs for W=1 windows (0->1,1->2,...)
    data = pd.read_csv(args.data_csv, header=None).values
    N, D = data.shape
    gene_start, gene_end = 2, 2 + G
    if D < gene_end + 3:
        raise RuntimeError(f"{args.data_csv}: has {D} cols; need at least {gene_end+3}")

    cids_all  = data[:, 0].astype(np.int64)
    times_all = data[:, 1].astype(np.int64)
    uniq_times = np.unique(times_all); uniq_times.sort()
    t_windows = [(t0, t0+1) for t0 in uniq_times if (t0+1) in uniq_times]  # W=1 calibration
    test_ids = load_test_ids(args.test_cells)

    Xs, Ys = [], []
    for (t0, t1) in t_windows:
        block_t0 = data[times_all == t0, :]
        block_t1 = data[times_all == t1, :]
        mask_t0 = np.isin(block_t0[:,0].astype(np.int64), test_ids)
        block_t0 = block_t0[mask_t0, :]
        if block_t0.shape[0] == 0:
            continue
        X_now, Y_true, _, _ = collate_truth_by_cid(block_t0, block_t1, (gene_start, gene_end))
        if X_now is not None and X_now.shape[0] > 0:
            Xs.append(X_now); Ys.append(Y_true)
    if not Xs:
        raise RuntimeError("No matched rows found for W=1")

    X = torch.tensor(np.vstack(Xs), dtype=torch.float64, device=DEVICE)
    Y = torch.tensor(np.vstack(Ys), dtype=torch.float64, device=DEVICE)
    print(f"[info] matched pairs: {X.shape[0]}  genes: {X.shape[1]}")

    # Optimizer on *only* b0 with consistent device/dtype
    params = [field.b0]
    # tiny device audit
    print("[audit] optimizer params:",
          [(p.shape, p.dtype, p.device) for p in params])

    # Try Adam with foreach=False; fall back to basic Adam if flag unsupported
    opt = None
    try:
        opt = torch.optim.Adam(params, lr=args.lr, foreach=False)
    except TypeError:
        opt = torch.optim.Adam(params, lr=args.lr)

    # Training loop
    for step in range(1, args.steps+1):
        opt.zero_grad(set_to_none=True)
        Y_pred = integrate_one_step(field, X, method=args.method,
                                    rtol=args.rtol, atol=args.atol, nseg=args.nseg)
        mse = torch.mean((Y_pred - Y)**2)
        reg = args.lambda_reg * torch.mean(field.b0**2)
        loss = mse + reg
        loss.backward()
        opt.step()

        if step == 1 or step % 20 == 0 or step == args.steps:
            with torch.no_grad():
                mae = torch.mean(torch.abs(Y_pred - Y))
            print(f"[step {step:4d}/{args.steps}] loss={loss.item():.9e}  "
                  f"mse={mse.item():.9e}  mae={mae.item():.9e}")

    # Save calibrated CSV
    base, ext = os.path.splitext(args.code_csv)
    out = f"{base}_b0Cal.csv"
    save_field_to_csv(out, field)
    print(f"[done] Saved calibrated model -> {out}")

    # quick final report
    with torch.no_grad():
        Y_pred = integrate_one_step(field, X, method=args.method,
                                    rtol=args.rtol, atol=args.atol, nseg=args.nseg)
        mse = torch.mean((Y_pred - Y)**2).item()
        mae = torch.mean(torch.abs(Y_pred - Y)).item()
    print(f"RESULT code_csv={os.path.basename(out)} pairs={X.shape[0]} genes={G} MAE={mae:.7f} MSE={mse:.7f}")

if __name__ == "__main__":
    main()
