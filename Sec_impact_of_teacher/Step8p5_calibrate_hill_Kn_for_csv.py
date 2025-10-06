#!/usr/bin/env python3
import os, argparse, numpy as np, pandas as pd, torch, torch.nn as nn
from torchdiffeq import odeint


"""
Run like this:
python Step8p5_calibrate_hill_Kn_for_csv.py \
   --code_csv YourHillModelParameterMatrix.csv \
   --data_csv data/all_ctgxyz_27genes_fillrand_fillzero_t012345_noshuffle.csv \
   --test_cells data/test_cells.csv \
   --method rk4 --nseg 5 \
   --steps 200 --lr 0.01 --v_prune_thresh 1e-3
   --tune nz --lambda_kn 1e-3
"""


torch.set_default_dtype(torch.float64)
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
EPS = 1e-12

class HillFieldKN(nn.Module):
    def __init__(self, b0, V, K_init, n_init, log_gamma, log_alpha=0.0,
                 n_min=0.2, n_max=10.0):
        super().__init__()
        G = V.shape[0]
        self.G = G
        # frozen parts
        self.b0 = nn.Parameter(b0.to(torch.float64).to(DEVICE), requires_grad=False)
        self.V  = nn.Parameter(V.to(torch.float64).to(DEVICE),  requires_grad=False)
        self.log_gamma = nn.Parameter(log_gamma.to(torch.float64).to(DEVICE), requires_grad=False)
        self.log_alpha = nn.Parameter(torch.tensor([log_alpha], dtype=torch.float64, device=DEVICE), requires_grad=False)
        # learnable
        logK0 = torch.log(K_init.to(torch.float64).clamp(min=1e-4, max=1e4)).to(DEVICE)
        self.logK_raw = nn.Parameter(logK0.clone(), requires_grad=True)
        self.n_raw    = nn.Parameter(n_init.to(torch.float64).to(DEVICE).clone(), requires_grad=True)
        self.n_min, self.n_max = float(n_min), float(n_max)

    def forward(self, t, x):
        if x.dim() == 1: x = x.unsqueeze(0)
        B, G = x.shape; assert G == self.G
        x = torch.clamp(x, min=EPS)

        K_eff = torch.exp(self.logK_raw).clamp(min=1e-4, max=1e4)[None, :, :]
        n_eff = self.n_raw.clamp(min=self.n_min, max=self.n_max)[None, :, :]

        xj  = x[:, None, :].expand(B, G, G)
        num = torch.pow(xj, n_eff)
        den = torch.pow(K_eff, n_eff) + num
        h   = num / (den + EPS)
        h   = torch.nan_to_num(h, nan=0.0, posinf=1.0, neginf=0.0)

        prod  = (h * self.V[None]).sum(dim=2) + self.b0
        gamma = torch.exp(self.log_gamma)
        alpha = torch.exp(self.log_alpha)[0]
        dxdt  = alpha * (prod - gamma * x)
        return torch.nan_to_num(dxdt, nan=0.0, posinf=1e6, neginf=-1e6)

def load_csv_field(path: str):
    M = np.loadtxt(path, delimiter=",")
    def try_parse(A):
        r, c = A.shape
        return c if (r >= 5 and r == 3*c + 2) else None
    G = try_parse(M)
    if G is None:
        Mt = M.T; Gt = try_parse(Mt)
        if Gt is None: raise RuntimeError(f"{path}: expect (3G+2,G) or (G,3G+2), got {M.shape}")
        M, G = Mt, Gt
    b0    = torch.tensor(M[0, :], dtype=torch.float64)
    V     = torch.tensor(M[1:1+G, :], dtype=torch.float64)
    K     = torch.tensor(M[1+G:1+2*G, :], dtype=torch.float64)
    n     = torch.tensor(M[1+2*G:1+3*G, :], dtype=torch.float64)
    gamma = torch.tensor(M[1+3*G, :], dtype=torch.float64)
    log_gamma = torch.log(gamma.clamp_min(1e-20))
    return b0, V, K, n, log_gamma, G

def save_csv_field(path_out: str, b0, V, K, n, log_gamma):
    G = V.shape[0]
    M = torch.zeros((3*G + 2, G), dtype=torch.float64)
    M[0, :]           = b0
    M[1:1+G, :]       = V
    M[1+G:1+2*G, :]   = K
    M[1+2*G:1+3*G, :] = n
    M[1+3*G, :]       = torch.exp(log_gamma)
    np.savetxt(path_out, M.cpu().numpy(), delimiter=",", fmt="%.10g")

def load_test_ids(path: str) -> np.ndarray:
    return pd.read_csv(path, header=None).iloc[:, 0].to_numpy().astype(np.int64).unique()

def collate_truth_by_cid(block_t, block_tp, gene_slice):
    CID = 0
    c_t  = block_t[:, CID].astype(np.int64)
    c_tp = block_tp[:, CID].astype(np.int64)
    ytp  = block_tp[:, gene_slice[0]:gene_slice[1]]
    nxt  = {int(c): ytp[i] for i, c in enumerate(c_tp)}
    X, Y = [], []
    for i, c in enumerate(c_t):
        if c in nxt:
            X.append(block_t[i, gene_slice[0]:gene_slice[1]])
            Y.append(nxt[c])
    if not X: return None, None
    return np.stack(X, 0), np.stack(Y, 0)

def integrate_one_step(field, X0, method="rk4", rtol=1e-12, atol=1e-12, nseg=5):
    """NO @torch.no_grad(): keep grads for K/n!"""
    X = torch.clamp(X0, min=EPS)
    for k in range(nseg):
        tseg = torch.tensor([k/nseg, (k+1)/nseg], dtype=torch.float64, device=X.device)
        traj = odeint(field, X, tseg, method=method, rtol=rtol, atol=atol)
        X = torch.clamp(traj[-1], min=EPS)
    return X

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code_csv", required=True)
    ap.add_argument("--data_csv", required=True)
    ap.add_argument("--test_cells", required=True)
    ap.add_argument("--method", default="rk4", choices=["rk4","dopri5","euler","bdf","adams"])
    ap.add_argument("--rtol", type=float, default=1e-12)
    ap.add_argument("--atol", type=float, default=1e-12)
    ap.add_argument("--nseg", type=int, default=5)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--v_prune_thresh", type=float, default=1e-3)
    ap.add_argument("--tune", choices=["nz","all","diag"], default="nz")
    ap.add_argument("--lambda_kn", type=float, default=1e-3)
    ap.add_argument("--learn_K", action="store_true")
    ap.add_argument("--learn_n", action="store_true")
    ap.add_argument("--n_min", type=float, default=0.2)
    ap.add_argument("--n_max", type=float, default=10.0)
    ap.add_argument("--max_logK_delta", type=float, default=2.0)
    args = ap.parse_args()
    if not (args.learn_K or args.learn_n):
        args.learn_K, args.learn_n = True, True

    b0, V, K0, n0, log_gamma, G = load_csv_field(args.code_csv)
    # prune |V| < thresh for stability and mask construction
    with torch.no_grad():
        V_mask = V.clone()
        V_mask[torch.abs(V_mask) < args.v_prune_thresh] = 0.0

    # collect (t->t+1) pairs on test IDs
    data = pd.read_csv(args.data_csv, header=None).values
    times = data[:, 1].astype(np.int64)
    uniq = np.unique(times); uniq.sort()
    pairs = [(t0, t0+1) for t0 in uniq if (t0+1) in uniq]
    test_ids = pd.read_csv(args.test_cells, header=None).iloc[:,0].to_numpy().astype(np.int64)
    g0, g1 = 2, 2 + G
    Xs, Ys = [], []
    for t0, t1 in pairs:
        B0 = data[times == t0, :]
        B1 = data[times == t1, :]
        B0 = B0[np.isin(B0[:,0].astype(np.int64), test_ids)]
        if B0.shape[0] == 0: continue
        Xn, Yn = collate_truth_by_cid(B0, B1, (g0, g1))
        if Xn is None: continue
        Xs.append(Xn); Ys.append(Yn)
    if not Xs: raise RuntimeError("No matched pairs.")
    X = torch.tensor(np.vstack(Xs), dtype=torch.float64, device=DEVICE)
    Y = torch.tensor(np.vstack(Ys), dtype=torch.float64, device=DEVICE)
    print(f"[info] matched pairs: {X.shape[0]}  genes: {G}")

    field = HillFieldKN(b0=b0, V=V_mask, K_init=K0, n_init=n0, log_gamma=log_gamma,
                        log_alpha=0.0, n_min=args.n_min, n_max=args.n_max).to(DEVICE)

    # tuning masks
    if args.tune == "all":
        tune_mask = torch.ones((G,G), dtype=torch.bool, device=DEVICE)
    elif args.tune == "diag":
        tune_mask = torch.zeros((G,G), dtype=torch.bool, device=DEVICE); idx = torch.arange(G, device=DEVICE); tune_mask[idx,idx] = True
    else:  # "nz"
        tune_mask = (field.V.abs() >= args.v_prune_thresh)

    learn_K_mask = tune_mask if args.learn_K else torch.zeros_like(tune_mask)
    learn_n_mask = tune_mask if args.learn_n else torch.zeros_like(tune_mask)

    logK_init = field.logK_raw.detach().clone()
    n_init    = field.n_raw.detach().clone()

    # separate optimizers (avoid foreach/device grouping issues)
    opts = []
    if args.learn_K:
        try: optK = torch.optim.Adam([field.logK_raw], lr=args.lr, foreach=False)
        except TypeError: optK = torch.optim.Adam([field.logK_raw], lr=args.lr)
        opts.append(optK)
    if args.learn_n:
        try: optN = torch.optim.Adam([field.n_raw], lr=args.lr, foreach=False)
        except TypeError: optN = torch.optim.Adam([field.n_raw], lr=args.lr)
        opts.append(optN)

    audit = []
    if args.learn_K: audit.append((field.logK_raw.shape, field.logK_raw.dtype, field.logK_raw.device))
    if args.learn_n: audit.append((field.n_raw.shape,    field.n_raw.dtype,    field.n_raw.device))
    print(f"[audit] optimizer params: {audit}")

    # baseline print
    with torch.no_grad():
        base_pred = integrate_one_step(field, X, method=args.method, rtol=args.rtol, atol=args.atol, nseg=args.nseg)
        base_mse = torch.mean((base_pred - Y)**2).item()
        base_mae = torch.mean(torch.abs(base_pred - Y)).item()
    print(f"[step 000] baseline mse={base_mse:.9e}  mae={base_mae:.9e}")

    for step in range(1, args.steps+1):
        for o in opts: o.zero_grad(set_to_none=True)

        Y_pred = integrate_one_step(field, X, method=args.method, rtol=args.rtol, atol=args.atol, nseg=args.nseg)
        mse = torch.mean((Y_pred - Y)**2)

        reg = torch.tensor(0.0, dtype=torch.float64, device=DEVICE)
        if args.learn_K:
            dlogK = field.logK_raw - logK_init
            reg += args.lambda_kn * torch.mean((dlogK[learn_K_mask])**2)
        if args.learn_n:
            dn = field.n_raw - n_init
            reg += args.lambda_kn * torch.mean((dn[learn_n_mask])**2)

        loss = mse + reg
        loss.backward()

        # mask gradients
        if args.learn_K and field.logK_raw.grad is not None:
            field.logK_raw.grad[~learn_K_mask] = 0.0
        if args.learn_n and field.n_raw.grad is not None:
            field.n_raw.grad[~learn_n_mask] = 0.0

        for o in opts: o.step()

        with torch.no_grad():
            if args.learn_K:
                lo = logK_init - args.max_logK_delta
                hi = logK_init + args.max_logK_delta
                field.logK_raw[:] = torch.max(torch.min(field.logK_raw, hi), lo)
            if args.learn_n:
                field.n_raw.clamp_(min=args.n_min, max=args.n_max)

        if step == 1 or step % 20 == 0 or step == args.steps:
            with torch.no_grad():
                mae = torch.mean(torch.abs(Y_pred - Y)).item()
            print(f"[step {step:4d}/{args.steps}] loss={loss.item():.9e}  mse={mse.item():.9e}  mae={mae:.9e}")

    # save updated K,n (keep b0,V,gamma same as CSV)
    b0_, V_, _, _, logg_, _ = load_csv_field(args.code_csv)
    with torch.no_grad():
        K_new = torch.exp(field.logK_raw).clamp(min=1e-4, max=1e4).cpu()
        n_new = field.n_raw.clamp(min=args.n_min, max=args.n_max).cpu()
    out = os.path.splitext(args.code_csv)[0] + "_KnCal.csv"
    save_csv_field(out, b0_, V_, K_new, n_new, logg_)
    print(f"[done] Saved calibrated model -> {out}")

    with torch.no_grad():
        Yp = integrate_one_step(field, X, method=args.method, rtol=args.rtol, atol=args.atol, nseg=args.nseg)
        mae = float(torch.mean(torch.abs(Yp - Y)).item())
        mse = float(torch.mean((Yp - Y)**2).item())
    print(f"RESULT code_csv={os.path.basename(out)} pairs={X.shape[0]} genes={G} MAE={mae} MSE={mse}")

if __name__ == "__main__":
    main()
