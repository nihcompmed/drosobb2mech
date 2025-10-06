#!/usr/bin/env python3
# calibrate_alpha_for_csv.py
import os, argparse, numpy as np, pandas as pd, torch, torch.nn as nn
from torchdiffeq import odeint
"""
Run like this:
python Step8p1_calibrate_alpha_for_csv.py \
   --code_csv YourHillModelParameterMatrix.csv \
   --data_csv data/all_ctgxyz_27genes_fillrand_fillzero_t012345_noshuffle.csv \
   --test_cells data/test_cells.csv \
   --method rk4 --nseg 5 \
   --steps 200 --lr 0.01 --v_prune_thresh 1e-3

"""
torch.set_default_dtype(torch.float64)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-12

class HillField(nn.Module):
    def __init__(self, G):
        super().__init__()
        self.G = G
        self.b0 = nn.Parameter(torch.zeros(G, dtype=torch.float64), requires_grad=False)
        self.V  = nn.Parameter(torch.zeros(G, G, dtype=torch.float64), requires_grad=False)
        self.K  = nn.Parameter(torch.ones(G, G,  dtype=torch.float64), requires_grad=False)
        self.n  = nn.Parameter(torch.ones(G, G,  dtype=torch.float64), requires_grad=False)
        self.log_gamma = nn.Parameter(torch.zeros(G, dtype=torch.float64), requires_grad=False)
        self.log_alpha = nn.Parameter(torch.zeros(1, dtype=torch.float64), requires_grad=True)  # only thing we learn

    def forward(self, t, x):
        if x.dim() == 1: x = x.unsqueeze(0)
        x = torch.clamp(x, min=EPS)
        B, G = x.shape
        n = torch.clamp(self.n, min=1e-2, max=10.0)[None]
        K = torch.clamp(self.K, min=1e-4, max=1e4)[None]
        xj = x[:, None, :].expand(B, G, G)
        num = torch.pow(xj, n); den = torch.pow(K, n) + num
        h = torch.nan_to_num(num / (den + EPS), nan=0.0, posinf=1.0, neginf=0.0)
        prod = (h * self.V[None]).sum(dim=2) + self.b0
        gamma = torch.exp(torch.clamp(self.log_gamma, -20, 20))
        alpha = torch.exp(torch.clamp(self.log_alpha, -20, 20))[0]
        dxdt = alpha * (prod - gamma * x)
        return torch.nan_to_num(dxdt, nan=0.0, posinf=1e6, neginf=-1e6)

def load_csv_params(path):
    M = np.loadtxt(path, delimiter=",")
    def _try(A):
        r, c = A.shape
        return c if (r >= 5 and r == 3*c + 2) else None
    G = _try(M)
    if G is None:
        Mt = M.T; G = _try(Mt)
        if G is None: raise RuntimeError(f"{path}: expect (3G+2,G) or transpose; got {M.shape}")
        M = Mt
    b0    = M[0, :]
    V     = M[1:1+G, :]
    K     = M[1+G:1+2*G, :]
    n     = M[1+2*G:1+3*G, :]
    gamma = M[1+3*G, :]
    return G, b0, V, K, n, gamma

def save_csv_params(path, b0, V, K, n, gamma):
    G = V.shape[0]
    out = np.vstack([b0[None, :], V, K, n, gamma[None, :]])
    pd.DataFrame(out).to_csv(path, header=False, index=False)

def integrate_step(field, X0, method="rk4", nseg=5, rtol=1e-12, atol=1e-12):
    # NOTE: no @torch.no_grad here – we need gradients for alpha
    X = torch.clamp(X0, min=EPS)
    for k in range(nseg):
        tt = torch.tensor([k/nseg, (k+1)/nseg], dtype=torch.float64, device=X.device)
        X = odeint(field, X, tt, method=method, rtol=rtol, atol=atol)[-1]
        X = torch.clamp(X, min=EPS)
    return X

def build_pairs(data_csv, test_csv, G):
    data = pd.read_csv(data_csv, header=None).values
    test_ids = pd.read_csv(test_csv, header=None).iloc[:,0].to_numpy().astype(np.int64)
    test_ids = np.unique(test_ids)
    times = data[:,1].astype(np.int64)
    gene_start, gene_end = 2, 2+G
    uniq = np.unique(times); uniq.sort()
    pairs = []
    for t0 in uniq:
        t1 = t0 + 1
        if t1 not in uniq: continue
        Bt0 = data[times == t0]; Bt1 = data[times == t1]
        mask = np.isin(Bt0[:,0].astype(np.int64), test_ids)
        Bt0 = Bt0[mask]
        if Bt0.shape[0] == 0: continue
        map_next = {int(cid): Bt1[i, gene_start:gene_end] for i, cid in enumerate(Bt1[:,0].astype(np.int64))}
        Xs, Ys = [], []
        for row in Bt0:
            cid = int(row[0])
            if cid in map_next:
                Xs.append(row[gene_start:gene_end]); Ys.append(map_next[cid])
        if Xs: pairs.append((np.stack(Xs), np.stack(Ys)))
    if not pairs: raise RuntimeError("No matched (t,t+1) pairs found.")
    X = np.vstack([p[0] for p in pairs]); Y = np.vstack([p[1] for p in pairs])
    return X, Y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code_csv", required=True)
    ap.add_argument("--data_csv", required=True)
    ap.add_argument("--test_cells", required=True)
    ap.add_argument("--method", default="rk4")
    ap.add_argument("--nseg", type=int, default=5)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--v_prune_thresh", type=float, default=1e-3)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    global DEVICE
    if args.cpu: DEVICE = torch.device("cpu")

    G, b0_np, V_np, K_np, n_np, gamma_np = load_csv_params(args.code_csv)

    field = HillField(G).to(DEVICE)
    with torch.no_grad():
        field.b0.copy_(torch.tensor(b0_np, device=DEVICE))
        field.V.copy_( torch.tensor(V_np,  device=DEVICE))
        field.K.copy_( torch.tensor(K_np,  device=DEVICE))
        field.n.copy_( torch.tensor(n_np,  device=DEVICE))
        field.log_gamma.copy_( torch.log(torch.tensor(np.clip(gamma_np, 1e-20, None), device=DEVICE)))
        field.log_alpha.zero_()  # start at alpha=1

        # prune tiny V for stability
        mask = field.V.abs() < args.v_prune_thresh
        field.V[mask] = 0.0

    X_np, Y_np = build_pairs(args.data_csv, args.test_cells, G)
    X = torch.tensor(X_np, dtype=torch.float64, device=DEVICE)
    Y = torch.tensor(Y_np, dtype=torch.float64, device=DEVICE)

    # optimize only log_alpha
    try:
        opt = torch.optim.Adam([field.log_alpha], lr=args.lr, foreach=False)
    except TypeError:
        opt = torch.optim.Adam([field.log_alpha], lr=args.lr)

    for s in range(args.steps):
        opt.zero_grad(set_to_none=True)
        Yhat = integrate_step(field, X, method=args.method, nseg=args.nseg)
        loss = ((Yhat - Y)**2).mean()
        loss.backward()
        opt.step()
        if (s % 10) == 0:
            print(f"[step {s:03d}] alpha={torch.exp(field.log_alpha).item():.6f}  loss={loss.item():.6e}")

    alpha = float(torch.exp(field.log_alpha).item())
    print(f"==> learned alpha = {alpha:.6f}")

    # bake alpha into (b0, V, gamma) and save in standard (3G+2, G) CSV
    b0_out   = (alpha * field.b0.detach().cpu().numpy())
    V_out    = (alpha * field.V.detach().cpu().numpy())
    gamma_out= (alpha * torch.exp(field.log_gamma).detach().cpu().numpy())
    K_out    = field.K.detach().cpu().numpy()
    n_out    = field.n.detach().cpu().numpy()

    out_path = os.path.splitext(args.code_csv)[0] + "_alphaCal.csv"
    save_csv_params(out_path, b0_out, V_out, K_out, n_out, gamma_out)
    print(f"[saved] {out_path}")

if __name__ == "__main__":
    main()
