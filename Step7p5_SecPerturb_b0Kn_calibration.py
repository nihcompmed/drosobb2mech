#!/usr/bin/env python3
# calibrate_b0Kn_noGamma.py
"""
Joint calibration of b0, K, and n for Hill models without gamma.
Assumes input CSV has already been alpha-calibrated (b0, V, K, n).


Run:
python Step7p5_SecPerturb_b0Kn_calibration.py \
  --code_csv HillModels/HillLassoNoPrune_250815001_a1em02_l1em05_epoch=135-val_mse=0.02318_params_298x99_alphaCal.csv \
  --data_csv dataset_forallsections/all_ctgxyz_99g_fillbyrandVAE_t012345_noshuffle.csv \
  --test_cells test_cells.csv \
  --steps 200 --lr 5e-3 --nseg 10




"""

import os, argparse, numpy as np, pandas as pd, torch, torch.nn as nn
from torchdiffeq import odeint

torch.set_default_dtype(torch.float64)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-12

# ---------------- Hill field ----------------
class HillField(nn.Module):
    def __init__(self, b0, V, K, n):
        super().__init__()
        self.G = V.shape[0]
        self.b0 = nn.Parameter(torch.as_tensor(b0, dtype=torch.float64, device=DEVICE), requires_grad=True)
        self.V  = nn.Parameter(torch.as_tensor(V,  dtype=torch.float64, device=DEVICE), requires_grad=False)
        self.logK = nn.Parameter(torch.log(torch.as_tensor(K, dtype=torch.float64, device=DEVICE).clamp(min=1e-4, max=1e4)), requires_grad=True)
        self.n    = nn.Parameter(torch.as_tensor(n, dtype=torch.float64, device=DEVICE), requires_grad=True)

    def forward(self, t, x):
        if x.dim() == 1: x = x.unsqueeze(0)
        x = torch.clamp(x, min=EPS)
        B, G = x.shape

        K_eff = torch.exp(self.logK).clamp(min=1e-4, max=1e4)[None]
        n_eff = torch.clamp(self.n, min=0.2, max=10.0)[None]

        xj  = x[:, None, :].expand(B, G, G)
        num = torch.pow(xj, n_eff)
        den = torch.pow(K_eff, n_eff) + num
        h   = num / (den + EPS)
        h   = torch.nan_to_num(h, nan=0.0, posinf=1.0, neginf=0.0)

        prod = (h * self.V[None]).sum(dim=2) + self.b0
        dxdt = prod  # no gamma, no alpha
        return torch.nan_to_num(dxdt, nan=0.0, posinf=1e6, neginf=-1e6)

# ---------------- CSV helpers ----------------
def load_hill_csv(path):
    M = np.loadtxt(path, delimiter=",")
    def try_parse(A):
        r, c = A.shape
        return c if (r >= 4 and r == 3*c + 1) else None
    G = try_parse(M)
    if G is None:
        Mt = M.T; Gt = try_parse(Mt)
        if Gt is None:
            raise RuntimeError(f"{path}: expect (3G+1,G) or (G,3G+1), got {M.shape}")
        M, G = Mt, Gt
    b0 = M[0, :]
    V  = M[1:1+G, :]
    K  = M[1+G:1+2*G, :]
    n  = M[1+2*G:1+3*G, :]
    return b0, V, K, n, G

def save_hill_csv(path_out, b0, V, K, n):
    M = np.vstack([b0[None, :], V, K, n])
    np.savetxt(path_out, M, delimiter=",", fmt="%.10g")

# ---------------- Data pairing ----------------
def build_pairs(data_csv, test_csv, G):
    data = pd.read_csv(data_csv, header=None).values
    test_ids = pd.read_csv(test_csv, header=None).iloc[:,0].to_numpy().astype(np.int64)
    test_ids = np.unique(test_ids)

    all_ids = np.unique(data[:,0].astype(np.int64))
    train_ids = np.setdiff1d(all_ids, test_ids)  # all but test set

    cids = data[:,0].astype(np.int64)
    times = data[:,1].astype(np.int64)
    uniq = np.unique(times); uniq.sort()

    g0, g1 = 2, 2+G
    Xs, Ys = [], []
    for t in uniq[:-1]:
        if (t+1) not in uniq: continue
        B0 = data[times==t]; B1 = data[times==t+1]
        mask = np.isin(B0[:,0].astype(np.int64), train_ids)
        B0 = B0[mask]
        if B0.shape[0]==0: continue
        nxt = {int(c): B1[i,g0:g1] for i,c in enumerate(B1[:,0].astype(np.int64))}
        for row in B0:
            cid = int(row[0])
            if cid in nxt:
                Xs.append(row[g0:g1]); Ys.append(nxt[cid])
    if not Xs: raise RuntimeError("No matched (t,t+1) pairs found.")
    return np.stack(Xs), np.stack(Ys)


# ---------------- Integration ----------------
def integrate_step(field, X0, method="rk4", nseg=5, rtol=1e-12, atol=1e-12):
    X = torch.clamp(X0, min=EPS)
    for k in range(nseg):
        tseg = torch.tensor([k/nseg, (k+1)/nseg], dtype=torch.float64, device=X.device)
        X = odeint(field, X, tseg, method=method, rtol=rtol, atol=atol)[-1]
        X = torch.clamp(X, min=EPS)
    return X

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code_csv", required=True)
    ap.add_argument("--data_csv", required=True)
    ap.add_argument("--test_cells", required=True)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--nseg", type=int, default=5)
    args = ap.parse_args()

    b0, V, K, n, G = load_hill_csv(args.code_csv)
    field = HillField(b0, V, K, n).to(DEVICE)

    X_np, Y_np = build_pairs(args.data_csv, args.test_cells, G)
    X = torch.tensor(X_np, dtype=torch.float64, device=DEVICE)
    Y = torch.tensor(Y_np, dtype=torch.float64, device=DEVICE)

    opt = torch.optim.Adam([field.b0, field.logK, field.n], lr=args.lr, foreach=False)

    batch_size = 256   # tune this (try 128–512 depending on memory)
    N = X.shape[0]

    for s in range(args.steps):
        opt.zero_grad(set_to_none=True)
        total_loss = 0.0
        n_batches = 0

        # shuffle indices each epoch
        perm = torch.randperm(N, device=DEVICE)
        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            Xb, Yb = X[idx], Y[idx]
            Yhat = integrate_step(field, Xb, nseg=args.nseg)
            loss = ((Yhat - Yb)**2).mean()
            loss.backward()
            total_loss += loss.item()
            n_batches += 1

        opt.step()
        if (s % 10) == 0 or s == args.steps-1:
            avg_loss = total_loss / n_batches
            print(f"[step {s:03d}] loss={avg_loss:.6e}")

    # Save updated params
    b0_new = field.b0.detach().cpu().numpy()
    K_new  = torch.exp(field.logK).detach().cpu().numpy()
    n_new  = field.n.detach().cpu().numpy()
    out_path = os.path.splitext(args.code_csv)[0] + "_b0KnCal.csv"
    save_hill_csv(out_path, b0_new, V, K_new, n_new)
    print(f"[done] saved -> {out_path}")

if __name__ == "__main__":
    main()
