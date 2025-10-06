#!/usr/bin/env python3
# calibrate_prod_decay_for_csv.py
import argparse, os, numpy as np, pandas as pd, torch, torch.nn as nn
from torchdiffeq import odeint

"""
Run like this:
python Step8p2_calibrate_prod_decay_for_csv.py \
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
        # fixed (non-trainable) base params loaded from CSV
        self.b0    = nn.Parameter(torch.zeros(G, dtype=torch.float64), requires_grad=False)
        self.V     = nn.Parameter(torch.zeros(G, G, dtype=torch.float64), requires_grad=False)
        self.K     = nn.Parameter(torch.ones(G, G,  dtype=torch.float64), requires_grad=False)
        self.n     = nn.Parameter(torch.ones(G, G,  dtype=torch.float64), requires_grad=False)
        self.gamma = nn.Parameter(torch.ones(G, dtype=torch.float64),  requires_grad=False)

    def forward_terms(self, x):
        B, G = x.shape
        x_safe = torch.clamp(x, min=EPS)
        n_eff  = torch.clamp(self.n, min=1e-2, max=10.0)[None]
        K_eff  = torch.clamp(self.K, min=1e-4, max=1e4)[None]

        xj  = x_safe[:, None, :].expand(B, G, G)
        num = torch.pow(xj, n_eff)
        den = torch.pow(K_eff, n_eff) + num
        h   = num / (den + EPS)
        h   = torch.nan_to_num(h, nan=0.0, posinf=1.0, neginf=0.0)

        prod  = (h * self.V[None]).sum(dim=2) + self.b0    # (B,G)
        decay = self.gamma[None] * x_safe                  # (B,G)
        return prod, decay

class ScaledField(nn.Module):
    # train only these two scalars: s_prod, s_decay
    def __init__(self, base: HillField):
        super().__init__()
        self.base = base
        self.log_s_prod  = nn.Parameter(torch.zeros(1, dtype=torch.float64, device=DEVICE))
        self.log_s_decay = nn.Parameter(torch.zeros(1, dtype=torch.float64, device=DEVICE))

    def forward(self, t, x):
        prod, decay = self.base.forward_terms(x)
        s_prod  = torch.exp(self.log_s_prod)[0]
        s_decay = torch.exp(self.log_s_decay)[0]
        dxdt = s_prod * prod - s_decay * decay
        dxdt = torch.nan_to_num(dxdt, nan=0.0, posinf=1e6, neginf=-1e6)
        return dxdt

def load_field_from_csv(path: str) -> HillField:
    M = np.loadtxt(path, delimiter=",")
    def try_parse(A):
        r, c = A.shape
        return c if (r >= 5 and r == 3*c + 2) else None
    G = try_parse(M) or try_parse(M.T)
    if G is None:
        raise RuntimeError(f"{path}: expect (3G+2,G) or (G,3G+2)")
    if M.shape[0] != 3*G+2: M = M.T

    b0    = M[0, :]
    V     = M[1:1+G, :]
    K     = M[1+G:1+2*G, :]
    n     = M[1+2*G:1+3*G, :]
    gamma = M[1+3*G, :]

    field = HillField(G)
    with torch.no_grad():
        field.b0.copy_(torch.as_tensor(b0,    dtype=torch.float64))
        field.V.copy_( torch.as_tensor(V,     dtype=torch.float64))
        field.K.copy_( torch.as_tensor(K,     dtype=torch.float64))
        field.n.copy_( torch.as_tensor(n,     dtype=torch.float64))
        field.gamma.copy_(torch.as_tensor(gamma, dtype=torch.float64))
    return field

def prune_small_V(field: HillField, thresh: float):
    with torch.no_grad():
        V = field.V.data
        mask = V.abs() < thresh
        pruned = int(mask.sum().item())
        V[mask] = 0.0
        return pruned, int((V != 0).sum().item()), V.numel()

def integrate_one(field, X0, method="rk4", rtol=1e-12, atol=1e-12, nseg=5):
    # NOTE: do NOT decorate with @torch.no_grad() — we need gradients
    X = torch.clamp(X0, min=EPS)
    for k in range(nseg):
        tseg = torch.tensor([k/nseg, (k+1)/nseg], dtype=torch.float64, device=DEVICE)
        traj = odeint(field, X, tseg, method=method, rtol=rtol, atol=atol)
        X = torch.clamp(traj[-1], min=EPS)
    return X

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code_csv", required=True)
    ap.add_argument("--data_csv", required=True)
    ap.add_argument("--test_cells", required=True)
    ap.add_argument("--method", default="rk4")
    ap.add_argument("--nseg", type=int, default=5)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--v_prune_thresh", type=float, default=1e-3)
    args = ap.parse_args()

    # Load & freeze base model
    field = load_field_from_csv(args.code_csv).to(DEVICE).eval()
    for p in field.parameters(): p.requires_grad_(False)

    npr, nnz, tot = prune_small_V(field, args.v_prune_thresh)
    print(f"[info] V-prune |V|<{args.v_prune_thresh:g}: pruned {npr}/{tot}, nnz after={nnz}")

    # Build (X(t), Y(t+1)) pairs on DEVICE
    data = pd.read_csv(args.data_csv, header=None).values
    test_ids = np.unique(pd.read_csv(args.test_cells, header=None).iloc[:,0].to_numpy().astype(np.int64))
    cids = data[:,0].astype(np.int64); times = data[:,1].astype(np.int64)
    uniq_t = np.unique(times); uniq_t.sort()
    if len(uniq_t) < 2: raise RuntimeError("Need at least two timepoints")

    X_list, Y_list = [], []
    for t in uniq_t[:-1]:
        if (t+1) not in uniq_t: continue
        blk0 = data[times==t]; blk1 = data[times==t+1]
        mask0 = np.isin(blk0[:,0].astype(np.int64), test_ids)
        blk0 = blk0[mask0]
        if blk0.shape[0]==0: continue
        nxt = {int(c): blk1[i,2:2+field.G] for i,c in enumerate(blk1[:,0].astype(np.int64))}
        for i in range(blk0.shape[0]):
            cid = int(blk0[i,0])
            if cid in nxt:
                X_list.append(blk0[i,2:2+field.G])
                Y_list.append(nxt[cid])

    X = torch.as_tensor(np.stack(X_list), dtype=torch.float64, device=DEVICE)
    Y = torch.as_tensor(np.stack(Y_list), dtype=torch.float64, device=DEVICE)
    print(f"[info] matched pairs: {X.shape[0]}  genes: {X.shape[1]}")

    # Train the two scaling logs
    sfield = ScaledField(field).to(DEVICE)
    print(f"[debug] log_s_prod  device={sfield.log_s_prod.device}  dtype={sfield.log_s_prod.dtype}")
    print(f"[debug] log_s_decay device={sfield.log_s_decay.device} dtype={sfield.log_s_decay.dtype}")

    # Key fix: disable foreach to avoid device/dtype grouping issues
    opt = torch.optim.Adam([sfield.log_s_prod, sfield.log_s_decay],
                           lr=args.lr, foreach=False)

    for it in range(args.steps):
        opt.zero_grad(set_to_none=True)
        Yhat = integrate_one(sfield, X, method=args.method, nseg=args.nseg)
        loss = torch.mean((Yhat - Y)**2)
        loss.backward()
        opt.step()
        if (it % 25) == 0 or it == args.steps-1:
            sp = torch.exp(sfield.log_s_prod).item()
            sd = torch.exp(sfield.log_s_decay).item()
            print(f"[step {it:03d}] MSE={loss.item():.6g}  s_prod={sp:.6g}  s_decay={sd:.6g}")

    # Bake-in scales into CSV: b0' = sp*b0, V' = sp*V, gamma' = sd*gamma
    sp = float(torch.exp(sfield.log_s_prod).item())
    sd = float(torch.exp(sfield.log_s_decay).item())
    with torch.no_grad():
        b0_new = (sp * field.b0).cpu().numpy()
        V_new  = (sp * field.V).cpu().numpy()
        K_new  = field.K.cpu().numpy()
        n_new  = field.n.cpu().numpy()
        g_new  = (sd * field.gamma).cpu().numpy()

    G = field.G
    M = np.zeros((3*G+2, G), dtype=np.float64)
    M[0,:] = b0_new
    M[1:1+G,:] = V_new
    M[1+G:1+2*G,:] = K_new
    M[1+2*G:1+3*G,:] = n_new
    M[1+3*G,:] = g_new

    out = os.path.splitext(args.code_csv)[0] + "_proddecayCal.csv"
    np.savetxt(out, M, delimiter=",")
    print(f"[done] saved -> {out}")

if __name__ == "__main__":
    main()
