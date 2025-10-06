#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step6_integrateBB.py

For each row of the imputed CSV, integrate the latent ODE from its VAE mean (mu0),
decode at npts times s in [0, s_max], and write npts consecutive rows.

Output row layout (no header):
    [cid, time, s_eval, g1..gG, x, y, z]

- One block of length --npts for each input row (so output has npts * N_in rows).
- G is auto-inferred (27 or 99) from the imputed file.
"""

import os, re, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchdiffeq import odeint

# ---------------- CLI ----------------
ap = argparse.ArgumentParser(description="Integrate VAE+Neural ODE in latent space and decode to gene space.")
ap.add_argument("--imputed",  required=True, help="Imputed CSV: [cid,time, G genes, xyz(3), (optional g_next(G))]")
ap.add_argument("--vae_ckpt", required=True, help="Lightning VAE checkpoint (.ckpt)")
ap.add_argument("--ode_ckpt", required=True, help="Neural ODE checkpoint (.ckpt)")
ap.add_argument("--batch", type=int, default=1000, help="Batch size for encoding/decoding")
ap.add_argument("--limit", type=int, default=0, help="Optional row cap for quick tests (0 = no cap)")
ap.add_argument("--npts", type=int, default=6, help="Number of latent samples along path (>=2)")
ap.add_argument("--s_max", type=float, default=1.0, help="Max latent time; samples in [0, s_max]")
ap.add_argument("--ode_method", type=str, default="dopri5", help="odeint method (e.g. dopri5, rk4)")
ap.add_argument("--rtol", type=float, default=1e-9)
ap.add_argument("--atol", type=float, default=1e-12)
ap.add_argument("--out", default=None, help="Output CSV path. If omitted, auto-named from ckpt IDs.")
args = ap.parse_args()

# ---------------- Torch / layout ----------------
torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCALE_AFTER_SIGMOID = 3.18

def infer_layout(n_cols: int):
    """
    Returns (G, gene_beg, gene_end, xyz_beg, xyz_end).

    Assumes imputed layout: [cid, time, g_now(G), xyz(3), (optional g_next(G))]
      - 27 genes: 2 + 27 + 3 [+ 27] = 32..59
      - 99 genes: 2 + 99 + 3 [+ 99] = 104..203
    """
    if 32 <= n_cols <= 60:
        G = 27
        gene_beg = 2
        gene_end = gene_beg + G
        xyz_beg  = gene_end
        xyz_end  = xyz_beg + 3
        return G, gene_beg, gene_end, xyz_beg, xyz_end
    if 104 <= n_cols <= 205:
        G = 99
        gene_beg = 2
        gene_end = gene_beg + G
        xyz_beg  = gene_end
        xyz_end  = xyz_beg + 3
        return G, gene_beg, gene_end, xyz_beg, xyz_end
    raise ValueError(f"Unrecognized imputed CSV layout with {n_cols} columns.")

def make_act(name: str):
    n = (name or "tanh").lower()
    return {
        "relu": nn.ReLU(), "tanh": nn.Tanh(), "gelu": nn.GELU(), "silu": nn.SiLU(),
        "mish": nn.Mish(), "elu": nn.ELU(), "celu": nn.CELU(), "softplus": nn.Softplus(),
        "softplus0": nn.Softplus(), "erf": type("Erf",(nn.Module,),{"forward":lambda self,x: torch.erf(x)})()
    }.get(n, nn.Tanh())

class Multiply(nn.Module):
    def __init__(self, c): super().__init__(); self.c=float(c)
    def forward(self, x): return self.c * x

# ---------------- VAE (match training) ----------------
class VAEInfer(nn.Module):
    def __init__(self, nd1, nd2, nd3, ldim, act_name="tanh",
                 in_dim=102, out_dim=99, enc_do=0.0, dec_do=0.0, final_act="sigmoid"):
        super().__init__()
        act = make_act(act_name)
        self.enc_drop = nn.Dropout(enc_do) if enc_do > 0 else nn.Identity()
        self.fc1 = nn.Linear(in_dim, nd1, dtype=torch.float64)
        self.fc2 = nn.Linear(nd1, nd1,  dtype=torch.float64)
        self.fc3 = nn.Linear(nd1, nd2,  dtype=torch.float64)
        self.fc4 = nn.Linear(nd2, nd3,  dtype=torch.float64)
        self.act = act
        self.mu  = nn.Linear(nd3, ldim, dtype=torch.float64)
        self.lv  = nn.Linear(nd3, ldim, dtype=torch.float64)
        dec = [
            nn.Linear(ldim, nd3, dtype=torch.float64), act, nn.Dropout(dec_do),
            nn.Linear(nd3, nd2, dtype=torch.float64),  act, nn.Dropout(dec_do),
            nn.Linear(nd2, nd1, dtype=torch.float64),  act, nn.Dropout(dec_do),
            nn.Linear(nd1, out_dim, dtype=torch.float64)
        ]
        if (final_act or "sigmoid").lower() == "sigmoid":
            dec += [nn.Sigmoid(), Multiply(SCALE_AFTER_SIGMOID)]
        self.dec = nn.Sequential(*dec)
        self.double().eval()

    @torch.no_grad()
    def encode_mu(self, x: torch.Tensor) -> torch.Tensor:
        x = self.enc_drop(self.act(self.fc1(x)))
        x = self.enc_drop(self.act(self.fc2(x)))
        x = self.enc_drop(self.act(self.fc3(x)))
        x = self.enc_drop(self.act(self.fc4(x)))
        return self.mu(x)

def load_vae(path):
    sd = torch.load(path, map_location="cpu")
    hp = sd.get("hyper_parameters", {})
    m = VAEInfer(
        hp.get("ndense1", 1024),
        hp.get("ndense2", 512),
        hp.get("ndense3", 256),
        hp.get("latent_dim", hp.get("ldim", 10)),
        act_name=hp.get("act_name", hp.get("actfcn", "tanh")),
        in_dim=hp.get("input_dim", 102),
        out_dim=hp.get("output_dim", 99),
        enc_do=hp.get("enc_dropout_p", 0.0),
        dec_do=hp.get("dec_dropout_p", 0.0),
        final_act=hp.get("final_act", hp.get("finalact", "sigmoid")),
    ).to(device).eval()
    m.load_state_dict(sd.get("state_dict", sd), strict=False)
    return m

# ---------------- ODE (rebuild from state_dict) ----------------
class ODECore(nn.Module):
    def __init__(self, shapes, act):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(s[1], s[0], dtype=torch.float64) for s in shapes])
        self.act = act
        self.double().eval()
    def forward(self, z):
        for i, lin in enumerate(self.layers):
            z = lin(z)
            if i != len(self.layers) - 1:
                z = self.act(z)
        return z

class Succ(nn.Module):
    def __init__(self, core): super().__init__(); self.core = core
    def forward(self, t, z): return self.core(z)

def load_ode(path, ldim):
    sd = torch.load(path, map_location="cpu"); st = sd["state_dict"]
    items = []
    for k in st:
        if k.endswith(".weight") and "successor" in k:
            m = re.search(r'\.(\d+)\.weight$', k)
            if m: items.append((int(m.group(1)), k))
    if not items:
        for k in st:
            if k.endswith(".weight") and ("v_net." in k):
                m = re.search(r'v_net\.(\d+)\.weight$', k)
                if m: items.append((int(m.group(1)), k))
    if not items:
        raise RuntimeError("No successor/v_net linear weights found in ODE ckpt.")
    items.sort()
    shapes = [st[k].shape for _, k in items]  # (out, in)
    act = make_act("erf" if "erf" in path.lower() else "tanh")
    core = ODECore(shapes, act).to(device).eval()
    for i, (_, k) in enumerate(items):
        core.layers[i].weight.data.copy_(st[k])
        bkey = k.replace("weight", "bias")
        core.layers[i].bias.data.copy_(st[bkey] if bkey in st else torch.zeros_like(core.layers[i].bias))
    if core.layers[0].in_features != ldim or core.layers[-1].out_features != ldim:
        raise RuntimeError(
            f"ldim mismatch: VAE={ldim}, ODE in={core.layers[0].in_features}, out={core.layers[-1].out_features}"
        )
    return Succ(core).to(device).eval()

def jid(path):
    m = re.search(r"(\d{9,})", os.path.basename(path))
    return m.group(1) if m else "NA"

# ---------------- Main ----------------
if __name__ == "__main__":
    print(f"[info] imputed={args.imputed}")
    print(f"[info] vae_ckpt={args.vae_ckpt}")
    print(f"[info] ode_ckpt={args.ode_ckpt}")
    print(f"[info] npts={args.npts}, s_max={args.s_max}, method={args.ode_method}")

    assert args.npts >= 2, "--npts must be >= 2"

    # Load imputed
    df = pd.read_csv(args.imputed, header=None)
    if args.limit and args.limit > 0:
        df = df.iloc[:args.limit].copy()

    n_cols = df.shape[1]
    G, gene_beg, gene_end, xyz_beg, xyz_end = infer_layout(n_cols)

    cid = df.iloc[:, 0].to_numpy()
    tim = df.iloc[:, 1].to_numpy()
    g   = np.abs(df.iloc[:, gene_beg:gene_end].to_numpy())  # ensure non-negative
    xyz = df.iloc[:, xyz_beg:xyz_end].to_numpy()

    # VAE input is [genes, xyz]
    X_in = torch.tensor(np.concatenate([g, xyz], axis=1), dtype=torch.float64, device=device)

    # Load models
    vae = load_vae(args.vae_ckpt)
    ldim = vae.mu.out_features
    ode  = load_ode(args.ode_ckpt, ldim)

    # Latent-time grid in [0, s_max]
    S = torch.linspace(0.0, float(args.s_max), int(args.npts), dtype=torch.float64, device=device)  # [npts]

    rows = []
    with torch.no_grad():
        for start in range(0, X_in.shape[0], args.batch):
            xb   = X_in[start:start+args.batch]                         # [B, G+3]
            k    = xb.shape[0]
            cidb = cid[start:start+k]
            timb = tim[start:start+k]
            xyzb = xyz[start:start+k]

            # encode -> mu0
            mu0 = vae.encode_mu(xb)                                     # [B, ldim]

            # integrate latent ODE from each mu0 at times S
            zb  = odeint(ode, mu0, S, method=args.ode_method,
                         rtol=float(args.rtol), atol=float(args.atol))   # [npts, B, ldim]
            zb  = zb.transpose(0, 1).contiguous()                        # [B, npts, ldim]

            # decode all at once for speed
            zflat = zb.reshape(-1, ldim)                                 # [B*npts, ldim]
            yflat = vae.dec(zflat).detach().cpu().numpy()                # [B*npts, G]

            # repeat meta columns per-sample, per-s_eval
            s_rep   = S.detach().cpu().numpy().reshape(1, args.npts).repeat(k, axis=0).reshape(-1, 1)  # [B*npts,1]
            cid_rep = np.repeat(cidb, args.npts).reshape(-1, 1)          # [B*npts,1]
            tim_rep = np.repeat(timb, args.npts).reshape(-1, 1)          # [B*npts,1]
            xyz_rep = np.repeat(xyzb, args.npts, axis=0)                 # [B*npts,3]

            block = np.column_stack([cid_rep, tim_rep, s_rep, yflat, xyz_rep])
            rows.append(block)

    out = np.vstack(rows) if rows else np.zeros((0, 3 + G + 3))

    # Output path
    if args.out:
        out_path = args.out
    else:
        out_path = f"IntegrationBB_G{G}_n{args.npts}_S{args.s_max:.2f}_VAE{jid(args.vae_ckpt)}_ODE{jid(args.ode_ckpt)}.csv"

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    pd.DataFrame(out).to_csv(out_path, index=False, header=None)

    print(f"[done] saved {out.shape} → {out_path}")
    print("Layout per row: [cid, time, s_eval, g1..gG, x, y, z]")
    print("Ordering: for each input row, npts consecutive rows with s_eval increasing.")
