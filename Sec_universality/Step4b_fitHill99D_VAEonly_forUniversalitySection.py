#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step4b_fitHill99D_VAEonly_forUniversalitySection.py
Pipeline for universality section:
  - Load VAE checkpoint from Step2
  - Load imputed CSV from Step3 (already done)
  - Build teacher pushforward pairs
  - Fit Hill function model with L1 penalty
  - Save Hill model parameters CSV

Usage:
  python Sec_universality/Step4b_fitHill99D_VAEonly_forUniversalitySection.py \
      --job_id 250923001 --l1_lambda 1e-4
"""

import os, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from Step3_impute_universality import (
    VAE, parse_swarm, parse_cmdline, select_ckpt
)

torch.set_default_dtype(torch.float64)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────── Utils ───────────────────
def erf_smooth_l1(residual: torch.Tensor, sigma_vec: torch.Tensor) -> torch.Tensor:
    sigma = torch.clamp(sigma_vec, min=1e-8)
    z = residual / (sigma * np.sqrt(2.0))
    term1 = residual * torch.erf(z)
    term2 = sigma * np.sqrt(2.0/np.pi) * torch.exp(-z*z)
    return (term1 + term2).mean()

def find_imputed_csv(job_id: int) -> Path:
    root = Path("Sec_universality/ImputedOut_universality")
    pats = list(root.glob(f"Imputed_{job_id}_ldim*_trainonly.csv"))
    if not pats:
        raise FileNotFoundError(f"No filtered imputed CSV for job {job_id} in {root}")
    pats.sort(key=os.path.getmtime, reverse=True)
    return pats[0]

# ─────────────────── Hill Model ───────────────────
class HillAll(nn.Module):
    def __init__(self, G: int):
        super().__init__()
        self.b0 = nn.Parameter(torch.zeros(G, dtype=torch.float64))
        self.V  = nn.Parameter(torch.randn(G, G, dtype=torch.float64) * 0.1)
        self.K  = nn.Parameter(torch.ones(G, G, dtype=torch.float64) * 0.5)
        self.n  = nn.Parameter(torch.ones(G, G, dtype=torch.float64))
        self.log_gamma = nn.Parameter(torch.zeros(G, dtype=torch.float64))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, G = x.shape
        x_safe = torch.clamp(x, min=1e-12)
        n_eff  = torch.clamp(self.n, min=1.0, max=3.0)
        K_eff  = torch.clamp(self.K, min=0.2, max=2.5)
        gamma  = torch.exp(torch.clamp(self.log_gamma, -20.0, 20.0))
        xj  = x_safe.unsqueeze(1).expand(B, G, G)
        n_b = n_eff.unsqueeze(0).expand(B, G, G)
        K_b = K_eff.unsqueeze(0).expand(B, G, G)
        xn  = torch.pow(xj, n_b)
        den = torch.pow(K_b, n_b) + xn
        h   = xn / (den + 1e-12)
        production = (h * self.V.unsqueeze(0)).sum(dim=2) + self.b0
        out = production - gamma * x_safe
        return torch.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)

def rk4(field: nn.Module, x0: torch.Tensor, T: torch.Tensor, n_sub: int) -> torch.Tensor:
    if T.ndim == 1: T = T.unsqueeze(1)
    steps = max(1, n_sub)
    h = T / float(steps)
    x = x0
    for _ in range(steps):
        k1 = field(x); k2 = field(x + 0.5*h*k1)
        k3 = field(x + 0.5*h*k2); k4 = field(x + h*k3)
        x  = x + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    return x

# ─────────────────── Dataset ───────────────────
class MixedDerivStateDataset(Dataset):
    def __init__(self, Xd, Yd, Xn, Xn1, T):
        self.Xd, self.Yd = map(lambda a: torch.as_tensor(a, dtype=torch.float64), (Xd, Yd))
        self.Xn, self.Xn1, self.T = map(lambda a: torch.as_tensor(a, dtype=torch.float64), (Xn, Xn1, T))
    def __len__(self): return self.Xd.shape[0]
    def __getitem__(self, i):
        j = np.random.randint(0, self.Xn.shape[0])
        return (self.Xd[i], self.Yd[i], self.Xn[j], self.Xn1[j], self.T[j])

# ─────────────────── Trainer ───────────────────
class HillTrainerTorch:
    def __init__(self, G, max_lr, max_epochs, l1_lambda,
                 int_substeps=5, sigma_vec=None, wd=0.0, deriv_only=False):
        self.field = HillAll(G).to(DEVICE)
        self.loss_mse = nn.MSELoss()
        self.max_lr, self.max_epochs = max_lr, max_epochs
        self.int_sub = int(int_substeps)
        self.sigma_vec = torch.as_tensor(sigma_vec if sigma_vec is not None else np.ones(G),
                                         dtype=torch.float64, device=DEVICE)
        self.optimizer = optim.Adam(self.field.parameters(), lr=max_lr, weight_decay=wd, foreach=False)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max_epochs, eta_min=max_lr*0.05
        )
        self.deriv_only = deriv_only
        self.l1_lambda = l1_lambda

    def _weights(self, epoch):
        if self.deriv_only: return 1.0, 0.0, 0.0
        return 0.5, 0.5, 0.1  # simplified

    def train(self, ds, bsize, ckpt_dir="HillModels", name_prefix="hill", job_id="000"):
        torch.manual_seed(42); np.random.seed(42)
        n_val = max(1, int(0.1*len(ds)))
        train_ds, val_ds = random_split(ds, [len(ds)-n_val, n_val],
                                        generator=torch.Generator().manual_seed(42))
        train_loader = DataLoader(train_ds, batch_size=bsize, shuffle=True)
        val_loader   = DataLoader(val_ds, batch_size=bsize, shuffle=False)
        best_val = np.inf
        os.makedirs(ckpt_dir, exist_ok=True)
        params_csv = os.path.join(ckpt_dir, f"{name_prefix}_{job_id}.csv")

        for epoch in range(self.max_epochs):
            self.field.train()
            w_deriv,w_int,w_coll = self._weights(epoch)
            for xd,yd,xn,xn1,T in train_loader:
                xd,yd,xn,xn1,T = xd.to(DEVICE),yd.to(DEVICE),xn.to(DEVICE),xn1.to(DEVICE),T.to(DEVICE)
                y_hat = self.field(xd)
                loss_deriv = erf_smooth_l1(y_hat-yd,self.sigma_vec)
                if self.deriv_only:
                    loss_state,loss_coll = torch.tensor(0.,device=DEVICE),torch.tensor(0.,device=DEVICE)
                else:
                    x_pred = rk4(self.field,xn,T,self.int_sub)
                    loss_state=self.loss_mse(x_pred,xn1)
                    loss_coll=(x_pred-xn1).pow(2).mean()
                l1_pen=self.l1_lambda*torch.abs(self.field.V).sum()
                loss=w_deriv*loss_deriv + w_int*loss_state + w_coll*loss_coll + l1_pen
                self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
            self.scheduler.step()
            # validation
            self.field.eval()
            with torch.no_grad():
                v_loss=0; vc=0
                for xd,yd,xn,xn1,T in val_loader:
                    xd,yd,xn,xn1,T = xd.to(DEVICE),yd.to(DEVICE),xn.to(DEVICE),xn1.to(DEVICE),T.to(DEVICE)
                    v_loss+=float(self.loss_mse(self.field(xd),yd)); vc+=1
                monitor=v_loss/max(1,vc)
            if monitor<best_val:
                best_val=monitor
                with torch.no_grad():
                    b0=self.field.b0.detach().cpu().numpy()[None,:]
                    V=self.field.V.detach().cpu().numpy()
                    K=self.field.K.detach().cpu().numpy()
                    n=self.field.n.detach().cpu().numpy()
                    gamma=torch.exp(self.field.log_gamma).detach().cpu().numpy()[None,:]
                    mat=np.vstack([b0,V,K,n,gamma])
                    pd.DataFrame(mat).to_csv(params_csv,header=False,index=False)
                print(f"  ✅ saved best -> {params_csv}")
        return params_csv

# ─────────────────── Pushforward ───────────────────
def build_pushfwd_from_vae(df_arr, vae_model, n_sub=3, G=99, bsize_pairs=4096):
    cid,time = df_arr[:,0].astype(int), df_arr[:,1].astype(int)
    genes,xyz = df_arr[:,2:2+G],df_arr[:,2+G:2+G+3]
    key2idx={(int(c),int(t)):i for i,(c,t) in enumerate(zip(cid,time))}
    pairs=[(int(c),int(t)) for c,t in zip(cid,time) if (int(c),int(t)+1) in key2idx]
    Xd_list,Yd_list,Xn_list,Xn1_list,T_list=[],[],[],[],[]
    for i in range(0,len(pairs),bsize_pairs):
        batch=pairs[i:i+bsize_pairs]
        idx_t=[key2idx[(c,t)] for c,t in batch]; idx_t1=[key2idx[(c,t+1)] for c,t in batch]
        g_t,g_t1=torch.tensor(genes[idx_t],device=DEVICE),torch.tensor(genes[idx_t1],device=DEVICE)
        x_t=torch.cat([g_t,torch.tensor(xyz[idx_t],device=DEVICE)],1)
        x_t1=torch.cat([g_t1,torch.tensor(xyz[idx_t1],device=DEVICE)],1)
        z_t,z_t1=vae_model.encode_mu(x_t),vae_model.encode_mu(x_t1); dz=z_t1-z_t
        s_grid=torch.linspace(0.,1.,n_sub+1,dtype=torch.float64,device=DEVICE)
        z_stack=torch.stack([z_t+s*dz for s in s_grid],0)
        g_stack=torch.stack([vae_model.decoder(z_stack[k]) for k in range(n_sub+1)],0)
        def jvp_batch(z,v):
            outs=[]
            for zi,vi in zip(z,v):
                zi=zi.detach().requires_grad_(True)
                _,jvpi=torch.autograd.functional.jvp(lambda _z: vae_model.decoder(_z.unsqueeze(0)).squeeze(0),zi,vi)
                outs.append(jvpi)
            return torch.stack(outs,0)
        gdots=torch.stack([jvp_batch(z_stack[k],dz) for k in range(n_sub+1)],0)
        K,B=n_sub+1,len(batch)
        Xd_list.append(g_stack.detach().reshape(K*B,G).cpu().numpy())
        Yd_list.append(gdots.detach().reshape(K*B,G).cpu().numpy())
        Xn_list.append(g_t.detach().cpu().numpy())
        Xn1_list.append(g_t1.detach().cpu().numpy())
        T_list.append(np.ones((B,1)))
    return (np.vstack(Xd_list),np.vstack(Yd_list),
            np.vstack(Xn_list),np.vstack(Xn1_list),np.vstack(T_list))

# ─────────────────── Main ───────────────────
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--job_id",type=int,required=True)
    ap.add_argument("--l1_lambda",type=float,default=1e-4)
    ap.add_argument("--n_sub",type=int,default=5)
    ap.add_argument("--int_substeps",type=int,default=5)
    ap.add_argument("--bsize",type=int,default=1024)
    ap.add_argument("--max_epochs",type=int,default=100)
    ap.add_argument("--maxlr",type=float,default=3e-3)
    args=ap.parse_args()

    job_id=args.job_id
    line=parse_swarm(job_id); settings=parse_cmdline(line)
    ckpt=select_ckpt(Path("checkpoints"),job_id)
    state=torch.load(ckpt,map_location="cpu")
    vae_model=VAE(settings["N1"],settings["N2"],settings["N3"],settings["ldim"],settings["actfcn"]).to(DEVICE).eval()
    vae_model.load_state_dict(state.get("state_dict",state),strict=False)

    # load imputed CSV
    imputed_csv=find_imputed_csv(job_id)
    print(f"[info] Using imputed CSV: {imputed_csv}")
    df_arr=pd.read_csv(imputed_csv,header=None).values

    # build pushforward
    Xd,Yd,Xn,Xn1,T=build_pushfwd_from_vae(df_arr,vae_model,n_sub=args.n_sub,G=99)
    sigma_vec=np.maximum(1.4826*np.median(np.abs(Yd-np.median(Yd,0)),0),1e-6)
    ds=MixedDerivStateDataset(Xd,Yd,Xn,Xn1,T)

    l1_str=f"{args.l1_lambda:.0e}".replace("e-0","en")
    save_job_id=f"{job_id}hill_l1{l1_str}"

    trainer=HillTrainerTorch(G=99,max_lr=args.maxlr,max_epochs=args.max_epochs,
                             l1_lambda=args.l1_lambda,int_substeps=args.int_substeps,
                             sigma_vec=sigma_vec,wd=0.0,deriv_only=False)
    params_csv = trainer.train(
        ds,
        bsize=args.bsize,
        ckpt_dir="Sec_universality/HillModels_trainonly",   # <── new folder
        name_prefix=f"Hill_fromVAEonly_sub{args.n_sub}_rk4{args.int_substeps}",
        job_id=save_job_id
    )



    print(f"[done] best params at: {params_csv}")

if __name__=="__main__":
    main()
