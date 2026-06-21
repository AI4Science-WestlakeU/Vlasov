#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Examples:|
  # run all
  python unified_test_esay.py --settings all --mode both --models all --epochs 300 --ae_epochs 500 --pi_epochs 3000 --pi_ic_steps 800 --lowrank_rank 8 --reset_master --save_every 1 --no_h1 --device cuda
  # quick pipeline test
  python unified_test_esay.py \
    --settings all --mode both --models lowrank --quick --epochs 10 --ae_epochs 10 \
    --pi_epochs 10 --pi_ic_steps 20 --no_h1 --device cuda

  # reproduce low-rank with one setting
  CUDA_VISIBLE_DEVICES=5 python run_kvp_unified_clean.py \
    --settings 2d2v --mode both --models lowrank --epochs 300 --pi_epochs 3000 \
    --lowrank_rank 8 --device cuda

  # custom data
  CUDA_VISIBLE_DEVICES=5 python run_kvp_unified_clean.py \
    --settings 2d2v --mode data --models all --data_2d2v /path/to/2d2v.npz --device cuda
"""

import os, csv, json, math, time, argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------- utils -----------------------------

def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

def save_json(path, obj):
    ensure_dir(Path(path).parent)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def write_md(path, text):
    ensure_dir(Path(path).parent)
    with open(path, 'w', encoding='utf-8') as f: f.write(text)

def append_csv(path, row, header):
    ensure_dir(Path(path).parent)
    exists = os.path.exists(path)
    with open(path, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists: w.writeheader()
        w.writerow({k: row.get(k, '') for k in header})

def set_seed(seed):
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

def rel_l2_batch(pred, gt, eps=1e-12):
    return torch.linalg.norm((pred-gt).reshape(pred.shape[0], -1), dim=1) / (torch.linalg.norm(gt.reshape(gt.shape[0], -1), dim=1)+eps)

def central_diff_periodic(u, dim, dx):
    return (torch.roll(u, -1, dims=dim)-torch.roll(u, 1, dims=dim))/(2.0*dx)

def parse_csv(s, all_items):
    if s == 'all': return list(all_items)
    return [x.strip() for x in s.split(',') if x.strip()]

# ----------------------------- setting -----------------------------

@dataclass
class Setting:
    name: str
    d: int
    x_shape: Tuple[int, ...]
    v_shape: Tuple[int, ...]
    Lx: float = 2*math.pi
    vmax: float = 6.0

    @property
    def dx(self): return tuple([self.Lx/n for n in self.x_shape])
    @property
    def v_grids(self):
        return tuple([np.linspace(-self.vmax, self.vmax, n, dtype=np.float32) for n in self.v_shape])
    @property
    def dv(self):
        return tuple([float(g[1]-g[0]) if len(g)>1 else 1.0 for g in self.v_grids])
    @property
    def dv_prod(self): return float(np.prod(self.dv))
    @property
    def x_grids(self):
        return tuple([np.linspace(0, self.Lx, n, endpoint=False, dtype=np.float32) for n in self.x_shape])

# ----------------------------- VP solver -----------------------------

def gaussian(v, mu, sig):
    return np.exp(-0.5*((v-mu)/sig)**2)/(np.sqrt(2*np.pi)*sig)

def initial_f(setting: Setting, alpha=0.15, beam_u=2.4, thermal=0.55, phase=None):
    d = setting.d
    if phase is None: phase = [0.0]*d
    X = np.meshgrid(*setting.x_grids, indexing='ij')
    # Use a stable, reproducible density family. For 2D this matches the
    # classic two-stream VP benchmark more closely than a generic summed cosine:
    # rho = 1 + alpha cos(x+px) cos(y+py). For 1D it is a single cosine; for
    # 3D it is a mild mixed wave to keep the generated data non-trivial but stable.
    rho = np.ones(setting.x_shape, dtype=np.float64)
    if d == 1:
        rho += alpha * np.cos(X[0] + phase[0])
    elif d == 2:
        rho += alpha * np.cos(X[0] + phase[0]) * np.cos(X[1] + phase[1])
    else:
        rho += (alpha / d) * sum(np.cos(X[j] + phase[j]) for j in range(d))
        rho += 0.25 * alpha * np.cos(X[0] + phase[0]) * np.cos(X[1] + phase[1])
    rho = np.maximum(rho, 0.05)
    V = np.meshgrid(*setting.v_grids, indexing='ij')
    q = 0.5*gaussian(V[0], -beam_u, thermal) + 0.5*gaussian(V[0], beam_u, thermal)
    for j in range(1, d):
        q = q * gaussian(V[j], 0.0, thermal)
    q = q / (q.sum()*setting.dv_prod + 1e-12)
    f = rho.reshape(setting.x_shape + (1,)*d) * q.reshape((1,)*d + setting.v_shape)
    return f.astype(np.float64)

def rho_np(f, setting):
    axes = tuple(range(setting.d, 2*setting.d))
    return f.sum(axis=axes) * setting.dv_prod

def spectral_E_np(rho, setting):
    d = setting.d
    charge = rho - rho.mean()
    ch = np.fft.fftn(charge)
    ks = [2*np.pi*np.fft.fftfreq(n, d=dx) for n, dx in zip(setting.x_shape, setting.dx)]
    K = np.meshgrid(*ks, indexing='ij')
    k2 = np.zeros_like(rho, dtype=np.float64)
    for Kj in K: k2 += Kj**2
    phi_hat = np.zeros_like(ch, dtype=np.complex128)
    nz = k2 > 1e-14
    phi_hat[nz] = ch[nz] / k2[nz]
    E = []
    for Kj in K:
        Eh = -1j*Kj*phi_hat
        Eh[tuple([0]*d)] = 0.0
        Ej = np.fft.ifftn(Eh).real.astype(np.float64)
        Ej -= Ej.mean()
        E.append(Ej)
    return np.stack(E, axis=0)

def advect_space_axis(f, setting, axis, dt):
    # f shape x_dims + v_dims; axis in spatial dims
    d = setting.d; out = np.empty_like(f)
    xgrid = setting.x_grids[axis]; dx = setting.dx[axis]; L = setting.Lx
    vgrid = setting.v_grids[axis]
    N = setting.x_shape[axis]
    for iv, vel in enumerate(vgrid):
        dep = (xgrid - vel*dt) % L
        coord = dep/dx
        i0 = np.floor(coord).astype(np.int64)
        w = (coord - i0).astype(np.float64)
        sl = [slice(None)]*(2*d); sl[d+axis] = iv
        fs = f[tuple(sl)]
        a = np.take(fs, i0 % N, axis=axis)
        b = np.take(fs, (i0+1) % N, axis=axis)
        shape = [1]*fs.ndim; shape[axis] = N
        ww = w.reshape(shape)
        out[tuple(sl)] = (1-ww)*a + ww*b
    return out

def advect_velocity_axis(f, Eaxis, setting, axis, dt):
    d = setting.d; out = np.zeros_like(f)
    vgrid = setting.v_grids[axis]; dv = setting.dv[axis]; vmax = setting.vmax; Nv = setting.v_shape[axis]
    for idx in np.ndindex(*setting.x_shape):
        E = Eaxis[idx]
        dep = vgrid - E*dt
        coord = (dep + vmax)/dv
        i0 = np.floor(coord).astype(np.int64)
        w = (coord - i0).astype(np.float64)
        valid = (i0 >= 0) & (i0 < Nv-1)
        fs = f[idx]  # velocity tensor
        vals = np.zeros_like(fs)
        if valid.any():
            a = np.take(fs, i0[valid], axis=axis)
            b = np.take(fs, i0[valid]+1, axis=axis)
            shape = [1]*fs.ndim; shape[axis] = int(valid.sum())
            ww = w[valid].reshape(shape)
            interp = (1-ww)*a + ww*b
            sl = [slice(None)]*d; sl[axis] = valid
            vals[tuple(sl)] = interp
        out[idx] = vals
    return out

def run_vp_solver(setting: Setting, n_seq, n_frames, T, dt, seed, alpha_min, alpha_max, beam_u, thermal):
    rng = np.random.default_rng(seed)
    nsteps = max(1, int(math.ceil(T/dt)))
    real_dt = T/nsteps
    save_steps = np.unique(np.round(np.linspace(0, nsteps, n_frames)).astype(int))
    if len(save_steps) != n_frames:
        save_steps = np.linspace(0, nsteps, n_frames, dtype=int)
    all_f=[]; all_rho=[]; all_E=[]
    for s in range(n_seq):
        phase = [float(rng.uniform(0, 2*np.pi)) for _ in range(setting.d)]
        alpha = float(rng.uniform(alpha_min, alpha_max))
        if s == 0:
            phase = [0.0]*setting.d; alpha = max(alpha_min, min(alpha_max, 0.01))
        f = initial_f(setting, alpha=alpha, beam_u=beam_u, thermal=thermal, phase=phase)
        fs=[]; rhos=[]; Es=[]
        def store():
            r = rho_np(f, setting); E = spectral_E_np(r, setting)
            fs.append(f.astype(np.float32)); rhos.append(r.astype(np.float32)); Es.append(E.astype(np.float32))
        store()
        next_save_idx = 1
        for step in range(1, nsteps+1):
            for a in range(setting.d): f = advect_space_axis(f, setting, a, 0.5*real_dt)
            r = rho_np(f, setting); E = spectral_E_np(r, setting)
            for a in range(setting.d): f = advect_velocity_axis(f, E[a], setting, a, real_dt)
            for a in reversed(range(setting.d)): f = advect_space_axis(f, setting, a, 0.5*real_dt)
            f = np.maximum(f, 0.0)
            if next_save_idx < len(save_steps) and step >= save_steps[next_save_idx]:
                store(); next_save_idx += 1
        # ensure frames
        while len(fs) < n_frames: store()
        all_f.append(np.stack(fs[:n_frames])); all_rho.append(np.stack(rhos[:n_frames])); all_E.append(np.stack(Es[:n_frames]))
        print(f"[data] seq {s+1}/{n_seq} generated frames={len(fs[:n_frames])}")
    times = np.linspace(0, T, n_frames, dtype=np.float32)
    data = {
        'f': np.stack(all_f).astype(np.float32),
        'rho': np.stack(all_rho).astype(np.float32),
        'E': np.stack(all_E).astype(np.float32),
        'times': times,
        'x_shape': np.array(setting.x_shape, dtype=np.int64),
        'v_shape': np.array(setting.v_shape, dtype=np.int64),
        'Lx': np.array(setting.Lx, dtype=np.float32),
        'vmax': np.array(setting.vmax, dtype=np.float32),
        'dx': np.array(setting.dx, dtype=np.float32),
        'dv': np.array(setting.dv, dtype=np.float32),
        'generator': np.array('split_semi_lagrangian_vp', dtype='<U64'),
    }
    return data

def normalize_q_np(q, dv_prod, d, eps=1e-12):
    axes = tuple(range(q.ndim-d, q.ndim))
    return q / (q.sum(axis=axes, keepdims=True)*dv_prod + eps)

def get_data(args, setting: Setting):
    # external path priority
    pmap = {'1d1v': args.data_1d1v, '2d2v': args.data_2d2v, '3d3v': args.data_3d3v}
    path = pmap.get(setting.name) or args.data_path
    if path:
        # allow mapping in --data_path: 1d1v=a;2d2v=b
        if '=' in path and ';' in path:
            mp = dict(x.split('=',1) for x in path.split(';') if '=' in x)
            path = mp.get(setting.name)
        if path:
            print(f"[data] load external {setting.name}: {path}")
            z = np.load(path, allow_pickle=True)
            return {k:z[k] for k in z.files}
    root = Path(args.outdir)/f"{setting.name}_{'x'.join(map(str,setting.x_shape))}_{'v'.join(map(str,setting.v_shape))}_seed{args.seed}"/'data_cache'
    ensure_dir(root)
    fname = (f"vp_{setting.name}_x{'x'.join(map(str,setting.x_shape))}_v{'x'.join(map(str,setting.v_shape))}"
             f"_seq{args.n_seq}_tr{args.n_train_seq}_frames{args.frames}_T{args.T}_dt{args.ref_dt}"
             f"_L{setting.Lx:.4g}_vmax{setting.vmax}_a{args.alpha_min}-{args.alpha_max}"
             f"_u{args.beam_u}_th{args.thermal}_seed{args.seed}.npz")
    cache = root/fname
    if cache.exists() and not args.force_data:
        print(f"[data] load {cache}")
        z = np.load(cache, allow_pickle=True); return {k:z[k] for k in z.files}
    print(f"[data] generate {setting.name}: n_seq={args.n_seq}, frames={args.frames}, x={setting.x_shape}, v={setting.v_shape}")
    data = run_vp_solver(setting, args.n_seq, args.frames, args.T, args.ref_dt, args.seed,
                         args.alpha_min, args.alpha_max, args.beam_u, args.thermal)
    np.savez_compressed(cache, **data)
    print(f"[data] saved {cache}")
    return data

# ----------------------------- torch spectral Poisson -----------------------------

def spectral_E_torch(rho, setting: Setting):
    # rho shape [B/T..., *x]
    d = setting.d
    spatial_shape = setting.x_shape
    charge = rho - rho.mean(dim=tuple(range(rho.ndim-d, rho.ndim)), keepdim=True)
    ch = torch.fft.fftn(charge, dim=tuple(range(rho.ndim-d, rho.ndim)))
    device = rho.device
    ks = [2*math.pi*torch.fft.fftfreq(n, d=dx, device=device) for n,dx in zip(spatial_shape, setting.dx)]
    K = torch.meshgrid(*ks, indexing='ij')
    k2 = torch.zeros(spatial_shape, device=device)
    for Kj in K: k2 = k2 + Kj**2
    k2 = torch.where(k2==0, torch.ones_like(k2), k2)
    E=[]
    for Kj in K:
        shape = (1,)*(rho.ndim-d)+Kj.shape
        Eh = -1j*Kj.reshape(shape)*(ch/k2.reshape(shape))
        Ej = torch.fft.ifftn(Eh, dim=tuple(range(rho.ndim-d, rho.ndim))).real
        Ej = Ej - Ej.mean(dim=tuple(range(Ej.ndim-d, Ej.ndim)), keepdim=True)
        E.append(Ej)
    return torch.stack(E, dim=-1)  # [...,*x,d]

# ----------------------------- losses -----------------------------

def q_from_f(f, rho, setting, eps=1e-10):
    d = setting.d
    q = torch.clamp(f / (rho.reshape(rho.shape + (1,)*d) + eps), min=0.0)
    axes = tuple(range(q.ndim-d, q.ndim))
    q = q / (q.sum(dim=axes, keepdim=True)*setting.dv_prod + eps)
    return q

def weighted_rel_q_loss(q, qgt, setting, peak_weight=10.0, eps=1e-12):
    axes = tuple(range(q.ndim-setting.d, q.ndim))
    qmax = qgt.amax(dim=axes, keepdim=True).clamp_min(eps)
    w = 1.0 + peak_weight*(qgt/qmax)
    num = (w*(q-qgt)**2).sum(dim=axes)
    den = (w*qgt**2).sum(dim=axes) + eps
    return (num/den).mean()

def marginal_loss(q, qgt, setting):
    d = setting.d
    loss=0.0
    for a in range(d):
        axes = [q.ndim-d+j for j in range(d) if j != a]
        dv_other = np.prod([setting.dv[j] for j in range(d) if j != a]) if d>1 else 1.0
        mq = q.sum(dim=tuple(axes))*float(dv_other) if axes else q
        mg = qgt.sum(dim=tuple(axes))*float(dv_other) if axes else qgt
        loss = loss + ((mq-mg)**2).mean()/(mg.pow(2).mean()+1e-12)
    return loss/d

def log_density_loss(q, qgt, eps=1e-8):
    return ((torch.log(q+eps)-torch.log(qgt+eps))**2).mean()

# ----------------------------- codecs -----------------------------

class BaseCodec(nn.Module):
    def encode_state(self, f, rho, E=None): raise NotImplementedError
    def decode_state(self, S): raise NotImplementedError

class NeuralLogCodec(BaseCodec):
    def __init__(self, setting: Setting, zdim=32, hidden=128, kind='conv'):
        super().__init__(); self.setting=setting; self.d=setting.d; self.zdim=zdim; self.kind=kind
        Nv = int(np.prod(setting.v_shape)); self.Nv=Nv
        # A robust MLP codec with log-density decoder. It is named conv/diffusion by kind;
        # diffusion adds a moment base below. This avoids dimension-specific decoder bugs.
        self.enc = nn.Sequential(nn.Linear(Nv, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, zdim))
        self.dec = nn.Sequential(nn.Linear(zdim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, Nv))
        self.state_dim = 1 + zdim
        if kind == 'diffusion':
            self.moment_head = nn.Sequential(nn.Linear(Nv, hidden), nn.GELU(), nn.Linear(hidden, max(2*self.d,1)))
        else:
            self.moment_head = None
        # velocity grids for moment base
        meshes = np.meshgrid(*setting.v_grids, indexing='ij')
        self.register_buffer('Vflat', torch.tensor(np.stack([m.reshape(-1) for m in meshes],axis=-1), dtype=torch.float32))

    def _softmax_q(self, logits):
        q = torch.softmax(logits, dim=-1) / self.setting.dv_prod
        return q.reshape(logits.shape[:-1] + self.setting.v_shape)

    def _moment_base_logits(self, flat_q):
        # compute moments directly from q, not learned head. flat_q [B,Nv] density.
        w = flat_q * self.setting.dv_prod
        mean = (w.unsqueeze(-1)*self.Vflat).sum(dim=1)
        var = (w.unsqueeze(-1)*(self.Vflat-mean.unsqueeze(1))**2).sum(dim=1).clamp_min(0.15**2)
        logits = torch.zeros(flat_q.shape[0], self.Nv, device=flat_q.device)
        for j in range(self.d):
            logits = logits -0.5*((self.Vflat[:,j].reshape(1,-1)-mean[:,j:j+1])**2/var[:,j:j+1]) -0.5*torch.log(var[:,j:j+1])
        return logits

    def encode_q(self, q):
        flat = q.reshape(-1, self.Nv)
        # use normalized probability values as encoder input scale
        inp = flat * self.setting.dv_prod
        z = self.enc(inp)
        return z.reshape(q.shape[:-self.d] + (self.zdim,))

    def decode_z(self, z):
        flat_z = z.reshape(-1, self.zdim)
        logits = self.dec(flat_z)
        q = self._softmax_q(logits)
        return q.reshape(z.shape[:-1] + self.setting.v_shape)

    def forward_q(self, q):
        z = self.encode_q(q)
        return self.decode_z(z), z

    def encode_state(self, f, rho, E=None):
        q = q_from_f(f, rho, self.setting)
        z = self.encode_q(q)
        log_rho = torch.log(rho.clamp_min(1e-8)).unsqueeze(-1)
        S = torch.cat([log_rho, z], dim=-1)
        return S

    def decode_state(self, S):
        rho = torch.exp(torch.clamp(S[...,0], -20, 20))
        z = S[...,1:1+self.zdim]
        q = self.decode_z(z)
        f = rho.reshape(rho.shape + (1,)*self.d) * q
        return f, rho, q


class LinearLowRankDensityCodec(BaseCodec):
    """Density-space separable low-rank codec used by the successful data-driven 2D2V low-rank run.

    q(v|x) ~= Normalize(clamp(sum C_alpha(x) prod_j Phi_j(v_j), eps)).
    This is intentionally a *density* low-rank model, separate from the exponential log-density PI variant.
    """
    def __init__(self, setting: Setting, rank=8, eps=1e-10):
        super().__init__(); self.setting=setting; self.d=setting.d; self.rank=int(rank); self.eps=eps
        for j,nv in enumerate(setting.v_shape):
            self.register_buffer(f'Phi{j}', torch.empty(self.rank, nv))
        self.state_dim = 1 + (self.rank ** self.d)
    def bases(self): return [getattr(self, f'Phi{j}') for j in range(self.d)]
    def _q_np_from_f(self, f, rho):
        d=self.d
        q=f/(rho.reshape(rho.shape+(1,)*d)+self.eps); q=np.maximum(q,0.0)
        return normalize_q_np(q, self.setting.dv_prod, d, self.eps)
    def fit_basis_np(self, f, rho, max_samples=200000):
        d=self.d; q=self._q_np_from_f(f, rho).reshape(-1,*self.setting.v_shape)
        if q.shape[0] > max_samples:
            idx=np.linspace(0,q.shape[0]-1,max_samples).astype(np.int64); q=q[idx]
        bases=[]
        for a,nv in enumerate(self.setting.v_shape):
            # HOSVD covariance along velocity axis a, using the full joint q, not marginals.
            A=np.moveaxis(q, 1+a, -1).reshape(-1, nv)
            C=(A.T@A)/max(1,A.shape[0])
            w,V=np.linalg.eigh(C); ids=np.argsort(w)[::-1][:self.rank]
            bases.append(V[:,ids].T.astype(np.float32))
        with torch.no_grad():
            for j,b in enumerate(bases): getattr(self, f'Phi{j}').copy_(torch.tensor(b, dtype=torch.float32, device=getattr(self,f'Phi{j}').device))
    def encode_q_np(self, q):
        d=self.d; shp=q.shape[:-d]; q=q.reshape(-1,*self.setting.v_shape)
        B=[getattr(self,f'Phi{j}').detach().cpu().numpy().astype(np.float64) for j in range(d)]
        if d==1:
            C=np.einsum('bv,rv->br', q, B[0])
        elif d==2:
            C=np.einsum('bvw,rv,sw->brs', q, B[0], B[1])
        else:
            C=np.einsum('buvw,ru,sv,tw->brst', q, B[0], B[1], B[2])
        return C.reshape(shp+(self.rank**d,)).astype(np.float32)
    def encode_state_np(self, f, rho):
        q=self._q_np_from_f(f,rho); C=self.encode_q_np(q)
        return np.concatenate([np.log(rho+1e-8)[...,None].astype(np.float32), C], axis=-1)
    def encode_state(self, f, rho, E=None):
        q=q_from_f(f,rho,self.setting); C_np=self.encode_q_np(q.detach().cpu().numpy())
        C=torch.tensor(C_np, dtype=torch.float32, device=f.device)
        return torch.cat([torch.log(rho.clamp_min(1e-8)).unsqueeze(-1), C], dim=-1)
    def decode_state(self, S):
        rho=torch.exp(torch.clamp(S[...,0], -20, 20))
        C=S[...,1:1+self.rank**self.d].reshape(S.shape[:-1]+(self.rank,)*self.d)
        B=self.bases()
        if self.d==1:
            raw=torch.einsum('...r,rv->...v', C, B[0])
        elif self.d==2:
            raw=torch.einsum('...rs,rv,sw->...vw', C, B[0], B[1])
        else:
            raw=torch.einsum('...rst,ru,sv,tw->...uvw', C, B[0], B[1], B[2])
        q=torch.clamp(raw, min=self.eps)
        axes=tuple(range(q.ndim-self.d, q.ndim))
        q=q/(q.sum(dim=axes, keepdim=True)*self.setting.dv_prod+self.eps)
        f=rho.reshape(rho.shape+(1,)*self.d)*q
        return f,rho,q

class ExpLowRankLogCodec(BaseCodec):
    def __init__(self, setting: Setting, rank=8, eps=1e-8, log_floor_rel=1e-5, raw_clip=40.0):
        super().__init__(); self.setting=setting; self.d=setting.d; self.rank=int(rank); self.eps=eps; self.log_floor_rel=log_floor_rel; self.raw_clip=raw_clip
        for j,nv in enumerate(setting.v_shape):
            self.register_buffer(f'Phi{j}', torch.empty(self.rank, nv))
        self.state_dim = 1 + (self.rank ** self.d)

    def bases(self): return [getattr(self, f'Phi{j}') for j in range(self.d)]

    def _q_np_from_f(self, f, rho):
        d=self.d; q=f/(rho.reshape(rho.shape+(1,)*d)+self.eps); q=np.maximum(q,0.0)
        return normalize_q_np(q, self.setting.dv_prod, d, self.eps)

    def _centered_logq_np(self, q):
        axes=tuple(range(q.ndim-self.d,q.ndim))
        qmax=q.max(axis=axes, keepdims=True)
        qc=np.maximum(q, self.log_floor_rel*qmax+self.eps)
        logq=np.log(qc)
        return logq-logq.mean(axis=axes, keepdims=True)

    def fit_basis_np(self, f, rho, max_samples=200000):
        # f shape [N,T,*x,*v] or [*x,*v]; rho matching without velocity dims.
        d=self.d
        if f.ndim == 2*d:
            q = self._q_np_from_f(f, rho).reshape(-1, *self.setting.v_shape)
        else:
            q = self._q_np_from_f(f, rho).reshape(-1, *self.setting.v_shape)
        if q.shape[0] > max_samples:
            idx=np.linspace(0,q.shape[0]-1,max_samples).astype(np.int64); q=q[idx]
        logq=self._centered_logq_np(q)
        bases=[]
        for a,nv in enumerate(self.setting.v_shape):
            A=np.moveaxis(logq, 1+a, -1).reshape(-1, nv)
            C=(A.T@A)/max(1,A.shape[0])
            w,V=np.linalg.eigh(C); ids=np.argsort(w)[::-1][:self.rank]
            basis=V[:,ids].T.astype(np.float32)
            bases.append(basis)
        with torch.no_grad():
            for j,b in enumerate(bases): getattr(self, f'Phi{j}').copy_(torch.tensor(b, dtype=torch.float32, device=getattr(self,f'Phi{j}').device))

    def encode_q_np(self, q):
        d=self.d; shp=q.shape[:-d]; logq=self._centered_logq_np(q.reshape(-1,*self.setting.v_shape))
        B=[getattr(self,f'Phi{j}').detach().cpu().numpy().astype(np.float64) for j in range(d)]
        if d==1:
            C=np.einsum('bv,rv->br', logq, B[0])
        elif d==2:
            C=np.einsum('bvw,rv,sw->brs', logq, B[0], B[1])
        else:
            C=np.einsum('buvw,ru,sv,tw->brst', logq, B[0], B[1], B[2])
        return C.reshape(shp + (self.rank**d,)).astype(np.float32)

    def encode_state_np(self, f, rho):
        q=self._q_np_from_f(f,rho)
        C=self.encode_q_np(q)
        return np.concatenate([np.log(rho+1e-8)[...,None].astype(np.float32), C], axis=-1)

    def encode_state(self, f, rho, E=None):
        # torch path, slower but okay
        q=q_from_f(f,rho,self.setting)
        q_np=q.detach().cpu().numpy(); C_np=self.encode_q_np(q_np)
        C=torch.tensor(C_np, dtype=torch.float32, device=f.device)
        return torch.cat([torch.log(rho.clamp_min(1e-8)).unsqueeze(-1), C], dim=-1)

    def decode_state(self, S):
        rho=torch.exp(torch.clamp(S[...,0], -20, 20))
        C=S[...,1:1+self.rank**self.d].reshape(S.shape[:-1]+(self.rank,)*self.d)
        B=self.bases()
        if self.d==1:
            raw=torch.einsum('...r,rv->...v', C, B[0])
        elif self.d==2:
            raw=torch.einsum('...rs,rv,sw->...vw', C, B[0], B[1])
        else:
            raw=torch.einsum('...rst,ru,sv,tw->...uvw', C, B[0], B[1], B[2])
        axes=tuple(range(raw.ndim-self.d, raw.ndim))
        raw=raw-raw.mean(dim=axes, keepdim=True)
        raw=torch.clamp(raw, -self.raw_clip, self.raw_clip)
        q=torch.softmax(raw.reshape(raw.shape[:-self.d]+(-1,)), dim=-1).reshape(raw.shape)/self.setting.dv_prod
        f=rho.reshape(rho.shape+(1,)*self.d)*q
        return f,rho,q

# ----------------------------- networks -----------------------------

class PeriodicConvD(nn.Module):
    def __init__(self, d, cin, cout, k=3):
        super().__init__(); self.d=d; self.pad=k//2
        Conv={1:nn.Conv1d,2:nn.Conv2d,3:nn.Conv3d}[d]
        self.conv=Conv(cin, cout, k, padding=0)
    def forward(self,x):
        # x [B,C,*spatial]
        pad=[]
        for _ in range(self.d): pad += [self.pad,self.pad]
        return self.conv(F.pad(x, pad, mode='circular'))

class ResNetDynamics(nn.Module):
    def __init__(self, setting, c, width=64, depth=4):
        super().__init__(); self.d=setting.d
        self.inp=PeriodicConvD(self.d, c, width)
        self.blocks=nn.ModuleList([nn.Sequential(nn.GELU(), PeriodicConvD(self.d,width,width), nn.GELU(), PeriodicConvD(self.d,width,width)) for _ in range(depth)])
        self.out=PeriodicConvD(self.d,width,c)
        nn.init.zeros_(self.out.conv.weight); nn.init.zeros_(self.out.conv.bias)
    def forward(self,S):
        # S [B,*x,C]
        perm=[0,S.ndim-1]+list(range(1,S.ndim-1)); inv=[0]+list(range(2,S.ndim))+[1]
        x=S.permute(*perm).contiguous(); h=self.inp(x)
        for b in self.blocks: h=h+b(h)
        y=self.out(F.gelu(h)).permute(*inv).contiguous()
        return S+y

class StateNorm:
    def __init__(self, mean, std): self.mean=mean; self.std=std
    @classmethod
    def fit(cls,S):
        dims=tuple(range(S.ndim-1)); mean=S.mean(dim=dims, keepdim=True); std=S.std(dim=dims, keepdim=True).clamp_min(1e-5); return cls(mean,std)
    def n(self,S): return (S-self.mean)/self.std
    def d(self,S): return S*self.std+self.mean

# ----------------------------- train/eval helpers -----------------------------

def make_setting(name,args):
    Lmap = {'1d1v': args.Lx1, '2d2v': args.Lx2, '3d3v': args.Lx3}
    Lx = Lmap[name]
    if args.quick:
        if name=='1d1v': return Setting(name, 1, (32,), (32,), Lx=Lx, vmax=args.vmax)
        if name=='2d2v': return Setting(name, 2, (16,16), (16,16), Lx=Lx, vmax=args.vmax)
        return Setting(name, 3, (6,6,6), (8,8,8), Lx=Lx, vmax=args.vmax)
    if name=='1d1v': return Setting(name, 1, (args.Nx1,), (args.Nv1,), Lx=Lx, vmax=args.vmax)
    if name=='2d2v': return Setting(name, 2, (args.Nx2,args.Nx2), (args.Nv2,args.Nv2), Lx=Lx, vmax=args.vmax)
    return Setting(name, 3, (args.Nx3,args.Nx3,args.Nx3), (args.Nv3,args.Nv3,args.Nv3), Lx=Lx, vmax=args.vmax)

def data_tensors(data, device):
    return {k: torch.tensor(v, dtype=torch.float32, device=device) for k,v in data.items() if k in ['f','rho','E','times']}

def train_neural_codec(codec, f_train, rho_train, setting, args, device, run_dir):
    if isinstance(codec, ExpLowRankLogCodec):
        codec.fit_basis_np(f_train.detach().cpu().numpy(), rho_train.detach().cpu().numpy(), max_samples=args.lowrank_max_samples)
        return {'codec_type':'exp_lowrank_log', 'codec_oracle_rel_l2_sampled': codec_oracle(codec, f_train, rho_train, setting, max_batches=3)}
    if isinstance(codec, LinearLowRankDensityCodec):
        codec.fit_basis_np(f_train.detach().cpu().numpy(), rho_train.detach().cpu().numpy(), max_samples=args.lowrank_max_samples)
        return {'codec_type':'linear_lowrank_density', 'codec_oracle_rel_l2_sampled': codec_oracle(codec, f_train, rho_train, setting, max_batches=3)}
    codec.to(device)
    q_train=q_from_f(f_train, rho_train, setting).reshape(-1,*setting.v_shape)
    N=q_train.shape[0]
    opt=torch.optim.Adam(codec.parameters(), lr=args.ae_lr)
    bs=min(args.codec_batch, N)
    hist=[]
    for ep in range(1,args.ae_epochs+1):
        idx=torch.randint(0,N,(bs,),device=device)
        q=q_train[idx]
        opt.zero_grad(set_to_none=True)
        qh,_=codec.forward_q(q)
        loss=weighted_rel_q_loss(qh,q,setting,args.peak_weight)+args.lambda_marg*marginal_loss(qh,q,setting)+args.lambda_log*log_density_loss(qh,q)
        loss.backward(); torch.nn.utils.clip_grad_norm_(codec.parameters(),5.0); opt.step()
        if ep==1 or ep%max(1,args.ae_epochs//5)==0 or ep==args.ae_epochs:
            with torch.no_grad():
                idv=torch.linspace(0,N-1,min(N,1024),device=device).long(); qv=q_train[idv]; qhv,_=codec.forward_q(qv)
                val=float(torch.mean(rel_l2_batch(qhv.reshape(qhv.shape[0],-1), qv.reshape(qv.shape[0],-1))).item())
            print(f"[codec {codec.kind}] ep {ep}/{args.ae_epochs} loss={loss.item():.3e} val={val:.3e}")
            hist.append({'epoch':ep,'loss':float(loss.item()),'val_rel_l2':val})
    save_json(Path(run_dir)/'codec_history.json', hist)
    return {'codec_type':codec.kind, 'codec_oracle_rel_l2_sampled': codec_oracle(codec, f_train, rho_train, setting, max_batches=3)}

def codec_oracle(codec, f, rho, setting, max_batches=3):
    with torch.no_grad():
        f_flat=f.reshape(-1,*setting.x_shape,*setting.v_shape) if f.ndim==2+2*setting.d else f
        rho_flat=rho.reshape(-1,*setting.x_shape) if rho.ndim==2+setting.d else rho
        S=codec.encode_state(f_flat, rho_flat)
        fh,_,_=codec.decode_state(S)
        return float(rel_l2_batch(fh, f_flat).mean().item())

def add_E_to_state(S, E):
    return torch.cat([S, E], dim=-1)

def strip_E(S, setting, codec):
    return S[..., :codec.state_dim]

def eval_rollout(codec, dyn, norm, S0, n_frames, setting, use_E_channels=True, state_clip=0.0):
    pred=[S0]
    cur=S0
    with torch.no_grad():
        for _ in range(n_frames-1):
            nxt_n = dyn(norm.n(cur))
            if state_clip and state_clip > 0:
                nxt_n = torch.clamp(nxt_n, -state_clip, state_clip)
            nxt=norm.d(nxt_n)
            pred.append(nxt); cur=nxt
    return torch.stack(pred, dim=1)

def decode_sequence(codec, Sseq, setting, has_E=True):
    B,T=Sseq.shape[:2]
    f_list=[]; rho_list=[]; q_list=[]
    with torch.no_grad():
        for t in range(T):
            S=strip_E(Sseq[:,t], setting, codec) if has_E else Sseq[:,t]
            f,rho,q=codec.decode_state(S)
            f_list.append(f); rho_list.append(rho); q_list.append(q)
    return torch.stack(f_list,1), torch.stack(rho_list,1), torch.stack(q_list,1)

def h1_metrics(f_pred, f_gt, setting):
    d=setting.d
    def h1(x):
        val=(x*x).mean()
        for j in range(d): val=val+(central_diff_periodic(x, dim=1+j, dx=setting.dx[j])**2).mean()
        for j in range(d): val=val+(central_diff_periodic(x, dim=1+d+j, dx=setting.dv[j])**2).mean()
        return torch.sqrt(val+1e-12)
    errs=[]
    for t in range(f_pred.shape[1]): errs.append(float((h1(f_pred[:,t]-f_gt[:,t])/(h1(f_gt[:,t])+1e-12)).item()))
    return float(np.mean(errs)), float(errs[-1])

# ----------------------------- figures -----------------------------

def save_im(path, arr, title='', cmap='viridis'):
    ensure_dir(Path(path).parent); plt.figure(figsize=(4,3)); plt.imshow(arr, origin='lower', aspect='auto', cmap=cmap); plt.colorbar(); plt.title(title); plt.tight_layout(); plt.savefig(path,dpi=150); plt.close()

def save_line(path, ys, labels, title=''):
    ensure_dir(Path(path).parent); plt.figure(figsize=(5,3))
    for y,l in zip(ys,labels): plt.plot(y,label=l)
    plt.legend(); plt.title(title); plt.tight_layout(); plt.savefig(path,dpi=150); plt.close()

def save_triplet(path, a,b,c,titles):
    ensure_dir(Path(path).parent); fig,axs=plt.subplots(1,3,figsize=(10,3))
    for ax,img,ti in zip(axs,[a,b,c],titles):
        if img.ndim==1: ax.plot(img)
        else: im=ax.imshow(img,origin='lower',aspect='auto'); fig.colorbar(im,ax=ax,fraction=0.046)
        ax.set_title(ti)
    plt.tight_layout(); plt.savefig(path,dpi=150); plt.close()

def reduce_rho_for_plot(rho, setting):
    if setting.d==1: return rho
    if setting.d==2: return rho
    return rho[:,:,rho.shape[2]//2]

def reduce_q_for_plot(q, setting):
    if setting.d==1: return q
    if setting.d==2: return q
    return q[:,:,q.shape[2]//2]

def save_evolution_figures(outdir, f_pred, f_gt, rho_pred, rho_gt, q_pred, q_gt, setting, max_frames=None):
    ensure_dir(outdir)
    B,T=f_pred.shape[:2]
    frames=range(T) if max_frames is None or max_frames<=0 else np.linspace(0,T-1,min(T,max_frames)).astype(int)
    # use first test sequence and center spatial point
    center=tuple([n//2 for n in setting.x_shape])
    for t in frames:
        rp=reduce_rho_for_plot(rho_pred[0,t].detach().cpu().numpy(), setting)
        rg=reduce_rho_for_plot(rho_gt[0,t].detach().cpu().numpy(), setting)
        re=np.abs(rp-rg)
        save_triplet(Path(outdir)/f'rho_triplet_t{t:03d}.png', rg,rp,re, ['rho GT','rho Pred','abs err'])
        qp=q_pred[(0,t)+center].detach().cpu().numpy()
        qg=q_gt[(0,t)+center].detach().cpu().numpy()
        qi=reduce_q_for_plot(qp,setting); qgi=reduce_q_for_plot(qg,setting); qe=np.abs(qi-qgi)
        save_triplet(Path(outdir)/f'q_center_triplet_t{t:03d}.png', qgi,qi,qe, ['q GT center','q Pred center','abs err'])

# ----------------------------- reports -----------------------------

def make_report_md(rows):
    lines=['# Master report','', '| setting | mode | model | mean L2 | final L2 | mean H1 | final H1 | codec oracle | note/error |', '|---|---|---|---:|---:|---:|---:|---:|---|']
    for r in rows:
        lines.append(f"| {r.get('setting','')} | {r.get('mode','')} | {r.get('model','')} | {r.get('test_mean_rel_l2','')} | {r.get('test_final_rel_l2','')} | {r.get('test_mean_rel_h1','')} | {r.get('test_final_rel_h1','')} | {r.get('codec_oracle_rel_l2_sampled','')} | {r.get('note', r.get('error',''))} |")
    return '\n'.join(lines)+'\n'

def update_master(outdir, row, reset=False):
    path=Path(outdir)/'master_report.json'
    if reset or not path.exists(): rows=[]
    else:
        try: rows=json.load(open(path,'r',encoding='utf-8'))
        except Exception: rows=[]
    # replace same key if exists
    key=(row.get('setting'), row.get('mode'), row.get('model'))
    rows=[r for r in rows if (r.get('setting'),r.get('mode'),r.get('model'))!=key]
    rows.append(row)
    rows=sorted(rows, key=lambda r:(r.get('setting',''),r.get('mode',''),r.get('model','')))
    save_json(path, rows); write_md(Path(outdir)/'master_report.md', make_report_md(rows))

# ----------------------------- data mode -----------------------------

def make_codec(model, setting, args):
    if model in ['lowrank_pi','lowrank_exp']:
        return ExpLowRankLogCodec(setting, rank=args.lowrank_rank, log_floor_rel=args.log_floor_rel, raw_clip=args.raw_clip)
    if model in ['lowrank_data','lowrank_linear']:
        return LinearLowRankDensityCodec(setting, rank=args.lowrank_rank)
    if model=='conv': return NeuralLogCodec(setting, zdim=args.zdim, hidden=args.codec_hidden, kind='conv')
    if model=='diffusion': return NeuralLogCodec(setting, zdim=args.zdim, hidden=args.codec_hidden, kind='diffusion')
    raise ValueError(model)

def train_data_mode(setting, model, data, args, run_dir, device):
    ensure_dir(run_dir); save_json(Path(run_dir)/'config.json', vars(args)|{'setting':setting.name,'mode':'data','model':model})
    D=data_tensors(data,device); f=D['f']; rho=D['rho']; E=D['E']
    n_train=max(1,args.n_train_seq); ftr=f[:n_train]; rhotr=rho[:n_train]; Etr=E[:n_train]
    fte=f[n_train:]; rhote=rho[n_train:]; Ete=E[n_train:]
    if fte.shape[0]==0: fte=f[-1:]; rhote=rho[-1:]; Ete=E[-1:]
    codec=make_codec(model,setting,args).to(device)
    meta=train_neural_codec(codec, ftr, rhotr, setting, args, device, run_dir)
    with torch.no_grad():
        S_train=codec.encode_state(ftr.reshape(-1,*setting.x_shape,*setting.v_shape), rhotr.reshape(-1,*setting.x_shape)).reshape(ftr.shape[0],ftr.shape[1],*setting.x_shape,codec.state_dim)
        S_test=codec.encode_state(fte.reshape(-1,*setting.x_shape,*setting.v_shape), rhote.reshape(-1,*setting.x_shape)).reshape(fte.shape[0],fte.shape[1],*setting.x_shape,codec.state_dim)
        E_train=Etr.permute(0,1,*range(3,3+setting.d),2).contiguous() if Etr.ndim==3+setting.d else Etr.movedim(2,-1)
        E_test=Ete.movedim(2,-1)
        S_trainE=torch.cat([S_train,E_train], dim=-1); S_testE=torch.cat([S_test,E_test], dim=-1)
    pairs_in=S_trainE[:,:-1].reshape(-1,*setting.x_shape,S_trainE.shape[-1])
    pairs_out=S_trainE[:,1:].reshape(-1,*setting.x_shape,S_trainE.shape[-1])
    # True next-frame tensors matched to flattened pairs. This is important: the
    # decoded-f loss must be against the real target f, not against decode(encoded target state).
    f_targets=ftr[:,1:].reshape(-1,*setting.x_shape,*setting.v_shape)
    rho_targets=rhotr[:,1:].reshape(-1,*setting.x_shape)
    E_targets=E_train[:,1:].reshape(-1,*setting.x_shape,setting.d)
    norm=StateNorm.fit(pairs_in); dyn=ResNetDynamics(setting,pairs_in.shape[-1],width=args.dyn_width,depth=args.dyn_depth).to(device)
    dyn_lr = args.lowrank_dyn_lr if model in ['lowrank_data','lowrank_linear'] else args.lr
    opt=torch.optim.AdamW(dyn.parameters(), lr=dyn_lr, weight_decay=args.weight_decay)
    best=None; last=None; t0=time.time()
    qgt_all=None
    for ep in range(1,args.epochs+1):
        idx=torch.randint(0,pairs_in.shape[0],(min(args.batch_size,pairs_in.shape[0]),),device=device)
        x=pairs_in[idx]; y=pairs_out[idx]
        opt.zero_grad(set_to_none=True)
        pred_n=dyn(norm.n(x))
        if args.state_clip and args.state_clip > 0:
            pred_n=torch.clamp(pred_n, -args.state_clip, args.state_clip)
        pred=norm.d(pred_n)
        loss_state=F.mse_loss(pred,y)
        fb=min(args.f_loss_batch, pred.shape[0])
        f_tgt_b=f_targets[idx[:fb]]; rho_tgt_b=rho_targets[idx[:fb]]; E_tgt_b=E_targets[idx[:fb]]
        f_pred_b,rho_pred_b,q_pred_b=codec.decode_state(strip_E(pred[:fb], setting, codec))
        q_tgt_b=q_from_f(f_tgt_b, rho_tgt_b, setting)
        pred_E_b=pred[:fb,...,codec.state_dim:codec.state_dim+setting.d]
        loss_f_mse=((f_pred_b-f_tgt_b)**2).mean()/(f_tgt_b.pow(2).mean().detach()+1e-12)
        loss_f_rel=rel_l2_batch(f_pred_b, f_tgt_b).mean()
        loss_q=weighted_rel_q_loss(q_pred_b, q_tgt_b, setting, peak_weight=args.peak_weight)
        loss_rho=rel_l2_batch(rho_pred_b, rho_tgt_b).mean()
        loss_E=((pred_E_b-E_tgt_b)**2).mean()/(E_tgt_b.pow(2).mean().detach()+1e-12)
        loss=(args.lambda_state*loss_state + args.lambda_f_data*loss_f_mse +
              args.lambda_f_rel_data*loss_f_rel + args.lambda_q_data*loss_q +
              args.lambda_rho_data*loss_rho + args.lambda_E_data*loss_E)
        loss.backward(); torch.nn.utils.clip_grad_norm_(dyn.parameters(),args.grad_clip); opt.step()
        if ep==1 or ep%args.eval_every==0 or ep==args.epochs:
            with torch.no_grad():
                Sp=eval_rollout(codec,dyn,norm,S_testE[:,0],S_testE.shape[1],setting,True,state_clip=args.state_clip)
                fp,rp,qp=decode_sequence(codec,Sp,setting,has_E=True)
                if qgt_all is None:
                    qgt_all=q_from_f(fte.reshape(-1,*setting.x_shape,*setting.v_shape), rhote.reshape(-1,*setting.x_shape), setting).reshape(fte.shape)
                mean=float(rel_l2_batch(fp.reshape(fp.shape[0],-1), fte.reshape(fte.shape[0],-1)).mean().item())
                final=float(rel_l2_batch(fp[:,-1], fte[:,-1]).mean().item())
                mh,fh=(None,None) if args.no_h1 else h1_metrics(fp,fte,setting)
            row={'epoch':ep,'time_sec':time.time()-t0,'loss':float(loss.item()),'loss_state':float(loss_state.item()),'loss_f_mse':float(loss_f_mse.item()),'loss_f_rel':float(loss_f_rel.item()),'loss_q':float(loss_q.item()),'loss_rho':float(loss_rho.item()),'loss_E':float(loss_E.item()),'test_mean_rel_l2':mean,'test_final_rel_l2':final,'test_mean_rel_h1':mh,'test_final_rel_h1':fh}
            last=row.copy()
            append_csv(Path(run_dir)/'history.csv',row,['epoch','time_sec','loss','loss_state','loss_f_mse','loss_f_rel','loss_q','loss_rho','loss_E','test_mean_rel_l2','test_final_rel_l2','test_mean_rel_h1','test_final_rel_h1'])
            print(f"[data {setting.name}/{model}] ep {ep} loss={loss.item():.3e} mean={mean:.3e} final={final:.3e}")
            if best is None or mean<best['test_mean_rel_l2']:
                best=row.copy(); torch.save({'dyn':dyn.state_dict(),'codec':codec.state_dict(),'epoch':ep,'best':best}, Path(run_dir)/'checkpoint_best.pt')
                save_evolution_figures(Path(run_dir)/'figures_evolution_best',fp,fte,rp,rhote,qp,qgt_all,setting,args.max_plot_frames)
                np.savez_compressed(Path(run_dir)/'prediction_all_frames_best.npz', f_pred=fp.detach().cpu().numpy(), f_gt=fte.detach().cpu().numpy(), rho_pred=rp.detach().cpu().numpy(), rho_gt=rhote.detach().cpu().numpy())
    # Save last and then reload best for the primary report. This prevents a late unstable
    # coefficient rollout from being reported as the method result.
    save_evolution_figures(Path(run_dir)/'figures_evolution_last',fp,fte,rp,rhote,qp,qgt_all,setting,args.max_plot_frames)
    np.savez_compressed(Path(run_dir)/'prediction_all_frames_last.npz', f_pred=fp.detach().cpu().numpy(), f_gt=fte.detach().cpu().numpy(), rho_pred=rp.detach().cpu().numpy(), rho_gt=rhote.detach().cpu().numpy())
    ckpt_path=Path(run_dir)/'checkpoint_best.pt'
    if ckpt_path.exists() and args.report_best:
        ckpt=torch.load(ckpt_path, map_location=device); dyn.load_state_dict(ckpt['dyn']); best=ckpt.get('best',best)
        with torch.no_grad():
            Sp=eval_rollout(codec,dyn,norm,S_testE[:,0],S_testE.shape[1],setting,True,state_clip=args.state_clip)
            fp,rp,qp=decode_sequence(codec,Sp,setting,has_E=True)
            mean=float(rel_l2_batch(fp.reshape(fp.shape[0],-1), fte.reshape(fte.shape[0],-1)).mean().item())
            final=float(rel_l2_batch(fp[:,-1], fte[:,-1]).mean().item())
            mh,fh=(None,None) if args.no_h1 else h1_metrics(fp,fte,setting)
        save_evolution_figures(Path(run_dir)/'figures_evolution_reported',fp,fte,rp,rhote,qp,qgt_all,setting,args.max_plot_frames)
        np.savez_compressed(Path(run_dir)/'prediction_all_frames_reported.npz', f_pred=fp.detach().cpu().numpy(), f_gt=fte.detach().cpu().numpy(), rho_pred=rp.detach().cpu().numpy(), rho_gt=rhote.detach().cpu().numpy())
    rep={'setting':setting.name,'model':model,'mode':'data','epoch':best.get('epoch',args.epochs) if best else args.epochs,'test_mean_rel_l2':mean,'test_final_rel_l2':final,'test_mean_rel_h1':mh,'test_final_rel_h1':fh,'best':best,'last':last,'reported':'best_checkpoint' if args.report_best else 'last_checkpoint','state_dim':codec.state_dim+setting.d,'dyn_params':count_params(dyn)}|meta
    if meta.get('codec_oracle_rel_l2_sampled',0)>args.codec_warn_threshold: rep['note']=f"WARNING: codec oracle > {args.codec_warn_threshold}; dynamics result may be codec-limited."
    if rep.get('reported') == 'best_checkpoint':
        rep['note'] = (rep.get('note','') + ' reported best checkpoint; last metrics are in report.json.' ).strip()
    save_json(Path(run_dir)/'report.json',rep); write_md(Path(run_dir)/'report.md',make_report_md([rep])); return rep

# ----------------------------- PI mode -----------------------------


def optimize_lowrank_ic(codec, S0, f0, rho0, setting, steps=500, lr=5e-2, peak_weight=10.0):
    """Optimize only the low-rank coefficients at t=0 against the given initial condition."""
    if steps <= 0:
        return S0
    log_rho = S0[..., :1].detach()
    C = nn.Parameter(S0[..., 1:].detach().clone())
    opt = torch.optim.Adam([C], lr=lr)
    q0 = q_from_f(f0, rho0, setting)[0]
    best = None; best_loss = 1e99
    for it in range(1, steps+1):
        opt.zero_grad(set_to_none=True)
        S = torch.cat([log_rho, C], dim=-1)
        _, _, q = codec.decode_state(S.unsqueeze(0))
        q = q[0]
        loss = weighted_rel_q_loss(q.unsqueeze(0), q0.unsqueeze(0), setting, peak_weight=peak_weight)
        loss = loss + 0.2*marginal_loss(q.unsqueeze(0), q0.unsqueeze(0), setting) + 1e-3*log_density_loss(q, q0)
        loss.backward(); torch.nn.utils.clip_grad_norm_([C], 5.0); opt.step()
        lv = float(loss.detach().item())
        if lv < best_loss:
            best_loss = lv; best = C.detach().clone()
    return torch.cat([log_rho, best], dim=-1).detach()

class PIStates(nn.Module):
    def __init__(self,Sinit,T):
        super().__init__(); self.S0=Sinit.detach().clone(); self.rest=nn.Parameter(Sinit.detach().clone().unsqueeze(0).repeat(T-1,*([1]*(Sinit.ndim))))
    def forward(self): return torch.cat([self.S0.unsqueeze(0), self.rest], dim=0)
    def clamp_initial(self): pass

def pde_residual_loss(codec, Sseq, setting, times):
    # Sseq [T,*x,C]
    T=Sseq.shape[0]; dt=float((times[1]-times[0]).detach().cpu().item()) if T>1 else 1.0
    f,rho,q=codec.decode_state(Sseq)  # [T,*x,*v]
    E=spectral_E_torch(rho, setting)  # [T,*x,d]
    res=[]
    # velocity broadcast tensors
    meshes=torch.meshgrid(*[torch.tensor(g,device=f.device,dtype=f.dtype) for g in setting.v_grids], indexing='ij')
    V=[m.reshape((1,)+(1,)*setting.d+setting.v_shape) for m in meshes]
    for t in range(T-1):
        ft=(f[t+1]-f[t])/dt
        r=ft
        for a in range(setting.d):
            gx=central_diff_periodic(f[t:t+1], dim=1+a, dx=setting.dx[a])[0]
            gv=central_diff_periodic(f[t:t+1], dim=1+setting.d+a, dx=setting.dv[a])[0]
            Ea=E[t,...,a].reshape(setting.x_shape+(1,)*setting.d)
            r = r + V[a].squeeze(0)*gx + Ea*gv
        res.append(r)
    res=torch.stack(res)
    scale=(f[:-1].pow(2).mean().detach()+1e-8)
    return (res.pow(2).mean()/scale), f, rho, q

def train_pi_mode(setting, model, data, args, run_dir, device):
    ensure_dir(run_dir); save_json(Path(run_dir)/'config.json', vars(args)|{'setting':setting.name,'mode':'pi','model':model})
    D=data_tensors(data,device); fgt=D['f']; rhogt=D['rho']; times=D['times']
    seq=args.pi_seq_index if args.pi_seq_index < fgt.shape[0] else 0
    fseq=fgt[seq]; rhoseq=rhogt[seq]
    codec=make_codec(model,setting,args).to(device)
    # Fit codec using IC only for all models to stay pure PI.
    f0=fseq[0:1]; rho0=rhoseq[0:1]
    if isinstance(codec,ExpLowRankLogCodec):
        codec.fit_basis_np(f0.detach().cpu().numpy(), rho0.detach().cpu().numpy(), max_samples=args.lowrank_max_samples)
        with torch.no_grad():
            S0=codec.encode_state(f0, rho0)[0]
        S0 = optimize_lowrank_ic(codec, S0, f0, rho0, setting, steps=args.pi_ic_steps, lr=args.pi_ic_lr, peak_weight=args.peak_weight)
    else:
        # train neural decoder on initial q only; use pi_ic_steps, not full ae_epochs.
        old_ae = args.ae_epochs
        args.ae_epochs = args.pi_ic_steps
        train_neural_codec(codec, f0, rho0, setting, args, device, run_dir)
        args.ae_epochs = old_ae
        with torch.no_grad():
            S0=codec.encode_state(f0, rho0)[0]
    traj=PIStates(S0, fseq.shape[0]).to(device)
    opt=torch.optim.Adam(list(traj.parameters())+([] if args.freeze_pi_codec else list(codec.parameters())), lr=args.pi_lr)
    best=None; t0=time.time(); header=['epoch','time_sec','active_T','loss','loss_pde','loss_ic','loss_mass','loss_smooth','ic_rel_l2','test_mean_rel_l2','test_final_rel_l2','test_mean_rel_h1','test_final_rel_h1']
    # initial sanity
    with torch.no_grad():
        fi,_,_=codec.decode_state(S0.unsqueeze(0)); ic0=float(rel_l2_batch(fi,f0).mean().item())
    for ep in range(1,args.pi_epochs+1):
        active_T=min(fseq.shape[0], max(2, int(2 + (fseq.shape[0]-2)*ep/max(1,args.pi_curriculum_epochs))))
        S=traj()[:active_T]
        opt.zero_grad(set_to_none=True)
        lpde, fp, rp, qp=pde_residual_loss(codec,S,setting,times[:active_T])
        f0p,_,_=codec.decode_state(S[0:1])
        lic=rel_l2_batch(f0p,f0).mean()
        mass=(fp.reshape(active_T,-1).sum(dim=1)*np.prod(setting.dx)*setting.dv_prod); lmass=((mass-mass[0])**2).mean()/(mass[0].detach()**2+1e-12)
        lsmooth=((S[1:]-S[:-1])**2).mean() if active_T>1 else torch.tensor(0.,device=device)
        loss=args.lambda_pde*lpde + args.lambda_ic*lic + args.lambda_mass*lmass + args.lambda_smooth*lsmooth
        loss.backward(); torch.nn.utils.clip_grad_norm_(list(traj.parameters())+list(codec.parameters()),5.0); opt.step()
        if ep==1 or ep%args.eval_every==0 or ep==args.pi_epochs:
            with torch.no_grad():
                Sfull=traj(); fpr,rpr,qpr=codec.decode_state(Sfull)
                qgt=q_from_f(fseq, rhoseq, setting)
                mean=float(torch.linalg.norm((fpr-fseq).reshape(fpr.shape[0],-1),dim=1).div(torch.linalg.norm(fseq.reshape(fseq.shape[0],-1),dim=1)+1e-12).mean().item())
                final=float(torch.linalg.norm((fpr[-1]-fseq[-1]).reshape(-1))/(torch.linalg.norm(fseq[-1].reshape(-1))+1e-12))
                mh,fh=(None,None) if args.no_h1 else h1_metrics(fpr.unsqueeze(0), fseq.unsqueeze(0), setting)
                ic=float(rel_l2_batch(f0p,f0).mean().item())
            row={'epoch':ep,'time_sec':time.time()-t0,'active_T':active_T,'loss':float(loss.item()),'loss_pde':float(lpde.item()),'loss_ic':float(lic.item()),'loss_mass':float(lmass.item()),'loss_smooth':float(lsmooth.item()),'ic_rel_l2':ic,'test_mean_rel_l2':mean,'test_final_rel_l2':final,'test_mean_rel_h1':mh,'test_final_rel_h1':fh}
            append_csv(Path(run_dir)/'history.csv',row,header)
            print(f"[PI {setting.name}/{model}] ep {ep} activeT={active_T} loss={loss.item():.3e} mean={mean:.3e} final={final:.3e} ic={ic:.3e}")
            if best is None or mean<best['test_mean_rel_l2']:
                best=row.copy(); torch.save({'traj':traj.state_dict(),'codec':codec.state_dict(),'epoch':ep,'best':best}, Path(run_dir)/'checkpoint_best.pt')
                save_evolution_figures(Path(run_dir)/'figures_evolution_best',fpr.unsqueeze(0),fseq.unsqueeze(0),rpr.unsqueeze(0),rhoseq.unsqueeze(0),qpr.unsqueeze(0),qgt.unsqueeze(0),setting,args.max_plot_frames)
    save_evolution_figures(Path(run_dir)/'figures_evolution_final',fpr.unsqueeze(0),fseq.unsqueeze(0),rpr.unsqueeze(0),rhoseq.unsqueeze(0),qpr.unsqueeze(0),qgt.unsqueeze(0),setting,args.max_plot_frames)
    rep={'setting':setting.name,'model':model,'mode':'pi','epoch':args.pi_epochs,'test_mean_rel_l2':mean,'test_final_rel_l2':final,'test_mean_rel_h1':mh,'test_final_rel_h1':fh,'best':best,'state_dim':codec.state_dim,'train_params':count_params(traj)+(0 if args.freeze_pi_codec else count_params(codec)),'codec_oracle_rel_l2_sampled':ic0,'note':'Pure PI: future GT used only for evaluation/figures; E is solved from predicted rho.'}
    save_json(Path(run_dir)/'report.json',rep); write_md(Path(run_dir)/'report.md',make_report_md([rep])); return rep

# ----------------------------- args/main -----------------------------

def build_parser():
    p=argparse.ArgumentParser()
    p.add_argument('--settings',type=str,default='2d2v')
    p.add_argument('--mode',type=str,default='both',choices=['data','pi','both'])
    p.add_argument('--models',type=str,default='all')
    p.add_argument('--outdir',type=str,default='./outputs_kvp_unified_repaired')
    p.add_argument('--device',type=str,default='cuda')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--quick',action='store_true')
    p.add_argument('--reset_master',action='store_true')
    # data dims
    p.add_argument('--Nx1',type=int,default=128); p.add_argument('--Nv1',type=int,default=128)
    p.add_argument('--Nx2',type=int,default=32); p.add_argument('--Nv2',type=int,default=32)
    p.add_argument('--Nx3',type=int,default=8); p.add_argument('--Nv3',type=int,default=16)
    p.add_argument('--vmax',type=float,default=6.0); p.add_argument('--T',type=float,default=5.0); p.add_argument('--frames',type=int,default=26); p.add_argument('--ref_dt',type=float,default=0.05)
    p.add_argument('--Lx1',type=float,default=6.283185307179586); p.add_argument('--Lx2',type=float,default=12.566370614359172); p.add_argument('--Lx3',type=float,default=6.283185307179586)
    p.add_argument('--n_seq',type=int,default=5); p.add_argument('--n_train_seq',type=int,default=4)
    p.add_argument('--alpha_min',type=float,default=0.005); p.add_argument('--alpha_max',type=float,default=0.03); p.add_argument('--beam_u',type=float,default=2.4); p.add_argument('--thermal',type=float,default=1.0)
    p.add_argument('--force_data',action='store_true')
    p.add_argument('--data_path',type=str,default=''); p.add_argument('--data_1d1v',type=str,default=''); p.add_argument('--data_2d2v',type=str,default=''); p.add_argument('--data_3d3v',type=str,default='')
    # model/train
    p.add_argument('--zdim',type=int,default=32); p.add_argument('--codec_hidden',type=int,default=160); p.add_argument('--lowrank_rank',type=int,default=8); p.add_argument('--lowrank_max_samples',type=int,default=200000)
    p.add_argument('--log_floor_rel',type=float,default=1e-5); p.add_argument('--raw_clip',type=float,default=40.0)
    p.add_argument('--ae_epochs',type=int,default=500); p.add_argument('--ae_lr',type=float,default=1e-3); p.add_argument('--codec_batch',type=int,default=2048)
    p.add_argument('--peak_weight',type=float,default=10.0); p.add_argument('--lambda_marg',type=float,default=0.2); p.add_argument('--lambda_log',type=float,default=1e-3)
    p.add_argument('--epochs',type=int,default=300); p.add_argument('--lr',type=float,default=5e-4); p.add_argument('--lowrank_dyn_lr',type=float,default=2e-4); p.add_argument('--weight_decay',type=float,default=1e-6); p.add_argument('--grad_clip',type=float,default=1.0); p.add_argument('--state_clip',type=float,default=8.0); p.add_argument('--batch_size',type=int,default=8); p.add_argument('--dyn_width',type=int,default=96); p.add_argument('--dyn_depth',type=int,default=4)
    p.add_argument('--lambda_state',type=float,default=0.2); p.add_argument('--lambda_f_data',type=float,default=1.0); p.add_argument('--lambda_f_rel_data',type=float,default=0.1); p.add_argument('--lambda_q_data',type=float,default=0.1); p.add_argument('--lambda_rho_data',type=float,default=0.1); p.add_argument('--lambda_E_data',type=float,default=0.1); p.add_argument('--f_loss_batch',type=int,default=2); p.add_argument('--report_best',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--pi_epochs',type=int,default=3000); p.add_argument('--pi_lr',type=float,default=2e-3); p.add_argument('--pi_ic_steps',type=int,default=500); p.add_argument('--pi_ic_lr',type=float,default=5e-2); p.add_argument('--pi_curriculum_epochs',type=int,default=500); p.add_argument('--pi_seq_index',type=int,default=0)
    p.add_argument('--lambda_pde',type=float,default=1.0); p.add_argument('--lambda_ic',type=float,default=10.0); p.add_argument('--lambda_mass',type=float,default=0.1); p.add_argument('--lambda_smooth',type=float,default=1e-4)
    p.add_argument('--freeze_pi_codec',action='store_true',default=True)
    p.add_argument('--eval_every',type=int,default=50); p.add_argument('--no_h1',action='store_true'); p.add_argument('--max_plot_frames',type=int,default=0); p.add_argument('--save_every',type=int,default=None)
    p.add_argument('--codec_warn_threshold',type=float,default=0.05)
    p.add_argument('--torch_threads',type=int,default=0)
    return p

def main():
    args=build_parser().parse_args()
    if args.torch_threads and args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    if getattr(args,'save_every',None) is not None:
        args.max_plot_frames = 0 if args.save_every <= 1 else args.max_plot_frames
    set_seed(args.seed)
    device=torch.device(args.device if args.device=='cpu' or torch.cuda.is_available() else 'cpu')
    settings=parse_csv(args.settings,['1d1v','2d2v','3d3v'])
    raw_models=parse_csv(args.models,['conv','diffusion','lowrank_data','lowrank_pi'])
    models=[]
    for m in raw_models:
        if m in ['lowrank','lowrank_all']:
            models += ['lowrank_data','lowrank_pi']
        else:
            models.append(m)
    # de-duplicate while preserving order
    models=list(dict.fromkeys(models))
    modes=['data','pi'] if args.mode=='both' else [args.mode]
    if args.reset_master:
        for fn in ['master_report.json','master_report.md']:
            p=Path(args.outdir)/fn
            if p.exists(): p.unlink()
    for sname in settings:
        setting=make_setting(sname,args)
        setting_dir=Path(args.outdir)/f"{sname}_{'x'.join(map(str,setting.x_shape))}_{'v'.join(map(str,setting.v_shape))}_seed{args.seed}"
        data=get_data(args,setting)
        setting_rows=[]
        for mode in modes:
            for model in models:
                if mode == 'data' and model == 'lowrank_pi':
                    continue
                if mode == 'pi' and model == 'lowrank_data':
                    continue
                run_dir=setting_dir/f"{mode}_{model}"
                print(f"\n========== RUN {sname} / {mode} / {model} ==========")
                try:
                    if mode=='data': rep=train_data_mode(setting,model,data,args,run_dir,device)
                    else: rep=train_pi_mode(setting,model,data,args,run_dir,device)
                except Exception as e:
                    import traceback; traceback.print_exc()
                    rep={'setting':sname,'mode':mode,'model':model,'error':repr(e)}
                    save_json(run_dir/'report.json',rep)
                update_master(args.outdir,rep,reset=False)
                setting_rows.append(rep); save_json(setting_dir/'setting_report.json',setting_rows); write_md(setting_dir/'setting_report.md',make_report_md(setting_rows))
        save_json(setting_dir/'setting_done.json',{'setting':sname,'done_time':time.time(),'n_runs':len(setting_rows)})
    print(f"\n[done] master report: {Path(args.outdir)/'master_report.md'}")

if __name__=='__main__': main()
