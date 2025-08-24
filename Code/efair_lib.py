
import numpy as np
import pandas as pd
import math, random, time
from typing import Dict, Tuple, List

from sklearn.metrics import roc_auc_score, accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

def make_synthetic(n=5000, d=10, seed=0, rho=0.6, base_rate_gap=0.2):
    rng = np.random.RandomState(seed)
    A = rng.binomial(1, 0.5, size=n).astype(np.int64)
    X = rng.randn(n, d).astype(np.float32)
    X[:, 0] += rho * (A * 2 - 1).astype(np.float32)
    w = rng.randn(d).astype(np.float32) / np.sqrt(d)
    score = X @ w + 0.2 * np.tanh(X[:, 1]) + 0.2 * (X[:, 2] ** 2 - 1.0)
    shift = base_rate_gap * (A - 0.5) * 2.0
    p = 1 / (1 + np.exp(-(score + shift)))
    Y = (rng.rand(n) < p).astype(np.int64)
    return X, A, Y

def demographic_parity(yhat, A):
    yhat = (yhat > 0.5).astype(int)
    p1 = yhat[A==1].mean() if (A==1).sum() > 0 else 0.0
    p0 = yhat[A==0].mean() if (A==0).sum() > 0 else 0.0
    return abs(p1 - p0), p1, p0

def equal_opportunity(yhat, y, A):
    yhat = (yhat > 0.5).astype(int)
    pos = (y == 1)
    if pos.sum() == 0:
        return 0.0, 0.0, 0.0
    tpr1 = (yhat[(A==1) & pos] == 1).mean() if ((A==1) & pos).sum() > 0 else 0.0
    tpr0 = (yhat[(A==0) & pos] == 1).mean() if ((A==0) & pos).sum() > 0 else 0.0
    return abs(tpr1 - tpr0), tpr1, tpr0

def plug_in_mi_binary(A, B):
    A = np.asarray(A).astype(int)
    B = np.asarray(B).astype(int)
    n = len(A)
    pA = np.bincount(A, minlength=2) / n
    pB = np.bincount(B, minlength=2) / n
    joint = np.zeros((2,2), dtype=float)
    for a in [0,1]:
        for b in [0,1]:
            joint[a,b] = ((A==a) & (B==b)).sum() / n
    mi = 0.0
    for a in [0,1]:
        for b in [0,1]:
            if joint[a,b] > 0 and pA[a] > 0 and pB[b] > 0:
                mi += joint[a,b] * math.log(joint[a,b] / (pA[a]*pB[b] + 1e-12) + 1e-12)
    return mi

def dp_upper_from_mi(mi, A):
    A = np.asarray(A).astype(int)
    p1 = A.mean(); p0 = 1 - p1
    return math.sqrt(max(mi, 0.0) / (2.0 * p0 * p1 + 1e-12))

def eo_upper_from_mi(mi_cond, A_pos):
    A_pos = np.asarray(A_pos).astype(int)
    p1 = A_pos.mean(); p0 = 1 - p1
    return math.sqrt(max(mi_cond, 0.0) / (2.0 * p0 * p1 + 1e-12))

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd; return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None

def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)

class Encoder(nn.Module):
    def __init__(self, d_in, d_z=8):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 64), nn.ReLU(), nn.Linear(64, d_z))
    def forward(self, x):
        return self.net(x)

class ClassifierHead(nn.Module):
    def __init__(self, d_z):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_z, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, z):
        return torch.sigmoid(self.net(z))

class AdvAHead(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, h):
        return torch.sigmoid(self.net(h))

class CLUBGaussian(nn.Module):
    def __init__(self, d_z):
        super().__init__()
        self.mu0 = nn.Parameter(torch.zeros(d_z))
        self.mu1 = nn.Parameter(torch.zeros(d_z))
        self.logvar0 = nn.Parameter(torch.zeros(d_z))
        self.logvar1 = nn.Parameter(torch.zeros(d_z))
    def log_prob(self, z, a):
        mu = torch.where(a.unsqueeze(-1) > 0.5, self.mu1, self.mu0)
        logvar = torch.where(a.unsqueeze(-1) > 0.5, self.logvar1, self.logvar0)
        return -0.5 * (((z - mu) ** 2) / torch.exp(logvar) + logvar).sum(dim=1)
    def club_upper(self, z, a):
        B = z.size(0)
        logp_matched = self.log_prob(z, a)
        idx = torch.randperm(B)
        a_shuf = a[idx]
        logp_mismatch = self.log_prob(z, a_shuf)
        return (logp_matched.mean() - logp_mismatch.mean())

class InfoNCE(nn.Module):
    def __init__(self, d_z, d_e=16):
        super().__init__()
        self.a_embed = nn.Embedding(2, d_e)
        nn.init.normal_(self.a_embed.weight, std=0.1)
        self.proj = nn.Sequential(nn.Linear(d_z, d_e), nn.ReLU(), nn.Linear(d_e, d_e))
    def forward(self, z, a):
        B = z.size(0)
        z_e = self.proj(z)
        a_e = self.a_embed(a.long()).view(B, -1)
        sim = z_e @ a_e.t()
        labels = torch.arange(B, device=z.device)
        loss = F.cross_entropy(sim, labels)
        mi_lower = math.log(B) - loss.item()
        return loss, mi_lower

def train_representation_model(
    X, A, Y, d_z=8, epochs=10, batch=256, lr=1e-3,
    lambda_grl=0.0, use_adv_on='z',
    club_weight=0.0, club_budget=None, info_nce_weight=0.0,
    seed=0, device='cpu'
):
    torch.manual_seed(seed)
    X = torch.tensor(X, dtype=torch.float32)
    A = torch.tensor(A, dtype=torch.float32).view(-1, 1)
    Y = torch.tensor(Y, dtype=torch.float32).view(-1, 1)

    enc = Encoder(X.shape[1], d_z).to(device)
    clf = ClassifierHead(d_z).to(device)
    adv_in_dim = d_z if use_adv_on == 'z' else 1
    adv = AdvAHead(adv_in_dim).to(device)

    club = CLUBGaussian(d_z).to(device) if club_weight > 0.0 else None
    info = InfoNCE(d_z).to(device) if info_nce_weight > 0.0 else None

    # Two optimizers: (i) model (enc/clf/adv), (ii) CLUB estimator
    opt_model = torch.optim.Adam(
        list(enc.parameters()) + list(clf.parameters()) + list(adv.parameters())
        + (list(info.parameters()) if info is not None else []),
        lr=lr
    )
    opt_club = torch.optim.Adam(list(club.parameters()), lr=1e-2) if club is not None else None

    ds = TensorDataset(X, A, Y)
    dl = DataLoader(ds, batch_size=batch, shuffle=True)

    for epoch in range(epochs):
        for xb, ab, yb in dl:
            xb = xb.to(device); ab = ab.to(device); yb = yb.to(device)

            # ----- Forward through encoder/classifier
            z = enc(xb)
            s = clf(z)
            task_loss = F.binary_cross_entropy(s, yb)

            # ----- Adversary on Z or score (optional)
            adv_loss = torch.tensor(0.0, device=device)
            if lambda_grl > 0.0:
                h = z if use_adv_on == 'z' else s
                a_pred = adv(grad_reverse(h, lambd=lambda_grl))
                adv_loss = F.binary_cross_entropy(a_pred, ab)

            # ----- Fit CLUB estimator (update CLUB params only; STOP grad to z)
            if club is not None:
                opt_club.zero_grad()
                with torch.no_grad():
                    z_det = z.detach()
                    a_det = ab.detach()
                club_fit = - club.log_prob(z_det, a_det).mean()   # maximize log p(z|a)
                club_fit.backward()
                opt_club.step()

            # ----- Apply CLUB budget penalty to the encoder (FREEZE CLUB params)
            club_pen = torch.tensor(0.0, device=device)
            if club is not None and club_budget is not None:
                for p in club.parameters():
                    p.requires_grad_(False)
                club_val = club.club_upper(z, ab)                # gradient flows to z (encoder)
                club_pen = club_weight * F.relu(club_val - club_budget)
                for p in club.parameters():
                    p.requires_grad_(True)

            # ----- Optional InfoNCE regularizer
            info_loss = torch.tensor(0.0, device=device)
            if info is not None and info_nce_weight > 0.0:
                loss_nce, _ = info(z, ab.view(-1))
                info_loss = info_nce_weight * loss_nce

            # ----- Update encoder/classifier/adv (not CLUB)
            loss = task_loss + adv_loss + club_pen + info_loss
            opt_model.zero_grad()
            loss.backward()
            opt_model.step()

    return enc, clf, adv, club, info


def predict_with_model(enc, clf, X):
    X = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        z = enc(X); s = clf(z).view(-1).numpy()
    return s

def bootstrap_ci(values, n_boot=500, alpha=0.05, seed=0):
    rng = np.random.RandomState(seed)
    values = np.asarray(values, float); n = len(values)
    if n == 0: return (np.nan, np.nan)
    boots = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        boots.append(values[idx].mean())
    lo = np.percentile(boots, 100 * (alpha/2))
    hi = np.percentile(boots, 100 * (1 - alpha/2))
    return lo, hi
