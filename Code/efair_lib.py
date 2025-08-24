
import math
import numpy as np
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

def make_synthetic(n: int = 5000, d: int = 10, seed: int = 0, rho: float = 0.6, base_rate_gap: float = 0.2):
    rng = np.random.RandomState(seed)
    A = rng.binomial(1, 0.5, size=n).astype(np.int64)
    X = rng.randn(n, d).astype(np.float32)
    X[:, 0] += rho * (A * 2 - 1).astype(np.float32)
    w = rng.randn(d).astype(np.float32) / np.sqrt(d)
    score = X @ w + 0.2 * np.tanh(X[:, 1]) + 0.2 * (X[:, 2] ** 2 - 1.0)
    shift = base_rate_gap * (A - 0.5) * 2.0
    p = 1.0 / (1.0 + np.exp(-(score + shift)))
    Y = (rng.rand(n) < p).astype(np.int64)
    return X, A, Y

def demographic_parity(y_score, A) -> Tuple[float, float, float]:
    yhat = (np.asarray(y_score) > 0.5).astype(int)
    A = np.asarray(A).astype(int)
    p1 = yhat[A == 1].mean() if (A == 1).sum() > 0 else 0.0
    p0 = yhat[A == 0].mean() if (A == 0).sum() > 0 else 0.0
    return abs(p1 - p0), p1, p0

def equal_opportunity(y_score, y_true, A) -> Tuple[float, float, float]:
    yhat = (np.asarray(y_score) > 0.5).astype(int)
    y_true = np.asarray(y_true).astype(int)
    A = np.asarray(A).astype(int)
    pos = (y_true == 1)
    if pos.sum() == 0:
        return 0.0, 0.0, 0.0
    tpr1 = (yhat[(A == 1) & pos] == 1).mean() if ((A == 1) & pos).sum() > 0 else 0.0
    tpr0 = (yhat[(A == 0) & pos] == 1).mean() if ((A == 0) & pos).sum() > 0 else 0.0
    return abs(tpr1 - tpr0), tpr1, tpr0

def plug_in_mi_binary(A, B) -> float:
    A = np.asarray(A).astype(int); B = np.asarray(B).astype(int); n = len(A)
    if n == 0: return 0.0
    pA = np.bincount(A, minlength=2) / n
    pB = np.bincount(B, minlength=2) / n
    joint = np.zeros((2, 2), dtype=float)
    for a in (0, 1):
        for b in (0, 1):
            joint[a, b] = ((A == a) & (B == b)).sum() / n
    mi = 0.0
    for a in (0, 1):
        for b in (0, 1):
            if joint[a, b] > 0 and pA[a] > 0 and pB[b] > 0:
                mi += joint[a, b] * math.log((joint[a, b] / (pA[a] * pB[b] + 1e-12)) + 1e-12)
    return float(mi)

def dp_upper_from_mi(mi: float, A) -> float:
    A = np.asarray(A).astype(int)
    p1 = A.mean(); p0 = 1.0 - p1
    return math.sqrt(max(mi, 0.0) / (2.0 * p0 * p1 + 1e-12))

def eo_upper_from_mi(mi_cond: float, A_pos) -> float:
    A_pos = np.asarray(A_pos).astype(int)
    p1 = A_pos.mean(); p0 = 1.0 - p1
    return math.sqrt(max(mi_cond, 0.0) / (2.0 * p0 * p1 + 1e-12))

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None

def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)

class Encoder(nn.Module):
    def __init__(self, d_in, d_z=8):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 64), nn.ReLU(), nn.Linear(64, d_z))
    def forward(self, x): return self.net(x)

class ClassifierHead(nn.Module):
    def __init__(self, d_z):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_z, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, z): return torch.sigmoid(self.net(z))

class AdvAHead(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, h): return torch.sigmoid(self.net(h))

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
        lp_m = self.log_prob(z, a)
        idx = torch.randperm(B, device=z.device)
        a_shuf = a[idx]
        lp_mm = self.log_prob(z, a_shuf)
        return (lp_m.mean() - lp_mm.mean())

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
        loss = torch.nn.functional.cross_entropy(sim, labels)
        mi_lower = math.log(B) - loss.item()
        return loss, mi_lower

def train_representation_model(
    X, A, Y, d_z=8, epochs=10, batch=256, lr=1e-3,
    lambda_grl=0.0, use_adv_on='z',
    club_weight=0.0, club_budget=None,
    seed=0, device='cpu',
    club_fit_lr=1e-2, club_warmup_steps=1
):
    torch.manual_seed(seed)
    X_t = torch.tensor(X, dtype=torch.float32)
    A_t = torch.tensor(A, dtype=torch.float32).view(-1, 1)
    Y_t = torch.tensor(Y, dtype=torch.float32).view(-1, 1)

    enc = Encoder(X_t.shape[1], d_z).to(device)
    clf = ClassifierHead(d_z).to(device)
    adv = AdvAHead(d_z if use_adv_on == 'z' else 1).to(device)

    need_club = (club_weight > 0.0) or (club_budget is not None)
    club = CLUBGaussian(d_z).to(device) if need_club else None

    opt_model = torch.optim.Adam(
        list(enc.parameters()) + list(clf.parameters()) + list(adv.parameters()),
        lr=lr
    )
    opt_club = torch.optim.Adam(list(club.parameters()), lr=club_fit_lr) if club else None

    dl = DataLoader(TensorDataset(X_t, A_t, Y_t), batch_size=batch, shuffle=True)

    for _ in range(epochs):
        for xb, ab, yb in dl:
            xb, ab, yb = xb.to(device), ab.to(device), yb.to(device)

            z = enc(xb)
            s = clf(z)
            task_loss = torch.nn.functional.binary_cross_entropy(s, yb)

            adv_loss = torch.tensor(0.0, device=device)
            if lambda_grl > 0.0:
                h = z if use_adv_on == 'z' else s
                a_pred = adv(GradReverse.apply(h, lambda_grl))
                adv_loss = torch.nn.functional.binary_cross_entropy(a_pred, ab)

            if club is not None:
                for _ in range(club_warmup_steps):
                    opt_club.zero_grad()
                    with torch.no_grad():
                        z_det, a_det = z.detach(), ab.detach()
                    club_fit = - club.log_prob(z_det, a_det).mean()
                    club_fit.backward()
                    opt_club.step()

            club_pen = torch.tensor(0.0, device=device)
            if club is not None and club_budget is not None:
                for p in club.parameters():
                    p.requires_grad_(False)
                club_val = club.club_upper(z, ab)
                club_pen = club_weight * torch.relu(club_val - club_budget)
                for p in club.parameters():
                    p.requires_grad_(True)

            loss = task_loss + adv_loss + club_pen
            opt_model.zero_grad()
            loss.backward()
            opt_model.step()

    return enc, clf, adv, club, None

def predict_with_model(enc, clf, X):
    X = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        z = enc(X)
        s = clf(z).view(-1).cpu().numpy()
    return s
