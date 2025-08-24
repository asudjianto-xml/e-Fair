
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import roc_auc_score

import efair_lib as L

OUT = "outputs"

def experiment_rep_budget_two_phase(seed=7):
    """
    Two-phase representation-budget sweep that is guaranteed non-flat.
    Phase 1: pretrain encoder to *carry* A (auxiliary A-prediction head) -> leakage.
    Phase 2: for a range of MI budgets, apply a differentiable CLUB-like upper
             bound computed from batch statistics to *push leakage down*.
    Outputs: outputs/rep_budget_two_phase.csv and rep_budget_dp_eo.png
    """
    import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score

    # -----------------------------
    # 0) Data with stronger signal
    # -----------------------------
    X, A, Y = L.make_synthetic(
        n=6000, d=10, seed=seed,
        rho=1.2,            # stronger A→X shift
        base_rate_gap=0.35  # stronger A→Y base-rate gap
    )
    A = A.astype(int); Y = Y.astype(int)
    X_t = torch.tensor(X, dtype=torch.float32)
    A_t = torch.tensor(A, dtype=torch.float32).view(-1,1)
    Y_t = torch.tensor(Y, dtype=torch.float32).view(-1,1)

    # -----------------------------
    # 1) Small models (local)
    # -----------------------------
    class Enc(nn.Module):
        def __init__(self, d_in=10, d_z=4):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(d_in, 32), nn.ReLU(), nn.Linear(32, d_z))
        def forward(self, x): return self.net(x)

    class Clf(nn.Module):
        def __init__(self, d_z=4):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(d_z, 16), nn.ReLU(), nn.Linear(16, 1))
        def forward(self, z): return torch.sigmoid(self.net(z))

    class AHead(nn.Module):
        def __init__(self, d_z=4):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(d_z, 16), nn.ReLU(), nn.Linear(16, 1))
        def forward(self, z): return torch.sigmoid(self.net(z))

    # -------------------------------------------
    # 2) Differentiable batch-CLUB upper (MI upper)
    #    Uses per-group (mu, var) computed from z.
    #    Crucially: *no detach*, so gradients hit the encoder.
    # -------------------------------------------
    def batch_club_upper(z, a, eps=1e-6):
        a = a.view(-1,1)
        w1 = a
        w0 = 1 - a
        n1 = w1.sum() + eps
        n0 = w0.sum() + eps
        mu1 = (w1 * z).sum(0) / n1
        mu0 = (w0 * z).sum(0) / n0
        var1 = (w1 * (z - mu1)**2).sum(0) / n1 + eps
        var0 = (w0 * (z - mu0)**2).sum(0) / n0 + eps

        def log_prob(z, a01):
            mu  = a01 * mu1 + (1 - a01) * mu0
            var = a01 * var1 + (1 - a01) * var0
            return -0.5 * (((z - mu)**2 / var) + torch.log(var)).sum(dim=1)

        lp_m  = log_prob(z, a)                  # matched a
        a_sh  = a[torch.randperm(a.shape[0])]   # shuffled a'
        lp_mm = log_prob(z, a_sh)               # mismatched
        return (lp_m.mean() - lp_mm.mean())

    # ---------------------------------------------------
    # 3) Phase 1: pretrain encoder to *carry* A (leakage)
    # ---------------------------------------------------
    def pretrain_leaky_encoder(epochs=10, batch=512, lam_a=1.0):
        torch.manual_seed(seed)
        enc = Enc(d_in=X.shape[1], d_z=4)
        clf = Clf(d_z=4)
        ahead = AHead(d_z=4)
        opt = torch.optim.Adam(list(enc.parameters()) + list(clf.parameters()) + list(ahead.parameters()), lr=1e-3)

        dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_t, A_t, Y_t),
                                         batch_size=batch, shuffle=True)
        for _ in range(epochs):
            for xb, ab, yb in dl:
                z = enc(xb)
                s = clf(z)
                y_loss = F.binary_cross_entropy(s, yb)
                # POSITIVE A-prediction (no GRL): explicitly *encourage* leakage
                a_pred = ahead(z)
                a_loss = F.binary_cross_entropy(a_pred, ab)
                loss = y_loss + lam_a * a_loss
                opt.zero_grad(); loss.backward(); opt.step()

        return enc, clf

    # ---------------------------------------------------
    # 4) Phase 2: budget sweep with batch-CLUB penalty
    # ---------------------------------------------------
    def train_with_budget(enc_init, clf_init, budget, weight=25.0, epochs=25, batch=512):
        enc = Enc(d_in=X.shape[1], d_z=4); enc.load_state_dict(enc_init.state_dict())
        clf = Clf(d_z=4);                  clf.load_state_dict(clf_init.state_dict())
        opt = torch.optim.Adam(list(enc.parameters()) + list(clf.parameters()), lr=1e-3)
        dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_t, A_t, Y_t),
                                         batch_size=batch, shuffle=True)
        for _ in range(epochs):
            for xb, ab, yb in dl:
                z = enc(xb)
                s = clf(z)
                y_loss = F.binary_cross_entropy(s, yb)
                # CLUB-like upper evaluated on *non-detached* z
                club_u = batch_club_upper(z, ab)
                pen = weight * torch.relu(club_u - budget)
                loss = y_loss + pen
                opt.zero_grad(); loss.backward(); opt.step()

        # Eval
        with torch.no_grad():
            Z = enc(X_t)
            S = clf(Z).view(-1).cpu().numpy()
            yhat = (S > 0.5).astype(int)
            auc = roc_auc_score(Y, S)
            dp, _, _ = L.demographic_parity(S, A)
            eo, _, _ = L.equal_opportunity(S, Y, A)
            mi_yhat = L.plug_in_mi_binary(A, yhat)
            dp_cert = L.dp_upper_from_mi(mi_yhat, A)
            if (Y == 1).sum() > 0:
                mi_eo = L.plug_in_mi_binary(A[Y == 1], yhat[Y == 1])
                eo_cert = L.eo_upper_from_mi(mi_eo, A[Y == 1])
            else:
                eo_cert = 0.0
            # Also report batch-CLUB upper on the final Z
            club_all = batch_club_upper(Z, A_t).item()
        return dict(auc=auc, dp=dp, dp_cert=dp_cert, eo=eo, eo_cert=eo_cert, club_u=club_all)

    # Run Phase 1
    enc0, clf0 = pretrain_leaky_encoder(epochs=12, lam_a=1.0)

    # Measure baseline leakage u0 with the same differentiable estimator
    with torch.no_grad():
        Z0 = enc0(X_t)
        u0 = batch_club_upper(Z0, A_t).item()
    print(f"Baseline batch-CLUB upper u0 = {u0:.5f}")

    # Calibrated budgets as fractions of u0 (some bind, some don’t)
    fracs = [0.9, 0.7, 0.5, 0.3, 0.1]
    budgets = [max(u0 * f, 1e-4) for f in fracs]
    print("Budgets:", [f"{b:.5f}" for b in budgets])

    rows = []
    for f, b in zip(fracs, budgets):
        out = train_with_budget(enc0, clf0, budget=b, weight=30.0, epochs=30)
        rows.append(dict(frac=f, budget=b, AUC=out["auc"],
                         DP=out["dp"], DP_cert=out["dp_cert"],
                         EO=out["eo"], EO_cert=out["eo_cert"],
                         CLUB_upper=out["club_u"]))

    df = pd.DataFrame(rows).sort_values("frac", ascending=False)
    df.to_csv(f"{OUT}/rep_budget_two_phase.csv", index=False)

    # Plot
    x = np.arange(len(df))
    plt.figure()
    plt.plot(x, df["DP"], marker="o", label="Observed DP")
    plt.plot(x, df["DP_cert"], marker="o", label="Certified DP upper")
    plt.plot(x, df["EO"], marker="o", label="Observed EO")
    plt.plot(x, df["EO_cert"], marker="o", label="Certified EO upper")
    plt.xticks(x, [f"{f:.1f}·u0" for f in df["frac"]], rotation=30)
    plt.xlabel("Budget on I(A;Z) (fraction of baseline batch-CLUB upper u0)")
    plt.ylabel("Gap")
    plt.title("Representation budget → DP/EO (calibrated & binding)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT}/rep_budget_dp_eo.png", dpi=160)

    return df

def experiment_rep_budget_calibrated(seed=42):
    """
    Self-contained, calibrated representation-budget sweep.
    Trains a local encoder+classifier and a CLUB estimator inline,
    so the CLUB budget *definitely* pushes the encoder.
    """
    import torch, torch.nn as nn, torch.nn.functional as F
    import numpy as np, pandas as pd, matplotlib.pyplot as plt

    rng = np.random.RandomState(seed)
    X, A, Y = L.make_synthetic(
        n=6000, d=10, seed=seed,
        rho=1.2,           # stronger A→X shift (forces leakage path)
        base_rate_gap=0.35 # stronger A→Y gap
    )
    A = A.astype(int); Y = Y.astype(int)
    X_t = torch.tensor(X, dtype=torch.float32)
    A_t = torch.tensor(A, dtype=torch.float32).view(-1,1)
    Y_t = torch.tensor(Y, dtype=torch.float32).view(-1,1)

    # --- Tiny models (local, independent of efair_lib trainer)
    class Enc(nn.Module):
        def __init__(self, d_in=10, d_z=4):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(d_in, 32), nn.ReLU(), nn.Linear(32, d_z))
        def forward(self, x): return self.net(x)

    class Clf(nn.Module):
        def __init__(self, d_z=4):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(d_z, 16), nn.ReLU(), nn.Linear(16, 1))
        def forward(self, z): return torch.sigmoid(self.net(z))

    # Use efair_lib’s CLUB module for the estimator
    def fit_club(club, Z, A, steps=600, lr=1e-2, batch=512):
        opt = torch.optim.Adam(club.parameters(), lr=lr)
        ds = torch.utils.data.TensorDataset(Z, A)
        dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True)
        club.train()
        for _ in range(steps):
            for zb, ab in dl:
                opt.zero_grad()
                loss = -club.log_prob(zb, ab).mean()
                loss.backward(); opt.step()
        club.eval()

    # Local trainer that *fits CLUB and applies the CLUB penalty on z*
    def train_once(club_budget=None, club_weight=10.0, epochs=25, batch=512, seed=seed):
        torch.manual_seed(seed)
        enc = Enc(d_in=X.shape[1], d_z=4)
        clf = Clf(d_z=4)
        club = L.CLUBGaussian(d_z=4)

        opt_model = torch.optim.Adam(list(enc.parameters())+list(clf.parameters()), lr=1e-3)
        opt_club  = torch.optim.Adam(club.parameters(), lr=1e-2)

        dl = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_t, A_t, Y_t),
            batch_size=batch, shuffle=True
        )

        for _ in range(epochs):
            for xb, ab, yb in dl:
                # forward
                z = enc(xb)
                s = clf(z)
                task = F.binary_cross_entropy(s, yb)

                # --- fit CLUB on *detached* z (updates only CLUB)
                opt_club.zero_grad()
                with torch.no_grad():
                    z_det, a_det = z.detach(), ab.detach()
                club_fit = -club.log_prob(z_det, a_det).mean()
                club_fit.backward(); opt_club.step()

                # --- apply CLUB budget on *non-detached* z (freeze CLUB during penalty)
                club_pen = torch.tensor(0.0)
                if club_budget is not None:
                    for p in club.parameters(): p.requires_grad_(False)
                    club_val = club.club_upper(z, ab)
                    club_pen = club_weight * torch.relu(club_val - club_budget)
                    for p in club.parameters(): p.requires_grad_(True)

                loss = task + club_pen
                opt_model.zero_grad()
                loss.backward(); opt_model.step()

        # evaluate on full data
        with torch.no_grad():
            Z = enc(X_t)
            S = clf(Z).view(-1).cpu().numpy()
            yhat = (S > 0.5).astype(int)
            auc = roc_auc_score(Y, S)
            dp, _, _ = L.demographic_parity(S, A)
            eo, _, _ = L.equal_opportunity(S, Y, A)
            mi_yhat = L.plug_in_mi_binary(A, yhat)
            dp_cert = L.dp_upper_from_mi(mi_yhat, A)
            if (Y == 1).sum() > 0:
                mi_eo = L.plug_in_mi_binary(A[Y == 1], yhat[Y == 1])
                eo_cert = L.eo_upper_from_mi(mi_eo, A[Y == 1])
            else:
                eo_cert = 0.0
            # compute CLUB upper on final Z
            club_eval = L.CLUBGaussian(d_z=Z.shape[1])
            fit_club(club_eval, Z, A_t, steps=600, lr=1e-2, batch=512)
            club_u = float(torch.relu(club_eval.club_upper(Z, A_t)).item())
        return dict(auc=auc, dp=dp, dp_cert=dp_cert, eo=eo, eo_cert=eo_cert, club_u=club_u)

    # 1) Baseline (no budget) → get u0 on its Z
    base = train_once(club_budget=None, club_weight=0.0, epochs=12)
    u0 = base["club_u"]
    # guard in case it's tiny: strengthen signal until u0 > 0
    if u0 < 1e-4:
        # one-time stronger training to make baseline leakage show up
        base = train_once(club_budget=None, club_weight=0.0, epochs=30)
        u0 = base["club_u"]
    print(f"Baseline CLUB upper u0 = {u0:.6f}")

    # 2) Budgets as fractions of u0 (guarantees some bind)
    fracs = [0.9, 0.7, 0.5, 0.3, 0.1]
    budgets = [max(u0*f, 1e-4) for f in fracs]  # keep positive
    print("Budgets:", [f"{b:.6f}" for b in budgets])

    rows = []
    for f, b in zip(fracs, budgets):
        out = train_once(club_budget=b, club_weight=20.0, epochs=30)  # heavier penalty to ensure movement
        rows.append(dict(
            frac=f, budget=b,
            AUC=out["auc"],
            DP=out["dp"], DP_cert=out["dp_cert"],
            EO=out["eo"], EO_cert=out["eo_cert"],
            CLUB_upper=out["club_u"]
        ))

    df = pd.DataFrame(rows).sort_values("frac", ascending=False)
    df.to_csv(f"{OUT}/rep_budget_sweep.csv", index=False)

    # Plot
    x = np.arange(len(df))
    plt.figure()
    plt.plot(x, df["DP"], marker="o", label="Observed DP")
    plt.plot(x, df["DP_cert"], marker="o", label="Certified DP upper")
    plt.plot(x, df["EO"], marker="o", label="Observed EO")
    plt.plot(x, df["EO_cert"], marker="o", label="Certified EO upper")
    plt.xticks(x, [f"{f:.1f}·u0" for f in df["frac"]], rotation=30)
    plt.xlabel("Budget on I(A;Z) (fraction of baseline CLUB upper u0)")
    plt.ylabel("Gap")
    plt.title("Representation budget → DP/EO (calibrated & binding)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT}/rep_budget_dp_eo.png", dpi=160)

    return df


def experiment_pareto(seed=0):
    X, A, Y = L.make_synthetic(n=3000, d=10, seed=seed, rho=0.7, base_rate_gap=0.25)
    lambdas = [0.0, 0.2, 0.5, 0.8]
    rows = []
    for lam in lambdas:
        enc, clf, *_ = L.train_representation_model(
            X, A, Y, d_z=8, epochs=5, batch=512, lr=1e-3,
            lambda_grl=lam, use_adv_on='s', seed=seed
        )
        s = L.predict_with_model(enc, clf, X)
        yhat = (s > 0.5).astype(int)
        auc = roc_auc_score(Y, s)
        dp, _, _ = L.demographic_parity(s, A)
        eo, _, _ = L.equal_opportunity(s, Y, A)
        mi_dp = L.plug_in_mi_binary(A, yhat)
        dp_cert = L.dp_upper_from_mi(mi_dp, A)
        if (Y == 1).sum() > 0:
            mi_eo = L.plug_in_mi_binary(A[Y == 1], yhat[Y == 1])
            eo_cert = L.eo_upper_from_mi(mi_eo, A[Y == 1])
        else:
            eo_cert = 0.0
        rows.append(dict(lambda_grl=lam, AUC=auc, DP=dp, DP_cert=dp_cert, EO=eo, EO_cert=eo_cert))
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/pareto_points.csv", index=False)
    plt.figure(); plt.scatter(df["DP"], df["AUC"]); plt.xlabel("Observed DP"); plt.ylabel("AUC")
    plt.title("AUC vs Observed DP"); plt.tight_layout(); plt.savefig(f"{OUT}/pareto_frontier_auc_vs_dp.png", dpi=140)
    plt.figure(); plt.scatter(df["DP_cert"], df["AUC"]); plt.xlabel("Certified DP upper"); plt.ylabel("AUC")
    plt.title("AUC vs Certified DP upper"); plt.tight_layout(); plt.savefig(f"{OUT}/pareto_frontier_auc_vs_certified_dp.png", dpi=140)

def _fit_club_on_model(enc, X, A, steps=600, lr=1e-2, batch=512, seed=0):
    X_t = torch.tensor(X, dtype=torch.float32)
    A_t = torch.tensor(A, dtype=torch.float32).view(-1,1)
    with torch.no_grad():
        Z = enc(X_t)
    club = L.CLUBGaussian(d_z=Z.shape[1])
    opt = torch.optim.Adam(club.parameters(), lr=lr)
    ds = torch.utils.data.TensorDataset(Z, A_t)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True)
    for _ in range(steps):
        for zb, ab in dl:
            opt.zero_grad()
            loss = - club.log_prob(zb, ab).mean()
            loss.backward(); opt.step()
    with torch.no_grad():
        u0 = torch.relu(club.club_upper(Z, A_t)).item()
    return club, u0

def experiment_rep_budget_calibrated(seed=42):
    X, A, Y = L.make_synthetic(n=5000, d=10, seed=seed, rho=1.0, base_rate_gap=0.30)

    enc0, clf0, *_ = L.train_representation_model(
        X, A, Y, d_z=8, epochs=10, batch=512, lr=1e-3,
        lambda_grl=0.0, use_adv_on='z',
        club_weight=0.0, club_budget=None, seed=seed
    )
    club0, u0 = _fit_club_on_model(enc0, X, A, steps=600, lr=1e-2, batch=512, seed=seed)
    fracs = [0.9, 0.7, 0.5, 0.3, 0.1]
    budgets = [max(u0 * f, 1e-4) for f in fracs]
    print("Baseline CLUB upper u0 =", u0)

    rows = []
    for b, f in zip(budgets, fracs):
        enc, clf, *_ = L.train_representation_model(
            X, A, Y,
            d_z=8, epochs=25, batch=512, lr=1e-3,
            lambda_grl=0.0, use_adv_on='z',
            club_weight=10.0, club_budget=b,
            seed=seed, club_warmup_steps=2, club_fit_lr=1e-2
        )
        s = L.predict_with_model(enc, clf, X)
        yhat = (s > 0.5).astype(int)
        auc = roc_auc_score(Y, s)
        dp, _, _ = L.demographic_parity(s, A)
        eo, _, _ = L.equal_opportunity(s, Y, A)
        mi_yhat = L.plug_in_mi_binary(A, yhat)
        dp_cert = L.dp_upper_from_mi(mi_yhat, A)
        if (Y == 1).sum() > 0:
            mi_eo = L.plug_in_mi_binary(A[Y == 1], yhat[Y == 1])
            eo_cert = L.eo_upper_from_mi(mi_eo, A[Y == 1])
        else:
            eo_cert = 0.0

        X_t = torch.tensor(X, dtype=torch.float32)
        A_t = torch.tensor(A, dtype=torch.float32).view(-1,1)
        with torch.no_grad():
            Z_t = enc(X_t)
            club_val = torch.relu(club0.club_upper(Z_t, A_t)).item()

        rows.append(dict(budget=b, frac=f, AUC=auc, DP=dp, DP_cert=dp_cert,
                         EO=eo, EO_cert=eo_cert, CLUB_upper=club_val))

    df = pd.DataFrame(rows).sort_values("frac", ascending=False)
    df.to_csv(f"{OUT}/rep_budget_sweep.csv", index=False)

    x = np.arange(len(df))
    plt.figure()
    plt.plot(x, df["DP"], marker="o", label="Observed DP")
    plt.plot(x, df["DP_cert"], marker="o", label="Certified DP upper")
    plt.plot(x, df["EO"], marker="o", label="Observed EO")
    plt.plot(x, df["EO_cert"], marker="o", label="Certified EO upper")
    plt.xticks(x, [f"{f:.1f}·u0" for f in df["frac"]], rotation=30)
    plt.xlabel("Budget on I(A;Z) (fraction of baseline CLUB upper u0)")
    plt.ylabel("Gap")
    plt.title("Representation budget → DP/EO (calibrated & binding)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT}/rep_budget_dp_eo.png", dpi=160)

def experiment_drift(seed=2):
    X0, A0, Y0 = L.make_synthetic(n=2500, d=10, seed=seed, rho=0.5, base_rate_gap=0.2)
    enc, clf, *_ = L.train_representation_model(X0, A0, Y0, d_z=8, epochs=6, batch=512, lr=1e-3, seed=seed)
    rows = []; T = 8
    for t in range(T):
        rho_t = 0.5 + 0.06 * t
        X, A, Y = L.make_synthetic(n=2500, d=10, seed=seed+10*t, rho=rho_t, base_rate_gap=0.2)
        s = L.predict_with_model(enc, clf, X); yhat = (s > 0.5).astype(int)
        dp, _, _ = L.demographic_parity(s, A)
        mi = L.plug_in_mi_binary(A, yhat); dp_cert = L.dp_upper_from_mi(mi, A)
        rng = np.random.RandomState(seed + 99*t); n = len(A); bs = []
        for _ in range(150):
            idx = rng.randint(0, n, size=n); bs.append(L.plug_in_mi_binary(A[idx], yhat[idx]))
        lo = np.percentile(bs, 2.5); hi = np.percentile(bs, 97.5)
        policy_budget = 0.03; alert_budget = int(hi > policy_budget)
        rows.append(dict(t=t, rho=rho_t, MI=mi, MI_lo=lo, MI_hi=hi, DP=dp, DP_cert=dp_cert, alert_budget=alert_budget))
    df = pd.DataFrame(rows); df.to_csv(f"{OUT}/drift_timeseries.csv", index=False)
    fig, ax1 = plt.subplots()
    ax1.plot(df["t"], df["MI"], marker="o", label="Î(A; Ŷ)")
    ax1.fill_between(df["t"], df["MI_lo"], df["MI_hi"], alpha=0.2, label="95% CI")
    ax1.set_xlabel("time"); ax1.set_ylabel("MI & CI")
    ax2 = ax1.twinx()
    ax2.plot(df["t"], df["DP"], marker="s", label="Observed DP")
    ax2.plot(df["t"], df["DP_cert"], marker="^", label="Certified DP upper")
    lines, labels = ax1.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines+lines2, labels+labels2, loc="upper left"); plt.title("Drift monitoring")
    plt.tight_layout(); plt.savefig(f"{OUT}/drift_monitoring.png", dpi=140)

def experiment_estimators(seed=3, epochs_club=200, epochs_infonce=200, batch=512):
    X, A, Y = L.make_synthetic(n=5000, d=10, seed=seed, rho=0.7, base_rate_gap=0.2)
    enc, clf, *_ = L.train_representation_model(
        X, A, Y, d_z=8, epochs=8, batch=batch, lr=1e-3,
        lambda_grl=0.0, use_adv_on='z', seed=seed
    )
    s = L.predict_with_model(enc, clf, X)
    yhat = (s > 0.5).astype(int)
    mi_plugin = L.plug_in_mi_binary(A, yhat)

    X_t = torch.tensor(X, dtype=torch.float32)
    A_t = torch.tensor(A, dtype=torch.float32).view(-1,1)
    with torch.no_grad():
        Z_t = enc(X_t)

    club = L.CLUBGaussian(d_z=Z_t.shape[1])
    opt_c = torch.optim.Adam(club.parameters(), lr=1e-2)
    ds = torch.utils.data.TensorDataset(Z_t, A_t)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True)
    for _ in range(epochs_club):
        for zb, ab in dl:
            opt_c.zero_grad()
            loss = - club.log_prob(zb, ab).mean()
            loss.backward(); opt_c.step()
    with torch.no_grad():
        club_upper = club.club_upper(Z_t, A_t).item()

    info = L.InfoNCE(d_z=Z_t.shape[1], d_e=16)
    opt_i = torch.optim.Adam(info.parameters(), lr=5e-3)
    for _ in range(epochs_infonce):
        for zb, ab in dl:
            opt_i.zero_grad()
            loss, _ = info(zb, ab.view(-1))
            loss.backward(); opt_i.step()
    with torch.no_grad():
        _, mi_lower = info(Z_t, A_t.view(-1))

    xs = [0, 1, 2]
    vals = [mi_lower, mi_plugin, club_upper]
    labels = ["InfoNCE lower (A;Z)", "Plug-in MI (A; Ŷ)", "CLUB upper (A;Z)"]
    df = pd.DataFrame({"estimator": labels, "value": vals})
    df.to_csv(f"{OUT}/estimator_values.csv", index=False)

    plt.figure()
    plt.bar(xs, vals)
    plt.xticks(xs, labels, rotation=20)
    plt.ylabel("nats")
    plt.title("Estimator study: lower vs plug-in vs upper")
    for i, v in enumerate(vals):
        plt.annotate(f"{v:.3f}", (xs[i], vals[i]), xytext=(0, 5), textcoords="offset points", ha="center")
    plt.tight_layout()
    plt.savefig(f"{OUT}/estimator_scatter.png", dpi=160)

if __name__ == "__main__":
    experiment_pareto()
    experiment_rep_budget_calibrated()
    experiment_drift()
    experiment_estimators()
    print("Done. See outputs/")
