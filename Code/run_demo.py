
# run_demo.py  — clean version (no escaped quotes)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score

from efair_lib import (
    make_synthetic, train_representation_model, predict_with_model,
    demographic_parity, equal_opportunity,
    plug_in_mi_binary, dp_upper_from_mi, eo_upper_from_mi,
    CLUBGaussian, InfoNCE
)
import torch

OUT = "outputs"

# -------------------------
# 1) Utility–Fairness Pareto
# -------------------------
def experiment_pareto(seed=0):
    X, A, Y = make_synthetic(n=3000, d=10, seed=seed, rho=0.7, base_rate_gap=0.25)
    lambdas = [0.0, 0.2, 0.5, 0.8]
    rows = []
    for lam in lambdas:
        enc, clf, *_ = train_representation_model(
            X, A, Y, d_z=8, epochs=5, batch=512, lr=1e-3,
            lambda_grl=lam, use_adv_on='s', seed=seed
        )
        s = predict_with_model(enc, clf, X)
        yhat = (s > 0.5).astype(int)
        auc = roc_auc_score(Y, s)
        dp, _, _ = demographic_parity(s, A)
        eo, _, _ = equal_opportunity(s, Y, A)
        mi_dp = plug_in_mi_binary(A, yhat)
        dp_cert = dp_upper_from_mi(mi_dp, A)
        mi_eo = plug_in_mi_binary(A[Y == 1], yhat[Y == 1]) if (Y == 1).sum() > 0 else 0.0
        eo_cert = eo_upper_from_mi(mi_eo, A[Y == 1]) if (Y == 1).sum() > 0 else 0.0
        rows.append(dict(lambda_grl=lam, AUC=auc, DP=dp, DP_cert=dp_cert, EO=eo, EO_cert=eo_cert))

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/pareto_points.csv", index=False)

    plt.figure()
    plt.scatter(df["DP"], df["AUC"])
    plt.xlabel("Observed DP")
    plt.ylabel("AUC")
    plt.title("AUC vs Observed DP")
    plt.tight_layout()
    plt.savefig(f"{OUT}/pareto_frontier_auc_vs_dp.png", dpi=140)

    plt.figure()
    plt.scatter(df["DP_cert"], df["AUC"])
    plt.xlabel("Certified DP upper")
    plt.ylabel("AUC")
    plt.title("AUC vs Certified DP upper")
    plt.tight_layout()
    plt.savefig(f"{OUT}/pareto_frontier_auc_vs_certified_dp.png", dpi=140)

# -----------------------------------
# 2) Representation budget propagation
# -----------------------------------
def experiment_rep_budget(seed=1):
    X, A, Y = make_synthetic(n=4000, d=10, seed=seed, rho=0.9, base_rate_gap=0.25)
    budgets = [0.03, 0.02, 0.01, 0.005, 0.002]
    rows = []
    for budget in budgets:
        enc, clf, adv, club, info = train_representation_model(
            X, A, Y,
            d_z=8, epochs=25, batch=512, lr=1e-3,
            lambda_grl=0.0, use_adv_on='z',
            club_weight=10.0, club_budget=budget,
            seed=seed
        )
        s = predict_with_model(enc, clf, X)
        yhat = (s > 0.5).astype(int)
        auc = roc_auc_score(Y, s)
        dp, _, _ = demographic_parity(s, A)
        eo, _, _ = equal_opportunity(s, Y, A)

        # MI on (A, Ŷ) for certification
        mi_yhat = plug_in_mi_binary(A, yhat)
        dp_cert = dp_upper_from_mi(mi_yhat, A)
        mi_eo = plug_in_mi_binary(A[Y==1], yhat[Y==1]) if (Y==1).sum() > 0 else 0.0
        eo_cert = eo_upper_from_mi(mi_eo, A[Y==1]) if (Y==1).sum() > 0 else 0.0

        # Log CLUB upper on full data to see if the budget is binding
        import torch
        X_t = torch.tensor(X, dtype=torch.float32)
        A_t = torch.tensor(A, dtype=torch.float32).view(-1,1)
        with torch.no_grad():
            Z_t = enc(X_t)
            club_val = club.club_upper(Z_t, A_t).item()

        rows.append(dict(budget=budget, AUC=auc, DP=dp, DP_cert=dp_cert,
                         EO=eo, EO_cert=eo_cert, CLUB_upper=club_val))

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/rep_budget_sweep.csv", index=False)

    # Plot
    import matplotlib.pyplot as plt
    x = list(range(len(df)))
    plt.figure()
    plt.plot(x, df["DP"], marker="o", label="Observed DP")
    plt.plot(x, df["DP_cert"], marker="o", label="Certified DP upper")
    plt.plot(x, df["EO"], marker="o", label="Observed EO")
    plt.plot(x, df["EO_cert"], marker="o", label="Certified EO upper")
    plt.xticks(x, [str(b) for b in df["budget"]], rotation=30)
    plt.xlabel("CLUB budget on I(A;Z)")
    plt.ylabel("Gap")
    plt.title("Representation budget → DP/EO (binding budgets)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT}/rep_budget_dp_eo.png", dpi=160)


# ---------------------
# 3) Drift monitoring
# ---------------------
def experiment_drift(seed=2):
    X0, A0, Y0 = make_synthetic(n=2500, d=10, seed=seed, rho=0.5, base_rate_gap=0.2)
    enc, clf, *_ = train_representation_model(
        X0, A0, Y0, d_z=8, epochs=6, batch=512, lr=1e-3, seed=seed
    )
    rows = []
    T = 8
    for t in range(T):
        rho_t = 0.5 + 0.06 * t
        X, A, Y = make_synthetic(n=2500, d=10, seed=seed + 10 * t, rho=rho_t, base_rate_gap=0.2)
        s = predict_with_model(enc, clf, X)
        yhat = (s > 0.5).astype(int)
        dp, _, _ = demographic_parity(s, A)
        mi = plug_in_mi_binary(A, yhat)
        dp_cert = dp_upper_from_mi(mi, A)
        rng = np.random.RandomState(seed + 99 * t)
        n = len(A)
        bs = []
        for _ in range(150):
            idx = rng.randint(0, n, size=n)
            bs.append(plug_in_mi_binary(A[idx], yhat[idx]))
        lo = np.percentile(bs, 2.5)
        hi = np.percentile(bs, 97.5)
        policy_budget = 0.03
        alert_budget = int(hi > policy_budget)
        rows.append(dict(t=t, rho=rho_t, MI=mi, MI_lo=lo, MI_hi=hi,
                         DP=dp, DP_cert=dp_cert, alert_budget=alert_budget))

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/drift_timeseries.csv", index=False)

    fig, ax1 = plt.subplots()
    ax1.plot(df["t"], df["MI"], marker="o", label="Î(A; Ŷ)")
    ax1.fill_between(df["t"], df["MI_lo"], df["MI_hi"], alpha=0.2, label="95% CI")
    ax1.set_xlabel("time")
    ax1.set_ylabel("MI & CI")
    ax2 = ax1.twinx()
    ax2.plot(df["t"], df["DP"], marker="s", label="Observed DP")
    ax2.plot(df["t"], df["DP_cert"], marker="^", label="Certified DP upper")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper left")
    plt.title("Drift monitoring")
    plt.tight_layout()
    plt.savefig(f"{OUT}/drift_monitoring.png", dpi=140)

# ----------------------------------
# 4) Estimator comparison experiment
# ----------------------------------
def experiment_estimators(seed=3, epochs_club=200, epochs_infonce=200, batch=512):
    X, A, Y = make_synthetic(n=5000, d=10, seed=seed, rho=0.7, base_rate_gap=0.2)
    enc, clf, *_ = train_representation_model(
        X, A, Y, d_z=8, epochs=8, batch=batch, lr=1e-3,
        lambda_grl=0.0, use_adv_on='z', seed=seed
    )

    # Plug-in MI for (A, Ŷ)
    s = predict_with_model(enc, clf, X)
    yhat = (s > 0.5).astype(int)
    mi_plugin = plug_in_mi_binary(A, yhat)

    # Z
    X_t = torch.tensor(X, dtype=torch.float32)
    A_t = torch.tensor(A, dtype=torch.float32).view(-1, 1)
    with torch.no_grad():
        Z_t = enc(X_t)

    # CLUB (upper)
    club = CLUBGaussian(d_z=Z_t.shape[1])
    opt_c = torch.optim.Adam(club.parameters(), lr=1e-2)
    ds = torch.utils.data.TensorDataset(Z_t, A_t)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True)
    for _ in range(epochs_club):
        for zb, ab in dl:
            opt_c.zero_grad()
            loss = -club.log_prob(zb, ab).mean()
            loss.backward()
            opt_c.step()
    with torch.no_grad():
        club_upper = club.club_upper(Z_t, A_t).item()

    # InfoNCE (lower)
    info = InfoNCE(d_z=Z_t.shape[1], d_e=16)
    opt_i = torch.optim.Adam(info.parameters(), lr=5e-3)
    for _ in range(epochs_infonce):
        for zb, ab in dl:
            opt_i.zero_grad()
            loss, _ = info(zb, ab.view(-1))
            loss.backward()
            opt_i.step()
    with torch.no_grad():
        _, mi_lower = info(Z_t, A_t.view(-1))

    # Save & plot
    xs = [0, 1, 2]
    vals = [mi_lower, mi_plugin, club_upper]
    labels = ["InfoNCE lower (A;Z)", "Plug-in MI (A; Ŷ)", "CLUB upper (A;Z)"]
    pd.DataFrame({"estimator": labels, "value": vals}).to_csv(f"{OUT}/estimator_values.csv", index=False)

    plt.figure()
    plt.bar(xs, vals)
    plt.xticks(xs, labels, rotation=20)
    plt.ylabel("nats")
    plt.title("Estimator study: lower vs plug-in vs upper")
    for i, v in enumerate(vals):
        plt.annotate(f"{v:.3f}", (xs[i], vals[i]), xytext=(0, 5), textcoords="offset points", ha="center")
    plt.tight_layout()
    plt.savefig(f"{OUT}/estimator_scatter.png", dpi=160)

if __name__ == '__main__':
    experiment_pareto()
    experiment_rep_budget()
    experiment_drift()
    experiment_estimators()
    print("Done. See outputs/")
