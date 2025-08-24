# ϵ-Fair: Unifying Algorithmic Fairness via Information Budgets

**ϵ-Fair** is an information-theoretic framework that turns fairness desiderata into **explicit budgets** on statistical dependence (mutual information). Those budgets are simple to **design with**, **monitor**, and **audit**, and they translate *mechanically* into certified bounds on common fairness metrics such as **Demographic Parity (DP)**, **Equalized Odds (EO)**, and **Calibration (CAL)**.

- **Applied paper (companion to this repo):**  
  *ϵ-Fair: Unifying Algorithmic Fairness via Information Budgets*  
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5403882

- **Theory paper (derivations & bounds):**  
  *Theoretical Foundations of Algorithmic Fairness: A Unified Framework Through Pinsker’s Inequality*  
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5402916

---

## Why ϵ-Fair?

- **One language for many notions.** DP/EO/CAL are unified by bounding the mutual information (MI) between the protected attribute $$A$$ and the model’s **prediction** $$\hat Y$$, **score** $$S$$, or **representation** $$Z$$.
- **Budgets → targets.** If you enforce $$I(A;\hat Y)\le \epsilon$$, you immediately get certified bounds (with prior-aware constants), e.g.
$$
\begin{align}
\mathrm{DP} &\le \sqrt{\frac{I(A;\hat{Y})}{2\pi_0\pi_1}} \\
\mathrm{EO} &\le \sqrt{\frac{I(A;\hat{Y} \mid Y = 1)}{2\pi_0^+\pi_1^+}}
\end{align}
$$

- **Model-agnostic and practitioner-ready.** Works with logistic regression, GBDT, calibrated scores, neural nets—anything that produces a score or decision.

---

## What’s in this repository?

- **`run_demo.py`** – Reproduces all figures/CSVs:
  - Utility–Fairness **Pareto** frontier (observed & **certified** DP)
  - **Representation budget** propagation $$I(A;Z)\ \rightarrow$$ DP/EO
  - **Drift monitoring** with MI time-series + bootstrap CIs
  - **Estimator study** (InfoNCE lower / plug-in / CLUB upper)
- **`outputs/`** – Auto-generated PNGs/CSVs (created when you run the script).
- **`requirements.txt`** – Minimal dependencies.

---

## Quick start

### Setup
```bash
git clone https://github.com/asudjianto-xml/e-Fair
cd e-Fair

python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# If PyTorch isn't installed:
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Reproduce the figures
```bash
python run_demo.py
```
This writes to `outputs/`:

- `pareto_frontier_auc_vs_dp.png`
- `pareto_frontier_auc_vs_certified_dp.png`
- `rep_budget_dp_eo.png`
- `drift_monitoring.png`
- `estimator_scatter.png`

### Run individual experiments (Jupyter/Python)
```python
import importlib, run_demo
importlib.reload(run_demo)

run_demo.experiment_pareto()
run_demo.experiment_rep_budget_two_phase()
run_demo.experiment_drift()
run_demo.experiment_estimators()
```

---

## The core idea (2 minutes)

- **Controlled dependence.** Fairness violations are upper-bounded by how much information about $$A$$ the system leaks through $$\hat Y$$ or \(S\).
- **Budgets in MI, targets in metrics.** Pick $$\epsilon$$ in MI-space; convert to **certified DP/EO/CAL** bounds via sharp, prior-aware inequalities.
- **Representation budgets propagate.** If $$\hat Y=g(Z)$$, then $$I(A;\hat Y)\le I(A;Z)$$. A budget on $$I(A;Z)$$ is a budget on prediction-level gaps.

---

## What the demos show
 
   Sweep a fairness knob; plot AUC vs observed DP and **AUC vs certified DP upper**. The certified curve **upper-envelopes** the observed—governance-ready.

2. **Representation budget propagation**  
   Tighten $$I(A;Z)$$ (CLUB upper) → **monotone** decreases in DP/EO; AUC stays steady; certified bounds track and upper-envelope observed gaps.

3. **Bias drift monitoring**  
   Track $$\widehat I_t(A;\hat Y)$$ with **bootstrap CIs**; convert to a certified DP bound; alert when the CI upper **exceeds budget**.

4. **Estimator study (diagnostics)**  
   InfoNCE (lower) / CLUB (upper) bracket $$I(A;Z)$$; plug-in MI on $$(A,\hat Y)$$ is exact for 2×2 tables. Certification uses **uppers or exact** estimators.

---

## Templates for practitioners 

### Logistic Regression (LR)
- **Granularity control:** supervised binning caps $$H(S)$$ and leakage capacity.  
- **Budgeted thresholding:** on validation, scan thresholds and keep only those with  
  $$\widehat I(A;\hat Y_t)\le \epsilon_{\mathrm{DP}}$$ (and $$\widehat I(A;\hat Y_t\mid Y{=}1)\le \epsilon_{\mathrm{EO}}$$); choose the best utility within the **feasible** set.  
- **Certify & monitor:** plug-in MI on test + bootstrap CIs → **certified** DP/EO bounds.

### Gradient-Boosted Decision Trees (GBDT)
- **Granularity control:** depth, min leaf, and rounds cap partition complexity and leakage capacity.  
- **Budget-aware early stopping:** after each round, run the same budgeted threshold scan; keep the best **feasible** round.  
- **Certify & monitor:** as above.

### Example using Neural Networks
> In all cases, “budget selection” is just choosing $$\epsilon$$ from a policy target via  
> $$\epsilon_{\mathrm{DP}}=2\,\pi_0\pi_1\,\tau_{\mathrm{DP}}^2$$ (and analogously for EO).

---

## Troubleshooting

- **`ModuleNotFoundError: torch`** → install CPU build (see setup).  
- **AttributeError: missing experiment** → available functions: `experiment_pareto`, `experiment_rep_budget_two_phase`, `experiment_drift`, `experiment_estimators`.  
- **CLUB/InfoNCE look inverted** → short training can narrow/invert the bracket; extend CLUB training. Certification still uses **upper** (CLUB) and **exact** plug-in.

---

## Papers

- **Applied (companion to this repo):**  
  Sudjianto, A. (2025). *ϵ-Fair: Unifying Algorithmic Fairness via Information Budgets*. SSRN.  
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5403882

- **Theory (derivations & bounds):**  
  Sudjianto, A. (2025). *Theoretical Foundations of Algorithmic Fairness: A Unified Framework Through Pinsker’s Inequality*. SSRN.  
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5402916

---

## How to cite

If you use the **code or applied workflow**, please cite the **Applied** paper (and optionally the **Theory** paper when referencing derivations/bounds).

**BibTeX**
```bibtex
@misc{Sudjianto2025EpsilonFairApplied,
  title        = {ϵ-Fair: Unifying Algorithmic Fairness via Information Budgets},
  author       = {Agus Sudjianto},
  year         = {2025},
  howpublished = {\url{https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5403882}},
  note         = {Companion code: \url{https://github.com/asudjianto-xml/e-Fair}}
}

@misc{Sudjianto2025FairnessTheoryPinsker,
  title        = {Theoretical Foundations of Algorithmic Fairness: A Unified Framework Through Pinsker's Inequality},
  author       = {Agus Sudjianto},
  year         = {2025},
  howpublished = {\url{https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5402916}}
}

@misc{Sudjianto2025EfairCode,
  title        = {e-Fair: Information-Budget Fairness Framework (code)},
  author       = {Agus Sudjianto},
  year         = {2025},
  howpublished = {\url{https://github.com/asudjianto-xml/e-Fair}}
}
```

**Plaintext**
> Sudjianto, A. (2025). *ϵ-Fair: Unifying Algorithmic Fairness via Information Budgets*. SSRN.  
> Sudjianto, A. (2025). *Theoretical Foundations of Algorithmic Fairness: A Unified Framework Through Pinsker’s Inequality*. SSRN.  
> Code: https://github.com/asudjianto-xml/e-Fair

---

## License

See `LICENSE` in this repository.

---

## Acknowledgments

If ϵ-Fair helps your work, please ⭐ the repo and cite the papers.
