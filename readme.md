# e-Fair: An ε-Budget Information-Theoretic Fairness Framework

**e-Fair** is a practical and rigorous framework for algorithmic fairness that treats fairness as an **explicit information budget**.  
Instead of relying on ad-hoc penalties, e-Fair constrains the mutual information leakage between protected attributes **A** and predictions **Ŷ**:

\[
I(A;\hat Y) \;\leq\; \epsilon
\]

This gives practitioners a clear **budget ε** to spend on fairness, while still optimizing utility.

---

## 🔑 Key Ideas

- **ε-Budget Fairness**  
  Control fairness explicitly via information-theoretic constraints (mutual information).
  
- **Theoretical Backbone**  
  Based on Pinsker-type inequalities, impossibility theorems, and ε-constraint optimization.

- **Practical Realization**  
  - **Binning & Monotonicity**: Stabilize predictions, enforce regulatory constraints.  
  - **Exact Optimization**: Dynamic programming (DP) or Mixed-Integer Programming (MIP) solves ε-constraint problems exactly.  
  - **Fairness–Utility Frontier**: Sweep over ε to visualize trade-offs.  

- **General & Extensible**  
  Works for categorical, continuous scores, multi-task settings, and temporal monitoring.

---

## 📚 What’s Included

- **LaTeX Tutorial**:  
  A self-contained supplemental document:  
  `supplemental_practical_tutorial.tex`  
  with full proofs, references, and practical guidelines.

- **Auditing Toolbox**:  
  `fairness_audit.py`  
  - Demographic Parity (DP)  
  - Equal Opportunity (EOp)  
  - Mutual Information (binary + discretized scores)  
  - Bootstrap CIs  
  - Permutation tests

- **Notebooks** (ready to run):  
  - `14_Categorical_Fairness_Diagnostics.ipynb` — DP, EOp, MI, Pinsker bounds  
  - `15_Continuous_Scores_MI_Bootstrap_Permutation.ipynb` — Scores MI, bootstrap CI, permutation  
  - `16_MultiTask_MI_and_UniformDP.ipynb` — Multi-task MI vs joint MI, uniform DP bound  
  - `17_Temporal_Monitoring_BiasDrift.ipynb` — Drift detection over time  
  - `12_IP_Binning_LogReg_EpsilonMIP.ipynb` — Bin merging via ε-constraint, monotone logistic  
  - `13_LDA_XGB1_EpsilonConstraint_BOP.ipynb` — IV-based ε-constraint program with LP export

- **LP Exporters with Monotonicity**:  
  `fairness_lp.py` for external MILP solvers (CBC, Gurobi, CPLEX).

---

## 🚀 Quick Start

1. Clone the repo and install dependencies:
   ```bash
   git clone https://github.com/yourusername/e-fair.git
   cd e-fair
   pip install -r requirements.txt
