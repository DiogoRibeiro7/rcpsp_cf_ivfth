# RCPSP-CF-IVFTH

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17382196.svg)](https://doi.org/10.5281/zenodo.17382196)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

Bi-objective **Resource-Constrained Project Scheduling with Cash-Flow** under **fuzzy uncertainty** (NIVTF), solved via an **extended IVF–TH** scalarization and MILP.

This repository implements the model from:

> _A New Bi-Objective Model for Resource-Constrained Project Scheduling and Cash Flow Problems with Financial Constraints under Uncertainty: A Case Study_ (multi-mode RCPSP, cash-flow constraints, delayed payments, interest on loans/excess cash, and interval-valued fuzzy numbers)

--------------------------------------------------------------------------------

## Features

- **Bi-objective MILP**: minimize **makespan** (Cmax) and maximize **final cash flow** (CF in the last period).
- **Multi-mode activities** with **renewable/non-renewable** resources.
- **Daily & periodic costs**, **initial capital**, **short-/long-term loans**, **credit limits**, **delayed payments** (≤ 1 period).
- **Uncertainty** in durations and resource demands via **Normalized Interval-Valued Triangular Fuzzy (NIVTF)** numbers.
- **Extended IVF–TH** scalarization (Torabi–Hassini) to convert the fuzzy bi-objective to a single-objective MILP.
- Written in pure Python with **Pyomo**. Works with GLPK/CBC/CPLEX/Gurobi.

--------------------------------------------------------------------------------

## Repository layout

```
rcpsp_cf_ivfth/
├── __init__.py           # Main package exports
├── fuzzy.py             # NIVTF fuzzy number definitions
├── data.py              # Data structures (Activity, FinanceParams, etc.)
├── model.py             # Main RCPSP_CF_IVFTH solver class
└── examples/
    ├── __init__.py
    └── toy_instance.py   # Example usage with toy data
```

--------------------------------------------------------------------------------

## Installation

### With Poetry (recommended)

```bash
poetry init --no-interaction
poetry add pyomo
# Add a MILP solver; choose one you have:
# GLPK (Linux): sudo apt-get install glpk-utils
# CBC (cross-platform via conda): conda install -c conda-forge coincbc
poetry run python -m rcpsp_cf_ivfth.examples.toy_instance
```

### With pip

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install pyomo
# Install a solver (e.g., cbc or glpk) and ensure it's on PATH
python -m rcpsp_cf_ivfth.examples.toy_instance
```

> **Solver note:** Set the solver in code: `ivfth.solve(model, solver_name="glpk")`. Alternatives: `"cbc"`, `"gurobi"`, `"cplex"` (if licensed/installed).

--------------------------------------------------------------------------------

## Quick start (toy instance)

Run the example with:

```bash
python -m rcpsp_cf_ivfth.examples.toy_instance
```

Expected console summary (varies by solver/params):

```
Solve summary:
  status: optimal
  objective: ...
  Cmax: ...
  CF_final: ...
  mu1: ...
  mu2: ...
  lambda: ...
```

--------------------------------------------------------------------------------

## How to use with your data

```python
from rcpsp_cf_ivfth import (
    RCPSP_CF_IVFTH, Activity, ModeData, FinanceParams, CalendarParams,
    IVFTHTargets, IVFTHWeights, NIVTF, create_triangle
)

# 1. Define your activities with modes, NIVTF durations, resource usage, and payments
activities = {
    "Start": Activity(
        name="Start", predecessors=[],
        modes={1: ModeData(
            duration=NIVTF(*create_triangle(0, 0, 0)),
            renewables={1: NIVTF(*create_triangle(0, 0, 0))},
            nonrenewables={1: NIVTF(*create_triangle(0, 0, 0))},
            payment=0.0
        )}
    ),
    # ... define your real activities
}

# 2. Define finance parameters
finance = FinanceParams(
    alpha_excess_cash=0.0125,   # Interest rates
    beta_delayed_pay=0.10,
    gamma_LTL=0.06,
    delta_STL=0.075,
    IC=10000.0,                 # Initial capital
    max_LTL=5000.0,             # Loan limits
    max_STL=4000.0,
    min_CF=0.0,                 # Credit floor
    CC_daily_cap=2000.0,        # Daily cost cap
    CR_k={1: 10.0, 2: 8.0},     # Resource unit costs
    CW_l={1: 50.0}
)

# 3. Define calendar with periods
calendar = CalendarParams(
    T_days=60,
    Y_periods=[(1, 30), (31, 60)]  # Two 30-day periods
)

# 4. Choose IVF–TH targets (PiS/NiS anchors) and weights
targets = IVFTHTargets(
    alpha_level=0.5,
    Z1_PIS=10.0,    # Best makespan (pre-run min-Cmax to estimate)
    Z1_NIS=60.0,    # Worst makespan bound
    Z2_PIS=30000.0, # Best final CF (pre-run max-CF to estimate)
    Z2_NIS=0.0      # Worst final CF bound
)

weights = IVFTHWeights(theta1=0.5, theta2=0.5, gamma_tradeoff=0.5)

# 5. Build & solve
ivfth = RCPSP_CF_IVFTH(activities, finance, calendar)
model = ivfth.build_model(targets, weights)
result = ivfth.solve(model, solver_name="cbc")
print(result)
```

## Finding PiS/NiS targets

For best results, pre-compute the PiS/NiS anchors by solving single-objective problems:

- **Z1 (Cmax) PiS/NiS:** Run a _min-Cmax_ single-objective to estimate a good PiS; set NiS as a safe upper bound (e.g., horizon).
- **Z2 (final CF) PiS/NiS:** Run a _max-CF_ single-objective for PiS; set NiS to a conservative lower bound (e.g., 0 or minCF).

--------------------------------------------------------------------------------

## Data schema (Python objects)

- **NIVTF(ao_L, am_L, ap_L, ao_U, am_U, ap_U)** With ordering: `ao_U < ao_L < am_L == am_U < ap_L < ap_U`.

- **ModeData**
  - `duration: NIVTF`
  - `renewables: Dict[int, NIVTF]` (k → NIVTF per day)
  - `nonrenewables: Dict[int, NIVTF]` (l → NIVTF per day)
  - `payment: float`

- **Activity**
  - `name: str`
  - `predecessors: List[str]`
  - `modes: Dict[int, ModeData]` (mode id → ModeData)

- **FinanceParams**
  - `alpha_excess_cash, beta_delayed_pay, gamma_LTL, delta_STL: float`
  - `IC, max_LTL, max_STL, min_CF, CC_daily_cap: float`
  - `CR_k: Dict[int, float]` (renewable unit costs)
  - `CW_l: Dict[int, float]` (non-renewable unit costs)

- **CalendarParams**
  - `T_days: int`
  - `Y_periods: List[Tuple[int,int]]` (1-based day windows)

--------------------------------------------------------------------------------

## Model overview (constraints)

- **Scheduling**
  - Start once per activity: `∑_{m,t} X_{i,m,t} = 1`
  - Precedence with fuzzy durations (NIVTF α-blend): `t_j ≥ t_i + [α E2_L + (1-α) E1_L]`
  - Completion linking with lower/upper bounds as in (33)–(34)
  - Unique completion per activity: `∑_{m,t} Xp_{i,m,t} = 1`
  - Period mapping `XYp` (ties completion day to a long period)

- **Resources & costs**
  - Daily renewable/non-renewable usage upper-bounded via α-blend linearization
  - Daily cost `BU_t` ≥ Σ(CR_k·BR_{k,t}) + Σ(CW_l·WR_{l,t})
  - Daily cap: `BU_t ≤ CC`

- **Finance & cash flow**
  - Periodic total cost: `TBU_y = ∑ BU_t` in `[a_y, b_y]`
  - Cash flow recurrences with interest on **excess**, **delayed payments**, **loans**
  - Loan caps (`LTL ≤ maxLTL`, `STL_y ≤ maxSTL`), CF floors (`CF_y ≥ minCF`)
  - Delayed payments balance (≤ 1 period delay)

- **Objectives (IVF–TH)**
  - Z1 = `Cmax`, Z2 = `CF_{Y_n}`
  - Memberships `μ1, μ2` linear in (Z1, Z2) using PiS/NiS anchors
  - Scalarization: `max γ·λ + (1-γ)(θ1 μ1 + θ2 μ2)` with `λ ≤ μ1, λ ≤ μ2`

--------------------------------------------------------------------------------

## Reproducibility tips

- Fix solver seeds where supported (CBC/Gurobi).
- Log solver output (`tee=True`) if you need detailed runs.
- Document PiS/NiS derivation (pre-runs) for each dataset.

--------------------------------------------------------------------------------

## Community & Support

- Read the [Contributing guide](CONTRIBUTING.md) for setup instructions, coding standards, and workflow tips.
- Participation is governed by our [Code of Conduct](CODE_OF_CONDUCT.md); please review it before engaging.
- Report security issues privately via the process described in [SECURITY.md](SECURITY.md).
- File bugs or feature requests using the GitHub issue templates; open pull requests against the `develop` branch.

--------------------------------------------------------------------------------

## Citation

If you use this repo in academic work, please cite the original article and this implementation:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1733326.svg)](https://doi.org/10.5281/zenodo.1733326)
```bibtex
@software{rcpsp_cf_ivfth_2025,
  author       = {Ribeiro, Diogo},
  title        = {{RCPSP-CF-IVFTH: Bi-objective Resource-Constrained 
                   Project Scheduling with Cash-Flow under 
                   Interval-Valued Fuzzy Uncertainty}},
  month        = sep,
  year         = 2025,
  publisher    = {Zenodo},
  version      = {1.0.0},
  doi          = {10.5281/zenodo.1733326},
  url          = {https://doi.org/10.5281/zenodo.1733326}
}
```

--------------------------------------------------------------------------------

## License

MIT -- see <LICENSE>.
