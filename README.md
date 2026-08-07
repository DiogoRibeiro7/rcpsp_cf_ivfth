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
- **Multi-mode activities** with **renewable/non-renewable** resources and optional **availability limits**.
- **Daily & periodic costs**, **initial capital**, **short-/long-term loans**, **credit limits**, **delayed payments** (≤ 1 period).
- **Uncertainty** in durations and resource demands via **Normalized Interval-Valued Triangular Fuzzy (NIVTF)** numbers.
- **Extended IVF–TH** scalarization (Torabi–Hassini) to convert the fuzzy bi-objective to a single-objective MILP.
- Written in pure Python with **Pyomo**. Works with HiGHS/GLPK/CBC/CPLEX/Gurobi.

> **Upgrading from 1.x?** Version 2.0.0 corrects four defects in the formulation, so
> every numerical result changes. See the [CHANGELOG](CHANGELOG.md) before comparing
> against previously published runs.

--------------------------------------------------------------------------------

## Repository layout

```text
rcpsp_cf_ivfth/
├── __init__.py           # Main package exports
├── fuzzy.py              # NIVTF fuzzy number definitions
├── data.py               # Data structures (Activity, FinanceParams, ResourceParams, ...)
├── model.py              # Main RCPSP_CF_IVFTH solver class
├── sensitivity.py        # Alpha / weight / finance sweeps (needs pandas)
├── visualization.py      # Gantt, resource, cash-flow plots and exports (needs matplotlib)
└── examples/
    ├── __init__.py
    ├── __main__.py       # `python -m rcpsp_cf_ivfth.examples`
    └── toy_instance.py   # Example usage with toy data
```

--------------------------------------------------------------------------------

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -e .                     # core: pyomo only
pip install -e ".[solvers]"          # adds HiGHS (a wheel, no system packages needed)
pip install -e ".[visualization]"    # adds matplotlib
pip install -e ".[sensitivity]"      # adds pandas + matplotlib
pip install -e ".[test,dev]"         # everything needed to run the test suite and linters

python -m rcpsp_cf_ivfth.examples
```

You need one MILP solver. **HiGHS** is the easiest — it installs as a wheel on every
platform via `pip install highspy` and needs nothing on `PATH`. Alternatives:

| Solver | Install | `solver_name` |
| --- | --- | --- |
| HiGHS | `pip install highspy` | `"appsi_highs"` |
| GLPK | `apt-get install glpk-utils` / `brew install glpk` | `"glpk"` |
| CBC | `conda install -c conda-forge coincbc` | `"cbc"` |
| Gurobi / CPLEX | vendor installer (licence required) | `"gurobi"` / `"cplex"` |

> **Solver note:** Set the solver in code: `ivfth.solve(model, solver_name="appsi_highs")`.
> If the model is infeasible, `solve()` raises a `RuntimeError` naming the termination
> condition rather than failing inside Pyomo.

--------------------------------------------------------------------------------

## Quick start (toy instance)

Run the example with:

```bash
python -m rcpsp_cf_ivfth.examples
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
    ResourceParams, IVFTHTargets, IVFTHWeights, NIVTF, create_triangle
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

# 2. Define finance parameters.
#    All four rates are effective rates *per accounting period*.
finance = FinanceParams(
    alpha_excess_cash=0.0125,   # Earned on cash carried into the next period
    beta_delayed_pay=0.10,      # Earned on a payment deferred by one period
    gamma_LTL=0.06,             # Charged per period on the long-term loan
    delta_STL=0.075,            # Charged over a short-term loan's one-period life
    IC=10000.0,                 # Initial capital
    max_LTL=5000.0,             # Loan limits
    max_STL=4000.0,
    min_CF=0.0,                 # Credit floor
    CC_daily_cap=2000.0,        # Daily cost cap
    CR_k={1: 10.0, 2: 8.0},     # Resource unit costs
    CW_l={1: 50.0}
)

# 3. Define resource availability. Optional: omit it and the schedule is limited only
#    by precedence and the daily cost cap.
resources = ResourceParams(
    renewable_capacity={1: 6.0, 2: 3.0},   # Available *each day*
    nonrenewable_capacity={1: 40.0},       # Available in *total* over the horizon
)

# 4. Define calendar with periods
calendar = CalendarParams(
    T_days=60,
    Y_periods=[(1, 30), (31, 60)]  # Two 30-day periods
)

# 5. Choose IVF–TH targets (PiS/NiS anchors) and weights
targets = IVFTHTargets(
    alpha_level=0.5,
    Z1_PIS=10.0,    # Best makespan (pre-run min-Cmax to estimate)
    Z1_NIS=60.0,    # Worst makespan bound
    Z2_PIS=30000.0, # Best final CF (pre-run max-CF to estimate)
    Z2_NIS=0.0      # Worst final CF bound
)

weights = IVFTHWeights(theta1=0.5, theta2=0.5, gamma_tradeoff=0.5)

# 6. Build & solve
ivfth = RCPSP_CF_IVFTH(activities, finance, calendar, resources)
model = ivfth.build_model(targets, weights)
result = ivfth.solve(model, solver_name="appsi_highs")
print(result)

# 7. Inspect, plot or export the full solution
solution = ivfth.extract_solution(model, solver_metadata=result)
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
    (effective rates **per accounting period**, applied directly)
  - `IC, max_LTL, max_STL, min_CF, CC_daily_cap: float`
  - `CR_k: Dict[int, float]` (renewable unit costs)
  - `CW_l: Dict[int, float]` (non-renewable unit costs)

- **ResourceParams** (optional)
  - `renewable_capacity: Dict[int, float]` (k → units available **each day**)
  - `nonrenewable_capacity: Dict[int, float]` (l → units available **in total**)
  - A resource omitted from either mapping is unlimited.

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
  - Daily demand counts only activities whose **active window covers day `t`**: an
    activity started at `h` occupies days `h … h + span − 1`, where `span` is the
    α-blend duration rounded up, and releases its resources afterwards.
  - Availability (when `ResourceParams` is supplied): `BR_{k,t} ≤ R_k` per day,
    and `∑_t WR_{l,t} ≤ W_l` across the horizon.
  - Daily cost `BU_t` ≥ Σ(CR_k·BR_{k,t}) + Σ(CW_l·WR_{l,t})
  - Daily cap: `BU_t ≤ CC`

- **Finance & cash flow**
  - Periodic total cost: `TBU_y = ∑ BU_t` in `[a_y, b_y]`
  - Payment balance (≤ 1 period delay), as an **equality**: revenue earned in period
    `y` is either collected then or deferred once — `∑ PA_{i,m}·XYp = PA_y + DP_y`,
    with `DP_{Y_n} = 0` so nothing is deferred past the horizon.
  - Cash flow recurrence with interest on **excess cash** and **delayed payments**,
    and debt service on **loans**: `− STL_{y−1}·(1+δ)` repays the prior short-term
    loan, `− LTL·γ` services the long-term loan.
  - Terminal settlement: the final period clears outstanding principal, so
    `Z2 = CF_{Y_n}` is cash kept, not cash still owed.
  - Loan caps (`LTL ≤ maxLTL`, `STL_y ≤ maxSTL`), CF floors (`CF_y ≥ minCF`)

- **Objectives (IVF–TH)**
  - Z1 = `Cmax`, Z2 = `CF_{Y_n}`
  - Memberships `μ1, μ2` linear in (Z1, Z2) using PiS/NiS anchors
  - Scalarization: `max γ·λ + (1-γ)(θ1 μ1 + θ2 μ2)` with `λ ≤ μ1, λ ≤ μ2`

--------------------------------------------------------------------------------

## Reproducibility tips

- Record the package version: the formulation changed in 2.0.0 and results are not
  comparable across that boundary. See the [CHANGELOG](CHANGELOG.md).
- Fix solver seeds where supported (CBC/Gurobi).
- Log solver output (`tee=True`) if you need detailed runs.
- Document PiS/NiS derivation (pre-runs) for each dataset.

--------------------------------------------------------------------------------

## Development

```bash
pip install -e ".[test,dev]"

pytest                          # full suite, 80% coverage gate
pytest -m "not solver"          # skip everything needing a MILP backend
python run_tests.py --no-solver # same, via the convenience wrapper

black rcpsp_cf_ivfth/ tests/ scripts/
isort rcpsp_cf_ivfth/ tests/ scripts/
flake8 rcpsp_cf_ivfth/ tests/ scripts/
```

Pytest settings live in `[tool.pytest.ini_options]` in `pyproject.toml`; flake8 reads
`.flake8` because it cannot read `pyproject.toml`. Regenerate `docs/api.md` with
`python docs/_generate_api_md.py` after changing public docstrings.

--------------------------------------------------------------------------------

## Community & Support

- Read the [Contributing guide](CONTRIBUTING.md) for setup instructions, coding standards, and workflow tips.
- Participation is governed by our [Code of Conduct](CODE_OF_CONDUCT.md); please review it before engaging.
- Report security issues privately via the process described in [SECURITY.md](SECURITY.md).
- File bugs or feature requests using the GitHub issue templates; open pull requests against the `develop` branch.

--------------------------------------------------------------------------------

## Citation

If you use this repo in academic work, please cite the original article and this implementation:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17382196.svg)](https://doi.org/10.5281/zenodo.17382196)
```bibtex
@software{rcpsp_cf_ivfth_2025,
  author       = {Ribeiro, Diogo},
  title        = {{RCPSP-CF-IVFTH: Bi-objective Resource-Constrained 
                   Project Scheduling with Cash-Flow under 
                   Interval-Valued Fuzzy Uncertainty}},
  month        = sep,
  year         = 2025,
  publisher    = {Zenodo},
  version      = {2.0.0},
  doi          = {10.5281/zenodo.17382196},
  url          = {https://doi.org/10.5281/zenodo.17382196}
}
```

--------------------------------------------------------------------------------

## License

MIT -- see <LICENSE>.
