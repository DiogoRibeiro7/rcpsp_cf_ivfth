# RCPSP-CF-IVFTH

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

- **`rcpsp_cf_ivfth.py`** -- all code in a single file (model, fuzzy numbers, toy instance, solver harness).
- (Optional later) `rcpsp_cf_ivfth/` package split for larger instances.

--------------------------------------------------------------------------------

## Installation

### With Poetry (recommended)

```bash
poetry init --no-interaction
poetry add pyomo
# Add a MILP solver; choose one you have:
# GLPK (Linux): sudo apt-get install glpk-utils
# CBC (cross-platform via conda): conda install -c conda-forge coincbc
poetry run python rcpsp_cf_ivfth.py
```

### With pip

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install pyomo
# Install a solver (e.g., cbc or glpk) and ensure it's on PATH
python rcpsp_cf_ivfth.py
```

> **Solver note:** Set the solver in code: `ivfth.solve(model, solver_name="glpk")`. Alternatives: `"cbc"`, `"gurobi"`, `"cplex"` (if licensed/installed).

--------------------------------------------------------------------------------

## Quick start (toy instance)

The file ships with a minimal, runnable instance:

```bash
python rcpsp_cf_ivfth.py
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

1. **Edit or replace** `build_toy_instance()`:

  - Define activities with modes, NIVTF durations, NIVTF resource usage per day, and a payment per mode.
  - Define finance parameters: interest rates (α, β, γ, δ), initial capital, loan caps, min CF, daily cost cap, resource unit costs.
  - Define calendar: total days `T_days` and long periods `Y_periods = [(a_y, b_y)]`.

2. **Choose IVF–TH targets** (PiS/NiS anchors) and weights:

  ```python
  targets = IVFTHTargets(
      alpha_level=0.5,
      Z1_PIS=..., Z1_NIS=...,
      Z2_PIS=..., Z2_NIS=...
  )
  weights = IVFTHWeights(theta1=0.5, theta2=0.5, gamma_tradeoff=0.5)
  ```

  - **Z1 (Cmax) PiS/NiS:** run a _min-Cmax_ single-objective to estimate a good PiS; set NiS as a safe upper bound (e.g., horizon).
  - **Z2 (final CF) PiS/NiS:** run a _max-CF_ single-objective for PiS; set NiS to a conservative lower bound (e.g., 0 or minCF).

3. **Build & solve:**

  ```python
  ivfth = RCPSP_CF_IVFTH(activities, finance, calendar)
  model = ivfth.build_model(targets, weights)
  result = ivfth.solve(model, solver_name="cbc")
  ```

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

## Citation

If you use this repo in academic work, please cite the original article and this implementation:

```bibtex
@misc{rcpsp_cf_ivfth_implementation_2025,
  title   = {RCPSP-CF-IVFTH: Bi-objective RCPSP with Cash-Flow under IVF Uncertainty},
  year    = {2025},
  note    = {Python/Pyomo implementation of extended IVF–TH scalarization for the model},
  url     = {https://github.com/<your-org>/rcpsp-cf-ivfth}
}
```

--------------------------------------------------------------------------------

## License

MIT -- see <LICENSE>.
