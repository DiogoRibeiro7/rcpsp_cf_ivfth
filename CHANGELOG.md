# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-08-07

This release corrects four defects in the MILP formulation. **Every numerical result
produced by 1.x differs from 2.0.0**, so results published from an earlier version
cannot be compared against this one without re-running them.

### Fixed

- **Payments were unbounded, so the model could invent cash.** Constraint (18) was
  written as `Σ PA_im·XYp − PA[y] ≤ DP[y]`, which bounds `PA[y]` only from *below*.
  Because `PA[y]` is a free non-negative variable that feeds the cash-flow
  recurrence, the solver could set it arbitrarily high. On the shipped toy instance
  it returned `PA[1] = 111,999.98` against total real payments of `8,400`.

  The practical effect was that `mu2` reached `1.0` in every run, so the second
  objective was always fully satisfied and the bi-objective trade-off never
  happened — the model effectively minimised makespan alone. The balance is now an
  equality, `Σ PA_im·XYp == PA[y] + DP[y]`, and nothing may be deferred past the
  final period.

- **Resources were never released.** The daily demand constraints summed
  `r·X[i,m,h]` over *every* start day `h ≤ t`, so an activity consumed resources
  from its start day to the end of the horizon and the resource profile could only
  ever rise. Demand is now summed over the activity's active window,
  `h ∈ [t − span + 1, t]`, where `span` is the alpha-blended duration rounded up to
  whole days — the same blend the precedence constraint uses. This also drops the
  constraint-building cost from quadratic in the horizon to linear in the span.

- **Loan repayment moved the wrong way with the interest rate.** The recurrence
  discounted repayments as `LTL / (1+γ)^30` and `STL[y-1] / (1+δ)^30`, which made
  debt *cheaper* as rates rose. Repayment is now `STL[y-1] · (1+δ)` for short-term
  loans and `LTL · γ` per period of interest service on the long-term loan.

  The final period additionally settles outstanding principal, so `Z2 = CF[Yn]`
  measures cash the project keeps rather than money it has borrowed and not repaid.
  Previously a loan drawn in the last period was pure profit.

- **Interest rates are applied per accounting period.** The rates were documented as
  "per 30-day period" but raised to the 30th power, turning the toy instance's 6%
  long-term rate into 474%. They are now effective per-period rates applied directly.

- **Reported activity durations were one day too long.** `extract_solution` computed
  `finish − start + 1`, but the model releases an activity on its `finish` day and
  lets successors start that same day. Durations are now `finish − start`, and
  zero-duration milestones render as markers rather than invisible Gantt bars.

- **`plot_metric_trends` rejected its own primary input.** It validated `x_column`
  against `df.columns` *before* calling `reset_index()`, so plotting the output of
  `run_alpha_sweep` — where `alpha_level` is the index — always raised.

- **`run_finance_scenarios` dropped resource limits.** Scenario models are rebuilt
  from scratch and were not given the parent's `ResourceParams`.

- **Infeasible models raised an opaque Pyomo error.** `solve()` now defers loading
  the solution and raises a `RuntimeError` naming the termination condition.

- **`solver_time` could not be serialized.** Solvers that report no timing return
  Pyomo's `UndefinedData` sentinel, which broke `export_solution_json` whenever
  solver metadata was attached to a solution. It is now coerced to `float` or `None`.

- **Cycle detection overflowed the stack** on activity graphs deeper than Python's
  recursion limit. The depth-first search is now iterative.

### Added

- `ResourceParams`, carrying renewable capacity (per day) and non-renewable capacity
  (total across the horizon). This is what makes the problem resource-*constrained*:
  before, the only limit on resource use was `FinanceParams.CC_daily_cap`, a cost cap.
  It is an optional fourth positional argument to `RCPSP_CF_IVFTH`; omitting it
  reproduces the unconstrained behaviour.
- `build_toy_resources()`, giving the toy instance limits that force a real
  makespan-versus-resource trade-off.
- `sensitivity`, `solvers` and `dev` extras. `pandas` is required by
  `rcpsp_cf_ivfth.sensitivity` but was previously declared nowhere.
- Regression tests for each defect above, plus coverage for the previously untested
  `visualization` and `sensitivity` modules.

### Changed

- Pytest configuration moved from `pytest.ini` to `[tool.pytest.ini_options]` in
  `pyproject.toml`. The old file used a `[tool:pytest]` header, which is only valid
  in `setup.cfg`, so every setting in it — including the 80% coverage gate — had been
  silently ignored.
- Solver-dependent tests carry the `solver` marker, so `-m "not solver"` now works.
- CI tests Python 3.9-3.13 (was 3.8-3.11, where 3.8 contradicted `requires-python`),
  installs HiGHS so solver-backed tests actually run, and the `lint` job passes.
- `__version__` is read from installed package metadata instead of being a second
  hand-maintained literal.
- All references to the software DOI use `10.5281/zenodo.17382196`; `CITATION.cff`,
  `codemeta.json` and the README previously carried a second, inconsistent DOI.

## [1.0.1]

- Dependency constraint upgrades.

## [1.0.0]

- Initial release.
