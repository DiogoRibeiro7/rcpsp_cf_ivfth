# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-08-07

This release corrects two defects where the implementation departed from the source
paper (Mirnezami, Ghasemi & Shahabi-Shahmiri, arXiv:2509.00002), plus a number of
reporting and tooling problems. **Numerical results differ from 1.x**, so earlier
results cannot be compared against this version without re-running them.

Constraint numbers below refer to the paper's equations.

### Fixed

- **Resources were never released** — the implementation did not match constraints
  (9)/(10) and (31)/(32). Those sum resource demand over a *window* the length of the
  activity's duration; the code instead summed `r·X[i,m,h]` over *every* start day
  `h ≤ t`, so an activity consumed resources from its start day to the end of the
  horizon and the profile could only ever rise. Verified on the toy instance: `BR[1]`
  stayed at 9.0 from day 8 to day 60 although A1 completed on day 8.

  Demand is now summed over `h ∈ [t − span + 1, t]`, with `span` the expected value
  `(d_o + 2·d_m + d_p)/4` of the lower triangle, as (31)/(32) specify. This also drops
  constraint building from quadratic in the horizon to linear in the span.

  Note that (31)/(32) print the window as `h = t … t + span − 1`, which would have an
  activity consuming resources *before* it starts. That reading is untenable, so the
  transposed window is used.

- **Payments were unbounded, so the model could invent cash.** Constraint (18) is
  printed as `Σ PA_im·XYp − PA[y] ≤ DP[y]`, which bounds `PA[y]` only from *below*.
  Because `PA[y]` is a free non-negative variable feeding the cash-flow recurrence,
  the solver could set it arbitrarily high: the toy instance returned
  `PA[1] = 111,999.98` against total real payments of `8,400`, and `CF_final` simply
  tracked whatever `Z2_PIS` asked for, pinning `mu2` at `1.0` in every run. The
  second objective was effectively absent from the problem.

  The balance is now an equality, `Σ PA_im·XYp == PA[y] + DP[y]`, with nothing
  deferred past the final period. **This is a deliberate departure from (18) as
  printed**, and the only one in this release. It matches the model the paper
  describes in prose — payments "permitted to be delayed partly or completely", with
  "the delayed payments of the last period received in the next period" — and without
  it the second objective cannot be optimised at all.

- **Reported activity durations were one day too long.** Constraint (13) sets
  completion to `start + d` and precedence (8) lets a successor start on that same
  day, so `extract_solution`'s `finish − start + 1` over-counted by one and made
  consecutive Gantt bars overlap. Durations are now `finish − start`, and
  zero-duration milestones render as markers rather than invisible bars.

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
  (total across the horizon). **This is an extension beyond the published model, not
  a fix.** The paper has no availability parameter — `BR_kt` and `WR_lt` are free
  variables and resource use is bounded only indirectly, through the maximum daily
  resource cost `CC` in constraint (12). `ResourceParams` is an optional fourth
  positional argument to `RCPSP_CF_IVFTH`; omitting it (the default) reproduces the
  paper's formulation exactly.
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
- **Minimum Python is now 3.10** (was a declared 3.9 that could not actually be
  installed). `pyproject.toml` asked for `pyomo >=6.10.0`, but pyomo 6.10 requires
  Python >= 3.10, so `pip install` on 3.9 failed to resolve. The old CI hid this by
  installing bare `pyomo` and ignoring the project's own constraint. Python 3.9
  reached end of life in October 2025.
- CI tests Python 3.10-3.13 (was 3.8-3.11, where 3.8 contradicted `requires-python`),
  installs the package with its real constraints via `pip install -e ".[test]"` so
  packaging errors surface, installs HiGHS so solver-backed tests actually run, and
  the `lint` job passes.
- `__version__` is read from installed package metadata instead of being a second
  hand-maintained literal.
- All references to the software DOI use `10.5281/zenodo.17382196`; `CITATION.cff`,
  `codemeta.json` and the README previously carried a second, inconsistent DOI.

### Deliberately unchanged

The cash-flow recurrence (20)-(21) is left exactly as published, including the two
loan terms `− LTL/(1+γ)^30` and `− STL[y-1]/(1+δ)^30`. Because these are divisions,
a *higher* interest rate deducts *less* from cash flow, which reads backwards
economically and invites a well-meaning correction. It is nonetheless what equations
(1), (2) and (21) specify, and reproducing the published model is the purpose of this
package. The behaviour is pinned by tests in `tests/test_model_semantics.py`.

For the same reason the four interest rates remain *daily* rates compounded over a
30-day period, per equations (1)-(4).

## [1.0.1]

- Dependency constraint upgrades.

## [1.0.0]

- Initial release.
