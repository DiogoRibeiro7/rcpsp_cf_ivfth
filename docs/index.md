# RCPSP-CF-IVFTH Documentation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1733326.svg)](https://doi.org/10.5281/zenodo.1733326)

Welcome to the documentation for the RCPSP-CF-IVFTH project. This package implements the bi-objective Resource-Constrained Project Scheduling Problem with Cash Flow (RCPSP-CF) under interval-valued fuzzy uncertainty, solved through the extended IVF-TH scalarization and a mixed-integer linear programming (MILP) formulation.

## Citation

If you use this software in your research, please cite:
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

## Original Paper

The implementation follows the case study described in:

- _A New Bi-Objective Model for Resource-Constrained Project Scheduling and Cash Flow Problems with Financial Constraints under Uncertainty: A Case Study_

You can read the original paper included in this repository at: [Project Scheduling.pdf](../Project%20Scheduling.pdf).

## Project Highlights

- **Bi-objective focus**: Minimizes makespan while maximizing end-of-horizon cash flow in one MILP objective via IVF-TH scalarization.
- **Rich activity model**: Supports multi-mode activities with renewable and non-renewable resources, delayed payments, and interest accrual.
- **Uncertainty modeling**: Encodes activity durations and resource usage as Normalized Interval-Valued Triangular Fuzzy (NIVTF) numbers.
- **Finance-aware constraints**: Tracks cash balances, loan usage, and credit limits across accounting periods.
- **Pyomo-based implementation**: Leverages Pyomo for model construction so you can run with open-source or commercial MILP solvers.

## Documentation Map

- [Usage guide](usage.md): Step-by-step instructions for adapting `build_toy_instance` to your scenario and tuning IVF-TH targets.
- [API reference](api.md): Auto-generated documentation for public classes, functions, and dataclasses.
- `README.md`: Installation instructions, detailed feature list, and problem background.

For development tips and solver configuration examples, see the project README and inline docstrings throughout the `rcpsp_cf_ivfth` package.
