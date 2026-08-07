"""
RCPSP-CF-IVFTH: Bi-objective Resource-Constrained Project Scheduling with
Cash-Flow under fuzzy uncertainty.

This package implements the model from:
"A New Bi-Objective Model for Resource-Constrained Project Scheduling and Cash Flow Problems
with Financial Constraints under Uncertainty: A Case Study"

Main components:
- Fuzzy numbers (NIVTF) for handling uncertainty
- Data structures for activities, modes, finance, and calendar parameters
- MILP model builder with Extended IVF-TH scalarization
- Solver interface for various MILP solvers

Example usage:
    from rcpsp_cf_ivfth import RCPSP_CF_IVFTH, IVFTHTargets, IVFTHWeights
    from rcpsp_cf_ivfth.examples import build_toy_instance

    activities, finance, calendar = build_toy_instance()
    ivfth = RCPSP_CF_IVFTH(activities, finance, calendar)

    targets = IVFTHTargets(alpha_level=0.5, Z1_PIS=10.0, Z1_NIS=60.0, Z2_PIS=30000.0, Z2_NIS=0.0)
    weights = IVFTHWeights(theta1=0.5, theta2=0.5, gamma_tradeoff=0.5)

    model = ivfth.build_model(targets, weights)
    results = ivfth.solve(model, solver_name="glpk")
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _package_version

try:
    # Single source of truth: the version declared in pyproject.toml. The literal
    # fallback only applies when running from a source tree that was never installed.
    __version__ = _package_version("rcpsp-cf-ivfth")
except PackageNotFoundError:  # pragma: no cover - source checkout without install
    __version__ = "2.0.0"

__author__ = "Diogo Ribeiro"

from .data import (
    Activity,
    CalendarParams,
    FinanceParams,
    IVFTHTargets,
    IVFTHWeights,
    ModeData,
    ResourceParams,
)
from .fuzzy import NIVTF, create_triangle

# Main exports
from .model import RCPSP_CF_IVFTH
from .sensitivity import (
    plot_metric_trends,
    run_alpha_sweep,
    run_finance_scenarios,
    run_weight_scenarios,
    sensitivity_analysis,
)
from .visualization import (
    create_gantt_chart,
    export_solution_csv,
    export_solution_json,
    plot_cash_flow,
    plot_loan_usage,
    plot_resource_usage,
)

__all__ = [
    "RCPSP_CF_IVFTH",
    "Activity",
    "ModeData",
    "FinanceParams",
    "CalendarParams",
    "ResourceParams",
    "IVFTHTargets",
    "IVFTHWeights",
    "NIVTF",
    "create_triangle",
    "create_gantt_chart",
    "plot_resource_usage",
    "plot_cash_flow",
    "plot_loan_usage",
    "export_solution_json",
    "export_solution_csv",
    "run_alpha_sweep",
    "run_weight_scenarios",
    "run_finance_scenarios",
    "sensitivity_analysis",
    "plot_metric_trends",
]
