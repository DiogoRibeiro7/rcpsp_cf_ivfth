"""
RCPSP-CF-IVFTH: Bi-objective Resource-Constrained Project Scheduling with Cash-Flow under fuzzy uncertainty.

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

__version__ = "1.0.0"
__author__ = "Diogo Ribeiro"

# Main exports
from .model import RCPSP_CF_IVFTH
from .data import (
    Activity,
    ModeData,
    FinanceParams,
    CalendarParams,
    IVFTHTargets,
    IVFTHWeights,
)
from .fuzzy import NIVTF, create_triangle

__all__ = [
    "RCPSP_CF_IVFTH",
    "Activity",
    "ModeData", 
    "FinanceParams",
    "CalendarParams",
    "IVFTHTargets",
    "IVFTHWeights",
    "NIVTF",
    "create_triangle",
]
