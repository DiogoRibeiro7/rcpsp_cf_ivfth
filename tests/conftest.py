"""
pytest configuration and fixtures for RCPSP-CF-IVFTH tests.
"""

import pytest
from typing import Dict, Tuple

from rcpsp_cf_ivfth import (
    RCPSP_CF_IVFTH,
    Activity,
    ModeData,
    FinanceParams,
    CalendarParams,
    IVFTHTargets,
    IVFTHWeights,
    NIVTF,
    create_triangle,
)


@pytest.fixture
def simple_nivtf():
    """Create a simple NIVTF for testing."""
    return NIVTF(*create_triangle(5, 7, 9, widen=0.5))


@pytest.fixture
def toy_instance():
    """Create the toy instance from the package."""
    # Renewables k ∈ {1,2}, Non-renewables l ∈ {1}
    CR_k = {1: 10.0, 2: 8.0}
    CW_l = {1: 50.0}

    # Activities
    A0 = Activity(
        name="Start",
        predecessors=[],
        modes={
            1: ModeData(
                duration=NIVTF(*create_triangle(0.0, 0.0, 0.0, widen=0.1)),
                renewables={
                    1: NIVTF(*create_triangle(0, 0, 0)),
                    2: NIVTF(*create_triangle(0, 0, 0)),
                },
                nonrenewables={1: NIVTF(*create_triangle(0, 0, 0))},
                payment=0.0,
            )
        },
    )
    A1 = Activity(
        name="A1",
        predecessors=["Start"],
        modes={
            1: ModeData(
                duration=NIVTF(*create_triangle(6, 8, 10, widen=0.6)),
                renewables={
                    1: NIVTF(*create_triangle(3, 4, 5)),
                    2: NIVTF(*create_triangle(0, 1, 2)),
                },
                nonrenewables={1: NIVTF(*create_triangle(0, 1, 2))},
                payment=3000.0,
            ),
            2: ModeData(
                duration=NIVTF(*create_triangle(5, 7, 9, widen=0.6)),
                renewables={
                    1: NIVTF(*create_triangle(4, 5, 6)),
                    2: NIVTF(*create_triangle(1, 2, 3)),
                },
                nonrenewables={1: NIVTF(*create_triangle(0, 2, 3))},
                payment=3600.0,
            ),
        },
    )
    A2 = Activity(
        name="A2",
        predecessors=["A1"],
        modes={
            1: ModeData(
                duration=NIVTF(*create_triangle(8, 10, 12, widen=0.6)),
                renewables={
                    1: NIVTF(*create_triangle(2, 3, 4)),
                    2: NIVTF(*create_triangle(1, 1, 2)),
                },
                nonrenewables={1: NIVTF(*create_triangle(1, 2, 3))},
                payment=4200.0,
            ),
            2: ModeData(
                duration=NIVTF(*create_triangle(6, 9, 11, widen=0.6)),
                renewables={
                    1: NIVTF(*create_triangle(3, 4, 5)),
                    2: NIVTF(*create_triangle(0, 1, 2)),
                },
                nonrenewables={1: NIVTF(*create_triangle(0, 1, 3))},
                payment=4800.0,
            ),
        },
    )
    A3 = Activity(
        name="End",
        predecessors=["A2"],
        modes={
            1: ModeData(
                duration=NIVTF(*create_triangle(0.0, 0.0, 0.0, widen=0.1)),
                renewables={
                    1: NIVTF(*create_triangle(0, 0, 0)),
                    2: NIVTF(*create_triangle(0, 0, 0)),
                },
                nonrenewables={1: NIVTF(*create_triangle(0, 0, 0))},
                payment=0.0,
            )
        },
    )

    acts = {"Start": A0, "A1": A1, "A2": A2, "End": A3}

    finance = FinanceParams(
        alpha_excess_cash=0.0125,  # α
        beta_delayed_pay=0.10,  # β
        gamma_LTL=0.06,  # γ
        delta_STL=0.075,  # δ
        IC=10000.0,  # initial capital
        max_LTL=5000.0,
        max_STL=4000.0,
        min_CF=0.0,
        CC_daily_cap=2000.0,  # daily resource cost cap
        CR_k=CR_k,
        CW_l=CW_l,
    )

    # Two 30-day periods, total 60 days
    calendar = CalendarParams(T_days=60, Y_periods=[(1, 30), (31, 60)])

    return acts, finance, calendar


@pytest.fixture
def ivfth_targets():
    """Create IVF-TH targets for testing."""
    return IVFTHTargets(
        alpha_level=0.5,
        Z1_PIS=10.0,  # optimistic (best) makespan target (days)
        Z1_NIS=60.0,  # pessimistic (worst) makespan bound
        Z2_PIS=30000.0,  # optimistic final CF
        Z2_NIS=0.0,  # pessimistic final CF
    )


@pytest.fixture
def ivfth_weights():
    """Create IVF-TH weights for testing."""
    return IVFTHWeights(theta1=0.5, theta2=0.5, gamma_tradeoff=0.5)


@pytest.fixture
def solver_name():
    """Return the solver name to use for tests."""
    # Try to use CBC first, then GLPK
    try:
        from pyomo.environ import SolverFactory

        opt = SolverFactory("cbc")
        if opt.available():
            return "cbc"
    except:
        pass

    try:
        from pyomo.environ import SolverFactory

        opt = SolverFactory("glpk")
        if opt.available():
            return "glpk"
    except:
        pass

    pytest.skip("No suitable MILP solver (CBC or GLPK) available")
