"""Toy instance helpers for RCPSP-CF-IVFTH.

Provides a canonical four-activity toy instance (Start -> A1 -> A2 -> End)
used by the test-suite, as well as a small helper for running the instance
through the solver when a MILP backend is available.
"""

from __future__ import annotations

from typing import Dict, Tuple

from ..data import Activity, ModeData, FinanceParams, CalendarParams
from ..fuzzy import NIVTF, create_triangle


def _zero_nivtf() -> NIVTF:
    """Return an NIVTF representing a deterministic zero value."""
    return NIVTF(*create_triangle(0.0, 0.0, 0.0, widen=0.1))


def build_toy_instance() -> Tuple[Dict[str, Activity], FinanceParams, CalendarParams]:
    """Construct the toy instance used by the tests."""
    renewable_costs = {1: 10.0, 2: 8.0}
    nonrenewable_costs = {1: 50.0}

    activities: Dict[str, Activity] = {
        "Start": Activity(
            name="Start",
            predecessors=[],
            modes={
                1: ModeData(
                    duration=_zero_nivtf(),
                    renewables={1: _zero_nivtf(), 2: _zero_nivtf()},
                    nonrenewables={1: _zero_nivtf()},
                    payment=0.0,
                )
            },
        ),
        "A1": Activity(
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
        ),
        "A2": Activity(
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
        ),
        "End": Activity(
            name="End",
            predecessors=["A2"],
            modes={
                1: ModeData(
                    duration=_zero_nivtf(),
                    renewables={1: _zero_nivtf(), 2: _zero_nivtf()},
                    nonrenewables={1: _zero_nivtf()},
                    payment=0.0,
                )
            },
        ),
    }

    finance = FinanceParams(
        alpha_excess_cash=0.0125,
        beta_delayed_pay=0.10,
        gamma_LTL=0.06,
        delta_STL=0.075,
        IC=10000.0,
        max_LTL=5000.0,
        max_STL=4000.0,
        min_CF=0.0,
        CC_daily_cap=2000.0,
        CR_k=renewable_costs,
        CW_l=nonrenewable_costs,
    )

    calendar = CalendarParams(
        T_days=60,
        Y_periods=[(1, 30), (31, 60)],
    )

    return activities, finance, calendar


def run_toy_example() -> None:
    """Solve the toy instance using the first available MILP solver."""
    from pyomo.environ import SolverFactory

    from .. import RCPSP_CF_IVFTH, IVFTHTargets, IVFTHWeights

    activities, finance, calendar = build_toy_instance()
    ivfth = RCPSP_CF_IVFTH(activities, finance, calendar)

    targets = IVFTHTargets(
        alpha_level=0.5,
        Z1_PIS=10.0,
        Z1_NIS=60.0,
        Z2_PIS=30000.0,
        Z2_NIS=0.0,
    )
    weights = IVFTHWeights(theta1=0.5, theta2=0.5, gamma_tradeoff=0.5)

    model = ivfth.build_model(targets, weights)

    for solver_name in ("cbc", "glpk"):
        solver = SolverFactory(solver_name)
        if solver.available():
            result = ivfth.solve(model, solver_name=solver_name)
            print(f"Solved with {solver_name}: {result}")
            break
    else:
        print("Toy instance model built, but no CBC or GLPK solver was found.")


if __name__ == "__main__":
    run_toy_example()
