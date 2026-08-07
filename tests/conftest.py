"""
pytest configuration and fixtures for RCPSP-CF-IVFTH tests.
"""

import pytest

from rcpsp_cf_ivfth import NIVTF, IVFTHTargets, IVFTHWeights, create_triangle
from rcpsp_cf_ivfth.examples import build_toy_instance, build_toy_resources

# Solvers are tried in this order; the first available one is used for solve tests.
CANDIDATE_SOLVERS = ("cbc", "glpk", "appsi_highs")


@pytest.fixture
def simple_nivtf():
    """Create a simple NIVTF for testing."""
    return NIVTF(*create_triangle(5, 7, 9, widen=0.5))


@pytest.fixture
def toy_instance():
    """
    Return the canonical toy instance.

    This delegates to :func:`rcpsp_cf_ivfth.examples.build_toy_instance` rather than
    redefining the activities, so the fixture and the shipped example cannot drift
    apart. Each call builds fresh objects, so tests are free to mutate the result.
    """
    return build_toy_instance()


@pytest.fixture
def toy_resources():
    """Resource availability limits matching the toy instance."""
    return build_toy_resources()


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


def find_solver():
    """Return the name of the first available MILP solver, or None."""
    try:
        from pyomo.environ import SolverFactory
    except ImportError:  # pragma: no cover - pyomo is a hard dependency
        return None

    for name in CANDIDATE_SOLVERS:
        try:
            opt = SolverFactory(name)
        except Exception:
            continue
        if opt is not None and opt.available(exception_flag=False):
            return name
    return None


@pytest.fixture
def solver_name():
    """Return the solver name to use for tests, skipping if none is installed."""
    name = find_solver()
    if name is None:
        pytest.skip(f"No MILP solver available (tried: {', '.join(CANDIDATE_SOLVERS)})")
    return name
