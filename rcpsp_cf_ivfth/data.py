"""
Data structures for RCPSP-CF-IVFTH problem instances.

This module contains all the data classes and parameters needed to define
a Resource-Constrained Project Scheduling Problem with Cash Flow under
fuzzy uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .fuzzy import NIVTF


@dataclass
class ModeData:
    """
    Per-activity, per-mode data.

    Attributes
    ----------
    duration : NIVTF
        Uncertain duration (NIVTF).
    renewables : Dict[int, NIVTF]
        Per renewable resource k, uncertain daily use NIVTF.
    nonrenewables : Dict[int, NIVTF]
        Per non-renewable resource l, uncertain daily use NIVTF.
    payment : float
        Payment amount PA_{i,m} devoted to the period where the activity
        completes (or the next one if delayed).
    """

    duration: NIVTF
    renewables: Dict[int, NIVTF]
    nonrenewables: Dict[int, NIVTF]
    payment: float


@dataclass
class Activity:
    """
    Activity with multiple modes and precedence list.

    Attributes
    ----------
    name : str
        Unique identifier for the activity.
    predecessors : List[str]
        List of activity names that must be completed before this activity can start.
    modes : Dict[int, ModeData]
        Dictionary mapping mode index to mode data.
    """

    name: str
    predecessors: List[str]
    modes: Dict[int, ModeData]  # mode index -> ModeData


@dataclass
class FinanceParams:
    """
    Financial parameters for the model.

    Attributes
    ----------
    alpha_excess_cash : float
        Effective interest earned on cash carried into the next accounting period.
    beta_delayed_pay : float
        Effective compensation earned on a payment deferred by one accounting period.
    gamma_LTL : float
        Effective interest charged on the long-term loan per accounting period.
    delta_STL : float
        Effective interest charged on a short-term loan over its one-period life.
    IC : float
        Initial capital (received at period 1).
    max_LTL : float
        Upper bound on long-term loan (single, period 1 only).
    max_STL : float
        Upper bound on short-term loan per period.
    min_CF : float
        Minimum cash-flow per period (credit floor).
    CC_daily_cap : float
        Upper bound on total daily resource cost (constraint (12)).
    CR_k : Dict[int, float]
        Cost per unit of renewable resource k per day.
    CW_l : Dict[int, float]
        Cost per unit of non-renewable resource l per day.

    Notes
    -----
    All four interest rates are *effective rates per accounting period*, applied
    directly (not compounded over 30 days). With the default toy values this means
    1.25% earned on cash carried forward, 6% charged per period on the long-term
    loan, and so on.
    """

    alpha_excess_cash: float
    beta_delayed_pay: float
    gamma_LTL: float
    delta_STL: float
    IC: float
    max_LTL: float
    max_STL: float
    min_CF: float
    CC_daily_cap: float
    CR_k: Dict[int, float]
    CW_l: Dict[int, float]


@dataclass
class ResourceParams:
    """
    Availability limits for renewable and non-renewable resources.

    Attributes
    ----------
    renewable_capacity : Dict[int, float]
        Per renewable resource k, the amount available *on each day*. A resource
        omitted from the mapping is treated as unlimited.
    nonrenewable_capacity : Dict[int, float]
        Per non-renewable resource l, the amount available *in total across the whole
        horizon*. A resource omitted from the mapping is treated as unlimited.

    Notes
    -----
    This is the constraint that makes the problem resource-*constrained*: without it
    the schedule is limited only by precedence and by ``FinanceParams.CC_daily_cap``.
    Passing ``resources=None`` to :class:`~rcpsp_cf_ivfth.model.RCPSP_CF_IVFTH`
    reproduces the unconstrained behaviour.

    Examples
    --------
    >>> ResourceParams(renewable_capacity={1: 6.0, 2: 3.0}, nonrenewable_capacity={1: 40.0})
    ResourceParams(renewable_capacity={1: 6.0, 2: 3.0}, nonrenewable_capacity={1: 40.0})
    """

    renewable_capacity: Dict[int, float] = field(default_factory=dict)
    nonrenewable_capacity: Dict[int, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for label, mapping in (
            ("renewable_capacity", self.renewable_capacity),
            ("nonrenewable_capacity", self.nonrenewable_capacity),
        ):
            for res_id, capacity in mapping.items():
                if capacity < 0:
                    raise ValueError(
                        f"{label}[{res_id}] must be non-negative; received {capacity}."
                    )

    def renewable_limit(self, resource_id: int) -> Optional[float]:
        """Return the daily limit for renewable ``resource_id``, or None if unlimited."""
        return self.renewable_capacity.get(resource_id)

    def nonrenewable_limit(self, resource_id: int) -> Optional[float]:
        """Return the horizon-wide limit for non-renewable ``resource_id``, or None."""
        return self.nonrenewable_capacity.get(resource_id)


@dataclass
class CalendarParams:
    """
    Time partitioning parameters.

    Attributes
    ----------
    T_days : int
        Total number of short-term daily periods considered (>= project horizon).
    Y_periods : List[Tuple[int, int]]
        Monthly (or long) periods y as (a_y, b_y) inclusive daily indices (1-based).
        TY_y = [a_y, b_y], with y in {1,...,Yn}. Constraint (19) sums BU_t over t in [a_y, b_y].

    Notes
    -----
    - The model uses days (t) and periods (y). You must set T_days and the period
      intervals coherently.
    """

    T_days: int
    Y_periods: List[Tuple[int, int]]


@dataclass
class IVFTHWeights:
    """
    Weights for the Torabi-Hassini scalarization.

    Attributes
    ----------
    theta1 : float
        Weight for objective 1 membership (makespan).
    theta2 : float
        Weight for objective 2 membership (final cash-flow).
    gamma_tradeoff : float
        Trade-off parameter in [0, 1].
        Objective: maximize
        ``gamma_tradeoff * zeta + (1 - gamma_tradeoff) * (theta1*mu1 + theta2*mu2)``.

    Notes
    -----
    ``mu1`` and ``mu2`` are non-negative and satisfy ``mu1 + mu2 = 1`` (enforced with tolerance).
    """

    theta1: float
    theta2: float
    gamma_tradeoff: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.gamma_tradeoff <= 1.0):
            raise ValueError("gamma_tradeoff must be in [0,1].")
        s = self.theta1 + self.theta2
        if abs(s - 1.0) > 1e-9:
            raise ValueError("theta1 + theta2 must sum to 1.0")


@dataclass
class IVFTHTargets:
    """
    Alpha-level and PiS/NiS targets for IVF-TH membership functions.

    Attributes
    ----------
    alpha_level : float
        Alpha in [0, 1] used in fuzzy-to-crisp conversion in constraints (30)-(34).
    Z1_PIS : float
        Positive ideal solution for Z1 (makespan), i.e., a *small* target (best).
    Z1_NIS : float
        Negative ideal solution for Z1 (worst makespan bound).
    Z2_PIS : float
        Positive ideal solution for Z2 (final cash-flow), i.e., a *large* target (best).
    Z2_NIS : float
        Negative ideal solution for Z2 (worst cash-flow bound).

    Notes
    -----
    These targets can be computed by separate runs (min Z1, max Z2) or set by domain knowledge.
    The membership functions ``mu1`` and ``mu2`` are linear in ``Z1`` and ``Z2``
    using these PiS/NiS anchors.
    """

    alpha_level: float
    Z1_PIS: float
    Z1_NIS: float
    Z2_PIS: float
    Z2_NIS: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.alpha_level <= 1.0):
            raise ValueError("alpha_level must be in [0,1].")
        if not (self.Z1_PIS <= self.Z1_NIS):
            raise ValueError("Expect Z1_PIS <= Z1_NIS (makespan best <= worst).")
        if not (self.Z2_NIS <= self.Z2_PIS):
            raise ValueError("Expect Z2_NIS <= Z2_PIS (cash-flow worst <= best).")
