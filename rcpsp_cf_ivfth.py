"""
Bi-Objective Multi-Mode RCPSP with Cash-Flow under Uncertainty (Extended IVF-TH)
=================================================================================

Implements the full model from:
"A New Bi-Objective Model for Resource-Constrained Project Scheduling and Cash Flow Problems
with Financial Constraints under Uncertainty: A Case Study"
- Activities with multiple modes
- Renewable and non-renewable resources
- Daily and periodic costs
- Initial capital, short/long-term loans, interest on excess cash, delayed payments
- Payments can be delayed at most 1 period
- Credit limit (min CF), loan caps
- Uncertainty on durations and resource requirements via NIVTF numbers
- Extended IVF-TH scalarization of the two objectives:
    Z1 = Cmax (project duration) -> Minimize
    Z2 = CF_{Y_n} (final period cash flow) -> Maximize

Solver:
    Pyomo backend (any MILP solver supported by Pyomo).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable, Union, Any, Set
import math

# Pyomo is used for the MILP
from pyomo.environ import (
    ConcreteModel, Set as PySet, Var, Param, NonNegativeReals, Binary, Integers,
    Objective, Constraint, SolverFactory, value, maximize, minimize, Reals
)
from pyomo.environ import summation
from pyomo.core.base.expression import Expression
from pyomo.opt import TerminationCondition


# ==========================
# 1) Fuzzy number machinery
# ==========================

@dataclass(frozen=True)
class NIVTF:
    """
    Normalized Interval-Valued Triangular Fuzzy number (NIVTF).
    This stores the **lower** and **upper** triangular fuzzy numbers.

    For a triangular fuzzy number (a_o, a_m, a_p) we follow the paper’s normalization:
    - normalized => peak membership = 1 at a_m
    - For NIVTF: lower triangle L = (a_o^L, a_m^L, a_p^L) and upper triangle U = (a_o^U, a_m^U, a_p^U)
      with a_m^L = a_m^U and  a_o^U < a_o^L < a_m^L(=a_m^U) < a_p^L < a_p^U.

    In the paper:
        E1(x) = (a_o + a_m) / 2
        E2(x) = (a_m + a_p) / 2
        EV(x) = (a_o + 2 a_m + a_p) / 4
    applied to lower (L) and upper (U) sides.

    Here we expose:
        E1_L(), E1_U(), E2_L(), E2_U(), EV_L(), EV_U()
    which return floats.

    Input Validation:
        - Ensures the order constraints for NIVTF are satisfied (best-effort checks).
    """
    ao_L: float
    am_L: float
    ap_L: float
    ao_U: float
    am_U: float
    ap_U: float

    def __post_init__(self) -> None:
        # Basic checks for monotonicity and normalization consistency
        if not (self.ao_U < self.ao_L < self.am_L <= self.am_U < self.ap_L < self.ap_U):
            raise ValueError(
                f"NIVTF ordering violated:\n"
                f"Require ao_U < ao_L < am_L(=am_U) <= am_U < ap_L < ap_U\n"
                f"Got: (ao_U, ao_L, am_L, am_U, ap_L, ap_U) = "
                f"({self.ao_U}, {self.ao_L}, {self.am_L}, {self.am_U}, {self.ap_L}, {self.ap_U})"
            )
        if abs(self.am_L - self.am_U) > 1e-12:
            raise ValueError("Normalization requires am_L == am_U for NIVTF.")

    # Expected (Jiménez-style) computations on LOWER side
    def E1_L(self) -> float:
        return 0.5 * (self.ao_L + self.am_L)

    def E2_L(self) -> float:
        return 0.5 * (self.am_L + self.ap_L)

    def EV_L(self) -> float:
        return 0.25 * (self.ao_L + 2.0 * self.am_L + self.ap_L)

    # Expected computations on UPPER side
    def E1_U(self) -> float:
        return 0.5 * (self.ao_U + self.am_U)

    def E2_U(self) -> float:
        return 0.5 * (self.am_U + self.ap_U)

    def EV_U(self) -> float:
        return 0.25 * (self.ao_U + 2.0 * self.am_U + self.ap_U)

    # Convenience: midpoints of expected intervals (optional)
    def E1_mid(self) -> float:
        return 0.5 * (self.E1_L() + self.E1_U())

    def E2_mid(self) -> float:
        return 0.5 * (self.E2_L() + self.E2_U())

    def EV_mid(self) -> float:
        return 0.5 * (self.EV_L() + self.EV_U())


# ======================================
# 2) Instance containers & input schema
# ======================================

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
        Payment amount PA_{i,m} devoted to the period where the activity completes (or next if delayed).
    """
    duration: NIVTF
    renewables: Dict[int, NIVTF]
    nonrenewables: Dict[int, NIVTF]
    payment: float


@dataclass
class Activity:
    """
    Activity with multiple modes and precedence list.
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
        Interest on excess cash (per 30-day period), used as (1+alpha)^30 in model equation.
    beta_delayed_pay : float
        Interest applied to delayed payments (per 30-day period).
    gamma_LTL : float
        Long-term loan interest (per 30-day period).
    delta_STL : float
        Short-term loan interest (per 30-day period).
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
    - The model uses days (t) and periods (y). You must set T_days and the period intervals coherently.
    """
    T_days: int
    Y_periods: List[Tuple[int, int]]


@dataclass
class IVFTHWeights:
    """
    Weights for Torabi–Hassini scalarization.

    Attributes
    ----------
    theta1 : float
        Weight for objective 1 membership (makespan).
    theta2 : float
        Weight for objective 2 membership (final cash-flow).
    gamma_tradeoff : float
        Trade-off parameter ∈ [0,1].
        Objective: maximize gamma_tradeoff * λ  + (1 - gamma_tradeoff) * (theta1*μ1 + theta2*μ2)

    Notes
    -----
    - θ1, θ2 ≥ 0 and θ1 + θ2 = 1 (we check with tolerance).
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
    α-level and PiS/NiS targets for IVF-TH membership functions.

    Attributes
    ----------
    alpha_level : float
        α ∈ [0,1] used in fuzzy-to-crisp conversion in constraints (30)-(34).
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
    - These targets can be computed by separate runs (min Z1, max Z2) or set by domain knowledge.
    - Memberships μ1, μ2 are then linear functions of Z1, Z2 using these PiS/NiS anchors.
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


# ============================
# 3) Example (toy) test data
# ============================

def _tri(a0: float, am: float, ap: float, widen: float = 1.0) -> Tuple[float, float, float, float, float, float]:
    """
    Convenience to create an NIVTF by widening lower/upper supports around am.

    Parameters
    ----------
    a0, am, ap : float
        Base triangle points with a0 < am < ap.
    widen : float
        Controls widening of U and narrowing of L around the same am.

    Returns
    -------
    NIVTF args: (ao_L, am_L, ap_L, ao_U, am_U, ap_U)
    """
    # Upper triangle further away, lower triangle closer, share am
    ao_U = a0 - 0.5 * widen * (am - a0)
    ap_U = ap + 0.5 * widen * (ap - am)

    ao_L = a0 + 0.25 * widen * (am - a0)
    ap_L = ap - 0.25 * widen * (ap - am)

    am_L = am_U = am
    return (ao_L, am_L, ap_L, ao_U, am_U, ap_U)


def build_toy_instance() -> Tuple[Dict[str, Activity], FinanceParams, CalendarParams]:
    """
    Builds a very small instance with:
    - 4 activities (including Start, End dummies) and 2 renewables, 1 non-renewable.
    - 2 long periods (months), 30 days each, total T=60.
    - Two modes per non-dummy activity.

    This is only to demonstrate structure and feasibility. Replace with your real data.
    """
    # Renewables k ∈ {1,2}, Non-renewables l ∈ {1}
    CR_k = {1: 10.0, 2: 8.0}
    CW_l = {1: 50.0}

    # Activities
    A0 = Activity(
        name="Start", predecessors=[],
        modes={
            1: ModeData(
                duration=NIVTF(*_tri(0.0, 0.0, 0.0, widen=0.1)),
                renewables={1: NIVTF(*_tri(0,0,0)), 2: NIVTF(*_tri(0,0,0))},
                nonrenewables={1: NIVTF(*_tri(0,0,0))},
                payment=0.0
            )
        }
    )
    A1 = Activity(
        name="A1", predecessors=["Start"],
        modes={
            1: ModeData(
                duration=NIVTF(*_tri(6, 8, 10, widen=0.6)),
                renewables={1: NIVTF(*_tri(3, 4, 5)), 2: NIVTF(*_tri(0, 1, 2))},
                nonrenewables={1: NIVTF(*_tri(0, 1, 2))},
                payment=3000.0
            ),
            2: ModeData(
                duration=NIVTF(*_tri(5, 7, 9, widen=0.6)),
                renewables={1: NIVTF(*_tri(4, 5, 6)), 2: NIVTF(*_tri(1, 2, 3))},
                nonrenewables={1: NIVTF(*_tri(0, 2, 3))},
                payment=3600.0
            )
        }
    )
    A2 = Activity(
        name="A2", predecessors=["A1"],
        modes={
            1: ModeData(
                duration=NIVTF(*_tri(8, 10, 12, widen=0.6)),
                renewables={1: NIVTF(*_tri(2, 3, 4)), 2: NIVTF(*_tri(1, 1, 2))},
                nonrenewables={1: NIVTF(*_tri(1, 2, 3))},
                payment=4200.0
            ),
            2: ModeData(
                duration=NIVTF(*_tri(6, 9, 11, widen=0.6)),
                renewables={1: NIVTF(*_tri(3, 4, 5)), 2: NIVTF(*_tri(0, 1, 2))},
                nonrenewables={1: NIVTF(*_tri(0, 1, 3))},
                payment=4800.0
            )
        }
    )
    A3 = Activity(
        name="End", predecessors=["A2"],
        modes={
            1: ModeData(
                duration=NIVTF(*_tri(0.0, 0.0, 0.0, widen=0.1)),
                renewables={1: NIVTF(*_tri(0,0,0)), 2: NIVTF(*_tri(0,0,0))},
                nonrenewables={1: NIVTF(*_tri(0,0,0))},
                payment=0.0
            )
        }
    )

    acts = {"Start": A0, "A1": A1, "A2": A2, "End": A3}

    finance = FinanceParams(
        alpha_excess_cash=0.0125,   # α
        beta_delayed_pay=0.10,      # β
        gamma_LTL=0.06,             # γ
        delta_STL=0.075,            # δ
        IC=10000.0,                 # initial capital
        max_LTL=5000.0,
        max_STL=4000.0,
        min_CF=0.0,
        CC_daily_cap=2000.0,        # daily resource cost cap
        CR_k=CR_k,
        CW_l=CW_l
    )

    # Two 30-day periods, total 60 days
    calendar = CalendarParams(
        T_days=60,
        Y_periods=[(1, 30), (31, 60)]
    )

    return acts, finance, calendar


# ==============================================
# 4) Model builder with extended IVF-TH approach
# ==============================================

class RCPSP_CF_IVFTH:
    """
    Builder/solver for the Extended IVF-TH MILP.

    Usage:
        acts, finance, calendar = build_toy_instance()
        ivfth = RCPSP_CF_IVFTH(acts, finance, calendar)

        # Set IVF-TH targets and weights (you can pre-run to compute PiS/NiS or set bounds)
        targets = IVFTHTargets(alpha_level=0.5, Z1_PIS=10.0, Z1_NIS=60.0, Z2_PIS=30000.0, Z2_NIS=0.0)
        weights = IVFTHWeights(theta1=0.5, theta2=0.5, gamma_tradeoff=0.5)

        model = ivfth.build_model(targets, weights)

        results = ivfth.solve(model, solver_name="glpk")
        print(results)
    """

    def __init__(self, activities: Dict[str, Activity], finance: FinanceParams, calendar: CalendarParams) -> None:
        self.activities = activities
        self.finance = finance
        self.calendar = calendar
        self._validate_inputs()

    # ---------- Validation ----------
    def _validate_inputs(self) -> None:
        # Check that Start and End exist and form a DAG
        if "Start" not in self.activities or "End" not in self.activities:
            raise ValueError("Activities must include 'Start' and 'End'.")
        # Every predecessor must be a known activity
        act_names = set(self.activities.keys())
        for a in self.activities.values():
            for p in a.predecessors:
                if p not in act_names:
                    raise ValueError(f"Unknown predecessor '{p}' for activity '{a.name}'.")

        # Period partitions
        if len(self.calendar.Y_periods) == 0:
            raise ValueError("At least one long period (Y_periods) is required.")
        max_day_cover = max(b for (_, b) in self.calendar.Y_periods)
        if self.calendar.T_days < max_day_cover:
            raise ValueError("T_days must cover all Y_periods upper bounds.")

        # Cost vectors must start at index 1..K and 1..L
        # Accept as given.

    # ---------- Helper: fuzzy crisp reductions ----------
    @staticmethod
    def _duration_alpha_L(nivtf: NIVTF, alpha: float, use_lower: bool = True) -> float:
        """
        Return α-weighted expected duration:
            lower:    α*E2_L + (1-α)*E1_L    (used in precedence lower bound)
            upper:    (1-α/2)*E2_L + (α/2)*E1_L or (α/2)*E2_L + (1-α/2)*E1_L (per paper’s (33),(34))
        This helper focuses on the LOWER side used in constraints (30), (33), (34).

        For constraints:
            (30): t_j >= t_i + [ α*E2_L + (1-α)*E1_L ]
            (33): sum t*X_P >= sum (t + α/2 *E2_L + (1-α/2)*E1_L) * X_start
            (34): sum t*X_P <= sum (t + (1-α/2)*E2_L + α/2 *E1_L) * X_start

        We expose E2_L and E1_L so we can combine as needed by callers.
        """
        if use_lower:
            E1L = nivtf.E1_L()
            E2L = nivtf.E2_L()
        else:
            # If ever needed the U side; not used in the main constraints here.
            E1L = nivtf.E1_U()
            E2L = nivtf.E2_U()
        return E1L, E2L

    @staticmethod
    def _res_use_alpha_L(nivtf: NIVTF, alpha: float) -> float:
        """
        Return α-blended expected resource use per day on LOWER side for constraints (31),(32):
            ((1 - α) * E2_L + α * E1_L)
        """
        E1L = nivtf.E1_L()
        E2L = nivtf.E2_L()
        return (1.0 - alpha) * E2L + alpha * E1L

    @staticmethod
    def _dur_mid_EV(nivtf: NIVTF) -> float:
        """
        Convenience: EV midpoint (not used in constraints directly).
        """
        return nivtf.EV_mid()

    # ---------- Build model ----------
    def build_model(self, targets: IVFTHTargets, weights: IVFTHWeights):
        """
        Build the Pyomo model with the extended IVF-TH scalarization.

        Decision variables:
            X[i,m,t]      : 1 if activity i starts at day t in mode m
            Xp[i,m,t]     : 1 if activity i completes at day t in mode m
            XYp[i,m,y,t]  : 1 if activity i completes at day t in period y, in mode m
            BR[k,t]       : renewable resource k used on day t
            WR[l,t]       : non-renewable resource l used on day t
            BU[t]         : daily cost of resource use
            TBU[y]        : total daily cost in period y
            CF[y]         : cash-flow in period y
            LTL           : long-term loan (only y=1)
            STL[y]        : short-term loan in period y
            PA[y]         : payment amount realized in period y
            DP[y]         : delayed payment arriving in period y (from previous)
            Cmax          : project completion time (makespan) via Xp on End

        IVF-TH membership:
            mu1 (for Z1=Cmax), mu2 (for Z2=CF_Yn), lambda_star in [0,1]
        """
        acts = self.activities
        finance = self.finance
        calendar = self.calendar
        alpha = targets.alpha_level

        # Basic index sets
        I = list(acts.keys())
        M_i = {i: list(acts[i].modes.keys()) for i in I}
        T = list(range(1, calendar.T_days + 1))
        Y = list(range(1, len(calendar.Y_periods) + 1))

        # Resource indices (renewable and non-renewable)
        K = sorted(finance.CR_k.keys())  # renewable
        L = sorted(finance.CW_l.keys())  # non-renewable

        # Precompute period bounds TY_y
        TY = {y: (calendar.Y_periods[y - 1][0], calendar.Y_periods[y - 1][1]) for y in Y}

        # ---- Pyomo model ----
        m = ConcreteModel()

        # Sets
        m.I = PySet(initialize=I, ordered=True)
        m.T = PySet(initialize=T, ordered=True)
        m.Y = PySet(initialize=Y, ordered=True)
        m.K = PySet(initialize=K, ordered=True)
        m.L = PySet(initialize=L, ordered=True)
        m.M_i = {i: PySet(initialize=M_i[i], ordered=True) for i in I}

        # Helper: map (y) -> [a_y, b_y]
        m.a = Param(m.Y, initialize={y: TY[y][0] for y in Y}, within=Integers)
        m.b = Param(m.Y, initialize={y: TY[y][1] for y in Y}, within=Integers)

        # --------------------------
        # Decision Variables
        # --------------------------
        # Start / complete variables
        m.X = Var(((i, m_, t) for i in I for m_ in M_i[i] for t in T), within=Binary)
        m.Xp = Var(((i, m_, t) for i in I for m_ in M_i[i] for t in T), within=Binary)
        m.XYp = Var(((i, m_, y, t) for i in I for m_ in M_i[i] for y in Y for t in T), within=Binary)

        # Resources and costs
        m.BR = Var(((k, t) for k in K for t in T), within=NonNegativeReals)
        m.WR = Var(((l_, t) for l_ in L for t in T), within=NonNegativeReals)
        m.BU = Var((t for t in T), within=NonNegativeReals)
        m.TBU = Var((y for y in Y), within=NonNegativeReals)

        # Finance
        m.CF = Var((y for y in Y), within=Reals)  # allow negative intermediate CF if min_CF < 0
        m.LTL = Var(within=NonNegativeReals)
        m.STL = Var((y for y in Y), within=NonNegativeReals)
        m.PA = Var((y for y in Y), within=NonNegativeReals)
        m.DP = Var((y for y in Y), within=NonNegativeReals)

        # Makespan and final cash flow
        m.Cmax = Var(within=NonNegativeReals)

        # IVF-TH membership variables
        m.mu1 = Var(bounds=(0.0, 1.0))
        m.mu2 = Var(bounds=(0.0, 1.0))
        m.lambda_star = Var(bounds=(0.0, 1.0))

        # --------------------------
        # Parameters (cost rates)
        # --------------------------
        m.CR = Param(m.K, initialize=finance.CR_k, within=Reals)
        m.CW = Param(m.L, initialize=finance.CW_l, within=Reals)
        m.CC = Param(initialize=finance.CC_daily_cap, within=Reals)
        m.minCF = Param(initialize=finance.min_CF, within=Reals)
        # Interest parameters
        m.alpha_ex = Param(initialize=finance.alpha_excess_cash, within=Reals)
        m.beta_dp = Param(initialize=finance.beta_delayed_pay, within=Reals)
        m.gamma_LTL = Param(initialize=finance.gamma_LTL, within=Reals)
        m.delta_STL = Param(initialize=finance.delta_STL, within=Reals)
        # Loans & IC
        m.IC = Param(initialize=finance.IC, within=Reals)
        m.maxLTL = Param(initialize=finance.max_LTL, within=Reals)
        m.maxSTL = Param(initialize=finance.max_STL, within=Reals)

        # --------------------------
        # Convenience maps
        # --------------------------
        # Predecessors set P: pairs (i,j) with j successor of i
        P_edges: List[Tuple[str, str]] = []
        for j in I:
            for i_pred in acts[j].predecessors:
                P_edges.append((i_pred, j))
        m.P = PySet(initialize=P_edges, dimen=2)

        # Payments per (i,m)
        PA_im: Dict[Tuple[str, int], float] = {}
        for i in I:
            for mm in M_i[i]:
                PA_im[(i, mm)] = acts[i].modes[mm].payment
        m.PA_im = Param(((i, mm) for i in I for mm in M_i[i]), initialize=PA_im, within=Reals, default=0.0)

        # α-level for fuzzy crisping
        m.alpha = Param(initialize=alpha, within=Reals)

        # --------------------------
        # Objective proxies Z1 and Z2
        # --------------------------
        # Z1 = Cmax
        def cmax_def_rule(_m):
            # Completion time of End activity: sum t * Xp[End, m, t] equals Cmax
            # But End has single mode in our setup. More generally, we can bind:
            return _m.Cmax == sum(t * _m.Xp[("End", mm, t)] for mm in M_i["End"] for t in T)
        m.Cmax_def = Constraint(rule=cmax_def_rule)

        # Z2 = CF at final period Yn
        Yn = Y[-1]

        # --------------------------
        # Constraints
        # --------------------------
        # (7) Each activity starts exactly once in one mode and one day
        def start_once_rule(_m, i):
            return sum(_m.X[(i, mm, t)] for mm in M_i[i] for t in T) == 1
        m.start_once = Constraint(m.I, rule=start_once_rule)

        # (8)/(30) Precedence with fuzzy duration (lower-side α-blend)
        def precedence_rule(_m, i, j):
            # sum t X[j] >= sum (t + alpha*E2_L + (1-alpha)*E1_L) X[i]
            # Build LHS and RHS
            lhs = sum(t * _m.X[(j, mmj, t)] for mmj in M_i[j] for t in T)
            rhs_expr = 0.0
            for mmi in M_i[i]:
                # α-blended expected duration LOWER side:
                E1L, E2L = self._duration_alpha_L(acts[i].modes[mmi].duration, alpha=_m.alpha.value, use_lower=True)
                dur_alpha = _m.alpha * E2L + (1.0 - _m.alpha) * E1L
                rhs_expr += sum((t + dur_alpha) * _m.X[(i, mmi, t)] for t in T)
            return lhs >= rhs_expr
        m.precedence = Constraint(m.P, rule=lambda _m, i, j: precedence_rule(_m, i, j))

        # (9) Renewable resources per day: sum over active activities at day t of r_i,k * X_i(h) ≤ BR[k,t]
        # Active if started at h and running h..(h+dur-1). We approximate with expected length at α-level:
        # We linearize by upper-bounding daily need using the α-blend per (31) with window average.
        # Exact convolution is combinatorial; the paper uses a similar linearization window.
        def renewable_rule(_m, k, t):
            rhs = _m.BR[(k, t)]
            # Sum_{i,m} Sum_{h} r_i,k(α-blend) * X[i,m,h] if activity spans t
            # Use expected span length for indexing window: we conservatively include starts h<=t
            lhs_terms = []
            for i in I:
                for mm in M_i[i]:
                    r_day = self._res_use_alpha_L(acts[i].modes[mm].renewables.get(k, NIVTF(*_tri(0, 0, 0))), alpha=_m.alpha.value)
                    E1L, E2L = self._duration_alpha_L(acts[i].modes[mm].duration, alpha=_m.alpha.value, use_lower=True)
                    dur_mid = (E1L + E2L)  # rough span proxy; safer upper approx
                    # Include start day h if t is within [h, h+ceil(dur_mid)-1]
                    # Implement via big window sum with on/off indicator X[i,mm,h] times r_day
                    # (linear over h)
                    for h in T:
                        if h <= t:
                            lhs_terms.append(r_day * _m.X[(i, mm, h)])
            return sum(lhs_terms) <= rhs
        m.renewable = Constraint(m.K, m.T, rule=renewable_rule)

        # (10) Non-renewable per day (same style as renewables)
        def nonrenewable_rule(_m, l_, t):
            rhs = _m.WR[(l_, t)]
            lhs_terms = []
            for i in I:
                for mm in M_i[i]:
                    r_day = self._res_use_alpha_L(acts[i].modes[mm].nonrenewables.get(l_, NIVTF(*_tri(0, 0, 0))), alpha=_m.alpha.value)
                    for h in T:
                        if h <= t:
                            lhs_terms.append(r_day * _m.X[(i, mm, h)])
            return sum(lhs_terms) <= rhs
        m.nonrenewable = Constraint(m.L, m.T, rule=nonrenewable_rule)

        # (11) Daily resource cost: sum_k CR_k * BR[k,t] + sum_l CW_l * WR[l,t] ≤ BU[t]
        def daily_cost_rule(_m, t):
            return sum(_m.CR[k] * _m.BR[(k, t)] for k in K) + sum(_m.CW[l_] * _m.WR[(l_, t)] for l_ in L) <= _m.BU[t]
        m.daily_cost = Constraint(m.T, rule=daily_cost_rule)

        # (12) BU[t] ≤ CC
        m.daily_cap = Constraint(m.T, rule=lambda _m, t: _m.BU[t] <= _m.CC)

        # (13), (14), (15)-(17): completion date variables and linking to periods
        # (13)+(14): For each activity i,m, sum_t Xp[i,m,t] = sum_t (t + dur_alpha_alt)*X[i,m,t].
        # Paper splits into (13) definition and (14) sum Xp over all m,t == 1. We'll adapt:
        def completion_link_rule(_m, i, mm):
            # Sum_t t * Xp[i,mm,t] ∈ [sum (t + α/2 E2 + (1-α/2)E1) X[i,mm,t], sum (t + (1-α/2)E2 + α/2 E1) X[i,mm,t]]
            E1L, E2L = self._duration_alpha_L(acts[i].modes[mm].duration, alpha=_m.alpha.value, use_lower=True)
            lower = sum((t + 0.5*_m.alpha * E2L + (1.0 - 0.5*_m.alpha) * E1L) * _m.X[(i, mm, t)] for t in T)
            upper = sum((t + (1.0 - 0.5*_m.alpha) * E2L + 0.5*_m.alpha * E1L) * _m.X[(i, mm, t)] for t in T)
            left = sum(t * _m.Xp[(i, mm, t)] for t in T)
            # Two inequalities:
            return (lower <= left, left <= upper)
        m.completion_link_lo = ConstraintList := []
        m.completion_link_hi = ConstraintList := []
        # Pyomo doesn't allow returning tuple lists from rule easily; add two constraints per (i,mm)
        from pyomo.environ import ConstraintList as _ConstraintList
        m.comp_lo = _ConstraintList()
        m.comp_hi = _ConstraintList()
        for i in I:
            for mm in M_i[i]:
                # Build expressions now:
                E1L, E2L = self._duration_alpha_L(acts[i].modes[mm].duration, alpha=alpha, use_lower=True)
                lower = sum((t + 0.5*alpha * E2L + (1.0 - 0.5*alpha) * E1L) * m.X[(i, mm, t)] for t in T)
                upper = sum((t + (1.0 - 0.5*alpha) * E2L + 0.5*alpha * E1L) * m.X[(i, mm, t)] for t in T)
                left = sum(t * m.Xp[(i, mm, t)] for t in T)
                m.comp_lo.add(lower <= left)
                m.comp_hi.add(left <= upper)

        # (14): ensure exactly one completion (over all m,t) per activity
        def complete_once_rule(_m, i):
            return sum(_m.Xp[(i, mm, t)] for mm in M_i[i] for t in T) == 1
        m.complete_once = Constraint(m.I, rule=complete_once_rule)

        # (15)-(17): tie completion day t to period y via XYp
        def XYp_sum_rule(_m, i, mm, t):
            return sum(_m.XYp[(i, mm, y, t)] for y in Y) == _m.Xp[(i, mm, t)]
        m.XYp_sum = Constraint(((i, mm, t) for i in I for mm in M_i[i] for t in T), rule=lambda _m, i, mm, t: XYp_sum_rule(_m, i, mm, t))

        def XYp_lb_rule(_m, i, mm, y, t):
            # TY_{y-1} * XYp ≤ t * Xp   -> with TY_{y-1} meaning lower bound a_y
            ay = _m.a[y]
            return ay * _m.XYp[(i, mm, y, t)] <= t * _m.Xp[(i, mm, t)]
        m.XYp_lb = Constraint(((i, mm, y, t) for i in I for mm in M_i[i] for y in Y for t in T), rule=XYp_lb_rule)

        def XYp_ub_rule(_m, i, mm, y, t):
            by = _m.b[y]
            return t * _m.XYp[(i, mm, y, t)] <= by * _m.XYp[(i, mm, y, t)]
        m.XYp_ub = Constraint(((i, mm, y, t) for i in I for mm in M_i[i] for y in Y for t in T), rule=XYp_ub_rule)

        # (18) Delayed payment balance per period:
        # sum_{i,m,t} PA_im * XYp[i,m,y,t] - PA[y] ≤ DP[y]
        def delayed_pay_rule(_m, y):
            lhs = sum(_m.PA_im[(i, mm)] * _m.XYp[(i, mm, y, t)] for i in I for mm in M_i[i] for t in T)
            return lhs - _m.PA[y] <= _m.DP[y]
        m.delayed_pay = Constraint(m.Y, rule=delayed_pay_rule)

        # (19) TBU[y] = sum_{t in [a_y, b_y]} BU[t]
        def TBU_rule(_m, y):
            ay, by = _m.a[y], _m.b[y]
            return _m.TBU[y] == sum(_m.BU[t] for t in T if (t >= ay and t <= by))
        m.tbu_def = Constraint(m.Y, rule=TBU_rule)

        # (20) CF[1] = IC + STL[1] + LTL + PA[1] - TBU[1]
        def CF1_rule(_m):
            return _m.CF[1] == _m.IC + _m.STL[1] + _m.LTL + _m.PA[1] - _m.TBU[1]
        m.cf1 = Constraint(rule=CF1_rule)

        # (21) CF[y] for y >= 2:
        # CF[y] = STL[y] + CF[y-1]*(1+alpha)^30 + PA[y] + DP[y-1]*(1+beta)^30 - TBU[y] - LTL/(1+gamma)^30 - STL[y-1]/(1+delta)^30
        def CFy_rule(_m, y):
            if y == 1:
                return Constraint.Skip
            # interest factors (assume period=30 days)
            ex = (1.0 + _m.alpha_ex) ** 30
            bd = (1.0 + _m.beta_dp) ** 30
            gL = (1.0 + _m.gamma_LTL) ** 30
            dS = (1.0 + _m.delta_STL) ** 30
            return _m.CF[y] == _m.STL[y] + _m.CF[y - 1] * ex + _m.PA[y] + _m.DP[y - 1] * bd - _m.TBU[y] - _m.LTL / gL - _m.STL[y - 1] / dS
        m.cfy = Constraint(m.Y, rule=CFy_rule)

        # (22) LTL ≤ maxLTL
        m.LTL_cap = Constraint(rule=lambda _m: _m.LTL <= _m.maxLTL)

        # (23) STL[y] ≤ maxSTL
        m.STL_cap = Constraint(m.Y, rule=lambda _m, y: _m.STL[y] <= _m.maxSTL)

        # (24) CF[y] ≥ minCF
        m.CF_floor = Constraint(m.Y, rule=lambda _m, y: _m.CF[y] >= _m.minCF)

        # --------------------------
        # Membership functions and TH scalarization
        # --------------------------
        Z1_PIS, Z1_NIS = targets.Z1_PIS, targets.Z1_NIS  # makespan (minimize)
        Z2_PIS, Z2_NIS = targets.Z2_PIS, targets.Z2_NIS  # final CF (maximize)

        # Membership μ1 for Z1 (smaller is better):
        #   μ1 = 1 if Cmax <= Z1_PIS
        #   μ1 = (Z1_NIS - Cmax) / (Z1_NIS - Z1_PIS)  for Z1_PIS <= Cmax <= Z1_NIS
        #   μ1 = 0 if Cmax >= Z1_NIS
        # Encode with linear constraints:
        #   μ1 ≥ 0
        #   μ1 ≤ 1
        #   μ1 ≤ (Z1_NIS - Cmax) / (Z1_NIS - Z1_PIS)
        #   μ1 ≥ (Z1_NIS - Cmax) / (Z1_NIS - Z1_PIS) - bigM*(binary intervals)  [we avoid binaries; use inequality in the right sense]
        # For TH, it suffices to *upper-bound* μ1 and μ2 and then maximize a convex combination; so:
        denom1 = max(1e-9, (Z1_NIS - Z1_PIS))
        m.mu1_le = Constraint(expr=m.mu1 <= (Z1_NIS - m.Cmax) / denom1)

        # Membership μ2 for Z2 (larger is better):
        #   μ2 = 1 if CF_Yn >= Z2_PIS
        #   μ2 = (CF_Yn - Z2_NIS) / (Z2_PIS - Z2_NIS)  for Z2_NIS <= CF_Yn <= Z2_PIS
        #   μ2 = 0 if CF_Yn <= Z2_NIS
        denom2 = max(1e-9, (Z2_PIS - Z2_NIS))
        m.mu2_le = Constraint(expr=m.mu2 <= (m.CF[Yn] - Z2_NIS) / denom2)

        # λ ≤ μ1, λ ≤ μ2
        m.lambda_le_mu1 = Constraint(expr=m.lambda_star <= m.mu1)
        m.lambda_le_mu2 = Constraint(expr=m.lambda_star <= m.mu2)

        # Final scalarization objective:
        # maximize: gamma * λ  + (1 - gamma) * (θ1 * μ1 + θ2 * μ2)
        gamma = weights.gamma_tradeoff
        theta1 = weights.theta1
        theta2 = weights.theta2

        m.OBJ = Objective(
            expr=gamma * m.lambda_star + (1.0 - gamma) * (theta1 * m.mu1 + theta2 * m.mu2),
            sense=maximize
        )

        return m

    # ---------- Solve ----------
    @staticmethod
    def solve(model: ConcreteModel, solver_name: str = "glpk", timelimit: Optional[int] = None) -> Dict[str, Any]:
        """
        Solve the Pyomo model with the chosen MILP solver.

        Parameters
        ----------
        model : ConcreteModel
            The built model.
        solver_name : str
            Name of the installed solver (e.g., "glpk", "cbc", "gurobi", "cplex").
        timelimit : Optional[int]
            Time limit in seconds (solver dependent).

        Returns
        -------
        Dict[str, Any]
            A summary with objective value, Cmax, final CF, memberships, and status.

        Notes
        -----
        - If your environment lacks the solver, install one or switch to an available one.
        """
        opt = SolverFactory(solver_name)
        if opt is None:
            raise RuntimeError(f"Solver '{solver_name}' is not available.")
        if timelimit is not None:
            try:
                opt.options["timelimit"] = timelimit
            except Exception:
                pass

        res = opt.solve(model, tee=False)
        status = str(res.solver.termination_condition)

        out = {
            "status": status,
            "objective": float(value(model.OBJ)),
            "Cmax": float(value(model.Cmax)),
            "CF_final": float(value(model.CF[max(model.Y)])),
            "mu1": float(value(model.mu1)),
            "mu2": float(value(model.mu2)),
            "lambda": float(value(model.lambda_star)),
        }
        return out


# ==========================
# 5) Example: run the model
# ==========================

if __name__ == "__main__":
    # Build the toy instance (replace with your case-study data)
    activities, finance, calendar = build_toy_instance()

    ivfth = RCPSP_CF_IVFTH(activities, finance, calendar)

    # For a full run you usually pre-compute Z1_PIS (min Cmax) and Z2_PIS (max CF_Yn) by
    # solving two single-objective variants. Here, we set coarse anchors to illustrate usage.
    # Adjust these to your data or run pre-solves to find tight PiS/NiS.
    targets = IVFTHTargets(
        alpha_level=0.5,
        Z1_PIS=10.0,   # optimistic (best) makespan target (days)
        Z1_NIS=60.0,   # pessimistic (worst) makespan bound
        Z2_PIS=30000.0,# optimistic final CF
        Z2_NIS=0.0     # pessimistic final CF
    )

    weights = IVFTHWeights(theta1=0.5, theta2=0.5, gamma_tradeoff=0.5)

    model = ivfth.build_model(targets, weights)

    # Solve (choose a solver you have installed: "glpk", "cbc", "gurobi", "cplex")
    try:
        summary = ivfth.solve(model, solver_name="glpk")
        print("Solve summary:")
        for k, v in summary.items():
            print(f"  {k}: {v}")
    except Exception as e:
        print("Solver run skipped or failed:", e)
        print("Model built successfully. Install a MILP solver (e.g., GLPK/CBC/Gurobi/CPLEX) to solve.")
