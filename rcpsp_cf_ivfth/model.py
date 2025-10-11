"""
Main MILP model for Bi-Objective Multi-Mode RCPSP with Cash-Flow under Uncertainty.

This module contains the RCPSP_CF_IVFTH class that builds and solves the 
Extended IVF-TH scalarization model using Pyomo.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Any, Union

from collections import deque

import logging
from time import perf_counter

# Pyomo is used for the MILP
from pyomo.environ import (
    ConcreteModel, Set as PySet, Var, Param, NonNegativeReals, Binary, Integers,
    Objective, Constraint, SolverFactory, value, maximize, Reals
)
from pyomo.environ import ConstraintList as _ConstraintList

from .data import Activity, FinanceParams, CalendarParams, IVFTHTargets, IVFTHWeights
from .fuzzy import NIVTF, create_triangle


class RCPSP_CF_IVFTH:
    """
    Builder/solver for the Extended IVF-TH MILP.

    This class implements the full model from:
    "A New Bi-Objective Model for Resource-Constrained Project Scheduling and Cash Flow Problems
    with Financial Constraints under Uncertainty: A Case Study"
    
    Features:
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

    def __init__(
        self,
        activities: Dict[str, Activity],
        finance: FinanceParams,
        calendar: CalendarParams,
        *,
        logging_enabled: bool = False,
        log_level: Union[int, str] = "INFO",
        log_file: Optional[str] = None,
        strict_validation: bool = False,
    ) -> None:
        self.activities = activities
        self.finance = finance
        self.calendar = calendar
        self._strict_validation = strict_validation

        self._logger = self._configure_logger(logging_enabled, log_level, log_file)
        self._logging_enabled = logging_enabled
        self._log_level = log_level
        self._log_file = log_file

        validation_start = perf_counter()
        self._log_info(
            "input_validation_start",
            activity_count=len(self.activities),
            period_count=len(self.calendar.Y_periods),
        )
        self._validate_inputs()
        self._log_info(
            "input_validation_complete",
            elapsed_seconds=round(perf_counter() - validation_start, 4),
        )

    def configure_logging(
        self,
        *,
        enabled: bool,
        log_level: Union[int, str] = "INFO",
        log_file: Optional[str] = None,
    ) -> None:
        """
        Reconfigure logging for the current instance.
        """
        self._logger = self._configure_logger(enabled, log_level, log_file)
        self._logging_enabled = enabled
        self._log_level = log_level
        self._log_file = log_file

    # ---------- Logging helpers ----------
    def _configure_logger(
        self,
        enabled: bool,
        log_level: Union[int, str],
        log_file: Optional[str],
    ) -> logging.Logger:
        logger_name = f"rcpsp_cf_ivfth.model.{id(self)}"
        logger = logging.getLogger(logger_name)
        logger.handlers.clear()

        if not enabled:
            logger.addHandler(logging.NullHandler())
            logger.setLevel(logging.CRITICAL)
            logger.propagate = False
            return logger

        level = self._resolve_log_level(log_level)
        logger.setLevel(level)

        formatter = logging.Formatter(
            fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        if log_file:
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        logger.propagate = False
        return logger

    @staticmethod
    def _resolve_log_level(log_level: Union[int, str]) -> int:
        if isinstance(log_level, int):
            return log_level
        if isinstance(log_level, str):
            level = logging.getLevelName(log_level.upper())
            if isinstance(level, int):
                return level
        return logging.INFO

    def _log(self, level: int, message: str, **fields: Any) -> None:
        if not getattr(self, "_logging_enabled", False):
            return
        if fields:
            kv = " ".join(f"{key}={value}" for key, value in fields.items())
            message = f"{message} | {kv}"
        self._logger.log(level, message)

    def _log_debug(self, message: str, **fields: Any) -> None:
        self._log(logging.DEBUG, message, **fields)

    def _log_info(self, message: str, **fields: Any) -> None:
        self._log(logging.INFO, message, **fields)

    def _log_warning(self, message: str, **fields: Any) -> None:
        self._log(logging.WARNING, message, **fields)

    @staticmethod
    def _component_size(component: Any) -> Optional[int]:
        try:
            return len(component)
        except TypeError:
            try:
                return len(component.index_set())
            except Exception:
                return None

    @staticmethod
    def _model_statistics(model: ConcreteModel) -> Dict[str, int]:
        stats: Dict[str, int] = {}
        try:
            stats["variables"] = sum(
                1 for _ in model.component_data_objects(Var, active=True)
            )
        except Exception:
            pass
        try:
            stats["constraints"] = sum(
                1 for _ in model.component_data_objects(Constraint, active=True)
            )
        except Exception:
            pass
        return stats

    # ---------- Validation ----------
    def _validate_inputs(self) -> None:
        strict = getattr(self, "_strict_validation", False)

        def warn_or_raise(code: str, message: str, suggestion: Optional[str] = None, **fields: Any) -> None:
            payload = dict(fields)
            if suggestion:
                payload["suggestion"] = suggestion
            self._log_warning(code, **payload)
            if strict:
                detail = message
                if suggestion:
                    detail = f"{detail} Suggested fix: {suggestion}"
                raise ValueError(detail)

        # Basic terminal nodes check
        if "Start" not in self.activities or "End" not in self.activities:
            self._log_warning(
                "validation_missing_terminal_nodes",
                has_start="Start" in self.activities,
                has_end="End" in self.activities,
            )
            raise ValueError("Activities must include 'Start' and 'End'. Add terminal nodes to the instance definition.")

        # Every predecessor must be a known activity
        act_names = set(self.activities.keys())
        self._log_debug("validation_activity_summary", activity_count=len(act_names))
        for a in self.activities.values():
            for p in a.predecessors:
                if p not in act_names:
                    self._log_warning(
                        "validation_unknown_predecessor",
                        activity=a.name,
                        predecessor=p,
                    )
                    raise ValueError(
                        f"Unknown predecessor '{p}' for activity '{a.name}'. "
                        "Ensure all predecessors reference existing activities."
                    )

        if len(self.calendar.Y_periods) == 0:
            self._log_warning("validation_no_periods_defined")
            raise ValueError("At least one long period (Y_periods) is required.")

        max_day_cover = self._validate_periods(warn_or_raise)
        if self.calendar.T_days < max_day_cover:
            self._log_warning(
                "validation_period_out_of_bounds",
                t_days=self.calendar.T_days,
                max_period_end=max_day_cover,
            )
            raise ValueError(
                f"T_days must cover all Y_periods upper bounds (max {max_day_cover}); current T_days={self.calendar.T_days}. "
                "Increase the planning horizon or adjust the period definitions."
            )

        cycle = self._detect_precedence_cycle()
        if cycle:
            cycle_path = " -> ".join(cycle)
            self._log_warning("validation_cycle_detected", cycle=cycle_path)
            raise ValueError(
                f"Precedence constraints form a cycle: {cycle_path}. "
                "Remove circular dependencies in the activity graph."
            )

        self._validate_nivtf_values()
        self._validate_resource_caps(warn_or_raise)
        self._validate_financial_feasibility(warn_or_raise)

        self._log_debug(
            "validation_complete_success",
            renewable_costs=len(self.finance.CR_k),
            nonrenewable_costs=len(self.finance.CW_l),
            strict=strict,
        )

    # ---------- Helper: fuzzy crisp reductions ----------
    @staticmethod
    def _duration_alpha_L(nivtf: NIVTF, alpha: float, use_lower: bool = True) -> Tuple[float, float]:
        """
        Return alpha-weighted expected duration components E1_L and E2_L:
            lower:    alpha*E2_L + (1-alpha)*E1_L    (used in precedence lower bound)
            upper:    (1-alpha/2)*E2_L + (alpha/2)*E1_L or (alpha/2)*E2_L + (1-alpha/2)*E1_L (per paper's (33),(34))
        This helper focuses on the LOWER side used in constraints (30), (33), (34).

        For constraints:
            (30): t_j >= t_i + [ alpha*E2_L + (1-alpha)*E1_L ]
            (33): sum t*X_P >= sum (t + alpha/2 *E2_L + (1-alpha/2)*E1_L) * X_start
            (34): sum t*X_P <= sum (t + (1-alpha/2)*E2_L + alpha/2 *E1_L) * X_start

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
        Return alpha-blended expected resource use per day on LOWER side for constraints (31),(32):
            ((1 - alpha) * E2_L + alpha * E1_L)
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

    @staticmethod
    def _nivtf_values(nivtf: NIVTF) -> Tuple[float, float, float, float, float, float]:
        return (
            nivtf.ao_L,
            nivtf.am_L,
            nivtf.ap_L,
            nivtf.ao_U,
            nivtf.am_U,
            nivtf.ap_U,
        )

    @staticmethod
    def _max_nivtf_value(nivtf: NIVTF) -> float:
        return max(RCPSP_CF_IVFTH._nivtf_values(nivtf))

    @staticmethod
    def _min_nivtf_value(nivtf: NIVTF) -> float:
        return min(RCPSP_CF_IVFTH._nivtf_values(nivtf))

    def _detect_precedence_cycle(self) -> Optional[List[str]]:
        graph: Dict[str, List[str]] = {name: [] for name in self.activities}
        for successor, activity in self.activities.items():
            for predecessor in activity.predecessors:
                if predecessor in graph:
                    graph[predecessor].append(successor)

        visited: Dict[str, str] = {}
        stack: Dict[str, bool] = {}
        parent: Dict[str, Optional[str]] = {}
        cycle: Optional[List[str]] = None

        def dfs(node: str) -> None:
            nonlocal cycle
            visited[node] = "gray"
            stack[node] = True
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    parent[neighbor] = node
                    dfs(neighbor)
                    if cycle:
                        return
                elif stack.get(neighbor):
                    path = [neighbor]
                    current = node
                    while current is not None and current != neighbor:
                        path.append(current)
                        current = parent.get(current)
                    path.append(neighbor)
                    path.reverse()
                    cycle = path
                    return
            stack[node] = False
            visited[node] = "black"

        for node in graph:
            if node not in visited:
                parent[node] = None
                dfs(node)
                if cycle:
                    break
        return cycle

    def _validate_nivtf_values(self) -> None:
        tolerance = -1e-6
        for act_name, activity in self.activities.items():
            for mode_id, mode in activity.modes.items():
                duration_min = self._min_nivtf_value(mode.duration)
                if duration_min < tolerance:
                    self._log_warning(
                        "validation_negative_duration",
                        activity=act_name,
                        mode=mode_id,
                        min_value=duration_min,
                    )
                    raise ValueError(
                        f"Activity '{act_name}' mode {mode_id} has a negative duration bound ({duration_min}). "
                        "Adjust the NIVTF parameters to be non-negative."
                    )
                elif duration_min < 0:
                    self._log_debug(
                        "validation_duration_clamped",
                        activity=act_name,
                        mode=mode_id,
                        min_value=duration_min,
                    )
                for res_id, niv in mode.renewables.items():
                    res_min = self._min_nivtf_value(niv)
                    if res_min < tolerance:
                        self._log_warning(
                            "validation_negative_renewable",
                            activity=act_name,
                            mode=mode_id,
                            resource=res_id,
                            min_value=res_min,
                        )
                        raise ValueError(
                            f"Renewable resource {res_id} for activity '{act_name}' mode {mode_id} "
                            f"has negative demand ({res_min}). Ensure resource demands are non-negative."
                        )
                    elif res_min < 0:
                        self._log_debug(
                            "validation_renewable_clamped",
                            activity=act_name,
                            mode=mode_id,
                            resource=res_id,
                            min_value=res_min,
                        )
                for res_id, niv in mode.nonrenewables.items():
                    res_min = self._min_nivtf_value(niv)
                    if res_min < tolerance:
                        self._log_warning(
                            "validation_negative_nonrenewable",
                            activity=act_name,
                            mode=mode_id,
                            resource=res_id,
                            min_value=res_min,
                        )
                        raise ValueError(
                            f"Non-renewable resource {res_id} for activity '{act_name}' mode {mode_id} "
                            f"has negative demand ({res_min}). Ensure resource demands are non-negative."
                        )
                    elif res_min < 0:
                        self._log_debug(
                            "validation_nonrenewable_clamped",
                            activity=act_name,
                            mode=mode_id,
                            resource=res_id,
                            min_value=res_min,
                        )

    def _estimate_mode_daily_cost(self, mode: ModeData) -> float:
        cost = 0.0
        for res_id, niv in mode.renewables.items():
            unit_cost = self.finance.CR_k.get(res_id, 0.0)
            cost += unit_cost * self._max_nivtf_value(niv)
        for res_id, niv in mode.nonrenewables.items():
            unit_cost = self.finance.CW_l.get(res_id, 0.0)
            cost += unit_cost * self._max_nivtf_value(niv)
        return cost

    def _estimate_mode_total_cost(self, mode: ModeData) -> float:
        daily_cost = self._estimate_mode_daily_cost(mode)
        duration = max(1.0, self._max_nivtf_value(mode.duration))
        return daily_cost * duration

    def _estimate_min_project_resource_cost(self) -> float:
        total = 0.0
        for activity in self.activities.values():
            mode_costs = []
            for mode in activity.modes.values():
                mode_costs.append(self._estimate_mode_total_cost(mode))
            if mode_costs:
                total += min(mode_costs)
        return total

    def _available_funds(self) -> float:
        periods = len(self.calendar.Y_periods)
        return (
            self.finance.IC
            + self.finance.max_LTL
            + periods * self.finance.max_STL
        )

    def _validate_resource_caps(self, warn_or_raise) -> None:
        cap = self.finance.CC_daily_cap
        if cap is None or cap <= 0:
            return

        problematic: List[Tuple[str, int, float]] = []
        for act_name, activity in self.activities.items():
            for mode_id, mode in activity.modes.items():
                estimated_cost = self._estimate_mode_daily_cost(mode)
                if estimated_cost > cap + 1e-6:
                    problematic.append((act_name, mode_id, estimated_cost))

        if problematic:
            act, mode_id, estimated_cost = max(problematic, key=lambda item: item[2])
            warn_or_raise(
                "validation_resource_cost_cap",
                f"Estimated daily resource cost {estimated_cost:.2f} exceeds CC_daily_cap ({cap:.2f}) "
                f"for activity '{act}' mode {mode_id}.",
                suggestion="Increase FinanceParams.CC_daily_cap or reduce resource usage for that mode.",
                activity=act,
                mode=mode_id,
                estimated_cost=round(estimated_cost, 4),
                cap=cap,
            )

    def _validate_financial_feasibility(self, warn_or_raise) -> None:
        available = self._available_funds()
        minimum_required = self._estimate_min_project_resource_cost()
        if minimum_required > available + 1e-6:
            warn_or_raise(
                "validation_finance_shortfall",
                f"Estimated minimum resource spending ({minimum_required:.2f}) exceeds available funds ({available:.2f}).",
                suggestion="Increase initial capital or loan limits, or reduce resource demands.",
                estimated_cost=round(minimum_required, 4),
                available_funds=available,
            )

    def _validate_periods(self, warn_or_raise) -> int:
        periods = list(self.calendar.Y_periods)
        sorted_periods = sorted(periods, key=lambda seg: seg[0])
        if sorted_periods != periods:
            warn_or_raise(
                "validation_periods_not_sorted",
                "Accounting periods are not sorted by start day.",
                suggestion="Order CalendarParams.Y_periods by ascending start day.",
                provided=periods,
            )

        prev_end: Optional[int] = None
        prev_start: Optional[int] = None
        max_end = 0
        for idx, (start, end) in enumerate(sorted_periods, start=1):
            if start > end:
                raise ValueError(
                    f"Period {idx} has start {start} greater than end {end}. "
                    "Ensure each Y_periods entry is (start, end) with start <= end."
                )
            if prev_end is not None:
                if start <= prev_end:
                    raise ValueError(
                        f"Periods {idx-1} ({prev_start}, {prev_end}) and {idx} ({start}, {end}) overlap. "
                        "Adjust Y_periods to eliminate overlap."
                    )
                if start != prev_end + 1:
                    warn_or_raise(
                        "validation_periods_not_contiguous",
                        f"Period {idx} starts at day {start} but previous period ended at day {prev_end}.",
                        suggestion="Ensure Y_periods are contiguous (next start = previous end + 1).",
                        previous_end=prev_end,
                        next_start=start,
                    )
            prev_start = start
            prev_end = end
            max_end = max(max_end, end)
        return max_end

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
        build_start = perf_counter()

        acts = self.activities
        finance = self.finance
        calendar = self.calendar
        alpha = targets.alpha_level

        if not (0.0 <= alpha <= 1.0):
            self._log_warning("validation_alpha_out_of_bounds", alpha=alpha)
            raise ValueError(
                f"alpha_level must be within [0, 1]; received {alpha}. "
                "Adjust IVFTHTargets.alpha_level accordingly."
            )
        if not (targets.Z1_PIS < targets.Z1_NIS):
            self._log_warning(
                "validation_target_order_makespan",
                z1_pis=targets.Z1_PIS,
                z1_nis=targets.Z1_NIS,
            )
            raise ValueError(
                f"Z1_PIS ({targets.Z1_PIS}) must be strictly less than Z1_NIS ({targets.Z1_NIS}). "
                "Compute realistic best/worst makespan anchors before building the model."
            )
        if not (targets.Z2_NIS < targets.Z2_PIS):
            self._log_warning(
                "validation_target_order_cashflow",
                z2_pis=targets.Z2_PIS,
                z2_nis=targets.Z2_NIS,
            )
            raise ValueError(
                f"Z2_NIS ({targets.Z2_NIS}) must be strictly less than Z2_PIS ({targets.Z2_PIS}). "
                "Set achievable cash-flow anchors (worst < best)."
            )

        self._log_info(
            "model_build_start",
            activities=len(acts),
            total_modes=sum(len(a.modes) for a in acts.values()),
            horizon_days=calendar.T_days,
            periods=len(calendar.Y_periods),
            renewable_resources=len(finance.CR_k),
            nonrenewable_resources=len(finance.CW_l),
            alpha_level=alpha,
        )

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
        m.M_i = PySet(m.I, initialize=lambda _m, i: M_i[i], ordered=True)
        self._log_debug(
            "model_sets_defined",
            activities=len(I),
            time_points=len(T),
            periods=len(Y),
            renewable=len(K),
            nonrenewable=len(L),
        )

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
        self._log_debug(
            "decision_variables_created",
            X=self._component_size(m.X),
            Xp=self._component_size(m.Xp),
            XYp=self._component_size(m.XYp),
            BR=self._component_size(m.BR),
            WR=self._component_size(m.WR),
            BU=self._component_size(m.BU),
            TBU=self._component_size(m.TBU),
            CF=self._component_size(m.CF),
            STL=self._component_size(m.STL),
            PA=self._component_size(m.PA),
            DP=self._component_size(m.DP),
        )

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
        self._log_debug(
            "parameters_initialized",
            cost_cap=finance.CC_daily_cap,
            min_cf=finance.min_CF,
        )

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

        # alpha-level for fuzzy crisping
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
        self._log_debug("constraint_start_once_defined", count=len(m.start_once))

        # (8)/(30) Precedence with fuzzy duration (lower-side alpha-blend)
        def precedence_rule(_m, i, j):
            # sum t X[j] >= sum (t + alpha*E2_L + (1-alpha)*E1_L) X[i]
            # Build LHS and RHS
            lhs = sum(t * _m.X[(j, mmj, t)] for mmj in M_i[j] for t in T)
            rhs_expr = 0.0
            for mmi in M_i[i]:
                # alpha-blended expected duration LOWER side:
                E1L, E2L = self._duration_alpha_L(acts[i].modes[mmi].duration, alpha=_m.alpha.value, use_lower=True)
                dur_alpha = _m.alpha * E2L + (1.0 - _m.alpha) * E1L
                rhs_expr += sum((t + dur_alpha) * _m.X[(i, mmi, t)] for t in T)
            return lhs >= rhs_expr
        m.precedence = Constraint(m.P, rule=lambda _m, i, j: precedence_rule(_m, i, j))
        self._log_debug("constraint_precedence_defined", arcs=len(P_edges))

        # (9) Renewable resources per day: sum over active activities at day t of r_i,k * X_i(h) <= BR[k,t]
        # Active if started at h and running h..(h+dur-1). We approximate with expected length at alpha-level:
        # We linearize by upper-bounding daily need using the alpha-blend per (31) with window average.
        # Exact convolution is combinatorial; the paper uses a similar linearization window.
        def renewable_rule(_m, k, t):
            rhs = _m.BR[(k, t)]
            # Sum_{i,m} Sum_{h} r_i,k(alpha-blend) * X[i,m,h] if activity spans t
            # Use expected span length for indexing window: we conservatively include starts h<=t
            lhs_terms = []
            for i in I:
                for mm in M_i[i]:
                    r_day = self._res_use_alpha_L(acts[i].modes[mm].renewables.get(k, NIVTF(*create_triangle(0, 0, 0))), alpha=_m.alpha.value)
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
        self._log_debug(
            "constraint_renewable_capacity_defined",
            count=len(m.renewable),
        )

        # (10) Non-renewable per day (same style as renewables)
        def nonrenewable_rule(_m, l_, t):
            rhs = _m.WR[(l_, t)]
            lhs_terms = []
            for i in I:
                for mm in M_i[i]:
                    r_day = self._res_use_alpha_L(acts[i].modes[mm].nonrenewables.get(l_, NIVTF(*create_triangle(0, 0, 0))), alpha=_m.alpha.value)
                    for h in T:
                        if h <= t:
                            lhs_terms.append(r_day * _m.X[(i, mm, h)])
            return sum(lhs_terms) <= rhs
        m.nonrenewable = Constraint(m.L, m.T, rule=nonrenewable_rule)
        self._log_debug(
            "constraint_nonrenewable_capacity_defined",
            count=len(m.nonrenewable),
        )

        # (11) Daily resource cost: sum_k CR_k * BR[k,t] + sum_l CW_l * WR[l,t] <= BU[t]
        def daily_cost_rule(_m, t):
            return sum(_m.CR[k] * _m.BR[(k, t)] for k in K) + sum(_m.CW[l_] * _m.WR[(l_, t)] for l_ in L) <= _m.BU[t]
        m.daily_cost = Constraint(m.T, rule=daily_cost_rule)
        self._log_debug("constraint_daily_cost_defined", count=len(m.daily_cost))

        # (12) BU[t] <= CC
        m.daily_cap = Constraint(m.T, rule=lambda _m, t: _m.BU[t] <= _m.CC)
        self._log_debug("constraint_daily_cap_defined", count=len(m.daily_cap))

        # (13), (14), (15)-(17): completion date variables and linking to periods
        # (13)+(14): For each activity i,m, sum_t Xp[i,m,t] = sum_t (t + dur_alpha_alt)*X[i,m,t].
        # Paper splits into (13) definition and (14) sum Xp over all m,t == 1. We'll adapt:
        def completion_link_rule(_m, i, mm):
            # Sum_t t * Xp[i,mm,t] in [sum (t + alpha/2 E2 + (1-alpha/2)E1) X[i,mm,t], sum (t + (1-alpha/2)E2 + alpha/2 E1) X[i,mm,t]]
            E1L, E2L = self._duration_alpha_L(acts[i].modes[mm].duration, alpha=_m.alpha.value, use_lower=True)
            lower = sum((t + 0.5*_m.alpha * E2L + (1.0 - 0.5*_m.alpha) * E1L) * _m.X[(i, mm, t)] for t in T)
            upper = sum((t + (1.0 - 0.5*_m.alpha) * E2L + 0.5*_m.alpha * E1L) * _m.X[(i, mm, t)] for t in T)
            left = sum(t * _m.Xp[(i, mm, t)] for t in T)
            # Two inequalities:
            return (lower <= left, left <= upper)
        # Pyomo doesn't allow returning tuple lists from rule easily; add two constraints per (i,mm)
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
        self._log_debug(
            "constraint_completion_link_defined",
            lower_bounds=len(m.comp_lo),
            upper_bounds=len(m.comp_hi),
        )

        # (14): ensure exactly one completion (over all m,t) per activity
        def complete_once_rule(_m, i):
            return sum(_m.Xp[(i, mm, t)] for mm in M_i[i] for t in T) == 1
        m.complete_once = Constraint(m.I, rule=complete_once_rule)

        # (15)-(17): tie completion day t to period y via XYp
        def XYp_sum_rule(_m, i, mm, t):
            return sum(_m.XYp[(i, mm, y, t)] for y in Y) == _m.Xp[(i, mm, t)]
        m.XYp_sum = Constraint(((i, mm, t) for i in I for mm in M_i[i] for t in T), rule=lambda _m, i, mm, t: XYp_sum_rule(_m, i, mm, t))
        self._log_debug(
            "constraint_completion_period_mapping_defined",
            count=len(m.XYp_sum),
        )

        def XYp_lb_rule(_m, i, mm, y, t):
            # TY_{y-1} * XYp <= t * Xp   -> with TY_{y-1} meaning lower bound a_y
            ay = _m.a[y]
            return ay * _m.XYp[(i, mm, y, t)] <= t * _m.Xp[(i, mm, t)]
        m.XYp_lb = Constraint(((i, mm, y, t) for i in I for mm in M_i[i] for y in Y for t in T), rule=XYp_lb_rule)

        def XYp_ub_rule(_m, i, mm, y, t):
            by = _m.b[y]
            return t * _m.XYp[(i, mm, y, t)] <= by * _m.XYp[(i, mm, y, t)]
        m.XYp_ub = Constraint(((i, mm, y, t) for i in I for mm in M_i[i] for y in Y for t in T), rule=XYp_ub_rule)
        self._log_debug(
            "constraint_completion_period_bounds_defined",
            lower=len(m.XYp_lb),
            upper=len(m.XYp_ub),
        )

        # (18) Delayed payment balance per period:
        # sum_{i,m,t} PA_im * XYp[i,m,y,t] - PA[y] <= DP[y]
        def delayed_pay_rule(_m, y):
            lhs = sum(_m.PA_im[(i, mm)] * _m.XYp[(i, mm, y, t)] for i in I for mm in M_i[i] for t in T)
            return lhs - _m.PA[y] <= _m.DP[y]
        m.delayed_pay = Constraint(m.Y, rule=delayed_pay_rule)
        self._log_debug("constraint_delayed_pay_defined", count=len(m.delayed_pay))

        # (19) TBU[y] = sum_{t in [a_y, b_y]} BU[t]
        def TBU_rule(_m, y):
            ay, by = _m.a[y], _m.b[y]
            return _m.TBU[y] == sum(_m.BU[t] for t in T if (t >= ay and t <= by))
        m.tbu_def = Constraint(m.Y, rule=TBU_rule)
        self._log_debug("constraint_period_cost_defined", count=len(m.tbu_def))

        # (20) CF[1] = IC + STL[1] + LTL + PA[1] - TBU[1]
        def CF1_rule(_m):
            return _m.CF[1] == _m.IC + _m.STL[1] + _m.LTL + _m.PA[1] - _m.TBU[1]
        m.cf1 = Constraint(rule=CF1_rule)
        self._log_debug("constraint_cashflow_initial_defined", count=1)

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
        self._log_debug(
            "constraint_cashflow_dynamic_defined",
            count=len(m.cfy) - 1,
        )

        # (22) LTL <= maxLTL
        m.LTL_cap = Constraint(rule=lambda _m: _m.LTL <= _m.maxLTL)

        # (23) STL[y] <= maxSTL
        m.STL_cap = Constraint(m.Y, rule=lambda _m, y: _m.STL[y] <= _m.maxSTL)

        # (24) CF[y] >= minCF
        m.CF_floor = Constraint(m.Y, rule=lambda _m, y: _m.CF[y] >= _m.minCF)
        self._log_debug(
            "constraint_financial_caps_defined",
            ltl=1,
            stl=len(m.STL_cap),
            cf=len(m.CF_floor),
        )

        # --------------------------
        # Membership functions and TH scalarization
        # --------------------------
        Z1_PIS, Z1_NIS = targets.Z1_PIS, targets.Z1_NIS  # makespan (minimize)
        Z2_PIS, Z2_NIS = targets.Z2_PIS, targets.Z2_NIS  # final CF (maximize)

        # Membership mu1 for Z1 (smaller is better):
        #   mu1 = 1 if Cmax <= Z1_PIS
        #   mu1 = (Z1_NIS - Cmax) / (Z1_NIS - Z1_PIS)  for Z1_PIS <= Cmax <= Z1_NIS
        #   mu1 = 0 if Cmax >= Z1_NIS
        # Encode with linear constraints:
        #   mu1 >= 0
        #   mu1 <= 1
        #   mu1 <= (Z1_NIS - Cmax) / (Z1_NIS - Z1_PIS)
        #   mu1 >= (Z1_NIS - Cmax) / (Z1_NIS - Z1_PIS) - bigM*(binary intervals)  [we avoid binaries; use inequality in the right sense]
        # For TH, it suffices to *upper-bound* mu1 and mu2 and then maximize a convex combination; so:
        denom1 = max(1e-9, (Z1_NIS - Z1_PIS))
        m.mu1_le = Constraint(expr=m.mu1 <= (Z1_NIS - m.Cmax) / denom1)

        # Membership mu2 for Z2 (larger is better):
        #   mu2 = 1 if CF_Yn >= Z2_PIS
        #   mu2 = (CF_Yn - Z2_NIS) / (Z2_PIS - Z2_NIS)  for Z2_NIS <= CF_Yn <= Z2_PIS
        #   mu2 = 0 if CF_Yn <= Z2_NIS
        denom2 = max(1e-9, (Z2_PIS - Z2_NIS))
        m.mu2_le = Constraint(expr=m.mu2 <= (m.CF[Yn] - Z2_NIS) / denom2)

        # lambda <= mu1, lambda <= mu2
        m.lambda_le_mu1 = Constraint(expr=m.lambda_star <= m.mu1)
        m.lambda_le_mu2 = Constraint(expr=m.lambda_star <= m.mu2)
        self._log_debug(
            "constraint_membership_defined",
            mu1_bounds=1,
            mu2_bounds=1,
        )

        # Final scalarization objective:
        # maximize: gamma * lambda  + (1 - gamma) * (theta1 * mu1 + theta2 * mu2)
        gamma = weights.gamma_tradeoff
        theta1 = weights.theta1
        theta2 = weights.theta2

        m.OBJ = Objective(
            expr=gamma * m.lambda_star + (1.0 - gamma) * (theta1 * m.mu1 + theta2 * m.mu2),
            sense=maximize
        )

        stats = self._model_statistics(m)
        self._log_info(
            "model_build_complete",
            elapsed_seconds=round(perf_counter() - build_start, 4),
            variables=stats.get("variables"),
            constraints=stats.get("constraints"),
        )

        return m

    # ---------- Solve ----------
    def solve(
        self,
        model: ConcreteModel,
        solver_name: str = "glpk",
        timelimit: Optional[int] = None,
        tee: bool = False,
    ) -> Dict[str, Any]:
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
        tee : bool
            When True, stream solver output to stdout.

        Returns
        -------
        Dict[str, Any]
            A summary with objective value, Cmax, final CF, memberships, and status.

        Notes
        -----
        - If your environment lacks the solver, install one or switch to an available one.
        """
        solve_start = perf_counter()
        stats = self._model_statistics(model)
        self._log_info(
            "solver_start",
            solver=solver_name,
            timelimit=timelimit,
            variables=stats.get("variables"),
            constraints=stats.get("constraints"),
        )

        opt = SolverFactory(solver_name)
        if opt is None or not opt.available(exception_flag=False):
            self._log_warning("solver_unavailable", solver=solver_name)
            raise RuntimeError(f"Solver '{solver_name}' is not available.")
        if timelimit is not None:
            try:
                opt.options["timelimit"] = timelimit
            except Exception:
                pass

        res = opt.solve(model, tee=tee)
        termination = str(res.solver.termination_condition)
        status = str(res.solver.status)
        solver_time = getattr(res.solver, "time", None)
        if solver_time is None:
            solver_time = getattr(res.solver, "wallclock_time", None)

        if termination.lower() != "optimal":
            self._log_warning(
                "solver_termination_non_optimal",
                termination=termination,
                status=status,
            )
        else:
            self._log_info(
                "solver_termination",
                termination=termination,
                status=status,
            )

        elapsed = round(perf_counter() - solve_start, 4)
        self._log_info(
            "solver_complete",
            elapsed_seconds=elapsed,
            solver_time=solver_time,
        )

        out = {
            "status": termination,
            "objective": float(value(model.OBJ)),
            "Cmax": float(value(model.Cmax)),
            "CF_final": float(value(model.CF[max(model.Y)])),
            "mu1": float(value(model.mu1)),
            "mu2": float(value(model.mu2)),
            "lambda": float(value(model.lambda_star)),
            "solver_status": status,
            "solver_time": solver_time,
        }
        return out

    def extract_solution(
        self,
        model: ConcreteModel,
        *,
        solver_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Extract a structured solution dictionary from a solved model.

        Parameters
        ----------
        model : ConcreteModel
            Solved Pyomo model.
        solver_metadata : Optional[Dict[str, Any]]
            Optional metadata (e.g., return value from :meth:`solve`) to attach.

        Returns
        -------
        Dict[str, Any]
            Structured solution data suitable for serialization or visualization.
        """
        extraction_start = perf_counter()
        self._log_info("solution_extraction_start")

        def best_key(raw_key: Any) -> Any:
            try:
                return int(raw_key)
            except (TypeError, ValueError):
                return raw_key

        schedule: List[Dict[str, Any]] = []
        for (activity, mode, day), var in model.X.items():
            if value(var) >= 0.5:
                start_day = int(best_key(day))
                chosen_mode = best_key(mode)
                finish_day = start_day
                for tau in model.T:
                    key = (activity, mode, tau)
                    if key in model.Xp and value(model.Xp[key]) >= 0.5:
                        finish_day = int(best_key(tau))
                        break
                schedule.append(
                    {
                        "activity": activity,
                        "mode": chosen_mode,
                        "start": start_day,
                        "finish": finish_day,
                        "duration": finish_day - start_day + 1,
                    }
                )
        schedule.sort(key=lambda item: (item["start"], item["activity"]))

        renewable_usage: Dict[str, Dict[int, float]] = {}
        for (res, day), var in model.BR.items():
            res_key = str(best_key(res))
            renewable_usage.setdefault(res_key, {})[int(best_key(day))] = float(value(var))

        nonrenewable_usage: Dict[str, Dict[int, float]] = {}
        for (res, day), var in model.WR.items():
            res_key = str(best_key(res))
            nonrenewable_usage.setdefault(res_key, {})[int(best_key(day))] = float(value(var))

        daily_cost = {int(best_key(t)): float(value(model.BU[t])) for t in model.T}
        period_cost = {int(best_key(y)): float(value(model.TBU[y])) for y in model.Y}

        cash_flow = {
            "periods": {int(best_key(y)): float(value(model.CF[y])) for y in model.Y},
            "payments": {int(best_key(y)): float(value(model.PA[y])) for y in model.Y},
            "delayed_payments": {int(best_key(y)): float(value(model.DP[y])) for y in model.Y},
        }

        loans = {
            "LTL": float(value(model.LTL)),
            "STL": {int(best_key(y)): float(value(model.STL[y])) for y in model.Y},
        }

        membership = {
            "mu1": float(value(model.mu1)),
            "mu2": float(value(model.mu2)),
            "lambda": float(value(model.lambda_star)),
        }

        solution: Dict[str, Any] = {
            "objective": float(value(model.OBJ)),
            "schedule": schedule,
            "resources": {
                "renewable": renewable_usage,
                "nonrenewable": nonrenewable_usage,
                "daily_cost": daily_cost,
                "period_cost": period_cost,
            },
            "cash_flow": cash_flow,
            "loans": loans,
            "membership": membership,
            "model_stats": self._model_statistics(model),
        }

        if solver_metadata:
            solution["solver"] = solver_metadata

        self._log_info(
            "solution_extraction_complete",
            elapsed_seconds=round(perf_counter() - extraction_start, 4),
            activities=len(schedule),
        )
        return solution

