"""
Regression tests for the model's economic and resource semantics.

Each test here pins down a defect that made the MILP solve a different problem from
the one the package documents. They are deliberately behavioural: they assert what a
*solution* must look like, not how the constraints happen to be written.
"""

import json
from dataclasses import replace

import pytest

from rcpsp_cf_ivfth import (
    NIVTF,
    RCPSP_CF_IVFTH,
    Activity,
    CalendarParams,
    FinanceParams,
    IVFTHTargets,
    IVFTHWeights,
    ModeData,
    ResourceParams,
    create_triangle,
)
from rcpsp_cf_ivfth.examples import build_toy_instance

TOL = 1e-6


def solved(activities, finance, calendar, targets, weights, solver_name, resources=None):
    """Build and solve an instance, returning (result, extracted solution)."""
    ivfth = RCPSP_CF_IVFTH(activities, finance, calendar, resources)
    model = ivfth.build_model(targets, weights)
    result = ivfth.solve(model, solver_name=solver_name)
    return result, ivfth.extract_solution(model, solver_metadata=result)


def total_payment_of_schedule(activities, solution):
    """Sum the payment of the mode actually selected for each activity."""
    return sum(
        activities[entry["activity"]].modes[entry["mode"]].payment for entry in solution["schedule"]
    )


@pytest.mark.solver
class TestPaymentBalance:
    """Payments must come from completed work, not from thin air."""

    def test_payments_match_realised_revenue(
        self, toy_instance, ivfth_targets, ivfth_weights, solver_name
    ):
        activities, finance, calendar = toy_instance
        _, solution = solved(
            activities, finance, calendar, ivfth_targets, ivfth_weights, solver_name
        )

        received = sum(solution["cash_flow"]["payments"].values())
        deferred = sum(solution["cash_flow"]["delayed_payments"].values())
        expected = total_payment_of_schedule(activities, solution)

        # Every unit of revenue is either collected in-period or deferred by one.
        assert received + deferred == pytest.approx(expected, abs=1e-3)

    def test_no_period_receives_more_than_total_revenue(
        self, toy_instance, ivfth_targets, ivfth_weights, solver_name
    ):
        """The original bug let a single period report many times the real revenue."""
        activities, finance, calendar = toy_instance
        _, solution = solved(
            activities, finance, calendar, ivfth_targets, ivfth_weights, solver_name
        )

        expected = total_payment_of_schedule(activities, solution)
        for period, amount in solution["cash_flow"]["payments"].items():
            assert amount <= expected + TOL, f"period {period} collected {amount} > {expected}"

    def test_nothing_is_deferred_past_the_horizon(
        self, toy_instance, ivfth_targets, ivfth_weights, solver_name
    ):
        activities, finance, calendar = toy_instance
        _, solution = solved(
            activities, finance, calendar, ivfth_targets, ivfth_weights, solver_name
        )

        last_period = max(solution["cash_flow"]["delayed_payments"])
        assert solution["cash_flow"]["delayed_payments"][last_period] == pytest.approx(0.0, abs=TOL)

    def test_final_cash_flow_does_not_track_the_target(
        self, toy_instance, ivfth_weights, solver_name
    ):
        """
        The sharpest symptom of the unbounded-payment bug: because PA[y] was free, the
        solver could manufacture whatever cash the aspiration level asked for, so
        CF_final rose with Z2_PIS and mu2 sat at 1.0 in every run.

        With payments tied to realised revenue, CF_final is decided by the cash the
        project actually generates and is therefore invariant to the anchor.
        """
        activities, finance, calendar = toy_instance
        results = []
        for z2_pis in (30_000.0, 3_000_000.0):
            targets = IVFTHTargets(
                alpha_level=0.5, Z1_PIS=10.0, Z1_NIS=60.0, Z2_PIS=z2_pis, Z2_NIS=0.0
            )
            result, _ = solved(activities, finance, calendar, targets, ivfth_weights, solver_name)
            results.append(result["CF_final"])

        assert results[0] == pytest.approx(results[1], rel=1e-6)


@pytest.mark.solver
class TestResourceProfile:
    """Resources are held for the activity's span and released afterwards."""

    def test_usage_returns_to_zero_after_the_project_ends(
        self, toy_instance, ivfth_targets, ivfth_weights, solver_name
    ):
        activities, finance, calendar = toy_instance
        result, solution = solved(
            activities, finance, calendar, ivfth_targets, ivfth_weights, solver_name
        )

        cmax = int(round(result["Cmax"]))
        for res_id, series in solution["resources"]["renewable"].items():
            after = [v for day, v in series.items() if day > cmax]
            assert all(v <= TOL for v in after), f"renewable {res_id} still in use after Cmax"

    def test_usage_is_not_monotonically_increasing(
        self, toy_instance, ivfth_targets, ivfth_weights, solver_name
    ):
        """
        The original window summed every start day h <= t, so an activity consumed
        resources from its start to the end of the horizon and the profile could
        never fall.
        """
        activities, finance, calendar = toy_instance
        _, solution = solved(
            activities, finance, calendar, ivfth_targets, ivfth_weights, solver_name
        )

        series = solution["resources"]["renewable"]["1"]
        values = [series[day] for day in sorted(series)]
        assert any(later < earlier - TOL for earlier, later in zip(values, values[1:]))

    def test_renewable_capacity_is_respected(
        self, toy_instance, ivfth_targets, ivfth_weights, solver_name
    ):
        activities, finance, calendar = toy_instance
        cap = 4.5
        _, solution = solved(
            activities,
            finance,
            calendar,
            ivfth_targets,
            ivfth_weights,
            solver_name,
            resources=ResourceParams(renewable_capacity={1: cap}),
        )

        assert max(solution["resources"]["renewable"]["1"].values()) <= cap + TOL

    def test_nonrenewable_capacity_is_a_horizon_budget(
        self, toy_instance, ivfth_targets, ivfth_weights, solver_name
    ):
        activities, finance, calendar = toy_instance
        budget = 25.0
        _, solution = solved(
            activities,
            finance,
            calendar,
            ivfth_targets,
            ivfth_weights,
            solver_name,
            resources=ResourceParams(nonrenewable_capacity={1: budget}),
        )

        assert sum(solution["resources"]["nonrenewable"]["1"].values()) <= budget + TOL

    def test_capacity_below_every_mode_is_infeasible(
        self, toy_instance, ivfth_targets, ivfth_weights, solver_name
    ):
        """A1 needs at least 4 units of renewable 1 in either mode."""
        activities, finance, calendar = toy_instance
        ivfth = RCPSP_CF_IVFTH(
            activities, finance, calendar, ResourceParams(renewable_capacity={1: 1.0})
        )
        model = ivfth.build_model(ivfth_targets, ivfth_weights)

        with pytest.raises(RuntimeError, match="no solution"):
            ivfth.solve(model, solver_name=solver_name)

    def test_tighter_capacity_never_improves_the_objective(
        self, toy_instance, ivfth_targets, ivfth_weights, solver_name
    ):
        activities, finance, calendar = toy_instance
        loose, _ = solved(activities, finance, calendar, ivfth_targets, ivfth_weights, solver_name)
        tight, _ = solved(
            activities,
            finance,
            calendar,
            ivfth_targets,
            ivfth_weights,
            solver_name,
            resources=ResourceParams(renewable_capacity={1: 4.5}),
        )

        assert tight["objective"] <= loose["objective"] + TOL


@pytest.mark.solver
class TestFinancing:
    """
    The cash-flow recurrence must match equations (20) and (21) exactly.

    Note that the two loan terms are *divisions* in the published model, so a higher
    interest rate deducts *less* from cash flow. That reads backwards economically and
    invites a well-meaning "fix", which is precisely why it is pinned here.
    """

    @staticmethod
    def _cash_starved_instance():
        """Toy instance with too little capital to self-fund, so loans are needed."""
        activities, finance, calendar = build_toy_instance()
        return activities, replace(finance, IC=200.0), calendar

    def test_first_period_matches_equation_20(
        self, toy_instance, ivfth_targets, ivfth_weights, solver_name
    ):
        """CF[1] = IC + STL[1] + LTL + PA[1] - TBU[1]."""
        activities, finance, calendar = toy_instance
        _, solution = solved(
            activities, finance, calendar, ivfth_targets, ivfth_weights, solver_name
        )

        expected = (
            finance.IC
            + solution["loans"]["STL"][1]
            + solution["loans"]["LTL"]
            + solution["cash_flow"]["payments"][1]
            - solution["resources"]["period_cost"][1]
        )
        assert solution["cash_flow"]["periods"][1] == pytest.approx(expected, rel=1e-6)

    def test_later_periods_match_equation_21(self, ivfth_targets, ivfth_weights, solver_name):
        """
        CF[y] = STL[y] + CF[y-1](1+a)^30 + PA[y] + DP[y-1](1+b)^30
                - TBU[y] - LTL/(1+g)^30 - STL[y-1]/(1+d)^30
        """
        activities, finance, calendar = self._cash_starved_instance()
        _, solution = solved(
            activities, finance, calendar, ivfth_targets, ivfth_weights, solver_name
        )

        cf = solution["cash_flow"]["periods"]
        pa = solution["cash_flow"]["payments"]
        dp = solution["cash_flow"]["delayed_payments"]
        stl = solution["loans"]["STL"]
        ltl = solution["loans"]["LTL"]
        tbu = solution["resources"]["period_cost"]

        for y in sorted(cf)[1:]:
            expected = (
                stl[y]
                + cf[y - 1] * (1.0 + finance.alpha_excess_cash) ** 30
                + pa[y]
                + dp[y - 1] * (1.0 + finance.beta_delayed_pay) ** 30
                - tbu[y]
                - ltl / (1.0 + finance.gamma_LTL) ** 30
                - stl[y - 1] / (1.0 + finance.delta_STL) ** 30
            )
            assert cf[y] == pytest.approx(expected, rel=1e-6), f"period {y}"

    def test_loan_interest_is_applied_as_a_division(
        self, ivfth_targets, ivfth_weights, solver_name
    ):
        """
        Guards the published direction of the loan terms. Under a multiplication the
        deduction would grow with the rate; under the paper's division it shrinks, so
        raising the rates cannot lower the objective.
        """
        activities, finance, calendar = self._cash_starved_instance()

        low, _ = solved(
            activities,
            replace(finance, delta_STL=0.02, gamma_LTL=0.02),
            calendar,
            ivfth_targets,
            ivfth_weights,
            solver_name,
        )
        high, _ = solved(
            activities,
            replace(finance, delta_STL=0.40, gamma_LTL=0.40),
            calendar,
            ivfth_targets,
            ivfth_weights,
            solver_name,
        )

        assert high["objective"] >= low["objective"] - TOL


@pytest.mark.solver
class TestScheduleConsistency:
    """The extracted schedule must agree with the constraints that produced it."""

    def test_duration_is_the_gap_between_start_and_finish(
        self, toy_instance, ivfth_targets, ivfth_weights, solver_name
    ):
        activities, finance, calendar = toy_instance
        _, solution = solved(
            activities, finance, calendar, ivfth_targets, ivfth_weights, solver_name
        )

        for entry in solution["schedule"]:
            assert entry["duration"] == entry["finish"] - entry["start"]
            assert entry["duration"] >= 0

    def test_successors_do_not_overlap_predecessors(
        self, toy_instance, ivfth_targets, ivfth_weights, solver_name
    ):
        activities, finance, calendar = toy_instance
        _, solution = solved(
            activities, finance, calendar, ivfth_targets, ivfth_weights, solver_name
        )
        by_name = {entry["activity"]: entry for entry in solution["schedule"]}

        for name, activity in activities.items():
            for predecessor in activity.predecessors:
                assert by_name[name]["start"] >= by_name[predecessor]["finish"] - TOL

    def test_milestones_have_zero_duration(
        self, toy_instance, ivfth_targets, ivfth_weights, solver_name
    ):
        activities, finance, calendar = toy_instance
        _, solution = solved(
            activities, finance, calendar, ivfth_targets, ivfth_weights, solver_name
        )
        by_name = {entry["activity"]: entry for entry in solution["schedule"]}

        assert by_name["Start"]["duration"] == 0
        assert by_name["End"]["duration"] == 0

    def test_solution_is_json_serializable(
        self, toy_instance, ivfth_targets, ivfth_weights, solver_name
    ):
        """Solvers that report no timing hand back a non-serializable sentinel."""
        activities, finance, calendar = toy_instance
        _, solution = solved(
            activities, finance, calendar, ivfth_targets, ivfth_weights, solver_name
        )

        payload = json.loads(json.dumps(solution))
        assert payload["solver"]["solver_time"] is None or isinstance(
            payload["solver"]["solver_time"], float
        )


class TestResourceSpanHelper:
    """
    Constraints (31)/(32) size the resource window with the expected value of the
    lower triangle, independently of alpha - unlike precedence (30), which uses the
    alpha blend. The two are meant to differ.
    """

    def test_zero_duration_occupies_no_days(self):
        milestone = NIVTF(*create_triangle(0, 0, 0))
        assert RCPSP_CF_IVFTH._resource_span_days(milestone) == 0

    def test_span_is_the_expected_value_rounded_up(self):
        duration = NIVTF(*create_triangle(6, 8, 10, widen=0.6))
        expected_value = duration.EV_L()
        span = RCPSP_CF_IVFTH._resource_span_days(duration)

        assert span >= expected_value
        assert span - expected_value < 1.0

    def test_span_does_not_depend_on_alpha(self):
        """The window length is EV-based, so alpha must not enter it."""
        duration = NIVTF(*create_triangle(3, 5, 7, widen=0.4))
        spans = {RCPSP_CF_IVFTH._resource_span_days(duration) for _ in range(3)}

        assert len(spans) == 1
        assert spans.pop() >= 1

    @pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
    def test_precedence_blend_does_depend_on_alpha(self, alpha):
        duration = NIVTF(*create_triangle(3, 5, 7, widen=0.4))
        blend = RCPSP_CF_IVFTH._alpha_blend_duration(duration, alpha=alpha)

        assert blend == pytest.approx(alpha * duration.E2_L() + (1.0 - alpha) * duration.E1_L())


class TestCycleDetection:
    """Cycle detection must survive activity graphs deeper than the recursion limit."""

    @staticmethod
    def _chain(length, close_cycle):
        zero = NIVTF(*create_triangle(0, 0, 0))

        def milestone(name, predecessors):
            return Activity(
                name=name,
                predecessors=predecessors,
                modes={
                    1: ModeData(
                        duration=zero, renewables={1: zero}, nonrenewables={1: zero}, payment=0.0
                    )
                },
            )

        names = ["Start"] + [f"A{i}" for i in range(length)] + ["End"]
        activities = {}
        for index, name in enumerate(names):
            predecessors = [names[index - 1]] if index else []
            activities[name] = milestone(name, predecessors)
        if close_cycle:
            # Make the first chain link depend on the last one.
            activities[names[1]].predecessors.append(names[-1])
        return activities

    @staticmethod
    def _params():
        finance = FinanceParams(
            alpha_excess_cash=0.01,
            beta_delayed_pay=0.05,
            gamma_LTL=0.06,
            delta_STL=0.08,
            IC=1000.0,
            max_LTL=500.0,
            max_STL=300.0,
            min_CF=0.0,
            CC_daily_cap=100.0,
            CR_k={1: 1.0},
            CW_l={1: 1.0},
        )
        return finance, CalendarParams(T_days=10, Y_periods=[(1, 10)])

    def test_deep_acyclic_chain_validates(self):
        finance, calendar = self._params()
        activities = self._chain(3000, close_cycle=False)

        # A recursive walk raises RecursionError well before this depth.
        ivfth = RCPSP_CF_IVFTH(activities, finance, calendar)
        assert len(ivfth.activities) == 3002

    def test_deep_cycle_is_reported(self):
        finance, calendar = self._params()
        activities = self._chain(3000, close_cycle=True)

        with pytest.raises(ValueError, match="cycle"):
            RCPSP_CF_IVFTH(activities, finance, calendar)


class TestResourceParamsValidation:
    """ResourceParams guards its own inputs."""

    def test_negative_renewable_capacity_is_rejected(self):
        with pytest.raises(ValueError, match="renewable_capacity"):
            ResourceParams(renewable_capacity={1: -1.0})

    def test_negative_nonrenewable_capacity_is_rejected(self):
        with pytest.raises(ValueError, match="nonrenewable_capacity"):
            ResourceParams(nonrenewable_capacity={1: -5.0})

    def test_missing_resource_means_unlimited(self):
        params = ResourceParams(renewable_capacity={1: 3.0})
        assert params.renewable_limit(1) == 3.0
        assert params.renewable_limit(2) is None
        assert params.nonrenewable_limit(1) is None

    def test_defaults_are_independent_between_instances(self):
        first = ResourceParams()
        first.renewable_capacity[1] = 5.0
        assert ResourceParams().renewable_capacity == {}


class TestBuildValidation:
    """build_model rejects target anchors it cannot form memberships from."""

    def test_alpha_outside_unit_interval_is_rejected(self, toy_instance, ivfth_weights):
        activities, finance, calendar = toy_instance
        ivfth = RCPSP_CF_IVFTH(activities, finance, calendar)
        targets = IVFTHTargets.__new__(IVFTHTargets)
        object.__setattr__(targets, "alpha_level", 1.5)
        object.__setattr__(targets, "Z1_PIS", 10.0)
        object.__setattr__(targets, "Z1_NIS", 60.0)
        object.__setattr__(targets, "Z2_PIS", 30000.0)
        object.__setattr__(targets, "Z2_NIS", 0.0)

        with pytest.raises(ValueError, match="alpha_level"):
            ivfth.build_model(targets, ivfth_weights)

    def test_equal_makespan_anchors_are_rejected(self, toy_instance, ivfth_weights):
        activities, finance, calendar = toy_instance
        ivfth = RCPSP_CF_IVFTH(activities, finance, calendar)
        targets = IVFTHTargets(alpha_level=0.5, Z1_PIS=30.0, Z1_NIS=30.0, Z2_PIS=1.0, Z2_NIS=0.0)

        with pytest.raises(ValueError, match="Z1_PIS"):
            ivfth.build_model(targets, IVFTHWeights(0.5, 0.5, 0.5))
