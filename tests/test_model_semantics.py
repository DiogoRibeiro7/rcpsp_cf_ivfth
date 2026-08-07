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

    def test_second_objective_is_not_degenerate(
        self, toy_instance, ivfth_targets, ivfth_weights, solver_name
    ):
        """
        With unbounded payments mu2 was 1.0 in every run, which removed the cash-flow
        objective from the problem entirely and made the bi-objective trade-off a no-op.
        """
        activities, finance, calendar = toy_instance
        result, _ = solved(activities, finance, calendar, ivfth_targets, ivfth_weights, solver_name)

        assert result["mu2"] < 1.0 - TOL
        assert result["CF_final"] < ivfth_targets.Z2_PIS


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
    """Debt must cost money, and it must be repaid inside the horizon."""

    @staticmethod
    def _cash_starved_instance():
        """Toy instance with too little capital to self-fund, so loans are needed."""
        activities, finance, calendar = build_toy_instance()
        return activities, replace(finance, IC=200.0), calendar

    def test_higher_loan_interest_never_improves_cash_flow(
        self, ivfth_targets, ivfth_weights, solver_name
    ):
        """
        Dividing by the rate factor made debt *cheaper* as interest rose, so raising a
        rate could increase the reported final cash flow.
        """
        activities, finance, calendar = self._cash_starved_instance()

        cheap, _ = solved(
            activities,
            replace(finance, delta_STL=0.02, gamma_LTL=0.02),
            calendar,
            ivfth_targets,
            ivfth_weights,
            solver_name,
        )
        expensive, _ = solved(
            activities,
            replace(finance, delta_STL=0.40, gamma_LTL=0.40),
            calendar,
            ivfth_targets,
            ivfth_weights,
            solver_name,
        )

        assert expensive["CF_final"] <= cheap["CF_final"] + TOL

    def test_final_cash_flow_is_net_of_outstanding_debt(
        self, ivfth_targets, ivfth_weights, solver_name
    ):
        """
        A loan drawn in the final period used to be pure profit, because nothing
        repaid it before the horizon ended.
        """
        activities, finance, calendar = self._cash_starved_instance()
        result, solution = solved(
            activities, finance, calendar, ivfth_targets, ivfth_weights, solver_name
        )

        last_period = max(solution["loans"]["STL"])
        borrowed = solution["loans"]["STL"][last_period] + solution["loans"]["LTL"]
        # Whatever is still borrowed at the end cannot also be counted as final cash.
        assert (
            result["CF_final"]
            <= sum(activities[e["activity"]].modes[e["mode"]].payment for e in solution["schedule"])
            + finance.IC
            + TOL
        )
        assert borrowed >= -TOL


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


class TestActiveDaysHelper:
    """The resource span must agree with the gap precedence enforces."""

    def test_zero_duration_occupies_no_days(self):
        milestone = NIVTF(*create_triangle(0, 0, 0))
        assert RCPSP_CF_IVFTH._active_days(milestone, alpha=0.5) == 0

    def test_span_rounds_up_to_whole_days(self):
        duration = NIVTF(*create_triangle(6, 8, 10, widen=0.6))
        blend = RCPSP_CF_IVFTH._alpha_blend_duration(duration, alpha=0.5)
        span = RCPSP_CF_IVFTH._active_days(duration, alpha=0.5)

        assert span >= blend
        assert span - blend < 1.0

    @pytest.mark.parametrize("alpha", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_span_is_positive_for_real_work(self, alpha):
        duration = NIVTF(*create_triangle(3, 5, 7, widen=0.4))
        assert RCPSP_CF_IVFTH._active_days(duration, alpha=alpha) >= 1


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
