"""
Tests for the sensitivity-analysis helpers.

These drive the solver repeatedly, so the sweeps are kept deliberately small.
"""

import pytest

pytest.importorskip("pandas")
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from rcpsp_cf_ivfth import RCPSP_CF_IVFTH, ResourceParams  # noqa: E402
from rcpsp_cf_ivfth.sensitivity import (  # noqa: E402
    plot_metric_trends,
    run_alpha_sweep,
    run_finance_scenarios,
    run_weight_scenarios,
    sensitivity_analysis,
)

pytestmark = pytest.mark.solver


@pytest.fixture
def ivfth(toy_instance):
    activities, finance, calendar = toy_instance
    return RCPSP_CF_IVFTH(activities, finance, calendar)


class TestAlphaSweep:
    def test_returns_one_row_per_alpha_indexed_by_alpha(
        self, ivfth, ivfth_targets, ivfth_weights, solver_name
    ):
        alphas = (0.2, 0.8)
        df = run_alpha_sweep(ivfth, ivfth_targets, ivfth_weights, alphas, solver_name=solver_name)

        assert df.index.name == "alpha_level"
        assert list(df.index) == sorted(alphas)
        assert {"objective", "Cmax", "CF_final", "mu1", "mu2"} <= set(df.columns)

    def test_alpha_changes_the_schedule_length(
        self, ivfth, ivfth_targets, ivfth_weights, solver_name
    ):
        """A higher alpha leans on the pessimistic duration, so Cmax cannot shrink."""
        df = run_alpha_sweep(
            ivfth, ivfth_targets, ivfth_weights, (0.0, 1.0), solver_name=solver_name
        )
        assert df.loc[1.0, "Cmax"] >= df.loc[0.0, "Cmax"]


class TestWeightScenarios:
    def test_records_every_configuration(self, ivfth, ivfth_targets, solver_name):
        configs = [(0.5, 0.5, 0.5), (0.9, 0.1, 0.5)]
        df = run_weight_scenarios(ivfth, ivfth_targets, configs, solver_name=solver_name)

        assert len(df) == len(configs)
        assert set(df["theta1"]) == {0.5, 0.9}

    def test_rejects_weights_that_do_not_sum_to_one(self, ivfth, ivfth_targets, solver_name):
        with pytest.raises(ValueError, match="must sum to 1"):
            run_weight_scenarios(ivfth, ivfth_targets, [(0.6, 0.6, 0.5)], solver_name=solver_name)

    def test_favouring_makespan_does_not_worsen_it(self, ivfth, ivfth_targets, solver_name):
        df = run_weight_scenarios(
            ivfth,
            ivfth_targets,
            [(0.1, 0.9, 0.0), (0.9, 0.1, 0.0)],
            solver_name=solver_name,
        )
        by_theta1 = df.set_index("theta1")
        assert by_theta1.loc[0.9, "Cmax"] <= by_theta1.loc[0.1, "Cmax"]


class TestFinanceScenarios:
    def test_applies_each_override(self, ivfth, ivfth_targets, ivfth_weights, solver_name):
        scenarios = [
            ("baseline", {}),
            ("lean_capital", {"IC": 4000.0}),
        ]
        df = run_finance_scenarios(
            ivfth, ivfth_targets, ivfth_weights, scenarios, solver_name=solver_name
        )

        assert set(df["scenario"]) == {"baseline", "lean_capital"}
        assert df.set_index("scenario").loc["lean_capital", "finance.IC"] == 4000.0

    def test_less_capital_never_improves_final_cash(
        self, ivfth, ivfth_targets, ivfth_weights, solver_name
    ):
        df = run_finance_scenarios(
            ivfth,
            ivfth_targets,
            ivfth_weights,
            [("rich", {"IC": 10000.0}), ("poor", {"IC": 3000.0})],
            solver_name=solver_name,
        )
        by_scenario = df.set_index("scenario")
        assert by_scenario.loc["poor", "CF_final"] <= by_scenario.loc["rich", "CF_final"] + 1e-6

    def test_resource_limits_carry_into_each_scenario(
        self, toy_instance, ivfth_targets, ivfth_weights, solver_name
    ):
        """Scenario models are rebuilt, so they must inherit the capacity limits."""
        activities, finance, calendar = toy_instance
        constrained = RCPSP_CF_IVFTH(
            activities, finance, calendar, ResourceParams(renewable_capacity={1: 4.5})
        )
        df = run_finance_scenarios(
            constrained,
            ivfth_targets,
            ivfth_weights,
            [("baseline", {})],
            solver_name=solver_name,
        )

        unconstrained = run_alpha_sweep(
            RCPSP_CF_IVFTH(activities, finance, calendar),
            ivfth_targets,
            ivfth_weights,
            (ivfth_targets.alpha_level,),
            solver_name=solver_name,
        )
        assert df.iloc[0]["Cmax"] >= unconstrained.iloc[0]["Cmax"]


class TestSensitivityBundle:
    def test_only_requested_families_are_run(
        self, ivfth, ivfth_targets, ivfth_weights, solver_name
    ):
        results = sensitivity_analysis(
            ivfth,
            ivfth_targets,
            ivfth_weights,
            alpha_levels=(0.5,),
            solver_name=solver_name,
        )
        assert set(results) == {"alpha"}

    def test_empty_request_returns_nothing(self, ivfth, ivfth_targets, ivfth_weights, solver_name):
        assert sensitivity_analysis(ivfth, ivfth_targets, ivfth_weights) == {}


class TestMetricTrends:
    def test_plots_against_an_index_column(self, ivfth, ivfth_targets, ivfth_weights, solver_name):
        """
        run_alpha_sweep returns alpha_level as the index; validating before
        reset_index() rejected the function's own primary input.
        """
        df = run_alpha_sweep(
            ivfth, ivfth_targets, ivfth_weights, (0.2, 0.8), solver_name=solver_name
        )
        ax = plot_metric_trends(df, "alpha_level", show=False)

        assert len(ax.lines) == 2  # Cmax and CF_final
        assert ax.get_xlabel() == "alpha_level"

    def test_unknown_column_lists_the_available_ones(
        self, ivfth, ivfth_targets, ivfth_weights, solver_name
    ):
        df = run_alpha_sweep(ivfth, ivfth_targets, ivfth_weights, (0.5,), solver_name=solver_name)
        with pytest.raises(ValueError, match="Available:"):
            plot_metric_trends(df, "not_a_column", show=False)

    def test_rejects_non_dataframe_input(self):
        with pytest.raises(TypeError, match="pandas DataFrame"):
            plot_metric_trends([1, 2, 3], "alpha_level", show=False)

    def test_missing_metrics_are_skipped(self, ivfth, ivfth_targets, ivfth_weights, solver_name):
        df = run_alpha_sweep(ivfth, ivfth_targets, ivfth_weights, (0.5,), solver_name=solver_name)
        ax = plot_metric_trends(df, "alpha_level", metrics=("Cmax", "nope"), show=False)
        assert len(ax.lines) == 1
