"""
Tests for the plotting and export helpers.

Plotting is exercised through matplotlib's non-interactive Agg backend, so these run
headless in CI. They assert on the artists and files produced rather than on pixels.
"""

import csv
import json

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from rcpsp_cf_ivfth.visualization import (  # noqa: E402
    create_gantt_chart,
    export_solution_csv,
    export_solution_json,
    plot_cash_flow,
    plot_loan_usage,
    plot_resource_usage,
)


@pytest.fixture
def solution():
    """A minimal solution dictionary shaped like extract_solution's output."""
    return {
        "objective": 0.64,
        "schedule": [
            {"activity": "Start", "mode": 1, "start": 1, "finish": 1, "duration": 0},
            {"activity": "A1", "mode": 2, "start": 1, "finish": 8, "duration": 7},
            {"activity": "A2", "mode": 2, "start": 8, "finish": 17, "duration": 9},
            {"activity": "End", "mode": 1, "start": 17, "finish": 17, "duration": 0},
        ],
        "resources": {
            "renewable": {"1": {1: 5.0, 2: 5.0, 3: 4.0}, "2": {1: 1.0, 2: 1.0, 3: 0.0}},
            "nonrenewable": {"1": {1: 2.0, 2: 2.0, 3: 1.0}},
            "daily_cost": {1: 120.0, 2: 120.0, 3: 90.0},
            "period_cost": {1: 2062.75, 2: 0.0},
        },
        "cash_flow": {
            "periods": {1: 7937.25, 2: 17276.47},
            "payments": {1: 0.0, 2: 0.0},
            "delayed_payments": {1: 8400.0, 2: 0.0},
        },
        "loans": {"LTL": 0.0, "STL": {1: 0.0, 2: 0.0}},
        "membership": {"mu1": 0.86, "mu2": 0.58, "lambda": 0.58},
        "model_stats": {"variables": 100, "constraints": 200},
    }


class TestGanttChart:
    def test_draws_one_row_per_activity(self, solution):
        """Rows are ordered by start day, then alphabetically within the same day."""
        ax = create_gantt_chart(solution, show=False)
        assert [label.get_text() for label in ax.get_yticklabels()] == [
            "A1",
            "Start",
            "A2",
            "End",
        ]

    def test_milestones_render_as_markers_not_bars(self, solution):
        """A zero-duration bar is invisible, so milestones are drawn as markers."""
        ax = create_gantt_chart(solution, show=False)

        # Two real activities produce bars; the two milestones produce line markers.
        assert len(ax.patches) == 2
        assert len(ax.lines) == 2

    def test_bars_do_not_overlap(self, solution):
        ax = create_gantt_chart(solution, show=False)
        spans = sorted((patch.get_x(), patch.get_x() + patch.get_width()) for patch in ax.patches)

        for (_, first_end), (second_start, _) in zip(spans, spans[1:]):
            assert second_start >= first_end - 1e-9

    def test_empty_schedule_is_rejected(self, solution):
        solution["schedule"] = []
        with pytest.raises(ValueError, match="schedule is empty"):
            create_gantt_chart(solution, show=False)

    def test_accepts_a_caller_supplied_axis(self, solution):
        import matplotlib.pyplot as plt

        _, ax = plt.subplots()
        assert create_gantt_chart(solution, ax=ax, show=False) is ax


class TestResourcePlots:
    @pytest.mark.parametrize("resource_type", ["renewable", "nonrenewable"])
    def test_plots_one_line_per_resource(self, solution, resource_type):
        ax = plot_resource_usage(solution, resource_type=resource_type, show=False)
        expected = len(solution["resources"][resource_type])
        assert len(ax.lines) == expected

    def test_unknown_resource_type_is_rejected(self, solution):
        with pytest.raises(ValueError, match="No resource usage"):
            plot_resource_usage(solution, resource_type="imaginary", show=False)


class TestFinancePlots:
    def test_cash_flow_plots_periods(self, solution):
        ax = plot_cash_flow(solution, show=False)
        assert ax.lines
        assert ax.get_ylabel() == "Cash Flow"

    def test_cash_flow_requires_period_data(self, solution):
        solution["cash_flow"]["periods"] = {}
        with pytest.raises(ValueError, match="Cash flow data is missing"):
            plot_cash_flow(solution, show=False)

    def test_loan_usage_plots_both_facilities(self, solution):
        ax = plot_loan_usage(solution, show=False)
        assert ax.lines

    def test_loan_usage_requires_loan_data(self, solution):
        solution["loans"] = {}
        with pytest.raises(ValueError, match="Loan information missing"):
            plot_loan_usage(solution, show=False)


class TestExports:
    def test_json_round_trips(self, solution, tmp_path):
        target = export_solution_json(solution, tmp_path / "solution.json")
        assert json.loads(target.read_text(encoding="utf-8"))["objective"] == solution["objective"]

    def test_csv_writes_three_files(self, solution, tmp_path):
        schedule_path, resources_path, finance_path = export_solution_csv(
            solution, tmp_path / "run"
        )
        assert schedule_path.exists() and resources_path.exists() and finance_path.exists()

    def test_schedule_csv_matches_the_schedule(self, solution, tmp_path):
        schedule_path, _, _ = export_solution_csv(solution, tmp_path / "run")
        with schedule_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))

        assert [row["activity"] for row in rows] == [e["activity"] for e in solution["schedule"]]
        assert rows[1]["duration"] == "7"

    def test_finance_csv_contains_every_category(self, solution, tmp_path):
        _, _, finance_path = export_solution_csv(solution, tmp_path / "run")
        with finance_path.open(encoding="utf-8", newline="") as handle:
            categories = {row[0] for row in csv.reader(handle)}

        assert {"cash_flow", "payments", "delayed_payments", "LTL", "STL", "membership"} <= (
            categories
        )

    def test_existing_suffix_is_replaced(self, solution, tmp_path):
        schedule_path, _, _ = export_solution_csv(solution, tmp_path / "run.csv")
        assert schedule_path.name == "run_schedule.csv"
