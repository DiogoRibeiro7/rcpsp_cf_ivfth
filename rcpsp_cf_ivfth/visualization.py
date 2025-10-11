"""
Visualization utilities for RCPSP-CF-IVFTH solutions.

This module provides helper functions to turn the structured solution dictionary
returned by :meth:`rcpsp_cf_ivfth.model.RCPSP_CF_IVFTH.extract_solution` into
plots and exportable formats.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple
import json
import csv

__all__ = [
    "create_gantt_chart",
    "plot_resource_usage",
    "plot_cash_flow",
    "plot_loan_usage",
    "export_solution_json",
    "export_solution_csv",
]


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "matplotlib is required for visualization utilities. "
            "Install it with `pip install matplotlib`."
        ) from exc
    return plt


def _schedule_entries(solution: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    return solution.get("schedule", [])


def create_gantt_chart(
    solution: Dict[str, Any],
    *,
    ax=None,
    show: bool = True,
) -> Any:
    """
    Plot a simple Gantt chart for the activity schedule.
    """
    plt = _require_matplotlib()
    schedule = list(_schedule_entries(solution))
    if not schedule:
        raise ValueError("Solution schedule is empty; cannot create Gantt chart.")

    schedule = sorted(schedule, key=lambda item: (item["start"], item["activity"]))

    if ax is None:
        height = max(4, len(schedule) * 0.6)
        fig, ax = plt.subplots(figsize=(10, height))
    else:
        fig = ax.figure

    yticks = []
    ylabels = []
    for idx, entry in enumerate(schedule):
        activity = entry["activity"]
        start = entry["start"]
        duration = entry["duration"]
        finish = entry["finish"]
        mode = entry.get("mode", "")
        ax.barh(
            idx,
            duration,
            left=start,
            height=0.4,
            align="center",
            color="#4C78A8",
            edgecolor="black",
        )
        ax.text(
            start + duration / 2,
            idx,
            f"{activity} (M{mode})\n{start}->{finish}",
            ha="center",
            va="center",
            color="white",
            fontsize=9,
            weight="bold",
        )
        yticks.append(idx)
        ylabels.append(activity)

    ax.set_xlabel("Day")
    ax.set_ylabel("Activity")
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_title("Project Schedule")
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()

    if show:
        plt.show()
    return ax


def plot_resource_usage(
    solution: Dict[str, Any],
    *,
    resource_type: str = "renewable",
    ax=None,
    show: bool = True,
) -> Any:
    """
    Plot resource usage over the planning horizon.
    """
    plt = _require_matplotlib()
    resources = solution.get("resources", {})
    usage = resources.get(resource_type)
    if not usage:
        raise ValueError(f"No resource usage found for type '{resource_type}'.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure

    for res_id, series in sorted(usage.items(), key=lambda item: item[0]):
        days = sorted(series)
        values = [series[day] for day in days]
        ax.plot(days, values, marker="o", label=f"{resource_type.title()} {res_id}")

    ax.set_xlabel("Day")
    ax.set_ylabel("Usage")
    ax.set_title(f"{resource_type.title()} Resource Usage")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()

    if show:
        plt.show()
    return ax


def plot_cash_flow(
    solution: Dict[str, Any],
    *,
    ax=None,
    show: bool = True,
) -> Any:
    """
    Plot the cash flow trajectory across accounting periods.
    """
    plt = _require_matplotlib()
    cash_flow = solution.get("cash_flow", {})
    periods = cash_flow.get("periods")
    if not periods:
        raise ValueError("Cash flow data is missing in solution.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure

    xs = sorted(periods)
    ys = [periods[p] for p in xs]
    ax.plot(xs, ys, marker="o", color="#59A14F", label="Cash Flow")

    payments = cash_flow.get("payments", {})
    if payments:
        ax.bar(xs, [payments.get(p, 0.0) for p in xs], alpha=0.2, label="Payments")

    ax.set_xlabel("Period")
    ax.set_ylabel("Cash Flow")
    ax.set_title("Cash Flow by Period")
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()

    if show:
        plt.show()
    return ax


def plot_loan_usage(
    solution: Dict[str, Any],
    *,
    ax=None,
    show: bool = True,
) -> Any:
    """
    Plot STL/LTL loan usage across periods.
    """
    plt = _require_matplotlib()
    loans = solution.get("loans", {})
    if not loans:
        raise ValueError("Loan information missing in solution.")

    ltl = loans.get("LTL", 0.0)
    stl = loans.get("STL", {})

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 3))
    else:
        fig = ax.figure

    if stl:
        periods = sorted(stl)
        values = [stl[p] for p in periods]
        ax.plot(periods, values, marker="o", color="#F28E2B", label="Short-term Loan")
    ax.axhline(ltl, color="#E15759", linestyle="--", label="Long-term Loan")

    ax.set_xlabel("Period")
    ax.set_ylabel("Loan Amount")
    ax.set_title("Loan Utilization")
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()

    if show:
        plt.show()
    return ax


def export_solution_json(solution: Dict[str, Any], path: Path | str) -> Path:
    """
    Export the solution dictionary to a JSON file.
    """
    target = Path(path)
    with target.open("w", encoding="utf-8") as fp:
        json.dump(solution, fp, indent=2)
    return target


def export_solution_csv(solution: Dict[str, Any], base_path: Path | str) -> Tuple[Path, Path, Path]:
    """
    Export solution components to CSV files.

    Returns
    -------
    Tuple[Path, Path, Path]
        Paths to the generated schedule, resources, and finance CSV files.
    """
    base = Path(base_path)
    if base.suffix:
        base = base.with_suffix("")

    schedule_path = base.with_name(base.name + "_schedule.csv")
    resources_path = base.with_name(base.name + "_resources.csv")
    finance_path = base.with_name(base.name + "_finance.csv")

    schedule = list(_schedule_entries(solution))
    with schedule_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["activity", "mode", "start", "finish", "duration"])
        writer.writeheader()
        for entry in schedule:
            writer.writerow(entry)

    resources = solution.get("resources", {})
    with resources_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["type", "resource", "time", "value"])
        for rtype in ("renewable", "nonrenewable"):
            usage = resources.get(rtype, {})
            for res_id, series in usage.items():
                for time_point, value_ in sorted(series.items()):
                    writer.writerow([rtype, res_id, time_point, value_])
        daily_cost = resources.get("daily_cost", {})
        for day, value_ in sorted(daily_cost.items()):
            writer.writerow(["daily_cost", "", day, value_])
        period_cost = resources.get("period_cost", {})
        for period, value_ in sorted(period_cost.items()):
            writer.writerow(["period_cost", "", period, value_])

    cash_flow = solution.get("cash_flow", {})
    loans = solution.get("loans", {})
    membership = solution.get("membership", {})
    with finance_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["category", "key", "value"])
        for period, value_ in sorted(cash_flow.get("periods", {}).items()):
            writer.writerow(["cash_flow", period, value_])
        for period, value_ in sorted(cash_flow.get("payments", {}).items()):
            writer.writerow(["payments", period, value_])
        for period, value_ in sorted(cash_flow.get("delayed_payments", {}).items()):
            writer.writerow(["delayed_payments", period, value_])
        writer.writerow(["LTL", "", loans.get("LTL", 0.0)])
        for period, value_ in sorted(loans.get("STL", {}).items()):
            writer.writerow(["STL", period, value_])
        for key, value_ in membership.items():
            writer.writerow(["membership", key, value_])

    return schedule_path, resources_path, finance_path
