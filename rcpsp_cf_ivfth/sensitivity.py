"""
Sensitivity analysis utilities for RCPSP-CF-IVFTH.

This module provides helpers to explore how IVF-TH model results change when
varying alpha levels, IVF-TH weights, or finance parameters.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .data import FinanceParams, IVFTHTargets, IVFTHWeights
from .model import RCPSP_CF_IVFTH

__all__ = [
    "run_alpha_sweep",
    "run_weight_scenarios",
    "run_finance_scenarios",
    "sensitivity_analysis",
    "plot_metric_trends",
]


def _require_pandas():
    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "pandas is required for sensitivity analysis. Install it with `pip install pandas`."
        ) from exc
    return pd


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "matplotlib is required for plotting sensitivity analysis. Install it with `pip install matplotlib`."
        ) from exc
    return plt


def _solve_scenario(
    ivfth: RCPSP_CF_IVFTH,
    targets: IVFTHTargets,
    weights: IVFTHWeights,
    *,
    solver_name: str = "cbc",
    solver_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    solve_kwargs = solver_kwargs or {}
    model = ivfth.build_model(targets, weights)
    result = ivfth.solve(model, solver_name=solver_name, **solve_kwargs)
    solution = ivfth.extract_solution(model, solver_metadata=result)
    metrics = {
        "objective": result.get("objective"),
        "Cmax": result.get("Cmax"),
        "CF_final": result.get("CF_final"),
        "mu1": result.get("mu1"),
        "mu2": result.get("mu2"),
        "lambda": result.get("lambda"),
        "status": result.get("status"),
        "solver_status": result.get("solver_status"),
        "solver_time": result.get("solver_time"),
    }
    metrics["solution"] = solution
    return metrics


def run_alpha_sweep(
    ivfth: RCPSP_CF_IVFTH,
    targets: IVFTHTargets,
    weights: IVFTHWeights,
    alpha_levels: Sequence[float] = (0.1, 0.3, 0.5, 0.7, 0.9),
    *,
    solver_name: str = "cbc",
    solver_kwargs: Optional[Dict[str, Any]] = None,
) -> "pd.DataFrame":
    """
    Evaluate the model across multiple alpha levels.
    """
    pd = _require_pandas()
    rows: List[Dict[str, Any]] = []
    for alpha in alpha_levels:
        target_variant = replace(targets, alpha_level=alpha)
        metrics = _solve_scenario(
            ivfth,
            target_variant,
            weights,
            solver_name=solver_name,
            solver_kwargs=solver_kwargs,
        )
        row = {k: v for k, v in metrics.items() if k != "solution"}
        row["alpha_level"] = alpha
        rows.append(row)
    df = pd.DataFrame(rows).set_index("alpha_level").sort_index()
    return df


def run_weight_scenarios(
    ivfth: RCPSP_CF_IVFTH,
    targets: IVFTHTargets,
    weight_configs: Iterable[Tuple[float, float, float]],
    *,
    solver_name: str = "cbc",
    solver_kwargs: Optional[Dict[str, Any]] = None,
) -> "pd.DataFrame":
    """
    Evaluate the model across a grid of (theta1, theta2, gamma) configurations.
    """
    pd = _require_pandas()
    rows: List[Dict[str, Any]] = []
    for theta1, theta2, gamma in weight_configs:
        if abs(theta1 + theta2 - 1.0) > 1e-9:
            raise ValueError(f"Theta weights must sum to 1. Received theta1={theta1}, theta2={theta2}.")
        weights = IVFTHWeights(theta1=theta1, theta2=theta2, gamma_tradeoff=gamma)
        metrics = _solve_scenario(
            ivfth,
            targets,
            weights,
            solver_name=solver_name,
            solver_kwargs=solver_kwargs,
        )
        row = {k: v for k, v in metrics.items() if k != "solution"}
        row.update({"theta1": theta1, "theta2": theta2, "gamma": gamma})
        rows.append(row)
    df = pd.DataFrame(rows).sort_values(["gamma", "theta1"])
    return df


def run_finance_scenarios(
    ivfth: RCPSP_CF_IVFTH,
    targets: IVFTHTargets,
    weights: IVFTHWeights,
    scenarios: Iterable[Tuple[str, Dict[str, float]]],
    *,
    solver_name: str = "cbc",
    solver_kwargs: Optional[Dict[str, Any]] = None,
) -> "pd.DataFrame":
    """
    Evaluate the model with modified finance parameters.

    Parameters
    ----------
    scenarios:
        Iterable of (label, overrides) where overrides is a mapping of attribute -> new value.
    """
    pd = _require_pandas()

    base_acts = ivfth.activities
    base_calendar = ivfth.calendar
    base_logging = {
        "logging_enabled": ivfth._logging_enabled,
        "log_level": ivfth._log_level,
        "log_file": ivfth._log_file,
        "strict_validation": ivfth._strict_validation,
    }

    rows: List[Dict[str, Any]] = []
    for label, overrides in scenarios:
        finance_variant = replace(ivfth.finance, **overrides)
        scenario_model = RCPSP_CF_IVFTH(
            base_acts,
            finance_variant,
            base_calendar,
            **base_logging,
        )
        metrics = _solve_scenario(
            scenario_model,
            targets,
            weights,
            solver_name=solver_name,
            solver_kwargs=solver_kwargs,
        )
        row = {k: v for k, v in metrics.items() if k != "solution"}
        row["scenario"] = label
        for key, value in overrides.items():
            row[f"finance.{key}"] = value
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("scenario")
    return df


def sensitivity_analysis(
    ivfth: RCPSP_CF_IVFTH,
    targets: IVFTHTargets,
    weights: IVFTHWeights,
    *,
    alpha_levels: Optional[Sequence[float]] = None,
    weight_configs: Optional[Iterable[Tuple[float, float, float]]] = None,
    finance_scenarios: Optional[Iterable[Tuple[str, Dict[str, float]]]] = None,
    solver_name: str = "cbc",
    solver_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, "pd.DataFrame"]:
    """
    Run a bundle of sensitivity analyses and return results per scenario family.
    """
    results: Dict[str, "pd.DataFrame"] = {}
    if alpha_levels:
        results["alpha"] = run_alpha_sweep(
            ivfth,
            targets,
            weights,
            alpha_levels,
            solver_name=solver_name,
            solver_kwargs=solver_kwargs,
        )
    if weight_configs:
        results["weights"] = run_weight_scenarios(
            ivfth,
            targets,
            weight_configs,
            solver_name=solver_name,
            solver_kwargs=solver_kwargs,
        )
    if finance_scenarios:
        results["finance"] = run_finance_scenarios(
            ivfth,
            targets,
            weights,
            finance_scenarios,
            solver_name=solver_name,
            solver_kwargs=solver_kwargs,
        )
    return results


def plot_metric_trends(
    df: "pd.DataFrame",
    x_column: str,
    *,
    metrics: Sequence[str] = ("Cmax", "CF_final"),
    ax=None,
    show: bool = True,
) -> Any:
    """
    Plot selected metrics against a given column (e.g., alpha levels or scenario index).
    """
    pd = _require_pandas()
    plt = _require_matplotlib()

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame for plotting.")
    if x_column not in df.columns:
        raise ValueError(f"Column '{x_column}' not found in DataFrame.")

    plot_df = df.reset_index(drop=False)
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure

    for metric in metrics:
        if metric not in plot_df.columns:
            continue
        ax.plot(plot_df[x_column], plot_df[metric], marker="o", label=metric)

    ax.set_xlabel(x_column)
    ax.set_ylabel("Metric value")
    ax.set_title("Sensitivity trends")
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()

    if show:
        plt.show()
    return ax
