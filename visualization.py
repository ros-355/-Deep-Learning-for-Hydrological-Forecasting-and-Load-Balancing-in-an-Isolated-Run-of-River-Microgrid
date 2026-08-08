"""Publication-quality Matplotlib figures for forecasting and control."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "blue": "#1768AC",
    "orange": "#F28E2B",
    "green": "#2A9D8F",
    "red": "#D1495B",
    "gray": "#636363",
    "light": "#E8EEF4",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_forecast_accuracy(
    hour_indices: np.ndarray,
    actual_flow: np.ndarray,
    predicted_flow: np.ndarray,
    output_path: str | Path,
    window_hours: int = 336,
) -> None:
    _style()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    n = min(window_hours, len(actual_flow))
    x = (hour_indices[:n] - hour_indices[0]) / 24.0
    fig, ax = plt.subplots(figsize=(6.45, 2.65))
    ax.plot(x, actual_flow[:n], color=COLORS["blue"], lw=1.35, label="Actual synthetic flow")
    ax.plot(
        x,
        predicted_flow[:n],
        color=COLORS["orange"],
        lw=1.15,
        ls="--",
        label="INT8 QAT-LSTM, 24-h lead",
    )
    ax.set_xlabel("Test-period day")
    ax.set_ylabel("River flow (m$^3$ s$^{-1}$)")
    ax.set_title("Twenty-four-hour-ahead flow prediction")
    ax.grid(alpha=0.20)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_stress_test(
    hour: np.ndarray,
    flow_m3s: np.ndarray,
    generation_kw: np.ndarray,
    critical_demand_kw: np.ndarray,
    noncritical_demand_kw: np.ndarray,
    critical_served_kw: np.ndarray,
    baseline_critical_served_kw: np.ndarray,
    noncritical_served_kw: np.ndarray,
    curtailment_fraction: np.ndarray,
    output_path: str | Path,
) -> None:
    _style()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(6.45, 5.8), sharex=True)

    axes[0].plot(hour, flow_m3s, color=COLORS["blue"], lw=1.4, label="Actual flow")
    axes[0].set_ylabel("Flow\n(m$^3$ s$^{-1}$)")
    axes[0].legend(frameon=False, loc="upper right")

    total_demand = critical_demand_kw + noncritical_demand_kw
    axes[1].plot(hour, generation_kw, color=COLORS["green"], lw=1.4, label="Generation")
    axes[1].plot(hour, total_demand, color=COLORS["gray"], lw=1.0, ls=":", label="Total demand")
    axes[1].plot(
        hour,
        critical_demand_kw,
        color=COLORS["red"],
        lw=1.1,
        label="Critical demand",
    )
    axes[1].fill_between(
        hour,
        noncritical_served_kw + critical_served_kw,
        critical_served_kw,
        color=COLORS["orange"],
        alpha=0.20,
        label="Non-critical served",
    )
    axes[1].set_ylabel("Power (kW)")
    axes[1].legend(frameon=False, ncol=2, loc="upper right")

    axes[2].plot(
        hour,
        critical_served_kw / np.maximum(critical_demand_kw, 1.0e-6) * 100.0,
        color=COLORS["green"],
        lw=1.4,
        label="Critical supply with MPC",
    )
    axes[2].plot(
        hour,
        baseline_critical_served_kw / np.maximum(critical_demand_kw, 1.0e-6) * 100.0,
        color=COLORS["red"],
        lw=1.1,
        ls="--",
        label="Critical supply without control",
    )
    axes[2].fill_between(
        hour,
        0.0,
        curtailment_fraction * 100.0,
        color=COLORS["orange"],
        alpha=0.25,
        label="Non-critical curtailment command",
    )
    axes[2].set_ylim(0, 106)
    axes[2].set_ylabel("Supply / command (%)")
    axes[2].set_xlabel("Stress-test hour")
    axes[2].legend(frameon=False, ncol=2, loc="lower left")

    for ax in axes:
        ax.grid(alpha=0.20)
    fig.suptitle("Look-ahead MPC response to an abrupt river-flow decline", y=0.995)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_annual_synthetic_profile(
    day: np.ndarray,
    flow_m3s: np.ndarray,
    critical_load_kw: np.ndarray,
    noncritical_load_kw: np.ndarray,
    output_path: str | Path,
) -> None:
    _style()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    daily_count = len(flow_m3s) // 24
    daily_flow = flow_m3s[: daily_count * 24].reshape(daily_count, 24).mean(axis=1)
    daily_critical = critical_load_kw[: daily_count * 24].reshape(daily_count, 24).mean(axis=1)
    daily_noncritical = noncritical_load_kw[: daily_count * 24].reshape(daily_count, 24).mean(axis=1)
    fig, axes = plt.subplots(2, 1, figsize=(6.45, 4.0), sharex=True)
    axes[0].plot(day[:daily_count], daily_flow, color=COLORS["blue"], lw=1.2)
    axes[0].set_ylabel("Mean flow\n(m$^3$ s$^{-1}$)")
    axes[0].set_title("Synthetic South Asian seasonal hydrology and tiered demand")
    axes[1].plot(day[:daily_count], daily_critical, color=COLORS["red"], lw=1.1, label="Critical")
    axes[1].plot(
        day[:daily_count],
        daily_noncritical,
        color=COLORS["orange"],
        lw=1.1,
        label="Non-critical",
    )
    axes[1].set_ylabel("Mean load (kW)")
    axes[1].set_xlabel("Day of year")
    axes[1].legend(frameon=False, ncol=2)
    for ax in axes:
        ax.grid(alpha=0.20)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
