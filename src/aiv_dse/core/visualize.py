"""Pareto front visualization.

Generates a PNG with three subplots showing pairwise tradeoffs
(latency vs area, latency vs power, area vs power).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional


def plot_pareto_front(
    all_points: List[Dict[str, Any]],
    front_points: List[Dict[str, Any]],
    selected_point: Optional[Dict[str, Any]],
    output_path: str,
) -> str:
    """Plot the Pareto front and save as PNG.

    Args:
        all_points: Every explored point (including VETO).
        front_points: Non-dominated subset (APPROVED only).
        selected_point: The weight-selected best point (red star), or None.
        output_path: Destination PNG path.

    Returns:
        The resolved output path string.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pairs = [
        ("latency_ns", "area_units"),
        ("latency_ns", "power_mw"),
        ("area_units", "power_mw"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, (xkey, ykey) in zip(axes, pairs):
        # All explored points (grey)
        xs = [p["metrics"].get(xkey, 0) for p in all_points]
        ys = [p["metrics"].get(ykey, 0) for p in all_points]
        ax.scatter(xs, ys, c="grey", alpha=0.4, s=20, label="explored")

        # Front points (blue with black edge)
        if front_points:
            fxs = [p["metrics"].get(xkey, 0) for p in front_points]
            fys = [p["metrics"].get(ykey, 0) for p in front_points]
            ax.scatter(fxs, fys, c="blue", edgecolors="black",
                       s=60, zorder=3, label="front")

        # Selected point (red star)
        if selected_point:
            sx = selected_point["metrics"].get(xkey, 0)
            sy = selected_point["metrics"].get(ykey, 0)
            ax.scatter([sx], [sy], c="red", marker="*",
                       s=200, zorder=4, label="selected")

        ax.set_xlabel(xkey)
        ax.set_ylabel(ykey)
        ax.legend(fontsize=7)

    fig.suptitle("Pareto Front", fontsize=12)
    fig.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=100)
    plt.close(fig)

    return str(Path(output_path).resolve())
