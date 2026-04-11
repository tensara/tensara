import json
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


def _runtime_formatter(value: float, _: int) -> str:
    if value < 1:
        return f"{value * 1000:.0f}μs"
    return f"{value:.2f}ms"


def _gflops_formatter(value: float, _: int) -> str:
    return f"{value:.1f}"


def _temperature_formatter(value: float, _: int) -> str:
    return f"{value:.1f}°C"


def _load_payload() -> dict[str, Any]:
    raw = sys.stdin.read()
    if not raw.strip():
        raise ValueError("No plot payload received")
    return json.loads(raw)


def main() -> int:
    payload = _load_payload()

    categories = payload.get("categories", [])
    series = payload.get("series", [])
    metric_mode = payload.get("metricMode", "runtime")
    y_label = payload.get("yLabel", "")
    title = payload.get("title", "Tensara Analysis")
    subtitle = payload.get("subtitle", "")
    x_label = payload.get("xLabel", "")

    if not categories or not series:
        raise ValueError("Plot payload requires categories and series")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.edgecolor": "#475569",
            "axes.labelcolor": "#e2e8f0",
            "xtick.color": "#cbd5e1",
            "ytick.color": "#cbd5e1",
            "text.color": "#f8fafc",
            "axes.titlecolor": "#f8fafc",
            "svg.fonttype": "none",
        }
    )

    fig, ax = plt.subplots(figsize=(13.2, 7.4), dpi=160)
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#181f2a")

    x = np.arange(len(categories))
    has_any_points = False

    for entry in series:
        points = [
            np.nan if point is None else float(point)
            for point in entry.get("points", [])
        ]
        if any(not np.isnan(point) for point in points):
            has_any_points = True

        ax.plot(
            x,
            points,
            color=entry.get("color", "#38bdf8"),
            label=entry.get("label", "Series"),
            linewidth=2.6,
            marker="o",
            markersize=7.5,
            markeredgewidth=1.6,
            markeredgecolor="#0f172a",
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=3,
        )

    if not has_any_points:
        raise ValueError("No plottable benchmark values found")

    formatter = {
        "runtime": FuncFormatter(_runtime_formatter),
        "gflops": FuncFormatter(_gflops_formatter),
        "temperature": FuncFormatter(_temperature_formatter),
    }.get(metric_mode, FuncFormatter(_runtime_formatter))

    ax.yaxis.set_major_formatter(formatter)
    ax.grid(axis="y", color="#334155", alpha=0.55, linewidth=0.8)
    ax.grid(axis="x", color="#1e293b", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#475569")
        ax.spines[spine].set_linewidth(1.1)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=24, ha="right")
    ax.set_xlabel(x_label, labelpad=14, color="#cbd5e1")
    ax.set_ylabel(y_label, labelpad=14, color="#cbd5e1")
    ax.tick_params(axis="x", pad=10)
    ax.tick_params(axis="y", pad=8)
    ax.margins(x=0.03)

    fig.text(
        0.055,
        0.95,
        title,
        fontsize=19,
        fontweight="bold",
        color="#f8fafc",
        ha="left",
        va="top",
    )

    if subtitle:
        fig.text(
            0.055,
            0.915,
            subtitle,
            fontsize=10.5,
            color="#94a3b8",
            ha="left",
            va="top",
        )

    legend = ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.0, -0.22),
        ncol=2,
        frameon=True,
        fancybox=False,
        framealpha=1,
        fontsize=10,
        borderpad=0.7,
        labelspacing=0.7,
        handlelength=2.2,
    )
    legend.get_frame().set_facecolor("#111827")
    legend.get_frame().set_edgecolor("#334155")
    legend.get_frame().set_linewidth(0.9)

    plt.subplots_adjust(left=0.09, right=0.985, top=0.82, bottom=0.28)
    fig.savefig(
        sys.stdout.buffer,
        format="svg",
        facecolor=fig.get_facecolor(),
        bbox_inches="tight",
    )
    plt.close(fig)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
