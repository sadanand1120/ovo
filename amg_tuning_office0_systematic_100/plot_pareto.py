#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
import numpy as np


ROOT_DIR = Path("/robodata/smodak/repos/ovo/amg_tuning_office0_systematic_100")
DEFAULT_DUMP_PATH = ROOT_DIR / "experiment_dump.tsv"
DEFAULT_PLAN_PATH = ROOT_DIR / "search_plan.json"
DEFAULT_OUTPUT_PATH = ROOT_DIR / "pareto_time_vs_metric.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot runtime vs metric Pareto frontier for the SAM AMG sweep.",
    )
    parser.add_argument("--dump", type=Path, default=DEFAULT_DUMP_PATH, help="TSV dump to read.")
    parser.add_argument(
        "--plan",
        type=Path,
        default=DEFAULT_PLAN_PATH,
        help="Search-plan JSON used to recover knob names for hover text.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="PNG output path.")
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Only save the PNG without opening an interactive window.",
    )
    return parser.parse_args()


def load_knob_names(plan_path: Path) -> list[str]:
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    return list(payload["search_space"])


def load_rows(dump_path: Path, knob_names: list[str]) -> list[dict]:
    rows = []
    with dump_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row["status"] != "ok":
                continue
            parsed = dict(row)
            parsed["run_id"] = int(row["run_id"])
            parsed["batch_idx"] = int(row["batch_idx"])
            parsed["gpu_id"] = int(row["gpu_id"])
            parsed["runtime_sec"] = float(row["runtime_sec"])
            parsed["metric_value"] = float(row["metric_value"])
            for knob in knob_names:
                raw = row[knob]
                if raw.isdigit():
                    parsed[knob] = int(raw)
                else:
                    try:
                        parsed[knob] = float(raw)
                    except ValueError:
                        parsed[knob] = raw
            rows.append(parsed)
    return rows


def pareto_front_rows(rows: list[dict]) -> list[dict]:
    frontier = []
    for row in rows:
        dominated = False
        for other in rows:
            if other is row:
                continue
            if (
                other["runtime_sec"] <= row["runtime_sec"]
                and other["metric_value"] >= row["metric_value"]
                and (
                    other["runtime_sec"] < row["runtime_sec"]
                    or other["metric_value"] > row["metric_value"]
                )
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(row)
    frontier.sort(key=lambda row: (row["runtime_sec"], -row["metric_value"]))
    return frontier


def hover_text(row: dict, knob_names: list[str]) -> str:
    knob_blob = "\n".join(f"{name}={row[name]}" for name in knob_names)
    return (
        f"run {row['run_id']} ({row['role']})\n"
        f"time={row['runtime_sec']:.2f}s ({row['runtime_sec'] / 60.0:.2f}m)\n"
        f"metric={row['metric_value']:.6f}\n"
        f"batch={row['batch_idx']} gpu={row['gpu_id']}\n"
        f"{knob_blob}"
    )


def main() -> None:
    args = parse_args()
    knob_names = load_knob_names(args.plan)
    rows = load_rows(args.dump, knob_names)
    if not rows:
        raise RuntimeError(f"No successful rows found in {args.dump}")

    frontier = pareto_front_rows(rows)
    frontier_ids = {row["run_id"] for row in frontier}
    baseline = next((row for row in rows if row["role"] == "baseline"), None)

    x = np.asarray([row["runtime_sec"] / 60.0 for row in rows], dtype=np.float32)
    y = np.asarray([row["metric_value"] for row in rows], dtype=np.float32)

    colors = []
    sizes = []
    for row in rows:
        if baseline is not None and row["run_id"] == baseline["run_id"]:
            colors.append("#111111")
            sizes.append(240)
        elif row["run_id"] in frontier_ids:
            colors.append("#d95f02")
            sizes.append(130)
        else:
            colors.append("#1f77b4")
            sizes.append(72)

    plt.close("all")
    fig, ax = plt.subplots(figsize=(18, 11), dpi=180)
    ax.set_title("SAM AMG Office0 Systematic Sweep\nRuntime vs Mean AP", fontsize=18, weight="bold")
    ax.set_xlabel("Runtime (minutes)", fontsize=13)
    ax.set_ylabel("Mean AP", fontsize=13)
    ax.grid(True, which="major", alpha=0.22, linestyle="--")
    ax.grid(True, which="minor", alpha=0.10, linestyle=":")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=22))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=18))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    scatter = ax.scatter(
        x,
        y,
        c=colors,
        s=sizes,
        alpha=0.92,
        edgecolors="white",
        linewidths=0.85,
        zorder=3,
    )

    frontier_sorted = sorted(frontier, key=lambda row: row["runtime_sec"])
    ax.plot(
        [row["runtime_sec"] / 60.0 for row in frontier_sorted],
        [row["metric_value"] for row in frontier_sorted],
        color="#d95f02",
        linewidth=2.0,
        alpha=0.82,
        zorder=2,
    )

    for row in rows:
        ax.annotate(
            str(row["run_id"]),
            (row["runtime_sec"] / 60.0, row["metric_value"]),
            textcoords="offset points",
            xytext=(4, 4),
            ha="left",
            fontsize=7.5,
            color="#202020",
            alpha=0.92,
            zorder=4,
        )

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", label="All runs", markerfacecolor="#1f77b4", markersize=9),
        Line2D([0], [0], marker="o", color="w", label="Pareto frontier", markerfacecolor="#d95f02", markersize=10),
    ]
    if baseline is not None:
        legend_handles.append(
            Line2D([0], [0], marker="o", color="w", label="Baseline", markerfacecolor="#111111", markersize=12)
        )
    ax.legend(handles=legend_handles, loc="lower right", frameon=True)

    annotation = ax.annotate(
        "",
        xy=(0, 0),
        xytext=(18, 18),
        textcoords="offset points",
        bbox={"boxstyle": "round,pad=0.35", "fc": "#fffef5", "ec": "#404040", "alpha": 0.97},
        fontsize=8.5,
        visible=False,
    )

    def on_move(event) -> None:
        if event.inaxes != ax:
            if annotation.get_visible():
                annotation.set_visible(False)
                fig.canvas.draw_idle()
            return
        contains, info = scatter.contains(event)
        if not contains:
            if annotation.get_visible():
                annotation.set_visible(False)
                fig.canvas.draw_idle()
            return
        idx = info["ind"][0]
        row = rows[idx]
        annotation.xy = (x[idx], y[idx])
        annotation.set_text(hover_text(row, knob_names))
        annotation.set_visible(True)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_move)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
