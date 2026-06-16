from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


ROOT_DIR = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT_DIR / "result"
BENCHMARK_RESULT_DIR = RESULT_DIR / "benchmark" / "pact"

PATCH_SIZES = [1, 2, 4, 8]
DATASETS = [
    ("BPI_Challenge_2012", "BPIC12", "wo_attr"),
    ("BPI_Challenge_2012_A", "BPIC12 A", "w_attr"),
    ("BPI_Challenge_2012_Complete", "BPIC12 Complete", "wo_attr"),
    ("BPI_Challenge_2012_O", "BPIC12 O", "w_attr"),
    ("BPI_Challenge_2012_W", "BPIC12 W", "wo_attr"),
    ("BPI_Challenge_2012_W_Complete", "BPIC12 W Complete", "wo_attr"),
    ("BPI_Challenge_2013_closed_problems", "BPIC13 Closed", "w_attr"),
    ("bpi_challenge_2013_incidents", "BPIC13 Incidents", "w_attr"),
    ("env_permit", "Env Permit", "w_attr"),
    ("Helpdesk", "Helpdesk", "w_attr"),
    ("nasa", "NASA", "wo_attr"),
    ("SEPSIS", "SEPSIS", "w_attr"),
]


def _load_data() -> dict[tuple[str, str, int], dict[str, float]]:
    data: dict[tuple[str, str, int], dict[str, float]] = {}
    for dataset, _, _ in DATASETS:
        found = list(BENCHMARK_RESULT_DIR.glob(f"*/{dataset}/results.csv"))
        if not found:
            raise RuntimeError(f"Missing results.csv for dataset: {dataset}")
        for csv_path in found:
            with csv_path.open("r", encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    if row.get("fold") != "mean":
                        continue
                    patch_size = int(row["prefix_patch_size"])
                    key = (dataset, row["prefix_mode"], patch_size)
                    if key not in data:
                        data[key] = {
                            "next_act": float(row["next_activity_accuracy"]),
                            "suffix": float(row["suffix_dl_similarity"]),
                            "time": float(row["remaining_time_mae_days"]),
                        }
    return data


def _validate(data: dict[tuple[str, str, int], dict[str, float]]) -> None:
    missing = []
    for dataset, _, prefix_mode in DATASETS:
        for patch_size in PATCH_SIZES:
            if (dataset, prefix_mode, patch_size) not in data:
                missing.append(f"{dataset}/{prefix_mode}/P={patch_size}")
    if missing:
        raise RuntimeError("Missing required mean rows: " + ", ".join(missing))


def _pad_limits(values: list[float], *, lower_bound: float | None = None) -> tuple[float, float]:
    lo = min(values)
    hi = max(values)
    if abs(hi - lo) < 1e-9:
        pad = max(abs(hi) * 0.02, 0.01)
    else:
        pad = (hi - lo) * 0.18
    lo -= pad
    hi += pad
    if lower_bound is not None:
        lo = max(lower_bound, lo)
    return lo, hi


def make_figure(output: Path) -> None:
    data = _load_data()
    _validate(data)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 7,
        "axes.titlesize": 8.5,
        "axes.titleweight": "semibold",
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 8,
        "svg.fonttype": "none",
    })

    fig, axes = plt.subplots(3, 4, figsize=(12.46, 7.18), sharex=True)
    colors = {"next_act": "#2563eb", "suffix": "#16a34a", "time": "#dc2626"}

    for idx, (ax, (dataset, label, prefix_mode)) in enumerate(zip(axes.flat, DATASETS)):
        next_act = [data[(dataset, prefix_mode, patch)]["next_act"] for patch in PATCH_SIZES]
        suffix = [data[(dataset, prefix_mode, patch)]["suffix"] for patch in PATCH_SIZES]
        time = [data[(dataset, prefix_mode, patch)]["time"] for patch in PATCH_SIZES]

        ax.plot(PATCH_SIZES, next_act, color=colors["next_act"], linewidth=1.8)
        ax.plot(PATCH_SIZES, suffix, color=colors["suffix"], linewidth=1.8)
        ax.set_title(label, pad=6)
        ax.set_xscale("log", base=2)
        ax.set_xticks(PATCH_SIZES)
        ax.set_xticklabels([str(patch) for patch in PATCH_SIZES])
        ax.grid(True, axis="both", color="#e5e7eb", linewidth=0.6)
        ax.tick_params(axis="y", colors="#374151")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(*_pad_limits(next_act + suffix, lower_bound=0.0))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

        rax = ax.twinx()
        rax.plot(PATCH_SIZES, time, color=colors["time"], linewidth=1.8)
        rax.tick_params(axis="y", colors=colors["time"], labelsize=7)
        rax.spines["top"].set_visible(False)
        rax.spines["left"].set_visible(False)
        rax.spines["right"].set_color(colors["time"])
        rax.set_ylim(*_pad_limits(time, lower_bound=0.0))
        rax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        if idx % 4 == 3:
            rax.set_ylabel("time MAE (days)", color=colors["time"])

    for ax in axes[-1, :]:
        ax.set_xlabel("Patch size")
    for ax in axes[:, 0]:
        ax.set_ylabel("next / suffix")

    legend_handles = [
        plt.Line2D([0], [0], color=colors["next_act"], linewidth=1.8, label="next"),
        plt.Line2D([0], [0], color=colors["suffix"], linewidth=1.8, label="suffix"),
        plt.Line2D([0], [0], color=colors["time"], linewidth=1.8, label="time"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.tight_layout(rect=(0, 0, 1, 0.965), w_pad=2.0, h_pad=2.0)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create mixed-mode patch-size sensitivity SVG for PaCT Fig. 2.")
    parser.add_argument("--output", type=Path, default=RESULT_DIR / "patch_size_4x3_mixed_mode_dual_axis_clean.svg")
    args = parser.parse_args()
    make_figure(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
