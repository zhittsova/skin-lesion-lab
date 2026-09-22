"""Plot MC Dropout predictive uncertainty distributions.

Use with real MC Dropout outputs when available:

    uv run python plot_mc_dropout_uncertainty.py --mc-probs path/to/mc_probs.npy

Expected `.npy` shape is `(n_passes, n_samples)`, where each row is one
dropout-active inference pass and each value is `P(melanoma)`.
CSV uses the same orientation. Its required header is `sample_0,sample_1,...`
in column order, with one stochastic pass per following row and no index column.

If no MC output is supplied, the script creates a clearly marked illustrative
figure for slides. It is not an experimental result.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from src.uncertainty import summarize_mc_dropout_probabilities

OUT_DIR = Path(__file__).parent / "outputs"
PROJECT_FIGURES_DIR = Path(__file__).parent / "results" / "figures"

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
    "blue": "#5477C4",
    "blue_light": "#CEDFFE",
    "orange": "#CC6F47",
    "orange_light": "#FFBDA1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mc-probs",
        type=Path,
        default=None,
        help=(
            "Optional .npy or .csv with n_passes rows and n_samples columns. "
            "CSV requires sample_0,sample_1,... headers and no index column."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUT_DIR / "mc_dropout_predictive_uncertainty_distribution.png",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=OUT_DIR / "mc_dropout_predictive_uncertainty_distribution.csv",
    )
    return parser.parse_args()


def load_mc_probabilities(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        raw = np.load(path, allow_pickle=False)
    elif path.suffix == ".csv":
        with path.open(newline="") as file:
            reader = csv.reader(file, strict=True)
            header = next(reader, [])
            if not header or header != [f"sample_{i}" for i in range(len(header))]:
                raise ValueError("CSV header must be sample_0,sample_1,... without IDs")
            rows = []
            for row in reader:
                if len(row) != len(header) or any(value.strip() == "" for value in row):
                    raise ValueError("CSV has an empty or ragged MC pass")
                try:
                    rows.append([float(value) for value in row])
                except ValueError as error:
                    raise ValueError("CSV probabilities must be numeric") from error
        raw = np.asarray(rows, dtype=float)
    else:
        raise ValueError("Use a .npy or .csv file for MC probabilities")
    try:
        values = np.asarray(raw, dtype=float)
        summarize_mc_dropout_probabilities(values)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "MC probabilities need at least two finite passes in [0, 1]"
        ) from error
    return values


def make_illustrative_mc_probabilities(seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Create plausible MC samples for a slide-only illustration."""
    rng = np.random.default_rng(seed)
    n_passes = 30
    n_low = 1250
    n_high = 360
    n_borderline = 260
    n_error_like = 130

    means = np.concatenate(
        [
            rng.beta(1.2, 12, size=n_low),
            rng.beta(9, 1.8, size=n_high),
            rng.uniform(0.35, 0.65, size=n_borderline),
            rng.uniform(0.15, 0.85, size=n_error_like),
        ]
    )
    sigmas = np.concatenate(
        [
            rng.uniform(0.005, 0.025, size=n_low),
            rng.uniform(0.006, 0.035, size=n_high),
            rng.uniform(0.045, 0.10, size=n_borderline),
            rng.uniform(0.08, 0.16, size=n_error_like),
        ]
    )

    samples = rng.normal(loc=means, scale=sigmas, size=(n_passes, len(means)))
    samples = np.clip(samples, 0.0, 1.0)

    groups = np.array(
        ["low uncertainty"] * (n_low + n_high)
        + ["near threshold"] * n_borderline
        + ["high uncertainty"] * n_error_like
    )
    return samples, groups


def build_uncertainty_frame(
    mc_probabilities: np.ndarray, groups: np.ndarray | None, source: str
) -> pd.DataFrame:
    summary = summarize_mc_dropout_probabilities(mc_probabilities)
    uncertainty = summary["variance"] ** 0.5
    frame = pd.DataFrame(
        {
            "mean_prob_melanoma": summary["mean_prob_melanoma"],
            "predictive_uncertainty": uncertainty,
            "predictive_entropy": summary["predictive_entropy"],
            "mutual_information": summary["mutual_information"],
            "source": source,
        }
    )
    if groups is None:
        frame["group"] = np.where(
            np.abs(frame["mean_prob_melanoma"] - 0.5) <= 0.05,
            "near threshold",
            "all cases",
        )
    else:
        frame["group"] = groups
    return frame


def plot_distribution(
    frame: pd.DataFrame, output_path: Path, illustrative: bool
) -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": TOKENS["surface"],
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "xtick.color": TOKENS["muted"],
            "ytick.color": TOKENS["muted"],
            "text.color": TOKENS["ink"],
            "font.family": "DejaVu Sans",
        }
    )

    fig, ax = plt.subplots(figsize=(11, 6.2), dpi=180)

    if illustrative:
        palette = {
            "low uncertainty": TOKENS["blue_light"],
            "near threshold": TOKENS["orange_light"],
            "high uncertainty": TOKENS["orange"],
        }
        order = ["low uncertainty", "near threshold", "high uncertainty"]
        for group in order:
            subset = frame[frame["group"] == group]
            sns.histplot(
                subset,
                x="predictive_uncertainty",
                bins=np.linspace(0, 0.18, 28),
                stat="density",
                element="step",
                fill=True,
                alpha=0.45,
                color=palette[group],
                edgecolor=palette[group],
                linewidth=1.4,
                label=group,
                ax=ax,
            )
    else:
        sns.histplot(
            frame,
            x="predictive_uncertainty",
            bins=28,
            stat="count",
            color=TOKENS["blue"],
            edgecolor=TOKENS["blue"],
            alpha=0.75,
            ax=ax,
        )

    median = frame["predictive_uncertainty"].median()
    p90 = frame["predictive_uncertainty"].quantile(0.9)
    ax.axvline(median, color=TOKENS["ink"], linestyle="--", linewidth=1.4)
    ax.axvline(p90, color=TOKENS["orange"], linestyle="--", linewidth=1.8)
    ax.text(median, ax.get_ylim()[1] * 0.92, f"median\n{median:.3f}", ha="center")
    ax.text(p90, ax.get_ylim()[1] * 0.78, f"90th pct.\n{p90:.3f}", ha="center")

    title = "Distribution of predictive uncertainties (MC Dropout)"
    subtitle = (
        "Illustrative slide figure: uncertainty is the standard deviation of "
        "P(melanoma) across stochastic dropout passes."
        if illustrative
        else "Uncertainty is the standard deviation of P(melanoma) across stochastic dropout passes."
    )
    fig.text(0.08, 0.965, title, fontsize=18, fontweight="bold", ha="left")
    fig.text(0.08, 0.92, subtitle, fontsize=10.5, color=TOKENS["muted"], ha="left")

    ax.set_xlabel("Predictive uncertainty: std of MC melanoma probability")
    ax.set_ylabel("Density" if illustrative else "Number of lesions")
    ax.grid(axis="y", color=TOKENS["grid"])
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper right")

    note = (
        "Use as concept figure only unless real MC Dropout probabilities are supplied."
        if illustrative
        else "Source: supplied MC Dropout probabilities."
    )
    fig.text(0.08, 0.045, note, fontsize=9.5, color=TOKENS["muted"], ha="left")
    fig.subplots_adjust(top=0.84, bottom=0.18, left=0.09, right=0.96)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.mc_probs is None:
        mc_probabilities, groups = make_illustrative_mc_probabilities()
        illustrative = True
        source = "illustrative"
    else:
        mc_probabilities = load_mc_probabilities(args.mc_probs)
        groups = None
        illustrative = False
        source = str(args.mc_probs)

    frame = build_uncertainty_frame(mc_probabilities, groups, source)
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.csv_output, index=False)
    plot_distribution(frame, args.output, illustrative)

    PROJECT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    project_copy = PROJECT_FIGURES_DIR / args.output.name
    plot_distribution(frame, project_copy, illustrative)

    print(f"Saved chart: {args.output}")
    print(f"Saved project chart copy: {project_copy}")
    print(f"Saved data: {args.csv_output}")


if __name__ == "__main__":
    main()
