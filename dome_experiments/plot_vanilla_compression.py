#!/usr/bin/env python3
import argparse

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_and_prepare(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required_cols = [
        "dataset",
        "epoch",
        "batch_size",
        "accuracy",
        "compression_rate",
        "debias",
        "remove_dom",
        "seed",
    ]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}' in CSV.")

    # Normalize remove_dom to boolean if possible
    df = df.copy()
    df["remove_dom"] = (
        df["remove_dom"]
        .astype(str)
        .str.strip()
        .map({"True": True, "False": False, "1": True, "0": False})
        .fillna(df["remove_dom"])
    )

    # Only keep the two methods we care about: remove_dom True/False
    df = df[df["remove_dom"].isin([True, False])].copy()

    # Define method label
    method_map = {
        True: "DOME (with filtering)",
        False: "Adam",
    }
    df["method"] = df["remove_dom"].map(method_map)

    return df


def compute_best_per_seed(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (dataset, dp_scale, method, seed), select the row with the
    best accuracy over epochs.
    """
    group_cols = ["dataset", "compression_rate", "method", "seed"]
    idx = df.groupby(group_cols)["accuracy"].idxmax()
    best = df.loc[idx].copy()
    return best


def plot_best_vs_dp_scale(best_df: pd.DataFrame, output_path: str, dataset=None):
    """
    Plot best accuracy vs dp_scale, averaging across seeds,
    with seaborn CI, comparing methods (Proposed vs DP-Adam).
    Optionally filter to a single dataset.
    """
    if dataset is not None:
        best_df = best_df[best_df["dataset"] == dataset].copy()
        if best_df.empty:
            raise ValueError(f"No rows left after filtering to dataset='{dataset}'")

    # ---- Styling: larger fonts & thicker lines ----
    sns.set_style("whitegrid")
    sns.set_context(
        "talk",  # alternatives: paper < notebook < talk < poster
        font_scale=1.2
    )

    plt.figure(figsize=(8, 6))

    sns.lineplot(
        data=best_df,
        x="compression_rate",
        y="accuracy",
        hue="method",
        marker="o",
        ci=95,
        linewidth=2.5,     # thicker lines
        markersize=9,      # larger markers
    )

    plt.xlabel("Compression rate", fontsize=16)
    plt.ylabel("Best accuracy over epochs", fontsize=16)

    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.xscale("log")
    plt.legend(
        title="Method",
        title_fontsize=20,
        fontsize=18,
        frameon=True,
    )

    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=300)  # higher DPI for papers/slides
        print(f"Saved plot to {output_path}")
        plt.show()
    else:
        plt.show()

    plt.close()



def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot best epoch accuracy for remove_dom=True (proposed) "
            "and remove_dom=False (DP-Adam) across dp_scale, "
            "averaging across seeds with seaborn CI."
        )
    )
    parser.add_argument("--csv_path", type=str, help="Path to the CSV file")
    parser.add_argument(
        "--output",
        type=str,
        default="best_vs_compression_rate.pdf",
        help="Output image path (PNG, PDF, etc.)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Optional dataset name to filter on (default: use all datasets).",
    )
    args = parser.parse_args()

    df = load_and_prepare(args.csv_path)
    best_df = compute_best_per_seed(df)
    plot_best_vs_dp_scale(best_df, args.output, args.dataset)


if __name__ == "__main__":
    main()
