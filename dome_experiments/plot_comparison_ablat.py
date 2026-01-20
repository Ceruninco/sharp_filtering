#!/usr/bin/env python3
import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt


def define_method_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'method' column with the three methods of interest:

    1. Random sketch = True, rescaling = True      -> "RS + Rescaling"
    2. Random sketch = True, rescaling = False     -> "RS + No Rescaling"
    3. Random sketch = False, rescaling = True     -> "No RS + Rescaling"

    Rows that do not match these patterns are dropped.
    """
    # Initialize with NaN
    df = df.copy()
    df["method"] = None

    mask_rs_rescale = (df["use_random_sketching"] == True) & (df["use_rescaling"] == True)
    mask_rs_norescale = (df["use_random_sketching"] == True) & (df["use_rescaling"] == False)
    mask_nors_rescale = (df["use_random_sketching"] == False) & (df["use_rescaling"] == True)

    df.loc[mask_rs_rescale, "method"] = "DOME (mom. rescal.)"
    df.loc[mask_rs_norescale, "method"] = "Random Sketching"
    df.loc[mask_nors_rescale, "method"] = "DOME (mom. rescal. + PCA)"

    # Keep only rows belonging to one of the three methods
    df = df[df["method"].notna()].copy()

    return df


def compute_best_per_compression(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (dp_scale, method, compression_rate, seed, dataset), pick
    the row with the best accuracy over epochs.

    Then average across seeds (and optionally datasets, if multiple)
    to get a single accuracy per (dp_scale, method, compression_rate).
    """
    # Best per (dp_scale, dataset, method, compression_rate, seed)
    group_cols = ["dp_scale", "dataset", "method", "compression_rate", "seed"]
    idx = df.groupby(group_cols)["accuracy"].idxmax()
    best = df.loc[idx].copy()

    # Now average across seeds (and datasets, if you have multiple) for plotting
    agg_cols = ["dp_scale", "method", "compression_rate"]
    best_mean = (
        best.groupby(agg_cols)["accuracy"]
        .mean()
        .reset_index()
        .sort_values(["dp_scale", "method", "compression_rate"])
    )
    return best_mean


def plot_per_dp_scale(best_df: pd.DataFrame, output_dir: str):
    """
    For each dp_scale, produce a plot:

    x-axis: compression_rate
    y-axis: best accuracy
    one line per method
    """
    os.makedirs(output_dir, exist_ok=True)

    dp_scales = sorted(best_df["dp_scale"].unique())

    methods_order = ["DOME (mom. rescal. + PCA)", "DOME (mom. rescal.)", "Random Sketching"]

    for dp in dp_scales:
        sub = best_df[best_df["dp_scale"] == dp]

        if sub.empty:
            continue

        plt.figure(figsize=(6, 4))

        for method in methods_order:
            sub_m = sub[sub["method"] == method]
            if sub_m.empty:
                continue

            sub_m = sub_m.sort_values("compression_rate")
            plt.plot(
                sub_m["compression_rate"],
                sub_m["accuracy"],
                marker="o",
                label=method,
            )

        plt.xlabel("Compression rate")
        plt.ylabel("Best accuracy over epochs")
        plt.xscale("log")
        plt.title(f"Best accuracy vs compression (dp_scale = {dp})")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()

        out_path = os.path.join(output_dir, f"dp_{dp}.pdf")
        plt.savefig(out_path, dpi=200)
        plt.show()
        # plt.close()
        print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot best epoch per method per compression factor, "
            "one plot per DP scale."
        )
    )
    # parser.add_argument("csv_path", type=str, help="Path to the CSV file")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="plots_dp_compression",
        help="Directory to save output plots",
    )
    args = parser.parse_args()

    csv_path = "../results/training_metrics_cifar10_ablation.csv"
    df = pd.read_csv(csv_path)

    # Ensure required columns exist
    required_cols = [
        "dataset",
        "epoch",
        "batch_size",
        "use_random_sketching",
        "accuracy",
        "compression_rate",
        "use_sketching",
        "dp_scale",
        "use_preconditioning",
        "debias",
        "use_rescaling",
        "seed",
    ]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found in CSV.")

    df = define_method_column(df)
    best_df = compute_best_per_compression(df)
    plot_per_dp_scale(best_df, args.output_dir)


if __name__ == "__main__":
    main()
