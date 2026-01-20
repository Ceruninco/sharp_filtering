#!/usr/bin/env python3

import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def compute_baseline(baseline_csv: str) -> float:
    """
    Baseline:
      - remove_dom = False
      - compression_rate = 5000
      - best accuracy per seed, then average over seeds
    """
    df = pd.read_csv(baseline_csv)

    required_cols = {
        "epoch",
        "accuracy",
        "compression_rate",
        "seed",
        "remove_dom",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in baseline CSV: {missing}")

    df = df[
        (df["compression_rate"] == 5000) &
        (df["remove_dom"] == False)
    ]

    best_per_seed = (
        df.groupby("seed", as_index=False)
          .agg(best_accuracy=("accuracy", "max"))
    )

    return best_per_seed["best_accuracy"].mean()


def main(csv_path: str, baseline_csv: str, out_path: str):
    # ------------------------------------------------------------------
    # Load main CSV (rank ablation)
    # ------------------------------------------------------------------
    df = pd.read_csv(csv_path)

    required_cols = {
        "epoch",
        "accuracy",
        "compression_rate",
        "seed",
        "nb_dims_pca",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # ------------------------------------------------------------------
    # Filter compression rate for ablation
    # ------------------------------------------------------------------
    df = df[df["compression_rate"] == 5000]

    # ------------------------------------------------------------------
    # Best accuracy per (nb_dims_pca, seed)
    # ------------------------------------------------------------------
    best_df = (
        df.groupby(["nb_dims_pca", "seed"], as_index=False)
          .agg(best_accuracy=("accuracy", "max"))
    )

    # ------------------------------------------------------------------
    # Compute baseline
    # ------------------------------------------------------------------
    baseline_acc = compute_baseline(baseline_csv)

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    sns.set(style="whitegrid", context="talk")

    plt.figure(figsize=(7, 5))

    sns.lineplot(
        data=best_df,
        x="nb_dims_pca",
        y="best_accuracy",
        marker="o",
        ci=95,
        label="Filtered",
    )

    plt.axhline(
        y=baseline_acc,
        linestyle="--",
        linewidth=4,
        label="Unfiltered baseline",
    )

    plt.xlabel("Filter dimension")
    plt.ylabel("Best accuracy")
    plt.xscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.show()
    plt.close()

    print(f"Saved figure to {out_path}")
    print(f"Baseline accuracy = {baseline_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="../results/training_metrics_mnist_study_rank.csv",
        help="Rank ablation CSV",
    )
    parser.add_argument(
        "--baseline_csv",
        default="../results/training_metrics_mnist_compression.csv",
        help="Baseline CSV",
    )
    parser.add_argument(
        "--out",
        default="../results/compression_k_ablation.pdf",
    )
    args = parser.parse_args()

    main(args.csv, args.baseline_csv, args.out)
