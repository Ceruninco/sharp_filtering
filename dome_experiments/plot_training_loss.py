#!/usr/bin/env python3
"""
Plot (1) training loss vs step, and (2) mean-gradient fraction vs step (EMA),
for BOTH filtered and unfiltered runs.

Convention:
  - filtered files:      <name>.csv
  - unfiltered files:    <name>_unfiltered.csv

Expected CSV schemas:
  - loss CSV:       columns ["step", "loss"]
  - fractions CSV:  columns ["step", "fraction_mean"] (can also include "fraction")

Outputs:
  - <outdir>/training_mnist.pdf
  - <outdir>/fraction_mnist.pdf
"""

import argparse
import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def _ensure_outdir(outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)


def _with_unfiltered_suffix(path: str) -> str:
    """
    Insert '_unfiltered' before the extension.
    e.g., 'foo.csv' -> 'foo_unfiltered.csv'
    """
    base, ext = os.path.splitext(path)
    return f"{base}_unfiltered{ext}"


def _load_loss(loss_csv: str) -> pd.DataFrame:
    df = pd.read_csv(loss_csv)
    required = {"step", "loss"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{loss_csv} is missing columns: {sorted(missing)}")
    df = df[["step", "loss"]].copy()
    df["step"] = pd.to_numeric(df["step"])
    df["loss"] = pd.to_numeric(df["loss"])
    return df.sort_values("step")


def _load_fractions(frac_csv: str) -> pd.DataFrame:
    df = pd.read_csv(frac_csv)
    required = {"step", "fraction_mean"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{frac_csv} is missing columns: {sorted(missing)}")
    df = df[["step", "fraction_mean"]].copy()
    df["step"] = pd.to_numeric(df["step"])
    df["fraction_mean"] = pd.to_numeric(df["fraction_mean"])
    return df.sort_values("step")


def _apply_ema(series: pd.Series, alpha: float) -> pd.Series:
    if not (0.0 < alpha <= 1.0):
        raise ValueError("--ema-alpha must be in (0, 1].")
    # Standard EMA recursion: y_t = alpha x_t + (1-alpha) y_{t-1}
    return series.ewm(alpha=alpha, adjust=False).mean()


def _set_icml_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "lines.linewidth": 2.0,
        }
    )


def plot_training_loss_both(
    loss_filtered: pd.DataFrame,
    loss_unfiltered: pd.DataFrame,
    outpath: str,
    smooth_loss_window: int = 0,
) -> None:
    _set_icml_style()

    df_f = loss_filtered.copy()
    df_u = loss_unfiltered.copy()

    if smooth_loss_window and smooth_loss_window > 1:
        df_f["loss_plot"] = df_f["loss"].rolling(smooth_loss_window, min_periods=1).mean()
        df_u["loss_plot"] = df_u["loss"].rolling(smooth_loss_window, min_periods=1).mean()
        y_label = f"Train loss (MA{smooth_loss_window})"
    else:
        df_f["loss_plot"] = df_f["loss"]
        df_u["loss_plot"] = df_u["loss"]
        y_label = "Train loss"

    df_f["method"] = "DOME (filtered)"
    df_u["method"] = "Baseline (unfiltered)"
    df = pd.concat([df_f[["step", "loss_plot", "method"]], df_u[["step", "loss_plot", "method"]]], ignore_index=True)

    fig, ax = plt.subplots(figsize=(4, 2.4))
    sns.lineplot(data=df, x="step", y="loss_plot", hue="method", ax=ax)

    ax.set_xlabel("Step")
    ax.set_ylabel(y_label)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_fraction_mean_ema_both(
    frac_filtered: pd.DataFrame,
    frac_unfiltered: pd.DataFrame,
    outpath: str,
    ema_alpha: float,
    show_raw: bool = False,
) -> None:
    _set_icml_style()

    df_f = frac_filtered.copy()
    df_u = frac_unfiltered.copy()

    df_f["fraction_mean_ema"] = _apply_ema(df_f["fraction_mean"], alpha=ema_alpha)
    df_u["fraction_mean_ema"] = _apply_ema(df_u["fraction_mean"], alpha=ema_alpha)

    df_f["method"] = "DOME (filtered)"
    df_u["method"] = "Baseline (unfiltered)"

    if show_raw:
        df_f_raw = df_f[["step", "fraction_mean", "method"]].rename(columns={"fraction_mean": "value"})
        df_f_raw["curve"] = "Raw"
        df_u_raw = df_u[["step", "fraction_mean", "method"]].rename(columns={"fraction_mean": "value"})
        df_u_raw["curve"] = "Raw"

        df_f_ema = df_f[["step", "fraction_mean_ema", "method"]].rename(columns={"fraction_mean_ema": "value"})
        df_f_ema["curve"] = f"EMA (alpha={ema_alpha:g})"
        df_u_ema = df_u[["step", "fraction_mean_ema", "method"]].rename(columns={"fraction_mean_ema": "value"})
        df_u_ema["curve"] = f"EMA (alpha={ema_alpha:g})"

        df = pd.concat([df_f_raw, df_u_raw, df_f_ema, df_u_ema], ignore_index=True)

        fig, ax = plt.subplots(figsize=(4, 2.4))
        # hue = method, style = raw vs ema
        sns.lineplot(data=df, x="step", y="value", hue="method", style="curve", ax=ax)
    else:
        df = pd.concat(
            [
                df_f[["step", "fraction_mean_ema", "method"]].rename(columns={"fraction_mean_ema": "value"}),
                df_u[["step", "fraction_mean_ema", "method"]].rename(columns={"fraction_mean_ema": "value"}),
            ],
            ignore_index=True,
        )
        fig, ax = plt.subplots(figsize=(4, 2.4))
        sns.lineplot(data=df, x="step", y="value", hue="method", ax=ax)
        ax.legend(title=None)

    ax.set_xlabel("Step")
    ax.set_ylabel("Grad. fract. in subspace")
    ax.set_ylim(0.0, 1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot training loss and mean-gradient fraction (EMA) vs step for filtered and unfiltered runs."
    )
    parser.add_argument(
        "--loss-csv",
        type=str,
        required=True,
        help="Filtered loss CSV path (expects <path>_unfiltered.csv alongside).",
    )
    parser.add_argument(
        "--fractions-csv",
        type=str,
        required=True,
        help="Filtered fractions CSV path (expects <path>_unfiltered.csv alongside).",
    )
    parser.add_argument(
        "--smooth-loss-window",
        type=int,
        default=0,
        help="Moving average window for loss (0 disables).",
    )
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=0.1,
        help="EMA alpha in (0,1]. Smaller = smoother.",
    )
    parser.add_argument(
        "--show-raw-fraction",
        action="store_true",
        help="Also plot raw fraction_mean alongside EMA for both methods.",
    )
    parser.add_argument("--outdir", type=str, default="plots", help="Output directory")
    args = parser.parse_args()

    _ensure_outdir(args.outdir)

    loss_csv_u = _with_unfiltered_suffix(args.loss_csv)
    frac_csv_u = _with_unfiltered_suffix(args.fractions_csv)

    if not os.path.isfile(args.loss_csv):
        raise FileNotFoundError(f"Filtered loss CSV not found: {args.loss_csv}")
    if not os.path.isfile(loss_csv_u):
        raise FileNotFoundError(f"Unfiltered loss CSV not found: {loss_csv_u}")

    if not os.path.isfile(args.fractions_csv):
        raise FileNotFoundError(f"Filtered fractions CSV not found: {args.fractions_csv}")
    if not os.path.isfile(frac_csv_u):
        raise FileNotFoundError(f"Unfiltered fractions CSV not found: {frac_csv_u}")

    loss_f = _load_loss(args.loss_csv)
    loss_u = _load_loss(loss_csv_u)

    frac_f = _load_fractions(args.fractions_csv)
    frac_u = _load_fractions(frac_csv_u)

    loss_out = os.path.join(args.outdir, "training_cifar10.pdf")
    frac_out = os.path.join(args.outdir, "fraction_cifar10.pdf")

    plot_training_loss_both(
        loss_filtered=loss_f,
        loss_unfiltered=loss_u,
        outpath=loss_out,
        smooth_loss_window=args.smooth_loss_window,
    )
    plot_fraction_mean_ema_both(
        frac_filtered=frac_f,
        frac_unfiltered=frac_u,
        outpath=frac_out,
        ema_alpha=args.ema_alpha,
        show_raw=args.show_raw_fraction,
    )

    print(f"Saved:\n  {loss_out}\n  {frac_out}")


if __name__ == "__main__":
    main()
