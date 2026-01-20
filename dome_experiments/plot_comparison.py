#!/usr/bin/env python3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# -------- HARD-CODED INPUT --------
CSV_PATH = Path("training_metrics_fashionmnist_noise_2411.csv")

# -------- CONFIG --------
BOOL_COLS   = ["use_random_sketching", "use_preconditioning",
               "use_sketching", "debias", "use_numertrick"]
# include use_sketching in the grouping so we pick best epoch per seed separately
METHOD_KEYS = ["dp_scale", "use_sketching", "use_random_sketching",
               "use_preconditioning", "use_numertrick"]

def booleanize(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.lower().isin(["1", "true", "t", "yes", "y"])
    return df

def best_epoch_per_seed(df):
    # best accuracy per (method, compression_rate, seed)
    keys = METHOD_KEYS + ["compression_rate", "seed"]
    idx  = df.groupby(keys)["accuracy"].idxmax()
    return df.loc[idx].copy()

def make_label(row):
    # Baseline label not used in seaborn curves (handled separately), but keep for completeness
    if not row["use_sketching"]:
        return "No Sketching"
    if row["use_random_sketching"]:
        return "Random Sketching"
    # Our correlation-aware without preconditioning is the DOME method
    return "DOME"

def compute_no_sketching_baseline(df_best):
    """
    For use_sketching == False, compute one accuracy per seed (take the max over any
    compression_rate) then average across seeds to get the horizontal level per dp_scale.
    Returns dict: dp_scale -> mean accuracy.
    """
    base = df_best[df_best["use_sketching"] == False]
    if base.empty:
        return {}

    # One value per (dp_scale, seed)
    per_seed = (base.groupby(["dp_scale", "seed"])["accuracy"]
                    .max()
                    .reset_index())

    # Mean across seeds per dp_scale
    return per_seed.groupby("dp_scale")["accuracy"].mean().to_dict()

def main():
    df = booleanize(pd.read_csv(CSV_PATH), BOOL_COLS)
    df[BOOL_COLS] = df[BOOL_COLS].fillna(False)

    df_best = best_epoch_per_seed(df)

    # pretty labels for the sketching curves
    df_best["pair"] = df_best.apply(make_label, axis=1)

    # --- Only plot sketching results WITHOUT preconditioning ---
    # Keep random sketching and correlation-aware w/o precond (DOME). Drop any with preconditioning=True.
    df_sketch = df_best[(df_best["use_sketching"] == True) & (df_best["use_preconditioning"] == False)].copy()

    sns.set_theme(style="whitegrid")

    # Facet by dp_scale in columns, and by use_numertrick in rows (separate panels)
    g = sns.relplot(
        data=df_sketch,
        x="compression_rate",
        y="accuracy",
        hue="pair",
        kind="line",
        col="dp_scale",
        row="use_numertrick",
        marker="o",
        linestyle="-",
        linewidth=2.6,
        markersize=8,
        estimator="mean",       # seaborn <=0.11
        errorbar=("ci", 95),    # seaborn >=0.12 (use ci=95 for older)
        facet_kws=dict(sharey=True, sharex=True),
    )

    # Log-scale on x, common y-scale in [0,1]
    g.set(xscale="log")
    g.set_axis_labels("Compression rate", "Accuracy")
    g.set(ylim=(0, 1))

    # Nice facet titles
    g.set_titles(row_template="use_numertrick = {row_name}", col_template="dp_scale = {col_name}")

    # --- add the horizontal "No Compression" baseline lines in every facet ---
    baseline_mean = compute_no_sketching_baseline(df_best)
    no_sk_label = "No Compression"
    baseline_handle = None

    # g.axes is 2D when both row and col are used
    if hasattr(g, "axes") and g.axes is not None:
        for i, row_val in enumerate(getattr(g, "row_names", [None])):
            for j, col_val in enumerate(getattr(g, "col_names", [None])):
                ax = g.axes[i, j]
                if col_val in baseline_mean:
                    y = baseline_mean[col_val]
                    xmin, xmax = 10, 1000
                    (baseline_handle,) = ax.plot(
                        [xmin, xmax], [y, y],
                        linestyle="--",
                        color="black",
                        linewidth=2.2,
                        label=no_sk_label
                    )

    # ---------- Legend above, boxed (+ add the baseline line) ----------
    # Grab handles/labels from the first visible axis
    first_ax = g.axes.flat[0] if hasattr(g.axes, "flat") else g.ax
    handles, labels = first_ax.get_legend_handles_labels()
    if baseline_handle is not None:
        handles.append(baseline_handle)
        labels.append(no_sk_label)

    # Deduplicate while preserving order
    uniq = {}
    for h, l in zip(handles, labels):
        if l not in uniq:
            uniq[l] = h
    labels, handles = list(uniq.keys()), list(uniq.values())

    if g._legend:
        g._legend.remove()

    legend = g.fig.legend(
        handles, labels,
        loc="upper center",
        ncol=len(labels),
        frameon=True,
        bbox_to_anchor=(0.5, 0.98),
        borderaxespad=0.3,
    )
    frame = legend.get_frame()
    frame.set_edgecolor("black")
    frame.set_linewidth(0.9)
    frame.set_alpha(0.95)

    # Tight layout with space for legend
    g.fig.tight_layout(rect=[0, 0, 1, 0.93])

    plt.savefig("training_metrics_fashionmnist_2111.pdf")
    plt.show()

if __name__ == "__main__":
    main()
