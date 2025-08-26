"""
Build a "confusion-matrix" style heatmap of **per-cluster average CLAP prompt scores**
using the **actual prompt names** from your raw CLAP CSVs.

This version plots clusters on the X-axis and prompts on the Y-axis, with
narrower columns so the cells are not overly wide.
"""
from __future__ import annotations
import argparse
import os
import pickle
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

CSV_1 = "eda/clap_similarity_scores_full_prompts_2.csv"
CSV_2 = "eda/clap_similarity_scores_full_prompts_1.csv"
JSON_1 = "preprocessing/labelled_data_01_08_2025.json"
JSON_2 = "preprocessing/labelled_data_20_06_2025.json"

try:
    from tools.data_utils import load_and_merge_clap_results
except ModuleNotFoundError:
    import sys
    sys.path.append(os.path.dirname(__file__))
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from tools.data_utils import load_and_merge_clap_results


def load_payload(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def get_labels_and_files(payload: dict) -> Tuple[np.ndarray, List[str]]:
    labels = payload.get("kmeans_labels")
    if labels is None:
        labels = payload.get("hier_labels")
    if labels is None:
        raise KeyError("Could not find cluster labels (kmeans_labels or hier_labels) in pickle")

    files = payload.get("filenames")
    if files is None:
        raise KeyError("Could not find 'filenames' in pickle")

    return np.asarray(labels).ravel(), list(files)


def rebuild_features(files_wanted: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    df1 = load_and_merge_clap_results(json_path=JSON_1, csv_path=CSV_1)
    df2 = load_and_merge_clap_results(json_path=JSON_2, csv_path=CSV_2)
    df = pd.concat([df1, df2], ignore_index=True)

    if "file" not in df.columns:
        raise KeyError("Merged dataframe missing 'file' column; check data_utils.merge function")
    df = df.set_index("file")

    missing = [f for f in files_wanted if f not in df.index]
    if missing:
        print(f"[WARN] {len(missing)} filenames from pickle not found in raw CSVs (showing first 5): {missing[:5]}")

    files_present = [f for f in files_wanted if f in df.index]
    df = df.loc[files_present]

    prompt_cols = df.select_dtypes(include="number").columns.tolist()
    if not prompt_cols:
        raise RuntimeError("No numeric prompt columns found in merged dataframe.")

    return df[prompt_cols].copy(), prompt_cols


def compute_centered_means(X_df: pd.DataFrame, labels: np.ndarray) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    X_df = X_df.copy()
    X_df["cluster"] = labels[: len(X_df)]

    means = X_df.groupby("cluster").mean(numeric_only=True)
    centered = means - means.mean(axis=0)
    sep = centered.abs().max(axis=0).sort_values(ascending=False)

    centered_ordered = centered.loc[:, sep.index]
    return centered_ordered, sep, means


def plot_heatmap(centered_ordered: pd.DataFrame, sep: pd.Series, fig_out: str | None, show: bool, title: str):
    data = centered_ordered.T

    # Adjust figure size: scale width with number of clusters so columns remain narrow
    n_clusters = data.shape[1]
    n_prompts = data.shape[0]
    plt.figure(figsize=(0.8 * n_clusters + 4, 0.35 * n_prompts))

    ax = sns.heatmap(
        data,
        cmap="RdBu_r",
        center=0.0,
        cbar=True,
        annot=False,
        linewidths=0.3,
        linecolor="white",
    )
    ax.set_title(title)
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("Prompts (ordered by inter-cluster separation)")
    plt.tight_layout()

    if fig_out:
        outdir = os.path.dirname(fig_out) or "."
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(fig_out, dpi=150, bbox_inches="tight")
        print(f"Saved heatmap: {fig_out}")

    if show:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Per-cluster prompt means & differences using real prompt names from raw CLAP CSVs.")
    parser.add_argument("--input", required=True, help="Path to clustering pickle (k-means or hierarchical; raw or normalised).")
    parser.add_argument("--csv-out", default=None, help="Optional CSV to save raw means + separation.")
    parser.add_argument("--fig-out", default="evaluation/prompt_differences.png", help="Path to save the heatmap image.")
    parser.add_argument("--show", action="store_true", help="Display the figure interactively.")
    args = parser.parse_args()

    payload = load_payload(args.input)
    labels, files = get_labels_and_files(payload)

    X_df, prompt_names = rebuild_features(files)
    centered_ordered, sep, means = compute_centered_means(X_df, labels)

    if args.csv_out:
        out = means.copy()
        out.loc["__sep__", :] = sep.reindex(out.columns)
        outdir = os.path.dirname(args.csv_out) or "."
        os.makedirs(outdir, exist_ok=True)
        out.to_csv(args.csv_out)
        print(f"Saved CSV: {args.csv_out}")

    title = "Per-Cluster Prompt Means Difference"
    plot_heatmap(centered_ordered, sep, args.fig_out, args.show, title)

    print("Top prompts by inter-cluster separation:")
    for name, val in sep.head(12).items():
        print(f"  {name}: max |deviation| = {val:.3f}")


if __name__ == "__main__":
    main()