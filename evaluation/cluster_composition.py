"""
Print the composition of each cluster by aircraft model for either
k-means or hierarchical clustering outputs.

Usage examples (run from project root):
    python -m evaluation.cluster_composition --input clustering/k_means_clusters.pkl
    python -m evaluation.cluster_composition --input clustering/hierarchical_clusters.pkl

    # also works with the *normalised* pickles:
    python -m evaluation.cluster_composition --input clustering/k_means_clusters_normalised.pkl
    python -m evaluation.cluster_composition --input clustering/hierarchical_clusters_normalised.pkl

If you pass multiple inputs, it will print a report for each in order.
"""

from __future__ import annotations
import argparse
import os
import pickle
import numpy as np
import pandas as pd


def load_pickle(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def extract_labels_and_models(payload: dict):
    """Return (cluster_labels, model_labels) from a clustering payload.
    Tries common keys from our k-means & hierarchical scripts.
    Avoids truth-testing on numpy arrays (which raises ValueError).
    """
    labels = payload.get("kmeans_labels", None)
    if labels is None:
        labels = payload.get("hier_labels", None)
    if labels is None:
        labels = payload.get("labels", None)

    y = payload.get("y", None)
    if y is None:
        y = payload.get("model_group", None)

    if labels is None:
        raise KeyError("Could not find cluster labels in pickle (expected one of 'kmeans_labels', 'hier_labels', 'labels').")
    if y is None:
        raise KeyError("Could not find model labels in pickle (expected 'y' or 'model_group').")

    # Convert to 1D numpy arrays for safety
    labels = np.asarray(labels).ravel()
    y = np.asarray(y).ravel()

    if labels.shape[0] != y.shape[0]:
        raise ValueError(f"Length mismatch: labels({labels.shape[0]}) vs y({y.shape[0]}).")
    return labels, y


def report_composition(labels: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Compute a tidy DataFrame with fractions per (cluster, model)."""
    df = pd.DataFrame({"cluster": labels, "model": y})

    # counts per (cluster, model)
    counts = df.groupby(["cluster", "model"], as_index=False).size()

    # total per cluster
    totals = counts.groupby("cluster", as_index=False)["size"].sum().rename(columns={"size": "n_cluster"})

    # percentage within each cluster
    out = counts.merge(totals, on="cluster")
    out["fraction"] = 100 * out["size"] / out["n_cluster"]

    # tidy columns + sort
    out = out.drop(columns=["size"]).sort_values(["cluster", "fraction"], ascending=[True, False]).reset_index(drop=True)
    return out



def print_report(df: pd.DataFrame, title: str | None = None):
    if title:
        print("=" * len(title))
        print(title)
        print("=" * len(title))
    for clus, chunk in df.groupby("cluster", sort=True):
        n = int(chunk["n_cluster"].iloc[0])
        print(f"Cluster {clus} (n={n}):")
        for _, row in chunk.iterrows():
            print(f"  - {row['model']}: {row['fraction']:.1f}%")
        print()


def main():
    parser = argparse.ArgumentParser(description="Print cluster composition by aircraft model.")
    parser.add_argument("--input", nargs="+", required=True, help="Path(s) to clustering pickle(s).")
    parser.add_argument("--csv-out", default=None, help="Optional path to save a CSV of the composition table.")
    args = parser.parse_args()

    all_tables = []
    for path in args.input:
        payload = load_pickle(path)
        labels, y = extract_labels_and_models(payload)
        table = report_composition(labels, y)
        print_report(table, title=os.path.basename(path))
        # tag source for optional CSV concat
        table.insert(0, "source", os.path.basename(path))
        all_tables.append(table)

    if args.csv_out:
        out = pd.concat(all_tables, ignore_index=True)
        os.makedirs(os.path.dirname(args.csv_out) or ".", exist_ok=True)
        out.to_csv(args.csv_out, index=False)
        print(f"Saved CSV: {args.csv_out}")


if __name__ == "__main__":
    main()
