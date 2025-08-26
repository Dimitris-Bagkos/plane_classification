"""
Compute silhouette scores for k-means over k = 2..11 on the MERGED dataset
coming from two (json, csv) pairs.

Inputs (edit below if paths differ):
    csv_path_1 = "eda/clap_similarity_scores_full_prompts_2.csv"
    csv_path_2 = "eda/clap_similarity_scores_full_prompts_1.csv"

    json_path_1 = "preprocessing/labelled_data_01_08_2025.json"
    json_path_2 = "preprocessing/labelled_data_20_06_2025.json"

Usage (from project root):
    python -m kmeans_silhouette
or
    python kmeans_silhouette.py
"""
from __future__ import annotations
import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ---------------------------------
# Config: input paths
# ---------------------------------
csv_path_1 = "eda/clap_similarity_scores_full_prompts_2.csv"
csv_path_2 = "eda/clap_similarity_scores_full_prompts_1.csv"

json_path_1 = "preprocessing/labelled_data_01_08_2025.json"
json_path_2 = "preprocessing/labelled_data_20_06_2025.json"

# ---------------------------------
# Import helper (make tools/ importable when run directly)
# ---------------------------------
try:
    from tools.data_utils import load_and_merge_clap_results
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(__file__))
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from tools.data_utils import load_and_merge_clap_results


def build_merged_dataframe() -> pd.DataFrame:
    """Load (json,csv) pairs and concatenate rows into a single DataFrame."""
    df1 = load_and_merge_clap_results(json_path=json_path_1, csv_path=csv_path_1)
    df1["session"] = 1
    df2 = load_and_merge_clap_results(json_path=json_path_2, csv_path=csv_path_2)
    df2["session"] = 2
    df = pd.concat([df1, df2], ignore_index=True)
    # Keep only rows that actually have numeric scores
    num_cols = df.select_dtypes(include="number").columns.tolist()
    # In case there are no numeric columns, fail loudly
    if not num_cols:
        raise RuntimeError("No numeric CLAP score columns found after merge. Check CSV columns.")
    return df


def get_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Return only the numeric CLAP score columns as feature matrix X."""
    X = df.select_dtypes(include="number").to_numpy()
    return X


def main():
    df = build_merged_dataframe()
    X = get_feature_matrix(df)

    # Optional: shuffle to avoid any weird ordering effects (not required)
    # rng = np.random.default_rng(42)
    # idx = rng.permutation(len(X))
    # X = X[idx]

    ks = list(range(2, 12))  # 2..11 inclusive
    scores: list[float] = []

    for k in ks:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        scores.append(score)
        print(f"k={k:2d} -> silhouette={score:.3f}")

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(ks, scores, marker="o")
    plt.xticks(ks)
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score vs Number of Clusters")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
