from tools.data_utils import load_and_merge_clap_results
from typing import List, Union, Tuple
import os
import pandas as pd
import pickle
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# -------------------------------
# CONFIG
# -------------------------------
csv_path_1 = "eda/clap_similarity_scores_full_prompts_2.csv"
csv_path_2 = "eda/clap_similarity_scores_full_prompts_1.csv"

json_path_1 = "preprocessing/labelled_data_01_08_2025.json"
json_path_2 = "preprocessing/labelled_data_20_06_2025.json"

MODE = "top_n"          # "top_n" or "all"
TOP_N = 4               # used only if MODE == "top_n"
N_CLUSTERS = None       # if None, uses number of distinct labels kept (TOP_N for top_n mode)

# Agglomerative settings
LINKAGE = "ward"        # one of: "ward", "complete", "average", "single"
METRIC = "euclidean"    # for linkage=="ward" this must be "euclidean"

OUTPUT_PKL = "clustering/hierarchical_clusters.pkl"
# -------------------------------


def build_dataset(
    csv_path: Union[str, List[str]],
    json_path: Union[str, List[str]],
    mode: str,
    top_n: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, List[str]]:
    """
    Returns:
      df_filtered (scores+meta),
      X (features, np.ndarray),
      y (labels, np.ndarray),
      models_included (sorted unique labels)
    Now supports multiple datasets by passing lists of paths (paired by index).
    """
    # Coerce to lists
    def _as_list(x):
        return x if isinstance(x, list) else [x]

    csv_paths = _as_list(csv_path)
    json_paths = _as_list(json_path)

    if len(csv_paths) != len(json_paths):
        raise ValueError("csv_path and json_path must have the same length when lists are provided.")

    # Merge each (json, csv) pair, tag with a session index, then concat
    merged_list = []
    for i, (jp, cp) in enumerate(zip(json_paths, csv_paths), start=1):
        df_i = load_and_merge_clap_results(json_path=jp, csv_path=cp).copy()
        df_i["session"] = i  # handy if you want to stratify/debug later
        merged_list.append(df_i)

    df = pd.concat(merged_list, ignore_index=True)

    # Filter to rows with a non-empty model_group
    df = df[df["model_group"].notna() & (df["model_group"].astype(str).str.strip() != "")]
    df["model_group"] = df["model_group"].astype(str)

    # Decide which models to include
    if mode == "top_n":
        keep = df["model_group"].value_counts().nlargest(top_n).index
        df_filtered = df[df["model_group"].isin(keep)].copy()
    elif mode == "all":
        df_filtered = df.copy()
    else:
        raise ValueError("MODE must be 'top_n' or 'all'")

    # Numeric features = CLAP scores
    X = df_filtered.select_dtypes(include="number").values
    y = df_filtered["model_group"].values
    models_included = sorted(df_filtered["model_group"].unique())

    return df_filtered, X, y, models_included


def main():
    # Build dataset
    df_filtered, X, y, models_included = build_dataset(
        csv_path=[csv_path_1, csv_path_2],
        json_path=[json_path_1, json_path_2],
        mode=MODE,
        top_n=TOP_N,
    )

    # Decide number of clusters
    n_labels = len(models_included)
    n_clusters = (TOP_N if MODE == "top_n" else n_labels) if N_CLUSTERS is None else N_CLUSTERS

    print("=== Hierarchical Clustering Config ===")
    print(f"MODE: {MODE}")
    if MODE == "top_n":
        print(f"TOP_N: {TOP_N}")
    print(f"n_clusters: {n_clusters}")
    print(f"linkage: {LINKAGE}, metric: {METRIC}")
    print(f"Records kept: {len(df_filtered)}")
    print("Models included & counts:")
    print(df_filtered["model_group"].value_counts().sort_index())
    print()

    # Run Agglomerative Clustering
    # Note: for linkage='ward', metric must be 'euclidean'.
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward", metric="euclidean")
    labels = model.fit_predict(X)

    # Silhouette (only valid if 2 <= k <= n_samples-1)
    sil = None
    if n_clusters >= 2 and len(df_filtered) > n_clusters:
        sil = silhouette_score(X, labels)
        print(f"Silhouette Score: {sil:.3f}")
    else:
        print("Silhouette Score: skipped (need at least 2 clusters and enough samples)")

    # Attach cluster labels for quick inspection
    out_df = df_filtered.copy()
    out_df["hier_cluster"] = labels
    print(out_df[["file", "model_group", "hier_cluster"]].sort_values("hier_cluster"))

    # Prepare output payload
    payload = {
        "X_raw": X,
        "y": y,
        "filenames": out_df["file"].values,
        "hier_labels": labels,
        "mode": MODE,
        "top_n": TOP_N if MODE == "top_n" else None,
        "models_included": models_included,
        "k": n_clusters,
        "linkage": LINKAGE,
        "metric": METRIC,
        "silhouette": sil,
    }

    # Ensure output folder exists and save
    os.makedirs(os.path.dirname(OUTPUT_PKL), exist_ok=True)
    with open(OUTPUT_PKL, "wb") as f:
        pickle.dump(payload, f)

    print(f"\nSaved: {OUTPUT_PKL}")


if __name__ == "__main__":
    main()
