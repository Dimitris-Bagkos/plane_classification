"""
Dendrogram exporter for hierarchical clustering results.

This version shows aircraft model groups as leaf labels instead of filenames.

Usage:
    python -m clustering.dendrogram --input clustering/hierarchical_clusters.pkl --show
"""
import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram


def load_payload(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def compute_linkage(X: np.ndarray, method: str, metric: str):
    method = (method or "ward").lower()
    if method == "ward":
        Z = linkage(X, method="ward", optimal_ordering=True)
    else:
        Z = linkage(X, method=method, metric=(metric or "euclidean"), optimal_ordering=True)
    return Z


def infer_color_threshold(Z: np.ndarray, k: int | None):
    if not k or k < 2:
        return None
    idx = -(k - 1)
    if -idx <= Z.shape[0]:
        return float(Z[idx, 2]) - 1e-12
    return None


def main():
    parser = argparse.ArgumentParser(description="Render a dendrogram from hierarchical clustering pickle output.")
    parser.add_argument("--input", default="clustering/hierarchical_clusters.pkl", help="Path to pickle from hierarchical_clustering.py")
    parser.add_argument("--output", default="clustering/dendrogram.png", help="Where to save the figure (png/pdf/svg). If omitted with --show, nothing is saved.")
    parser.add_argument("--show", action="store_true", help="Display the figure interactively.")
    parser.add_argument("--orientation", choices=["top", "bottom", "left", "right"], default="top", help="Dendrogram orientation.")
    parser.add_argument("--leaf-font-size", type=float, default=8.0, help="Leaf label font size.")
    parser.add_argument("--width", type=float, default=18.0, help="Figure width in inches.")
    parser.add_argument("--height", type=float, default=10.0, help="Figure height in inches.")
    parser.add_argument("--dpi", type=int, default=150, help="Figure DPI when saving.")
    parser.add_argument("--sample", type=int, default=None, help="Randomly sample this many observations before plotting.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used with --sample.")
    parser.add_argument("--title", default="Hierarchical Clustering Dendrogram (k=4)", help="Plot title.")
    args = parser.parse_args()

    payload = load_payload(args.input)
    X = np.asarray(payload.get("X_raw"))
    model_groups = np.asarray(payload.get("y"))  # labels for each observation
    method = str(payload.get("linkage", "ward"))
    metric = str(payload.get("metric", "euclidean"))
    k = payload.get("k")

    if X is None or model_groups is None:
        raise RuntimeError("Input pickle is missing 'X_raw' or 'y'. Re-run hierarchical_clustering.py.")

    if args.sample is not None and 0 < args.sample < len(X):
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(X), size=args.sample, replace=False)
        X = X[idx]
        model_groups = model_groups[idx]

    Z = compute_linkage(X, method=method, metric=metric)
    color_threshold = infer_color_threshold(Z, k)

    labels = [str(mg) for mg in model_groups]

    fig, ax = plt.subplots(figsize=(args.width, args.height))

    dendro_kwargs = dict(
        labels=labels,
        orientation=args.orientation,
        color_threshold=color_threshold,
        above_threshold_color="grey",
        leaf_rotation=90 if args.orientation in ("top", "bottom") else 0,
        leaf_font_size=args.leaf_font_size,
    )

    dendrogram(Z, **dendro_kwargs)

    ax.set_ylabel("Distance")
    ax.set_title(args.title)
    plt.tight_layout()

    if args.output:
        outdir = os.path.dirname(args.output) or "."
        os.makedirs(outdir, exist_ok=True)
        fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved dendrogram to {args.output}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
