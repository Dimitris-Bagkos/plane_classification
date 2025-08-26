# dimensionality_reduction/umap_hier.py
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, rand_score
import umap

# --- Load preprocessed hierarchical data ---
with open("clustering/hierarchical_clusters_normalised.pkl", "rb") as f:
    data = pickle.load(f)

X = data["X"]                          # 22D standardised CLAP scores
y = data["y"]                          # Aircraft labels (e.g., Airbus A320, Boeing 737)
# prefer explicit hierarchical labels, but fall back to kmeans_labels if shim is present
cluster_labels = data.get("hier_labels", data.get("kmeans_labels"))
filenames = data.get("filenames")

# --- UMAP params (tweak as needed) ---
N_NEIGHBORS = 15
MIN_DIST = 0.1
METRIC = "euclidean"
RANDOM_STATE = 42

# --- Run UMAP (2D) ---
reducer = umap.UMAP(
    n_components=2,
    n_neighbors=N_NEIGHBORS,
    min_dist=MIN_DIST,
    metric=METRIC,
    random_state=RANDOM_STATE
)
Z_umap = reducer.fit_transform(X)

# --- Evaluate clustering performance in UMAP space ---
sil_score = silhouette_score(Z_umap, cluster_labels)
ri = rand_score(y, cluster_labels)
print(f"Silhouette Score (UMAP 2D): {sil_score:.3f}")
print(f"Adjusted Rand Index (vs true labels): {ri:.3f}")

# --- Plot UMAP result (color by hierarchical cluster, style by label) ---
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=Z_umap[:, 0], y=Z_umap[:, 1],
    hue=cluster_labels,
    style=y,
    palette='Set1',
    s=100
)

plt.title("UMAP of CLAP Scores - Hierarchical Clusters")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.grid(True)
plt.legend(title="Cluster / Aircraft", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# --- Optional: Save 2D projection for future comparisons ---
umap_output = {
    "Z_umap": Z_umap,
    "filenames": filenames,
    "y": y,
    "hier_labels": data.get("hier_labels"),
    "kmeans_labels": data.get("kmeans_labels"),  # include if present for parity
    "params": {
        "n_neighbors": N_NEIGHBORS,
        "min_dist": MIN_DIST,
        "metric": METRIC,
        "random_state": RANDOM_STATE
    }
}

with open("dimensionality_reduction/umap_hier_2d.pkl", "wb") as f:
    pickle.dump(umap_output, f)
