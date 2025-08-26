import pickle
from sklearn.preprocessing import StandardScaler

# --- Paths ---
INPUT_PKL = "clustering/hierarchical_clusters.pkl"
OUTPUT_PKL = "clustering/hierarchical_clusters_normalised.pkl"

# --- Load raw cluster data ---
with open(INPUT_PKL, "rb") as f:
    data = pickle.load(f)

# Expect the same structure produced by hierarchical_clustering.py
X_raw = data["X_raw"]
y = data.get("y")
filenames = data.get("filenames")
hier_labels = data.get("hier_labels")

# --- Standardise features ---
scaler = StandardScaler()
X_std = scaler.fit_transform(X_raw)

# --- Save normalised data ---
# Keep original payload, but add/override with standardised matrix under key "X"
clusters_normalised = {
    **data,
    "X": X_std,
    "y": y,
    "filenames": filenames,
    "hier_labels": hier_labels,
}

# Optional compatibility shim: expose labels under kmeans-style key if not present
if "kmeans_labels" not in clusters_normalised and hier_labels is not None:
    clusters_normalised["kmeans_labels"] = hier_labels

with open(OUTPUT_PKL, "wb") as f:
    pickle.dump(clusters_normalised, f)

print(f"Saved normalised hierarchical clusters to {OUTPUT_PKL}")
