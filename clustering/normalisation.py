import pickle
from sklearn.preprocessing import StandardScaler

# Load raw cluster data
with open("clustering/k_means_clusters.pkl", "rb") as f:
    data = pickle.load(f)

X_raw = data["X_raw"]
y = data["y"]
filenames = data["filenames"]
kmeans_labels = data["kmeans_labels"]

# Standardise features
scaler = StandardScaler()
X_std = scaler.fit_transform(X_raw)

# Save normalised data
clusters_normalised = {
    **data,
    "X": X_std,
    "y": y,
    "filenames": filenames,
    "kmeans_labels": kmeans_labels
}

with open("clustering/k_means_clusters_normalised.pkl", "wb") as f:
    pickle.dump(clusters_normalised, f)
