import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score

# --- Load preprocessed data ---
with open("../clustering/k_means_clusters_normalised.pkl", "rb") as f:
    data = pickle.load(f)

X = data["X"]                          # 22D standardised CLAP scores
y = data["y"]                          # Aircraft labels (A320 / A321)
kmeans_labels = data["kmeans_labels"]  # Cluster assignments from k-means
filenames = data["filenames"]
altitude = data.get("altitude", None)      # Optional altitude data


# --- Run PCA (3 components) ---
pca = PCA(n_components=3, random_state=42)
Z_pca = pca.fit_transform(X)
explained = pca.explained_variance_ratio_

print(f"Explained variance:")
print(f"  PC1 = {explained[0]:.2%}")
print(f"  PC2 = {explained[1]:.2%}")
print(f"  PC3 = {explained[2]:.2%}")

# --- Evaluate clustering performance in 3D PCA space ---
sil_score = silhouette_score(Z_pca, kmeans_labels)
ari = adjusted_rand_score(y, kmeans_labels)
print(f"Silhouette Score (PCA 3D): {sil_score:.3f}")
print(f"Adjusted Rand Index (vs true labels): {ari:.3f}")

# --- Plot PC1 vs PC2 with PC3 as colour ---
plt.figure(figsize=(10, 6))
scatter = plt.scatter(Z_pca[:, 0], Z_pca[:, 1],
                      c=Z_pca[:, 2], cmap='turbo', s=100, alpha=0.85)

plt.xlabel(f"PC1 ({explained[0]*100:.1f}% var)")
plt.ylabel(f"PC2 ({explained[1]*100:.1f}% var)")
plt.title("PCA: PC1 vs PC2, colour = PC3")
cbar = plt.colorbar(scatter)
cbar.set_label(f"PC3 ({explained[2]*100:.1f}% variance)")

plt.grid(True)
plt.tight_layout()
plt.show()

# --- Save updated PCA output ---
pca_output = {
    "Z_pca": Z_pca,  # Now (n x 3)
    "filenames": filenames,
    "y": y,
    "kmeans_labels": kmeans_labels,
    "explained_variance": explained
}

with open("pca_3d.pkl", "wb") as f:
    pickle.dump(pca_output, f)




