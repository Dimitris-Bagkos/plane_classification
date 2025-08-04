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

# --- Run PCA ---
pca = PCA(n_components=2, random_state=42)
Z_pca = pca.fit_transform(X)

explained = pca.explained_variance_ratio_
print(f"Explained variance: PC1 = {explained[0]:.2%}, PC2 = {explained[1]:.2%}")

# --- Evaluate clustering performance in PCA space ---
sil_score = silhouette_score(Z_pca, kmeans_labels)
ari = adjusted_rand_score(y, kmeans_labels)
print(f"Silhouette Score (PCA 2D): {sil_score:.3f}")
print(f"Adjusted Rand Index (vs true labels): {ari:.3f}")

# --- Plot PCA result (color by cluster) ---
plt.figure(figsize=(10, 6))
sns.scatterplot(x=Z_pca[:, 0], y=Z_pca[:, 1],
                hue=kmeans_labels,
                style=y,
                palette='Set1',
                s=100)

plt.title("PCA (2D) of CLAP Scores - K-means Clusters")
plt.xlabel(f"PC1 ({explained[0]*100:.1f}% var)")
plt.ylabel(f"PC2 ({explained[1]*100:.1f}% var)")
plt.grid(True)
plt.legend(title="Cluster / Aircraft", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# --- Optional: Save 2D projection for future comparisons ---
pca_output = {
    "Z_pca": Z_pca,
    "filenames": filenames,
    "y": y,
    "kmeans_labels": kmeans_labels,
    "explained_variance": explained
}

with open("pca_2d.pkl", "wb") as f:
    pickle.dump(pca_output, f)
