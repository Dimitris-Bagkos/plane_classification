import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, rand_score

# --- Load preprocessed data ---
with open("clustering/k_means_clusters_normalised.pkl", "rb") as f:
    data = pickle.load(f)

X = data["X"]
y = data["y"]
kmeans_labels = data["kmeans_labels"]
filenames = data["filenames"]

# --- Run t-SNE (2D) ---
tsne = TSNE(n_components=2, perplexity=5, init='pca', learning_rate='auto', random_state=42)
Z_tsne = tsne.fit_transform(X)

# --- Evaluate clustering in t-SNE space ---
sil_score = silhouette_score(Z_tsne, kmeans_labels)
ri = rand_score(y, kmeans_labels)
print(f"Silhouette Score (t-SNE 2D): {sil_score:.3f}")
print(f"Adjusted Rand Index (vs true labels): {ri:.3f}")

# --- Plot t-SNE result ---
plt.figure(figsize=(10, 6))
sns.scatterplot(x=Z_tsne[:, 0], y=Z_tsne[:, 1],
                hue=kmeans_labels,
                style=y,
                palette='Set1',
                s=100)

plt.title("t-SNE of CLAP Scores - K-means Clusters")
plt.xlabel("t-SNE Dim 1")
plt.ylabel("t-SNE Dim 2")
plt.grid(True)
plt.legend(title="Cluster / Aircraft", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# --- Save result ---
tsne_output = {
    "Z_tsne": Z_tsne,
    "filenames": filenames,
    "y": y,
    "kmeans_labels": kmeans_labels
}

with open("dimensionality_reductuion/tsne_2d.pkl", "wb") as f:
    pickle.dump(tsne_output, f)

print("✅ Saved: tsne_2d.pkl")
