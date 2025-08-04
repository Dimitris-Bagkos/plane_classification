import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score

# --- Load preprocessed data ---
with open("../clustering/k_means_clusters_normalised.pkl", "rb") as f:
    data = pickle.load(f)

X = data["X"]
y = data["y"]                                # Aircraft labels (e.g., A320 / A321)
kmeans_labels = data["kmeans_labels"]
filenames = data["filenames"]

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
"""
# --- Plot PC1 vs PC2 with PC3 as colour and model as style ---
plt.figure(figsize=(10, 6))

# Create scatterplot with seaborn for aircraft style and PC3 coloring
scatter = sns.scatterplot(
    x=Z_pca[:, 0],
    y=Z_pca[:, 1],
    hue=Z_pca[:, 2],
    style=y,
    palette='turbo',
    s=120,
    edgecolor='black',  # Optional: outline by cluster?
    alpha=0.9
)

plt.xlabel(f"PC1 ({explained[0]*100:.1f}% var)")
plt.ylabel(f"PC2 ({explained[1]*100:.1f}% var)")
plt.title("PCA: PC1 vs PC2, colour = PC3, style = Aircraft Type")

# Add colorbar with explicit axes
norm = plt.Normalize(Z_pca[:, 2].min(), Z_pca[:, 2].max())
sm = plt.cm.ScalarMappable(cmap='turbo', norm=norm)
sm.set_array([])

ax = plt.gca()  # get current Axes
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label(f"PC3 ({explained[2]*100:.1f}% variance)")

plt.grid(True)
plt.tight_layout()
plt.show()

# --- Save updated PCA output ---
pca_output = {
    "Z_pca": Z_pca,
    "filenames": filenames,
    "y": y,
    "kmeans_labels": kmeans_labels,
    "explained_variance": explained
}

with open("pca_3d.pkl", "wb") as f:
    pickle.dump(pca_output, f)

print("✅ Saved: pca_3d.pkl")
"""
import pandas as pd

# --- Bin PC3 into 3 categories (quantiles) ---
pc3 = Z_pca[:, 2]
pc3_labels = pd.qcut(pc3, q=3, labels=["Low PC3", "Medium PC3", "High PC3"])

# --- Create dataframe for easier seaborn plotting ---
plot_df = pd.DataFrame({
    "PC1": Z_pca[:, 0],
    "PC2": Z_pca[:, 1],
    "PC3_bin": pc3_labels,
    "Aircraft": y
})

# --- Plot PC1 vs PC2 with binned PC3 as discrete hue, aircraft as style ---
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=plot_df,
    x="PC1",
    y="PC2",
    hue="PC3_bin",
    style="Aircraft",
    palette="Dark2",
    s=120,
    edgecolor='black',
    alpha=0.9
)

plt.xlabel(f"PC1 ({explained[0]*100:.1f}% var)")
plt.ylabel(f"PC2 ({explained[1]*100:.1f}% var)")
plt.title("PCA: PC1 vs PC2, colour = PC3 level (Low/Med/High)")

plt.legend(title="PC3 Level / Aircraft", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()


