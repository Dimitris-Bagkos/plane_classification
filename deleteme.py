import json
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import umap
import matplotlib.pyplot as plt

# Load data
with open("preprocessing/labelled_data_01_08_2025.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)
df_meta = pd.DataFrame(metadata)

# Simplify aircraft model
def simplify_aircraft_model(model_name: str):
    if not isinstance(model_name, str) or not model_name:
        return "Unknown"
    parts = model_name.split()
    if len(parts) >= 2:
        brand = parts[0]
        base = parts[1].split('-')[0]
        return f"{brand} {base}"
    return model_name

df_meta["model_group"] = df_meta["aircraft_model"].apply(simplify_aircraft_model)

# Load CLAP scores
df_scores = pd.read_csv("eda/clap_similarity_scores_full_prompts_2.csv")

# Normalize filenames
df_meta["filename"] = df_meta["filename"].str.strip().str.lower()
df_scores["file"] = df_scores["file"].str.strip().str.lower()

# Merge
df_merged = pd.merge(
    df_scores,
    df_meta[["filename", "model_group", "heading"]],
    left_on="file",
    right_on="filename",
    how="inner"
)

# Filter eastbound only
df_east = df_merged[df_merged["heading"].str.lower() == "east"].copy()

# Get top 4 most common parent types
top4 = df_east["model_group"].value_counts().nlargest(4).index
df_top4 = df_east[df_east["model_group"].isin(top4)].copy()

# Features
X = df_top4.select_dtypes(include="number").values

# Standardize
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=1, n_init=10)
clusters = kmeans.fit_predict(X_std)
df_top4["cluster"] = clusters

# UMAP 2D
reducer = umap.UMAP(random_state=1)
embedding = reducer.fit_transform(X_std)

# Plot
plt.figure(figsize=(8, 6))
for cluster_id in range(4):
    mask = df_top4["cluster"] == cluster_id
    plt.scatter(
        embedding[mask, 0], embedding[mask, 1],
        label=f"Cluster {cluster_id}", alpha=0.7
    )
# Label points with model_group
for i, txt in enumerate(df_top4["model_group"].values):
    plt.annotate(txt, (embedding[i, 0], embedding[i, 1]), fontsize=6, alpha=0.7)

plt.title("UMAP projection of eastbound top-4 aircraft clusters")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.legend()
plt.tight_layout()
plt.show()
