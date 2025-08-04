from tools.data_utils import simplify_aircraft_model
import pandas as pd
import matplotlib.pyplot as plt
import json
import pickle
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# Load your data
df_scores = pd.read_csv("../eda/clap_similarity_scores_full_prompts_1.csv")

# Load JSON metadata
with open("../preprocessing/labelled_data_20_06_2025.json") as f:
    metadata = json.load(f)

df_meta = pd.DataFrame(metadata)

# Apply simplify function to aircraft models
df_meta['simple_model'] = df_meta['aircraft_model'].apply(simplify_aircraft_model)

# Merge datasets on filenames
df_merged = pd.merge(df_scores,
                     df_meta[['filename', 'simple_model']],
                     left_on='file',
                     right_on='filename',
                     how='inner')

# Check model frequencies
top_models = df_merged['simple_model'].value_counts().nlargest(2)
print("Top two aircraft models:\n", top_models)

# Keep only recordings from the two most frequent models
df_top2 = df_merged[df_merged['simple_model'].isin(top_models.index)]

# Select numeric columns only (excluding filenames and non-numeric)
X_top2 = df_top2.select_dtypes(include=['float64', 'int']).values

# Run k-means clustering (2 clusters since you have 2 models)
kmeans = KMeans(n_clusters=2, random_state=1, n_init=10)
clusters_top2 = kmeans.fit_predict(X_top2)

# Silhouette score evaluates cluster clarity
sil_score_top2 = silhouette_score(X_top2, clusters_top2)
print(f"Silhouette Score (Top 2 Models): {sil_score_top2:.3f}")

# Append cluster labels to your dataframe
df_top2 = df_top2.copy()  # avoid SettingWithCopyWarning
df_top2['cluster'] = clusters_top2

print(df_top2[['filename', 'simple_model', 'cluster']].sort_values('cluster'))

# Prepare output
raw_cluster_data = {
    "X_raw": X_top2,
    "y": df_top2['simple_model'].values,
    "filenames": df_top2['file'].values,
    "kmeans_labels": clusters_top2
}

# Save to file
with open("k_means_clusters.pkl", "wb") as f:
    pickle.dump(raw_cluster_data, f)




# 0: 5x320, 4x321
# 1:
"""
# Select only numeric columns (assuming the first column might be filenames)
X = df_scores.select_dtypes(include=['float64', 'int']).values

# Select number of clusters, e.g., 3
kmeans = KMeans(n_clusters=3, random_state=2, n_init=10)

# Fit the model and predict clusters
clusters = kmeans.fit_predict(X)

# Print cluster assignments
print(clusters)

# Compute silhouette score
score = silhouette_score(X, clusters)

print(f'Silhouette Score: {score:.3f}')

silhouette_scores = []

# Try k values from 2 to 10
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
    score = silhouette_score(X, clusters)
    silhouette_scores.append(score)

# Plot results
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Optimal number of clusters')
plt.grid()
plt.show()
"""











