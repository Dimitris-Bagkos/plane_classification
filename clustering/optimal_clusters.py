from tools.data_utils import simplify_aircraft_model
import pandas as pd
import matplotlib.pyplot as plt
import json
import pickle
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load your data
df_scores = pd.read_csv("../preprocessing/clap_similarity_scores_log.csv")

# Load JSON metadata
with open("../preprocessing/labelled_data_20_06_2025.json") as f:
    metadata = json.load(f)


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