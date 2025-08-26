# clustering/cluster_stats.py
import pickle, pandas as pd

data = pickle.load(open("./clustering/k_means_clusters_merged.pkl","rb"))
models = pd.Series(data["y"], name="model")
clusters = pd.Series(data["kmeans_labels"], name="cluster")
df = pd.concat([models, clusters], axis=1)

counts = df.groupby(["model","cluster"]).size().unstack(fill_value=0)
print("Counts:\n", counts, "\n")
print("Percentages (%):\n", (counts.div(counts.sum(axis=1), axis=0)*100).round(2))
