import json
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# --- Step 1: Load JSON with aircraft model info ---
json_path = Path(r"./labelled_data_20_06_2025.json")
with open(json_path, "r", encoding="utf-8") as f:
    json_data = json.load(f)

# Convert JSON to DataFrame
json_df = pd.DataFrame(json_data)

# Keep only relevant columns (filename and aircraft model)
json_df = json_df[["filename", "aircraft_model"]]

# --- Step 2: Load CLAP CSV data ---
csv_path = Path(r"../eda/clap_similarity_scores_new.csv")
clap_df = pd.read_csv(csv_path, header=None)

# Add column names for the CSV
clap_df.columns = ["filename", "dog_bark", "plane_cruising", "train_passing", "quiet_forest", "car_engine", "birds_chirping"]

# --- Step 3: Merge CSV + JSON on filename ---
df = pd.merge(clap_df, json_df, on="filename")

# --- Step 4: PCA Visualization ---
features = df[["dog_bark", "plane_cruising", "train_passing", "quiet_forest", "car_engine", "birds_chirping"]]

# --- Utility function to group similar models together
def simplify_aircraft_model(model_name):
    parts = model_name.split()
    if len(parts) >= 2:
        brand = parts[0]
        model = parts[1].split('-')[0]  # Split at dash and keep base model
        return f"{brand} {model}"
    else:
        return model_name  # Just in case something unexpected shows up


# --- Group similar models together ---
df["model_group"] = df["aircraft_model"].apply(simplify_aircraft_model)

pca = PCA(n_components=2)
pca_result = pca.fit_transform(features)
df["PC1"] = pca_result[:, 0]
df["PC2"] = pca_result[:, 1]

plt.figure(figsize=(8,6))
sns.scatterplot(
    data=df,
    x="PC1",
    y="PC2",
    hue="model_group",   # ✅ Use grouped model names for coloring
    palette="tab10",
    s=100
)
plt.title("PCA of CLAP Scores by Aircraft Model Group")
plt.grid(True)
plt.tight_layout()
plt.show()
