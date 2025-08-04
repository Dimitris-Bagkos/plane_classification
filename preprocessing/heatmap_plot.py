import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tools import data_utils

# Load enriched results
df = data_utils.load_and_merge_clap_results(
    "labelled_data_20_06_2025.json",
    "../eda/clap_similarity_scores_full_prompts.csv"
)

# Drop non-score columns
score_cols = [col for col in df.columns if col not in ["file", "aircraft_model", "model_group"]]

# Group by model group and average scores
heatmap_data = df.groupby("model_group")[score_cols].mean()

# Plot
plt.figure(figsize=(16, 12))
ax = sns.heatmap(heatmap_data.T, cmap="viridis", annot=False, cbar=True)

ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top')

plt.title("Average CLAP Similarity Scores by Aircraft Model Group")
plt.xlabel("Prompts")
plt.ylabel("Aircraft Model Group")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
