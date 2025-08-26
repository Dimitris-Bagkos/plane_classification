import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tools import data_utils

# Load enriched results
df = data_utils.load_and_merge_clap_results(
    "preprocessing/labelled_data_01_08_2025.json",
    "eda/clap_similarity_scores_full_prompts_2.csv"
)

# Drop non-score columns
score_cols = [col for col in df.columns if col not in ["file", "aircraft_model", "model_group"]]

# Group by model group and average scores
grouped = df.groupby("model_group")[score_cols].mean()

# Ensure both models exist
if "Airbus A320" not in grouped.index or "Airbus A321" not in grouped.index:
    raise ValueError("One of the models (A320 or A321) is missing from the dataset.")

# Compute the difference
diff_vector = grouped.loc["Airbus A321"] - grouped.loc["Airbus A320"]

# Plot the difference as a horizontal bar chart
plt.figure(figsize=(12, 8))
sns.barplot(x=diff_vector.values, y=diff_vector.index, palette="coolwarm")
plt.title("CLAP Score Difference (A321 - A320) per Prompt")
plt.xlabel("Score Difference")
plt.ylabel("Prompt")
plt.axvline(0, color='gray', linestyle='--')
plt.tight_layout()
plt.show()
