import pandas as pd
import matplotlib.pyplot as plt

# Load the similarity scores
df = pd.read_csv("clap_similarity_scores_full_prompts.csv", index_col=0)

# Set up the plot
fig, ax = plt.subplots(figsize=(12, 6))

# Bar width and positions
bar_width = 0.15
index = range(len(df))
num_prompts = len(df.columns)
offsets = [(i - (num_prompts - 1) / 2) * bar_width for i in range(num_prompts)]

# Plot each column (prompt)
for i, col in enumerate(df.columns):
    ax.bar([x + offsets[i] for x in index], df[col], width=bar_width, label=col)

# Labeling
ax.set_xlabel("Audio Files")
ax.set_ylabel("Similarity Score")
ax.set_title("CLAP Similarity Scores for Each Prompt")
ax.set_xticks(index)
ax.set_xticklabels(df.index, rotation=45, ha='right')
ax.legend(title="Prompts")

plt.tight_layout()
plt.show()
