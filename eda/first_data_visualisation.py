import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# how many samples to display
n = 3  # change this as you like

# Load the similarity scores
df = pd.read_csv("clap_similarity_scores_full_prompts_2.csv", index_col=0)

# take a random subset of n rows
df = df.sample(n)

# Set up the plot
fig, ax = plt.subplots(figsize=(12, 6))

# Bar width and positions
bar_width = 0.8 / len(df.columns)   # shrink bars so each group fits in width ~0.8
index = np.arange(len(df))          # one position per audio file

# Plot each column (prompt)
for i, col in enumerate(df.columns):
    ax.bar(index + i * bar_width, df[col], width=bar_width, label=col)

# Labeling
ax.set_xlabel("Audio Files")
ax.set_ylabel("Similarity Score")
ax.set_title("CLAP Similarity Scores for Each Prompt")
ax.set_xticks(index + bar_width * (len(df.columns) - 1) / 2)
ax.set_xticklabels(df.index, rotation=45, ha='right')
ax.legend(title="Prompts", fontsize="small", ncol=2)

plt.tight_layout(rect=[0, 0.05, 1, 1])  # leave 5% space at bottom
plt.show()
