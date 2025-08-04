import pandas as pd
import matplotlib.pyplot as plt

# Load the data
filepath = r"clap_similarity_scores_full_prompts.csv"
df = pd.read_csv(filepath)

# Drop 'iteration' column if it's just an index
data = df.drop(columns=['iteration'])

# Compute cumulative mean for each column
cumulative_means = data.expanding().mean()

# Plotting
plt.figure(figsize=(10, 6))

for column in cumulative_means.columns:
    plt.plot(cumulative_means.index, cumulative_means[column], label=column)

plt.title('Cumulative Average Similarity Scores Over Iterations')
plt.xlabel('Sample Count')
plt.ylabel('Average Similarity Score')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


