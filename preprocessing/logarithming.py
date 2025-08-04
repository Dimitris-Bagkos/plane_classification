import pandas as pd
import numpy as np

# Load the file
df = pd.read_csv("../eda/clap_similarity_scores_full_prompts.csv")

# Separate non-numeric columns (e.g. filename)
non_numeric_cols = df.select_dtypes(exclude=['float64', 'int']).columns
numeric_cols = df.select_dtypes(include=['float64', 'int']).columns

# Check for values ≤ 0 and shift if needed
if (df[numeric_cols] <= 0).any().any():
    min_val = df[numeric_cols].min().min()
    shift = abs(min_val) + 1e-5
    print(f"Warning: Non-positive values found. Shifting data by +{shift:.5f}")
    df[numeric_cols] = df[numeric_cols] + shift

# Apply natural log to numeric columns only
df[numeric_cols] = np.log(df[numeric_cols])

# Export
df.to_csv("clap_similarity_scores_log.csv", index=False)
print("Log-transformed data saved to clap_similarity_scores_log.csv")

