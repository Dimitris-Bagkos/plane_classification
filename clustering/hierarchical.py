from tools.data_utils import simplify_aircraft_model
import pandas as pd
import json

# Load your data
df_scores = pd.read_csv("../eda/clap_similarity_scores_full_prompts.csv")

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
