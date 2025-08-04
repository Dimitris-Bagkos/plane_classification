import pandas as pd
import matplotlib.pyplot as plt
import itertools

# Load data
df_scores = pd.read_csv("../eda/clap_similarity_scores_new.csv")
df_meta = pd.read_json("./labelled_data_20_06_2025.json")

# Merge on filename
df_scores = df_scores.rename(columns={'file': 'filename'})
df = pd.merge(df_scores, df_meta[['filename', 'aircraft_model']], on='filename', how='left')

# Simplify aircraft model function
def simplify_aircraft_model(model_name):
    parts = model_name.split()
    if len(parts) >= 2:
        brand = parts[0]
        model = parts[1].split('-')[0]
        return f"{brand} {model}"
    else:
        return model_name

df['simplified_model'] = df['aircraft_model'].apply(simplify_aircraft_model)

# Define a list of marker styles
marker_styles = ['o', 's', 'D', '^', 'v', '<', '>', 'P', 'X', '*', 'H', 'h', '+', 'x', '1', '2', '3', '4']

# Create a marker cycle and assign a unique marker per model
marker_cycle = itertools.cycle(marker_styles)
unique_models = df['simplified_model'].unique()
model_to_marker = {model: next(marker_cycle) for model in unique_models}

# Plotting
plt.figure(figsize=(10, 6))
for model, group in df.groupby('simplified_model'):
    plt.scatter(
        group['a plane cruising'],
        group['birds chirping'],
        label=model,
        marker=model_to_marker[model]
    )

plt.xlabel('Similarity to "a plane cruising"')
plt.ylabel('Similarity to "birds chirping"')
plt.title('CLAP Similarity: Plane Cruising vs Birds Chirping by Aircraft Model')
plt.legend(title='Aircraft Model', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()
