import json
import pandas as pd

with open("preprocessing/labelled_data_01_08_2025.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

df_meta = pd.DataFrame(metadata)
unique_models = df_meta['aircraft_model'].dropna().unique()

print(f"Found {len(unique_models)} unique aircraft models:\n")
for model in sorted(unique_models):
    print(model)
