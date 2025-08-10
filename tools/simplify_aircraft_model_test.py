import json
import pandas as pd
from tools.data_utils import simplify_aircraft_model

with open("preprocessing/labelled_data_01_08_2025.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

df = pd.DataFrame(metadata)

# Show original unique models
print("Full models (unique):")
for model in sorted(df['aircraft_model'].dropna().unique()):
    print(model)

# Apply simplification
df['model_group'] = df['aircraft_model'].apply(simplify_aircraft_model)

print("\nSimplified groups (unique):")
for group in sorted(df['model_group'].dropna().unique()):
    print(group)

# Show mapping from full -> simplified
print("\nMapping from full model to simplified group:")
for full, simp in sorted(set(zip(df['aircraft_model'], df['model_group']))):
    print(f"{full}  -->  {simp}")
