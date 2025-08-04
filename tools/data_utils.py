import json
import pandas as pd


def simplify_aircraft_model(model_name: str) -> str:
    """
    Simplify an aircraft model name by keeping only brand and base model.

    e.g. "Airbus A321-231" → "Airbus A321"
    """
    if not isinstance(model_name, str) or not model_name:
        return "Unknown"
    parts = model_name.split()
    if len(parts) >= 2:
        brand = parts[0]
        base = parts[1].split('-')[0]
        return f"{brand} {base}"
    return model_name


def load_and_merge_clap_results(
    json_path: str,
    csv_path: str
) -> pd.DataFrame:
    """
    Load CLAP similarity scores from a CSV and merge with aircraft metadata from a JSON file.

    Parameters
    ----------
    json_path : str
        Filepath to the JSON metadata file. Must contain 'filename' and 'aircraft_model'.
    csv_path : str
        Filepath to the CLAP results CSV. Must contain 'file' plus prompt score columns.

    Returns
    -------
    pd.DataFrame
        DataFrame containing columns:
        - file: original audio filename
        - aircraft_model: full model string
        - model_group: simplified model name
        - one column per prompt score (numeric)

    """
    # Load similarity scores
    df = pd.read_csv(csv_path)

    # Load metadata JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    meta_df = pd.DataFrame(metadata)[["filename", "aircraft_model"]]

    # Normalize and merge
    df['file'] = df['file'].str.strip().str.lower()
    meta_df['filename'] = meta_df['filename'].str.strip().str.lower()
    merged = df.merge(
        meta_df,
        left_on='file',
        right_on='filename',
        how='left'
    )

    # Simplify model grouping
    merged['model_group'] = merged['aircraft_model'].apply(simplify_aircraft_model)

    # Drop helper column
    merged.drop(columns=['filename'], inplace=True)

    return merged
