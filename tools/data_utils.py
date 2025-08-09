def simplify_aircraft_model(model_name: str):
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


def load_and_merge_clap_results(json_path: str, csv_path: str):
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
    import json
    import pandas as pd


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


def split_by_heading(json_path: str, csv_path: str):
    """
    Load CLAP results merged with metadata, then split into subsets by heading.

    Parameters
    ----------
    json_path : str
        Path to JSON metadata file (must include 'heading').
    csv_path : str
        Path to CLAP results CSV.

    Returns
    -------
    dict
        Keys: 'east', 'north', 'south'.
        Values: DataFrames filtered by heading.
    """
    import json
    import pandas as pd

    # Load base merged DataFrame from existing function
    df = load_and_merge_clap_results(json_path, csv_path)

    # Reload metadata just for the heading column
    with open(json_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    heading_df = pd.DataFrame(metadata)[["filename", "heading"]]
    heading_df['filename'] = heading_df['filename'].str.strip().str.lower()

    # Merge heading into df
    df['file'] = df['file'].str.strip().str.lower()
    df = df.merge(heading_df, left_on='file', right_on='filename', how='left')
    df.drop(columns=['filename'], inplace=True)

    # Create subsets
    subsets = {}
    for direction in ['east', 'north', 'south']:
        subsets[direction] = df[df['heading'].str.lower() == direction].copy()

    return subsets


