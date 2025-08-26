# clustering/cluster_merger.py
import argparse
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd


def remap_labels(arr):
    """Map 0→0, 1→1, 2→1, 3→0 for any integer-like array."""
    arr = np.asarray(arr)
    mapping = {0: 0, 1: 1, 2: 1, 3: 0}
    out = np.array([mapping.get(int(x), int(x)) for x in arr])
    return out


def merge_any(obj, logs):
    """Try to find and remap cluster labels in a variety of common containers."""
    changed = False

    # Case 1: plain numpy/list/Series of labels
    if isinstance(obj, (np.ndarray, list, pd.Series)):
        logs.append("Detected labels as array/series; remapped directly.")
        return remap_labels(obj), True

    # Case 2: pandas DataFrame with a likely labels column
    if isinstance(obj, pd.DataFrame):
        df = obj.copy()
        candidate_cols = [c for c in df.columns if c.lower() in (
            "cluster", "clusters", "label", "labels",
            "kmeans_label", "kmeans_labels", "cluster_id"
        )]
        used_col = None
        if candidate_cols:
            used_col = candidate_cols[0]
        else:
            # fallback: exactly one integer-like column
            intlike_cols = [
                c for c in df.columns
                if pd.api.types.is_integer_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c])
            ]
            if len(intlike_cols) == 1:
                used_col = intlike_cols[0]
        if used_col is not None:
            df[used_col] = remap_labels(df[used_col].values)
            logs.append(f"Detected DataFrame; remapped column '{used_col}'.")
            return df, True
        else:
            logs.append("DataFrame with no obvious labels column; left unchanged.")
            return df, False

    # Case 3: dict holding labels / DataFrames / arrays
    if isinstance(obj, dict):
        new_obj = dict(obj)
        local_changed = False

        for key in ("labels", "labels_", "cluster", "clusters", "kmeans_labels"):
            if key in new_obj:
                try:
                    new_obj[key] = remap_labels(new_obj[key])
                    logs.append(f"Dict: remapped key '{key}'.")
                    local_changed = True
                except Exception as e:
                    logs.append(f"Dict: failed remapping key '{key}': {e!r}")

        # Also check any DataFrames inside
        for k, v in list(new_obj.items()):
            if isinstance(v, pd.DataFrame):
                df = v.copy()
                candidate_cols = [c for c in df.columns if c.lower() in (
                    "cluster", "clusters", "label", "labels",
                    "kmeans_label", "kmeans_labels", "cluster_id"
                )]
                if candidate_cols:
                    col = candidate_cols[0]
                    df[col] = remap_labels(df[col])
                    new_obj[k] = df
                    logs.append(f"Dict contained DataFrame under '{k}'; remapped column '{col}'.")
                    local_changed = True

        return new_obj, local_changed

    # Case 4: sklearn-like estimator with labels_ / labels attributes
    new_obj = obj
    if hasattr(obj, "labels_"):
        try:
            setattr(obj, "labels_", remap_labels(getattr(obj, "labels_")))
            logs.append("Generic object: remapped 'labels_'.")
            changed = True
        except Exception as e:
            logs.append(f"Failed to remap 'labels_': {e!r}")
    if hasattr(obj, "labels"):
        try:
            setattr(obj, "labels", remap_labels(getattr(obj, "labels")))
            logs.append("Generic object: remapped 'labels'.")
            changed = True
        except Exception as e:
            logs.append(f"Failed to remap 'labels': {e!r}")

    return new_obj, changed


def main():
    parser = argparse.ArgumentParser(
        description="Merge 4 k-means clusters into 2 (3→0 and 2→1)."
    )
    parser.add_argument("--in", dest="in_path", required=True,
                        help="Path to the input pickle (e.g. C:\\path\\k_means_clusters.pkl)")
    parser.add_argument("--out", dest="out_path", required=False,
                        help="Path for the output pickle (defaults to <input>_merged.pkl)")
    args = parser.parse_args()

    in_path = Path(args.in_path).expanduser().resolve() if not Path(args.in_path).is_absolute() \
        else Path(args.in_path)
    if not in_path.exists():
        sys.exit(f"ERROR: File not found: {in_path}")

    out_path = Path(args.out_path) if args.out_path else in_path.with_name(in_path.stem + "_merged.pkl")

    # Load
    with open(in_path, "rb") as f:
        obj = pickle.load(f)

    logs = [f"Loaded: {in_path} (type: {type(obj)})"]
    new_obj, changed = merge_any(obj, logs)

    # Save
    with open(out_path, "wb") as f:
        pickle.dump(new_obj, f)

    print("\n".join(logs))
    if changed:
        print(f"Success: clusters remapped (3→0, 2→1).")
    else:
        print("Note: no obvious labels found to remap; output saved unchanged.")
    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
