import os
import glob
import config
import pandas as pd
from pprint import pprint


def evaluate_all_metrics_folder():
    """
    Scan the metrics output folder for '*_results.csv' files and print the best row
    (highest mAP) for each file. If multiple rows tie on mAP, pick the latest row by
    the 'timestamp' column.
    """
    metrics_folder = os.path.join(config.output_folder, config.output_metrics_subfolder)
    if not os.path.isdir(metrics_folder):
        print(f"Metrics folder does not exist: {metrics_folder}")
        return

    csv_files = sorted(glob.glob(os.path.join(metrics_folder, '*_results.csv')))
    if not csv_files:
        print(f"No result CSV files found in {metrics_folder}")
        return

    print(f"Found {len(csv_files)} CSV files in {metrics_folder}\n")

    for csv_path in csv_files:
        print(f"== File: {os.path.basename(csv_path)} ==")
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"  Failed to read CSV: {e}")
            continue

        if 'mAP' not in df.columns:
            print("  No 'mAP' column found in CSV")
            continue

        # Convert mAP to numeric, coerce errors to NaN
        df['mAP'] = pd.to_numeric(df['mAP'], errors='coerce')
        if df['mAP'].isna().all():
            print("  No valid numeric mAP values found in CSV")
            continue

        max_map = df['mAP'].max()
        candidates = df[df['mAP'] == max_map].copy()

        # If timestamp exists, parse and pick the latest
        if 'timestamp' in candidates.columns:
            candidates['__ts'] = pd.to_datetime(candidates['timestamp'], errors='coerce')
            if candidates['__ts'].notna().any():
                best_row = candidates.loc[candidates['__ts'].idxmax()]
            else:
                # fallback to last candidate row
                best_row = candidates.iloc[-1]
            # drop helper
            if '__ts' in candidates.columns:
                candidates.drop(columns='__ts', inplace=True, errors='ignore')
        else:
            best_row = candidates.iloc[-1]

        # Print best row as dict
        pprint(best_row.dropna().to_dict())
        print()


if __name__ == '__main__':
    evaluate_all_metrics_folder()
