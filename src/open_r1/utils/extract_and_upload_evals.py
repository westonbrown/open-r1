import json
import pandas as pd
import argparse
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download, upload_file

def append_csv_and_upload_to_hf(
    local_csv: str,
    repo_id: str,
    csv_filename: str,
):
    """
    Append a local CSV file to an existing CSV in HuggingFace Hub.
    
    Args:
        local_csv: Path to the local CSV file to append
        repo_id: HuggingFace repository ID (e.g., 'username/repo-name')
        csv_filename: Name of the CSV file in the HF repo
    """
    # Read the new CSV file
    df_new = pd.read_csv(local_csv)
    
    try:
        # Try to download existing CSV from HuggingFace
        csv_path = hf_hub_download(
            repo_id=repo_id,
            filename=csv_filename,
        )
        # Read existing CSV and append new data
        df_existing = pd.read_csv(csv_path)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    except Exception:
        # If file doesn't exist or other error, use only new data
        df_combined = df_new
    
    # Save combined DataFrame to temporary file
    temp_csv = "temp_results.csv"
    df_combined.to_csv(temp_csv, index=False)
    
    # Upload to HuggingFace
    api = HfApi()
    api.upload_file(
        path_or_fileobj=temp_csv,
        path_in_repo=csv_filename,
        repo_id=repo_id,
    )
    
    # Clean up temporary file
    Path(temp_csv).unlink()

def main():
    parser = argparse.ArgumentParser(description='Upload and append CSV file to HuggingFace')
    parser.add_argument('--local_csv', required=True, help='Path to the local CSV file')
    parser.add_argument('--repo_id', required=True, help='HuggingFace repository ID')
    parser.add_argument('--csv_filename', required=True, help='Name of the CSV file in the HF repo')
    
    args = parser.parse_args()
    
    append_csv_and_upload_to_hf(
        args.local_csv,
        args.repo_id,
        args.csv_filename,
    )

if __name__ == "__main__":
    main()