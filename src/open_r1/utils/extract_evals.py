import json
import pandas as pd
import argparse
from pathlib import Path

def append_results_to_csv(results_file: str, csv_file: str, step: int | None = None):
    """
    Extract evaluation results from a JSON file and append them to a CSV file.
    
    Args:
        results_file: Path to the JSON results file
        csv_file: Path to the CSV file to append results to
        step: Training step number (optional)
    """
    # Read the JSON file
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    # Extract the model name and results
    model_name = data['config_general']['model_name']
    if step is not None:
        model_name = f"{model_name}_step{step}"
    results = data['results']
    
    # Create a row dictionary with default None values
    row = {'model_name': model_name}
    
    # Add AIME24 scores if present
    if 'custom|aime24|0' in results:
        row.update({
            'aime24_score': results['custom|aime24|0']['extractive_match'],
            'aime24_stderr': results['custom|aime24|0']['extractive_match_stderr'],
        })
    
    # Add Math500 scores if present
    if 'custom|math_500|0' in results:
        row.update({
            'math500_score': results['custom|math_500|0']['extractive_match'],
            'math500_stderr': results['custom|math_500|0']['extractive_match_stderr'],
        })
    
    # Add overall scores if present
    if 'all' in results:
        row.update({
            'overall_score': results['all']['extractive_match'],
            'overall_stderr': results['all']['extractive_match_stderr']
        })
    
    # Convert to DataFrame
    df_new = pd.DataFrame([row])
    
    # If CSV exists, append to it; otherwise create new
    csv_path = Path(csv_file)
    if csv_path.exists():
        df_new.to_csv(csv_file, mode='a', header=False, index=False)
    else:
        df_new.to_csv(csv_file, mode='w', header=True, index=False)

def main():
    parser = argparse.ArgumentParser(description='Append evaluation results to CSV')
    parser.add_argument('--results_file', required=True, help='Path to the JSON results file')
    parser.add_argument('--csv_file', required=True, help='Path to the CSV file to append results to')
    parser.add_argument('--step', type=int, help='Training step number (optional)')
    
    args = parser.parse_args()
    
    append_results_to_csv(args.results_file, args.csv_file, args.step)

if __name__ == "__main__":
    main()