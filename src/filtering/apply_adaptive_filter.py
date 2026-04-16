#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apply Adaptive Filter to All Benchmark Daily CSV Files

This script applies the adaptive threshold filter to all CSV files in the
benchmark_daily/ directory and adds an 'adaptive_filter' column to each file.

Convention:
- 0 = accepted observation
- 1 = rejected observation
"""

import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm
import sys

# Get the script's directory and find project root
script_dir = os.path.dirname(os.path.abspath(__file__))
# Go up two levels: src/filter/ -> src/ -> project_root/
project_root = os.path.dirname(os.path.dirname(script_dir))

# Change to project root directory
os.chdir(project_root)
print(f"Working directory set to: {project_root}")

# Add project root and src directories to path
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))
sys.path.insert(0, os.path.join(project_root, 'src', 'filter'))

try:
    from src.filter.adaptive_filter import swot_adaptive_lakeSP
    print("Successfully imported adaptive filter function")
except ImportError as e:
    print(f"Error importing adaptive filter: {e}")
    print("Make sure you're running from the correct directory")
    sys.exit(1)

def apply_adaptive_filter_to_file(csv_path):
    """
    Apply adaptive filter to a single CSV file and add 'adaptive_filter' column.
    
    Parameters:
    -----------
    csv_path : str
        Path to the CSV file
        
    Returns:
    --------
    bool : True if successful, False otherwise
    """
    try:
        # Load the CSV file with proper dtypes
        df = pd.read_csv(csv_path, dtype={'gage_id': str, 'swot_lake_id': str})
        
        # Check if adaptive_filter column already exists
        #if 'adaptive_filter' in df.columns:
        #    print(f"  Skipping {os.path.basename(csv_path)} - adaptive_filter column already exists")
        #    return True
        
        # Initialize adaptive_filter column for all data as rejected (1)
        df['adaptive_filter'] = 1
        
        # Check if we have SWOT data to filter (use all data like filter_evaluation.py)
        swot_data = df[df['swot_wse'].notna()]
        if len(swot_data) < 5:
            # Not enough SWOT data - all remain rejected (1)
            print(f"  {os.path.basename(csv_path)}: Insufficient SWOT data ({len(swot_data)} obs) - all marked as rejected")
        else:
            try:
                # Apply the adaptive filter to ALL data (like filter_evaluation.py does)
                filter_result = swot_adaptive_lakeSP(df, verbose=True)
                
                # Apply date filtering to the results (like filter_evaluation.py does)
                df['date'] = pd.to_datetime(df['date'])
                date_mask = df['date'] >= '2023-07-21'
                
                # Combine adaptive filter results with date filter
                combined_mask = filter_result & date_mask
                
                # Only observations that pass both filters get marked as accepted (0)
                df.loc[combined_mask, 'adaptive_filter'] = 0
                
                n_accepted = (df['adaptive_filter'] == 0).sum()
                n_total = len(df)
                n_adaptive_passed = filter_result.sum()
                print(f"  {os.path.basename(csv_path)}: {n_accepted}/{n_total} observations accepted ({n_adaptive_passed} passed adaptive, {n_accepted} after date filter)")
                
            except Exception as e:
                # If adaptive filter fails, all remain rejected (1)
                print(f"  {os.path.basename(csv_path)}: Adaptive filter failed ({str(e)[:50]}) - all marked as rejected")
        
        # Save the updated CSV file
        df.to_csv(csv_path, index=False)
        return True
        
    except Exception as e:
        print(f"  Error processing {csv_path}: {e}")
        return False

def main():
    """
    Main function to apply adaptive filter to all benchmark_daily CSV files.
    """
    print("Adaptive Filter Application Script")
    print("=" * 50)
    
    # Define the benchmark_daily directory
    benchmark_dir = "data/timeseries/benchmark_daily"
    
    if not os.path.exists(benchmark_dir):
        print(f"Error: Directory {benchmark_dir} not found!")
        print("Make sure you're running this script from the project root directory")
        return
    
    # Find all CSV files in the benchmark_daily directory
    csv_files = glob.glob(os.path.join(benchmark_dir, "*_daily.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {benchmark_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV files to process")
    print()
    
    # Process each file
    successful = 0
    failed = 0
    
    for csv_file in tqdm(csv_files, desc="Processing files"):
        if apply_adaptive_filter_to_file(csv_file):
            successful += 1
        else:
            failed += 1
    
    print()
    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Total files processed: {len(csv_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if failed > 0:
        print(f"\nWarning: {failed} files failed to process")
    else:
        print("\nAll files processed successfully!")
        print("The 'adaptive_filter' column has been added to all CSV files.")
        print("Values: 0 = accepted observation, 1 = rejected observation")

if __name__ == "__main__":
    main()