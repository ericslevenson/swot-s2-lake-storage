#!/usr/bin/env python3
"""
Normalized Storage Uncertainty Attribution Analysis

Implements first-order error propagation to attribute normalized storage uncertainty to:
1. WSE measurement errors 
2. Elevation-area relationship errors (from WSA uncertainty)
3. Comparison with observed storage errors

All errors are normalized by storage variability (1st to 99th percentile range) per lake.
Uses analytical derivatives of the trapezoidal integration storage calculation.
Processes ALL model variants in a single run.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def acre_feet_to_km3(acre_feet_value):
    """Convert acre-feet to cubic kilometers"""
    if pd.isna(acre_feet_value):
        return np.nan
    # 1 acre-foot = 1233.48 m³, 1 km³ = 1e9 m³
    return (acre_feet_value * 1233.48) / 1e9

def calculate_storage_uncertainty_components_two_way_normalized(df, wse_std=0.1, wse_temporal_std=0.0, wse_filt_std=0.0, wsa_percent_error=15.0, filter_type='opt'):
    """
    Calculate normalized uncertainty components for storage estimates.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Lake data with WSE and WSA columns
    wse_std : float
        Base WSE measurement uncertainty (meters)
    wse_temporal_std : float
        Temporal interpolation uncertainty (meters)
    wse_filt_std : float
        Filtering uncertainty (meters)
    wsa_percent_error : float  
        WSA measurement uncertainty (percent)
    filter_type : str
        'opt' or 'filt' to determine which reference column to use
        
    Returns:
    --------
    pandas.DataFrame with normalized uncertainty breakdown (percentages)
    """
    
    results = []
    
    # Calculate combined WSE uncertainty (linear addition)
    wse_total_std = wse_std + wse_temporal_std + wse_filt_std
    
    # Determine the reference column based on filter type
    if filter_type == 'opt' and 'storage_anomaly_opt' in df.columns:
        gauge_storage = df['storage_anomaly_opt'].apply(acre_feet_to_km3)
    elif filter_type == 'filt' and 'storage_anomaly_filt' in df.columns:
        gauge_storage = df['storage_anomaly_filt'].apply(acre_feet_to_km3)
    elif 'storage_anomaly' in df.columns:
        # Fallback to regular storage_anomaly if specific column not available
        gauge_storage = df['storage_anomaly'].apply(acre_feet_to_km3)
    else:
        # If no storage anomaly columns, compute from storage column if available
        if 'storage' in df.columns:
            storage_km3 = df['storage'].apply(acre_feet_to_km3)
            # Compute anomaly as deviation from mean
            gauge_storage = storage_km3 - storage_km3.mean()
        else:
            # No storage data available for normalization
            return pd.DataFrame(results)
    
    # Calculate storage range (1st to 99th percentile) for this lake
    valid_storage = gauge_storage.dropna()
    if len(valid_storage) < 10:  # Need reasonable sample size
        return pd.DataFrame(results)
        
    storage_range = np.percentile(valid_storage, 99) - np.percentile(valid_storage, 1)
    
    if storage_range <= 1e-6:  # Very small threshold in km³
        return pd.DataFrame(results)
    
    # Process each observation
    for idx, row in df.iterrows():
        
        # Skip if missing essential data
        if pd.isna(row['swot_wse_anomaly']) or pd.isna(row['wsa']):
            continue
            
        # Basic parameters
        wse = row['stage_anomaly_swotdates']  # WSE above some reference (meters)
        current_area = row['wsa']  # Current area in km²
        height_above_ref = abs(wse)  # meters above reference
        
        # Convert WSA percentage error to absolute area uncertainty  
        sigma_area_km2 = current_area * (wsa_percent_error / 100.0)  # km²
        
        # Component 1: Total WSE uncertainty
        partial_S_partial_WSE = current_area  # km²
        sigma_storage_wse_total = abs((partial_S_partial_WSE * wse_total_std) / 1000.0)  # km³
        
        # Component 2: WSA measurement uncertainty  
        partial_S_partial_A = height_above_ref / 1000.0  # Convert m to km for units
        sigma_storage_area = abs(partial_S_partial_A * sigma_area_km2)  # km³
        
        # Total predicted uncertainty in km³
        sigma_storage_total = np.sqrt(sigma_storage_wse_total**2 + sigma_storage_area**2)
        
        # Normalize to percentage of storage variability
        sigma_storage_wse_total_norm = (sigma_storage_wse_total / storage_range) * 100.0
        sigma_storage_area_norm = (sigma_storage_area / storage_range) * 100.0
        sigma_storage_total_norm = (sigma_storage_total / storage_range) * 100.0
        
        # Attribution percentages (relative contributions)
        if sigma_storage_total > 0:
            linear_total = sigma_storage_wse_total + sigma_storage_area
            
            wse_contribution_pct = (sigma_storage_wse_total / linear_total) * 100
            area_contribution_pct = (sigma_storage_area / linear_total) * 100
            
            # Break down WSE contribution
            if wse_total_std > 0:
                wse_base_frac = wse_std / wse_total_std
                wse_temporal_frac = wse_temporal_std / wse_total_std
                wse_filt_frac = wse_filt_std / wse_total_std
            else:
                wse_base_frac = wse_temporal_frac = wse_filt_frac = 0
            
            wse_base_contribution_pct = wse_contribution_pct * wse_base_frac
            wse_temporal_contribution_pct = wse_contribution_pct * wse_temporal_frac
            wse_filt_contribution_pct = wse_contribution_pct * wse_filt_frac
        else:
            wse_contribution_pct = area_contribution_pct = 0
            wse_base_contribution_pct = wse_temporal_contribution_pct = wse_filt_contribution_pct = 0
        
        # Calculate equal weighting (inverse height weighting)
        height_weight = 1.0 / max(abs(height_above_ref), 0.1)  # Avoid division by zero with minimum 0.1m
        
        # Store normalized results (all in percentages)
        results.append({
            'date': row['date'] if 'date' in row else None,
            'swot_wse_anomaly': wse,
            'wsa': current_area,
            'area_km2': current_area,
            'storage_range_km3': storage_range,
            'height_above_ref': height_above_ref,
            'height_weight': height_weight,
            
            # Normalized uncertainties (% of storage variability)
            'sigma_storage_wse_total_norm': round(sigma_storage_wse_total_norm, 1),
            'sigma_storage_area_norm': round(sigma_storage_area_norm, 1),
            'sigma_storage_total_norm': round(sigma_storage_total_norm, 1),
            
            # Relative contributions (sum to 100%)
            'wse_contribution_pct': wse_contribution_pct,
            'area_contribution_pct': area_contribution_pct,
            'wse_base_contribution_pct': wse_base_contribution_pct,
            'wse_temporal_contribution_pct': wse_temporal_contribution_pct,
            'wse_filt_contribution_pct': wse_filt_contribution_pct,
            
            # Original values in km³ for reference
            'sigma_storage_wse_total_km3': sigma_storage_wse_total,
            'sigma_storage_area_km3': sigma_storage_area,
            'sigma_storage_total_km3': sigma_storage_total
        })
    
    return pd.DataFrame(results)

def load_benchmark_normalized_errors(model, swot_filter, temporal):
    """Load normalized benchmark error statistics for a specific model configuration"""
    
    # Load the normalized benchmark summary statistics
    benchmark_file = Path("/Users/ericlevenson/University of Oregon Dropbox/Eric Levenson/SWOT/production/experiments/storage_normalized/results/benchmark_storage_variants_summary_stats_normalized.csv")
    
    if not benchmark_file.exists():
        print(f"Warning: Benchmark file not found at {benchmark_file}")
        return None
        
    benchmark_df = pd.read_csv(benchmark_file)
    
    # Map model configuration to variant name
    model_map = {
        'swot': 'swot',
        'swots2': 'swots2',
        's2': 's2',
        'static': 'static'
    }
    
    filter_map = {
        'optimal': 'opt',
        'filtered': 'filt'
    }
    
    temporal_map = {
        'discrete': 'dis',
        'continuous': 'con',
        'daily': 'con'  # Map 'daily' to 'con' for benchmark file matching
    }
    
    # Construct variant name
    model_key = model_map.get(model, model)
    filter_key = filter_map.get(swot_filter, swot_filter)
    temporal_key = temporal_map.get(temporal, temporal)
    
    variant_name = f"{model_key}_{filter_key}_{temporal_key}"
    
    # Find the matching row
    row = benchmark_df[benchmark_df['variant_name'] == variant_name]
    
    if row.empty:
        return None
        
    row = row.iloc[0]
    
    return {
        'observed_mae': row['mae_median'],
        'observed_rmse': row['rmse_median'], 
        'observed_std': row['std_error_median'],
        'observed_nse_median': row['nse_median'],
        'observed_nse_mean': row['nse_mean'],
        'n_lakes': row['n_lakes']
    }

def calculate_weighted_statistics(data_series, weights_series, use_weights=True):
    """Calculate mean and median with optional weighting"""
    if len(data_series) == 0:
        return np.nan, np.nan
    
    # Remove NaN values
    valid_mask = ~(pd.isna(data_series) | pd.isna(weights_series))
    if valid_mask.sum() == 0:
        return np.nan, np.nan
    
    valid_data = data_series[valid_mask]
    valid_weights = weights_series[valid_mask]
    
    if use_weights:
        # Weighted mean
        weighted_mean = np.average(valid_data, weights=valid_weights)
        
        # Weighted median (approximate using sorted weighted values)
        sorted_indices = np.argsort(valid_data)
        sorted_data = valid_data.values[sorted_indices] if hasattr(valid_data, 'values') else valid_data[sorted_indices]
        sorted_weights = valid_weights.values[sorted_indices] if hasattr(valid_weights, 'values') else valid_weights[sorted_indices]
        
        cumulative_weights = np.cumsum(sorted_weights)
        total_weight = cumulative_weights[-1]
        median_weight = total_weight / 2.0
        
        # Find the value where cumulative weight crosses 50%
        median_idx = np.searchsorted(cumulative_weights, median_weight)
        if median_idx >= len(sorted_data):
            median_idx = len(sorted_data) - 1
        weighted_median = sorted_data[median_idx]
        
        return round(weighted_mean, 1), round(weighted_median, 1)
    else:
        # Regular unweighted statistics
        return round(valid_data.mean(), 1), round(valid_data.median(), 1)

def process_lake_file_all_models(csv_file, uncertainty_combinations):
    """Process a single lake file for ALL model configurations
    
    Parameters:
    -----------
    csv_file : Path
        Path to the lake CSV file
    uncertainty_combinations : list
        List of uncertainty configuration dictionaries from input_uncertainties.csv
    """
    
    try:
        # Read CSV with string dtypes for ID columns
        df = pd.read_csv(csv_file, dtype={'gage_id': str, 'swot_lake_id': str})
        
        if df.empty:
            return None
            
        lake_id = csv_file.stem.replace("_daily", "")
        
        # Check for required columns
        if 'swot_wse_anomaly' not in df.columns or 'wsa' not in df.columns:
            return None
            
        print(f"Processing {lake_id}...")
        
        # Filter to rows with valid WSE and area data
        df_filtered = df[(df['swot_wse_anomaly'].notna()) & (df['wsa'].notna())].copy()
        
        if len(df_filtered) < 10:
            print(f"  Insufficient data for {lake_id}")
            return None
        
        all_results = []
        
        # Process each model configuration from input file
        for config in uncertainty_combinations:
            # Handle column name variations in input file
            swot_filter = config.get('swot filter', config.get('swot_filter', 'optimal'))
            filter_type = 'opt' if swot_filter == 'optimal' else 'filt'
            
            # Get uncertainty parameters with correct column names
            wse_std = config.get('wse', config.get('wse_std', 0.1))
            wse_temporal_std = config.get('wse_temporal', config.get('wse_temporal_std', 0.0))
            wse_filt_std = config.get('wse_filt', config.get('wse_filt_std', 0.0))
            wsa_percent_error = config.get('wsa', config.get('wsa_percent_error', 15.0))
            
            # Calculate normalized uncertainty breakdown for this configuration
            uncertainty_results = calculate_storage_uncertainty_components_two_way_normalized(
                df_filtered, 
                wse_std=wse_std,
                wse_temporal_std=wse_temporal_std,
                wse_filt_std=wse_filt_std,
                wsa_percent_error=wsa_percent_error,
                filter_type=filter_type
            )
            
            if not uncertainty_results.empty:
                # Add metadata (handle column name variations)
                uncertainty_results['lake_id'] = lake_id
                uncertainty_results['model'] = config['model']
                uncertainty_results['swot_filter'] = config.get('swot filter', config.get('swot_filter', 'optimal'))
                uncertainty_results['temporal'] = config['temporal']
                all_results.append(uncertainty_results)
        
        print(f"  Processed {len(all_results)} model configurations for {lake_id}")
        
        if all_results:
            return pd.concat(all_results, ignore_index=True)
        else:
            return None
            
    except Exception as e:
        print(f"Error processing {csv_file.name}: {e}")
        return None

def main(use_height_weighting=False):
    """Main function to run normalized uncertainty attribution analysis for ALL models
    
    Parameters:
    -----------
    use_height_weighting : bool
        If True, use inverse height weighting for equal contribution across water levels.
        If False, use standard unweighted statistics (current approach).
    """
    
    print("NORMALIZED STORAGE UNCERTAINTY ATTRIBUTION ANALYSIS")
    if use_height_weighting:
        print("Using HEIGHT-WEIGHTED statistics (equal water level contribution)")
    else:
        print("Using UNWEIGHTED statistics (current approach)")
    print("Processing ALL model variants in single run")
    print("=" * 80)
    
    # Define paths
    data_dir = Path("/Users/ericlevenson/University of Oregon Dropbox/Eric Levenson/SWOT/production/data/timeseries/benchmark_daily")
    output_dir = Path("/Users/ericlevenson/University of Oregon Dropbox/Eric Levenson/SWOT/production/experiments/storage_normalized/results/storage_uncertainty")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load uncertainty combinations from input file (from the CORRECT location)
    input_file = Path("/Users/ericlevenson/University of Oregon Dropbox/Eric Levenson/SWOT/production/experiments/results/storage_uncertainty/input_uncertainties.csv")
    
    if not input_file.exists():
        print(f"Error: Input file not found at {input_file}")
        return
    
    uncertainty_df = pd.read_csv(input_file)
    print(f"Loaded {len(uncertainty_df)} uncertainty combinations from {input_file}")
    
    # Convert to list of dictionaries
    uncertainty_combinations = uncertainty_df.to_dict('records')
    
    # Get all CSV files
    csv_files = list(data_dir.glob("*_daily.csv"))
    print(f"Found {len(csv_files)} lake files to process")
    
    # Process each lake file for ALL models
    all_lake_results = []
    processed_count = 0
    skipped_count = 0
    
    for i, csv_file in enumerate(csv_files):
        if i % 20 == 0 and i > 0:
            print(f"Progress: {i}/{len(csv_files)} files checked, {processed_count} lakes with valid data")
        
        result = process_lake_file_all_models(csv_file, uncertainty_combinations)
        
        if result is not None:
            all_lake_results.append(result)
            processed_count += 1
        else:
            skipped_count += 1
    
    print(f"\nFinal: Processed {processed_count} lakes, skipped {skipped_count} (no valid data)")
    
    if not all_lake_results:
        print("No valid results obtained")
        return
    
    # Combine all results
    combined_results = pd.concat(all_lake_results, ignore_index=True)
    print(f"Total observations: {len(combined_results)}")
    
    # Group by model configuration and calculate summary statistics matching original output
    summary_stats = []
    
    # Get unique combinations from input file to ensure we include all models
    for idx, config in uncertainty_df.iterrows():
        model = config['model']
        swot_filter = config.get('swot filter', config.get('swot_filter', 'optimal'))
        temporal = config['temporal']
        
        # Filter combined results for this configuration
        group_df = combined_results[
            (combined_results['model'] == model) &
            (combined_results['swot_filter'] == swot_filter) &
            (combined_results['temporal'] == temporal)
        ]
        
        # Load benchmark errors for comparison
        benchmark_errors = load_benchmark_normalized_errors(model, swot_filter, temporal)
        
        # Calculate all the statistics that the original outputs
        stats = {
            'model': model,
            'swot_filter': swot_filter,
            'temporal': temporal,
            
            # Propagated uncertainty values (from input file, with correct column names)
            'propagated_wse': config.get('wse', config.get('wse_std', 0.1)),
            'propagated_wse_temporal': config.get('wse_temporal', config.get('wse_temporal_std', 0.0)),
            'propagated_wse_filt': config.get('wse_filt', config.get('wse_filt_std', 0.0)),
            'propagated_wsa': config.get('wsa', config.get('wsa_percent_error', 15.0)),
            
            # Counts
            'n_observations': len(group_df) if len(group_df) > 0 else 0,
            'n_lakes': group_df['lake_id'].nunique() if len(group_df) > 0 else 0,
            
        }
        
        # Calculate statistics using weighted or unweighted approach
        if len(group_df) > 0:
            # WSE contributions (mean and median)
            wse_total_mean, wse_total_median = calculate_weighted_statistics(
                group_df['wse_contribution_pct'], group_df['height_weight'], use_height_weighting)
            stats['wse_total_contribution_mean'] = wse_total_mean
            stats['wse_total_contribution_median'] = wse_total_median
            
            wse_base_mean, wse_base_median = calculate_weighted_statistics(
                group_df['wse_base_contribution_pct'], group_df['height_weight'], use_height_weighting)
            stats['wse_base_contribution_mean'] = wse_base_mean
            stats['wse_base_contribution_median'] = wse_base_median
            
            wse_temp_mean, wse_temp_median = calculate_weighted_statistics(
                group_df['wse_temporal_contribution_pct'], group_df['height_weight'], use_height_weighting)
            stats['wse_temporal_contribution_mean'] = wse_temp_mean
            stats['wse_temporal_contribution_median'] = wse_temp_median
            
            wse_filt_mean, wse_filt_median = calculate_weighted_statistics(
                group_df['wse_filt_contribution_pct'], group_df['height_weight'], use_height_weighting)
            stats['wse_filt_contribution_mean'] = wse_filt_mean
            stats['wse_filt_contribution_median'] = wse_filt_median
            
            # Area contributions
            area_mean, area_median = calculate_weighted_statistics(
                group_df['area_contribution_pct'], group_df['height_weight'], use_height_weighting)
            stats['area_contribution_mean'] = area_mean
            stats['area_contribution_median'] = area_median
            
            # Total normalized uncertainty
            total_mean, total_median = calculate_weighted_statistics(
                group_df['sigma_storage_total_norm'], group_df['height_weight'], use_height_weighting)
            stats['total_uncertainty_mean'] = total_mean
            stats['total_uncertainty_median'] = total_median
            
            # Predicted values (same as total uncertainty for normalized)
            stats['predicted_mean'] = total_mean
            stats['predicted_median'] = total_median
        else:
            # No data case
            stats.update({
                'wse_total_contribution_mean': np.nan,
                'wse_total_contribution_median': np.nan,
                'wse_base_contribution_mean': np.nan,
                'wse_base_contribution_median': np.nan,
                'wse_temporal_contribution_mean': np.nan,
                'wse_temporal_contribution_median': np.nan,
                'wse_filt_contribution_mean': np.nan,
                'wse_filt_contribution_median': np.nan,
                'area_contribution_mean': np.nan,
                'area_contribution_median': np.nan,
                'total_uncertainty_mean': np.nan,
                'total_uncertainty_median': np.nan,
                'predicted_mean': np.nan,
                'predicted_median': np.nan,
            })
        
        # Analysis type
        stats['analysis_type'] = temporal
        
        # Add benchmark comparison if available
        if benchmark_errors:
            stats['observed_mae'] = benchmark_errors['observed_mae']
            stats['observed_rmse'] = benchmark_errors['observed_rmse']
            stats['observed_std'] = benchmark_errors['observed_std']
            stats['observed_nse_median'] = benchmark_errors['observed_nse_median']
            stats['observed_nse_mean'] = benchmark_errors['observed_nse_mean']
            stats['benchmark_n_lakes'] = benchmark_errors['n_lakes']
            
            # Calculate ratios (predicted/observed)
            if benchmark_errors['observed_mae'] > 0:
                stats['ratio_mae'] = stats['predicted_median'] / benchmark_errors['observed_mae']
            else:
                stats['ratio_mae'] = np.nan
                
            if benchmark_errors['observed_rmse'] > 0:
                stats['ratio_rmse'] = stats['predicted_median'] / benchmark_errors['observed_rmse']
            else:
                stats['ratio_rmse'] = np.nan
        else:
            stats['observed_mae'] = np.nan
            stats['observed_rmse'] = np.nan
            stats['observed_std'] = np.nan
            stats['observed_nse_median'] = np.nan
            stats['observed_nse_mean'] = np.nan
            stats['benchmark_n_lakes'] = 0
            stats['ratio_mae'] = np.nan
            stats['ratio_rmse'] = np.nan
        
        summary_stats.append(stats)
    
    # Create summary DataFrame with all columns matching original
    summary_df = pd.DataFrame(summary_stats)
    
    # Save results
    suffix = "_weighted" if use_height_weighting else ""
    output_file = output_dir / f"storage_uncertainty_attribution_normalized{suffix}.csv"
    summary_df.to_csv(output_file, index=False)
    print(f"\nSaved normalized uncertainty attribution results to {output_file}")
    
    # Print summary table
    print("\n" + "="*80)
    print("NORMALIZED UNCERTAINTY ATTRIBUTION SUMMARY")
    print("="*80)
    print("\nAll values in % of storage variability:")
    display_cols = ['model', 'swot_filter', 'temporal', 
                   'predicted_uncertainty_median', 'observed_rmse',
                   'wse_contribution_median', 'area_contribution_median', 'n_lakes']
    
    # Filter to columns that exist
    display_cols = [col for col in display_cols if col in summary_df.columns]
    print(summary_df[display_cols].to_string(index=False))
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print(f"Processed {len(summary_df)} model configurations")
    print(f"Total lakes analyzed: {processed_count}")
    print("="*80)

if __name__ == "__main__":
    # Configuration: Set to True to use height-weighted statistics for equal water level contribution
    # Set to False for standard unweighted statistics (current approach)
    USE_HEIGHT_WEIGHTING = False # Change this to True to enable weighting
    
    main(use_height_weighting=USE_HEIGHT_WEIGHTING)