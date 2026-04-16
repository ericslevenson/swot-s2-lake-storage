#!/usr/bin/env python3
"""
Normalized Storage Analysis Script

Processes benchmark daily CSV files and calculates normalized storage error metrics
where errors are expressed as percentage of storage variability (1st to 99th percentile range).

Final outputs:
1. benchmark_storage_error_summary_stats_normalized.csv
2. benchmark_storage_normalized_error_metrics_boxplots.png  
3. benchmark_storage_error_metrics_boxplots_comparison_normalized.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from pathlib import Path
import warnings
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
warnings.filterwarnings('ignore')

def acre_feet_to_km3(acre_feet_value):
    """Convert acre-feet to cubic kilometers"""
    if pd.isna(acre_feet_value):
        return np.nan
    # 1 acre-foot = 1233.48 m³, 1 km³ = 1e9 m³
    return (acre_feet_value * 1233.48) / 1e9

def calculate_normalized_error_metrics(observed, predicted, lake_id=None):
    """Calculate normalized error metrics between observed and predicted values
    
    Parameters:
    -----------
    observed : array-like
        Observed values (storage anomalies in km³)
    predicted : array-like  
        Predicted values (storage anomalies in km³)
    lake_id : str, optional
        Lake ID for debugging purposes
    
    Returns:
    --------
    Dictionary with error metrics where errors are normalized by storage variability
    """
    # Remove NaN values for calculation
    mask = ~(pd.isna(observed) | pd.isna(predicted))
    obs = observed[mask]
    pred = predicted[mask]
    
    if len(obs) == 0:
        return {
            'count': 0,
            'mae': np.nan,
            'rmse': np.nan,
            'bias': np.nan,
            'std_error': np.nan,
            'p68_abs_error': np.nan,
            'pearson_r': np.nan,
            'nse': np.nan,
            'storage_range_p1_p99': np.nan
        }
    
    # Calculate storage range (1st to 99th percentile) for normalization
    storage_range = np.percentile(obs, 99) - np.percentile(obs, 1)
    
    if storage_range <= 0:
        # If range is zero or negative, cannot normalize
        return {
            'count': len(obs),
            'mae': np.nan,
            'rmse': np.nan,
            'bias': np.nan,
            'std_error': np.nan,
            'p68_abs_error': np.nan,
            'pearson_r': np.nan,
            'nse': np.nan,
            'storage_range_p1_p99': storage_range
        }
    
    # Calculate errors and normalize by storage range
    errors = (pred - obs) / storage_range * 100.0  # Convert to percentage
    abs_errors = np.abs(errors)
    
    # Normalized metrics (as percentages)
    mae = np.mean(abs_errors)
    rmse = np.sqrt(np.mean(errors**2))
    bias = np.mean(errors)
    std_error = np.std(errors)
    p68_abs_error = np.percentile(abs_errors, 68)
    
    # Correlation and NSE remain unchanged (dimensionless)
    if len(obs) > 1 and np.std(obs) > 0 and np.std(pred) > 0:
        pearson_r = np.corrcoef(obs, pred)[0, 1]
    else:
        pearson_r = np.nan
    
    # Nash-Sutcliffe Efficiency
    if np.var(obs) > 0:
        nse = 1 - np.sum((obs - pred)**2) / np.sum((obs - np.mean(obs))**2)
    else:
        nse = np.nan
    
    return {
        'count': len(obs),
        'mae': round(mae, 1),  # Round to 1 decimal place
        'rmse': round(rmse, 1),
        'bias': round(bias, 1),
        'std_error': round(std_error, 1),
        'p68_abs_error': round(p68_abs_error, 1),
        'pearson_r': pearson_r,
        'nse': nse,
        'storage_range_p1_p99': storage_range
    }

def calculate_overall_metrics_from_pooled_residuals(data_dir, storage_variants):
    """Calculate overall metrics by pooling all residuals across lakes"""
    
    csv_files = list(data_dir.glob("*_daily.csv"))
    
    overall_results = []
    
    for variant_name, variant_col in storage_variants.items():
        print(f"Calculating overall metrics for {variant_name}...")
        
        # Extract model, filter, temporal info
        parts = variant_name.split('_')
        model = parts[0]
        filter_type = parts[1] 
        temporal_type = parts[2]
        
        # Collect all residuals across lakes for this variant
        all_observed = []
        all_predicted = []
        lake_count = 0
        
        for csv_file in csv_files:
            try:
                # Read CSV
                df = pd.read_csv(csv_file, dtype={'gage_id': str, 'swot_lake_id': str})
                
                if df.empty or variant_col not in df.columns:
                    continue
                
                # Convert storage columns from acre-feet to km³
                all_storage_cols = ['storage_anomaly', 'storage_anomaly_opt', 'storage_anomaly_filt', variant_col]
                for col in all_storage_cols:
                    if col in df.columns:
                        df[col] = df[col].apply(acre_feet_to_km3)
                
                # Determine reference column
                if filter_type == 'opt':
                    reference_col = 'storage_anomaly_opt'
                elif filter_type == 'filt':
                    reference_col = 'storage_anomaly_filt'
                else:
                    continue
                
                if reference_col not in df.columns:
                    continue
                
                # Apply filtering for discrete SWOT-based models
                if temporal_type == 'dis' and model in ['swot', 'swots2', 'static']:
                    if filter_type == 'opt':
                        if 'swot_wse_abs_error' not in df.columns:
                            continue
                        quality_filter = ((df['date'] > '2023-07-21') & (df['swot_wse_abs_error'] < 0.283))
                    elif filter_type == 'filt':
                        if 'adaptive_filter' not in df.columns:
                            continue
                        quality_filter = ((df['date'] > '2023-07-21') & (df['adaptive_filter'] == 0))
                    else:
                        continue
                    mask = df[reference_col].notna() & df[variant_col].notna() & quality_filter
                else:
                    mask = df[reference_col].notna() & df[variant_col].notna()
                
                if mask.sum() < 2:
                    continue
                
                # Get gauge storage for normalization
                if reference_col in df.columns:
                    gauge_storage = df[reference_col]
                elif 'storage_anomaly' in df.columns:
                    gauge_storage = df['storage_anomaly']
                else:
                    continue
                
                # Calculate storage range for this lake
                valid_storage = gauge_storage.dropna()
                if len(valid_storage) < 10:
                    continue
                
                storage_range = np.percentile(valid_storage, 99) - np.percentile(valid_storage, 1)
                if storage_range <= 1e-6:
                    continue
                
                # Get observed and predicted values for this lake
                observed = df.loc[mask, reference_col].values
                predicted = df.loc[mask, variant_col].values
                
                # Normalize residuals by this lake's storage range
                residuals = (predicted - observed) / storage_range * 100.0  # Convert to %
                
                # Add to pooled data (store as normalized residuals)
                all_observed.extend([0] * len(residuals))  # Baseline is 0 for residuals
                all_predicted.extend(residuals)  # Normalized residuals
                lake_count += 1
                
            except Exception as e:
                continue
        
        if len(all_predicted) == 0:
            continue
        
        # Calculate overall metrics from pooled normalized residuals
        all_predicted = np.array(all_predicted)
        abs_residuals = np.abs(all_predicted)
        
        # Apply P1-P99 outlier-robust filtering to remove gauge errors
        p1_threshold = np.percentile(all_predicted, 1)
        p99_threshold = np.percentile(all_predicted, 99)
        robust_mask = (all_predicted >= p1_threshold) & (all_predicted <= p99_threshold)
        
        robust_residuals = all_predicted[robust_mask]
        robust_abs_residuals = np.abs(robust_residuals)
        
        overall_metrics = {
            'model': model,
            'filter_type': filter_type,
            'temporal_type': temporal_type,
            'variant_name': variant_name,
            'n_lakes': lake_count,
            'n_observations': len(all_predicted),
            'n_observations_robust': len(robust_residuals),
            
            # Overall metrics from P1-P99 outlier-robust pooled residuals (% of storage variability)
            'overall_mae': round(np.mean(robust_abs_residuals), 1),
            'overall_rmse': round(np.sqrt(np.mean(robust_residuals**2)), 1),
            'overall_bias': round(np.mean(robust_residuals), 1),
            'overall_std_error': round(np.std(robust_residuals), 1),
            'overall_p68_abs_error': round(np.percentile(robust_abs_residuals, 68), 1),
        }
        
        overall_results.append(overall_metrics)
    
    return pd.DataFrame(overall_results)

def process_storage_variants_data():
    """Process all benchmark_daily CSV files for normalized storage anomaly variants"""
    
    # Define paths
    data_dir = PROJECT_ROOT / "data/benchmark_timeseries"
    output_dir = PROJECT_ROOT / "analysis/storage_estimation_assessment/results_normalized"
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all CSV files
    csv_files = list(data_dir.glob("*_daily.csv"))
    print(f"Found {len(csv_files)} CSV files to process")
    
    # Storage for all data
    all_data = []
    
    # Column mapping for the 16 storage variants
    storage_variants = {
        'swot_opt_dis': 'swot_opt_dis',
        'swot_opt_con': 'swot_opt_con', 
        'swot_filt_dis': 'swot_filt_dis',
        'swot_filt_con': 'swot_filt_con',
        'swots2_opt_dis': 'swots2_opt_dis',
        'swots2_opt_con': 'swots2_opt_con',
        'swots2_filt_dis': 'swots2_filt_dis', 
        'swots2_filt_con': 'swots2_filt_con',
        's2_opt_dis': 's2_opt_dis',
        's2_opt_con': 's2_opt_con',
        's2_filt_dis': 's2_filt_dis',
        's2_filt_con': 's2_filt_con',
        'static_opt_dis': 'static_opt_dis',
        'static_opt_con': 'static_opt_con', 
        'static_filt_dis': 'static_filt_dis',
        'static_filt_con': 'static_filt_con'
    }
    
    for csv_file in csv_files:
        try:
            # Read CSV
            df = pd.read_csv(csv_file, dtype={'gage_id': str, 'swot_lake_id': str})
            
            if df.empty:
                continue
                
            lake_id = csv_file.stem.replace("_daily", "")
            
            # Convert all storage columns from acre-feet to km³
            all_storage_cols = ['storage_anomaly', 'storage_anomaly_opt', 'storage_anomaly_filt'] + list(storage_variants.keys())
            for col in all_storage_cols:
                if col in df.columns:
                    df[col] = df[col].apply(acre_feet_to_km3)
            
            # Check if we have any reference data (either opt or filt)
            has_opt_ref = 'storage_anomaly_opt' in df.columns and df['storage_anomaly_opt'].notna().any()
            has_filt_ref = 'storage_anomaly_filt' in df.columns and df['storage_anomaly_filt'].notna().any()
            
            if not has_opt_ref and not has_filt_ref:
                continue
                
            # For each storage variant, calculate normalized error metrics
            for variant_name, variant_col in storage_variants.items():
                if variant_col in df.columns:
                    # Extract model, filter, and temporal info from column name
                    parts = variant_name.split('_')
                    model = parts[0]
                    filter_type = parts[1] 
                    temporal_type = parts[2]
                    
                    # Determine reference column based on filter type
                    if filter_type == 'opt':
                        reference_col = 'storage_anomaly_opt'
                    elif filter_type == 'filt':
                        reference_col = 'storage_anomaly_filt'
                    else:
                        continue
                    
                    # Check if reference column exists
                    if reference_col not in df.columns:
                        continue
                    
                    # Apply appropriate filter for discrete SWOT-based models only
                    if temporal_type == 'dis' and model in ['swot', 'swots2', 'static']:
                        # Apply quality filter for discrete SWOT-based models
                        if filter_type == 'opt':
                            if 'swot_wse_abs_error' not in df.columns:
                                print(f"    Warning: Cannot apply optimal filter for {variant_name} - swot_wse_abs_error missing")
                                continue
                            quality_filter = ((df['date'] > '2023-07-21') & (df['swot_wse_abs_error'] < 0.283))
                        elif filter_type == 'filt':
                            if 'adaptive_filter' not in df.columns:
                                print(f"    Warning: Cannot apply adaptive filter for {variant_name} - adaptive_filter missing")
                                continue
                            quality_filter = ((df['date'] > '2023-07-21') & (df['adaptive_filter'] == 0))
                        else:
                            continue
                        
                        # Get pairs that pass quality filter and have valid data
                        mask = df[reference_col].notna() & df[variant_col].notna() & quality_filter
                    else:
                        # For continuous models and S2 model, use all available pairs
                        mask = df[reference_col].notna() & df[variant_col].notna()
                    
                    if mask.sum() < 2:  # Need at least 2 data points
                        continue
                        
                    observed = df.loc[mask, reference_col].values
                    predicted = df.loc[mask, variant_col].values
                    
                    # Calculate normalized error metrics
                    metrics = calculate_normalized_error_metrics(observed, predicted, lake_id)
                    
                    # Store results
                    result_row = {
                        'lake_id': lake_id,
                        'model': model,
                        'filter_type': filter_type,
                        'temporal_type': temporal_type,
                        'variant_name': variant_name,
                        **metrics
                    }
                    all_data.append(result_row)
                    
        except Exception as e:
            print(f"Error processing {csv_file.name}: {e}")
            continue
    
    if len(all_data) == 0:
        print("No data processed successfully")
        return None
        
    # Create comprehensive DataFrame
    results_df = pd.DataFrame(all_data)
    
    # Save per-lake results
    per_lake_file = output_dir / 'benchmark_storage_variants_per_lake_stats_normalized.csv'
    results_df.to_csv(per_lake_file, index=False)
    print(f"Saved per-lake normalized statistics to {per_lake_file}")
    
    # Create aggregated summary statistics
    summary_stats = []
    
    for variant_name in storage_variants.keys():
        variant_data = results_df[results_df['variant_name'] == variant_name]
        
        if len(variant_data) == 0:
            continue
            
        # Extract model, filter, temporal info
        parts = variant_name.split('_')
        model = parts[0]
        filter_type = parts[1]
        temporal_type = parts[2]
        
        # Calculate aggregated metrics (all in normalized percentages)
        summary_row = {
            'model': model,
            'filter_type': filter_type,
            'temporal_type': temporal_type,
            'variant_name': variant_name,
            'n_lakes': len(variant_data),
            'mae_mean': round(variant_data['mae'].mean(), 1),
            'mae_std': round(variant_data['mae'].std(), 1),
            'mae_median': round(variant_data['mae'].median(), 1),
            'rmse_mean': round(variant_data['rmse'].mean(), 1),
            'rmse_std': round(variant_data['rmse'].std(), 1), 
            'rmse_median': round(variant_data['rmse'].median(), 1),
            'bias_mean': round(variant_data['bias'].mean(), 1),
            'bias_std': round(variant_data['bias'].std(), 1),
            'bias_median': round(variant_data['bias'].median(), 1),
            'nse_mean': variant_data['nse'].mean(),
            'nse_std': variant_data['nse'].std(),
            'nse_median': variant_data['nse'].median(),
            'pearson_r_mean': variant_data['pearson_r'].mean(),
            'pearson_r_std': variant_data['pearson_r'].std(),
            'pearson_r_median': variant_data['pearson_r'].median(),
            'r_squared_mean': (variant_data['pearson_r'] ** 2).mean(),
            'r_squared_std': (variant_data['pearson_r'] ** 2).std(),
            'r_squared_median': (variant_data['pearson_r'] ** 2).median(),
            'std_error_mean': round(variant_data['std_error'].mean(), 1),
            'std_error_std': round(variant_data['std_error'].std(), 1),
            'std_error_median': round(variant_data['std_error'].median(), 1),
            'p68_abs_error_mean': round(variant_data['p68_abs_error'].mean(), 1),
            'p68_abs_error_std': round(variant_data['p68_abs_error'].std(), 1),
            'p68_abs_error_median': round(variant_data['p68_abs_error'].median(), 1)
        }
        summary_stats.append(summary_row)
    
    summary_df = pd.DataFrame(summary_stats)
    
    # Calculate overall metrics from pooled residuals
    print("\nCalculating overall metrics from pooled residuals...")
    overall_df = calculate_overall_metrics_from_pooled_residuals(data_dir, storage_variants)
    
    # Merge overall metrics into summary_df as additional columns
    # Create merge key for joining
    summary_df['merge_key'] = summary_df['variant_name']
    overall_df['merge_key'] = overall_df['variant_name']
    
    # Merge on variant_name, keeping all summary rows
    summary_with_overall = summary_df.merge(
        overall_df[['merge_key', 'overall_mae', 'overall_rmse', 'overall_bias', 'overall_std_error', 'overall_p68_abs_error', 'n_observations', 'n_observations_robust']], 
        on='merge_key', 
        how='left'
    )
    
    # Clean up merge column
    summary_with_overall = summary_with_overall.drop('merge_key', axis=1)
    
    # Save combined summary with overall metrics as additional columns
    summary_file = output_dir / 'benchmark_storage_variants_summary_stats_normalized.csv'
    summary_with_overall.to_csv(summary_file, index=False)
    print(f"Saved combined normalized summary statistics (with overall metrics) to {summary_file}")
    
    return results_df, summary_with_overall

def create_inverted_storage_boxplots(df, output_dir):
    """Create inverted 2-panel plot: x-axis = filter-temporal combo, colors = models"""
    
    # Toggle to include/exclude S2 models
    include_s2 = False  # Set to True to include S2 models in plots
    
    # Prepare data for plotting
    plot_data = []
    if include_s2:
        models = ['swot', 'swots2', 's2', 'static']
    else:
        models = ['static','swot',  'swots2',]
    filter_types = ['opt', 'filt']
    temporal_types = ['dis', 'con']
    
    # Define color mapping for models
    model_color_mapping = {
        'SWOT': '#377eb8',      # Blue
        'SWOTS2': '#984ea3',    # Purple  
        'S2': '#e41a1c',        # Red (kept for when S2 is toggled on)
        'STATIC': '#4daf4a'     # Green
    }
    
    for model in models:
        for filt in filter_types:
            for temp in temporal_types:
                model_data = df[(df['model'] == model) & 
                               (df['filter_type'] == filt) & 
                               (df['temporal_type'] == temp)]
                
                # Create full descriptive labels
                if filt == 'opt' and temp == 'dis':
                    filter_temporal_label = 'Optimal\nDiscrete'
                elif filt == 'opt' and temp == 'con':
                    filter_temporal_label = 'Optimal\nContinuous'
                elif filt == 'filt' and temp == 'dis':
                    filter_temporal_label = 'Functional\nDiscrete'
                elif filt == 'filt' and temp == 'con':
                    filter_temporal_label = 'Functional\nContinuous'
                
                for _, row in model_data.iterrows():
                    plot_data.append({
                        'model': model.upper(),
                        'filter_temporal': filter_temporal_label,
                        'rmse': row['rmse'],
                        'nse': row['nse']
                    })
    
    if len(plot_data) == 0:
        print("No plot data available for inverted plot")
        return
        
    plot_df = pd.DataFrame(plot_data)
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    # Remove title
    
    metrics = ['rmse', 'nse']
    metric_labels = ['Normalized RMSE (%)', 'Nash-Sutcliffe Efficiency']
    
    # Define desired order for filter-temporal combinations
    filter_temporal_order = ['Optimal\nDiscrete', 'Functional\nDiscrete', 'Functional\nContinuous']
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[i]
        
        # Set grid BEFORE creating the boxplot to ensure it appears below
        ax.set_axisbelow(True)
        ax.grid(True, alpha=0.7, zorder=0)
        
        # Create grouped boxplot with models as colors, filter-temporal as x-axis
        box_plot = sns.boxplot(data=plot_df, x='filter_temporal', y=metric, hue='model', 
                              palette=model_color_mapping, ax=ax,
                              flierprops={'alpha': 0.3, 'markersize': 3},
                              order=filter_temporal_order,
                              zorder=3)
        
        # Add defined boundaries around the plot
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.2)
            spine.set_visible(True)
        
        # Remove title
        ax.set_xlabel('')  # Remove x-axis label
        ax.set_ylabel('')  # Remove y-axis label
        
        # Remove x-axis tick labels but keep tick marks
        ax.set_xticklabels([])
        
        # Set y-axis limits
        if metric == 'rmse':
            ax.set_ylim(0, 40)  # Changed to 0-40 as requested
        elif metric == 'nse':
            ax.set_ylim(0, 1)
        
        # Remove legend
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    
    plt.tight_layout(w_pad=3.0)
    
    # Save plot
    filename = 'storage_error_inverted_boxplots_normalized.png'
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved normalized inverted boxplot to {filepath}")

def main_variants():
    """Main execution function for normalized storage variants analysis"""
    
    print("Starting benchmark normalized storage variants analysis...")
    
    # Process files and calculate normalized error metrics for all variants
    result = process_storage_variants_data()
    
    if result is not None:
        results_df, summary_df = result
        output_dir = PROJECT_ROOT / "analysis/storage_estimation_assessment/results_normalized"
        
        # Create plot using the per-lake results DataFrame
        create_inverted_storage_boxplots(results_df, output_dir)
        
        print("\nNormalized benchmark storage variants analysis completed successfully!")
        print(f"\nSummary: Processed {len(results_df)} model-lake combinations")
        print(f"Models: {sorted(results_df['model'].unique())}")
        print(f"Filter types: {sorted(results_df['filter_type'].unique())}")
        print(f"Temporal types: {sorted(results_df['temporal_type'].unique())}")
        
        # Print comparison between per-lake averages and overall pooled metrics
        print("\nComparison: Per-lake averages vs Overall pooled metrics")
        print("="*70)
        print(f"{'Variant':<20} {'Avg RMSE':<10} {'Overall RMSE':<12} {'Avg MAE':<10} {'Overall MAE':<12}")
        print("-"*70)
        
        for variant in ['swot_opt_dis', 'swot_filt_con', 'swots2_opt_dis', 'static_filt_dis']:
            if variant in summary_df['variant_name'].values:
                row = summary_df[summary_df['variant_name'] == variant].iloc[0]
                overall_rmse = row.get('overall_rmse', 'N/A')
                overall_mae = row.get('overall_mae', 'N/A')
                if overall_rmse != 'N/A':
                    print(f"{variant:<20} {row['rmse_median']:>7.1f}%   {overall_rmse:>9.1f}%   {row['mae_median']:>7.1f}%   {overall_mae:>9.1f}%")
                else:
                    print(f"{variant:<20} {row['rmse_median']:>7.1f}%   {'N/A':>9}   {row['mae_median']:>7.1f}%   {'N/A':>9}")
    else:
        print("\nNo data was processed successfully.")

if __name__ == "__main__":
    # Run the normalized storage variants analysis
    main_variants()