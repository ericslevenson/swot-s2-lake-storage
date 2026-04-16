#!/usr/bin/env python3
"""
Consolidated Storage Analysis Script

Combines data processing and visualization for benchmark storage error metrics.
Processes benchmark daily CSV files and creates final plots and summary statistics.

Final outputs:
1. benchmark_storage_error_summary_stats.csv
2. benchmark_storage_normalized_error_metrics_boxplots.png  
3. benchmark_storage_error_metrics_boxplots_comparison.png
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

def calculate_error_metrics(observed, predicted, capacity=None):
    """Calculate error metrics between observed and predicted values
    
    Parameters:
    -----------
    observed : array-like
        Observed values (storage anomalies in km³)
    predicted : array-like  
        Predicted values (storage anomalies in km³)
    capacity : float, optional
        Storage range (max - min storage in km³) for normalization
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
            'mae_norm': np.nan,
            'rmse_norm': np.nan,
            'bias_norm': np.nan,
            'std_error_norm': np.nan,
            'p68_abs_error_norm': np.nan
        }
    
    # Calculate errors
    errors = pred - obs
    abs_errors = np.abs(errors)
    
    # Basic metrics
    mae = np.mean(abs_errors)
    rmse = np.sqrt(np.mean(errors**2))
    bias = np.mean(errors)
    
    # Additional metrics
    std_error = np.std(errors)  # 1 standard deviation of errors
    p68_abs_error = np.percentile(abs_errors, 68)  # 68th percentile absolute error
    
    # Pearson correlation coefficient
    if len(obs) > 1 and np.std(obs) > 0 and np.std(pred) > 0:
        pearson_r = np.corrcoef(obs, pred)[0, 1]
    else:
        pearson_r = np.nan
    
    # Nash-Sutcliffe Efficiency
    if np.var(obs) > 0:
        nse = 1 - np.sum((obs - pred)**2) / np.sum((obs - np.mean(obs))**2)
    else:
        nse = np.nan
    
    # Normalized metrics (divide by capacity if provided)
    if capacity is not None and capacity > 0:
        mae_norm = mae / capacity
        rmse_norm = rmse / capacity
        bias_norm = bias / capacity
        std_error_norm = std_error / capacity
        p68_abs_error_norm = p68_abs_error / capacity
    else:
        mae_norm = np.nan
        rmse_norm = np.nan
        bias_norm = np.nan
        std_error_norm = np.nan
        p68_abs_error_norm = np.nan
    
    return {
        'count': len(obs),
        'mae': mae,
        'rmse': rmse,
        'bias': bias,
        'std_error': std_error,
        'p68_abs_error': p68_abs_error,
        'pearson_r': pearson_r,
        'nse': nse,
        'mae_norm': mae_norm,
        'rmse_norm': rmse_norm,
        'bias_norm': bias_norm,
        'std_error_norm': std_error_norm,
        'p68_abs_error_norm': p68_abs_error_norm
    }

def calculate_overall_metrics_from_pooled_residuals_km3(data_dir, storage_variants):
    """Calculate overall metrics by pooling all residuals across lakes (in km³ units)"""
    
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
                
                # Get observed and predicted values for this lake (in km³)
                observed = df.loc[mask, reference_col].values
                predicted = df.loc[mask, variant_col].values
                
                # Store raw residuals in km³
                residuals = predicted - observed
                
                # Add to pooled data
                all_observed.extend(observed)
                all_predicted.extend(predicted)
                lake_count += 1
                
            except Exception as e:
                continue
        
        if len(all_predicted) == 0:
            continue
        
        # Calculate overall metrics from pooled residuals in km³
        all_observed = np.array(all_observed)
        all_predicted = np.array(all_predicted)
        residuals = all_predicted - all_observed
        abs_residuals = np.abs(residuals)
        
        # Apply P1-P99 outlier-robust filtering to remove gauge errors
        p1_threshold = np.percentile(residuals, 1)
        p99_threshold = np.percentile(residuals, 99)
        robust_mask = (residuals >= p1_threshold) & (residuals <= p99_threshold)
        
        robust_residuals = residuals[robust_mask]
        robust_abs_residuals = np.abs(robust_residuals)
        
        overall_metrics = {
            'model': model,
            'filter_type': filter_type,
            'temporal_type': temporal_type,
            'variant_name': variant_name,
            'n_lakes': lake_count,
            'n_observations': len(all_predicted),
            'n_observations_robust': len(robust_residuals),
            
            # Overall metrics from P1-P99 outlier-robust pooled residuals (in km³)
            'overall_mae': np.mean(robust_abs_residuals),
            'overall_rmse': np.sqrt(np.mean(robust_residuals**2)),
            'overall_bias': np.mean(robust_residuals),
            'overall_std_error': np.std(robust_residuals),
            'overall_p68_abs_error': np.percentile(robust_abs_residuals, 68),
        }
        
        overall_results.append(overall_metrics)
    
    return pd.DataFrame(overall_results)

def process_storage_variants_data():
    """Process all benchmark_daily CSV files for the new storage anomaly variants"""
    
    # Define paths
    data_dir = PROJECT_ROOT / "data/benchmark_timeseries"
    output_dir = PROJECT_ROOT / "analysis/storage_estimation_assessment/results"
    
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
                
            # For each storage variant, calculate error metrics against appropriate reference
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
                        continue  # Skip unknown filter types
                    
                    # Check if reference column exists
                    if reference_col not in df.columns:
                        continue
                    
                    # Apply appropriate filter for discrete SWOT-based models only
                    # Continuous models already include gap filling, so don't re-filter
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
                        # (S2 already has its own built-in quality filtering)
                        mask = df[reference_col].notna() & df[variant_col].notna()
                    
                    if mask.sum() < 2:  # Need at least 2 data points
                        continue
                        
                    observed = df.loc[mask, reference_col].values
                    predicted = df.loc[mask, variant_col].values
                    
                    # Calculate error metrics
                    metrics = calculate_error_metrics(observed, predicted)
                    
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
    per_lake_file = output_dir / 'benchmark_storage_variants_per_lake_stats.csv'
    results_df.to_csv(per_lake_file, index=False)
    print(f"Saved per-lake statistics to {per_lake_file}")
    
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
        
        # Calculate aggregated metrics
        summary_row = {
            'model': model,
            'filter_type': filter_type,
            'temporal_type': temporal_type,
            'variant_name': variant_name,
            'n_lakes': len(variant_data),
            'mae_mean': variant_data['mae'].mean(),
            'mae_std': variant_data['mae'].std(),
            'mae_median': variant_data['mae'].median(),
            'rmse_mean': variant_data['rmse'].mean(),
            'rmse_std': variant_data['rmse'].std(), 
            'rmse_median': variant_data['rmse'].median(),
            'bias_mean': variant_data['bias'].mean(),
            'bias_std': variant_data['bias'].std(),
            'bias_median': variant_data['bias'].median(),
            'nse_mean': variant_data['nse'].mean(),
            'nse_std': variant_data['nse'].std(),
            'nse_median': variant_data['nse'].median(),
            'pearson_r_mean': variant_data['pearson_r'].mean(),
            'pearson_r_std': variant_data['pearson_r'].std(),
            'pearson_r_median': variant_data['pearson_r'].median(),
            'r_squared_mean': (variant_data['pearson_r'] ** 2).mean(),
            'r_squared_std': (variant_data['pearson_r'] ** 2).std(),
            'r_squared_median': (variant_data['pearson_r'] ** 2).median(),
            'std_error_mean': variant_data['std_error'].mean(),
            'std_error_std': variant_data['std_error'].std(),
            'std_error_median': variant_data['std_error'].median(),
            'p68_abs_error_mean': variant_data['p68_abs_error'].mean(),
            'p68_abs_error_std': variant_data['p68_abs_error'].std(),
            'p68_abs_error_median': variant_data['p68_abs_error'].median()
        }
        summary_stats.append(summary_row)
    
    summary_df = pd.DataFrame(summary_stats)
    
    # Calculate overall metrics from pooled residuals
    print("\nCalculating overall metrics from pooled residuals...")
    overall_df = calculate_overall_metrics_from_pooled_residuals_km3(data_dir, storage_variants)
    
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
    summary_file = output_dir / 'benchmark_storage_variants_summary_stats.csv'
    summary_with_overall.to_csv(summary_file, index=False)
    print(f"Saved combined summary statistics (with overall metrics) to {summary_file}")
    
    return results_df, summary_with_overall

def remove_outliers_percentile(data, columns, lower_percentile=5, upper_percentile=95):
    """Remove outliers using percentile-based clipping"""
    data_clean = data.copy()
    for col in columns:
        if col in data_clean.columns:
            lower = data_clean[col].quantile(lower_percentile/100)
            upper = data_clean[col].quantile(upper_percentile/100)
            data_clean = data_clean[(data_clean[col] >= lower) & (data_clean[col] <= upper)]
    return data_clean

def create_temporal_comparison_boxplots(df, filter_type, output_dir):
    """Create boxplots comparing discrete vs continuous for a given filter type"""
    
    # Filter data for the specified filter type
    df_filtered = df[df['filter_type'] == filter_type].copy()
    
    if len(df_filtered) == 0:
        print(f"No data for filter type: {filter_type}")
        return
    
    # Prepare data for plotting
    plot_data = []
    models = ['swot', 'swots2', 's2', 'static']
    temporals = ['dis', 'con']
    
    for model in models:
        for temporal in temporals:
            model_data = df_filtered[(df_filtered['model'] == model) & (df_filtered['temporal_type'] == temporal)]
            for _, row in model_data.iterrows():
                plot_data.append({
                    'model': model.upper(),
                    'temporal': 'Discrete' if temporal == 'dis' else 'Continuous', 
                    'mae': row['mae'],
                    'rmse': row['rmse'],
                    'bias': row['bias'],
                    'nse': row['nse']
                })
    
    if len(plot_data) == 0:
        print(f"No plot data for filter type: {filter_type}")
        return
        
    plot_df = pd.DataFrame(plot_data)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Storage Anomaly Error Metrics: Discrete vs Continuous ({filter_type.title()} Filter)', fontsize=16, fontweight='bold')
    
    metrics = ['mae', 'rmse', 'bias', 'nse']
    metric_labels = ['Mean Absolute Error (km³)', 'Root Mean Square Error (km³)', 'Bias (km³)', 'Nash-Sutcliffe Efficiency']
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        row = i // 2
        col = i % 2
        
        sns.boxplot(data=plot_df, x='model', y=metric, hue='temporal', ax=axes[row, col])
        axes[row, col].set_title(label, fontweight='bold')
        axes[row, col].set_xlabel('Model')
        axes[row, col].set_ylabel(label.split('(')[0].strip())
        axes[row, col].grid(True, alpha=0.3)
        
        # Set readable y-axis limits
        if metric in ['mae', 'rmse']:
            axes[row, col].set_ylim(0, .1)
        elif metric == 'bias':
            axes[row, col].set_ylim(-.2, .2)
            axes[row, col].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        elif metric == 'nse':
            axes[row, col].set_ylim(-1, 1)
            axes[row, col].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        if i == 0:
            axes[row, col].legend(title='Temporal Type', loc='upper right')
        else:
            axes[row, col].get_legend().remove()
    
    plt.tight_layout()
    
    # Save plot
    filename = f'storage_error_temporal_comparison_{filter_type}.png'
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved temporal comparison plot to {filepath}")

def create_filter_comparison_boxplots(df, temporal_type, output_dir):
    """Create boxplots comparing opt vs filt for a given temporal type"""
    
    # Filter data for the specified temporal type
    df_filtered = df[df['temporal_type'] == temporal_type].copy()
    
    if len(df_filtered) == 0:
        print(f"No data for temporal type: {temporal_type}")
        return
    
    # Prepare data for plotting
    plot_data = []
    models = ['swot', 'swots2', 's2', 'static']
    filters = ['opt', 'filt']
    
    for model in models:
        for filter_type in filters:
            model_data = df_filtered[(df_filtered['model'] == model) & (df_filtered['filter_type'] == filter_type)]
            for _, row in model_data.iterrows():
                plot_data.append({
                    'model': model.upper(),
                    'filter': 'Optimal' if filter_type == 'opt' else 'Filtered',
                    'mae': row['mae'],
                    'rmse': row['rmse'], 
                    'bias': row['bias'],
                    'nse': row['nse']
                })
    
    if len(plot_data) == 0:
        print(f"No plot data for temporal type: {temporal_type}")
        return
        
    plot_df = pd.DataFrame(plot_data)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Storage Anomaly Error Metrics: Optimal vs Filtered ({temporal_type.title()} Temporal)', fontsize=16, fontweight='bold')
    
    metrics = ['mae', 'rmse', 'bias', 'nse']
    metric_labels = ['Mean Absolute Error (km³)', 'Root Mean Square Error (km³)', 'Bias (km³)', 'Nash-Sutcliffe Efficiency']
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        row = i // 2
        col = i % 2
        
        sns.boxplot(data=plot_df, x='model', y=metric, hue='filter', ax=axes[row, col])
        axes[row, col].set_title(label, fontweight='bold')
        axes[row, col].set_xlabel('Model')
        axes[row, col].set_ylabel(label.split('(')[0].strip())
        axes[row, col].grid(True, alpha=0.3)
        
        # Set readable y-axis limits
        if metric in ['mae', 'rmse']:
            axes[row, col].set_ylim(0, .1)
        elif metric == 'bias':
            axes[row, col].set_ylim(-1, 1)
            axes[row, col].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        elif metric == 'nse':
            axes[row, col].set_ylim(-1, 1)
            axes[row, col].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        if i == 0:
            axes[row, col].legend(title='Filter Type', loc='upper right')
        else:
            axes[row, col].get_legend().remove()
    
    plt.tight_layout()
    
    # Save plot
    filename = f'storage_error_filter_comparison_{temporal_type}.png'
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved filter comparison plot to {filepath}")

def create_combined_storage_boxplots(df, output_dir):
    """Create combined 2-panel plot showing RMSE and NSE by model with filter/temporal combinations"""
    
    # Toggle to include/exclude S2 models
    include_s2 = False  # Set to True to include S2 models in plots
    
    # Prepare data for plotting
    plot_data = []
    if include_s2:
        models = ['swot', 'swots2', 's2', 'static']
    else:
        models = ['swot', 'swots2', 'static']
    filter_types = ['opt', 'filt']
    temporal_types = ['dis', 'con']
    
    # Define color mapping for filter/temporal combinations
    color_mapping = {
        'Optimal\nDiscrete': '#1f77b4',        # Blue
        'Optimal\nContinuous': '#ff7f0e',      # Orange  
        'Functional\nDiscrete': '#2ca02c',     # Green
        'Functional\nContinuous': '#d62728'    # Red
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
        print("No plot data available")
        return
        
    plot_df = pd.DataFrame(plot_data)
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Storage Anomaly Error Metrics by Model and Filter/Temporal Combination', 
                fontsize=16, fontweight='bold')
    
    metrics = ['rmse', 'nse']
    metric_labels = ['Root Mean Square Error (km³)', 'Nash-Sutcliffe Efficiency']
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[i]
        
        # Create grouped boxplot with custom colors using palette
        box_plot = sns.boxplot(data=plot_df, x='model', y=metric, hue='filter_temporal', 
                              palette=color_mapping, ax=ax,
                              flierprops={'alpha': 0.3, 'markersize': 4})
        
        ax.set_title(label, fontweight='bold', fontsize=14)
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel(label.split('(')[0].strip(), fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Make x-axis tick labels larger and bold
        ax.tick_params(axis='x', labelsize=14)
        for label in ax.get_xticklabels():
            label.set_fontweight('bold')
        
        # Set y-axis limits
        if metric == 'rmse':
            ax.set_ylim(0, 0.045)
        elif metric == 'nse':
            ax.set_ylim(-.75, 1)
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # Customize legend with larger text
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, title='Filter-Temporal', loc='upper right', 
                     fontsize=14, title_fontsize=16)
        else:
            ax.get_legend().remove()
    
    plt.tight_layout()
    
    # Save plot
    filename = 'storage_error_combined_boxplots.png'
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved combined boxplot to {filepath}")

def create_inverted_storage_boxplots(df, output_dir):
    """Create inverted 2-panel plot: x-axis = filter-temporal combo, colors = models"""
    
    # Toggle to include/exclude S2 models (same as original plot)
    include_s2 = False  # Set to True to include S2 models in plots
    
    # Prepare data for plotting (same as before)
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
        'SWOTS2': '#984ea3',    # Orange  
        'S2': '#e41a1c',        # Green (kept for when S2 is toggled on)
        'STATIC': '#4daf4a'     # Red
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
    fig.suptitle('Storage Anomaly Error Metrics by Filter-Temporal Combination and Model', 
                fontsize=16, fontweight='bold')
    
    metrics = ['rmse', 'nse']
    metric_labels = ['Root Mean Square Error (km³)', 'Nash-Sutcliffe Efficiency']
    
    # Define desired order for filter-temporal combinations
    filter_temporal_order = ['Optimal\nDiscrete', 'Functional\nDiscrete', 'Optimal\nContinuous', 'Functional\nContinuous']
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[i]
        
        # Create grouped boxplot with models as colors, filter-temporal as x-axis
        box_plot = sns.boxplot(data=plot_df, x='filter_temporal', y=metric, hue='model', 
                              palette=model_color_mapping, ax=ax,
                              flierprops={'alpha': 0.3, 'markersize': 3},
                              order=filter_temporal_order)
        
        # Add grid AFTER creating the boxplot to ensure it appears below
        ax.grid(True, alpha=0.7, zorder=0)
        
        # Add defined boundaries around the plot
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.2)
            spine.set_visible(True)
        
        ax.set_title(label, fontweight='bold', fontsize=14)
        #ax.set_xlabel('Filter-Temporal Combination', fontsize=12)
        ax.set_ylabel(label.split('(')[0].strip(), fontsize=16)
        
        
        # Make x-axis tick labels larger and bold
        ax.tick_params(axis='x', labelsize=14)
        for label in ax.get_xticklabels():
            label.set_fontweight('bold')
        
        # Set y-axis limits (same as original)
        if metric == 'rmse':
            ax.set_ylim(0, 0.025)
        elif metric == 'nse':
            ax.set_ylim(-.25, 1)
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # Customize legend with larger text
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, title='Model', loc='upper right', 
                     fontsize=14, title_fontsize=16)
        else:
            ax.get_legend().remove()
    
    plt.tight_layout(w_pad=3.0)
    
    # Save plot
    filename = 'storage_error_inverted_boxplots.png'
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved inverted boxplot to {filepath}")

def create_all_storage_variant_plots(df, output_dir):
    """Create both original and inverted storage variant plots"""
    
    print("Creating combined storage variant plots...")
    create_combined_storage_boxplots(df, output_dir)
    create_inverted_storage_boxplots(df, output_dir)
    print("Both storage variant plots created!")

def main_variants():
    """Main execution function for storage variants analysis"""
    
    print("Starting benchmark storage variants analysis...")
    
    # Process files and calculate error metrics for all variants
    result = process_storage_variants_data()
    
    if result is not None:
        results_df, summary_df = result
        output_dir = PROJECT_ROOT / "analysis/storage_estimation_assessment/results"
        
        # Create all comparison plots using the per-lake results DataFrame
        create_all_storage_variant_plots(results_df, output_dir)
        
        print("\nBenchmark storage variants analysis completed successfully!")
        print(f"\nSummary: Processed {len(results_df)} model-lake combinations")
        print(f"Models: {sorted(results_df['model'].unique())}")
        print(f"Filter types: {sorted(results_df['filter_type'].unique())}")
        print(f"Temporal types: {sorted(results_df['temporal_type'].unique())}")
    else:
        print("\nNo data was processed successfully.")

def process_benchmark_daily_files_original():
    """Process all benchmark_daily CSV files and calculate storage error metrics (original version)"""
    
    # Define paths
    data_dir = PROJECT_ROOT / "data/benchmark_timeseries"
    output_dir = PROJECT_ROOT / "analysis/storage_estimation_assessment/results"
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all CSV files
    csv_files = list(data_dir.glob("*_daily.csv"))
    print(f"Found {len(csv_files)} CSV files to process")
    
    # Storage for results
    error_metrics_summary = []
    
    for csv_file in csv_files:
        print(f"Processing {csv_file.name}...")
        
        try:
            # Read CSV with string dtypes for ID columns
            df = pd.read_csv(csv_file, dtype={'gage_id': str, 'swot_lake_id': str})
            
            # Skip if no data
            if df.empty:
                continue
            
            # Extract lake ID from filename
            lake_id = csv_file.stem.replace("_daily", "")
            
            # Convert storage anomalies from acre-feet to km³
            storage_cols = ['storage_anomaly', 'swot_storage_anomaly', 'swots2_storage_anomaly', 
                           's2_storage_anomaly', 'static_area_storage_anomaly']
            for col in storage_cols:
                if col in df.columns:
                    df[col] = df[col].apply(acre_feet_to_km3)
            
            # Convert storage column for capacity calculation
            if 'storage' in df.columns:
                df['storage'] = df['storage'].apply(acre_feet_to_km3)
            
            # Apply filtering for different model types
            
            # SWOT-based models filters
            # Optimal filter: using hard-coded threshold
            swot_optimal_filter = ((df['date'] > '2023-07-21') & 
                                  (df['swot_wse_abs_error'] < 0.283))
            df_swot_optimal = df[swot_optimal_filter].copy()
            
            # Adaptive filter: using pre-computed adaptive filter results
            if 'adaptive_filter' in df.columns:
                swot_adaptive_filter = ((df['date'] > '2023-07-21') & 
                                       (df['adaptive_filter'] == 0))  # 0 = accepted
                df_swot_adaptive = df[swot_adaptive_filter].copy()
            else:
                df_swot_adaptive = pd.DataFrame()  # Empty dataframe
            
            # S2 model filter (uses its own quality criteria)
            s2_filter = ((df['s2_coverage'] > 99) & (df['ice'] == 0) & (df['date'] > '2023-07-21'))
            df_s2_filtered = df[s2_filter].copy()
            
            # Calculate capacity (storage range) for normalization
            if 'storage' in df.columns:
                storage_values = df['storage'].dropna()
                if len(storage_values) > 1:
                    capacity = storage_values.max() - storage_values.min()  # storage range in km³
                else:
                    capacity = None
            else:
                capacity = None
            
            # Calculate error metrics using appropriate filters for each model
            
            # SWOT-based models with optimal filter
            if len(df_swot_optimal) > 0:
                swot_reference_optimal = df_swot_optimal['storage_anomaly']
                
                # SWOT vs reference (optimal filter)
                swot_metrics_optimal = calculate_error_metrics(swot_reference_optimal, df_swot_optimal['swot_storage_anomaly'], capacity)
                swot_metrics_optimal['lake_id'] = lake_id
                swot_metrics_optimal['model'] = 'swot'
                swot_metrics_optimal['filter_type'] = 'optimal'
                swot_metrics_optimal['capacity'] = capacity
                
                # SWOT+S2 vs reference (optimal filter)
                swots2_metrics_optimal = calculate_error_metrics(swot_reference_optimal, df_swot_optimal['swots2_storage_anomaly'], capacity)
                swots2_metrics_optimal['lake_id'] = lake_id
                swots2_metrics_optimal['model'] = 'swot_s2'
                swots2_metrics_optimal['filter_type'] = 'optimal'
                swots2_metrics_optimal['capacity'] = capacity
                
                # Static area vs reference (optimal filter)
                static_metrics_optimal = calculate_error_metrics(swot_reference_optimal, df_swot_optimal['static_area_storage_anomaly'], capacity)
                static_metrics_optimal['lake_id'] = lake_id
                static_metrics_optimal['model'] = 'static_area'
                static_metrics_optimal['filter_type'] = 'optimal'
                static_metrics_optimal['capacity'] = capacity
                
                error_metrics_summary.extend([swot_metrics_optimal, swots2_metrics_optimal, static_metrics_optimal])
            
            # SWOT-based models with adaptive filter
            if len(df_swot_adaptive) > 0:
                swot_reference_adaptive = df_swot_adaptive['storage_anomaly']
                
                # SWOT vs reference (adaptive filter)
                swot_metrics_adaptive = calculate_error_metrics(swot_reference_adaptive, df_swot_adaptive['swot_storage_anomaly'], capacity)
                swot_metrics_adaptive['lake_id'] = lake_id
                swot_metrics_adaptive['model'] = 'swot'
                swot_metrics_adaptive['filter_type'] = 'adaptive'
                swot_metrics_adaptive['capacity'] = capacity
                
                # SWOT+S2 vs reference (adaptive filter)
                swots2_metrics_adaptive = calculate_error_metrics(swot_reference_adaptive, df_swot_adaptive['swots2_storage_anomaly'], capacity)
                swots2_metrics_adaptive['lake_id'] = lake_id
                swots2_metrics_adaptive['model'] = 'swot_s2'
                swots2_metrics_adaptive['filter_type'] = 'adaptive'
                swots2_metrics_adaptive['capacity'] = capacity
                
                # Static area vs reference (adaptive filter)
                static_metrics_adaptive = calculate_error_metrics(swot_reference_adaptive, df_swot_adaptive['static_area_storage_anomaly'], capacity)
                static_metrics_adaptive['lake_id'] = lake_id
                static_metrics_adaptive['model'] = 'static_area'
                static_metrics_adaptive['filter_type'] = 'adaptive'
                static_metrics_adaptive['capacity'] = capacity
                
                error_metrics_summary.extend([swot_metrics_adaptive, swots2_metrics_adaptive, static_metrics_adaptive])
            
            # S2 model (use S2 filter)
            if len(df_s2_filtered) > 0:
                s2_reference = df_s2_filtered['storage_anomaly']
                
                s2_metrics = calculate_error_metrics(s2_reference, df_s2_filtered['s2_storage_anomaly'], capacity)
                s2_metrics['lake_id'] = lake_id
                s2_metrics['model'] = 's2'
                s2_metrics['filter_type'] = 's2'
                s2_metrics['capacity'] = capacity
                
                error_metrics_summary.append(s2_metrics)
            
        except Exception as e:
            print(f"Error processing {csv_file.name}: {e}")
            continue
    
    # Convert to DataFrame
    error_metrics_df = pd.DataFrame(error_metrics_summary)
    
    # Filter to common lakes and prepare final datasets
    print(f"\n=== FILTERING TO COMMON LAKE SET BY FILTER TYPE ===")
    
    # Create separate datasets for optimal and adaptive filter types
    optimal_metrics = error_metrics_df[error_metrics_df['filter_type'] == 'optimal'].copy()
    adaptive_metrics = error_metrics_df[error_metrics_df['filter_type'] == 'adaptive'].copy() 
    s2_metrics = error_metrics_df[error_metrics_df['filter_type'] == 's2'].copy()
    
    print(f"Optimal filter records: {len(optimal_metrics)}")
    print(f"Adaptive filter records: {len(adaptive_metrics)}")
    print(f"S2 filter records: {len(s2_metrics)}")
    
    # For optimal subset: Find lakes with all 4 models
    if len(optimal_metrics) > 0:
        optimal_lake_model_counts = optimal_metrics.groupby('lake_id')['model'].count()
        lakes_with_all_optimal_models = optimal_lake_model_counts[optimal_lake_model_counts == 3].index.tolist()  # 3 SWOT-based models
        
        # Add S2 lakes that also exist in optimal set
        s2_lake_ids = set(s2_metrics['lake_id'].unique())
        final_optimal_lakes = [lake for lake in lakes_with_all_optimal_models if lake in s2_lake_ids]
        
        # Create final optimal dataset
        optimal_final = optimal_metrics[optimal_metrics['lake_id'].isin(final_optimal_lakes)].copy()
        s2_for_optimal = s2_metrics[s2_metrics['lake_id'].isin(final_optimal_lakes)].copy()
        error_metrics_optimal_final = pd.concat([optimal_final, s2_for_optimal], ignore_index=True)
        error_metrics_optimal_final = error_metrics_optimal_final[error_metrics_optimal_final['count'] > 0]
    else:
        error_metrics_optimal_final = pd.DataFrame()
        final_optimal_lakes = []
    
    # For adaptive subset: Find lakes with all 4 models
    if len(adaptive_metrics) > 0:
        adaptive_lake_model_counts = adaptive_metrics.groupby('lake_id')['model'].count()
        lakes_with_all_adaptive_models = adaptive_lake_model_counts[adaptive_lake_model_counts == 3].index.tolist()  # 3 SWOT-based models
        
        # Add S2 lakes that also exist in adaptive set 
        final_adaptive_lakes = [lake for lake in lakes_with_all_adaptive_models if lake in s2_lake_ids]
        
        # Create final adaptive dataset
        adaptive_final = adaptive_metrics[adaptive_metrics['lake_id'].isin(final_adaptive_lakes)].copy()
        s2_for_adaptive = s2_metrics[s2_metrics['lake_id'].isin(final_adaptive_lakes)].copy()
        error_metrics_adaptive_final = pd.concat([adaptive_final, s2_for_adaptive], ignore_index=True)
        error_metrics_adaptive_final = error_metrics_adaptive_final[error_metrics_adaptive_final['count'] > 0]
    else:
        error_metrics_adaptive_final = pd.DataFrame()
        final_adaptive_lakes = []
    
    print(f"Lakes with all 4 models (optimal): {len(final_optimal_lakes)}")
    print(f"Lakes with all 4 models (adaptive): {len(final_adaptive_lakes)}")
    
    # Combine both datasets for plotting
    if len(error_metrics_optimal_final) > 0 and len(error_metrics_adaptive_final) > 0:
        combined_metrics = pd.concat([error_metrics_optimal_final, error_metrics_adaptive_final], ignore_index=True)
    elif len(error_metrics_optimal_final) > 0:
        combined_metrics = error_metrics_optimal_final
    elif len(error_metrics_adaptive_final) > 0:
        combined_metrics = error_metrics_adaptive_final
    else:
        combined_metrics = pd.DataFrame()
        print("Warning: No valid data found for plotting")
    
    return combined_metrics

def prepare_plot_data(df):
    """Prepare data for plotting"""
    
    # Map model names for better display
    model_mapping = {
        'swot': 'SWOT',
        'swot_s2': 'SWOT+S2', 
        's2': 'S2',
        'static_area': 'Static Area'
    }
    df['Model'] = df['model'].map(model_mapping)
    
    # Map filter types for better display
    filter_mapping = {
        'optimal': 'Optimal',
        'adaptive': 'Adaptive',
        's2': 'S2'  # S2 uses its own filtering criteria
    }
    df['Filter_Type'] = df['filter_type'].map(filter_mapping)
    
    # Error metrics are already in km³, no conversion needed
    df['mae_km3'] = df['mae']
    df['rmse_km3'] = df['rmse']
    df['bias_km3'] = df['bias']
    df['std_error_km3'] = df['std_error']
    df['p68_abs_error_km3'] = df['p68_abs_error']
    df['pearson_r'] = df['pearson_r']**2  # Convert to R²
    
    return df

def create_error_metrics_boxplots(df, output_dir):
    """Create grouped boxplots comparing Optimal vs Adaptive filters by model"""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("Set2")
    
    # Create figure with subplots for absolute metrics
    fig, axes = plt.subplots(3, 3, figsize=(24, 18))
    fig.suptitle('Storage Anomaly Error Metrics: Optimal vs Adaptive Filtering by Model\n(Benchmark Daily Data)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Define metrics to plot
    metrics_config = [
        {'col': 'mae_km3', 'title': 'Mean Absolute Error', 'ylabel': 'MAE (km³)', 'pos': (0, 0)},
        {'col': 'rmse_km3', 'title': 'Root Mean Square Error', 'ylabel': 'RMSE (km³)', 'pos': (0, 1)},
        {'col': 'bias_km3', 'title': 'Bias (Mean Error)', 'ylabel': 'Bias (km³)', 'pos': (0, 2)},
        {'col': 'std_error_km3', 'title': 'Standard Deviation of Errors', 'ylabel': 'Std Error (km³)', 'pos': (1, 0)},
        {'col': 'p68_abs_error_km3', 'title': '68th Percentile Absolute Error', 'ylabel': 'P68 Abs Error (km³)', 'pos': (1, 1)},
        {'col': 'pearson_r', 'title': 'Pearson Correlation (R²)', 'ylabel': 'R²', 'pos': (1, 2)},
        {'col': 'nse', 'title': 'Nash-Sutcliffe Efficiency', 'ylabel': 'NSE', 'pos': (2, 0)},
        {'col': 'count', 'title': 'Sample Size', 'ylabel': 'Number of Observations', 'pos': (2, 1)}
    ]
    
    # Include all models: SWOT-based models with Optimal/Adaptive filters + S2 model with S2 filter
    swot_models = ['SWOT', 'SWOT+S2', 'Static Area']
    df_swot_filtered = df[df['Model'].isin(swot_models) & df['Filter_Type'].isin(['Optimal', 'Adaptive'])].copy()
    df_s2 = df[df['Model'] == 'S2'].copy()  # S2 uses its own filter type
    
    # Combine SWOT-based models and S2 model
    df_plot = pd.concat([df_swot_filtered, df_s2], ignore_index=True)
    
    print(f"Plotting data: {len(df_plot)} total records")
    print(f"Filter types: {df_plot['Filter_Type'].value_counts()}")
    print(f"Models: {df_plot['Model'].value_counts()}")
    
    # Create boxplots for each metric
    for metric in metrics_config:
        row, col = metric['pos']
        ax = axes[row, col]

        # Create grouped boxplot: Model on x-axis, Filter_Type as hue
        sns.boxplot(data=df_plot, x='Model', y=metric['col'], hue='Filter_Type', ax=ax)
        
        # Customize plot
        ax.set_title(metric['title'], fontsize=12, fontweight='bold')
        ax.set_ylabel(metric['ylabel'], fontsize=10)
        ax.set_xlabel('Model', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(title='Filter Type', fontsize=9, title_fontsize=10)
        
        # Set readable y-axis limits for specific metrics
        if metric['col'] == 'pearson_r':
            ax.set_ylim(0, 1)
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        elif metric['col'] == 'nse':
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
            ax.set_ylim(-1, 1)
        elif metric['col'] in ['mae_km3', 'rmse_km3', 'std_error_km3', 'p68_abs_error_km3']:
            # Set readable range for error metrics (0-10,000 in current units)
            ax.set_ylim(0, .1)
        elif metric['col'] == 'bias_km3':
            # Bias can be negative, so center around 0 with symmetric range  
            ax.set_ylim(-.1, .1)
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        else:
            # For other metrics (like count), use automatic scaling
            pass
        
        # Add sample sizes as text (after setting y-limits)
        # Count by Model and Filter_Type combination
        counts_by_group = df_plot.groupby(['Model', 'Filter_Type'])[metric['col']].count().unstack(fill_value=0)
        
        # Get all models in order they appear in the plot
        all_models = ['SWOT', 'SWOT+S2', 'Static Area', 'S2']
        
        # Position text labels above each boxplot
        for model_idx, model in enumerate(all_models):
            if model in counts_by_group.index:
                # Get counts for different filter types for this model
                optimal_count = counts_by_group.loc[model, 'Optimal'] if 'Optimal' in counts_by_group.columns else 0
                adaptive_count = counts_by_group.loc[model, 'Adaptive'] if 'Adaptive' in counts_by_group.columns else 0
                s2_count = counts_by_group.loc[model, 'S2'] if 'S2' in counts_by_group.columns else 0
                
                y_pos = ax.get_ylim()[1] * 0.95
                
                # For S2 model, only show S2 filter count (centered)
                if model == 'S2':
                    if s2_count > 0:
                        ax.text(model_idx, y_pos, f'n={s2_count}', 
                               ha='center', va='top', fontsize=8, 
                               bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.7))
                else:
                    # For SWOT-based models, show optimal and adaptive counts
                    # Position text based on seaborn's default box positions
                    num_filters = sum([optimal_count > 0, adaptive_count > 0])
                    if num_filters > 1:
                        box_width = 0.8 / 2  # Total width divided by number of categories
                        optimal_x = model_idx - box_width/2
                        adaptive_x = model_idx + box_width/2
                    else:
                        optimal_x = adaptive_x = model_idx
                    
                    if optimal_count > 0:
                        ax.text(optimal_x, y_pos, f'n={optimal_count}', 
                               ha='center', va='top', fontsize=8, 
                               bbox=dict(boxstyle='round,pad=0.2', facecolor='lightblue', alpha=0.7))
                    
                    if adaptive_count > 0:
                        ax.text(adaptive_x, y_pos, f'n={adaptive_count}', 
                               ha='center', va='top', fontsize=8, 
                               bbox=dict(boxstyle='round,pad=0.2', facecolor='lightgreen', alpha=0.7))
    
    # Remove the unused subplot (bottom right)
    fig.delaxes(axes[2, 2])
    
    plt.tight_layout()
    
    # Save the plot
    output_file = output_dir / "benchmark_storage_error_metrics_boxplots_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved comparison boxplots to {output_file}")
    
    return fig

def create_normalized_metrics_boxplots(df, output_dir):
    """Create grouped boxplots for normalized error metrics"""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("Set2")
    
    # Create figure with subplots for normalized metrics
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Normalized Storage Anomaly Error Metrics: Optimal vs Adaptive Filtering\n(Relative to Storage Range)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Define normalized metrics to plot
    normalized_metrics_config = [
        {'col': 'mae_norm', 'title': 'Normalized MAE', 'ylabel': 'MAE / Storage Range', 'pos': (0, 0)},
        {'col': 'rmse_norm', 'title': 'Normalized RMSE', 'ylabel': 'RMSE / Storage Range', 'pos': (0, 1)},
        {'col': 'bias_norm', 'title': 'Normalized Bias', 'ylabel': 'Bias / Storage Range', 'pos': (0, 2)},
        {'col': 'std_error_norm', 'title': 'Normalized Std Error', 'ylabel': 'Std Error / Storage Range', 'pos': (1, 0)},
        {'col': 'p68_abs_error_norm', 'title': 'Normalized P68 Abs Error', 'ylabel': 'P68 Error / Storage Range', 'pos': (1, 1)},
        {'col': 'capacity', 'title': 'Storage Range (Capacity)', 'ylabel': 'Storage Range (km³)', 'pos': (1, 2)}
    ]
    
    # Include all models: SWOT-based models with Optimal/Adaptive filters + S2 model with S2 filter
    swot_models = ['SWOT', 'SWOT+S2', 'Static Area']
    df_swot_filtered = df[df['Model'].isin(swot_models) & df['Filter_Type'].isin(['Optimal', 'Adaptive'])].copy()
    df_s2 = df[df['Model'] == 'S2'].copy()  # S2 uses its own filter type
    
    # Combine SWOT-based models and S2 model
    df_plot = pd.concat([df_swot_filtered, df_s2], ignore_index=True)
    
    # Create boxplots for each normalized metric
    for metric in normalized_metrics_config:
        row, col = metric['pos']
        ax = axes[row, col]
        
        # Create grouped boxplot: Model on x-axis, Filter_Type as hue
        sns.boxplot(data=df_plot, x='Model', y=metric['col'], hue='Filter_Type', ax=ax)
        
        # Customize plot
        ax.set_title(metric['title'], fontsize=12, fontweight='bold')
        ax.set_ylabel(metric['ylabel'], fontsize=10)
        ax.set_xlabel('Model', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(title='Filter Type', fontsize=9, title_fontsize=10)
        
        # Set readable y-axis limits for normalized metrics
        if metric['col'] == 'bias_norm':
            # Bias can be negative, center around 0
            ax.set_ylim(-2, 2)
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        elif metric['col'] in ['mae_norm', 'rmse_norm', 'std_error_norm', 'p68_abs_error_norm']:
            # Normalized error metrics (ratio to storage range)
            ax.set_ylim(0, 2)
        elif metric['col'] == 'capacity':
            # Storage range in km³ - let this auto-scale but set minimum to 0
            ax.set_ylim(0, None)
        else:
            # Auto-scale for other metrics
            pass
        
        # Add sample sizes as text
        counts_by_group = df_plot.groupby(['Model', 'Filter_Type'])[metric['col']].count().unstack(fill_value=0)
        all_models = ['SWOT', 'SWOT+S2', 'Static Area', 'S2']
        
        for model_idx, model in enumerate(all_models):
            if model in counts_by_group.index:
                optimal_count = counts_by_group.loc[model, 'Optimal'] if 'Optimal' in counts_by_group.columns else 0
                adaptive_count = counts_by_group.loc[model, 'Adaptive'] if 'Adaptive' in counts_by_group.columns else 0
                s2_count = counts_by_group.loc[model, 'S2'] if 'S2' in counts_by_group.columns else 0
                
                y_pos = ax.get_ylim()[1] * 0.95
                
                if model == 'S2':
                    if s2_count > 0:
                        ax.text(model_idx, y_pos, f'n={s2_count}', 
                               ha='center', va='top', fontsize=8, 
                               bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.7))
                else:
                    num_filters = sum([optimal_count > 0, adaptive_count > 0])
                    if num_filters > 1:
                        box_width = 0.8 / 2
                        optimal_x = model_idx - box_width/2
                        adaptive_x = model_idx + box_width/2
                    else:
                        optimal_x = adaptive_x = model_idx
                    
                    if optimal_count > 0:
                        ax.text(optimal_x, y_pos, f'n={optimal_count}', 
                               ha='center', va='top', fontsize=8, 
                               bbox=dict(boxstyle='round,pad=0.2', facecolor='lightblue', alpha=0.7))
                    
                    if adaptive_count > 0:
                        ax.text(adaptive_x, y_pos, f'n={adaptive_count}', 
                               ha='center', va='top', fontsize=8, 
                               bbox=dict(boxstyle='round,pad=0.2', facecolor='lightgreen', alpha=0.7))
    
    plt.tight_layout()
    
    # Save the plot
    output_file = output_dir / "benchmark_storage_normalized_error_metrics_boxplots.png"
    plt.show()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved normalized metrics boxplots to {output_file}")
    
    return fig

def create_summary_statistics_table(df, output_dir):
    """Create and save summary statistics table"""
    
    # Calculate summary statistics for each model and filter type combination
    summary_stats = []
    
    # Group by both Model and Filter_Type
    for (model, filter_type), group_data in df.groupby(['Model', 'Filter_Type']):
        stats = {
            'Model': model,
            'Filter_Type': filter_type,
            'N_Lakes': len(group_data),
            'MAE_Mean': group_data['mae_km3'].mean(),
            'MAE_Median': group_data['mae_km3'].median(),
            'MAE_Std': group_data['mae_km3'].std(),
            'RMSE_Mean': group_data['rmse_km3'].mean(),
            'RMSE_Median': group_data['rmse_km3'].median(), 
            'RMSE_Std': group_data['rmse_km3'].std(),
            'Bias_Mean': group_data['bias_km3'].mean(),
            'Bias_Median': group_data['bias_km3'].median(),
            'Bias_Std': group_data['bias_km3'].std(),
            'StdError_Mean': group_data['std_error_km3'].mean(),
            'StdError_Median': group_data['std_error_km3'].median(),
            'StdError_Std': group_data['std_error_km3'].std(),
            'P68_Mean': group_data['p68_abs_error_km3'].mean(),
            'P68_Median': group_data['p68_abs_error_km3'].median(),
            'P68_Std': group_data['p68_abs_error_km3'].std(),
            'Corr_Mean': group_data['pearson_r'].mean(),
            'Corr_Median': group_data['pearson_r'].median(),
            'Corr_Std': group_data['pearson_r'].std(),
            'NSE_Mean': group_data['nse'].mean(),
            'NSE_Median': group_data['nse'].median(),
            'NSE_Std': group_data['nse'].std(),
            'MAE_Norm_Mean': group_data['mae_norm'].mean(),
            'RMSE_Norm_Mean': group_data['rmse_norm'].mean(),
            'Count_Mean': group_data['count'].mean(),
            'Count_Median': group_data['count'].median(),
            'Count_Std': group_data['count'].std()
        }
        summary_stats.append(stats)
    
    summary_df = pd.DataFrame(summary_stats)
    
    # Round numerical columns
    numeric_cols = summary_df.select_dtypes(include=[np.number]).columns
    summary_df[numeric_cols] = summary_df[numeric_cols].round(3)
    
    # Save summary table
    output_file = output_dir / "benchmark_storage_error_summary_stats.csv"
    summary_df.to_csv(output_file, index=False)
    print(f"Saved summary statistics to {output_file}")
    
    # Print summary to console
    print("\n=== SUMMARY STATISTICS TABLE ===")
    print("\nMean values by model and filter type:")
    display_cols = ['Model', 'Filter_Type', 'N_Lakes', 'MAE_Mean', 'RMSE_Mean', 'Bias_Mean', 'Corr_Mean', 'NSE_Mean']
    print(summary_df[display_cols].to_string(index=False))
    
    return summary_df

def main():
    """Main function to run consolidated storage analysis"""
    
    print("CONSOLIDATED STORAGE ANALYSIS")
    print("=" * 50)
    
    # Define output directory
    output_dir = PROJECT_ROOT / "analysis/storage_estimation_assessment/results"
    
    # Step 1: Process benchmark daily files and calculate error metrics
    print("Processing benchmark daily files...")
    combined_metrics = process_benchmark_daily_files()
    
    if combined_metrics.empty:
        print("No valid data found. Exiting.")
        return
    
    # Step 2: Prepare data for plotting
    print("Preparing data for visualization...")
    plot_data = prepare_plot_data(combined_metrics)
    
    # Step 3: Create visualizations and summary
    print("Creating error metrics comparison boxplots...")
    fig1 = create_error_metrics_boxplots(plot_data, output_dir)
    
    print("Creating normalized metrics boxplots...")
    fig2 = create_normalized_metrics_boxplots(plot_data, output_dir)
    
    print("Creating summary statistics table...")
    summary_df = create_summary_statistics_table(plot_data, output_dir)
    
    print("\n=== ANALYSIS COMPLETE ===")
    print("Generated final outputs:")
    print("1. benchmark_storage_error_summary_stats.csv")
    print("2. benchmark_storage_normalized_error_metrics_boxplots.png")
    print("3. benchmark_storage_error_metrics_boxplots_comparison.png")
    
    # Show plots
    plt.show()

if __name__ == "__main__":
    # Run the new storage variants analysis
    main_variants()
    
    # Uncomment below to run original analysis instead
    # main()