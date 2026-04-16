#!/usr/bin/env python3
"""
SWOT Range Analysis at Multiple Time Scales - Combined Version

Combines original (all windows) with v2 (≥2 SWOT only) approach:
- Heatmap: Uses ALL windows including those with 0/1 SWOT observations
- Boxplots: Triple grouping showing all windows, ≥2 SWOT only, and coverage
- Uses v2 time and WSE bins for consistency

Key principles:
1. Random sampling of time windows (no assumptions about optimal timing)
2. Separate metrics for full daily data vs SWOT-only observations
3. Explicit handling of zero ranges (no interpolation fallbacks)
4. Proper ratio calculations including zeros
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
import random
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import seaborn as sns
warnings.filterwarnings('ignore')


def analyze_lake_ranges(lake_data, lake_id, n_samples_per_window=10):
    """
    Analyze storage and WSE ranges for one lake across multiple time windows.
    Same logic as original version (includes all windows).
    """
    
    # Check required columns
    required_cols = ['date', 'storage_anomaly_opt', 'stage_anomaly_swotdates', 
                     'swot_opt_dis', 'swot_wse_anomaly']
    missing_cols = [col for col in required_cols if col not in lake_data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in {lake_id}: {missing_cols}")
    
    # Prepare data
    lake_subset = lake_data.copy()
    lake_subset['date'] = pd.to_datetime(lake_subset['date'])
    lake_subset = lake_subset.sort_values('date').reset_index(drop=True)
    
    # Check minimum data requirements
    if len(lake_subset) < 30:
        raise ValueError(f"Insufficient data for {lake_id}: only {len(lake_subset)} points")
    
    # Check for valid benchmark storage data
    n_valid_storage = lake_subset['storage_anomaly_opt'].notna().sum()
    if n_valid_storage < 30:
        raise ValueError(f"Insufficient benchmark storage data for {lake_id}: only {n_valid_storage} valid points")
    
    # Check for valid WSE data
    n_valid_wse = lake_subset['stage_anomaly_swotdates'].notna().sum()
    if n_valid_wse < 30:
        raise ValueError(f"Insufficient benchmark WSE data for {lake_id}: only {n_valid_wse} valid points")
    
    # Remove outliers from stage_anomaly_swotdates (only negative anomalies exceeding 3*IQR)
    stage_values = lake_subset['stage_anomaly_swotdates'].dropna()
    if len(stage_values) > 0:
        q1 = stage_values.quantile(0.25)
        q3 = stage_values.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 3 * iqr
        
        # Only remove negative outliers that are too extreme
        outlier_mask = lake_subset['stage_anomaly_swotdates'] < lower_bound
        n_outliers = outlier_mask.sum()
        if n_outliers > 0:
            lake_subset.loc[outlier_mask, 'stage_anomaly_swotdates'] = np.nan
            print(f"  Removed {n_outliers} negative WSE outliers from {lake_id}")
    
    # Time windows to analyze (from swot_variability_capture.py)
    time_windows = [3, 7, 14, 21, 30, 45, 60, 90, 120, 180, 270, 365, 455, 545, 635, 725]
    
    # Results storage
    results = []
    
    # Get date range for sampling
    start_date = lake_subset['date'].min()
    end_date = lake_subset['date'].max()
    total_days = (end_date - start_date).days
    
    for window_days in time_windows:
        
        # Skip if window is longer than available data
        if window_days > total_days:
            continue
        
        # Storage for this window's results
        window_storage_ranges = []
        window_wse_ranges = []
        window_n_swot_obs = []
        window_swot_wse_ranges = []
        window_swot_storage_ranges = []
        
        # Generate random start dates for this window size
        max_start_day = total_days - window_days
        if max_start_day <= 0:
            continue
            
        # Sample random windows
        for _ in range(n_samples_per_window):
            
            # Random start day offset
            start_day_offset = random.randint(0, max_start_day)
            window_start = start_date + pd.Timedelta(days=start_day_offset)
            window_end = window_start + pd.Timedelta(days=window_days)
            
            # Get data in this window (all daily observations)
            window_mask = (lake_subset['date'] >= window_start) & (lake_subset['date'] < window_end)
            window_data = lake_subset[window_mask]
            
            if len(window_data) == 0:
                continue
            
            # Calculate storage range (from all daily data)
            storage_values = window_data['storage_anomaly_opt'].dropna()
            if len(storage_values) >= 2:
                storage_range = storage_values.max() - storage_values.min()
            else:
                storage_range = 0.0
            window_storage_ranges.append(storage_range)
            
            # Calculate WSE range (from all daily data)
            wse_values = window_data['stage_anomaly_swotdates'].dropna()
            if len(wse_values) >= 2:
                wse_range = wse_values.max() - wse_values.min()
            else:
                wse_range = 0.0
            window_wse_ranges.append(wse_range)
            
            # Get SWOT-only data in this window
            swot_mask = window_data['swot_opt_dis'].notna()
            swot_window_data = window_data[swot_mask]
            
            # Number of SWOT observations
            n_swot = len(swot_window_data)
            window_n_swot_obs.append(n_swot)
            
            # SWOT WSE range
            if n_swot >= 2:
                swot_wse_values = swot_window_data['swot_wse_anomaly'].dropna()
                if len(swot_wse_values) >= 2:
                    swot_wse_range = swot_wse_values.max() - swot_wse_values.min()
                else:
                    swot_wse_range = 0.0
            else:
                swot_wse_range = 0.0
            window_swot_wse_ranges.append(swot_wse_range)
            
            # SWOT storage range
            if n_swot >= 2:
                swot_storage_values = swot_window_data['swot_opt_dis'].dropna()
                if len(swot_storage_values) >= 2:
                    swot_storage_range = swot_storage_values.max() - swot_storage_values.min()
                else:
                    swot_storage_range = 0.0
            else:
                swot_storage_range = 0.0
            window_swot_storage_ranges.append(swot_storage_range)
        
        # Skip if no valid windows
        if len(window_storage_ranges) == 0:
            continue
        
        # Calculate summary statistics for this time window
        storage_ranges = np.array(window_storage_ranges)
        wse_ranges = np.array(window_wse_ranges)
        n_swot_obs = np.array(window_n_swot_obs)
        swot_wse_ranges = np.array(window_swot_wse_ranges)
        swot_storage_ranges = np.array(window_swot_storage_ranges)
        
        # Calculate percent observed ratios (including zeros) - ORIGINAL METHOD
        # Storage percent observed
        storage_percent_observed = np.zeros(len(storage_ranges))
        for i in range(len(storage_ranges)):
            if storage_ranges[i] > 0:
                storage_percent_observed[i] = (swot_storage_ranges[i] / storage_ranges[i]) * 100
            else:
                storage_percent_observed[i] = 0.0
        
        # WSE percent observed  
        wse_percent_observed = np.zeros(len(wse_ranges))
        for i in range(len(wse_ranges)):
            if wse_ranges[i] > 0:
                wse_percent_observed[i] = (swot_wse_ranges[i] / wse_ranges[i]) * 100
            else:
                wse_percent_observed[i] = 0.0
        
        # Calculate V2 METHOD (only windows with ≥2 SWOT observations)
        storage_percent_observed_v2 = []
        wse_percent_observed_v2 = []
        for i in range(len(storage_ranges)):
            if n_swot_obs[i] >= 2:
                if storage_ranges[i] > 0:
                    storage_percent_observed_v2.append((swot_storage_ranges[i] / storage_ranges[i]) * 100)
                # Note: we don't append WSE here as the original v2 focused on storage
        
        # Count valid windows for v2 method
        n_windows_with_valid_percent = len(storage_percent_observed_v2)
        
        # Coverage statistics
        pct_windows_with_2plus_swot = (n_swot_obs >= 2).mean() * 100
        pct_windows_with_0_swot = (n_swot_obs == 0).mean() * 100
        pct_windows_with_1_swot = (n_swot_obs == 1).mean() * 100
        
        # Store results for this time window
        results.append({
            'lake_id': lake_id,
            'time_window_days': window_days,
            'n_windows_sampled': len(storage_ranges),
            'n_windows_with_valid_percent': n_windows_with_valid_percent,
            
            # Storage range statistics
            'storage_range_mean': storage_ranges.mean(),
            'storage_range_median': np.median(storage_ranges),
            'storage_range_std': storage_ranges.std(),
            'storage_range_min': storage_ranges.min(),
            'storage_range_max': storage_ranges.max(),
            
            # WSE range statistics
            'wse_range_mean': wse_ranges.mean(),
            'wse_range_median': np.median(wse_ranges),
            'wse_range_std': wse_ranges.std(),
            'wse_range_min': wse_ranges.min(),
            'wse_range_max': wse_ranges.max(),
            
            # SWOT observation statistics
            'n_swot_obs_mean': n_swot_obs.mean(),
            'n_swot_obs_median': np.median(n_swot_obs),
            'n_swot_obs_std': n_swot_obs.std(),
            'n_swot_obs_min': n_swot_obs.min(),
            'n_swot_obs_max': n_swot_obs.max(),
            
            # SWOT WSE range statistics
            'swot_wse_range_mean': swot_wse_ranges.mean(),
            'swot_wse_range_median': np.median(swot_wse_ranges),
            'swot_wse_range_std': swot_wse_ranges.std(),
            'swot_wse_range_min': swot_wse_ranges.min(),
            'swot_wse_range_max': swot_wse_ranges.max(),
            
            # SWOT storage range statistics
            'swot_storage_range_mean': swot_storage_ranges.mean(),
            'swot_storage_range_median': np.median(swot_storage_ranges),
            'swot_storage_range_std': swot_storage_ranges.std(),
            'swot_storage_range_min': swot_storage_ranges.min(),
            'swot_storage_range_max': swot_storage_ranges.max(),
            
            # Coverage statistics
            'pct_windows_with_2plus_swot': pct_windows_with_2plus_swot,
            'pct_windows_with_0_swot': pct_windows_with_0_swot,
            'pct_windows_with_1_swot': pct_windows_with_1_swot,
            
            # Percent observed statistics - ORIGINAL (all windows)
            'percent_storage_observed_mean': storage_percent_observed.mean(),
            'percent_storage_observed_median': np.median(storage_percent_observed),
            'percent_storage_observed_std': storage_percent_observed.std(),
            
            'percent_wse_observed_mean': wse_percent_observed.mean(),
            'percent_wse_observed_median': np.median(wse_percent_observed),
            'percent_wse_observed_std': wse_percent_observed.std(),
            
            # Percent observed statistics - V2 (≥2 SWOT only)
            'percent_storage_observed_mean_v2': np.mean(storage_percent_observed_v2) if storage_percent_observed_v2 else np.nan,
            'percent_storage_observed_median_v2': np.median(storage_percent_observed_v2) if storage_percent_observed_v2 else np.nan,
            'percent_storage_observed_std_v2': np.std(storage_percent_observed_v2) if storage_percent_observed_v2 else np.nan,
            
            # Total ranges (for aggregation)
            'total_storage_range': storage_ranges.sum(),
            'total_swot_storage_range': swot_storage_ranges.sum(),
            'total_wse_range': wse_ranges.sum(),
            'total_swot_wse_range': swot_wse_ranges.sum()
        })
    
    return results


def analyze_all_lakes(daily_dir, n_samples_per_window=10, max_lakes=None):
    """Analyze range capture for all lakes."""
    
    csv_files = list(daily_dir.glob("*_daily.csv"))
    if max_lakes is not None:
        csv_files = csv_files[:max_lakes]
    
    print(f"Processing {len(csv_files)} lake files...")
    
    all_results = []
    failed_lakes = []
    processed_count = 0
    
    for csv_file in csv_files:
        processed_count += 1
        lake_id = csv_file.stem.split('_')[0]
        
        try:
            print(f"Processing lake {processed_count}/{len(csv_files)}: {lake_id}...", end='', flush=True)
            
            # Load lake data with proper dtypes
            lake_data = pd.read_csv(csv_file, dtype={'swot_lake_id': str})
            
            # Analyze ranges for this lake
            lake_results = analyze_lake_ranges(lake_data, lake_id, n_samples_per_window)
            all_results.extend(lake_results)
            print(" ✓")
            
        except (ValueError, KeyError) as e:
            print(f" ✗ ({str(e)[:50]}...)")
            failed_lakes.append((lake_id, str(e)))
            continue
        except Exception as e:
            print(f"\nERROR processing {lake_id}: {e}")
            raise
    
    print(f"\nProcessed {processed_count} lakes")
    print(f"Failed: {len(failed_lakes)} lakes")
    print(f"Total results: {len(all_results)} lake-window combinations")
    
    return all_results, failed_lakes


def create_summary_statistics(detailed_results_df):
    """Create summary statistics aggregated by time window."""
    
    time_windows = [3, 7, 14, 21, 30, 45, 60, 90, 120, 180, 270, 365, 455, 545, 635, 725]
    
    summary_results = []
    
    for window_days in time_windows:
        window_data = detailed_results_df[detailed_results_df['time_window_days'] == window_days]
        
        if len(window_data) == 0:
            continue
        
        # Calculate total days analyzed
        total_days_analyzed = window_data['n_windows_sampled'].sum() * window_days
        total_windows_analyzed = window_data['n_windows_sampled'].sum()
        
        # Sum all ranges across lakes
        total_storage_range = window_data['total_storage_range'].sum()
        total_swot_storage_range = window_data['total_swot_storage_range'].sum()
        total_wse_range = window_data['total_wse_range'].sum()
        total_swot_wse_range = window_data['total_swot_wse_range'].sum()
        
        # Calculate overall percent observed (ratio of sums)
        overall_percent_storage_observed = (total_swot_storage_range / total_storage_range * 100) if total_storage_range > 0 else 0.0
        overall_percent_wse_observed = (total_swot_wse_range / total_wse_range * 100) if total_wse_range > 0 else 0.0
        
        # Calculate average percent observed (mean of individual percentages)
        # Original method (all windows)
        all_storage_percentages = []
        all_wse_percentages = []
        
        for _, lake_row in window_data.iterrows():
            if not np.isnan(lake_row['percent_storage_observed_mean']):
                n_windows = lake_row['n_windows_sampled']
                all_storage_percentages.extend([lake_row['percent_storage_observed_mean']] * n_windows)
            
            if not np.isnan(lake_row['percent_wse_observed_mean']):
                n_windows = lake_row['n_windows_sampled']
                all_wse_percentages.extend([lake_row['percent_wse_observed_mean']] * n_windows)
        
        avg_percent_storage_observed = np.mean(all_storage_percentages) if all_storage_percentages else np.nan
        avg_percent_wse_observed = np.mean(all_wse_percentages) if all_wse_percentages else np.nan
        
        # V2 method (≥2 SWOT only)
        all_storage_percentages_v2 = []
        
        for _, lake_row in window_data.iterrows():
            if not np.isnan(lake_row['percent_storage_observed_mean_v2']):
                n_valid_windows = lake_row['n_windows_with_valid_percent']
                all_storage_percentages_v2.extend([lake_row['percent_storage_observed_mean_v2']] * n_valid_windows)
        
        avg_percent_storage_observed_v2 = np.mean(all_storage_percentages_v2) if all_storage_percentages_v2 else np.nan
        
        # Count windows with SWOT data
        total_windows_with_swot_data = (window_data['pct_windows_with_2plus_swot'] * window_data['n_windows_sampled'] / 100).sum()
        
        summary_results.append({
            'time_window_days': window_days,
            'total_days_analyzed': total_days_analyzed,
            'total_windows_analyzed': total_windows_analyzed,
            'total_storage_range': total_storage_range,
            'total_swot_storage_range': total_swot_storage_range,
            'total_wse_range': total_wse_range,
            'total_swot_wse_range': total_swot_wse_range,
            'overall_percent_storage_observed': overall_percent_storage_observed,
            'overall_percent_wse_observed': overall_percent_wse_observed,
            'avg_percent_storage_observed': avg_percent_storage_observed,
            'avg_percent_wse_observed': avg_percent_wse_observed,
            'avg_percent_storage_observed_v2': avg_percent_storage_observed_v2,
            'total_windows_with_swot_data': total_windows_with_swot_data,
            'n_lakes_analyzed': len(window_data)
        })
    
    return pd.DataFrame(summary_results)


def create_visualization(summary_df, output_dir):
    """Create visualization of results."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Overall percent observed vs time window (ratio of sums)
    ax1 = axes[0, 0]
    ax1.plot(summary_df['time_window_days'], summary_df['overall_percent_storage_observed'], 
             'bo-', linewidth=2, markersize=6, label='Storage (ratio)')
    ax1.plot(summary_df['time_window_days'], summary_df['overall_percent_wse_observed'], 
             'ro-', linewidth=2, markersize=6, label='WSE (ratio)')
    ax1.plot(summary_df['time_window_days'], summary_df['avg_percent_storage_observed'], 
             'b--', linewidth=2, markersize=4, label='Storage (avg)', alpha=0.7)
    ax1.plot(summary_df['time_window_days'], summary_df['avg_percent_wse_observed'], 
             'r--', linewidth=2, markersize=4, label='WSE (avg)', alpha=0.7)
    ax1.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='100%')
    ax1.axhline(y=50, color='gray', linestyle=':', alpha=0.5, label='50%')
    ax1.set_xlabel('Time Window (days)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('% Range Observed by SWOT', fontsize=14, fontweight='bold')
    #ax1.set_title('SWOT Range Capture vs Time Scale', fontsize=16, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, max(summary_df['time_window_days']) * 1.05)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax1.tick_params(axis='y', rotation=45)
    
    # Plot 2: Total ranges vs time window
    ax2 = axes[0, 1]
    ax2.semilogy(summary_df['time_window_days'], summary_df['total_storage_range'], 
                 'b-', linewidth=2, label='Total Storage Range')
    ax2.semilogy(summary_df['time_window_days'], summary_df['total_swot_storage_range'], 
                 'r-', linewidth=2, label='SWOT Storage Range')
    ax2.set_xlabel('Time Window (days)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Total Range (log scale)', fontsize=14, fontweight='bold')
    ax2.set_title('Total Range Magnitudes', fontsize=16, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    
    # Plot 3: Number of windows analyzed
    ax3 = axes[1, 0]
    ax3.bar(summary_df['time_window_days'], summary_df['total_windows_analyzed'], 
            alpha=0.7, color='green')
    ax3.set_xlabel('Time Window (days)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Total Windows Analyzed', fontsize=14, fontweight='bold')
    ax3.set_title('Sampling Coverage by Time Scale', fontsize=16, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='both', which='major', labelsize=12)
    
    # Plot 4: Windows with SWOT data
    ax4 = axes[1, 1]
    pct_with_swot = (summary_df['total_windows_with_swot_data'] / summary_df['total_windows_analyzed']) * 100
    ax4.plot(summary_df['time_window_days'], pct_with_swot, 'go-', linewidth=2, markersize=6)
    ax4.set_xlabel('Time Window (days)', fontsize=14, fontweight='bold')
    ax4.set_ylabel('% Windows with SWOT Data', fontsize=14, fontweight='bold')
    ax4.set_title('SWOT Data Availability by Time Scale', fontsize=16, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 100)
    ax4.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'swot_range_analysis_multiwindow_combined.png', dpi=300, bbox_inches='tight')
    plt.show()


def analyze_wse_time_window_relationship(detailed_results_df, output_dir):
    """
    Combined analysis with heatmap (all windows) and triple boxplots.
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use ALL data including zeros for heatmap (original method)
    analysis_df_all = detailed_results_df.copy()
    analysis_df_all = analysis_df_all[analysis_df_all['percent_storage_observed_mean'] < 500].copy()
    
    # Use ≥2 SWOT data for v2 boxplots
    analysis_df_v2 = detailed_results_df.copy()
    analysis_df_v2 = analysis_df_v2.dropna(subset=['percent_storage_observed_mean_v2'])
    analysis_df_v2 = analysis_df_v2[analysis_df_v2['percent_storage_observed_mean_v2'] < 500].copy()
    
    print("\nAnalyzing WSE-Time Window Relationship")
    print(f"All windows: {len(analysis_df_all)} lake-window combinations")
    print(f"≥2 SWOT windows: {len(analysis_df_v2)} lake-window combinations")
    
    # Create WSE range bins (v2 style)
    wse_bins = [0, 0.1, 0.2, 0.5, 1.0, 2.0, np.inf]
    wse_labels = ['0-10', '10-20', '20-50', '50-100', '100-200', '>200']
    
    # Time window bins (v2 style)
    time_bins = [0, 7, 14, 30, 60, 90, 180, 365, 1000]
    time_labels = ['7', '14', '30', '60', '90', '180', '365', '>365']
    
    # Create bins for all datasets
    analysis_df_all['wse_range_bin'] = pd.cut(analysis_df_all['wse_range_mean'], 
                                               bins=wse_bins, labels=wse_labels)
    analysis_df_all['time_window_bin'] = pd.cut(analysis_df_all['time_window_days'], 
                                                 bins=time_bins, labels=time_labels)
    
    analysis_df_v2['wse_range_bin'] = pd.cut(analysis_df_v2['wse_range_mean'], 
                                              bins=wse_bins, labels=wse_labels)
    analysis_df_v2['time_window_bin'] = pd.cut(analysis_df_v2['time_window_days'], 
                                                bins=time_bins, labels=time_labels)
    
    detailed_results_df['wse_range_bin'] = pd.cut(detailed_results_df['wse_range_mean'], 
                                                   bins=wse_bins, labels=wse_labels)
    detailed_results_df['time_window_bin'] = pd.cut(detailed_results_df['time_window_days'], 
                                                     bins=time_bins, labels=time_labels)
    
    # Create pivot table for heatmap using ALL windows (keep as percentages)
    heatmap_data = analysis_df_all.pivot_table(
        values='percent_storage_observed_mean',
        index='wse_range_bin',
        columns='time_window_bin',
        aggfunc='mean'
    ).reindex(index=wse_labels, columns=time_labels, fill_value=np.nan)
    
    # Create figure with three panels (6.35 inches total width)
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.17))
    
    # Plot 1: Heatmap of storage ratios (using ALL windows)
    ax1 = axes[0]
    heatmap_data_reversed = heatmap_data.iloc[::-1]
    # Create custom annotations with percentage signs
    annot_data = heatmap_data_reversed.applymap(lambda x: f'{x:.0f}%' if not pd.isna(x) else '')
    
    heatmap = sns.heatmap(heatmap_data_reversed, annot=annot_data, fmt='', cmap='BrBG', 
                vmin=0, vmax=200, center=100, ax=ax1, 
                cbar=False,  # Disable automatic colorbar
                annot_kws={'size': 6, 'weight': 'normal', 'rotation': 0})
    
    # Create manual colorbar positioned to align with bottom of other axes
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("bottom", size="5%", pad=0.05)
    
    # Create the colorbar
    mappable = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=0, vmax=200), cmap='BrBG')
    cbar = plt.colorbar(mappable, cax=cax, orientation='horizontal', pad = 0.5)
    cbar.set_ticks([])
    
    ax1.set_xlabel('')
    ax1.set_ylabel('')
    ax1.set_title('')
    # Set ticks at every column/row position
    ax1.set_xticks(np.arange(len(heatmap_data.columns)) + 0.5)
    ax1.set_yticks(np.arange(len(heatmap_data.index)) + 0.5)
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.tick_params(left=True, bottom=True, length=2, width=1)
    
    # Plot 2: Triple grouped plots by time window
    ax2 = axes[1]
    
    time_window_bins = time_labels
    box_data_all = []
    box_data_v2 = []
    coverage_percentages = []
    positions_all = []
    positions_v2 = []
    bar_positions = []
    labels_time_used = []
    
    for i, tw_bin in enumerate(time_window_bins):
        tw_data_all = analysis_df_all[analysis_df_all['time_window_bin'] == tw_bin]
        tw_data_v2 = analysis_df_v2[analysis_df_v2['time_window_bin'] == tw_bin]
        tw_data_coverage = detailed_results_df[detailed_results_df['time_window_bin'] == tw_bin]
        
        if len(tw_data_all) > 0:
            # All windows (keep as percentages)
            box_data_all.append(tw_data_all['percent_storage_observed_mean'].values)
            positions_all.append(i * 3)
            
            # Calculate overall storage ratio for this time bin (like heatmap data)
            overall_storage_ratio = tw_data_all['percent_storage_observed_mean'].mean()
            coverage_percentages.append(overall_storage_ratio)
            
            labels_time_used.append(tw_bin)
    
    # Plot boxplots for all windows only (wider boxes)
    bp_all = ax2.boxplot([data for data in box_data_all if len(data) > 0], 
                         positions=[pos for i, pos in enumerate(positions_all) if len(box_data_all[i]) > 0], 
                         widths=1.5, patch_artist=True, 
                         boxprops=dict(facecolor='lightblue', alpha=0.7),
                         medianprops=dict(color='black', linewidth=1),
                         flierprops=dict(alpha=0.3))
    
    # Plot coverage as scatter with connected line
    x_positions = [i * 3 for i in range(len(labels_time_used))]
    ax2.plot(x_positions, coverage_percentages, 'o-', color='coral', linewidth=1.25, 
             markersize=6, markerfacecolor='coral', markeredgecolor='darkred', 
             markeredgewidth=1, alpha=0.8)
    
    ax2.set_xticks([i * 3 for i in range(len(labels_time_used))])
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    #ax2.axhline(y=50, color='orange', linestyle='--', alpha=0.5)
    ax2.axhline(y=100, color='black', linestyle='--', alpha=0.35)
    
    ax2.set_xlabel('')
    ax2.set_ylabel('')
    ax2.set_title('')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 200)
    ax2.tick_params(left=True, bottom=True, length=2, width=1)
    
    # Plot 3: Triple grouped plots by WSE range bins
    ax3 = axes[2]
    
    wse_bins_ordered = ['0-10', '10-20', '20-50', '50-100', '100-200', '>200']
    box_data_wse_all = []
    box_data_wse_v2 = []
    coverage_percentages_wse = []
    positions_wse_all = []
    positions_wse_v2 = []
    bar_positions_wse = []
    labels_wse_used = []
    
    for i, wse_bin in enumerate(wse_bins_ordered):
        wse_data_v2 = analysis_df_v2[analysis_df_v2['wse_range_bin'] == wse_bin]
        
        if len(wse_data_v2) > 0:
            # ≥2 SWOT windows only (keep as percentages)
            box_data_wse_v2.append(wse_data_v2['percent_storage_observed_mean_v2'].values)
            positions_wse_v2.append(i * 3)
            
            labels_wse_used.append(wse_bin)
    
    # Plot ≥2 SWOT WSE boxplots only (wider boxes)
    bp_wse_v2 = ax3.boxplot([data for data in box_data_wse_v2 if len(data) > 0], 
                            positions=[pos for i, pos in enumerate(positions_wse_v2) if len(box_data_wse_v2[i]) > 0], 
                            widths=1.5, patch_artist=True, 
                            boxprops=dict(facecolor='steelblue', alpha=0.7),
                            medianprops=dict(color='black', linewidth=1),
                            flierprops=dict(alpha=0.3))
    
    ax3.set_xticks([i * 3 for i in range(len(labels_wse_used))])
    ax3.set_xticklabels([])
    ax3.set_yticklabels([])
    #ax3.axhline(y=50, color='orange', linestyle='--', alpha=0.5)
    ax3.axhline(y=100, color='black', linestyle='--', alpha=0.35)
    
    ax3.set_xlabel('')
    ax3.set_ylabel('')
    ax3.set_title('')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 200)
    ax3.tick_params(left=True, bottom=True, length=2, width=1)
    
    plt.tight_layout(pad=0.5, w_pad=0.5)
    plt.savefig(output_dir / 'storage_change_detection.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*60)
    print("COMBINED ANALYSIS COMPLETE")
    print("="*60)


def main():
    """Main execution function."""
    print("SWOT Range Analysis at Multiple Time Scales - COMBINED VERSION")
    print("=" * 70)
    print("Analyzing storage and WSE range capture across time windows...")
    print("Combines original (all windows) with v2 (≥2 SWOT) methods")
    
    # Configuration
    daily_dir = Path("/Users/ericlevenson/University of Oregon Dropbox/Eric Levenson/SWOT/production/data/timeseries/benchmark_daily")
    output_dir = Path("/Users/ericlevenson/University of Oregon Dropbox/Eric Levenson/SWOT/production/experiments/02_storage_estimation_accuracy/results/changedetection/")
    
    n_samples_per_window = 100  # Random samples per time window per lake
    max_lakes = None  # None = analyze all lakes
    
    if not daily_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {daily_dir}")
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nConfiguration:")
    print(f"  Data directory: {daily_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Samples per window: {n_samples_per_window}")
    print(f"  Max lakes: {'All' if max_lakes is None else max_lakes}")
    
    # Analyze all lakes
    all_results, failed_lakes = analyze_all_lakes(daily_dir, n_samples_per_window, max_lakes)
    
    # Convert to DataFrame
    print("\nCreating results DataFrame...")
    detailed_results_df = pd.DataFrame(all_results)
    
    # Save detailed results
    detailed_file = output_dir / "change_detection_detailed.csv"
    detailed_results_df.to_csv(detailed_file, index=False, float_format='%.3f')
    print(f"Detailed results saved to: {detailed_file}")
    
    # Create summary statistics
    print("\nCreating summary statistics...")
    summary_df = create_summary_statistics(detailed_results_df)
    
    # Save summary results
    summary_file = output_dir / "change_detection_summary.csv"
    summary_df.to_csv(summary_file, index=False, float_format='%.3f')
    print(f"Summary results saved to: {summary_file}")
    
    # Create visualizations
    print("\nCreating basic visualizations...")
    create_visualization(summary_df, output_dir)
    
    # Create detailed analysis
    print("\nCreating WSE-time window analysis...")
    analyze_wse_time_window_relationship(detailed_results_df, output_dir)
    
    # Print key findings
    print("\n" + "="*50)
    print("KEY FINDINGS")
    print("="*50)
    
    print(f"\nSummary across time windows:")
    print(f"{'Window':<8} {'Storage%(All)':<12} {'Storage%(≥2)':<12} {'WSE%(All)':<10} {'N Windows':<10}")
    print("-" * 68)
    
    for _, row in summary_df.iterrows():
        storage_v2 = row.get('avg_percent_storage_observed_v2', np.nan)
        storage_v2_str = f"{storage_v2:.1f}" if not np.isnan(storage_v2) else "N/A"
        print(f"{row['time_window_days']:<8.0f} {row['avg_percent_storage_observed']:<12.1f} "
              f"{storage_v2_str:<12} {row['avg_percent_wse_observed']:<10.1f} "
              f"{row['total_windows_analyzed']:<10.0f}")
    
    print("\nCombined analysis complete!")


if __name__ == "__main__":
    main()