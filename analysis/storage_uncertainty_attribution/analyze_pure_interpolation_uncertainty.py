#!/usr/bin/env python3
"""
Analyze pure interpolation uncertainty by interpolating gauge data to itself.

This isolates the temporal aliasing effects from SWOT measurement errors by:
1. Using gauge stage anomaly as "perfect truth"
2. Subsampling gauge data at SWOT-filtered observation dates 
3. Interpolating back to all gauge dates
4. Measuring interpolation error as |interpolated_gauge - actual_gauge|

Creates two-panel plot showing pure interpolation error vs temporal distance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from scipy.interpolate import interp1d
from pathlib import Path
import warnings
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
warnings.filterwarnings('ignore')

# Change to project directory
os.chdir(PROJECT_ROOT)

def interpolate_gauge_to_gauge(gauge_dates_filtered, gauge_dates_all, gauge_values_all, filter_name):
    """
    Interpolate subsampled gauge data back to full gauge time series.
    
    Args:
        gauge_dates_filtered: Dates where SWOT observations passed filter
        gauge_dates_all: All gauge observation dates
        gauge_values_all: All gauge stage anomaly values
        filter_name: 'optimal' or 'functional' for debugging
    
    Returns:
        dict with interpolation_errors, temporal_distances, valid_dates
    """
    
    if len(gauge_dates_filtered) < 2:
        return None
    
    # Get gauge values at filtered dates (subsampled "SWOT-like" observations)
    gauge_dates_filtered_pd = pd.to_datetime(gauge_dates_filtered)
    gauge_dates_all_pd = pd.to_datetime(gauge_dates_all) 
    
    # Find gauge values corresponding to filtered dates
    subsampled_values = []
    subsampled_dates = []
    
    for swot_date in gauge_dates_filtered_pd:
        # Find closest gauge observation to this SWOT date (within 1 day tolerance)
        time_diffs = np.abs((gauge_dates_all_pd - swot_date).days if hasattr((gauge_dates_all_pd - swot_date), 'days') else (gauge_dates_all_pd - swot_date) / pd.Timedelta(days=1))
        closest_idx = time_diffs.argmin()
        
        if time_diffs[closest_idx] <= 1:  # Must be within 1 day
            subsampled_dates.append(gauge_dates_all_pd[closest_idx])
            subsampled_values.append(gauge_values_all[closest_idx])
    
    if len(subsampled_values) < 2:
        return None
    
    subsampled_dates = np.array(subsampled_dates)
    subsampled_values = np.array(subsampled_values)
    
    # Remove any NaN values from subsampled data
    valid_subsample_mask = ~np.isnan(subsampled_values)
    if valid_subsample_mask.sum() < 2:
        return None
        
    subsampled_dates = subsampled_dates[valid_subsample_mask]
    subsampled_values = subsampled_values[valid_subsample_mask]
    
    # Only interpolate to dates between first and last subsampled observation
    interpolation_range = ((gauge_dates_all_pd >= subsampled_dates.min()) & 
                          (gauge_dates_all_pd <= subsampled_dates.max()))
    
    dates_to_interpolate = gauge_dates_all_pd[interpolation_range]
    values_to_compare = gauge_values_all[interpolation_range]
    
    # Remove NaN values from target data
    valid_target_mask = ~np.isnan(values_to_compare)
    if valid_target_mask.sum() < 2:
        return None
        
    dates_to_interpolate = dates_to_interpolate[valid_target_mask] 
    values_to_compare = values_to_compare[valid_target_mask]
    
    try:
        # Convert dates to numeric for interpolation
        subsample_dates_numeric = pd.to_datetime(subsampled_dates).values.astype(np.int64)
        interp_dates_numeric = pd.to_datetime(dates_to_interpolate).values.astype(np.int64)
        
        # Create interpolation function
        if len(subsampled_values) == 2:
            # Linear interpolation for exactly 2 points
            interp_func = interp1d(subsample_dates_numeric, subsampled_values, 
                                  kind='linear', bounds_error=False, fill_value='extrapolate')
        else:
            # Linear interpolation for multiple points
            interp_func = interp1d(subsample_dates_numeric, subsampled_values, 
                                  kind='linear', bounds_error=False, fill_value=np.nan)
        
        # Interpolate gauge values
        interpolated_gauge_values = interp_func(interp_dates_numeric)
        
        # Remove NaN values from interpolation
        valid_interp_mask = ~np.isnan(interpolated_gauge_values)
        
        if valid_interp_mask.sum() == 0:
            return None
        
        # Calculate pure interpolation errors
        interpolation_errors = np.abs(interpolated_gauge_values[valid_interp_mask] - 
                                    values_to_compare[valid_interp_mask])
        valid_dates = dates_to_interpolate[valid_interp_mask]
        
        # Calculate temporal distance to nearest subsampled observation
        temporal_distances = []
        subsampled_dates_pd = pd.to_datetime(subsampled_dates)
        
        for interp_date in valid_dates:
            interp_date_pd = pd.to_datetime(interp_date)
            # Calculate distance to all subsampled observations
            time_diff = subsampled_dates_pd - interp_date_pd
            distances_to_subsample = np.abs(time_diff.days if hasattr(time_diff, 'days') else time_diff / pd.Timedelta(days=1))
            # Get minimum distance (nearest subsampled observation)
            min_distance = distances_to_subsample.min()
            temporal_distances.append(min_distance)
        
        temporal_distances = np.array(temporal_distances)
        
        # Filter for reasonable gaps (0-50 days) and reasonable errors (<10m)
        reasonable_mask = ((temporal_distances > 0) & (temporal_distances <= 50) & 
                          (interpolation_errors <= 10.0))
        
        if reasonable_mask.sum() == 0:
            return None
        
        return {
            'interpolation_errors': interpolation_errors[reasonable_mask],
            'temporal_distances': temporal_distances[reasonable_mask],
            'n_total_interpolated': len(interpolation_errors),
            'n_reasonable': reasonable_mask.sum(),
            'n_subsampled': len(subsampled_values)
        }
        
    except Exception as e:
        print(f"    Pure interpolation failed for {filter_name}: {e}")
        return None

def process_all_lakes_pure_interpolation():
    """Process all benchmark lakes for pure interpolation uncertainty analysis."""
    
    # Get all benchmark daily files
    data_dir = Path("data/benchmark_timeseries")
    csv_files = list(data_dir.glob("*_daily.csv"))
    print(f"Found {len(csv_files)} benchmark daily files")
    
    # Storage for results
    results = {
        'optimal': {'errors': [], 'distances': []},
        'functional': {'errors': [], 'distances': []}
    }
    
    processed_count = 0
    skipped_count = 0
    
    for csv_file in csv_files:
        try:
            lake_id = csv_file.stem.replace("_daily", "")
            
            # Read data
            df = pd.read_csv(csv_file, dtype={'gage_id': str, 'swot_lake_id': str})
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            
            # Check for required columns
            required_cols = ['swot_wse_abs_error', 'adaptive_filter', 'stage_anomaly_swotdates', 'ice']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"  Skipping {lake_id}: missing columns {missing_cols}")
                skipped_count += 1
                continue
            
            # Check minimum observations for SWOT quality metrics
            valid_swot_obs = df['swot_wse_abs_error'].notna().sum()
            if valid_swot_obs < 5:
                print(f"  Skipping {lake_id}: only {valid_swot_obs} valid SWOT observations")
                skipped_count += 1
                continue
                
            # Ice-aware filtering for gauge data
            if 'ice' in df.columns:
                df_ice_free = df[df['ice'] == 0].copy()
            else:
                df_ice_free = df.copy()
            
            # Get all gauge data (stage anomaly)
            gauge_data = df_ice_free[['date', 'stage_anomaly_swotdates']].dropna()
            
            # Filter out unreasonable stage anomalies  
            reasonable_stage_mask = (gauge_data['stage_anomaly_swotdates'].abs() <= 50.0)
            gauge_data = gauge_data[reasonable_stage_mask]
            
            if len(gauge_data) < 10:  # Need substantial gauge record
                print(f"  Skipping {lake_id}: insufficient gauge data ({len(gauge_data)} points)")
                skipped_count += 1
                continue
            
            print(f"  Processing {lake_id} ({len(gauge_data)} gauge observations, {valid_swot_obs} SWOT observations)...")
            
            # Process optimal filter: get dates where SWOT passed optimal threshold
            optimal_mask = ((df_ice_free['swot_wse_abs_error'] < 0.283) & 
                          df_ice_free['swot_wse_abs_error'].notna())
            if optimal_mask.sum() >= 2:
                optimal_dates = df_ice_free[optimal_mask]['date'].values
                
                optimal_result = interpolate_gauge_to_gauge(
                    optimal_dates, gauge_data['date'].values, 
                    gauge_data['stage_anomaly_swotdates'].values, 'optimal')
                
                if optimal_result is not None:
                    results['optimal']['errors'].extend(optimal_result['interpolation_errors'])
                    results['optimal']['distances'].extend(optimal_result['temporal_distances'])
                    print(f"    Optimal: {optimal_result['n_reasonable']}/{optimal_result['n_total_interpolated']} reasonable interpolations from {optimal_result['n_subsampled']} subsampled points")
            
            # Process functional filter: get dates where SWOT passed functional threshold
            functional_mask = ((df_ice_free['adaptive_filter'] == 0) & 
                             df_ice_free['adaptive_filter'].notna())
            if functional_mask.sum() >= 2:
                functional_dates = df_ice_free[functional_mask]['date'].values
                
                functional_result = interpolate_gauge_to_gauge(
                    functional_dates, gauge_data['date'].values,
                    gauge_data['stage_anomaly_swotdates'].values, 'functional')
                
                if functional_result is not None:
                    results['functional']['errors'].extend(functional_result['interpolation_errors'])
                    results['functional']['distances'].extend(functional_result['temporal_distances'])
                    print(f"    Functional: {functional_result['n_reasonable']}/{functional_result['n_total_interpolated']} reasonable interpolations from {functional_result['n_subsampled']} subsampled points")
            
            processed_count += 1
            
        except Exception as e:
            print(f"  Error processing {csv_file.name}: {e}")
            skipped_count += 1
            continue
    
    print(f"\nProcessed {processed_count} lakes, skipped {skipped_count} lakes")
    print(f"Optimal filter: {len(results['optimal']['errors'])} interpolation points")
    print(f"Functional filter: {len(results['functional']['errors'])} interpolation points")
    
    return results

def create_pure_interpolation_plot(results, output_dir):
    """Create two-panel plot of pure interpolation error vs temporal distance."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Pure Interpolation Error vs Temporal Distance (Gauge-to-Gauge Analysis)', 
                fontsize=16, fontweight='bold')
    
    colors = {'optimal': '#1f77b4', 'functional': '#ff7f0e'}  # Blue, Orange
    
    for i, (filter_name, ax) in enumerate(zip(['optimal', 'functional'], axes)):
        if len(results[filter_name]['errors']) == 0:
            ax.text(0.5, 0.5, f'No data for {filter_name} filter', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title(f'{filter_name.capitalize()} Filter (Pure Interpolation)')
            continue
        
        errors = np.array(results[filter_name]['errors'])
        distances = np.array(results[filter_name]['distances'])
        
        # Create scatter plot
        ax.scatter(distances, errors, alpha=0.3, s=20, color=colors[filter_name], edgecolors='none')
        
        # Add trend line (bin averages)
        unique_distances = np.unique(distances)
        if len(unique_distances) > 1:
            bin_centers = []
            bin_means = []
            bin_stds = []
            
            for dist in unique_distances:
                if np.sum(distances == dist) >= 3:  # At least 3 points for meaningful average
                    mask = distances == dist
                    bin_centers.append(dist)
                    bin_means.append(np.mean(errors[mask]))
                    bin_stds.append(np.std(errors[mask]))
            
            if len(bin_centers) > 1:
                ax.plot(bin_centers, bin_means, color='red', linewidth=2, 
                       marker='o', markersize=4, label=f'Bin averages (n≥3)')
                
                # Add error bars
                ax.errorbar(bin_centers, bin_means, yerr=bin_stds, 
                          color='red', alpha=0.7, capsize=3, linestyle='none')
        
        # Formatting
        ax.set_xlabel('Temporal Distance to Nearest Subsampled Observation (days)', fontsize=12)
        ax.set_ylabel('Pure Interpolation Error (m)', fontsize=12)
        ax.set_title(f'{filter_name.capitalize()} Filter\n(n={len(errors):,} interpolations)', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max(50, distances.max() + 1))
        
        # Add statistics text
        mean_error = np.mean(errors)
        median_error = np.median(errors)
        std_error = np.std(errors)
        
        stats_text = f'Mean: {mean_error:.3f}m\\nMedian: {median_error:.3f}m\\nStd: {std_error:.3f}m'
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        if len(bin_centers) > 1:
            ax.legend(loc='upper right')
    
    plt.tight_layout()
    
    # Save plot
    output_file = output_dir / "pure_interpolation_uncertainty_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved plot to {output_file}")
    plt.show()
    
    return fig

def calculate_pure_interpolation_uncertainty(results):
    """Calculate pure interpolation uncertainty statistics."""
    
    print("\n" + "="*60)
    print("PURE INTERPOLATION UNCERTAINTY ANALYSIS")
    print("="*60)
    
    for filter_name in ['optimal', 'functional']:
        if len(results[filter_name]['errors']) == 0:
            print(f"\n{filter_name.upper()} FILTER: No data available")
            continue
            
        errors = np.array(results[filter_name]['errors'])
        distances = np.array(results[filter_name]['distances'])
        
        print(f"\n{filter_name.upper()} FILTER (Pure Interpolation):")
        print(f"  Total interpolation points: {len(errors):,}")
        print(f"  Pure interpolation uncertainty (std): {np.std(errors):.4f} m")
        print(f"  Pure interpolation uncertainty (mean): {np.mean(errors):.4f} m")
        print(f"  Pure interpolation uncertainty (median): {np.median(errors):.4f} m")
        print(f"  Pure interpolation uncertainty (95th percentile): {np.percentile(errors, 95):.4f} m")
        print(f"  Temporal distance range: {distances.min():.0f} - {distances.max():.0f} days")
        print(f"  Mean temporal distance: {np.mean(distances):.1f} days")
        
        # Distance-dependent uncertainty - DAILY BINS
        print(f"\n  Distance-dependent uncertainty (daily bins):")
        for day in range(1, 51):  # 1 to 50 days
            mask = (distances >= day - 0.5) & (distances < day + 0.5)
            if mask.sum() > 0:
                bin_std = np.std(errors[mask])
                bin_mean = np.mean(errors[mask])
                print(f"    Day {day:2d}: {bin_mean:.4f} m mean, {bin_std:.4f} m std (n={mask.sum():,})")
    
    # Comparison between filters
    if len(results['optimal']['errors']) > 0 and len(results['functional']['errors']) > 0:
        opt_std = np.std(results['optimal']['errors'])
        func_std = np.std(results['functional']['errors'])
        ratio = func_std / opt_std
        
        print(f"\nCOMPARISON (Pure Interpolation):")
        print(f"  Functional/Optimal uncertainty ratio: {ratio:.2f}")
        print(f"  Functional filter shows {(ratio-1)*100:+.1f}% {'higher' if ratio > 1 else 'lower'} pure interpolation uncertainty")
        
        print(f"\nCONCLUSION:")
        print(f"  These values represent PURE temporal aliasing uncertainty")
        print(f"  Add these to measurement errors (not replace them)")
        print(f"  Total continuous uncertainty = sqrt(measurement² + temporal²)")
    
    return {
        'optimal_std': np.std(results['optimal']['errors']) if len(results['optimal']['errors']) > 0 else np.nan,
        'functional_std': np.std(results['functional']['errors']) if len(results['functional']['errors']) > 0 else np.nan
    }

def main():
    """Main execution function."""
    
    print("PURE INTERPOLATION UNCERTAINTY ANALYSIS")
    print("="*50)
    print("Analyzing pure temporal aliasing by interpolating gauge data to itself")
    print("This isolates interpolation effects from SWOT measurement errors")
    print("Processing all benchmark lakes with ≥5 SWOT observations...")
    
    # Process all lakes
    results = process_all_lakes_pure_interpolation()
    
    if len(results['optimal']['errors']) == 0 and len(results['functional']['errors']) == 0:
        print("ERROR: No pure interpolation data found. Check data availability.")
        return
    
    # Create output directory
    output_dir = Path("experiments/results/temporal_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create plots
    create_pure_interpolation_plot(results, output_dir)
    
    # Calculate pure interpolation uncertainty
    pure_uncertainties = calculate_pure_interpolation_uncertainty(results)
    
    # Save summary results
    summary_df = pd.DataFrame([
        {
            'filter_type': 'optimal',
            'n_interpolations': len(results['optimal']['errors']),
            'pure_interpolation_uncertainty_std': pure_uncertainties['optimal_std'],
            'pure_interpolation_uncertainty_mean': np.mean(results['optimal']['errors']) if len(results['optimal']['errors']) > 0 else np.nan
        },
        {
            'filter_type': 'functional', 
            'n_interpolations': len(results['functional']['errors']),
            'pure_interpolation_uncertainty_std': pure_uncertainties['functional_std'],
            'pure_interpolation_uncertainty_mean': np.mean(results['functional']['errors']) if len(results['functional']['errors']) > 0 else np.nan
        }
    ])
    
    summary_file = output_dir / "pure_interpolation_uncertainty_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"\nSaved summary to {summary_file}")
    
    print(f"\nPure interpolation analysis complete! Results saved to {output_dir}")

if __name__ == "__main__":
    main()