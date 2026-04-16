#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Input Uncertainties Calculation

Calculates uncertainties for WSE and WSA using different data subsets and columns.
Generates summary CSV files with error metrics (MAE, RMSE, 1std_error, p68).

WSE subsets:
- optimal, optimal_no_outliers, optimal_daily, optimal_daily_no_outliers
- filtered, filtered_no_outliers, filtered_daily, filtered_daily_no_outliers

WSA subsets (all reported as percentage errors):
- swot_optimal, swot_optimal_no_outliers
- swot_filt, swot_filt_no_outliers  
- s2, s2_no_outliers, s2_daily, s2_daily_no_outliers
- static_optimal, static_filt

Output files:
- analysis/storage_uncertainty_attribution/results/wse_error_metrics_summary.csv
- analysis/storage_uncertainty_attribution/results/wsa_error_metrics_summary.csv
"""

import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path
from scipy.interpolate import interp1d
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Set working directory
os.chdir(PROJECT_ROOT)

# Create output directory if it doesn't exist
output_dir = Path('analysis/storage_uncertainty_attribution/results')
output_dir.mkdir(parents=True, exist_ok=True)


def calculate_error_metrics(errors):
    """
    Calculate error metrics for a given array of errors.
    
    Parameters:
    -----------
    errors : array-like
        Array of error values
        
    Returns:
    --------
    dict : Dictionary with MAE, RMSE, 1std_error, p68
    """
    errors = np.array(errors)
    errors = errors[~np.isnan(errors)]
    
    if len(errors) == 0:
        return {'MAE': np.nan, 'RMSE': np.nan, '1std_error': np.nan, 'p68': np.nan}
    
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    std_error = np.std(errors)
    p68 = np.percentile(np.abs(errors), 68)
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        '1std_error': std_error,
        'p68': p68
    }


def remove_outliers(df, error_col, percentile=99):
    """
    Remove outliers based on percentile of absolute errors.
    
    Parameters:
    -----------
    df : DataFrame
        Data frame to filter
    error_col : str
        Column name containing errors
    percentile : float
        Percentile threshold for outlier removal
        
    Returns:
    --------
    DataFrame : Filtered dataframe
    """
    # Remove NaN values first
    df_clean = df.dropna(subset=[error_col])
    
    if len(df_clean) == 0:
        return df_clean
    
    abs_errors = np.abs(df_clean[error_col])
    threshold = np.percentile(abs_errors, percentile)
    return df_clean[abs_errors <= threshold]


def interpolate_swot_to_insitu_dates(lake_data_target_filtered, lake_data_all, max_gap_days=60):
    """
    Interpolate SWOT WSE values to in-situ dates with ice-aware filtering and gap limits.
    Adapted from filter_evaluation.py
    
    Args:
        lake_data_target_filtered: Lake data already filtered
        lake_data_all: All lake data (for in-situ dates)
        max_gap_days: Maximum interpolation gap in days
    
    Returns:
        Array of interpolated WSE errors, or None if insufficient data
    """
    # Ice-aware filtering
    ice_available = 'ice' in lake_data_all.columns
    if ice_available:
        lake_data_ice_free = lake_data_all[lake_data_all['ice'] == 0].copy()
        lake_data_target_ice_free = lake_data_target_filtered[lake_data_target_filtered['ice'] == 0].copy()
    else:
        lake_data_ice_free = lake_data_all.copy()
        lake_data_target_ice_free = lake_data_target_filtered.copy()
    
    if len(lake_data_target_ice_free) < 2:
        return None
    
    # Convert dates to datetime
    lake_data_ice_free['date'] = pd.to_datetime(lake_data_ice_free['date'])
    lake_data_target_ice_free['date'] = pd.to_datetime(lake_data_target_ice_free['date'])
    
    # Get SWOT WSE data for interpolation
    target_dates_with_data = lake_data_target_ice_free[['date', 'swot_wse_anomaly']].dropna()
    
    if len(target_dates_with_data) < 2:
        return None
        
    # Get all ice-free dates with in-situ stage anomaly data
    insitu_dates_ice_free = lake_data_ice_free[['date', 'stage_anomaly_swotdates']].dropna()
    
    # Filter out physically unreasonable stage anomalies
    reasonable_stage_mask = (insitu_dates_ice_free['stage_anomaly_swotdates'].abs() <= 50.0)
    insitu_dates_ice_free = insitu_dates_ice_free[reasonable_stage_mask]
    
    if len(insitu_dates_ice_free) < 2:
        return None
        
    # Only interpolate to dates within SWOT observation range that have true values
    swot_date_range = (insitu_dates_ice_free['date'] >= target_dates_with_data['date'].min()) & \
                     (insitu_dates_ice_free['date'] <= target_dates_with_data['date'].max())
    
    dates_to_interpolate = insitu_dates_ice_free[swot_date_range].copy()
    
    if len(dates_to_interpolate) < 2:
        return None
        
    try:
        # Convert dates to numeric for interpolation
        swot_dates_numeric = target_dates_with_data['date'].astype(np.int64)
        interp_dates_numeric = dates_to_interpolate['date'].astype(np.int64)
        
        # Apply gap limitation
        target_dates_sorted = target_dates_with_data.sort_values('date')
        max_gap = pd.Timedelta(days=max_gap_days)
        
        valid_interp_mask = np.ones(len(dates_to_interpolate), dtype=bool)
        
        for i, interp_date in enumerate(dates_to_interpolate['date']):
            before_swot = target_dates_sorted[target_dates_sorted['date'] <= interp_date]['date']
            after_swot = target_dates_sorted[target_dates_sorted['date'] >= interp_date]['date']
            
            if len(before_swot) > 0 and len(after_swot) > 0:
                gap_before = interp_date - before_swot.iloc[-1]
                gap_after = after_swot.iloc[0] - interp_date
                
                if gap_before > max_gap or gap_after > max_gap:
                    valid_interp_mask[i] = False
        
        dates_to_interpolate_filtered = dates_to_interpolate[valid_interp_mask].copy()
        
        if len(dates_to_interpolate_filtered) < 2:
            return None
            
        interp_dates_numeric_filtered = dates_to_interpolate_filtered['date'].astype(np.int64)
        
        # Interpolate SWOT WSE anomaly values  
        anomaly_interp_func = interp1d(swot_dates_numeric, target_dates_with_data['swot_wse_anomaly'], 
                                      kind='linear', bounds_error=False, fill_value=np.nan)
        
        interpolated_swot_anomaly = anomaly_interp_func(interp_dates_numeric_filtered)
        
        # Get corresponding in-situ stage anomalies
        insitu_stage_anomalies = dates_to_interpolate_filtered['stage_anomaly_swotdates'].values
        
        # Calculate interpolated errors: interpolated SWOT anomaly - in-situ stage anomaly
        valid_mask = ~np.isnan(interpolated_swot_anomaly) & ~np.isnan(insitu_stage_anomalies)
        
        if valid_mask.sum() < 2:
            return None
            
        interpolated_errors = interpolated_swot_anomaly[valid_mask] - insitu_stage_anomalies[valid_mask]
        return interpolated_errors
        
    except Exception as e:
        return None


def interpolate_s2_wsa_to_insitu_dates(lake_data_target_filtered, lake_data_all, max_gap_days=30):
    """
    Interpolate S2 WSA values to in-situ dates with ice-aware filtering and gap limits.
    
    Args:
        lake_data_target_filtered: Lake data already filtered for S2 (s2_coverage > 99, ice == 0)
        lake_data_all: All lake data (for in-situ dates)
        max_gap_days: Maximum interpolation gap in days
    
    Returns:
        Array of interpolated S2 WSA percentage errors, or None if insufficient data
    """
    # Ice-aware filtering
    ice_available = 'ice' in lake_data_all.columns
    if ice_available:
        lake_data_ice_free = lake_data_all[lake_data_all['ice'] == 0].copy()
        lake_data_target_ice_free = lake_data_target_filtered[lake_data_target_filtered['ice'] == 0].copy()
    else:
        lake_data_ice_free = lake_data_all.copy()
        lake_data_target_ice_free = lake_data_target_filtered.copy()
    
    if len(lake_data_target_ice_free) < 2:
        return None
    
    # Convert dates to datetime
    lake_data_ice_free['date'] = pd.to_datetime(lake_data_ice_free['date'])
    lake_data_target_ice_free['date'] = pd.to_datetime(lake_data_target_ice_free['date'])
    
    # Get S2 WSA data for interpolation
    target_dates_with_data = lake_data_target_ice_free[['date', 's2_wsa_cor']].dropna()
    
    if len(target_dates_with_data) < 2:
        return None
        
    # Get all ice-free dates with true WSA data
    wsa_dates_ice_free = lake_data_ice_free[['date', 'wsa']].dropna()
    
    # Filter out cases where wsa <= 0 to avoid division by zero
    wsa_dates_ice_free = wsa_dates_ice_free[wsa_dates_ice_free['wsa'] > 0]
    
    if len(wsa_dates_ice_free) < 2:
        return None
        
    # Only interpolate to dates within S2 observation range that have true values
    s2_date_range = (wsa_dates_ice_free['date'] >= target_dates_with_data['date'].min()) & \
                    (wsa_dates_ice_free['date'] <= target_dates_with_data['date'].max())
    
    dates_to_interpolate = wsa_dates_ice_free[s2_date_range].copy()
    
    if len(dates_to_interpolate) < 2:
        return None
        
    try:
        # Convert dates to numeric for interpolation
        s2_dates_numeric = target_dates_with_data['date'].astype(np.int64)
        interp_dates_numeric = dates_to_interpolate['date'].astype(np.int64)
        
        # Apply gap limitation
        target_dates_sorted = target_dates_with_data.sort_values('date')
        max_gap = pd.Timedelta(days=max_gap_days)
        
        valid_interp_mask = np.ones(len(dates_to_interpolate), dtype=bool)
        
        for i, interp_date in enumerate(dates_to_interpolate['date']):
            before_s2 = target_dates_sorted[target_dates_sorted['date'] <= interp_date]['date']
            after_s2 = target_dates_sorted[target_dates_sorted['date'] >= interp_date]['date']
            
            if len(before_s2) > 0 and len(after_s2) > 0:
                gap_before = interp_date - before_s2.iloc[-1]
                gap_after = after_s2.iloc[0] - interp_date
                
                if gap_before > max_gap or gap_after > max_gap:
                    valid_interp_mask[i] = False
        
        dates_to_interpolate_filtered = dates_to_interpolate[valid_interp_mask].copy()
        
        if len(dates_to_interpolate_filtered) < 2:
            return None
            
        interp_dates_numeric_filtered = dates_to_interpolate_filtered['date'].astype(np.int64)
        
        # Interpolate S2 WSA values  
        s2_interp_func = interp1d(s2_dates_numeric, target_dates_with_data['s2_wsa_cor'], 
                                  kind='linear', bounds_error=False, fill_value=np.nan)
        
        interpolated_s2_wsa = s2_interp_func(interp_dates_numeric_filtered)
        
        # Get corresponding true WSA values
        true_wsa_values = dates_to_interpolate_filtered['wsa'].values
        
        # Calculate interpolated percentage errors: (s2_wsa - true_wsa) / true_wsa * 100
        valid_mask = ~np.isnan(interpolated_s2_wsa) & ~np.isnan(true_wsa_values) & (true_wsa_values > 0)
        
        if valid_mask.sum() < 2:
            return None
            
        interpolated_percentage_errors = ((interpolated_s2_wsa[valid_mask] - true_wsa_values[valid_mask]) / true_wsa_values[valid_mask]) * 100
        return interpolated_percentage_errors
        
    except Exception as e:
        return None


def calculate_daily_wse_errors(df_filtered, df):
    """
    Calculate WSE errors using ice-aware interpolation with gap limits.
    
    Args:
        df_filtered: DataFrame with filtered SWOT observations
        df: Full DataFrame with all data
        
    Returns:
        Array of WSE errors from interpolation
    """
    all_wse_errors = []
    
    for lake_id in df_filtered['swot_lake_id'].unique():
        lake_data_filtered = df_filtered[df_filtered['swot_lake_id'] == lake_id].copy()
        lake_data_all = df[df['swot_lake_id'] == lake_id].copy()
        
        interpolated_errors = interpolate_swot_to_insitu_dates(lake_data_filtered, lake_data_all)
        
        if interpolated_errors is not None:
            all_wse_errors.extend(interpolated_errors)
    
    return np.array(all_wse_errors)


def calculate_daily_s2_wsa_errors(df_s2, df):
    """
    Calculate S2 WSA errors using ice-aware interpolation with gap limits.
    
    Args:
        df_s2: DataFrame with S2 observations (s2_coverage > 99, ice == 0, wsa > 0)
        df: Full DataFrame with all data
        
    Returns:
        Array of S2 WSA percentage errors from interpolation
    """
    all_s2_wsa_errors = []
    
    for lake_id in df_s2['swot_lake_id'].unique():
        lake_data_filtered = df_s2[df_s2['swot_lake_id'] == lake_id].copy()
        lake_data_all = df[df['swot_lake_id'] == lake_id].copy()
        
        interpolated_errors = interpolate_s2_wsa_to_insitu_dates(lake_data_filtered, lake_data_all)
        
        if interpolated_errors is not None:
            all_s2_wsa_errors.extend(interpolated_errors)
    
    return np.array(all_s2_wsa_errors)


def main():
    """Main function to calculate all error metrics."""
    
    print("Loading merged data...")
    
    # Load all merged CSV files
    csv_files = glob.glob('data/benchmark_timeseries/*.csv')
    
    if not csv_files:
        print("No merged CSV files found in data/benchmark_timeseries/")
        return
    
    # Read and concatenate all CSV files
    df_list = []
    for file in csv_files:
        df = pd.read_csv(file, dtype={'gage_id': str, 'nwis_gage_id': str})
        df_list.append(df)
    
    df = pd.concat(df_list, ignore_index=True)
    
    # Convert date column
    df['date'] = pd.to_datetime(df['date'])
    
    # Calculate swot_wse_error if not present
    if 'swot_wse_error' not in df.columns:
        df['swot_wse_error'] = df['swot_wse_anomaly'] - df['stage_anomaly_swotdates']
    
    # Calculate swot_wse_abs_error if not present
    if 'swot_wse_abs_error' not in df.columns:
        df['swot_wse_abs_error'] = np.abs(df['swot_wse_error'])
    
    print(f"Loaded {len(df)} observations from {len(csv_files)} files")
    
    # Initialize results dictionaries
    wse_results = {}
    wsa_results = {}
    
    # =====================
    # WSE CALCULATIONS
    # =====================
    print("\nCalculating WSE error metrics...")
    
    # OPTIMAL SUBSET (swot_wse_abs_error < 0.283)
    optimal_mask = df['swot_wse_abs_error'] < 0.283
    df_optimal = df[optimal_mask].copy()
    
    print(f"  Optimal subset: {len(df_optimal)} observations")
    
    # Optimal
    wse_results['optimal'] = calculate_error_metrics(df_optimal['swot_wse_error'])
    
    # Optimal no outliers
    df_optimal_no_outliers = remove_outliers(df_optimal, 'swot_wse_error')
    wse_results['optimal_no_outliers'] = calculate_error_metrics(df_optimal_no_outliers['swot_wse_error'])
    
    # Optimal daily (using interpolation)
    optimal_daily_errors = calculate_daily_wse_errors(df_optimal, df)
    wse_results['optimal_daily'] = calculate_error_metrics(optimal_daily_errors)
    
    # Optimal daily no outliers
    if len(optimal_daily_errors) > 0:
        abs_errors = np.abs(optimal_daily_errors)
        threshold = np.percentile(abs_errors, 99)
        optimal_daily_errors_no_outliers = optimal_daily_errors[abs_errors <= threshold]
        wse_results['optimal_daily_no_outliers'] = calculate_error_metrics(optimal_daily_errors_no_outliers)
    else:
        wse_results['optimal_daily_no_outliers'] = {'MAE': np.nan, 'RMSE': np.nan, '1std_error': np.nan, 'p68': np.nan}
    
    # FILTERED SUBSET (adaptive_filter == 0)
    filtered_mask = df['adaptive_filter'] == 0
    df_filtered = df[filtered_mask].copy()
    
    print(f"  Filtered subset: {len(df_filtered)} observations")
    
    # Filtered
    wse_results['filtered'] = calculate_error_metrics(df_filtered['swot_wse_error'])
    
    # Filtered no outliers
    df_filtered_no_outliers = remove_outliers(df_filtered, 'swot_wse_error')
    wse_results['filtered_no_outliers'] = calculate_error_metrics(df_filtered_no_outliers['swot_wse_error'])
    
    # Filtered daily (using interpolation)
    filtered_daily_errors = calculate_daily_wse_errors(df_filtered, df)
    wse_results['filtered_daily'] = calculate_error_metrics(filtered_daily_errors)
    
    # Filtered daily no outliers
    if len(filtered_daily_errors) > 0:
        abs_errors = np.abs(filtered_daily_errors)
        threshold = np.percentile(abs_errors, 99)
        filtered_daily_errors_no_outliers = filtered_daily_errors[abs_errors <= threshold]
        wse_results['filtered_daily_no_outliers'] = calculate_error_metrics(filtered_daily_errors_no_outliers)
    else:
        wse_results['filtered_daily_no_outliers'] = {'MAE': np.nan, 'RMSE': np.nan, '1std_error': np.nan, 'p68': np.nan}
    
    # =====================
    # WSA CALCULATIONS
    # =====================
    print("\nCalculating WSA error metrics...")
    
    # SWOT WSA - OPTIMAL
    swot_wsa_optimal_mask = optimal_mask & (df['swot_partial_f'] == 0) & (df['wsa'] > 0)
    df_swot_wsa_optimal = df[swot_wsa_optimal_mask].copy()
    # Calculate percentage errors: (swot_wsa - wsa) / wsa * 100
    df_swot_wsa_optimal['wsa_error'] = (df_swot_wsa_optimal['swot_wsa'] - df_swot_wsa_optimal['wsa']) / df_swot_wsa_optimal['wsa'] * 100
    
    print(f"  SWOT optimal subset: {len(df_swot_wsa_optimal)} observations")
    
    # SWOT optimal
    wsa_results['swot_optimal'] = calculate_error_metrics(df_swot_wsa_optimal['wsa_error'])
    
    # No daily calculations for WSA metrics
    
    # SWOT optimal no outliers
    df_swot_wsa_optimal_no_outliers = remove_outliers(df_swot_wsa_optimal, 'wsa_error')
    wsa_results['swot_optimal_no_outliers'] = calculate_error_metrics(df_swot_wsa_optimal_no_outliers['wsa_error'])
    
    
    # SWOT WSA - FILTERED
    swot_wsa_filtered_mask = filtered_mask & (df['swot_partial_f'] == 0) & (df['wsa'] > 0)
    df_swot_wsa_filtered = df[swot_wsa_filtered_mask].copy()
    # Calculate percentage errors: (swot_wsa - wsa) / wsa * 100
    df_swot_wsa_filtered['wsa_error'] = (df_swot_wsa_filtered['swot_wsa'] - df_swot_wsa_filtered['wsa']) / df_swot_wsa_filtered['wsa'] * 100
    
    print(f"  SWOT filtered subset: {len(df_swot_wsa_filtered)} observations")
    
    # SWOT filtered
    wsa_results['swot_filt'] = calculate_error_metrics(df_swot_wsa_filtered['wsa_error'])
    
    # No daily calculations for WSA metrics
    
    # SWOT filtered no outliers
    df_swot_wsa_filtered_no_outliers = remove_outliers(df_swot_wsa_filtered, 'wsa_error')
    wsa_results['swot_filt_no_outliers'] = calculate_error_metrics(df_swot_wsa_filtered_no_outliers['wsa_error'])
    
    
    # S2 WSA
    print("  Calculating S2 metrics...")
    s2_mask = (df['s2_coverage'] > 99) & (df['ice'] == 0) & (df['wsa'] > 0)
    df_s2 = df[s2_mask].copy()
    
    # Check if s2_wsa_cor column exists
    if 's2_wsa_cor' in df_s2.columns:
        # Calculate discrete percentage errors: (s2_wsa_cor - wsa) / wsa * 100
        df_s2['s2_wsa_error'] = (df_s2['s2_wsa_cor'] - df_s2['wsa']) / df_s2['wsa'] * 100
        
        print(f"  S2 subset: {len(df_s2)} observations")
        
        # S2 discrete
        wsa_results['s2'] = calculate_error_metrics(df_s2['s2_wsa_error'])
        
        # S2 discrete no outliers
        df_s2_no_outliers = remove_outliers(df_s2, 's2_wsa_error')
        wsa_results['s2_no_outliers'] = calculate_error_metrics(df_s2_no_outliers['s2_wsa_error'])
        
        # S2 daily (using interpolation)
        s2_daily_errors = calculate_daily_s2_wsa_errors(df_s2, df)
        wsa_results['s2_daily'] = calculate_error_metrics(s2_daily_errors)
        
        # S2 daily no outliers
        if len(s2_daily_errors) > 0:
            abs_errors = np.abs(s2_daily_errors)
            threshold = np.percentile(abs_errors, 99)
            s2_daily_errors_no_outliers = s2_daily_errors[abs_errors <= threshold]
            wsa_results['s2_daily_no_outliers'] = calculate_error_metrics(s2_daily_errors_no_outliers)
        else:
            wsa_results['s2_daily_no_outliers'] = {'MAE': np.nan, 'RMSE': np.nan, '1std_error': np.nan, 'p68': np.nan}
    else:
        print("  Warning: s2_wsa_cor column not found, skipping S2 metrics")
        wsa_results['s2'] = {'MAE': np.nan, 'RMSE': np.nan, '1std_error': np.nan, 'p68': np.nan}
        wsa_results['s2_no_outliers'] = {'MAE': np.nan, 'RMSE': np.nan, '1std_error': np.nan, 'p68': np.nan}
        wsa_results['s2_daily'] = {'MAE': np.nan, 'RMSE': np.nan, '1std_error': np.nan, 'p68': np.nan}
        wsa_results['s2_daily_no_outliers'] = {'MAE': np.nan, 'RMSE': np.nan, '1std_error': np.nan, 'p68': np.nan}
    
    # STATIC WSA
    print("  Calculating static WSA metrics...")
    
    # Static optimal (using swot_p_ref_area instead of swot_wsa)
    df_static_optimal = df[optimal_mask & (df['wsa'] > 0)].copy()
    # Calculate percentage errors: (swot_p_ref_area - wsa) / wsa * 100
    df_static_optimal['static_wsa_error'] = (df_static_optimal['swot_p_ref_area'] - df_static_optimal['wsa']) / df_static_optimal['wsa'] * 100
    wsa_results['static_optimal'] = calculate_error_metrics(df_static_optimal['static_wsa_error'])
    
    # Static filtered
    df_static_filtered = df[filtered_mask & (df['wsa'] > 0)].copy()
    # Calculate percentage errors: (swot_p_ref_area - wsa) / wsa * 100
    df_static_filtered['static_wsa_error'] = (df_static_filtered['swot_p_ref_area'] - df_static_filtered['wsa']) / df_static_filtered['wsa'] * 100
    wsa_results['static_filt'] = calculate_error_metrics(df_static_filtered['static_wsa_error'])
    
    # No daily calculations for WSA metrics
    
    # =====================
    # SAVE RESULTS
    # =====================
    print("\nSaving results...")
    
    # Create WSE summary dataframe
    wse_df = pd.DataFrame(wse_results).T
    wse_df.index.name = 'subset'
    wse_output_path = output_dir / 'wse_error_metrics_summary.csv'
    wse_df.to_csv(wse_output_path)
    print(f"  - WSE metrics saved to {wse_output_path}")
    
    # Create WSA summary dataframe
    wsa_df = pd.DataFrame(wsa_results).T
    wsa_df.index.name = 'subset'
    wsa_output_path = output_dir / 'wsa_error_metrics_summary.csv'
    wsa_df.to_csv(wsa_output_path)
    print(f"  - WSA metrics saved to {wsa_output_path}")
    
    # Note: input_uncertainties.csv will be manually created and populated
    
    # Print summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    
    print("\nWSE Error Metrics (meters):")
    print(wse_df.round(4))
    
    print("\nWSA Error Metrics (percentage):")
    print(wsa_df.round(4))
    
    print("\nProcessing complete!")
    print(f"Results saved to {output_dir}/")


if __name__ == "__main__":
    main()