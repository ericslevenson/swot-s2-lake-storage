#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filter Evaluation Analysis

Evaluates different filtering approaches using the centralized filter system.
Based on quality_analysis_wse_error.py but using the new filter framework.

Compares multiple filters:
- SWOT quality flag filter
- Custom standard filter  
- Strict filter
- Relaxed filter
- Adaptive LakeSP filter

Evaluates each filter on:
- Temporal resolution
- Precision/recall for good observations
- Error metrics (MAE, RMSE, NRMSE)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import glob
from datetime import datetime
from sklearn.metrics import f1_score
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar

# Set working directory and import path
import sys
os.chdir('/Users/ericlevenson/Dropbox/science/phd/SWOT/production')
sys.path.append('/Users/ericlevenson/Dropbox/science/phd/SWOT/production')

# Import all filters from centralized system
from src.filter.filters import (
    swot_quality_flag,
    swot_custom_standard, 
    swot_strict,
    swot_relaxed,
    date_range_swot_era
)

# Note: Adaptive filter is now applied using pre-computed 'adaptive_filter' column
# from CSV files. No longer need to import adaptive filter functions.

# CONFIGURATION
TARGET_WSE_STD = 0.1  # meters - target standard deviation for idealized filter threshold optimization
START_DATE = '2023-07-21'  # SWOT era start date

# Define filters to evaluate
FILTERS_TO_EVALUATE = {
    'all_observations': {
        'name': 'All Observations',
        'filter_func': lambda df: pd.Series(True, index=df.index) & date_range_swot_era(df),
        'description': 'No filtering - all observations after date filter'
    },
    'valid_subset': {
        'name': 'Idealized',  # Name updated dynamically with computed threshold
        'filter_func': lambda df: (df['good_observation'] == True) & date_range_swot_era(df),
        'description': 'Observations meeting computed error threshold'  # Updated dynamically
    },
    'quality_flag': {
        'name': 'SWOT Quality Flag',
        'filter_func': swot_quality_flag,
        'description': 'Official SWOT quality flag filter'
    },
    'custom_standard': {
        'name': 'Custom Standard',
        'filter_func': swot_custom_standard,
        'description': 'Standard custom filter (most common)'
    },
    'adaptive': {
        'name': 'Adaptive LakeSP',
        'filter_func': None,  # Uses pre-computed 'adaptive_filter' column
        'description': 'Lake-specific calibrated thresholds (pre-computed)'
    }
}

def find_threshold_for_target_std(df, target_std=0.1, error_col='swot_wse_error'):
    """
    Find the WSE error threshold that achieves a target standard deviation.

    Uses scipy optimization to find the threshold where filtering observations
    by |swot_wse_error| <= threshold produces a distribution with std ≈ target_std.

    Args:
        df: DataFrame with WSE error data
        target_std: Target standard deviation in meters (default 0.1m = 10cm)
        error_col: Column name for signed WSE errors

    Returns:
        dict with keys:
            - threshold: The computed threshold in meters
            - achieved_std: The actual std achieved with this threshold
            - n_observations: Number of observations passing the threshold
            - success: Whether optimization succeeded
            - filtered_data: The filtered DataFrame
    """
    valid_data = df.dropna(subset=[error_col, 'swot_wse_abs_error'])

    if len(valid_data) < 100:
        raise ValueError(f"Insufficient data for threshold optimization: {len(valid_data)} observations (need >= 100)")

    def objective(threshold):
        if threshold <= 0:
            return float('inf')

        filtered_data = valid_data[valid_data['swot_wse_abs_error'] <= threshold]

        if len(filtered_data) < 50:
            return float('inf')

        achieved_std = filtered_data[error_col].std()
        return abs(achieved_std - target_std)

    result = minimize_scalar(objective, bounds=(0.01, 2.0), method='bounded')

    if result.success and result.fun < 0.005:
        threshold = result.x
        filtered_data = valid_data[valid_data['swot_wse_abs_error'] <= threshold]
        achieved_std = filtered_data[error_col].std()

        return {
            'threshold': threshold,
            'achieved_std': achieved_std,
            'n_observations': len(filtered_data),
            'success': True,
            'filtered_data': filtered_data
        }

    raise ValueError(f"WSE threshold optimization failed: could not achieve target_std={target_std}m "
                     f"(best result: fun={result.fun:.4f}, threshold={result.x:.4f}m)")

def interpolate_swot_to_insitu_dates(lake_data_target_filtered, lake_data_all, variable='wse'):
    """
    Interpolate SWOT values to in-situ dates and calculate errors correctly.
    Adapted from experiments/3.1.py
    
    Args:
        lake_data_target_filtered: Lake data already filtered 
        lake_data_all: All lake data (for in-situ dates)
        variable: 'wse' or 'wsa'
    
    Returns:
        dict with interpolated_errors, interpolated_values, valid_dates, target_dates
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
    
    if variable == 'wse':
        # Get SWOT WSE anomaly data for interpolation (from target distribution only)
        target_dates_with_data = lake_data_target_ice_free[['date', 'swot_wse_anomaly']].dropna()
        
        if len(target_dates_with_data) >= 2:
            # Get all ice-free dates with in-situ stage anomaly data
            insitu_dates_ice_free = lake_data_ice_free[['date', 'stage_anomaly_swotdates']].dropna()
            
            # Filter out physically unreasonable stage anomalies
            reasonable_stage_mask = (insitu_dates_ice_free['stage_anomaly_swotdates'].abs() <= 50.0)
            insitu_dates_ice_free = insitu_dates_ice_free[reasonable_stage_mask]
            
            if len(insitu_dates_ice_free) >= 2:
                # Only interpolate to dates between first and last valid SWOT observation
                swot_date_range = (insitu_dates_ice_free['date'] >= target_dates_with_data['date'].min()) & \
                                 (insitu_dates_ice_free['date'] <= target_dates_with_data['date'].max())
                
                dates_to_interpolate = insitu_dates_ice_free[swot_date_range].copy()
                
                if len(dates_to_interpolate) >= 2:
                    try:
                        # Convert dates to numeric for interpolation
                        swot_dates_numeric = target_dates_with_data['date'].astype(np.int64)
                        interp_dates_numeric = dates_to_interpolate['date'].astype(np.int64)
                        
                        # Interpolate SWOT WSE anomaly values
                        anomaly_interp_func = interp1d(swot_dates_numeric, target_dates_with_data['swot_wse_anomaly'], 
                                                     kind='linear', bounds_error=False, fill_value=np.nan)
                        
                        interpolated_swot_anomaly = anomaly_interp_func(interp_dates_numeric)
                        
                        # Remove NaN values and calculate errors properly
                        valid_interp_mask = ~np.isnan(interpolated_swot_anomaly)
                        if valid_interp_mask.sum() >= 2:
                            valid_indices = np.where(valid_interp_mask)[0]
                            insitu_stage_anomalies = dates_to_interpolate.iloc[valid_indices]['stage_anomaly_swotdates'].values
                            
                            # Calculate interpolated errors: interpolated SWOT anomaly - in-situ stage anomaly
                            interpolated_errors = interpolated_swot_anomaly[valid_interp_mask] - insitu_stage_anomalies
                            
                            return {
                                'interpolated_errors': interpolated_errors,
                                'interpolated_values': interpolated_swot_anomaly[valid_interp_mask],
                                'valid_dates': dates_to_interpolate.iloc[valid_indices]['date'].values,
                                'target_dates_with_data': target_dates_with_data
                            }
                    except Exception as e:
                        return None
    
    return None

def calculate_nse_for_filter(df, filter_mask, filter_name, verbose=False):
    """
    Calculate Nash-Sutcliffe Efficiency for each lake using filtered data.
    Adapted from experiments/3.1.py calculate_nse_by_lake function.
    
    Args:
        df: Full dataframe with all lakes
        filter_mask: Boolean mask indicating which observations pass the filter
        filter_name: Name of the filter for reporting
        verbose: Print debug info
    
    Returns:
        dict: NSE metrics including min, mean, median, max NSE, per-lake NSE values
    """
    if verbose:
        print(f"Calculating NSE for {filter_name}...")
    
    nse_results = []
    
    for lake_id in df['swot_lake_id'].unique():
        lake_data = df[df['swot_lake_id'] == lake_id].copy()
        lake_data['date'] = pd.to_datetime(lake_data['date'])
        lake_data = lake_data.sort_values('date')
        
        # Get filtered data for this lake using original indices
        lake_filter_mask = filter_mask.loc[lake_data.index]
        lake_data_filtered = lake_data[lake_filter_mask].copy()
        
        # Now reset index for internal processing
        lake_data = lake_data.reset_index(drop=True)
        lake_data_filtered = lake_data_filtered.reset_index(drop=True)
        
        if len(lake_data_filtered) < 3:  # Need minimum observations for NSE calculation
            # If filter misses a lake, NSE = -999 (very poor)
            nse_results.append({
                'swot_lake_id': lake_id,
                'nse_target_alldates': -999.0,
                'nse_target_swotdates': -999.0,
                'n_observations_filtered': len(lake_data_filtered)
            })
            continue
        
        # Initialize variables
        nse_target_alldates = np.nan  # daily approach
        nse_target_swotdates = np.nan  # synoptic approach
        
        # Ice-aware filtering
        ice_available = 'ice' in lake_data.columns
        if ice_available:
            lake_data_ice_free = lake_data[lake_data['ice'] == 0].copy()
        else:
            lake_data_ice_free = lake_data.copy()
        
        if len(lake_data_ice_free) < 2:
            nse_results.append({
                'swot_lake_id': lake_id,
                'nse_target_alldates': -999.0,
                'nse_target_swotdates': -999.0,
                'n_observations_filtered': len(lake_data_filtered)
            })
            continue
        
        # Calculate NSE for synoptic approach (direct comparison on SWOT dates)
        filtered_ice_free = lake_data_filtered[lake_data_filtered['ice'] == 0] if ice_available else lake_data_filtered
        if len(filtered_ice_free) >= 2:
            # Get SWOT and in-situ data on SWOT dates
            swot_anomaly = filtered_ice_free['swot_wse_anomaly'].dropna()
            stage_anomaly = filtered_ice_free['stage_anomaly_swotdates'].dropna()
            
            # Find common indices for both datasets
            common_indices = filtered_ice_free.dropna(subset=['swot_wse_anomaly', 'stage_anomaly_swotdates']).index
            if len(common_indices) >= 3:
                swot_vals = filtered_ice_free.loc[common_indices, 'swot_wse_anomaly'].values
                stage_vals = filtered_ice_free.loc[common_indices, 'stage_anomaly_swotdates'].values
                
                # Calculate NSE = 1 - (SS_res / SS_tot)
                ss_res = np.sum((stage_vals - swot_vals) ** 2)
                ss_tot = np.sum((stage_vals - np.mean(stage_vals)) ** 2)
                
                if ss_tot > 0:
                    nse_target_swotdates = 1 - (ss_res / ss_tot)
                else:
                    nse_target_swotdates = -999.0  # No variance in observed data
        
        # Calculate NSE for daily approach (interpolation)
        if len(lake_data_filtered) >= 3:
            interp_result = interpolate_swot_to_insitu_dates(lake_data_filtered, lake_data, variable='wse')
            
            if interp_result is not None:
                interpolated_swot = interp_result['interpolated_values']
                valid_dates = interp_result['valid_dates']
                
                # Get corresponding in-situ stage anomaly values
                dates_df = pd.DataFrame({'date': valid_dates})
                merged_dates = pd.merge(dates_df, lake_data_ice_free[['date', 'stage_anomaly_alldates']], 
                                      on='date', how='inner')
                
                if len(merged_dates) >= 3 and len(interpolated_swot) >= 3:
                    # Match the arrays by taking minimum length
                    min_len = min(len(interpolated_swot), len(merged_dates))
                    swot_interp = interpolated_swot[:min_len]
                    stage_obs = merged_dates['stage_anomaly_alldates'].values[:min_len]
                    
                    # Remove any remaining NaN values
                    valid_mask = ~(np.isnan(swot_interp) | np.isnan(stage_obs))
                    if valid_mask.sum() >= 3:
                        swot_clean = swot_interp[valid_mask]
                        stage_clean = stage_obs[valid_mask]
                        
                        # Calculate NSE
                        ss_res = np.sum((stage_clean - swot_clean) ** 2)
                        ss_tot = np.sum((stage_clean - np.mean(stage_clean)) ** 2)
                        
                        if ss_tot > 0:
                            nse_target_alldates = 1 - (ss_res / ss_tot)
                        else:
                            nse_target_alldates = -999.0  # No variance in observed data
        
        nse_results.append({
            'swot_lake_id': lake_id,
            'nse_target_alldates': nse_target_alldates if not np.isnan(nse_target_alldates) else -999.0,
            'nse_target_swotdates': nse_target_swotdates if not np.isnan(nse_target_swotdates) else -999.0,
            'n_observations_filtered': len(lake_data_filtered)
        })
    
    # Calculate summary statistics
    nse_df = pd.DataFrame(nse_results)
    
    # Filter out failed cases (-999) for statistics
    valid_daily = nse_df['nse_target_alldates'][(nse_df['nse_target_alldates'] > -990)]
    valid_synoptic = nse_df['nse_target_swotdates'][(nse_df['nse_target_swotdates'] > -990)]
    
    # Calculate metrics
    nse_min_daily = valid_daily.min() if len(valid_daily) > 0 else np.nan
    nse_mean_daily = valid_daily.mean() if len(valid_daily) > 0 else np.nan
    nse_median_daily = valid_daily.median() if len(valid_daily) > 0 else np.nan
    nse_max_daily = valid_daily.max() if len(valid_daily) > 0 else np.nan
    
    nse_min_synoptic = valid_synoptic.min() if len(valid_synoptic) > 0 else np.nan
    nse_mean_synoptic = valid_synoptic.mean() if len(valid_synoptic) > 0 else np.nan
    nse_median_synoptic = valid_synoptic.median() if len(valid_synoptic) > 0 else np.nan
    nse_max_synoptic = valid_synoptic.max() if len(valid_synoptic) > 0 else np.nan
    
    if verbose:
        print(f"  {filter_name}: Calculated NSE for {len(nse_df)} lakes")
        print(f"  Daily NSE range: {nse_min_daily:.3f} to {nse_max_daily:.3f}")
        print(f"  Synoptic NSE range: {nse_min_synoptic:.3f} to {nse_max_synoptic:.3f}")
    
    return {
        'nse_min_daily': nse_min_daily,
        'nse_mean_daily': nse_mean_daily,
        'nse_median_daily': nse_median_daily,
        'nse_max_daily': nse_max_daily,
        'nse_min_synoptic': nse_min_synoptic,
        'nse_mean_synoptic': nse_mean_synoptic,
        'nse_median_synoptic': nse_median_synoptic,
        'nse_max_synoptic': nse_max_synoptic,
        'nse_df': nse_df,
        'lake_nse_daily': valid_daily.values if len(valid_daily) > 0 else [],
        'lake_nse_synoptic': valid_synoptic.values if len(valid_synoptic) > 0 else []
    }

def apply_filter_with_date_range(df, filter_func):
    """Apply filter combined with date range filter."""
    return filter_func(df) & date_range_swot_era(df)

def calculate_filter_metrics(df, filter_mask):
    """Calculate precision, recall, and error metrics for a filter."""
    
    # Apply filter
    df_filtered = df[filter_mask].copy()
    
    if len(df_filtered) == 0:
        return {
            'n_observations': 0,
            'n_good_observations': 0,
            'precision': np.nan,
            'recall': np.nan,
            'mae': np.nan,
            'rmse': np.nan,
            'nrmse': np.nan,
            'temporal_resolution': np.nan
        }
    
    # Count good/bad observations
    n_good_filtered = (df_filtered['good_observation'] == True).sum()
    n_total_filtered = len(df_filtered)
    
    # Calculate precision and recall
    # Precision: what fraction of kept observations are actually good?
    precision = n_good_filtered / n_total_filtered if n_total_filtered > 0 else 0
    
    # Recall: what fraction of all good observations did we keep?
    n_total_good = (df['good_observation'] == True).sum()
    recall = n_good_filtered / n_total_good if n_total_good > 0 else 0
    
    # Calculate F1 score
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0
    
    # Error metrics
    error_data = df_filtered['swot_wse_abs_error'].dropna()
    if len(error_data) > 0:
        mae = error_data.mean()
        rmse = np.sqrt(np.mean(error_data**2))
        p68_error = np.percentile(error_data, 68)  # 68th percentile error
        # NRMSE relative to range of WSE values
        wse_range = df_filtered['swot_wse_anomaly'].max() - df_filtered['swot_wse_anomaly'].min()
        nrmse = rmse / wse_range if wse_range > 0 else np.nan
        
        # Additional error statistics
        error_mean = error_data.mean()
        error_median = error_data.median()
        error_std = error_data.std()
    else:
        mae = rmse = p68_error = nrmse = np.nan
        error_mean = error_median = error_std = np.nan
    
    # Temporal resolution using SUM method (per-lake then sum across lakes)
    total_days_all_lakes = 0
    total_obs_all_lakes = 0
    
    for lake_id in df_filtered['swot_lake_id'].unique():
        lake_filtered = df_filtered[df_filtered['swot_lake_id'] == lake_id].copy()
        if len(lake_filtered) >= 2:
            lake_filtered = lake_filtered.sort_values('date')
            first_date = pd.to_datetime(lake_filtered['date'].min())
            last_date = pd.to_datetime(lake_filtered['date'].max())
            total_days = (last_date - first_date).days + 1
            
            total_days_all_lakes += total_days
            total_obs_all_lakes += len(lake_filtered)
    
    temporal_resolution = total_days_all_lakes / total_obs_all_lakes if total_obs_all_lakes > 0 else np.nan
    
    return {
        'n_observations': n_total_filtered,
        'n_good_observations': n_good_filtered,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'mae': mae,
        'rmse': rmse,
        'p68_error': p68_error,
        'nrmse': nrmse,
        'temporal_resolution': temporal_resolution,
        'error_mean': error_mean,
        'error_median': error_median,
        'error_std': error_std
    }

def evaluate_filters_overall(df):
    """Evaluate all filters on the entire dataset (overall statistics)."""
    
    print("Evaluating filters on overall dataset...")
    
    overall_results = {}
    
    # Calculate total good observations for recall calculation
    total_good_observations = (df['good_observation'] == True).sum()
    total_observations = len(df)
    total_lakes = len(df['swot_lake_id'].unique())
    
    print(f"Total observations: {total_observations}")
    print(f"Total lakes: {total_lakes}")
    print(f"Total good observations: {total_good_observations} ({total_good_observations/total_observations*100:.1f}%)")
    
    # Evaluate each filter
    for filter_key, filter_info in FILTERS_TO_EVALUATE.items():
        print(f"\nEvaluating {filter_info['name']}...")
        
        try:
            if filter_key == 'all_observations':
                # All observations filter already includes date range
                filter_mask = filter_info['filter_func'](df)
                # Count lakes with valid results for all_observations
                lakes_with_valid_results = 0
                lakes_attempted = 0
                for lake_id in df['swot_lake_id'].unique():
                    lake_data = df[df['swot_lake_id'] == lake_id].copy()
                    if len(lake_data) >= 5:
                        lakes_attempted += 1
                        lake_mask = filter_mask[lake_data.index]
                        if lake_mask.any():
                            lakes_with_valid_results += 1
            elif filter_key == 'adaptive':
                # Use pre-computed adaptive filter results from CSV column
                if 'adaptive_filter' not in df.columns:
                    print(f"    Warning: 'adaptive_filter' column not found in data!")
                    print(f"    Please run 'src/filter/apply_adaptive_filter.py' first to generate adaptive filter results")
                    filter_mask = pd.Series(False, index=df.index)
                else:
                    # adaptive_filter column: 0 = accepted, 1 = rejected
                    # Convert to boolean mask: True = keep (accepted), False = reject
                    adaptive_mask = (df['adaptive_filter'] == 0)
                    # Apply date range filter
                    date_mask = date_range_swot_era(df)
                    filter_mask = adaptive_mask & date_mask
                    
                    # Count lakes with valid results
                    lakes_with_valid_results = 0
                    lakes_attempted = 0
                    for lake_id in df['swot_lake_id'].unique():
                        lake_data = df[df['swot_lake_id'] == lake_id].copy()
                        # Count lakes that have any SWOT data (regardless of whether filter was applied)
                        swot_data = lake_data[lake_data['swot_wse'].notna()]
                        if len(swot_data) >= 5:
                            lakes_attempted += 1
                            lake_mask = filter_mask[lake_data.index]
                            if lake_mask.any():
                                lakes_with_valid_results += 1
                    
                    n_accepted = filter_mask.sum()
                    n_total_with_swot = (~df['swot_wse'].isna()).sum()
                    print(f"    Using pre-computed adaptive filter results: {n_accepted}/{n_total_with_swot} SWOT observations accepted")
            else:
                filter_mask = apply_filter_with_date_range(df, filter_info['filter_func'])
                # Count lakes with valid results for non-adaptive filters
                if filter_key != 'all_observations':
                    lakes_with_valid_results = 0
                    lakes_attempted = 0
                    for lake_id in df['swot_lake_id'].unique():
                        lake_data = df[df['swot_lake_id'] == lake_id].copy()
                        if len(lake_data) >= 5:
                            lakes_attempted += 1
                            lake_mask = filter_mask[lake_data.index]
                            if lake_mask.any():
                                lakes_with_valid_results += 1
                else:
                    # For all_observations, every lake has "valid" results
                    lakes_with_valid_results = total_lakes
                    lakes_attempted = total_lakes
            
            # Apply filter and calculate metrics
            df_filtered = df[filter_mask].copy()
            
            n_kept = len(df_filtered)
            n_good_kept = (df_filtered['good_observation'] == True).sum()
            n_bad_kept = n_kept - n_good_kept
            
            # Calculate precision and recall
            precision = n_good_kept / n_kept if n_kept > 0 else 0
            recall = n_good_kept / total_good_observations if total_good_observations > 0 else 0
            
            # Calculate F1 score
            if precision + recall > 0:
                f1_score_value = 2 * (precision * recall) / (precision + recall)
            else:
                f1_score_value = 0
            
            # Error metrics (overall across all observations)
            error_data = df_filtered['swot_wse_abs_error'].dropna()
            signed_error_data = df_filtered['swot_wse_error'].dropna()
            
            if len(error_data) > 0:
                mae = error_data.mean()
                rmse = np.sqrt(np.mean(error_data**2))
                p68_error = np.percentile(error_data, 68)  # 68th percentile error
                # NRMSE relative to overall WSE range
                wse_range = df['swot_wse_anomaly'].max() - df['swot_wse_anomaly'].min()
                nrmse = rmse / wse_range if wse_range > 0 else np.nan
                
                # Additional error statistics (overall)
                error_mean = error_data.mean()
                error_median = error_data.median()
                error_std = error_data.std()
            else:
                mae = rmse = p68_error = nrmse = np.nan
                error_mean = error_median = error_std = np.nan
            
            # Signed error statistics
            if len(signed_error_data) > 0:
                signed_error_std = signed_error_data.std()
                signed_error_mean = signed_error_data.mean()
            else:
                signed_error_std = signed_error_mean = np.nan
            
            # Per-lake metrics, then statistics across lakes
            lake_maes = []
            lake_rmses = []
            lake_p68s = []
            lake_temp_ress = []
            lake_precisions = []
            lake_recalls = []
            lake_signed_error_stds = []
            lake_n_observations = []
            lake_r2s = []
            
            for lake_id in df_filtered['swot_lake_id'].unique():
                lake_filtered = df_filtered[df_filtered['swot_lake_id'] == lake_id].copy()
                lake_error_data = lake_filtered['swot_wse_abs_error'].dropna()
                lake_signed_error_data = lake_filtered['swot_wse_error'].dropna()
                
                if len(lake_error_data) >= 2:  # Need at least 2 observations for meaningful stats
                    # Error metrics for this lake
                    lake_mae = lake_error_data.mean()
                    lake_rmse = np.sqrt(np.mean(lake_error_data**2))
                    lake_p68 = np.percentile(lake_error_data, 68)
                    
                    lake_maes.append(lake_mae)
                    lake_rmses.append(lake_rmse)
                    lake_p68s.append(lake_p68)
                    
                    # Signed error std for this lake
                    if len(lake_signed_error_data) > 0:
                        lake_signed_error_std = np.std(lake_signed_error_data)
                        lake_signed_error_stds.append(lake_signed_error_std)
                    
                    # R² calculation for this lake
                    swot_wse = lake_filtered['swot_wse_anomaly'].dropna()
                    stage_anomaly = lake_filtered['stage_anomaly_swotdates'].dropna()
                    
                    # Find common indices for both datasets
                    common_indices = lake_filtered.dropna(subset=['swot_wse_anomaly', 'stage_anomaly_swotdates']).index
                    if len(common_indices) >= 3:  # Need at least 3 points for R²
                        swot_vals = lake_filtered.loc[common_indices, 'swot_wse_anomaly'].values
                        stage_vals = lake_filtered.loc[common_indices, 'stage_anomaly_swotdates'].values
                        
                        # Calculate R²
                        from scipy.stats import pearsonr
                        r, p_value = pearsonr(swot_vals, stage_vals)
                        r2 = r**2
                        lake_r2s.append(r2)
                    else:
                        lake_r2s.append(np.nan)
                    
                    # Number of observations for this lake
                    lake_total_observations = len(lake_filtered)
                    lake_n_observations.append(lake_total_observations)
                    
                    # Precision and recall for this lake
                    lake_good_observations = (lake_filtered['good_observation'] == True).sum()
                    lake_precision = lake_good_observations / lake_total_observations if lake_total_observations > 0 else 0
                    lake_precisions.append(lake_precision)
                    
                    # For recall, need to get total good observations for this lake from original data
                    lake_original = df[df['swot_lake_id'] == lake_id].copy()
                    lake_total_good = (lake_original['good_observation'] == True).sum()
                    lake_recall = lake_good_observations / lake_total_good if lake_total_good > 0 else 0
                    lake_recalls.append(lake_recall)
                    
                    # Temporal resolution for this lake
                    if len(lake_filtered) >= 2:
                        lake_filtered_sorted = lake_filtered.sort_values('date')
                        first_date = pd.to_datetime(lake_filtered_sorted['date'].min())
                        last_date = pd.to_datetime(lake_filtered_sorted['date'].max())
                        total_days = (last_date - first_date).days + 1
                        lake_temp_res = total_days / len(lake_filtered)
                        lake_temp_ress.append(lake_temp_res)
            
            # Statistics across lakes
            if lake_maes:
                mae_mean_across_lakes = np.mean(lake_maes)
                mae_median_across_lakes = np.median(lake_maes)
                mae_std_across_lakes = np.std(lake_maes)
                
                rmse_mean_across_lakes = np.mean(lake_rmses)
                rmse_median_across_lakes = np.median(lake_rmses)
                rmse_std_across_lakes = np.std(lake_rmses)
                
                p68_mean_across_lakes = np.mean(lake_p68s)
                p68_median_across_lakes = np.median(lake_p68s)
                p68_std_across_lakes = np.std(lake_p68s)
                
                temp_res_mean_across_lakes = np.mean(lake_temp_ress) if lake_temp_ress else np.nan
                temp_res_median_across_lakes = np.median(lake_temp_ress) if lake_temp_ress else np.nan
                temp_res_std_across_lakes = np.std(lake_temp_ress) if lake_temp_ress else np.nan
                
                precision_mean_across_lakes = np.mean(lake_precisions)
                precision_median_across_lakes = np.median(lake_precisions)
                precision_std_across_lakes = np.std(lake_precisions)
                
                recall_mean_across_lakes = np.mean(lake_recalls)
                recall_median_across_lakes = np.median(lake_recalls)
                recall_std_across_lakes = np.std(lake_recalls)
                
                # R² statistics
                r2_mean_across_lakes = np.nanmean(lake_r2s) if lake_r2s else np.nan
                r2_median_across_lakes = np.nanmedian(lake_r2s) if lake_r2s else np.nan
                r2_std_across_lakes = np.nanstd(lake_r2s) if lake_r2s else np.nan
            else:
                mae_mean_across_lakes = mae_median_across_lakes = mae_std_across_lakes = np.nan
                rmse_mean_across_lakes = rmse_median_across_lakes = rmse_std_across_lakes = np.nan
                p68_mean_across_lakes = p68_median_across_lakes = p68_std_across_lakes = np.nan
                temp_res_mean_across_lakes = temp_res_median_across_lakes = temp_res_std_across_lakes = np.nan
                precision_mean_across_lakes = precision_median_across_lakes = precision_std_across_lakes = np.nan
                recall_mean_across_lakes = recall_median_across_lakes = recall_std_across_lakes = np.nan
                r2_mean_across_lakes = r2_median_across_lakes = r2_std_across_lakes = np.nan
            
            # Overall temporal resolution using SUM method (consistent with quality_analysis_wse_error.py)
            total_days_all_lakes = 0
            total_obs_all_lakes = 0
            
            for lake_id in df_filtered['swot_lake_id'].unique():
                lake_filtered = df_filtered[df_filtered['swot_lake_id'] == lake_id].copy()
                if len(lake_filtered) >= 2:
                    lake_filtered = lake_filtered.sort_values('date')
                    first_date = pd.to_datetime(lake_filtered['date'].min())
                    last_date = pd.to_datetime(lake_filtered['date'].max())
                    total_days = (last_date - first_date).days + 1
                    
                    total_days_all_lakes += total_days
                    total_obs_all_lakes += len(lake_filtered)
            
            overall_temp_res = total_days_all_lakes / total_obs_all_lakes if total_obs_all_lakes > 0 else np.nan

            # Calculate NSE for this filter
            print(f"  Calculating NSE metrics for {filter_info['name']}...")
            nse_metrics = calculate_nse_for_filter(df, filter_mask, filter_info['name'], verbose=False)
            
            overall_results[filter_key] = {
                'name': filter_info['name'],
                'description': filter_info['description'],
                'total_observations': n_kept,
                'good_observations': n_good_kept,
                'bad_observations': n_bad_kept,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score_value,
                'mae': mae,
                'rmse': rmse,
                'p68_error': p68_error,
                'nrmse': nrmse,
                'temporal_resolution': overall_temp_res,
                'retention_rate': n_kept / total_observations,
                'lakes_with_results': lakes_with_valid_results,
                'lakes_attempted': lakes_attempted,
                'lake_success_rate': lakes_with_valid_results / lakes_attempted if lakes_attempted > 0 else 0,
                'error_mean': error_mean,
                'error_median': error_median,
                'error_std': error_std,
                'signed_error_mean': signed_error_mean,
                'signed_error_std': signed_error_std,
                'mae_mean_across_lakes': mae_mean_across_lakes,
                'mae_median_across_lakes': mae_median_across_lakes,
                'mae_std_across_lakes': mae_std_across_lakes,
                'rmse_mean_across_lakes': rmse_mean_across_lakes,
                'rmse_median_across_lakes': rmse_median_across_lakes,
                'rmse_std_across_lakes': rmse_std_across_lakes,
                'p68_mean_across_lakes': p68_mean_across_lakes,
                'p68_median_across_lakes': p68_median_across_lakes,
                'p68_std_across_lakes': p68_std_across_lakes,
                'temp_res_mean_across_lakes': temp_res_mean_across_lakes,
                'temp_res_median_across_lakes': temp_res_median_across_lakes,
                'temp_res_std_across_lakes': temp_res_std_across_lakes,
                'precision_mean_across_lakes': precision_mean_across_lakes,
                'precision_median_across_lakes': precision_median_across_lakes,
                'precision_std_across_lakes': precision_std_across_lakes,
                'recall_mean_across_lakes': recall_mean_across_lakes,
                'recall_median_across_lakes': recall_median_across_lakes,
                'recall_std_across_lakes': recall_std_across_lakes,
                # Store filtered data for histogram generation
                'signed_errors': signed_error_data.values,
                'abs_errors': error_data.values,
                # Store per-lake data for boxplot generation
                'lake_rmses': lake_rmses,
                'lake_maes': lake_maes,
                'lake_p68s': lake_p68s,
                'lake_signed_error_stds': lake_signed_error_stds,
                'lake_precisions': lake_precisions,
                'lake_recalls': lake_recalls,
                'lake_n_observations': lake_n_observations,
                'lake_temp_ress': lake_temp_ress,
                'lake_r2s': lake_r2s,
                # NSE metrics
                'nse_min_daily': nse_metrics['nse_min_daily'],
                'nse_mean_daily': nse_metrics['nse_mean_daily'],
                'nse_median_daily': nse_metrics['nse_median_daily'],
                'nse_max_daily': nse_metrics['nse_max_daily'],
                'nse_min_synoptic': nse_metrics['nse_min_synoptic'],
                'nse_mean_synoptic': nse_metrics['nse_mean_synoptic'],
                'nse_median_synoptic': nse_metrics['nse_median_synoptic'],
                'nse_max_synoptic': nse_metrics['nse_max_synoptic'],
                'nse_df': nse_metrics['nse_df'],
                'lake_nse_daily': nse_metrics['lake_nse_daily'],
                'lake_nse_synoptic': nse_metrics['lake_nse_synoptic']
            }
            
            print(f"  Kept: {n_kept}/{total_observations} ({n_kept/total_observations*100:.1f}%)")
            print(f"  Lakes with results: {lakes_with_valid_results}/{lakes_attempted} ({lakes_with_valid_results/lakes_attempted*100:.1f}%)")
            print(f"  Good: {n_good_kept}, Bad: {n_bad_kept}")
            print(f"  Precision: {precision:.3f}, Recall: {recall:.3f}")
            
        except Exception as e:
            print(f"  Error evaluating {filter_info['name']}: {e}")
            overall_results[filter_key] = {
                'name': filter_info['name'],
                'description': filter_info['description'],
                'total_observations': 0,
                'good_observations': 0,
                'bad_observations': 0,
                'precision': np.nan,
                'recall': np.nan,
                'f1_score': np.nan,
                'mae': np.nan,
                'rmse': np.nan,
                'p68_error': np.nan,
                'nrmse': np.nan,
                'temporal_resolution': np.nan,
                'retention_rate': 0,
                'lakes_with_results': 0,
                'lakes_attempted': total_lakes,
                'lake_success_rate': 0,
                'error_mean': np.nan,
                'error_median': np.nan,
                'error_std': np.nan,
                'signed_error_mean': np.nan,
                'signed_error_std': np.nan,
                'mae_mean_across_lakes': np.nan,
                'mae_median_across_lakes': np.nan,
                'mae_std_across_lakes': np.nan,
                'rmse_mean_across_lakes': np.nan,
                'rmse_median_across_lakes': np.nan,
                'rmse_std_across_lakes': np.nan,
                'p68_mean_across_lakes': np.nan,
                'p68_median_across_lakes': np.nan,
                'p68_std_across_lakes': np.nan,
                'temp_res_mean_across_lakes': np.nan,
                'temp_res_median_across_lakes': np.nan,
                'temp_res_std_across_lakes': np.nan,
                'precision_mean_across_lakes': np.nan,
                'precision_median_across_lakes': np.nan,
                'precision_std_across_lakes': np.nan,
                'recall_mean_across_lakes': np.nan,
                'recall_median_across_lakes': np.nan,
                'recall_std_across_lakes': np.nan,
                # Store empty data for histogram generation
                'signed_errors': np.array([]),
                'abs_errors': np.array([]),
                # Store empty per-lake data for boxplot generation
                'lake_rmses': [],
                'lake_maes': [],
                'lake_p68s': [],
                'lake_signed_error_stds': [],
                'lake_precisions': [],
                'lake_recalls': [],
                'lake_n_observations': [],
                'lake_temp_ress': [],
                'lake_r2s': [],
                # NSE metrics (failed case)
                'nse_min_daily': np.nan,
                'nse_mean_daily': np.nan,
                'nse_median_daily': np.nan,
                'nse_max_daily': np.nan,
                'nse_min_synoptic': np.nan,
                'nse_mean_synoptic': np.nan,
                'nse_median_synoptic': np.nan,
                'nse_max_synoptic': np.nan,
                'nse_df': pd.DataFrame(),
                'lake_nse_daily': [],
                'lake_nse_synoptic': []
            }
    
    return overall_results

def create_comparison_plots(overall_results, output_dir):
    """Create comparison plots for all filters using overall results."""
    
    # Set up plotting style
    plt.style.use('default')
    
    # Create observation composition plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Observation counts (good vs bad)
    filter_names = [results['name'] for results in overall_results.values()]
    good_counts = [results['good_observations'] for results in overall_results.values()]
    bad_counts = [results['bad_observations'] for results in overall_results.values()]
    
    x = np.arange(len(filter_names))
    width = 0.6
    
    bars_good = ax1.bar(x, good_counts, width, label='Good Observations', color='lightgreen', alpha=0.8)
    bars_bad = ax1.bar(x, bad_counts, width, bottom=good_counts, label='Bad Observations', color='lightcoral', alpha=0.8)
    
    ax1.set_xlabel('Filter')
    ax1.set_ylabel('Number of Observations')
    ax1.set_title('Observation Composition by Filter', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(filter_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add count labels on bars
    for i, (good, bad) in enumerate(zip(good_counts, bad_counts)):
        total = good + bad
        if total > 0:
            ax1.text(i, total + max(good_counts + bad_counts) * 0.01, str(total), 
                    ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Precision vs Recall
    precisions = [results['precision'] for results in overall_results.values()]
    recalls = [results['recall'] for results in overall_results.values()]
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(filter_names)))
    for i, (name, prec, rec, color) in enumerate(zip(filter_names, precisions, recalls, colors)):
        ax2.scatter(rec, prec, s=100, color=color, alpha=0.8, label=name)
        ax2.annotate(name, (rec, prec), xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision vs Recall by Filter', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'filter_composition_plots.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Metrics are now only saved to CSV, not plotted
    
    # Create and save summary table
    summary_data = []
    for filter_key, results in overall_results.items():
        summary_data.append({
            'Filter': results['name'],
            'Description': results['description'],
            'Total_Obs': results['total_observations'],
            'Good_Obs': results['good_observations'],
            'Bad_Obs': results['bad_observations'],
            'Precision': results['precision'],
            'Recall': results['recall'],
            'F1_Score': results['f1_score'],
            'MAE': results['mae'],
            'RMSE': results['rmse'],
            'P68_Error': results['p68_error'],
            'NRMSE': results['nrmse'],
            'Temporal_Resolution': results['temporal_resolution'],
            'Retention_Rate': results['retention_rate'],
            'Lakes_With_Results': results['lakes_with_results'],
            'Lakes_Attempted': results['lakes_attempted'],
            'Lake_Success_Rate': results['lake_success_rate'],
            'Error_Mean': results['error_mean'],
            'Error_Median': results['error_median'],
            'Error_Std': results['error_std'],
            'Signed_Error_Mean': results['signed_error_mean'],
            'Signed_Error_Std': results['signed_error_std'],
            'MAE_Mean_Across_Lakes': results['mae_mean_across_lakes'],
            'MAE_Median_Across_Lakes': results['mae_median_across_lakes'],
            'MAE_Std_Across_Lakes': results['mae_std_across_lakes'],
            'RMSE_Mean_Across_Lakes': results['rmse_mean_across_lakes'],
            'RMSE_Median_Across_Lakes': results['rmse_median_across_lakes'],
            'RMSE_Std_Across_Lakes': results['rmse_std_across_lakes'],
            'P68_Mean_Across_Lakes': results['p68_mean_across_lakes'],
            'P68_Median_Across_Lakes': results['p68_median_across_lakes'],
            'P68_Std_Across_Lakes': results['p68_std_across_lakes'],
            'TempRes_Mean_Across_Lakes': results['temp_res_mean_across_lakes'],
            'TempRes_Median_Across_Lakes': results['temp_res_median_across_lakes'],
            'TempRes_Std_Across_Lakes': results['temp_res_std_across_lakes'],
            'Precision_Mean_Across_Lakes': results['precision_mean_across_lakes'],
            'Precision_Median_Across_Lakes': results['precision_median_across_lakes'],
            'Precision_Std_Across_Lakes': results['precision_std_across_lakes'],
            'Recall_Mean_Across_Lakes': results['recall_mean_across_lakes'],
            'Recall_Median_Across_Lakes': results['recall_median_across_lakes'],
            'Recall_Std_Across_Lakes': results['recall_std_across_lakes'],
            # NSE metrics
            'NSE_Min_Daily': results['nse_min_daily'],
            'NSE_Mean_Daily': results['nse_mean_daily'],
            'NSE_Median_Daily': results['nse_median_daily'],
            'NSE_Max_Daily': results['nse_max_daily'],
            'NSE_Min_Synoptic': results['nse_min_synoptic'],
            'NSE_Mean_Synoptic': results['nse_mean_synoptic'],
            'NSE_Median_Synoptic': results['nse_median_synoptic'],
            'NSE_Max_Synoptic': results['nse_max_synoptic']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_dir, 'filter_comparison_summary.csv'), index=False)
    
    return summary_df

def create_signed_error_comparison(overall_results, output_dir):
    """Create 4-panel histogram comparing signed error distributions with 68th percentile vs std dev."""
    
    # Select the four specific filters for comparison
    target_filters = ['valid_subset', 'quality_flag', 'custom_standard', 'adaptive']
    filter_data = {}
    
    # Collect data for each filter (we need the original dataframe data)
    # Since we don't have direct access to the filtered data here, we need to reconstruct it
    # This is a limitation - ideally we'd store the filtered data during evaluation
    print("Note: Signed error comparison requires re-filtering data for histogram generation")
    
    return  # For now, return early - need access to original dataframe

def create_signed_error_comparison_with_data(df, overall_results, output_dir):
    """Create 4-panel histogram comparing signed error distributions with 68th percentile vs std dev."""

    # Select four filters for comparison (including idealized)
    target_filters = ['valid_subset', 'quality_flag', 'custom_standard', 'adaptive']
    panel_labels = ['(a)', '(b)', '(c)', '(d)']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, filter_key in enumerate(target_filters):
        if filter_key not in overall_results:
            continue

        ax = axes[i]
        filter_info = FILTERS_TO_EVALUATE[filter_key]
        results = overall_results[filter_key]

        # Use stored data from main evaluation (no re-filtering needed!)
        signed_errors = results['signed_errors']
        abs_errors = results['abs_errors']

        try:
            if len(signed_errors) == 0:
                ax.text(0.5, 0.5, 'No data available', ha='center', va='center',
                       transform=ax.transAxes, fontsize=12)
                ax.set_title(f'{panel_labels[i]} {filter_info["name"]}', fontweight='bold')
                continue

            # Calculate statistics
            mean_signed = np.mean(signed_errors)
            std_signed = np.std(signed_errors)
            p68_abs = np.percentile(abs_errors, 68) if len(abs_errors) > 0 else np.nan

            # For non-idealized filters, truncate to 2nd and 98th percentiles
            if filter_key in ['quality_flag', 'custom_standard', 'adaptive']:
                p2 = np.percentile(signed_errors, 2)
                p98 = np.percentile(signed_errors, 98)
                signed_errors_truncated = signed_errors[(signed_errors >= p2) & (signed_errors <= p98)]
                hist_data = signed_errors_truncated
                title_suffix = f' (n={len(hist_data):,})'
            else:
                hist_data = signed_errors
                title_suffix = f' (n={len(signed_errors):,})'

            # Create histogram
            ax.hist(hist_data, bins=30, alpha=0.7, color='skyblue', density=True,
                   edgecolor='black', linewidth=0.5)

            # Add vertical lines for mean
            ax.axvline(mean_signed, color='red', linestyle='-', linewidth=2,
                      label=f'Mean: {mean_signed:.3f}m')

            # Add 68th percentile lines (±p68 around mean based on absolute errors)
            p68_pos = mean_signed + p68_abs
            p68_neg = mean_signed - p68_abs
            ax.axvline(p68_pos, color='orange', linestyle='--', linewidth=2,
                      label=f'68th percentile: ±{p68_abs:.3f}m')
            ax.axvline(p68_neg, color='orange', linestyle='--', linewidth=2)

            # Add standard deviation lines (±0.5σ around mean based on signed errors)
            std_pos = mean_signed + (std_signed / 2)
            std_neg = mean_signed - (std_signed / 2)
            ax.axvline(std_pos, color='green', linestyle=':', linewidth=2,
                      label=f'Half std dev: ±{std_signed/2:.3f}m')
            ax.axvline(std_neg, color='green', linestyle=':', linewidth=2)

            # Formatting
            ax.set_xlabel('Signed WSE Error (m)')
            ax.set_ylabel('Density')
            ax.set_title(f'{panel_labels[i]} {filter_info["name"]}{title_suffix}', fontweight='bold')
            ax.set_xlim(-1, 1)  # Set x-axis range to -1 to 1 meters
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9, loc='upper right')

            # Add summary statistics as text
            stats_text = f'Mean: {mean_signed:.3f}m\nStd: {std_signed:.3f}m\n68th %ile: {p68_abs:.3f}m'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)[:50]}...', ha='center', va='center',
                   transform=ax.transAxes, fontsize=10)
            ax.set_title(f'{panel_labels[i]} {filter_info["name"]}', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'signed_error_distributions.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

def create_three_panel_comparison(overall_results, output_dir):
    """Create 3-panel comparison: 1σ error, R², temporal resolution."""

    # Define the three target filters that we want to compare
    target_filters = {
        'quality_flag': 'Quality Flag',
        'custom_standard': 'Simple',
        'adaptive': 'Adaptive'
    }

    print("Creating 3-panel filter comparison...")

    # Set seaborn style for prettier plots
    sns.set_style("white")
    sns.set_palette("colorblind")

    # Create 3-panel plot (1x3)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Define metrics and their labels
    metrics = ['signed_error_std', 'r2', 'temporal_resolution']
    metric_keys = ['lake_signed_error_stds', 'lake_r2s', 'lake_temp_ress']
    metric_labels = {
        'signed_error_std': '1σ Error',
        'r2': 'R²',
        'temporal_resolution': 'Temporal Resolution'
    }

    # Create 3 panels as boxplots
    for i, (metric, key) in enumerate(zip(metrics, metric_keys)):
        ax = axes[i]

        # Collect data for each filter
        filter_data = []
        filter_labels = []

        for filter_key, filter_name in target_filters.items():
            if filter_key in overall_results and key in overall_results[filter_key]:
                values = overall_results[filter_key][key]
                if values:  # Only add if we have data
                    # Remove NaN values
                    clean_values = [v for v in values if not np.isnan(v)]
                    if clean_values:
                        # Convert 1σ error from meters to centimeters
                        if metric == 'signed_error_std':
                            clean_values = [v * 100 for v in clean_values]
                        filter_data.append(clean_values)
                        filter_labels.append(filter_name)

        if filter_data:
            # Prepare data for seaborn boxplot
            plot_data = []
            for j, (data, label) in enumerate(zip(filter_data, filter_labels)):
                for value in data:
                    plot_data.append({'Filter': label, 'Value': value})

            if plot_data:
                plot_df = pd.DataFrame(plot_data)

                # Create seaborn boxplot using current palette
                sns.boxplot(data=plot_df, x='Filter', y='Value', ax=ax,
                           width=0.8, flierprops={'alpha': 0.3, 'markersize': 4})

        ax.set_title(metric_labels[metric], fontweight='bold', pad=10)

        # Set y-axis labels and x-axis labels
        if i == 0:  # Panel 1: 1σ Error
            ax.set_ylabel('centimeters', fontsize=12)
            ax.set_xlabel('')
            ax.set_xticklabels([])
            ax.set_ylim(0, 100)
        elif i == 1:  # Panel 2: R²
            ax.set_ylabel('')
            ax.set_xlabel('')
            ax.set_xticklabels([])
            ax.set_ylim(0, 1)
        elif i == 2:  # Panel 3: Temporal Resolution
            ax.set_ylabel('days/observation')
            ax.set_xlabel('')
            ax.tick_params(axis='x', rotation=0)
            # Set y-axis to show 0 to 95th percentile
            if filter_data:
                all_values = [val for sublist in filter_data for val in sublist if not np.isnan(val)]
                if all_values:
                    p95 = np.percentile(all_values, 95)
                    ax.set_ylim(0, p95)

    # Overall title and layout
    fig.suptitle('Filter Performance Summary', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, wspace=0.3)
    plt.savefig(os.path.join(output_dir, 'three_panel_filter_comparison.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # Reset style after plotting
    sns.reset_defaults()

def create_per_lake_boxplot_comparison(overall_results, output_dir):
    """Create 8-panel boxplot comparing per-lake results using stored per-lake data."""
    
    # Define the three target filters that we want to compare
    target_filters = {
        'quality_flag': 'Quality Flag', 
        'custom_standard': 'Simple',
        'adaptive': 'Adaptive'
    }
    
    
    print("Creating per-lake boxplot comparison using stored data...")

    # Set seaborn style for prettier plots
    sns.set_style("white")
    sns.set_palette("colorblind")
    plt.rcParams['font.family'] = 'Gill Sans'
    
    
    # Create 8-panel boxplot
    fig, axes = plt.subplots(2, 4, figsize=(12, 10))
    axes = axes.flatten()
    
    # Define metrics and their labels - match the stored data keys
    metrics = ['rmse', 'mae', 'p68_error', 'signed_error_std', 'precision', 'recall', 'temporal_resolution']
    metric_keys = ['lake_rmses', 'lake_maes', 'lake_p68s', 'lake_signed_error_stds', 'lake_precisions', 'lake_recalls', 'lake_temp_ress']
    metric_labels = {
        'rmse': '(a) RMSE',
        'mae': '(b) MAE', 
        'p68_error': '(c) P68 Error',
        'signed_error_std': '(d) 1-Sigma Error',
        'precision': '(e) Precision',
        'recall': '(f) Recall',
        'temporal_resolution': '(g) Temporal Resolution'
    }
    
    # Create boxplots for each metric (first 7 panels)
    for i, (metric, key) in enumerate(zip(metrics, metric_keys)):
        ax = axes[i]
        
        # Collect data for each filter
        filter_data = []
        filter_labels = []
        
        for filter_key, filter_name in target_filters.items():
            if filter_key in overall_results and key in overall_results[filter_key]:
                values = overall_results[filter_key][key]
                if values:  # Only add if we have data
                    # Remove NaN values
                    clean_values = [v for v in values if not np.isnan(v)]
                    if clean_values:
                        # Convert first 4 metrics from meters to centimeters
                        if i < 4:  # First 4 panels: rmse, mae, p68_error, signed_error_std
                            clean_values = [v * 100 for v in clean_values]
                        filter_data.append(clean_values)
                        filter_labels.append(filter_name)
        
        if filter_data:
            # Prepare data for seaborn boxplot
            plot_data = []
            for j, (data, label) in enumerate(zip(filter_data, filter_labels)):
                for value in data:
                    plot_data.append({'Filter': label, 'Value': value})
            
            if plot_data:
                plot_df = pd.DataFrame(plot_data)
                
                # Create seaborn boxplot using current palette
                sns.boxplot(data=plot_df, x='Filter', y='Value', hue='Filter', ax=ax,
                           width=0.8, flierprops={'alpha': 0.3, 'markersize': 4}, legend=False)
        
        # Set titles with letter labels and minimal y-axis labels
        ax.set_title(metric_labels[metric], fontweight='bold', pad=10, fontsize=16)
        if i < 4:  # Panels 1-4: Error metrics in cm
            if i == 0:  # Only first panel gets y-axis label
                ax.set_ylabel('Centimeters', fontsize=14, labelpad=2)
            else:
                ax.set_ylabel('')
            ax.set_xlabel('')
            ax.set_xticklabels([])
        elif i < 6:  # Panels 5-6: Precision/Recall
            ax.set_ylabel('')
            ax.set_xlabel('')
            ax.tick_params(axis='x', rotation=30, labelsize=12)
        elif i == 6:  # Panel 7: Temporal Resolution
            ax.set_ylabel('Days / Observation', fontsize=14, labelpad=2)
            ax.set_xlabel('')
            ax.tick_params(axis='x', rotation=30, labelsize=12)
        
        # Increase tick label font sizes and move closer to axis
        ax.tick_params(axis='y', labelsize=12, pad=.25)
        ax.tick_params(axis='x', labelsize=12, pad=.25)
        
        # Set y-axis limits for better visualization
        if i < 4:  # First 4 panels: error metrics in cm with common 0-100 range
            ax.set_ylim(0, 80)
        elif metric in ['precision', 'recall']:
            ax.set_ylim(0, 1)
        elif metric == 'temporal_resolution':
            # For temporal resolution, set y-axis to show 0 to 95th percentile
            if filter_data:
                all_values = [val for sublist in filter_data for val in sublist if not np.isnan(val)]
                if all_values:
                    p95 = np.percentile(all_values, 95)
                    ax.set_ylim(0, 45)
    
    # Create NSE boxplot in the 8th panel
    ax_nse = axes[7]  # 8th panel (0-indexed)
    
    # Collect NSE data for each filter
    filter_data = []
    filter_labels = []
    
    for filter_key, filter_name in target_filters.items():
        if filter_key in overall_results and 'lake_nse_daily' in overall_results[filter_key]:
            nse_values = overall_results[filter_key]['lake_nse_daily']
            if len(nse_values) > 0:  # Only add if we have data
                # Remove failed cases (NSE < -990)
                clean_values = [v for v in nse_values if v > -990]
                if clean_values:
                    filter_data.append(clean_values)
                    filter_labels.append(filter_name)
    
    if filter_data:
        # Prepare data for seaborn boxplot
        plot_data = []
        for j, (data, label) in enumerate(zip(filter_data, filter_labels)):
            for value in data:
                plot_data.append({'Filter': label, 'NSE': value})
        
        if plot_data:
            plot_df = pd.DataFrame(plot_data)
            
            # Create seaborn boxplot using current palette
            sns.boxplot(data=plot_df, x='Filter', y='NSE', hue='Filter', ax=ax_nse,
                       width=0.8, flierprops={'alpha': 0.3, 'markersize': 4}, legend=False)
    
    # Formatting for NSE panel with letter label
    ax_nse.set_title('(h) Nash-Sutcliffe Efficiency\n(Daily Interpolation)', fontweight='bold', pad=10, fontsize=16)
    ax_nse.set_xlabel('')
    ax_nse.set_ylabel('NSE', fontsize=14, labelpad=2)
    ax_nse.tick_params(axis='x', rotation=30, labelsize=12, pad=.25)
    ax_nse.tick_params(axis='y', labelsize=12, pad=.25)
    for label in ax_nse.get_yticklabels():
        label.set_rotation(45)
    ax_nse.set_ylim(-1, 1)  # NSE typically ranges from -∞ to 1, but -1 to 1 shows useful range
  
    
    # Overall title and layout
    fig.suptitle('Per-Lake Filter Performance Comparison', fontsize=18, fontweight='bold', y=0.96)
    #sns.despine()
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, hspace=0.25, wspace=0.35)
    plt.savefig(os.path.join(output_dir, 'filter_comparison_boxplot.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Reset style after plotting
    sns.reset_defaults()
    
    # Print summary statistics for the boxplot data
    print("\nPer-lake boxplot comparison summary:")
    for filter_key, filter_name in target_filters.items():
        if filter_key in overall_results and 'lake_rmses' in overall_results[filter_key]:
            n_lakes = len(overall_results[filter_key]['lake_rmses'])
            print(f"  {filter_name}: {n_lakes} lakes with valid data")



def main():
    """Main execution function."""
    
    print("SWOT Filter Evaluation Analysis")
    print("=" * 50)
    
    # Create output directory
    output_dir = 'experiments/01_swot_measurement_accuracy/filter_evaluation'
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print("Loading benchmark daily data...")
    data_dir = "data/timeseries/benchmark_daily"
    csv_files = glob.glob(os.path.join(data_dir, "*_daily.csv"))
    
    # Load problematic lake IDs
    problematic_ids = pd.read_csv('data/benchmark/problematic_swot_lake_ids.csv', 
                                  dtype={'swot_lake_id': str})
    problematic_lake_ids = set(str(lake_id) for lake_id in problematic_ids['swot_lake_id'].dropna().values)
    csv_files = [f for f in csv_files if not any(str(lake_id) in os.path.basename(f) for lake_id in problematic_lake_ids)]
    
    print(f"Processing {len(csv_files)} individual gage files...")
    
    # Limit to subset for testing (remove this line for full analysis)
    #csv_files = csv_files[:5]  # Test on first 5 lakes
    print(f"Testing on first {len(csv_files)} lakes...")
    
    # Load and combine data
    all_data = []
    for csv_file in csv_files:
        try:
            file_df = pd.read_csv(csv_file, dtype={'gage_id': str, 'swot_lake_id': str})
            #file_df = file_df[file_df['swot_crid']=='PID0'].copy()
            all_data.append(file_df)
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
            continue
    
    if len(all_data) == 0:
        print("No data loaded! Exiting.")
        return
    
    # Combine all data
    df = pd.concat(all_data, ignore_index=True)
    print(f"Loaded {len(df)} observations from {len(all_data)} individual files")
    
    # Filter data
    swot_cols_to_check = ['swot_wse_abs_error', 'swot_wse_error', 'swot_quality_f']
    df = df.dropna(subset=swot_cols_to_check, how='all')
    print(f"After excluding rows without SWOT observations: {len(df)} observations")
    
    # Exclude lakes with fewer than 5 WSE error observations
    lakes_wse_counts = df.groupby('swot_lake_id')['swot_wse_abs_error'].apply(lambda x: x.notna().sum())
    valid_lake_ids = lakes_wse_counts[lakes_wse_counts >= 5].index
    df = df[df['swot_lake_id'].isin(valid_lake_ids)]
    print(f"After excluding lakes with <5 WSE error observations: {len(df)} observations")
    
    # Apply date filter
    if START_DATE:
        df['date'] = pd.to_datetime(df['date'])
        df = df[df['date'] >= START_DATE]
        print(f"After date filtering (>= {START_DATE}): {len(df)} observations")
    
    # Calculate optimal WSE error threshold dynamically
    print(f"\nCalculating optimal WSE error threshold (target std = {TARGET_WSE_STD}m)...")
    threshold_result = find_threshold_for_target_std(df, target_std=TARGET_WSE_STD)
    wse_threshold = threshold_result['threshold']

    print(f"  Computed threshold: {wse_threshold:.3f}m")
    print(f"  Achieved std: {threshold_result['achieved_std']:.4f}m")
    print(f"  Observations meeting threshold: {threshold_result['n_observations']}")

    # Define good/bad observations using the computed threshold
    df['good_observation'] = df['swot_wse_abs_error'] <= wse_threshold
    good_obs_total = df['good_observation'].sum()

    print(f"Total good observations: {good_obs_total}/{len(df)} ({good_obs_total/len(df)*100:.1f}%)")

    # Update FILTERS_TO_EVALUATE with the computed threshold
    FILTERS_TO_EVALUATE['valid_subset']['name'] = f'Idealized ({wse_threshold:.3f}m)'
    FILTERS_TO_EVALUATE['valid_subset']['description'] = f'Observations meeting {wse_threshold:.3f}m error threshold (target std={TARGET_WSE_STD}m)'
    
    # Evaluate filters on overall dataset
    print(f"\nEvaluating {len(FILTERS_TO_EVALUATE)} filters on overall dataset...")
    overall_results = evaluate_filters_overall(df)
    
    # Create comparison plots and summary
    print("\nCreating comparison plots and summary...")
    summary_df = create_comparison_plots(overall_results, output_dir)
    
    # Create signed error distribution comparison
    print("\nCreating signed error distribution comparison...")
    create_signed_error_comparison_with_data(df, overall_results, output_dir)
    
    # Create per-lake boxplot comparison
    print("\nCreating per-lake boxplot comparison...")
    create_per_lake_boxplot_comparison(overall_results, output_dir)
    
    # Create three-panel comparison
    print("\nCreating three-panel filter comparison...")
    create_three_panel_comparison(overall_results, output_dir)
    
    # Print final summary table
    print(f"\n{'='*60}")
    print("FILTER EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total lakes: {len(df['swot_lake_id'].unique())}")
    print(f"Total observations: {len(df)}")
    print(f"WSE error threshold: {wse_threshold:.3f}m (computed for target std={TARGET_WSE_STD}m)")
    print(f"Results saved to: {output_dir}/")
    print()
    
    # Print summary table
    print("Summary Statistics:")
    print("="*170)
    print(f"{'Filter':<20} {'Lakes':<8} {'Total':<8} {'Good':<8} {'Bad':<8} {'Precision':<10} {'Recall':<8} {'RMSE':<8} {'P68_Err':<8} {'Temp_Res':<10}")
    print(f"{'':20} {'w/Data':<8} {'Obs':<8} {'Obs':<8} {'Obs':<8} {'':10} {'':8} {'(m)':<8} {'(m)':<8} {'(days)':<10}")
    print("-"*170)
    
    for _, row in summary_df.iterrows():
        print(f"{row['Filter']:<20} {row['Lakes_With_Results']:<8.0f} {row['Total_Obs']:<8.0f} {row['Good_Obs']:<8.0f} {row['Bad_Obs']:<8.0f} "
              f"{row['Precision']:<10.3f} {row['Recall']:<8.3f} {row['RMSE']:<8.3f} {row['P68_Error']:<8.3f} {row['Temporal_Resolution']:<10.1f}")
    
    print()
    
    # Show which filter performs best 
    best_precision_idx = summary_df['Precision'].idxmax()
    best_recall_idx = summary_df['Recall'].idxmax()
    best_rmse_idx = summary_df['RMSE'].idxmin()  # Lower is better
    best_p68_idx = summary_df['P68_Error'].idxmin()  # Lower is better
    best_temporal_idx = summary_df['Temporal_Resolution'].idxmin()  # Lower is better
    
    print("Best performing filters:")
    print(f"  Best precision: {summary_df.loc[best_precision_idx, 'Filter']} ({summary_df.loc[best_precision_idx, 'Precision']:.3f})")
    print(f"  Best recall: {summary_df.loc[best_recall_idx, 'Filter']} ({summary_df.loc[best_recall_idx, 'Recall']:.3f})")
    print(f"  Best RMSE: {summary_df.loc[best_rmse_idx, 'Filter']} ({summary_df.loc[best_rmse_idx, 'RMSE']:.3f} m)")
    print(f"  Best 68th percentile error: {summary_df.loc[best_p68_idx, 'Filter']} ({summary_df.loc[best_p68_idx, 'P68_Error']:.3f} m)")
    print(f"  Best temporal resolution: {summary_df.loc[best_temporal_idx, 'Filter']} ({summary_df.loc[best_temporal_idx, 'Temporal_Resolution']:.1f} days/obs)")

if __name__ == "__main__":
    main()