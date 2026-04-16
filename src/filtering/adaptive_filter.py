#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adaptive SWOT filter implementation for benchmark_daily data format

This module adapts the LakeSP v9 adaptive filter for use with the benchmark_daily
time series format used in this project.
"""

import pandas as pd
import numpy as np
try:
    from .customized_functions import calibrate_heuristic_thresholds, apply_customized_filter
except ImportError:
    from customized_functions import calibrate_heuristic_thresholds, apply_customized_filter


def prepare_data_for_adaptive_filter(df):
    """
    Convert benchmark_daily format to adaptive filter format
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame in benchmark_daily format with 'swot_' prefixed columns
        
    Returns:
    --------
    pandas.DataFrame : DataFrame with columns expected by adaptive filter
    """
    # Create mapping from benchmark format to adaptive filter format
    column_mapping = {
        'date': 'time',
        'swot_lake_id': 'lake_id',
        'swot_wse': 'wse',
        'swot_wse_std': 'wse_std',
        'swot_wse_u': 'wse_u',
        'swot_xtrk_dist': 'xtrk_dist',
        'swot_xovr_cal_q': 'xovr_cal_q',
        'swot_quality_f': 'quality_f',
        'swot_ice_clim_f': 'ice_clim_f',
        'swot_partial_f': 'partial_f',
        'swot_wsa': 'area_total',
        'swot_crid': 'crid',
        'swot_pass_id': 'pass_id',
        'swot_cycle_id': 'cycle_id'
    }
    
    # Create new dataframe with mapped columns
    df_mapped = df.copy()
    
    # Rename columns
    for old_col, new_col in column_mapping.items():
        if old_col in df_mapped.columns:
            df_mapped = df_mapped.rename(columns={old_col: new_col})
    
    # Convert date to datetime and add datetime column expected by customized_functions
    if 'time' in df_mapped.columns:
        df_mapped['time'] = pd.to_datetime(df_mapped['time'])
        df_mapped['datetime'] = df_mapped['time']  # Add datetime column for compatibility
    
    # Add required columns for the adaptive filter
    # Create crid_scenario based on CRID values
    if 'crid' in df_mapped.columns:
        df_mapped['crid_scenario'] = df_mapped['crid'].apply(
            lambda x: 'PIC2_or_PID0' if x in ['PIC2', 'PID0'] else 'early_versions'
        )
    else:
        df_mapped['crid_scenario'] = 'early_versions'
    
    # Create ice_condition based on ice_clim_f
    if 'ice_clim_f' in df_mapped.columns:
        df_mapped['ice_condition'] = df_mapped['ice_clim_f'].apply(
            lambda x: 'ice-covered' if x >= 2 else 'ice-free'
        )
    else:
        df_mapped['ice_condition'] = 'ice-free'
    
    # Add index_col for filter tracking
    df_mapped['index_col'] = df_mapped.index
    
    # Filter to only SWOT observations (non-null WSE)
    if 'wse' in df_mapped.columns:
        df_mapped = df_mapped.dropna(subset=['wse'])
    
    return df_mapped


def swot_adaptive_lakeSP(df, verbose=False, **kwargs):
    """
    Apply adaptive LakeSP filter to benchmark_daily format data
    
    This function adapts the LakeSP v9 adaptive filter for use with 
    benchmark_daily time series format.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame in benchmark_daily format for a single lake
    verbose : bool
        If True, print filtering details
    **kwargs : dict
        Additional parameters for the adaptive filter
        
    Returns:
    --------
    pandas.Series : Boolean mask indicating which observations to keep
    
    Example:
    --------
    >>> df = pd.read_csv('data/timeseries/benchmark_daily/7420077053_daily.csv')
    >>> mask = swot_adaptive_lakeSP(df)
    >>> df_filtered = df[mask]
    """
    
    # Check if we have SWOT data
    if df['swot_wse'].isna().all():
        if verbose:
            print("No SWOT data available, returning empty mask")
        return pd.Series(False, index=df.index)
    
    # Prepare data for adaptive filter
    df_prepared = prepare_data_for_adaptive_filter(df)
    
    # Define conservative subset for threshold calibration as SQL string
    # Similar to the original conservative_SQL but adapted for our column names
    conservative_SQL = (
        '(xovr_cal_q < 1) & ('
        '  (quality_f < 1) '
        '  | ( (quality_f == 1) & ((crid == "PIC2") | (crid == "PID0")) )'
        ')'
    )
    
    # Check if we have enough conservative data for calibration
    try:
        conservative_data = df_prepared.query(conservative_SQL)
    except Exception as e:
        if verbose:
            print(f"Conservative SQL query failed: {e}")
            print("Available columns:", df_prepared.columns.tolist())
        # Use a simpler conservative selection
        conservative_data = df_prepared[
            (df_prepared['xovr_cal_q'] == 0) & 
            (df_prepared['quality_f'] == 0)
        ]
    
    if len(conservative_data) < 5:
        if verbose:
            print(f"Insufficient high-quality data for adaptive filtering: {len(conservative_data)} observations")
            print("Skipping adaptive filter - need at least 5 high-quality observations")
        return pd.Series(False, index=df.index)
    
    try:
        # Calibrate heuristic thresholds
        # Grouping parameters can be adjusted based on your needs
        df_heuristic_thresholds = calibrate_heuristic_thresholds(
            df_prepared, 
            conservative_SQL,
            by_crid_scenario=[False, False, False],  # [wse_std, wse_u, xtrk_dist]
            by_pass_id=[False, False, True],         # Pass grouping mainly for xtrk_dist
            by_ice=[True, True, True]                # Ice condition grouping
        )
        
        if verbose:
            print("Calibrated thresholds:")
            print(df_heuristic_thresholds)
        
        # Apply the adaptive filter
        df_filtered, n_iterations, filter_status = apply_customized_filter(
            df_prepared,
            df_heuristic_thresholds,
            
            # Threshold bounds
            wse_std_threshold_bounds=[0, 3],
            wse_u_threshold_bounds=[0, 0.5],
            xtrk_dist_threshold_bounds=[0, 75000],
            
            # Ice overrides
            wse_std_ice_min=3,
            wse_u_ice_min=0.1,
            
            # Temporal constraints
            allow_major_gap='yes',
            max_temporal_gap=95,
            min_temporal_range=365,
            
            # Ice rules
            rules_for_ice_free_data=['ice-free', 'ice-free', 'not apply'],
            rules_for_ice_covered_data=['ice-free', 'ice-free', 'not apply'],
            
            # Filter settings
            apply_low_pass_filter='yes',
            evaluating_at_full_data='no',
            r2_filter='yes',
            filter_type='savgol',
            z_score_thresholds=[2.576, 3.5],
            maximum_residual_spreads=[0.08, 0.06],
            show_filtering_evolution='no',
            
            **kwargs
        )
        
        if verbose:
            print(f"Filter status: {filter_status}")
            print(f"Iterations: {n_iterations}")
            print(f"Retained {len(df_filtered)}/{len(df_prepared)} observations")
        
        # Create boolean mask for original dataframe
        # Match based on time and WSE values
        mask = pd.Series(False, index=df.index)
        
        if not df_filtered.empty:
            # Find matching rows in original dataframe
            for _, row in df_filtered.iterrows():
                matching_rows = (
                    (df['date'] == row['time'].strftime('%Y-%m-%d')) &
                    (np.abs(df['swot_wse'] - row['wse']) < 1e-6)
                )
                mask = mask | matching_rows
        
        return mask
        
    except Exception as e:
        if verbose:
            print(f"Adaptive filter failed: {e}")
            print("Falling back to standard filter")
        
        # Return empty mask if adaptive filter fails
        if verbose:
            print("Adaptive filter completely failed - returning no observations")
        return pd.Series(False, index=df.index)


def apply_adaptive_filter_to_lake(lake_file_path, verbose=False, **kwargs):
    """
    Apply adaptive filter to a single lake CSV file
    
    Parameters:
    -----------
    lake_file_path : str
        Path to the lake CSV file in benchmark_daily format
    verbose : bool
        If True, print filtering details
    **kwargs : dict
        Additional parameters for the adaptive filter
        
    Returns:
    --------
    tuple : (df_filtered, df_original, mask)
        - df_filtered: Filtered dataframe
        - df_original: Original dataframe 
        - mask: Boolean mask used for filtering
        
    Example:
    --------
    >>> df_filt, df_orig, mask = apply_adaptive_filter_to_lake(
    ...     'data/timeseries/benchmark_daily/7420077053_daily.csv'
    ... )
    """
    # Load data with proper dtypes for ID columns
    df = pd.read_csv(lake_file_path, dtype={'swot_lake_id': str, 'gage_id': str})
    
    # Apply adaptive filter
    mask = swot_adaptive_lakeSP(df, verbose=verbose, **kwargs)
    
    # Create filtered dataframe
    df_filtered = df[mask].copy()
    
    if verbose:
        total_obs = len(df)
        swot_obs = df['swot_wse'].notna().sum()
        filtered_obs = mask.sum()
        
        print(f"\nFiltering Results:")
        print(f"Total observations: {total_obs}")
        print(f"SWOT observations: {swot_obs}")
        print(f"Filtered observations: {filtered_obs}")
        if swot_obs > 0:
            print(f"Retention rate: {filtered_obs/swot_obs:.1%}")
    
    return df_filtered, df, mask


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        lake_file = sys.argv[1]
    else:
        lake_file = "data/timeseries/benchmark_daily/7420077053_daily.csv"
    
    print(f"Testing adaptive filter on {lake_file}")
    df_filtered, df_original, mask = apply_adaptive_filter_to_lake(lake_file, verbose=True)
    
    # Show some results
    if mask.any():
        print("\nFirst few filtered SWOT observations:")
        swot_data = df_filtered[df_filtered['swot_wse'].notna()]
        if not swot_data.empty:
            print(swot_data[['date', 'swot_wse', 'swot_wse_std', 'swot_wse_u']].head())