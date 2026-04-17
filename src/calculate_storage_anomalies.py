#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate storage anomalies using elevation-area relationships for benchmark lakes.

This script processes benchmark_daily CSV files and adds 18 columns:
1. variable_area: 0 if either elevation-area relationship is significant, 1 otherwise
2. storage_anomaly: Anomaly from median storage on SWOT coincident dates
3-18. Storage anomalies for 4 models x 2 filters x 2 temporal approaches (16 variants):
   - Models: swot, swots2, s2, static
   - Filters: opt (optimal threshold 0.283), filt (adaptive filter)
   - Temporal: dis (discrete), con (continuous with ice-aware interpolation)
   
Column naming: {model}_{filter}_{temporal} (e.g., swot_opt_dis, s2_filt_con)
"""
import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path
from scipy import stats
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Project root is one level up from src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)

def calculate_storage_from_area_relationship(wse_values, area_func, reference_wse=None):
    """
    Calculate water storage by integrating area with respect to elevation.
    
    Parameters:
    -----------
    wse_values : array-like
        Water surface elevations at which to calculate storage
    area_func : function
        Function that predicts area from wse (should return values in km²)
    reference_wse : float, optional
        Reference elevation (minimum elevation as default)
        
    Returns:
    --------
    storage : array
        Storage volume at each wse value (in acre-feet)
    """
    wse_values = np.asarray(wse_values)
    
    if reference_wse is None:
        reference_wse = np.nanmin(wse_values)
    
    storage = np.zeros_like(wse_values, dtype=float)
    
    for i, h in enumerate(wse_values):
        # Skip NaN values
        if np.isnan(h):
            storage[i] = np.nan
            continue
            
        # Calculate storage by integrating area with respect to elevation
        if h <= reference_wse:
            storage[i] = 0
        else:
            # Create integration points
            elev_points = np.linspace(reference_wse, h, 100)
            # Get area in km² - convert final result to acre-feet
            area_points_km2 = area_func(elev_points)
            
            # Ensure area_points_km2 is the right shape
            if np.isscalar(area_points_km2):
                area_points_km2 = np.full_like(elev_points, area_points_km2)
            elif len(area_points_km2) != len(elev_points):
                # If shapes don't match, skip this calculation
                storage[i] = np.nan
                continue
            
            # Calculate storage using trapezoidal integration
            # Integration of km² * m = km²⋅m
            # Convert km²⋅m to m³, then to acre-feet
            storage_km2_m = np.trapz(area_points_km2, elev_points)  # This gives km²⋅m
            storage_m3 = storage_km2_m * 1e6  # Convert km²⋅m to m³ (1 km² = 1e6 m²)
            storage[i] = storage_m3 / 1233.48  # Convert m³ to acre-feet
    
    return storage

def interpolate_wse_to_ice_free_dates(df, filter_type='opt'):
    """
    Interpolate WSE values to all ice-free dates for continuous storage calculations.
    
    Parameters:
    -----------
    df : DataFrame
        Lake data
    filter_type : str
        'opt' for optimal threshold (0.283), 'filt' for adaptive filter
        
    Returns:
    --------
    dict with interpolated_wse_values and valid_dates, or None if insufficient data
    """
    # Apply appropriate filter - NO FALLBACKS
    if filter_type == 'opt':
        if 'swot_wse_abs_error' not in df.columns:
            print(f"    Warning: Cannot apply optimal filter - swot_wse_abs_error column missing")
            return None
        filter_mask = ((df['date'] > '2023-07-21') & (df['swot_wse_abs_error'] < 0.283))
    elif filter_type == 'filt':
        if 'adaptive_filter' not in df.columns:
            print(f"    Warning: Cannot apply adaptive filter - adaptive_filter column missing")
            return None
        filter_mask = ((df['date'] > '2023-07-21') & (df['adaptive_filter'] == 0))
    else:
        print(f"    Error: Unknown filter type: {filter_type}")
        return None
    
    # Ice-aware filtering
    ice_available = 'ice' in df.columns
    if ice_available:
        df_ice_free = df[df['ice'] == 0].copy()
        df_target_ice_free = df[filter_mask & (df['ice'] == 0)].copy()
    else:
        df_ice_free = df.copy()
        df_target_ice_free = df[filter_mask].copy()
    
    if len(df_target_ice_free) < 2:
        return None
    
    # Get target WSE data for interpolation
    target_dates_with_data = df_target_ice_free[['date', 'swot_wse_anomaly']].dropna()
    
    if len(target_dates_with_data) < 2:
        return None
    
    # Get all ice-free dates
    all_ice_free_dates = df_ice_free[['date']].dropna()
    
    if len(all_ice_free_dates) < 2:
        return None
    
    # Only interpolate to dates between first and last valid SWOT observation
    date_range_mask = ((all_ice_free_dates['date'] >= target_dates_with_data['date'].min()) &
                      (all_ice_free_dates['date'] <= target_dates_with_data['date'].max()))
    
    dates_to_interpolate = all_ice_free_dates[date_range_mask].copy()
    
    if len(dates_to_interpolate) < 2:
        return None
    
    try:
        # Convert dates to datetime then numeric for interpolation
        target_dates_with_data = target_dates_with_data.copy()
        dates_to_interpolate = dates_to_interpolate.copy()
        target_dates_with_data['date'] = pd.to_datetime(target_dates_with_data['date'])
        dates_to_interpolate['date'] = pd.to_datetime(dates_to_interpolate['date'])
        
        swot_dates_numeric = target_dates_with_data['date'].astype(np.int64)
        interp_dates_numeric = dates_to_interpolate['date'].astype(np.int64)
        
        # Interpolate SWOT WSE anomaly values
        anomaly_interp_func = interp1d(swot_dates_numeric, target_dates_with_data['swot_wse_anomaly'], 
                                     kind='linear', bounds_error=False, fill_value=np.nan)
        
        interpolated_wse_anomaly = anomaly_interp_func(interp_dates_numeric)
        
        # Remove NaN values
        valid_interp_mask = ~np.isnan(interpolated_wse_anomaly)
        if valid_interp_mask.sum() < 2:
            return None
        
        valid_indices = np.where(valid_interp_mask)[0]
        
        return {
            'interpolated_wse_values': interpolated_wse_anomaly[valid_interp_mask],
            'valid_dates': dates_to_interpolate.iloc[valid_indices]['date'].values,
            'target_dates_with_data': target_dates_with_data
        }
        
    except Exception as e:
        print(f"    WSE interpolation failed: {e}")
        return None

def interpolate_s2_areas_to_ice_free_dates(df):
    """
    Interpolate S2 area values to all ice-free dates for continuous S2 storage calculations.
    Maximum interpolation gap: 30 days. Gaps longer than 30 days are left as NaN.
    
    Parameters:
    -----------
    df : DataFrame
        Lake data
        
    Returns:
    --------
    dict with interpolated_s2_areas and valid_dates, or None if insufficient data
    """
    # Apply S2 quality filter
    s2_filter = ((df['s2_coverage'] > 99) & (df['ice'] == 0))
    
    # Ice-aware filtering
    ice_available = 'ice' in df.columns
    if ice_available:
        df_ice_free = df[df['ice'] == 0].copy()
        df_s2_valid = df[s2_filter].copy()
    else:
        df_ice_free = df.copy()
        df_s2_valid = df[s2_filter].copy()
    
    if len(df_s2_valid) < 2:
        return None
    
    # Get S2 data for interpolation
    s2_data_with_areas = df_s2_valid[['date', 's2_wsa']].dropna()
    
    if len(s2_data_with_areas) < 2:
        return None
    
    # Get all ice-free dates
    all_ice_free_dates = df_ice_free[['date']].dropna()
    
    if len(all_ice_free_dates) < 2:
        return None
    
    # Only interpolate to dates between first and last valid S2 observation
    date_range_mask = ((all_ice_free_dates['date'] >= s2_data_with_areas['date'].min()) &
                      (all_ice_free_dates['date'] <= s2_data_with_areas['date'].max()))
    
    dates_to_interpolate = all_ice_free_dates[date_range_mask].copy()
    
    if len(dates_to_interpolate) < 2:
        return None
    
    try:
        # Convert dates to datetime then numeric for interpolation
        s2_data_with_areas = s2_data_with_areas.copy()
        dates_to_interpolate = dates_to_interpolate.copy()
        s2_data_with_areas['date'] = pd.to_datetime(s2_data_with_areas['date'])
        dates_to_interpolate['date'] = pd.to_datetime(dates_to_interpolate['date'])
        
        s2_dates_numeric = s2_data_with_areas['date'].astype(np.int64)
        interp_dates_numeric = dates_to_interpolate['date'].astype(np.int64)
        
        # Check for gaps longer than 30 days
        s2_dates_sorted = s2_data_with_areas.sort_values('date')
        date_diffs = s2_dates_sorted['date'].diff().dt.days
        
        # If any gap is longer than 30 days, we need to handle it
        long_gaps = date_diffs > 30
        if long_gaps.any():
            # Find valid interpolation ranges (within 30 days of S2 observations)
            valid_interp_mask = np.zeros(len(dates_to_interpolate), dtype=bool)
            
            for _, s2_row in s2_dates_sorted.iterrows():
                s2_date = s2_row['date']
                # Mark dates within 30 days of this S2 observation as valid for interpolation
                time_diffs = abs((dates_to_interpolate['date'] - s2_date).dt.days)
                within_30days = time_diffs <= 30
                valid_interp_mask |= within_30days
            
            # Only interpolate to dates that are within 30 days of some S2 observation
            dates_to_interpolate = dates_to_interpolate[valid_interp_mask]
            interp_dates_numeric = interp_dates_numeric[valid_interp_mask]
            
            if len(dates_to_interpolate) == 0:
                return None
        
        # Interpolate S2 area values
        area_interp_func = interp1d(s2_dates_numeric, s2_data_with_areas['s2_wsa'], 
                                   kind='linear', bounds_error=False, fill_value=np.nan)
        
        interpolated_s2_areas = area_interp_func(interp_dates_numeric)
        
        # Remove NaN values
        valid_interp_mask = ~np.isnan(interpolated_s2_areas)
        if valid_interp_mask.sum() < 2:
            return None
        
        valid_indices = np.where(valid_interp_mask)[0]
        
        return {
            'interpolated_s2_areas': interpolated_s2_areas[valid_interp_mask],
            'valid_dates': dates_to_interpolate.iloc[valid_indices]['date'].values,
            's2_data_with_areas': s2_data_with_areas
        }
        
    except Exception as e:
        print(f"    S2 area interpolation failed: {e}")
        return None

def build_area_elevation_relationships(df, filter_type='opt'):
    """
    Build elevation-area relationships using specified filtering strategy.
    NO FALLBACKS - returns None functions if data is missing.
    
    Parameters:
    -----------
    df : DataFrame
        Lake data
    filter_type : str
        'opt' for optimal threshold (0.283), 'filt' for adaptive filter
    
    Returns:
    --------
    dict with keys: swot_func, swot_significant, s2_func, s2_significant
    """
    results = {
        'swot_func': None, 'swot_significant': False,
        's2_func': None, 's2_significant': False
    }
    
    # Base SWOT WSE filter - NO FALLBACKS
    if filter_type == 'opt':
        if 'swot_wse_abs_error' not in df.columns:
            print(f"    Cannot build {filter_type} relationships - swot_wse_abs_error missing")
            return results
        swot_wse_filter = ((df['date'] > '2023-07-21') & (df['swot_wse_abs_error'] < 0.283))
    elif filter_type == 'filt':
        if 'adaptive_filter' not in df.columns:
            print(f"    Cannot build {filter_type} relationships - adaptive_filter missing")
            return results
        swot_wse_filter = ((df['date'] > '2023-07-21') & (df['adaptive_filter'] == 0))
    else:
        print(f"    Error: Unknown filter type: {filter_type}")
        return results
    
    # SWOT relationship: SWOT WSE vs SWOT WSA
    # Only use complete observations (partial_f == 0)
    swot_mask = swot_wse_filter & (df['swot_partial_f'] == 0)
    df_swot = df[swot_mask].copy()
    df_swot = df_swot.dropna(subset=['swot_wsa', 'swot_wse_anomaly'])
    
    if len(df_swot) > 2:
        swot_area = df_swot['swot_wsa'].values
        swot_wse = df_swot['swot_wse_anomaly'].values
        
        # Calculate significance
        swot_r, swot_p_value = stats.pearsonr(swot_area, swot_wse)
        results['swot_significant'] = swot_p_value < 0.05
        
        # Fit both linear and quadratic models
        # Linear model
        linear_model = LinearRegression()
        linear_model.fit(swot_wse.reshape(-1, 1), swot_area)
        linear_pred = linear_model.predict(swot_wse.reshape(-1, 1))
        linear_rmse = np.sqrt(np.mean((swot_area - linear_pred)**2))
        
        # Quadratic model
        swot_poly = Pipeline([
            ('poly', PolynomialFeatures(degree=2)),
            ('linear', LinearRegression())
        ])
        swot_poly.fit(swot_wse.reshape(-1, 1), swot_area)
        quad_pred = swot_poly.predict(swot_wse.reshape(-1, 1))
        quad_rmse = np.sqrt(np.mean((swot_area - quad_pred)**2))
        
        # Select model with lower RMSE
        if linear_rmse <= quad_rmse:
            results['swot_func'] = lambda x: linear_model.predict(np.asarray(x).reshape(-1, 1))
            results['swot_model_type'] = 'linear'
        else:
            results['swot_func'] = lambda x: swot_poly.predict(np.asarray(x).reshape(-1, 1))
            results['swot_model_type'] = 'quadratic'
    
    # S2 relationship: SWOT WSE vs S2 WSA (matched within 3 days)
    # Get S2 observations that pass coverage filter
    s2_filter = ((df['s2_coverage'] > 99) & (df['ice'] == 0))
    df_s2_valid = df[s2_filter].dropna(subset=['s2_wsa']).copy()
    
    # Get SWOT WSE observations that pass base filter
    df_swot_wse_valid = df[swot_wse_filter].dropna(subset=['swot_wse_anomaly']).copy()
    
    if len(df_s2_valid) > 0 and len(df_swot_wse_valid) > 0:
        # Convert date columns to datetime
        df_s2_valid['date'] = pd.to_datetime(df_s2_valid['date'])
        df_swot_wse_valid['date'] = pd.to_datetime(df_swot_wse_valid['date'])
        
        # Match S2 observations to SWOT WSE observations within 3 days
        df_s2_matched = []
        
        for idx, swot_row in df_swot_wse_valid.iterrows():
            swot_date = swot_row['date']
            
            # Calculate time differences
            time_diffs = abs((df_s2_valid['date'] - swot_date).dt.days)
            
            # Find S2 observations within 3 days
            within_3days = time_diffs <= 3
            
            if within_3days.any():
                # Use the closest S2 observation within 3 days
                closest_idx = time_diffs[within_3days].idxmin()
                s2_value = df_s2_valid.loc[closest_idx, 's2_wsa']
            else:
                # No S2 observation within 3 days - exclude this observation
                s2_value = np.nan
            
            # Create matched row
            matched_row = swot_row.copy()
            matched_row['s2_matched'] = s2_value
            df_s2_matched.append(matched_row)
        
        if len(df_s2_matched) > 2:
            df_s2 = pd.DataFrame(df_s2_matched)
            # Filter out rows where S2 matching failed (NaN values)
            df_s2 = df_s2.dropna(subset=['s2_matched'])
            
            if len(df_s2) > 2:
                s2_area = df_s2['s2_matched'].values
                s2_wse = df_s2['swot_wse_anomaly'].values
                
                # Calculate significance
                s2_r, s2_p_value = stats.pearsonr(s2_area, s2_wse)
                results['s2_significant'] = s2_p_value < 0.05
                
                # Fit both linear and quadratic models
                # Linear model
                linear_model = LinearRegression()
                linear_model.fit(s2_wse.reshape(-1, 1), s2_area)
                linear_pred = linear_model.predict(s2_wse.reshape(-1, 1))
                linear_rmse = np.sqrt(np.mean((s2_area - linear_pred)**2))
                
                # Quadratic model
                s2_poly = Pipeline([
                    ('poly', PolynomialFeatures(degree=2)),
                    ('linear', LinearRegression())
                ])
                s2_poly.fit(s2_wse.reshape(-1, 1), s2_area)
                quad_pred = s2_poly.predict(s2_wse.reshape(-1, 1))
                quad_rmse = np.sqrt(np.mean((s2_area - quad_pred)**2))
                
                # Select model with lower RMSE
                if linear_rmse <= quad_rmse:
                    results['s2_func'] = lambda x: linear_model.predict(np.asarray(x).reshape(-1, 1))
                    results['s2_model_type'] = 'linear'
                else:
                    results['s2_func'] = lambda x: s2_poly.predict(np.asarray(x).reshape(-1, 1))
                    results['s2_model_type'] = 'quadratic'
    
    return results

def calculate_storage_anomalies_for_model(df, relationships, model_type, filter_type, temporal_type, reference_median_acre_feet=None):
    """
    Calculate storage anomalies for a specific model, filter, and temporal combination.
    Uses provided reference median for consistent anomaly calculation.
    
    Parameters:
    -----------
    df : DataFrame
        Lake data
    relationships : dict
        Elevation-area relationship functions
    model_type : str
        'swot', 'swots2', 's2', or 'static'
    filter_type : str
        'opt' or 'filt'
    temporal_type : str
        'dis' (discrete) or 'con' (continuous)
    reference_median_acre_feet : float or None
        Reference median from discrete storage for consistent anomaly calculation
        
    Returns:
    --------
    array: storage_anomaly_values (storage - reference_median)
    """
    # Initialize output array
    storage_anomalies = np.full(len(df), np.nan)
    
    # Apply appropriate filter - NO FALLBACKS
    if filter_type == 'opt':
        if 'swot_wse_abs_error' not in df.columns:
            return storage_anomalies
        swot_wse_filter = ((df['date'] > '2023-07-21') & (df['swot_wse_abs_error'] < 0.283))
    elif filter_type == 'filt':
        if 'adaptive_filter' not in df.columns:
            return storage_anomalies
        swot_wse_filter = ((df['date'] > '2023-07-21') & (df['adaptive_filter'] == 0))
    else:
        return storage_anomalies
    
    # Get appropriate elevation-area relationship
    if model_type == 'swot':
        area_func = relationships['swot_func']
        if area_func is None:
            return storage_anomalies
    elif model_type in ['swots2', 's2']:
        area_func = relationships['s2_func']
        if area_func is None:
            return storage_anomalies
    elif model_type == 'static':
        # Static model uses reference area
        if 'swot_p_ref_area' not in df.columns:
            return storage_anomalies
        static_ref_area = df['swot_p_ref_area'].dropna()
        if len(static_ref_area) == 0:
            return storage_anomalies
        reference_area_km2 = static_ref_area.iloc[0]
    
    # Get reference WSE
    valid_wse = df[swot_wse_filter]['swot_wse_anomaly'].dropna()
    if len(valid_wse) == 0:
        return storage_anomalies
    reference_wse = valid_wse.min()
    
    if temporal_type == 'dis':  # Discrete
        if model_type == 'swot':
            # Use SWOT WSE directly, but only for observations that pass the filter
            mask = swot_wse_filter & df['swot_wse_anomaly'].notna()
            if mask.any():
                wse_values = df.loc[mask, 'swot_wse_anomaly'].values
                storage_values = calculate_storage_from_area_relationship(
                    wse_values, area_func, reference_wse)
                storage_anomalies[mask] = storage_values
        
        elif model_type == 'swots2':
            # Use SWOT WSE with S2 relationship, but only for observations that pass the filter
            mask = swot_wse_filter & df['swot_wse_anomaly'].notna()
            if mask.any():
                wse_values = df.loc[mask, 'swot_wse_anomaly'].values
                storage_values = calculate_storage_from_area_relationship(
                    wse_values, area_func, reference_wse)
                storage_anomalies[mask] = storage_values
        
        elif model_type == 's2':
            # Use S2 area to predict WSE, then calculate storage
            # S2 uses its own quality criteria - not aligned with SWOT temporal filtering
            s2_filter = ((df['s2_coverage'] > 99) & (df['ice'] == 0))
            df_s2_valid = df[s2_filter].dropna(subset=['s2_wsa']).copy()
            df_swot_wse_valid = df[swot_wse_filter].dropna(subset=['swot_wse_anomaly']).copy()
            
            if len(df_s2_valid) > 0 and len(df_swot_wse_valid) > 0:
                # Build inverse relationship: S2 area -> WSE
                inverse_func = build_s2_area_to_wse_function(df_s2_valid, df_swot_wse_valid)
                if inverse_func is not None:
                    # Use S2's own quality filter for S2 observations
                    s2_mask = s2_filter & df['s2_wsa'].notna()
                    if s2_mask.any():
                        s2_areas = df.loc[s2_mask, 's2_wsa'].values
                        predicted_wse = inverse_func(s2_areas)
                        storage_values = calculate_storage_from_area_relationship(
                            predicted_wse, area_func, reference_wse)
                        storage_anomalies[s2_mask] = storage_values
        
        elif model_type == 'static':
            # Static area calculation, but only for observations that pass the filter
            mask = swot_wse_filter & df['swot_wse_anomaly'].notna()
            if mask.any():
                wse_values = df.loc[mask, 'swot_wse_anomaly'].values
                height_changes = wse_values - reference_wse
                volumes_km2_m = reference_area_km2 * height_changes
                volumes_m3 = volumes_km2_m * 1e6
                volumes_acre_feet = volumes_m3 / 1233.48
                storage_anomalies[mask] = volumes_acre_feet
    
    else:  # Continuous (temporal_type == 'con')
        # FIRST: Calculate discrete values where we have actual observations that pass the filter
        # This ensures continuous matches discrete on dates with real data
        if model_type == 'swot' or model_type == 'swots2':
            # Use SWOT WSE directly where available and passes filter
            mask = swot_wse_filter & df['swot_wse_anomaly'].notna()
            if mask.any():
                wse_values = df.loc[mask, 'swot_wse_anomaly'].values
                storage_values = calculate_storage_from_area_relationship(
                    wse_values, area_func, reference_wse)
                storage_anomalies[mask] = storage_values
        elif model_type == 's2':
            # Use S2 area to predict WSE, then calculate storage
            # For continuous S2: use discrete observations + interpolated S2 areas
            s2_filter = ((df['s2_coverage'] > 99) & (df['ice'] == 0))
            df_s2_valid = df[s2_filter].dropna(subset=['s2_wsa']).copy()
            df_swot_wse_valid = df[swot_wse_filter].dropna(subset=['swot_wse_anomaly']).copy()
            
            if len(df_s2_valid) > 0 and len(df_swot_wse_valid) > 0:
                # Build inverse relationship: S2 area -> WSE
                inverse_func = build_s2_area_to_wse_function(df_s2_valid, df_swot_wse_valid)
                if inverse_func is not None:
                    # FIRST: Calculate discrete values for actual S2 observations
                    s2_mask = s2_filter & df['s2_wsa'].notna()
                    if s2_mask.any():
                        s2_areas = df.loc[s2_mask, 's2_wsa'].values
                        predicted_wse = inverse_func(s2_areas)
                        storage_values = calculate_storage_from_area_relationship(
                            predicted_wse, area_func, reference_wse)
                        storage_anomalies[s2_mask] = storage_values
                    
                    # SECOND: Interpolate S2 areas to fill gaps in ice-free dates
                    s2_interp_result = interpolate_s2_areas_to_ice_free_dates(df)
                    
                    if s2_interp_result is not None:
                        interpolated_s2_areas = s2_interp_result['interpolated_s2_areas']
                        valid_dates = s2_interp_result['valid_dates']
                        
                        # Convert dates back to DataFrame indices for assignment
                        df_copy = df.copy()
                        df_copy['date'] = pd.to_datetime(df_copy['date'])
                        
                        for i, date in enumerate(valid_dates):
                            date_mask = df_copy['date'] == pd.to_datetime(date)
                            if date_mask.any():
                                idx = df_copy[date_mask].index[0]
                                
                                # ONLY fill if we don't already have a value (gap filling only)
                                if np.isnan(storage_anomalies[idx]):
                                    s2_area_val = interpolated_s2_areas[i]
                                    predicted_wse_val = inverse_func([s2_area_val])[0]
                                    storage_val = calculate_storage_from_area_relationship(
                                        [predicted_wse_val], area_func, reference_wse)[0]
                                    storage_anomalies[idx] = storage_val
        elif model_type == 'static':
            # Static area calculation for discrete observations that pass filter
            mask = swot_wse_filter & df['swot_wse_anomaly'].notna()
            if mask.any():
                wse_values = df.loc[mask, 'swot_wse_anomaly'].values
                height_changes = wse_values - reference_wse
                volumes_km2_m = reference_area_km2 * height_changes
                volumes_m3 = volumes_km2_m * 1e6
                volumes_acre_feet = volumes_m3 / 1233.48
                storage_anomalies[mask] = volumes_acre_feet
        
        # SECOND: Interpolate WSE to fill gaps in ice-free dates
        interp_result = interpolate_wse_to_ice_free_dates(df, filter_type)
        
        if interp_result is not None:
            interpolated_wse = interp_result['interpolated_wse_values']
            valid_dates = interp_result['valid_dates']
            
            # Convert dates back to DataFrame indices for assignment
            df_copy = df.copy()
            df_copy['date'] = pd.to_datetime(df_copy['date'])
            
            for i, date in enumerate(valid_dates):
                date_mask = df_copy['date'] == pd.to_datetime(date)
                if date_mask.any():
                    idx = df_copy[date_mask].index[0]
                    
                    # ONLY fill if we don't already have a value (gap filling only)
                    if np.isnan(storage_anomalies[idx]):
                        wse_val = interpolated_wse[i]
                        
                        if model_type == 'static':
                            height_change = wse_val - reference_wse
                            volume_km2_m = reference_area_km2 * height_change
                            volume_m3 = volume_km2_m * 1e6
                            volume_acre_feet = volume_m3 / 1233.48
                            storage_anomalies[idx] = volume_acre_feet
                        else:
                            storage_val = calculate_storage_from_area_relationship(
                                [wse_val], area_func, reference_wse)[0]
                            storage_anomalies[idx] = storage_val
    
    # Apply reference median if provided
    if reference_median_acre_feet is not None:
        storage_anomalies = storage_anomalies - reference_median_acre_feet
    else:
        # No reference median provided - calculate from discrete filtered data
        if temporal_type == 'dis':
            filtered_storage = storage_anomalies[swot_wse_filter]
            valid_filtered = filtered_storage[~np.isnan(filtered_storage)]
            if len(valid_filtered) > 0:
                calculated_median = np.median(valid_filtered)
                storage_anomalies = storage_anomalies - calculated_median
    
    return storage_anomalies

def build_s2_area_to_wse_function(df_s2_valid, df_swot_wse_valid):
    """
    Build inverse relationship: S2 area -> WSE for S2 model calculations.
    """
    # Convert dates for matching
    df_s2_valid = df_s2_valid.copy()
    df_swot_wse_valid = df_swot_wse_valid.copy()
    df_s2_valid['date'] = pd.to_datetime(df_s2_valid['date'])
    df_swot_wse_valid['date'] = pd.to_datetime(df_swot_wse_valid['date'])
    
    # Build matched dataset
    matched_data = []
    for idx, swot_row in df_swot_wse_valid.iterrows():
        swot_date = swot_row['date']
        time_diffs = abs((df_s2_valid['date'] - swot_date).dt.days)
        
        within_3days = time_diffs <= 3
        if within_3days.any():
            closest_idx = time_diffs[within_3days].idxmin()
            s2_area_value = df_s2_valid.loc[closest_idx, 's2_wsa']
            matched_data.append({
                's2_area': s2_area_value,
                'wse': swot_row['swot_wse_anomaly']
            })
    
    if len(matched_data) < 3:
        return None
    
    matched_df = pd.DataFrame(matched_data).dropna()
    
    if len(matched_df) < 3:
        return None
    
    try:
        # Build inverse polynomial relationship: S2 area -> WSE
        s2_inv_poly = Pipeline([
            ('poly', PolynomialFeatures(degree=2)),
            ('linear', LinearRegression())
        ])
        s2_inv_poly.fit(matched_df['s2_area'].values.reshape(-1, 1), matched_df['wse'].values)
        return lambda x: s2_inv_poly.predict(np.asarray(x).reshape(-1, 1))
    except:
        return None

def process_lake_file(csv_file):
    """Process a single lake CSV file and add the required columns."""
    
    try:
        gage_id = os.path.basename(csv_file).replace('_daily.csv', '')
        print(f"Processing {gage_id}...")
        
        # Load data with proper dtypes
        df = pd.read_csv(csv_file, dtype={'gage_id': str, 'swot_lake_id': str})
        
        # Check for required columns
        required_cols = ['swot_wsa', 'swot_wse_anomaly', 's2_wsa', 
                        'ice', 'swot_partial_f', 'swot_ice_clim_f', 's2_coverage', 'swot_xovr_cal_q', 
                        'swot_wse_std', 'swot_wse_u', 'date']
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"  Skipping {gage_id}: missing required columns {missing_cols}")
            return
        
        # Check what filtering capabilities we have
        has_optimal = 'swot_wse_abs_error' in df.columns
        has_adaptive = 'adaptive_filter' in df.columns
        
        print(f"  {gage_id} - Optimal filter: {has_optimal}, Adaptive filter: {has_adaptive}")
        
        # Build elevation-area relationships for available filters
        relationships_opt = None
        relationships_filt = None
        
        if has_optimal:
            relationships_opt = build_area_elevation_relationships(df, filter_type='opt')
            if relationships_opt:
                swot_model = relationships_opt.get('swot_model_type', 'N/A')
                s2_model = relationships_opt.get('s2_model_type', 'N/A')
                print(f"    Optimal - SWOT model: {swot_model}, S2 model: {s2_model}")
        
        if has_adaptive:
            relationships_filt = build_area_elevation_relationships(df, filter_type='filt')
            if relationships_filt:
                swot_model = relationships_filt.get('swot_model_type', 'N/A')
                s2_model = relationships_filt.get('s2_model_type', 'N/A')
                print(f"    Adaptive - SWOT model: {swot_model}, S2 model: {s2_model}")
        
        # Calculate variable_area column (using optimal if available)
        if relationships_opt:
            either_significant = relationships_opt['swot_significant'] or relationships_opt['s2_significant']
            df['variable_area'] = 0 if either_significant else 1
        else:
            df['variable_area'] = 1  # Default to 1 if no relationships available
        
        # Calculate storage_anomaly from gauge storage (if available)
        if 'storage' in df.columns and has_optimal:
            swot_wse_filter = ((df['date'] > '2023-07-21') & (df['swot_wse_abs_error'] < 0.283))
            storage_filter = (df['storage'] >= 0) & (df['stage'] >= 0)
            combined_filter = swot_wse_filter & storage_filter
            
            swot_coincident_storage = df[combined_filter]['storage'].dropna()
            if len(swot_coincident_storage) > 0:
                storage_median = swot_coincident_storage.median()
                storage_anomaly_m3 = df['storage'] - storage_median
                df['storage_anomaly'] = storage_anomaly_m3 / 1233.48
                df.loc[~storage_filter, 'storage_anomaly'] = np.nan
            else:
                df['storage_anomaly'] = np.nan
        else:
            df['storage_anomaly'] = np.nan
        
        # Calculate all storage anomaly variants
        models = ['swot', 'swots2', 's2', 'static']
        filter_types = ['opt', 'filt']
        
        for filter_type in filter_types:
            # Skip if filter not available
            if filter_type == 'opt' and not has_optimal:
                print(f"    Skipping optimal variants - no swot_wse_abs_error column")
                for model in models:
                    df[f'{model}_{filter_type}_dis'] = np.nan
                    df[f'{model}_{filter_type}_con'] = np.nan
                continue
            
            if filter_type == 'filt' and not has_adaptive:
                print(f"    Skipping filtered variants - no adaptive_filter column")
                for model in models:
                    df[f'{model}_{filter_type}_dis'] = np.nan
                    df[f'{model}_{filter_type}_con'] = np.nan
                continue
            
            # Get appropriate relationships
            relationships = relationships_opt if filter_type == 'opt' else relationships_filt
            
            # Calculate gauge storage anomaly for this filter
            if 'storage' in df.columns:
                if filter_type == 'opt':
                    swot_wse_filter = ((df['date'] > '2023-07-21') & (df['swot_wse_abs_error'] < 0.283))
                else:
                    swot_wse_filter = ((df['date'] > '2023-07-21') & (df['adaptive_filter'] == 0))
                
                storage_filter = (df['storage'] >= 0) & (df['stage'] >= 0)
                combined_filter = swot_wse_filter & storage_filter
                swot_coincident_storage = df[combined_filter]['storage'].dropna()
                
                if len(swot_coincident_storage) > 0:
                    storage_median_m3 = swot_coincident_storage.median()
                    storage_anomaly_m3 = df['storage'] - storage_median_m3
                    df[f'storage_anomaly_{filter_type}'] = storage_anomaly_m3 / 1233.48
                    df.loc[~storage_filter, f'storage_anomaly_{filter_type}'] = np.nan
                else:
                    df[f'storage_anomaly_{filter_type}'] = np.nan
            else:
                df[f'storage_anomaly_{filter_type}'] = np.nan
            
            # Calculate discrete storage medians first
            model_discrete_medians = {}
            
            for model in models:
                # Calculate discrete storage
                discrete_storage = calculate_storage_anomalies_for_model(
                    df, relationships, model, filter_type, 'dis', reference_median_acre_feet=0.0)
                
                # Get median from filtered dates
                if filter_type == 'opt':
                    filter_mask = ((df['date'] > '2023-07-21') & (df['swot_wse_abs_error'] < 0.283))
                else:
                    filter_mask = ((df['date'] > '2023-07-21') & (df['adaptive_filter'] == 0))
                
                filtered_discrete = discrete_storage[filter_mask]
                valid_discrete = filtered_discrete[~np.isnan(filtered_discrete)]
                
                if len(valid_discrete) > 0:
                    model_discrete_medians[model] = np.median(valid_discrete)
                else:
                    model_discrete_medians[model] = None
            
            # Calculate all variants using consistent reference medians
            for temporal_type in ['dis', 'con']:
                for model in models:
                    # Use model's discrete median for both dis and con
                    model_reference_median = model_discrete_medians.get(model, None)
                    
                    storage_anomalies = calculate_storage_anomalies_for_model(
                        df, relationships, model, filter_type, temporal_type, 
                        reference_median_acre_feet=model_reference_median)
                    
                    df[f'{model}_{filter_type}_{temporal_type}'] = storage_anomalies
        
        # Save the updated file
        df.to_csv(csv_file, index=False)
        print(f"  Successfully added storage anomaly variants to {gage_id}")
        
        # Return summary information
        return {
            'gage_id': gage_id,
            'has_optimal': has_optimal,
            'has_adaptive': has_adaptive,
            'swot_significant_opt': relationships_opt['swot_significant'] if relationships_opt else None,
            'swot_significant_filt': relationships_filt['swot_significant'] if relationships_filt else None,
            's2_significant_opt': relationships_opt['s2_significant'] if relationships_opt else None,
            's2_significant_filt': relationships_filt['s2_significant'] if relationships_filt else None,
            'swot_model_opt': relationships_opt.get('swot_model_type', None) if relationships_opt else None,
            'swot_model_filt': relationships_filt.get('swot_model_type', None) if relationships_filt else None,
            's2_model_opt': relationships_opt.get('s2_model_type', None) if relationships_opt else None,
            's2_model_filt': relationships_filt.get('s2_model_type', None) if relationships_filt else None,
            'variable_area': df['variable_area'].iloc[0] if 'variable_area' in df.columns else None,
            'has_storage': 'storage' in df.columns
        }
        
    except Exception as e:
        print(f"  Error processing {csv_file}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main execution function."""
    
    # Create output directory
    os.makedirs('experiments/results/storage_anomalies', exist_ok=True)
    
    # Get all benchmark daily CSV files
    csv_files = glob.glob('data/benchmark_timeseries/*_daily.csv')
    csv_files.sort()
    
    print(f"Found {len(csv_files)} benchmark daily files")
    print("="*60)
    
    # Process all files
    results = []
    for csv_file in csv_files:
        result = process_lake_file(csv_file)
        if result:
            results.append(result)
    
    # Create summary
    summary_df = pd.DataFrame(results)
    
    if len(summary_df) > 0:
        # Summary statistics
        print(f"\n{'='*60}")
        print(f"PROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"Successfully processed: {len(summary_df)} lakes")
        print(f"Lakes with optimal filter: {summary_df['has_optimal'].sum()}")
        print(f"Lakes with adaptive filter: {summary_df['has_adaptive'].sum()}")
        print(f"Lakes with storage data: {summary_df['has_storage'].sum()}")
        print(f"\nRelationship Significance:")
        print(f"  SWOT relationships significant (opt): {summary_df['swot_significant_opt'].sum()}")
        print(f"  SWOT relationships significant (filt): {summary_df['swot_significant_filt'].sum()}")
        print(f"  S2 relationships significant (opt): {summary_df['s2_significant_opt'].sum()}")
        print(f"  S2 relationships significant (filt): {summary_df['s2_significant_filt'].sum()}")
        
        print(f"\nElevation-Area Model Selection (SWOT/SWOTS2):")
        print(f"  Optimal filter:")
        swot_linear_opt = (summary_df['swot_model_opt'] == 'linear').sum()
        swot_quad_opt = (summary_df['swot_model_opt'] == 'quadratic').sum()
        print(f"    Linear models: {swot_linear_opt}")
        print(f"    Quadratic models: {swot_quad_opt}")
        
        print(f"  Adaptive filter:")
        swot_linear_filt = (summary_df['swot_model_filt'] == 'linear').sum()
        swot_quad_filt = (summary_df['swot_model_filt'] == 'quadratic').sum()
        print(f"    Linear models: {swot_linear_filt}")
        print(f"    Quadratic models: {swot_quad_filt}")
        
        print(f"\nElevation-Area Model Selection (S2):")
        print(f"  Optimal filter:")
        s2_linear_opt = (summary_df['s2_model_opt'] == 'linear').sum()
        s2_quad_opt = (summary_df['s2_model_opt'] == 'quadratic').sum()
        print(f"    Linear models: {s2_linear_opt}")
        print(f"    Quadratic models: {s2_quad_opt}")
        
        print(f"  Adaptive filter:")
        s2_linear_filt = (summary_df['s2_model_filt'] == 'linear').sum()
        s2_quad_filt = (summary_df['s2_model_filt'] == 'quadratic').sum()
        print(f"    Linear models: {s2_linear_filt}")
        print(f"    Quadratic models: {s2_quad_filt}")
        
        # Save summary
        summary_df.to_csv('experiments/results/storage_anomalies/processing_summary_clean.csv', index=False)
        print(f"\nSaved processing summary to experiments/results/storage_anomalies/processing_summary_clean.csv")
    
if __name__ == "__main__":
    main()