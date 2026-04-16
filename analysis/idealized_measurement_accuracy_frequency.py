import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize_scalar
import os
import glob
from datetime import datetime, timedelta

os.chdir('/Users/ericlevenson/University of Oregon Dropbox/Eric Levenson/SWOT/production')

def load_benchmark_data_with_ice():
    """Load benchmark daily data from individual files that contain ice column"""
    
    # Load problematic SWOT lake IDs to exclude
    try:
        problematic_ids = pd.read_csv('data/benchmark/problematic_swot_lake_ids.csv', dtype={'swot_lake_id': str})
        exclude_ids = set(problematic_ids['swot_lake_id'].values)
        print(f"Loaded {len(exclude_ids)} problematic lake IDs to exclude")
    except Exception as e:
        print(f"Warning: Could not load problematic lake IDs: {e}")
        exclude_ids = set()
    
    # Find all benchmark_daily files
    benchmark_files = glob.glob('data/timeseries/benchmark_daily/*_daily.csv')
    print(f"Found {len(benchmark_files)} benchmark_daily files")
    
    # Load and combine all data
    all_data = []
    files_with_ice = 0
    excluded_files = 0
    
    for file in benchmark_files:
        try:
            # Extract swot_lake_id from filename to check if problematic
            filename = os.path.basename(file)
            swot_lake_id = filename.split('_')[0]
            
            # Skip problematic lakes entirely
            if swot_lake_id in exclude_ids:
                excluded_files += 1
                continue
            
            df = pd.read_csv(file, dtype={'swot_lake_id': str, 'gage_id': str})
            
            # Add swot_lake_id from filename if not in dataframe
            if 'swot_lake_id' not in df.columns:
                df['swot_lake_id'] = swot_lake_id
            
            # Only include files that have ice column
            if 'ice' in df.columns:
                all_data.append(df)
                files_with_ice += 1
        
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    print(f"Loaded {files_with_ice} files with ice column")
    print(f"Excluded {excluded_files} files from problematic lakes")
    
    if not all_data:
        print("No files with ice column found! Falling back to combined file.")
        return pd.read_csv('data/timeseries/benchmark_daily_combined.csv', 
                          dtype={'swot_lake_id': str, 'swot_lake_id_y': str, 'gage_id': str})
    
    # Combine all data
    df_combined = pd.concat(all_data, ignore_index=True)
    print(f"Combined data shape: {df_combined.shape}")
    
    return df_combined

def load_lake_size_mapping():
    """Load lake size mapping from gage_swot_mapping.csv"""
    try:
        lake_mapping = pd.read_csv('data/benchmark/gage_swot_mapping.csv', 
                                   dtype={'swot_lake_id': str, 'gage_id': str})
        lake_size_dict = dict(zip(lake_mapping['swot_lake_id'].astype(str), 
                                  lake_mapping['ref_area_swot']))
        print(f"Loaded size data for {len(lake_size_dict)} lakes")
        return lake_size_dict
    except Exception as e:
        print(f"Warning: Could not load lake size mapping: {e}")
        return {}

def add_lake_size_categories(df, lake_size_dict):
    """Add lake size categories to dataframe"""
    def get_lake_size_category(lake_id):
        ref_area = lake_size_dict.get(str(lake_id), None)
        if ref_area is not None:
            return 'Small (<1 km²)' if ref_area < 1.0 else 'Large (≥1 km²)'
        else:
            return 'Unknown size'
    
    df['lake_size_category'] = df['swot_lake_id'].apply(get_lake_size_category)
    return df

def filter_lakes_by_observation_count(df, min_observations=5):
    """Filter out lakes with less than minimum SWOT observations"""
    
    print(f"\nFiltering lakes with <{min_observations} SWOT observations...")
    
    # Count SWOT observations per lake (non-null swot_wse_abs_error)
    lake_obs_counts = df.dropna(subset=['swot_wse_abs_error']).groupby('swot_lake_id').size()
    valid_lakes = lake_obs_counts[lake_obs_counts >= min_observations].index
    
    initial_lakes = df['swot_lake_id'].nunique()
    initial_obs = len(df)
    
    df_filtered = df[df['swot_lake_id'].isin(valid_lakes)].copy()
    
    final_lakes = df_filtered['swot_lake_id'].nunique()
    final_obs = len(df_filtered)
    
    print(f"Lakes: {initial_lakes} → {final_lakes} (removed {initial_lakes - final_lakes})")
    print(f"Observations: {initial_obs} → {final_obs} (removed {initial_obs - final_obs})")
    
    return df_filtered

def calculate_wsa_percentage_errors(df):
    """Calculate WSA percentage errors for analysis"""
    
    df = df.copy()
    
    # Filter for observations where both WSA values are available and wsa > 0
    valid_mask = (df['swot_wsa'].notna() & df['wsa'].notna() & (df['wsa'] > 0))
    df_valid = df[valid_mask].copy()
    
    if len(df_valid) == 0:
        print("No valid WSA data found!")
        return df
    
    # Calculate absolute residuals in km²
    df_valid['wsa_residual'] = df_valid['swot_wsa'] - df_valid['wsa']
    df_valid['wsa_abs_residual'] = np.abs(df_valid['wsa_residual'])
    
    # Calculate percentage errors
    df_valid['wsa_percentage_error'] = (df_valid['wsa_residual'] / df_valid['wsa']) * 100
    df_valid['wsa_abs_percentage_error'] = np.abs(df_valid['wsa_percentage_error'])
    
    # Merge back to original dataframe
    df.loc[valid_mask, ['wsa_residual', 'wsa_abs_residual', 'wsa_percentage_error', 'wsa_abs_percentage_error']] = \
        df_valid[['wsa_residual', 'wsa_abs_residual', 'wsa_percentage_error', 'wsa_abs_percentage_error']]
    
    return df

def find_threshold_for_target_std(df, target_std=0.1, error_col='swot_wse_error'):
    """Find threshold for WSE errors with target std"""
    
    valid_data = df.dropna(subset=[error_col, 'swot_wse_abs_error'])
    
    if len(valid_data) < 100:
        return {'threshold': np.nan, 'achieved_std': np.nan, 'n_observations': 0, 'success': False}
    
    def objective(threshold):
        if threshold <= 0:
            return float('inf')
        
        filtered_data = valid_data[valid_data['swot_wse_abs_error'] <= threshold]
        
        if len(filtered_data) < 50:
            return float('inf')
        
        achieved_std = filtered_data[error_col].std()
        return abs(achieved_std - target_std)
    
    try:
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
            
    except Exception as e:
        print(f"WSE optimization failed: {e}")
    
    return {'threshold': np.nan, 'achieved_std': np.nan, 'n_observations': 0, 'success': False}

def find_threshold_for_target_std_percentage(df, target_std=15.0, error_col='wsa_percentage_error'):
    """Find threshold for WSA percentage errors with target std"""
    
    valid_data = df.dropna(subset=[error_col, 'wsa_abs_percentage_error'])
    # Filter out partial lake detections for WSA analysis
    valid_data = valid_data[valid_data['swot_partial_f'] == 0]
    
    if len(valid_data) < 100:
        return {'threshold': np.nan, 'achieved_std': np.nan, 'n_observations': 0, 'success': False}
    
    def objective(threshold):
        if threshold <= 0:
            return float('inf')
        
        filtered_data = valid_data[valid_data['wsa_abs_percentage_error'] <= threshold]
        
        if len(filtered_data) < 50:
            return float('inf')
        
        achieved_std = filtered_data[error_col].std()
        return abs(achieved_std - target_std)
    
    try:
        result = minimize_scalar(objective, bounds=(1.0, 200.0), method='bounded')
        
        if result.success and result.fun < 1.0:
            threshold = result.x
            filtered_data = valid_data[valid_data['wsa_abs_percentage_error'] <= threshold]
            achieved_std = filtered_data[error_col].std()
            
            return {
                'threshold': threshold,
                'achieved_std': achieved_std,
                'n_observations': len(filtered_data),
                'success': True,
                'filtered_data': filtered_data
            }
            
    except Exception as e:
        print(f"WSA optimization failed: {e}")
    
    return {'threshold': np.nan, 'achieved_std': np.nan, 'n_observations': 0, 'success': False}

def calculate_temporal_resolution(df, error_thresholds, use_ice_aware=True, error_type='wse'):
    """Calculate temporal resolution for WSE or WSA errors"""
    
    results = []
    
    for threshold in error_thresholds:
        lake_stats = []
        
        for lake_id in df['swot_lake_id'].unique():
            lake_data = df[df['swot_lake_id'] == lake_id].copy()
            lake_data['date'] = pd.to_datetime(lake_data['date'])
            lake_data = lake_data.sort_values('date')
            
            if len(lake_data) < 2:
                continue
            
            # Step 1: Mask out ice observations
            if use_ice_aware and 'ice' in lake_data.columns:
                non_ice_data = lake_data[lake_data['ice'] == 0].copy()
                if len(non_ice_data) == 0:
                    continue
            else:
                non_ice_data = lake_data.copy()
            
            # Step 2: Filter by error threshold
            if error_type == 'wse':
                error_col = 'swot_wse_abs_error'
                if threshold == np.inf:
                    valid_obs_data = non_ice_data.dropna(subset=[error_col]).copy()
                else:
                    valid_obs_data = non_ice_data.dropna(subset=[error_col])
                    valid_obs_data = valid_obs_data[valid_obs_data[error_col] <= threshold]
            else:  # wsa
                error_col = 'wsa_abs_percentage_error'
                # Filter out partial lake detections for WSA analysis
                non_ice_data_full_lakes = non_ice_data[non_ice_data['swot_partial_f'] == 0]
                if threshold == np.inf:
                    valid_obs_data = non_ice_data_full_lakes.dropna(subset=[error_col]).copy()
                else:
                    valid_obs_data = non_ice_data_full_lakes.dropna(subset=[error_col])
                    valid_obs_data = valid_obs_data[valid_obs_data[error_col] <= threshold]
            
            if len(valid_obs_data) < 3:
                continue
            
            # Step 3: Calculate temporal resolution
            non_ice_days = len(non_ice_data)
            valid_observations = len(valid_obs_data)
            days_per_obs = non_ice_days / valid_observations
            
            lake_stats.append({
                'swot_lake_id': lake_id,
                'threshold': threshold,
                'total_days': non_ice_days,
                'n_observations': valid_observations,
                'days_per_observation': days_per_obs
            })
        
        if lake_stats:
            lake_df = pd.DataFrame(lake_stats)
            
            total_days = lake_df['total_days'].sum()
            total_observations = lake_df['n_observations'].sum()
            overall_days_per_obs = total_days / total_observations if total_observations > 0 else np.nan
            
            results.append({
                'threshold': threshold,
                'overall_days_per_obs': overall_days_per_obs,
                'n_lakes': len(lake_df),
                'total_obs': total_observations
            })
    
    return results

def calculate_sampling_gaps(df, wse_threshold=0.28):
    """Calculate sampling gaps for optimal WSE error distribution"""
    
    all_gaps = []
    
    for lake_id in df['swot_lake_id'].unique():
        lake_data = df[df['swot_lake_id'] == lake_id].copy()
        
        if len(lake_data) < 2:
            continue
        
        lake_data['date'] = pd.to_datetime(lake_data['date'])
        lake_data = lake_data.sort_values('date').reset_index(drop=True)
        
        # Filter for valid WSE observations
        valid_obs = lake_data.dropna(subset=['swot_wse_abs_error'])
        valid_obs = valid_obs[valid_obs['swot_wse_abs_error'] <= wse_threshold].copy()
        
        if len(valid_obs) < 2:
            continue
        
        valid_obs = valid_obs.reset_index(drop=True)
        
        # Calculate gaps between consecutive valid observations
        for i in range(len(valid_obs) - 1):
            start_date = valid_obs.loc[i, 'date']
            end_date = valid_obs.loc[i + 1, 'date']
            
            gap_days = (end_date - start_date).days
            
            if gap_days <= 0:
                continue
            
            # Check for ice in gap period
            gap_date_range = []
            current_date = start_date
            while current_date <= end_date:
                gap_date_range.append(current_date)
                current_date += timedelta(days=1)
            
            ice_in_gap = False
            for gap_date in gap_date_range:
                date_obs = lake_data[lake_data['date'] == gap_date]
                if len(date_obs) > 0:
                    if date_obs['ice'].max() == 1:
                        ice_in_gap = True
                        break
            
            if not ice_in_gap:
                all_gaps.append(gap_days)
    
    return all_gaps

def calculate_error_metrics(data, error_col, abs_error_col, dataset_name, baseline_data=None, threshold=None, full_df=None):
    """Calculate comprehensive error metrics for a given dataset"""
    
    valid_data = data.dropna(subset=[error_col, abs_error_col])
    
    if len(valid_data) == 0:
        return {
            'Distribution': dataset_name,
            'Count': '0',
            'Retention Rate': '0%',
            'Mean Error': '',
            'MAE': '',
            'RMSE': '',
            '1σ signed error': '',
            '1σ absolute error': '',
            '68th percentile error': '',
            'Temporal Resolution': '',
            'Mean Sampling Gap': '',
            'Median Sampling Gap': ''
        }
    
    # Basic metrics
    n_observations = len(valid_data)
    
    # Calculate retention rate against baseline dataset (if provided)
    if baseline_data is not None:
        baseline_valid = baseline_data.dropna(subset=[error_col, abs_error_col])
        retention_rate = len(valid_data) / len(baseline_valid) * 100 if len(baseline_valid) > 0 else 0.0
    else:
        retention_rate = 100.0  # If no baseline, assume this is the full dataset
    
    mean_error = valid_data[error_col].mean()
    mean_absolute_error = valid_data[abs_error_col].mean()
    rmse = np.sqrt(np.mean(valid_data[error_col] ** 2))
    std_signed = valid_data[error_col].std()
    std_absolute = valid_data[abs_error_col].std()
    percentile_68_absolute = np.percentile(valid_data[abs_error_col], 68)
    
    # Calculate temporal resolution using the EXACT same function as the figure
    if threshold is not None and full_df is not None:
        # Determine error type based on column
        if abs_error_col == 'swot_wse_abs_error':
            error_type = 'wse'
        else:
            error_type = 'wsa'
        
        # Use the exact same function as the figure
        ice_available = 'ice' in full_df.columns
        temporal_results = calculate_temporal_resolution(full_df, [threshold], use_ice_aware=ice_available, error_type=error_type)
        
        if temporal_results and len(temporal_results) > 0:
            overall_temporal_resolution = temporal_results[0]['overall_days_per_obs']
        else:
            overall_temporal_resolution = np.nan
    else:
        overall_temporal_resolution = np.nan
    
    # Calculate sampling gaps separately (this is different from temporal resolution)
    sampling_gaps = []
    # Check if data has lake IDs (interpolated data may not)
    if 'swot_lake_id' in valid_data.columns:
        for lake_id in valid_data['swot_lake_id'].unique():
            lake_data = valid_data[valid_data['swot_lake_id'] == lake_id].copy()
            lake_data['date'] = pd.to_datetime(lake_data['date'])
            lake_data = lake_data.sort_values('date')
            
            if len(lake_data) >= 2:
                for i in range(len(lake_data) - 1):
                    gap_days = (lake_data.iloc[i+1]['date'] - lake_data.iloc[i]['date']).days
                    if gap_days > 0:
                        sampling_gaps.append(gap_days)
    
    mean_sampling_gap = np.mean(sampling_gaps) if sampling_gaps else np.nan
    median_sampling_gap = np.median(sampling_gaps) if sampling_gaps else np.nan
    
    return {
        'Distribution': dataset_name,
        'Count': f"{n_observations:,}",
        'Retention Rate': f"{retention_rate:.0f}%" if retention_rate == 100 else f"{retention_rate:.0f}%",
        'Mean Error': f"{mean_error:.2f}",
        'MAE': f"{mean_absolute_error:.2f}",
        'RMSE': f"{rmse:.2f}",
        '1σ signed error': f"{std_signed:.2f}",
        '1σ absolute error': f"{std_absolute:.2f}",
        '68th percentile error': f"{percentile_68_absolute:.2f}",
        'Temporal Resolution': f"{overall_temporal_resolution:.2f}" if not np.isnan(overall_temporal_resolution) else '',
        'Mean Sampling Gap': f"{mean_sampling_gap:.2f}" if not np.isnan(mean_sampling_gap) else '',
        'Median Sampling Gap': f"{median_sampling_gap:.0f}" if not np.isnan(median_sampling_gap) else ''
    }

def interpolate_swot_to_insitu_dates(lake_data_target_filtered, lake_data_all, variable='wse'):
    """
    Consolidated function to interpolate SWOT values to in-situ dates and calculate errors correctly.
    
    Args:
        lake_data_target_filtered: Lake data already filtered by error threshold  
        lake_data_all: All lake data (for in-situ dates)
        variable: 'wse' or 'wsa'
    
    Returns:
        dict with interpolated_errors, interpolated_values, valid_dates, target_dates
    """
    from scipy.interpolate import interp1d
    
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
        # Filter to good SWOT observations but keep their anomaly measurements
        target_dates_with_data = lake_data_target_ice_free[['date', 'swot_wse_anomaly']].dropna()
        
        if len(target_dates_with_data) >= 2:
            # Get all ice-free dates with in-situ stage anomaly data
            insitu_dates_ice_free = lake_data_ice_free[['date', 'stage_anomaly_swotdates']].dropna()
            
            # Filter out physically unreasonable stage anomalies (likely data errors)
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
                        
                        # Interpolate SWOT WSE anomaly values (from good observations)
                        anomaly_interp_func = interp1d(swot_dates_numeric, target_dates_with_data['swot_wse_anomaly'], 
                                                     kind='linear', bounds_error=False, fill_value=np.nan)
                        
                        interpolated_swot_anomaly = anomaly_interp_func(interp_dates_numeric)
                        
                        # Remove NaN values and calculate errors properly
                        valid_interp_mask = ~np.isnan(interpolated_swot_anomaly)
                        if valid_interp_mask.sum() >= 2:
                            # Get corresponding in-situ stage anomalies - use iloc for position-based indexing
                            valid_indices = np.where(valid_interp_mask)[0]
                            insitu_stage_anomalies = dates_to_interpolate.iloc[valid_indices]['stage_anomaly_swotdates'].values
                            
                            # Calculate interpolated errors: interpolated SWOT anomaly - in-situ stage anomaly
                            interpolated_errors = interpolated_swot_anomaly[valid_interp_mask] - insitu_stage_anomalies
                            
                            # Debug: Check for unreasonably large errors during interpolation
                            max_error = np.abs(interpolated_errors).max() if len(interpolated_errors) > 0 else 0
                            if max_error > 10.0:
                                print(f"DEBUG: Large interpolation error detected in function")
                                print(f"  Interpolated SWOT anomaly range: {interpolated_swot_anomaly[valid_interp_mask].min():.3f} to {interpolated_swot_anomaly[valid_interp_mask].max():.3f}")
                                print(f"  In-situ stage anomaly range: {insitu_stage_anomalies.min():.3f} to {insitu_stage_anomalies.max():.3f}")
                                print(f"  Max interpolated error: {max_error:.3f}m")
                                print(f"  First few interpolated errors: {interpolated_errors[:5]}")
                                print(f"  First few interpolated SWOT anomalies: {interpolated_swot_anomaly[valid_interp_mask][:5]}")
                                print(f"  First few stage anomalies: {insitu_stage_anomalies[:5]}")
                            
                            return {
                                'interpolated_errors': interpolated_errors,
                                'interpolated_values': interpolated_swot_anomaly[valid_interp_mask],
                                'valid_dates': dates_to_interpolate.iloc[valid_indices]['date'].values,
                                'target_dates_with_data': target_dates_with_data,
                                'dates_to_interpolate': dates_to_interpolate
                            }
                    except Exception as e:
                        print(f"WSE interpolation failed: {e}")
                        return None
                        
    elif variable == 'wsa':
        # Get SWOT WSA data for interpolation (from target distribution) 
        target_dates_with_data = lake_data_target_ice_free[['date', 'swot_wsa']].dropna()
        
        if len(target_dates_with_data) >= 2:
            # Get all ice-free dates with in-situ WSA data
            insitu_dates_ice_free = lake_data_ice_free[['date', 'wsa']].dropna()
            
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
                        
                        # Create interpolation function for SWOT WSA
                        interp_func = interp1d(swot_dates_numeric, target_dates_with_data['swot_wsa'], 
                                             kind='linear', bounds_error=False, fill_value=np.nan)
                        
                        # Interpolate SWOT WSA values
                        interpolated_swot_wsa = interp_func(interp_dates_numeric)
                        
                        # Calculate percentage errors (remove NaN values)
                        valid_interp_mask = ~np.isnan(interpolated_swot_wsa)
                        if valid_interp_mask.sum() >= 2:
                            insitu_wsa_interp = dates_to_interpolate.loc[dates_to_interpolate.index[valid_interp_mask], 'wsa'].values
                            
                            # Filter out zero or negative WSA values to avoid division by zero
                            valid_wsa_mask = insitu_wsa_interp > 0
                            if valid_wsa_mask.sum() >= 2:
                                insitu_wsa_valid = insitu_wsa_interp[valid_wsa_mask]
                                interpolated_swot_valid = interpolated_swot_wsa[valid_interp_mask][valid_wsa_mask]
                                interpolated_wsa_errors = ((interpolated_swot_valid - insitu_wsa_valid) / insitu_wsa_valid) * 100
                                
                                return {
                                    'interpolated_errors': interpolated_wsa_errors,
                                    'interpolated_values': interpolated_swot_valid,
                                    'valid_dates': dates_to_interpolate.loc[dates_to_interpolate.index[valid_interp_mask], 'date'].values[valid_wsa_mask],
                                    'target_dates_with_data': target_dates_with_data,
                                    'dates_to_interpolate': dates_to_interpolate
                                }
                    except Exception as e:
                        print(f"WSA interpolation failed: {e}")
                        return None
    
    return None

def calculate_snr_by_lake(df, wse_optimal_result=None):
    """Calculate Signal-to-Noise Ratio for each lake with target distribution only"""
    
    print("\nCalculating Signal-to-Noise Ratios by lake...")
    
    snr_results = []
    
    for lake_id in df['swot_lake_id'].unique():
        lake_data = df[df['swot_lake_id'] == lake_id].copy()
        lake_data['date'] = pd.to_datetime(lake_data['date'])
        lake_data = lake_data.sort_values('date').reset_index(drop=True)
        
        if len(lake_data) < 3:  # Need minimum observations for variance calculation
            continue
        
        # Initialize target distribution variables
        snr_target_alldates = np.nan
        snr_target_swotdates = np.nan
        rmse_target_swotdates = np.nan
        rmse_target_alldates = np.nan
        signal_var_alldates = np.nan
        signal_var_swotdates = np.nan
        residual_var_target_alldates = np.nan
        residual_var_target_swotdates = np.nan
        
        # Calculate for target/optimal distribution if threshold exists
        if wse_optimal_result is not None and wse_optimal_result['success']:
            # Filter lake data to target distribution
            lake_data_target = lake_data[lake_data['swot_wse_abs_error'] <= wse_optimal_result['threshold']].copy()
            
            if len(lake_data_target) >= 3:  # Need minimum for interpolation
                # Ice-aware filtering for all dates
                ice_available = 'ice' in lake_data.columns
                if ice_available:
                    lake_data_ice_free = lake_data[lake_data['ice'] == 0].copy()
                else:
                    lake_data_ice_free = lake_data.copy()
                
                if len(lake_data_ice_free) < 2:
                    continue
                
                # Get signal variance from all ice-free dates
                stage_anomaly_alldates = lake_data_ice_free['stage_anomaly_alldates'].dropna()
                # Filter out physically unreasonable stage anomalies (likely data errors)
                reasonable_stage_mask = stage_anomaly_alldates.abs() <= 50.0
                stage_anomaly_alldates = stage_anomaly_alldates[reasonable_stage_mask]
                
                if len(stage_anomaly_alldates) < 2:
                    continue
                
                signal_var_alldates = np.var(stage_anomaly_alldates, ddof=1)
                
                # Get signal variance from SWOT dates only (target distribution)
                if ice_available:
                    lake_data_target_ice_free = lake_data_target[lake_data_target['ice'] == 0].copy()
                else:
                    lake_data_target_ice_free = lake_data_target.copy()
                
                if len(lake_data_target_ice_free) < 2:
                    continue
                
                stage_anomaly_swotdates = lake_data_target_ice_free['stage_anomaly_swotdates'].dropna()
                # Filter out physically unreasonable stage anomalies (likely data errors)
                reasonable_stage_mask = stage_anomaly_swotdates.abs() <= 50.0
                stage_anomaly_swotdates = stage_anomaly_swotdates[reasonable_stage_mask]
                
                wse_errors_target = lake_data_target_ice_free['swot_wse_error'].dropna()
                
                if len(stage_anomaly_swotdates) < 2 or len(wse_errors_target) < 2:
                    continue
                
                signal_var_swotdates = np.var(stage_anomaly_swotdates, ddof=1)
                residual_var_target_swotdates = np.var(wse_errors_target, ddof=1)
                
                # Calculate SNR for SWOT dates (using AllDates signal variance for fair comparison)
                snr_target_swotdates = signal_var_alldates / residual_var_target_swotdates if residual_var_target_swotdates > 0 else np.nan
                
                # Calculate RMSE for target distribution (SWOT dates)
                rmse_target_swotdates = np.sqrt(np.mean(wse_errors_target ** 2))
                
                # For alldates SNR: interpolate SWOT values to all ice-free dates
                interp_result = interpolate_swot_to_insitu_dates(lake_data_target, lake_data, variable='wse')
                
                if interp_result is not None:
                    interpolated_errors = interp_result['interpolated_errors']
                    
                    # Calculate residual variance for interpolated errors
                    residual_var_target_alldates = np.var(interpolated_errors, ddof=1)
                    
                    # Calculate RMSE for interpolated errors (all dates)
                    rmse_target_alldates = np.sqrt(np.mean(interpolated_errors ** 2))
                    
                    # Calculate alldates SNR
                    snr_target_alldates = signal_var_alldates / residual_var_target_alldates if residual_var_target_alldates > 0 else np.nan
        
        snr_results.append({
            'swot_lake_id': lake_id,
            'snr_target_alldates': snr_target_alldates,
            'snr_target_swotdates': snr_target_swotdates,
            'signal_var_alldates': signal_var_alldates,
            'signal_var_swotdates': signal_var_swotdates,
            'error_var_alldates': residual_var_target_alldates,
            'error_var_swotdates': residual_var_target_swotdates,
            'rmse_target_alldates': rmse_target_alldates,
            'rmse_target_swotdates': rmse_target_swotdates,
            'n_observations_target': len(lake_data[lake_data['swot_wse_abs_error'] <= wse_optimal_result['threshold']]) if wse_optimal_result and wse_optimal_result['success'] else 0
        })
    
    snr_df = pd.DataFrame(snr_results)
    print(f"Calculated SNR for {len(snr_df)} lakes")
    
    return snr_df

def create_snr_analysis_plot(snr_df, wsa_snr_df, df_with_size):
    """Create comprehensive SNR analysis visualization for both WSE and WSA"""
    
    print("\nCreating SNR analysis plot...")
    
    # Set up plotting style
    plt.rcParams.update({
        'font.size': 9,
        'text.color': 'black',
        'axes.labelcolor': 'black',
        'axes.edgecolor': 'black',
        'xtick.color': 'black',
        'ytick.color': 'black'
    })
    
    fig = plt.figure(figsize=(20, 12))
    
    # Create 2x4 grid layout (WSE top row, WSA bottom row)
    ax1 = plt.subplot(2, 4, 1)  # WSE Box plots
    ax2 = plt.subplot(2, 4, 2)  # WSE Histograms
    ax3 = plt.subplot(2, 4, 3)  # WSE RMSE vs SNR scatter
    ax4 = plt.subplot(2, 4, 4)  # WSE Lake size vs SNR scatter
    
    ax5 = plt.subplot(2, 4, 5)  # WSA Box plots
    ax6 = plt.subplot(2, 4, 6)  # WSA Histograms
    ax7 = plt.subplot(2, 4, 7)  # WSA RMSE vs SNR scatter
    ax8 = plt.subplot(2, 4, 8)  # WSA Lake size vs SNR scatter
    
    # Add lake size information to both SNR dataframes
    lake_size_dict = {}
    for _, row in df_with_size[['swot_lake_id', 'lake_size_category']].drop_duplicates().iterrows():
        lake_size_dict[row['swot_lake_id']] = row['lake_size_category']
    
    snr_df['lake_size_category'] = snr_df['swot_lake_id'].map(lake_size_dict)
    wsa_snr_df['lake_size_category'] = wsa_snr_df['swot_lake_id'].map(lake_size_dict)
    
    # ==========================================
    # WSE SNR PANELS (TOP ROW)
    # ==========================================
    
    # Panel 1: WSE Box plots of SNR distributions (target distribution only)
    snr_columns = ['snr_target_alldates', 'snr_target_swotdates']
    snr_labels = ['Target AllDates', 'Target SWOTDates']
    
    box_data = []
    valid_labels = []
    for col, label in zip(snr_columns, snr_labels):
        valid_data = snr_df[col].dropna()
        if len(valid_data) > 0:
            log_data = np.log10(np.clip(valid_data, 1e-6, 1e6))
            box_data.append(log_data)
            valid_labels.append(f"{label}\n(n={len(valid_data)})")
    
    if box_data:
        box_plot = ax1.boxplot(box_data, labels=valid_labels, patch_artist=True)
        colors = ['#fdb863', '#b2abd2']
        for patch, color in zip(box_plot['boxes'], colors[:len(box_data)]):
            patch.set_facecolor(color)
        
        ax1.set_ylabel('WSE SNR (log10)')
        ax1.set_title('WSE SNR Distributions (Target Distribution)')
        ax1.grid(True, alpha=0.3)
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # Panel 2: WSE Histograms of SNR (target distribution only)
    snr_alldates_valid = snr_df['snr_target_alldates'].dropna()
    snr_swotdates_valid = snr_df['snr_target_swotdates'].dropna()
    
    if len(snr_alldates_valid) > 0:
        log_snr_alldates = np.log10(np.clip(snr_alldates_valid, 1e-6, 1e6))
        ax2.hist(log_snr_alldates, bins=30, alpha=0.7, color='#fdb863', 
                label=f'AllDates (n={len(snr_alldates_valid)})', edgecolor='#fdb863')
    
    if len(snr_swotdates_valid) > 0:
        log_snr_swotdates = np.log10(np.clip(snr_swotdates_valid, 1e-6, 1e6))
        ax2.hist(log_snr_swotdates, bins=30, alpha=0.7, color='#b2abd2',
                label=f'SWOTDates (n={len(snr_swotdates_valid)})', edgecolor='#b2abd2')
    
    ax2.set_xlabel('WSE SNR (log10)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('WSE SNR Distribution (Target Distribution)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: WSE SNR vs RMSE scatter plot (target distribution only)
    target_valid_mask = snr_df['snr_target_alldates'].notna() & snr_df['rmse_target_alldates'].notna()
    target_swot_valid_mask = snr_df['snr_target_swotdates'].notna() & snr_df['rmse_target_swotdates'].notna()
    
    if target_valid_mask.any():
        ax3.scatter(snr_df.loc[target_valid_mask, 'rmse_target_alldates'],
                   snr_df.loc[target_valid_mask, 'snr_target_alldates'],
                   alpha=0.6, color='red', s=30, label='AllDates')
    
    if target_swot_valid_mask.any():
        ax3.scatter(snr_df.loc[target_swot_valid_mask, 'rmse_target_swotdates'],
                   snr_df.loc[target_swot_valid_mask, 'snr_target_swotdates'],
                   alpha=0.6, color='orange', s=30, label='SWOTDates')
    
    ax3.set_xlabel('RMSE (m)')
    ax3.set_ylabel('WSE SNR')
    ax3.set_title('WSE SNR vs RMSE (Target Distribution)')
    ax3.set_xlim(0, 0.5)
    ax3.set_yscale('log')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: WSE Lake size vs SNR scatter plot (target distribution only)
    small_lakes = snr_df[snr_df['lake_size_category'] == 'Small (<1 km²)']
    large_lakes = snr_df[snr_df['lake_size_category'] == 'Large (≥1 km²)']
    
    if len(small_lakes) > 0:
        small_valid = small_lakes['snr_target_alldates'].notna()
        ax4.scatter(small_lakes.loc[small_valid, 'n_observations_target'],
                   small_lakes.loc[small_valid, 'snr_target_alldates'],
                   alpha=0.6, color='red', s=30, label=f'Small (n={small_valid.sum()})')
    
    if len(large_lakes) > 0:
        large_valid = large_lakes['snr_target_alldates'].notna()
        ax4.scatter(large_lakes.loc[large_valid, 'n_observations_target'],
                   large_lakes.loc[large_valid, 'snr_target_alldates'],
                   alpha=0.6, color='blue', s=30, label=f'Large (n={large_valid.sum()})')
    
    ax4.set_xlabel('N Observations (Target)')
    ax4.set_ylabel('WSE SNR (AllDates)')
    ax4.set_title('WSE SNR vs Observations (Target Distribution)')
    ax4.set_yscale('log')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # ==========================================
    # WSA SNR PANELS (BOTTOM ROW)
    # ==========================================
    
    # Panel 5: WSA Box plots of SNR distributions (target distribution only)
    wsa_snr_columns = ['wsa_snr_target_alldates', 'wsa_snr_target_swotdates']
    wsa_snr_labels = ['Target AllDates', 'Target SWOTDates']
    
    wsa_box_data = []
    wsa_valid_labels = []
    for col, label in zip(wsa_snr_columns, wsa_snr_labels):
        valid_data = wsa_snr_df[col].dropna()
        if len(valid_data) > 0:
            log_data = np.log10(np.clip(valid_data, 1e-6, 1e6))
            wsa_box_data.append(log_data)
            wsa_valid_labels.append(f"{label}\n(n={len(valid_data)})")
    
    if wsa_box_data:
        wsa_box_plot = ax5.boxplot(wsa_box_data, labels=wsa_valid_labels, patch_artist=True)
        colors = ['#fdb863', '#b2abd2']
        for patch, color in zip(wsa_box_plot['boxes'], colors[:len(wsa_box_data)]):
            patch.set_facecolor(color)
        
        ax5.set_ylabel('WSA SNR (log10)')
        ax5.set_title('WSA SNR Distributions (Target Distribution)')
        ax5.grid(True, alpha=0.3)
        plt.setp(ax5.get_xticklabels(), rotation=45, ha='right')
    
    # Panel 6: WSA Histograms of SNR (target distribution only)
    wsa_snr_alldates_valid = wsa_snr_df['wsa_snr_target_alldates'].dropna()
    wsa_snr_swotdates_valid = wsa_snr_df['wsa_snr_target_swotdates'].dropna()
    
    if len(wsa_snr_alldates_valid) > 0:
        log_wsa_snr_alldates = np.log10(np.clip(wsa_snr_alldates_valid, 1e-6, 1e6))
        ax6.hist(log_wsa_snr_alldates, bins=30, alpha=0.7, color='#fdb863', 
                label=f'AllDates (n={len(wsa_snr_alldates_valid)})', edgecolor='#fdb863')
    
    if len(wsa_snr_swotdates_valid) > 0:
        log_wsa_snr_swotdates = np.log10(np.clip(wsa_snr_swotdates_valid, 1e-6, 1e6))
        ax6.hist(log_wsa_snr_swotdates, bins=30, alpha=0.7, color='#b2abd2',
                label=f'SWOTDates (n={len(wsa_snr_swotdates_valid)})', edgecolor='#b2abd2')
    
    ax6.set_xlabel('WSA SNR (log10)')
    ax6.set_ylabel('Frequency')
    ax6.set_title('WSA SNR Distribution (Target Distribution)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Panel 7: WSA SNR vs RMSE scatter plot (target distribution only)
    wsa_target_valid_mask = wsa_snr_df['wsa_snr_target_alldates'].notna() & wsa_snr_df['wsa_rmse_target_alldates'].notna()
    wsa_target_swot_valid_mask = wsa_snr_df['wsa_snr_target_swotdates'].notna() & wsa_snr_df['wsa_rmse_target_swotdates'].notna()
    
    if wsa_target_valid_mask.any():
        ax7.scatter(wsa_snr_df.loc[wsa_target_valid_mask, 'wsa_rmse_target_alldates'],
                   wsa_snr_df.loc[wsa_target_valid_mask, 'wsa_snr_target_alldates'],
                   alpha=0.6, color='red', s=30, label='AllDates')
    
    if wsa_target_swot_valid_mask.any():
        ax7.scatter(wsa_snr_df.loc[wsa_target_swot_valid_mask, 'wsa_rmse_target_swotdates'],
                   wsa_snr_df.loc[wsa_target_swot_valid_mask, 'wsa_snr_target_swotdates'],
                   alpha=0.6, color='orange', s=30, label='SWOTDates')
    
    ax7.set_xlabel('RMSE (%)')
    ax7.set_ylabel('WSA SNR')
    ax7.set_title('WSA SNR vs RMSE (Target Distribution)')
    ax7.set_xlim(0, 50)  
    ax7.set_yscale('log')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # Panel 8: WSA Lake size vs SNR scatter plot (target distribution only)
    wsa_small_lakes = wsa_snr_df[wsa_snr_df['lake_size_category'] == 'Small (<1 km²)']
    wsa_large_lakes = wsa_snr_df[wsa_snr_df['lake_size_category'] == 'Large (≥1 km²)']
    
    if len(wsa_small_lakes) > 0:
        wsa_small_valid = wsa_small_lakes['wsa_snr_target_alldates'].notna()
        ax8.scatter(wsa_small_lakes.loc[wsa_small_valid, 'wsa_n_observations_target'],
                   wsa_small_lakes.loc[wsa_small_valid, 'wsa_snr_target_alldates'],
                   alpha=0.6, color='red', s=30, label=f'Small (n={wsa_small_valid.sum()})')
    
    if len(wsa_large_lakes) > 0:
        wsa_large_valid = wsa_large_lakes['wsa_snr_target_alldates'].notna()
        ax8.scatter(wsa_large_lakes.loc[wsa_large_valid, 'wsa_n_observations_target'],
                   wsa_large_lakes.loc[wsa_large_valid, 'wsa_snr_target_alldates'],
                   alpha=0.6, color='blue', s=30, label=f'Large (n={wsa_large_valid.sum()})')
    
    ax8.set_xlabel('N Observations (Target)')
    ax8.set_ylabel('WSA SNR (AllDates)')
    ax8.set_title('WSA SNR vs Observations (Target Distribution)')
    ax8.set_yscale('log')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs('experiments/results/3.1/snr', exist_ok=True)
    plt.savefig('experiments/results/3.1/snr/snr_analysis.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return fig

def calculate_wsa_snr_by_lake(df, wsa_optimal_result=None):
    """Calculate WSA Signal-to-Noise Ratio for each lake with target distribution only"""
    
    print("\nCalculating WSA Signal-to-Noise Ratios by lake...")
    
    wsa_snr_results = []
    
    for lake_id in df['swot_lake_id'].unique():
        lake_data = df[df['swot_lake_id'] == lake_id].copy()
        lake_data['date'] = pd.to_datetime(lake_data['date'])
        lake_data = lake_data.sort_values('date').reset_index(drop=True)
        
        if len(lake_data) < 3:  # Need minimum observations for variance calculation
            continue
        
        # Initialize target distribution variables
        wsa_snr_target_alldates = np.nan
        wsa_snr_target_swotdates = np.nan
        wsa_rmse_target_swotdates = np.nan
        wsa_rmse_target_alldates = np.nan
        signal_var_all_pct = np.nan
        signal_var_swot_pct = np.nan
        error_var_target_alldates = np.nan
        error_var_target_swotdates = np.nan
        
        # Calculate for target/optimal distribution if threshold exists
        if wsa_optimal_result is not None and wsa_optimal_result['success']:
            # Filter lake data to target distribution (error threshold + no partial lakes)
            lake_data_target = lake_data[
                (lake_data['wsa_abs_percentage_error'] <= wsa_optimal_result['threshold']) &
                (lake_data['swot_partial_f'] == 0)
            ].copy()
            
            if len(lake_data_target) >= 3:  # Need minimum for interpolation
                # Ice-aware filtering for all dates
                ice_available = 'ice' in lake_data.columns
                if ice_available:
                    lake_data_ice_free = lake_data[lake_data['ice'] == 0].copy()
                else:
                    lake_data_ice_free = lake_data.copy()
                
                if len(lake_data_ice_free) < 2:
                    continue
                
                # Get WSA data for signal variance calculations (all ice-free dates)
                # Filter out dates with unreasonable stage anomalies (data quality filter)
                reasonable_stage_mask = lake_data_ice_free['stage_anomaly_alldates'].abs() <= 50.0
                wsa_values_all = lake_data_ice_free.loc[reasonable_stage_mask, 'wsa'].dropna()
                if len(wsa_values_all) < 2:
                    continue
                
                wsa_mean_all = wsa_values_all.mean()
                if wsa_mean_all <= 0:
                    continue
                
                # Calculate percentage anomalies for all dates
                wsa_percentage_anomalies_all = ((wsa_values_all - wsa_mean_all) / wsa_mean_all) * 100
                signal_var_all_pct = np.var(wsa_percentage_anomalies_all, ddof=1)
                
                # Get target distribution data (ice-free)
                if ice_available:
                    lake_data_target_ice_free = lake_data_target[lake_data_target['ice'] == 0].copy()
                else:
                    lake_data_target_ice_free = lake_data_target.copy()
                
                if len(lake_data_target_ice_free) < 2:
                    continue
                
                # Filter out dates with unreasonable stage anomalies for WSA analysis
                reasonable_stage_mask_target = lake_data_target_ice_free['stage_anomaly_swotdates'].abs() <= 50.0
                
                wsa_swot_coincident_target = lake_data_target_ice_free.loc[
                    reasonable_stage_mask_target & lake_data_target_ice_free['swot_wsa'].notna(), 'wsa'
                ].dropna()
                
                wsa_percentage_errors_target = lake_data_target_ice_free.loc[
                    reasonable_stage_mask_target, 'wsa_percentage_error'
                ].dropna()
                
                if len(wsa_swot_coincident_target) < 2 or len(wsa_percentage_errors_target) < 2:
                    continue
                
                # Calculate signal variance for SWOT dates
                wsa_swot_mean_target = wsa_swot_coincident_target.mean()
                if wsa_swot_mean_target <= 0:
                    continue
                
                wsa_percentage_anomalies_swot = ((wsa_swot_coincident_target - wsa_swot_mean_target) / wsa_swot_mean_target) * 100
                signal_var_swot_pct = np.var(wsa_percentage_anomalies_swot, ddof=1)
                error_var_target_swotdates = np.var(wsa_percentage_errors_target, ddof=1)
                
                # Calculate SNR for SWOT dates (using AllDates signal variance for fair comparison)
                wsa_snr_target_swotdates = signal_var_all_pct / error_var_target_swotdates if error_var_target_swotdates > 0 else np.nan
                
                # Calculate RMSE for target distribution (SWOT dates)
                wsa_rmse_target_swotdates = np.sqrt(np.mean(wsa_percentage_errors_target ** 2))
                
                # For alldates SNR: interpolate SWOT WSA values to all ice-free dates
                wsa_interp_result = interpolate_swot_to_insitu_dates(lake_data_target, lake_data, variable='wsa')
                
                if wsa_interp_result is not None:
                    interpolated_wsa_errors = wsa_interp_result['interpolated_errors']
                    
                    # Calculate error variance for interpolated errors
                    error_var_target_alldates = np.var(interpolated_wsa_errors, ddof=1)
                    
                    # Calculate RMSE for interpolated WSA errors (all dates)
                    wsa_rmse_target_alldates = np.sqrt(np.mean(interpolated_wsa_errors ** 2))
                    
                    # Calculate alldates SNR
                    wsa_snr_target_alldates = signal_var_all_pct / error_var_target_alldates if error_var_target_alldates > 0 else np.nan
        
        wsa_snr_results.append({
            'swot_lake_id': lake_id,
            'wsa_snr_target_alldates': wsa_snr_target_alldates,
            'wsa_snr_target_swotdates': wsa_snr_target_swotdates,
            'wsa_signal_var_alldates': signal_var_all_pct,
            'wsa_signal_var_swotdates': signal_var_swot_pct,
            'wsa_error_var_alldates': error_var_target_alldates,
            'wsa_error_var_swotdates': error_var_target_swotdates,
            'wsa_rmse_target_alldates': wsa_rmse_target_alldates,
            'wsa_rmse_target_swotdates': wsa_rmse_target_swotdates,
            'wsa_n_observations_target': len(lake_data_target) if wsa_optimal_result and wsa_optimal_result['success'] else 0
        })
    
    wsa_snr_df = pd.DataFrame(wsa_snr_results)
    print(f"Calculated WSA SNR for {len(wsa_snr_df)} lakes")
    
    return wsa_snr_df

def calculate_nse_by_lake(df, wse_optimal_result=None):
    """Calculate Nash-Sutcliffe Efficiency for each lake with target distribution only"""
    
    print("\nCalculating Nash-Sutcliffe Efficiency by lake...")
    
    nse_results = []
    
    for lake_id in df['swot_lake_id'].unique():
        lake_data = df[df['swot_lake_id'] == lake_id].copy()
        lake_data['date'] = pd.to_datetime(lake_data['date'])
        lake_data = lake_data.sort_values('date').reset_index(drop=True)
        
        if len(lake_data) < 3:  # Need minimum observations for calculation
            continue
        
        # Initialize target distribution variables
        nse_target_alldates = np.nan
        nse_target_swotdates = np.nan
        rmse_target_swotdates = np.nan
        rmse_target_alldates = np.nan
        
        # Calculate for target/optimal distribution if threshold exists
        if wse_optimal_result is not None and wse_optimal_result['success']:
            # Filter lake data to target distribution
            lake_data_target = lake_data[lake_data['swot_wse_abs_error'] <= wse_optimal_result['threshold']].copy()
            
            if len(lake_data_target) >= 3:  # Need minimum for interpolation
                # Ice-aware filtering for all dates
                ice_available = 'ice' in lake_data.columns
                if ice_available:
                    lake_data_ice_free = lake_data[lake_data['ice'] == 0].copy()
                    lake_data_target_ice_free = lake_data_target[lake_data_target['ice'] == 0].copy()
                else:
                    lake_data_ice_free = lake_data.copy()
                    lake_data_target_ice_free = lake_data_target.copy()
                
                if len(lake_data_ice_free) < 2 or len(lake_data_target_ice_free) < 2:
                    continue
                
                # Get observed stage anomalies from all ice-free dates (true values)
                stage_anomaly_alldates = lake_data_ice_free['stage_anomaly_alldates'].dropna()
                # Filter out physically unreasonable stage anomalies
                reasonable_stage_mask = stage_anomaly_alldates.abs() <= 50.0
                stage_anomaly_alldates = stage_anomaly_alldates[reasonable_stage_mask]
                
                if len(stage_anomaly_alldates) < 2:
                    continue
                
                # Calculate signal variance from all dates (for alldates NSE calculation)
                signal_var_alldates = np.var(stage_anomaly_alldates, ddof=1)
                
                # Calculate NSE for SWOT dates only
                stage_anomaly_swotdates = lake_data_target_ice_free['stage_anomaly_swotdates'].dropna()
                wse_errors_target = lake_data_target_ice_free['swot_wse_error'].dropna()
                
                if len(stage_anomaly_swotdates) >= 2 and len(wse_errors_target) >= 2:
                    # Calculate NSE for SWOT dates using same approach as SNR
                    signal_var_swotdates = np.var(stage_anomaly_swotdates, ddof=1)
                    error_var_swotdates = np.var(wse_errors_target, ddof=1)
                    
                    # NSE = 1 - (error_variance / signal_variance) = 1 - (1/SNR)
                    nse_target_swotdates = 1 - (error_var_swotdates / signal_var_swotdates) if signal_var_swotdates > 0 else np.nan
                    
                    # Calculate RMSE for target distribution (SWOT dates)
                    rmse_target_swotdates = np.sqrt(np.mean(wse_errors_target ** 2))
                
                # For alldates NSE: interpolate SWOT values to all ice-free dates
                interp_result = interpolate_swot_to_insitu_dates(lake_data_target, lake_data, variable='wse')
                
                if interp_result is not None:
                    interpolated_errors = interp_result['interpolated_errors']
                    
                    if len(interpolated_errors) > 0:
                        # Calculate NSE for alldates using same approach as SNR
                        # signal_var_alldates was already calculated above (from stage_anomaly_alldates)
                        error_var_alldates = np.var(interpolated_errors, ddof=1)
                        
                        # NSE = 1 - (error_variance / signal_variance) = 1 - (1/SNR)
                        nse_target_alldates = 1 - (error_var_alldates / signal_var_alldates) if signal_var_alldates > 0 else np.nan
                        
                        # Calculate RMSE for interpolated errors (all dates)
                        rmse_target_alldates = np.sqrt(np.mean(interpolated_errors ** 2))
        
        nse_results.append({
            'swot_lake_id': lake_id,
            'nse_target_alldates': nse_target_alldates,
            'nse_target_swotdates': nse_target_swotdates,
            'rmse_target_alldates': rmse_target_alldates,
            'rmse_target_swotdates': rmse_target_swotdates,
            'n_observations_target': len(lake_data[lake_data['swot_wse_abs_error'] <= wse_optimal_result['threshold']]) if wse_optimal_result and wse_optimal_result['success'] else 0
        })
    
    nse_df = pd.DataFrame(nse_results)
    print(f"Calculated NSE for {len(nse_df)} lakes")
    
    return nse_df

def calculate_wsa_nse_by_lake(df, wsa_optimal_result=None):
    """Calculate Nash-Sutcliffe Efficiency for WSA with target distribution only"""
    
    print("\nCalculating WSA Nash-Sutcliffe Efficiency by lake...")
    
    wsa_nse_results = []
    
    for lake_id in df['swot_lake_id'].unique():
        lake_data = df[df['swot_lake_id'] == lake_id].copy()
        lake_data['date'] = pd.to_datetime(lake_data['date'])
        lake_data = lake_data.sort_values('date').reset_index(drop=True)
        
        if len(lake_data) < 3:  # Need minimum observations for calculation
            continue
        
        # Initialize target distribution variables
        wsa_nse_target_alldates = np.nan
        wsa_nse_target_swotdates = np.nan
        wsa_rmse_target_swotdates = np.nan
        wsa_rmse_target_alldates = np.nan
        
        # Calculate for target/optimal distribution if threshold exists
        if wsa_optimal_result is not None and wsa_optimal_result['success']:
            # Filter lake data to target distribution (error threshold + no partial lakes) - same as SNR
            lake_data_target = lake_data[
                (lake_data['wsa_abs_percentage_error'] <= wsa_optimal_result['threshold']) &
                (lake_data['swot_partial_f'] == 0)
            ].copy()
            
            if len(lake_data_target) >= 3:  # Need minimum for interpolation
                # Ice-aware filtering for all dates - same as SNR
                ice_available = 'ice' in lake_data.columns
                if ice_available:
                    lake_data_ice_free = lake_data[lake_data['ice'] == 0].copy()
                else:
                    lake_data_ice_free = lake_data.copy()
                
                if len(lake_data_ice_free) < 2:
                    continue
                
                # Get WSA data for signal calculations (all ice-free dates) - same as SNR
                # Filter out dates with unreasonable stage anomalies (data quality filter)
                reasonable_stage_mask = lake_data_ice_free['stage_anomaly_alldates'].abs() <= 50.0
                wsa_values_all = lake_data_ice_free.loc[reasonable_stage_mask, 'wsa'].dropna()
                if len(wsa_values_all) < 2:
                    continue
                
                wsa_mean_all = wsa_values_all.mean()
                if wsa_mean_all <= 0:
                    continue
                
                # Calculate percentage anomalies for all dates - same as SNR
                wsa_percentage_anomalies_all = ((wsa_values_all - wsa_mean_all) / wsa_mean_all) * 100
                signal_var_all_pct = np.var(wsa_percentage_anomalies_all, ddof=1)
                
                # Get target distribution data (ice-free) - same as SNR
                if ice_available:
                    lake_data_target_ice_free = lake_data_target[lake_data_target['ice'] == 0].copy()
                else:
                    lake_data_target_ice_free = lake_data_target.copy()
                
                if len(lake_data_target_ice_free) < 2:
                    continue
                
                # Get SWOT dates data - same as SNR
                wsa_values_swot = lake_data_target_ice_free['wsa'].dropna()
                wsa_errors_target = lake_data_target_ice_free['wsa_percentage_error'].dropna()
                
                if len(wsa_values_swot) < 2 or len(wsa_errors_target) < 2:
                    continue
                
                wsa_mean_swot = wsa_values_swot.mean()
                if wsa_mean_swot <= 0:
                    continue
                
                # Calculate percentage anomalies for SWOT dates - same as SNR
                wsa_percentage_anomalies_swot = ((wsa_values_swot - wsa_mean_swot) / wsa_mean_swot) * 100
                
                # Calculate NSE for SWOT dates
                # NSE = 1 - SS_res/SS_tot where:
                # SS_res = sum of squared errors = sum(errors²)
                # SS_tot = sum of squared deviations = sum((observed - mean)²)
                signal_var_swot_pct = np.var(wsa_percentage_anomalies_swot, ddof=1)
                error_var_swot = np.var(wsa_errors_target, ddof=1)
                
                # NSE = 1 - (error_variance / signal_variance) = 1 - (1/SNR)
                wsa_nse_target_swotdates = 1 - (error_var_swot / signal_var_swot_pct) if signal_var_swot_pct > 0 else np.nan
                
                # Calculate RMSE for target distribution (SWOT dates)
                wsa_rmse_target_swotdates = np.sqrt(np.mean(wsa_errors_target ** 2))
                
                # For alldates NSE: interpolate SWOT WSA values to all ice-free dates - same as SNR
                interp_result = interpolate_swot_to_insitu_dates(lake_data_target, lake_data, variable='wsa')
                
                if interp_result is not None:
                    interpolated_errors = interp_result['interpolated_errors']
                    
                    if len(interpolated_errors) > 0:
                        # Calculate NSE for alldates using same approach as SNR
                        # signal_var_all_pct was already calculated above
                        error_var_alldates = np.var(interpolated_errors, ddof=1)
                        
                        # NSE = 1 - (error_variance / signal_variance) = 1 - (1/SNR)
                        wsa_nse_target_alldates = 1 - (error_var_alldates / signal_var_all_pct) if signal_var_all_pct > 0 else np.nan
                        
                        # Calculate RMSE for interpolated errors (all dates)
                        wsa_rmse_target_alldates = np.sqrt(np.mean(interpolated_errors ** 2))
        
        wsa_nse_results.append({
            'swot_lake_id': lake_id,
            'wsa_nse_target_alldates': wsa_nse_target_alldates,
            'wsa_nse_target_swotdates': wsa_nse_target_swotdates,
            'wsa_rmse_target_alldates': wsa_rmse_target_alldates,
            'wsa_rmse_target_swotdates': wsa_rmse_target_swotdates,
            'n_observations_target': len(lake_data[lake_data['wsa_abs_percentage_error'] <= wsa_optimal_result['threshold']]) if wsa_optimal_result and wsa_optimal_result['success'] else 0
        })
    
    wsa_nse_df = pd.DataFrame(wsa_nse_results)
    print(f"Calculated WSA NSE for {len(wsa_nse_df)} lakes")
    
    return wsa_nse_df

def export_nse_metrics_csv(nse_df, output_dir='experiments/results/3.1/nse'):
    """Export NSE summary statistics to CSV (target distribution only)"""
    
    print("\nExporting NSE metrics to CSV...")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate summary statistics for target distribution only
    nse_columns = ['nse_target_alldates', 'nse_target_swotdates']
    summary_stats = []
    
    for col in nse_columns:
        valid_data = nse_df[col].dropna()
        if len(valid_data) > 0:
            summary_stats.append({
                'NSE_Type': col.replace('nse_target_', '').replace('_', ' ').title(),
                'Count': len(valid_data),
                'Mean': np.mean(valid_data),
                'Median': np.median(valid_data),
                'Std': np.std(valid_data),
                'Min': np.min(valid_data),
                'Max': np.max(valid_data),
                'Q25': np.percentile(valid_data, 25),
                'Q75': np.percentile(valid_data, 75)
            })
    
    summary_df = pd.DataFrame(summary_stats)
    summary_csv_path = os.path.join(output_dir, 'nse_summary_statistics.csv')
    summary_df.to_csv(summary_csv_path, index=False)
    
    # Export per-lake NSE data
    lake_csv_path = os.path.join(output_dir, 'wse_nse_by_lake.csv')
    nse_df.to_csv(lake_csv_path, index=False)
    
    print(f"NSE summary statistics exported to: {summary_csv_path}")
    print(f"Per-lake NSE data exported to: {lake_csv_path}")
    
    return summary_csv_path, lake_csv_path

def export_wsa_nse_metrics_csv(wsa_nse_df, output_dir='experiments/results/3.1/nse'):
    """Export WSA NSE summary statistics to CSV (target distribution only)"""
    
    print("\nExporting WSA NSE metrics to CSV...")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate summary statistics for WSA NSE target distribution only
    wsa_nse_columns = ['wsa_nse_target_alldates', 'wsa_nse_target_swotdates']
    summary_stats = []
    
    for col in wsa_nse_columns:
        valid_data = wsa_nse_df[col].dropna()
        if len(valid_data) > 0:
            summary_stats.append({
                'WSA_NSE_Type': col.replace('wsa_nse_target_', '').replace('_', ' ').title(),
                'Count': len(valid_data),
                'Mean': np.mean(valid_data),
                'Median': np.median(valid_data),
                'Std': np.std(valid_data),
                'Min': np.min(valid_data),
                'Max': np.max(valid_data),
                'Q25': np.percentile(valid_data, 25),
                'Q75': np.percentile(valid_data, 75)
            })
    
    summary_df = pd.DataFrame(summary_stats)
    wsa_summary_csv_path = os.path.join(output_dir, 'wsa_nse_summary_statistics.csv')
    summary_df.to_csv(wsa_summary_csv_path, index=False)
    
    # Export per-lake WSA NSE data
    wsa_lake_csv_path = os.path.join(output_dir, 'wsa_nse_by_lake.csv')
    wsa_nse_df.to_csv(wsa_lake_csv_path, index=False)
    
    print(f"WSA NSE summary statistics exported to: {wsa_summary_csv_path}")
    print(f"Per-lake WSA NSE data exported to: {wsa_lake_csv_path}")
    
    return wsa_summary_csv_path, wsa_lake_csv_path

def export_snr_metrics_csv(snr_df, output_dir='experiments/results/3.1/snr'):
    """Export SNR summary statistics to CSV (target distribution only)"""
    
    print("\nExporting SNR metrics to CSV...")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate summary statistics for target distribution only
    snr_columns = ['snr_target_alldates', 'snr_target_swotdates']
    summary_stats = []
    
    for col in snr_columns:
        valid_data = snr_df[col].dropna()
        if len(valid_data) > 0:
            summary_stats.append({
                'SNR_Type': col.replace('snr_target_', '').replace('_', ' ').title(),
                'Count': len(valid_data),
                'Mean': valid_data.mean(),
                'Std': valid_data.std(),
                'Min': valid_data.min(),
                'Max': valid_data.max(),
                'Median': valid_data.median(),
                'Q25': valid_data.quantile(0.25),
                'Q75': valid_data.quantile(0.75)
            })
    
    # Export summary statistics
    summary_df = pd.DataFrame(summary_stats)
    summary_csv_path = os.path.join(output_dir, 'snr_summary_statistics.csv')
    summary_df.to_csv(summary_csv_path, index=False)
    
    # Export per-lake SNR data (rename to wse_snr_by_lake.csv)
    lake_csv_path = os.path.join(output_dir, 'wse_snr_by_lake.csv')
    snr_df.to_csv(lake_csv_path, index=False)
    
    print(f"SNR summary statistics exported to: {summary_csv_path}")
    print(f"Per-lake SNR data exported to: {lake_csv_path}")
    
    return summary_csv_path, lake_csv_path

def export_wsa_snr_metrics_csv(wsa_snr_df, output_dir='experiments/results/3.1/snr'):
    """Export WSA SNR summary statistics to CSV (target distribution only)"""
    
    print("\nExporting WSA SNR metrics to CSV...")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate summary statistics for WSA SNR target distribution only
    wsa_snr_columns = ['wsa_snr_target_alldates', 'wsa_snr_target_swotdates']
    wsa_summary_stats = []
    
    for col in wsa_snr_columns:
        valid_data = wsa_snr_df[col].dropna()
        if len(valid_data) > 0:
            wsa_summary_stats.append({
                'WSA_SNR_Type': col.replace('wsa_snr_target_', '').replace('_', ' ').title(),
                'Count': len(valid_data),
                'Mean': valid_data.mean(),
                'Std': valid_data.std(),
                'Min': valid_data.min(),
                'Max': valid_data.max(),
                'Median': valid_data.median(),
                'Q25': valid_data.quantile(0.25),
                'Q75': valid_data.quantile(0.75)
            })
    
    # Export summary statistics
    wsa_summary_df = pd.DataFrame(wsa_summary_stats)
    wsa_summary_csv_path = os.path.join(output_dir, 'wsa_snr_summary_statistics.csv')
    wsa_summary_df.to_csv(wsa_summary_csv_path, index=False)
    
    # Export per-lake WSA SNR data
    wsa_lake_csv_path = os.path.join(output_dir, 'wsa_snr_by_lake.csv')
    wsa_snr_df.to_csv(wsa_lake_csv_path, index=False)
    
    print(f"WSA SNR summary statistics exported to: {wsa_summary_csv_path}")
    print(f"Per-lake WSA SNR data exported to: {wsa_lake_csv_path}")
    
    return wsa_summary_csv_path, wsa_lake_csv_path

def export_error_metrics_csv(df, df_full_temporal=None, output_dir='experiments/results/3.1'):
    """Export WSE and WSA error metrics to CSV files"""
    
    print("\nCalculating and exporting error metrics to CSV...")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Use df_full_temporal for temporal resolution calculations, df for error calculations
    if df_full_temporal is None:
        df_full_temporal = df
    
    # Calculate optimal thresholds using the filtered dataset
    wse_optimal_result = find_threshold_for_target_std(df, target_std=0.1)
    wsa_optimal_result = find_threshold_for_target_std_percentage(df, target_std=15.0)
    
    # WSE Error Metrics
    wse_metrics = []
    
    # Full WSE distribution (baseline)
    full_wse_data = df.dropna(subset=['swot_wse_error', 'swot_wse_abs_error'])
    wse_full_metrics = calculate_error_metrics(
        full_wse_data, 'swot_wse_error', 'swot_wse_abs_error', 'Full', 
        threshold=np.inf, full_df=df_full_temporal
    )
    wse_metrics.append(wse_full_metrics)
    
    # Optimal WSE distribution (compare against full distribution)
    if wse_optimal_result['success']:
        optimal_wse_data = wse_optimal_result['filtered_data']
        wse_optimal_metrics = calculate_error_metrics(
            optimal_wse_data, 'swot_wse_error', 'swot_wse_abs_error', 'Target', 
            baseline_data=full_wse_data, threshold=wse_optimal_result['threshold'], full_df=df_full_temporal
        )
        wse_metrics.append(wse_optimal_metrics)
        
        # Collect interpolated WSE errors for Target-Daily distribution
        print("Collecting interpolated WSE errors for Target-Daily distribution...")
        interpolated_wse_errors_all = []
        
        for lake_id in df['swot_lake_id'].unique():
            lake_data = df[df['swot_lake_id'] == lake_id].copy()
            lake_data['date'] = pd.to_datetime(lake_data['date'])
            lake_data = lake_data.sort_values('date').reset_index(drop=True)
            
            if len(lake_data) < 3:
                continue
            
            lake_data_target = lake_data[lake_data['swot_wse_abs_error'] <= wse_optimal_result['threshold']].copy()
            
            if len(lake_data_target) >= 3:
                # Use consolidated interpolation function
                interp_result = interpolate_swot_to_insitu_dates(lake_data_target, lake_data, variable='wse')
                
                if interp_result is not None:
                    interpolated_wse_errors_all.extend(interp_result['interpolated_errors'])
        
        # Create dataframe with interpolated errors for metrics calculation
        if len(interpolated_wse_errors_all) > 0:
            interpolated_wse_df = pd.DataFrame({
                'swot_wse_error': interpolated_wse_errors_all,
                'swot_wse_abs_error': np.abs(interpolated_wse_errors_all)
            })
            
            wse_interpolated_metrics = calculate_error_metrics(
                interpolated_wse_df, 'swot_wse_error', 'swot_wse_abs_error', 'Target-Daily',
                baseline_data=full_wse_data, threshold=wse_optimal_result['threshold'], full_df=df_full_temporal
            )
            wse_metrics.append(wse_interpolated_metrics)
        else:
            print("Warning: No interpolated WSE errors collected")
    else:
        print("Warning: Could not find optimal WSE distribution")
    
    # Export WSE metrics
    wse_df = pd.DataFrame(wse_metrics)
    wse_csv_path = os.path.join(output_dir, 'wse_error_metrics.csv')
    wse_df.to_csv(wse_csv_path, index=False)
    print(f"WSE error metrics exported to: {wse_csv_path}")
    
    # WSA Error Metrics
    wsa_metrics = []
    
    # Full WSA distribution (baseline) - exclude partial lake detections
    full_wsa_data = df.dropna(subset=['wsa_percentage_error', 'wsa_abs_percentage_error'])
    full_wsa_data = full_wsa_data[full_wsa_data['swot_partial_f'] == 0]
    wsa_full_metrics = calculate_error_metrics(
        full_wsa_data, 'wsa_percentage_error', 'wsa_abs_percentage_error', 'Full',
        threshold=np.inf, full_df=df_full_temporal
    )
    wsa_metrics.append(wsa_full_metrics)
    
    # Optimal WSA distribution (compare against full distribution)
    if wsa_optimal_result['success']:
        optimal_wsa_data = wsa_optimal_result['filtered_data']
        wsa_optimal_metrics = calculate_error_metrics(
            optimal_wsa_data, 'wsa_percentage_error', 'wsa_abs_percentage_error', 'Target',
            baseline_data=full_wsa_data, threshold=wsa_optimal_result['threshold'], full_df=df_full_temporal
        )
        wsa_metrics.append(wsa_optimal_metrics)
        
        # Collect interpolated WSA errors for Target-Daily distribution
        print("Collecting interpolated WSA errors for Target-Daily distribution...")
        interpolated_wsa_errors_all = []
        
        for lake_id in df['swot_lake_id'].unique():
            lake_data = df[df['swot_lake_id'] == lake_id].copy()
            lake_data['date'] = pd.to_datetime(lake_data['date'])
            lake_data = lake_data.sort_values('date').reset_index(drop=True)
            
            if len(lake_data) < 3:
                continue
            
            lake_data_target = lake_data[
                (lake_data['wsa_abs_percentage_error'] <= wsa_optimal_result['threshold']) &
                (lake_data['swot_partial_f'] == 0)
            ].copy()
            
            if len(lake_data_target) >= 3:
                # Use consolidated interpolation function
                wsa_interp_result = interpolate_swot_to_insitu_dates(lake_data_target, lake_data, variable='wsa')
                
                if wsa_interp_result is not None:
                    interpolated_wsa_errors_all.extend(wsa_interp_result['interpolated_errors'])
        
        # Create dataframe with interpolated errors for metrics calculation
        if len(interpolated_wsa_errors_all) > 0:
            interpolated_wsa_df = pd.DataFrame({
                'wsa_percentage_error': interpolated_wsa_errors_all,
                'wsa_abs_percentage_error': np.abs(interpolated_wsa_errors_all)
            })
            
            wsa_interpolated_metrics = calculate_error_metrics(
                interpolated_wsa_df, 'wsa_percentage_error', 'wsa_abs_percentage_error', 'Target-Daily',
                baseline_data=full_wsa_data, threshold=wsa_optimal_result['threshold'], full_df=df_full_temporal
            )
            wsa_metrics.append(wsa_interpolated_metrics)
        else:
            print("Warning: No interpolated WSA errors collected")
    else:
        print("Warning: Could not find optimal WSA distribution")
    
    # Export WSA metrics
    wsa_df = pd.DataFrame(wsa_metrics)
    wsa_csv_path = os.path.join(output_dir, 'wsa_error_metrics.csv')
    wsa_df.to_csv(wsa_csv_path, index=False)
    print(f"WSA error metrics exported to: {wsa_csv_path}")
    
    return wse_csv_path, wsa_csv_path

def create_combined_comprehensive_plot(df, df_with_size):
    """Create the combined 6-panel comprehensive plot"""
    
    print("\nCreating combined 6-panel comprehensive analysis plot...")
    
    # Set up plotting style
    plt.rcParams.update({
        'font.size': 14,
        'text.color': 'black',
        'axes.labelcolor': 'black',
        'axes.edgecolor': 'black',
        'xtick.color': 'black',
        'ytick.color': 'black'
    })
    
    fig = plt.figure(figsize=(18, 12))
    
    # Create 3x2 grid layout 
    ax1 = plt.subplot(3, 2, 1)  # Panel 1: WSE CDF
    ax2 = plt.subplot(3, 2, 2)  # Panel 2: WSE Histogram  
    ax3 = plt.subplot(3, 2, 3)  # Panel 3: WSA CDF
    ax4 = plt.subplot(3, 2, 4)  # Panel 4: WSA Histogram
    ax5 = plt.subplot(3, 2, 5)  # Panel 5: Sampling Gap Histogram
    ax6 = plt.subplot(3, 2, 6)  # Panel 6: Combined Temporal Resolution
    
    # Find optimal thresholds
    wse_optimal_result = find_threshold_for_target_std(df, target_std=0.1)
    wsa_optimal_result = find_threshold_for_target_std_percentage(df, target_std=15.0)
    
    # ========================================
    # Panel 1: WSE CDF 
    # ========================================
    print("Creating Panel 1: WSE CDF...")

    all_lakes_errors = df_with_size['swot_wse_abs_error'].dropna()
    
    if len(all_lakes_errors) > 0:
        all_sorted = np.sort(all_lakes_errors)
        all_p = np.arange(1, len(all_sorted) + 1) / len(all_sorted)
        
        ax1.plot(all_sorted, all_p * 100, color='#d95f02', linewidth=2.5,
                label=f'All SWOT Measurements [n={len(all_lakes_errors):,}]')
        
        if wse_optimal_result['success']:
            optimal_errors = wse_optimal_result['filtered_data']['swot_wse_abs_error'].dropna()
            optimal_sorted = np.sort(optimal_errors)
            optimal_p = np.arange(1, len(optimal_sorted) + 1) / len(optimal_sorted)
            ax1.plot(optimal_sorted, optimal_p * 100, color='#1b9e77', linewidth=3,
                    label=f'Valid SWOT Measurements\n(Idealized Filter) [n={len(optimal_errors):,}]')
        
        # Add interpolated AllDates errors from SNR analysis
        print("Collecting interpolated AllDates errors for WSE CDF...")
        interpolated_errors_all = []
        
        for lake_id in df['swot_lake_id'].unique():
            lake_data = df[df['swot_lake_id'] == lake_id].copy()
            lake_data['date'] = pd.to_datetime(lake_data['date'])
            lake_data = lake_data.sort_values('date').reset_index(drop=True)
            
            if len(lake_data) < 3:
                continue
            
            if wse_optimal_result is not None and wse_optimal_result['success']:
                lake_data_target = lake_data[lake_data['swot_wse_abs_error'] <= wse_optimal_result['threshold']].copy()
                
                if len(lake_data_target) >= 3:
                    # Use consolidated interpolation function
                    interp_result = interpolate_swot_to_insitu_dates(lake_data_target, lake_data, variable='wse')
                    
                    if interp_result is not None:
                        interpolated_errors_all.extend(np.abs(interp_result['interpolated_errors']))
        
        if len(interpolated_errors_all) > 0:
            interp_sorted = np.sort(interpolated_errors_all)
            interp_p = np.arange(1, len(interp_sorted) + 1) / len(interp_sorted)
            ax1.plot(interp_sorted, interp_p * 100, color='#7570b3', linewidth=3, linestyle='--',
                    label=f'Interpolated Valid SWOT Measurements\n(Idealized Filter) [n={len(interpolated_errors_all):,}]')
        
        ax1.set_xlabel('Absolute WSE Error (m)', fontsize=18)
        ax1.set_ylabel('Cumulative Percentage (%)', fontsize=18)
        ax1.set_title('(a) CDF of SWOT WSE Errors', fontsize=18, fontweight='bold')
        ax1.set_xlim(0, 1)
        ax1.grid(True, alpha=0.3, color='gray')
        ax1.legend(fontsize=12)
    
    # ========================================
    # Panel 2: WSE Histogram
    # ========================================
    print("Creating Panel 2: WSE Histogram...")
    
    all_signed_errors = df_with_size['swot_wse_error'].dropna()
    
    if len(all_signed_errors) > 0:
        error_5th = np.percentile(all_signed_errors, 5)
        error_95th = np.percentile(all_signed_errors, 95)
        all_truncated = all_signed_errors[(all_signed_errors >= error_5th) & (all_signed_errors <= error_95th)]
        
        bins = np.linspace(error_5th, error_95th, 51)
        if wse_optimal_result['success']:
            optimal_signed_errors = wse_optimal_result['filtered_data']['swot_wse_error'].dropna()
            optimal_truncated = optimal_signed_errors[(optimal_signed_errors >= error_5th) & (optimal_signed_errors <= error_95th)]
        
        counts, _, _ = ax2.hist(all_truncated, bins=bins, alpha=1.0, edgecolor='black', color='#d95f02', 
                               label=f'Invalid SWOT Measurements\n(Idealized Filter) [n={len(all_truncated) - len(optimal_truncated):,}]')
        
        if wse_optimal_result['success']:     
            ax2.hist(optimal_truncated, bins=bins, alpha=1.0, edgecolor='black', 
                    color='#1b9e77', label=f'Valid SWOT Measurements\n(Idealized Filter) [n={len(optimal_truncated):,}]')
        
    
    ax2.set_xlabel('WSE Error (m)', fontsize=18)
    ax2.set_ylabel('Frequency', fontsize=18)
    ax2.set_title('(b) Histogram of SWOT WSE Errors', fontsize=18, fontweight='bold')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.5, color='gray')
    
    # ========================================
    # Panel 3: WSA CDF
    # ========================================
    print("Creating Panel 3: WSA CDF...")
    
    # Filter out partial lake detections for WSA analysis
    all_lakes_wsa_errors = df_with_size[df_with_size['swot_partial_f'] == 0]['wsa_abs_percentage_error'].dropna()
    
    if len(all_lakes_wsa_errors) > 0:
        all_wsa_sorted = np.sort(all_lakes_wsa_errors)
        all_wsa_p = np.arange(1, len(all_wsa_sorted) + 1) / len(all_wsa_sorted)
        
        ax3.plot(all_wsa_sorted, all_wsa_p * 100, color='#fc8d62', linewidth=2.5,
                label=f'All SWOT Measurements [n={len(all_lakes_wsa_errors):,}]')
        
        if wsa_optimal_result['success']:
            optimal_wsa_errors = wsa_optimal_result['filtered_data']['wsa_abs_percentage_error'].dropna()
            optimal_wsa_sorted = np.sort(optimal_wsa_errors)
            optimal_wsa_p = np.arange(1, len(optimal_wsa_sorted) + 1) / len(optimal_wsa_sorted)
            ax3.plot(optimal_wsa_sorted, optimal_wsa_p * 100, color='#66c2a5', linewidth=3,
                    label=f'Valid SWOT Measurements\n(Idealized Filter) [n={len(optimal_wsa_errors):,}]')
        
        # Add interpolated AllDates errors from SNR analysis
        print("Collecting interpolated AllDates errors for WSA CDF...")
        interpolated_wsa_errors_all = []
        
        for lake_id in df['swot_lake_id'].unique():
            lake_data = df[df['swot_lake_id'] == lake_id].copy()
            lake_data['date'] = pd.to_datetime(lake_data['date'])
            lake_data = lake_data.sort_values('date').reset_index(drop=True)
            
            if len(lake_data) < 3:
                continue
            
            if wsa_optimal_result is not None and wsa_optimal_result['success']:
                lake_data_target = lake_data[
                    (lake_data['wsa_abs_percentage_error'] <= wsa_optimal_result['threshold']) &
                    (lake_data['swot_partial_f'] == 0)
                ].copy()
                
                if len(lake_data_target) >= 3:
                    # Use consolidated interpolation function
                    wsa_interp_result = interpolate_swot_to_insitu_dates(lake_data_target, lake_data, variable='wsa')
                    
                    if wsa_interp_result is not None:
                        interpolated_wsa_errors_all.extend(np.abs(wsa_interp_result['interpolated_errors']))
        
        if len(interpolated_wsa_errors_all) > 0:
            interp_wsa_sorted = np.sort(interpolated_wsa_errors_all)
            interp_wsa_p = np.arange(1, len(interp_wsa_sorted) + 1) / len(interp_wsa_sorted)
            ax3.plot(interp_wsa_sorted, interp_wsa_p * 100, color='#8da0cb', linewidth=3, linestyle='--',
                    label=f'Interpolated Valid SWOT Measurements\n(Idealized Filter) [n={len(interpolated_wsa_errors_all):,}]')
        
        ax3.set_xlabel('Absolute WSA Percentage Error (%)', fontsize=18)
        ax3.set_ylabel('Cumulative Percentage (%)', fontsize=18)
        ax3.set_title('(c) CDF of SWOT WSA Percentage Errors', fontsize=18, fontweight='bold')
        ax3.set_xlim(0, 100)
        ax3.grid(True, alpha=0.3, color='gray')
        ax3.legend(fontsize=12)
    
    # ========================================
    # Panel 4: WSA Histogram
    # ========================================
    print("Creating Panel 4: WSA Histogram...")
    
    # Filter out partial lake detections for WSA analysis
    all_signed_wsa_errors = df_with_size[df_with_size['swot_partial_f'] == 0]['wsa_percentage_error'].dropna()
    
    if len(all_signed_wsa_errors) > 0:
        wsa_error_5th = np.percentile(all_signed_wsa_errors, 5)
        wsa_error_95th = np.percentile(all_signed_wsa_errors, 95)
        all_wsa_truncated = all_signed_wsa_errors[(all_signed_wsa_errors >= wsa_error_5th) & (all_signed_wsa_errors <= wsa_error_95th)]
        
        wsa_bins = np.linspace(wsa_error_5th, wsa_error_95th, 51)
        
        ax4.hist(all_wsa_truncated, bins=wsa_bins, alpha=1.0, edgecolor='black', color='#fc8d62', 
                label=f'Invalid SWOT Measurements\n(Idealized Filter) [n={len(all_wsa_truncated) - len(optimal_wsa_errors):,}]')
        
        if wsa_optimal_result['success']:
            optimal_wsa_signed_errors = wsa_optimal_result['filtered_data']['wsa_percentage_error'].dropna()
            optimal_wsa_truncated = optimal_wsa_signed_errors[(optimal_wsa_signed_errors >= wsa_error_5th) & (optimal_wsa_signed_errors <= wsa_error_95th)]
            
            ax4.hist(optimal_wsa_truncated, bins=wsa_bins, alpha=1.0, edgecolor='black', 
                    color='#66c2a5', label=f'Valid SWOT Measurements\n(Idealized Filter) [n={len(optimal_wsa_truncated):,}]')
        
    
    ax4.set_xlabel('WSA Percentage Error (%)', fontsize=18)
    ax4.set_ylabel('Frequency', fontsize=18)
    ax4.set_title('(d) Histogram of SWOT WSA Percentage Errors', fontsize=18, fontweight='bold')
    ax4.legend(fontsize=12)
    ax4.grid(True, alpha=0.5, color='gray')
    
    # ========================================
    # Panel 5: Sampling Gap Histogram
    # ========================================
    print("Creating Panel 5: Sampling Gap Histogram...")
    
    wse_threshold = 0.28
    gaps = calculate_sampling_gaps(df, wse_threshold=wse_threshold)
    
    if len(gaps) > 0:
        max_gap = max(gaps)
        min_gap = min(gaps)
        bins = range(min_gap, max_gap + 2, 1)
        
        ax5.hist(gaps, bins=bins, alpha=1.0, edgecolor='black', color='#1b9e77')
        
        mean_gap = np.mean(gaps)
        median_gap = np.median(gaps)
        
        ax5.axvline(mean_gap, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_gap:.1f} days')
        ax5.axvline(median_gap, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_gap:.1f} days')
        
        ax5.set_xlabel('Sampling Gap (days between valid observations)', fontsize=18)
        ax5.set_ylabel('Frequency', fontsize=18)
        ax5.set_title('(e) Sampling Gap Distribution', fontsize=18, fontweight='bold')
        ax5.grid(True, alpha=0.3, color='gray')
        ax5.legend(fontsize=12)
        ax5.set_xlim(0, min(max_gap, np.percentile(gaps, 98)) + 2)
    
    # ========================================
    # Panel 6: Combined Temporal Resolution
    # ========================================
    print("Creating Panel 6: Combined Temporal Resolution...")
    
    # Calculate temporal resolution for both WSE and WSA
    ice_available = 'ice' in df.columns
    
    wse_thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 1.0]
    wsa_thresholds = [10, 15, 20, 25, 30, 40, 50, 100]
    
    wse_temporal_results = calculate_temporal_resolution(df, wse_thresholds, use_ice_aware=ice_available, error_type='wse')
    wsa_temporal_results = calculate_temporal_resolution(df, wsa_thresholds, use_ice_aware=ice_available, error_type='wsa')
    
    if wse_temporal_results and wsa_temporal_results:
        # Extract data for plotting
        wse_days_per_obs = [r['overall_days_per_obs'] for r in wse_temporal_results]
        wse_thresholds_plot = [r['threshold'] for r in wse_temporal_results]
        
        wsa_days_per_obs = [r['overall_days_per_obs'] for r in wsa_temporal_results]  
        wsa_thresholds_plot = [r['threshold'] for r in wsa_temporal_results]
        
        # Create dual y-axis plot with days/obs on x-axis
        ax6_twin = ax6.twinx()
        
        # Plot WSE (left y-axis) - switched from WSA color
        line1 = ax6.plot(wse_days_per_obs, wse_thresholds_plot, 'o-', color='#1b9e77', linewidth=3, markersize=6, label='WSE')
        ax6.set_xlabel('Days per Valid Observation', fontsize=18)
        ax6.set_ylabel('WSE Absolute Error Threshold (m)', color='black', fontsize=13)
        ax6.tick_params(axis='y', labelcolor='black')
        
        # Plot WSA (right y-axis) - switched from WSE color
        line2 = ax6_twin.plot(wsa_days_per_obs, wsa_thresholds_plot, 's-', color='#66c2a5', linewidth=3, markersize=6, label='WSA')
        ax6_twin.set_ylabel('WSA Percentage Error Threshold (%)', color='black', fontsize=13)
        ax6_twin.tick_params(axis='y', labelcolor='black')
        
        ax6.set_title('(f) Temporal Resolution vs Error Thresholds', fontsize=18, fontweight='bold')
        ax6.grid(True, alpha=0.3, color='gray')
        
        # Combine legends
        lines1, labels1 = ax6.get_legend_handles_labels()
        lines2, labels2 = ax6_twin.get_legend_handles_labels()
        ax6.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=12)
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs('experiments/results/3.1', exist_ok=True)
    plt.savefig('experiments/results/3.1/combined_comprehensive_analysis.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return fig

def create_target_vs_interpolated_comparison_plot(df, wse_optimal_result, wsa_optimal_result):
    """Create comparison plot of Target (direct sampling) vs Target-Daily (interpolated) distributions"""
    
    print("\nCreating Target vs Target-Daily comparison plot...")
    
    # Set up plotting style
    plt.rcParams.update({
        'font.size': 14,
        'text.color': 'black',
        'axes.labelcolor': 'black',
        'axes.edgecolor': 'black',
        'xtick.color': 'black',
        'ytick.color': 'black'
    })
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # ========================================
    # WSE Comparison (Top Row)
    # ========================================
    
    if wse_optimal_result is not None and wse_optimal_result['success']:
        # Get Target distribution errors (direct sampling)
        target_wse_data = wse_optimal_result['filtered_data']
        target_wse_errors = target_wse_data['swot_wse_abs_error'].dropna()
        
        # Collect Target-Daily errors (interpolated)
        interpolated_wse_errors_all = []
        
        for lake_id in df['swot_lake_id'].unique():
            lake_data = df[df['swot_lake_id'] == lake_id].copy()
            lake_data['date'] = pd.to_datetime(lake_data['date'])
            lake_data = lake_data.sort_values('date').reset_index(drop=True)
            
            if len(lake_data) < 3:
                continue
            
            lake_data_target = lake_data[lake_data['swot_wse_abs_error'] <= wse_optimal_result['threshold']].copy()
            
            if len(lake_data_target) >= 3:
                interp_result = interpolate_swot_to_insitu_dates(lake_data_target, lake_data, variable='wse')
                
                if interp_result is not None:
                    interpolated_wse_errors_all.extend(np.abs(interp_result['interpolated_errors']))
        
        # Panel 1: WSE CDF Comparison
        if len(target_wse_errors) > 0:
            target_sorted = np.sort(target_wse_errors)
            target_p = np.arange(1, len(target_sorted) + 1) / len(target_sorted)
            ax1.plot(target_sorted, target_p * 100, color='#b2abd2', linewidth=3,
                    label=f'Target (Direct) [n={len(target_wse_errors):,}]')
        
        if len(interpolated_wse_errors_all) > 0:
            interp_sorted = np.sort(interpolated_wse_errors_all)
            interp_p = np.arange(1, len(interp_sorted) + 1) / len(interp_sorted)
            ax1.plot(interp_sorted, interp_p * 100, color='#fdb863', linewidth=3, linestyle='--',
                    label=f'Target-Daily (Interpolated) [n={len(interpolated_wse_errors_all):,}]')
        
        ax1.set_xlabel('Absolute WSE Error (m)', fontsize=14)
        ax1.set_ylabel('Cumulative Percentage (%)', fontsize=14)
        ax1.set_title('WSE Error CDF: Target vs Target-Daily', fontsize=14, fontweight='bold')
        ax1.set_xlim(0, 0.5)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=11)
        
        # Panel 2: WSE Histogram Comparison (Density)
        if len(target_wse_errors) > 0 and len(interpolated_wse_errors_all) > 0:
            # Use same bins for both
            max_error = max(target_wse_errors.max(), max(interpolated_wse_errors_all))
            bins = np.linspace(0, min(max_error, 0.5), 51)
            
            ax2.hist(target_wse_errors, bins=bins, alpha=0.7, density=True, 
                    edgecolor='#b2abd2', color='#b2abd2', 
                    label=f'Target (Direct) [n={len(target_wse_errors):,}]')
            
            ax2.hist(interpolated_wse_errors_all, bins=bins, alpha=0.7, density=True,
                    edgecolor='#fdb863', color='#fdb863', 
                    label=f'Target-Daily (Interpolated) [n={len(interpolated_wse_errors_all):,}]')
        
        ax2.set_xlabel('Absolute WSE Error (m)', fontsize=14)
        ax2.set_ylabel('Probability Density', fontsize=14)
        ax2.set_title('WSE Error Distribution: Target vs Target-Daily', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
    
    # ========================================
    # WSA Comparison (Bottom Row)
    # ========================================
    
    if wsa_optimal_result is not None and wsa_optimal_result['success']:
        # Get Target distribution errors (direct sampling)
        target_wsa_data = wsa_optimal_result['filtered_data']
        target_wsa_errors = target_wsa_data['wsa_abs_percentage_error'].dropna()
        
        # Collect Target-Daily errors (interpolated)
        interpolated_wsa_errors_all = []
        
        for lake_id in df['swot_lake_id'].unique():
            lake_data = df[df['swot_lake_id'] == lake_id].copy()
            lake_data['date'] = pd.to_datetime(lake_data['date'])
            lake_data = lake_data.sort_values('date').reset_index(drop=True)
            
            if len(lake_data) < 3:
                continue
            
            lake_data_target = lake_data[
                (lake_data['wsa_abs_percentage_error'] <= wsa_optimal_result['threshold']) &
                (lake_data['swot_partial_f'] == 0)
            ].copy()
            
            if len(lake_data_target) >= 3:
                wsa_interp_result = interpolate_swot_to_insitu_dates(lake_data_target, lake_data, variable='wsa')
                
                if wsa_interp_result is not None:
                    interpolated_wsa_errors_all.extend(np.abs(wsa_interp_result['interpolated_errors']))
        
        # Panel 3: WSA CDF Comparison
        if len(target_wsa_errors) > 0:
            target_wsa_sorted = np.sort(target_wsa_errors)
            target_wsa_p = np.arange(1, len(target_wsa_sorted) + 1) / len(target_wsa_sorted)
            ax3.plot(target_wsa_sorted, target_wsa_p * 100, color='#b2abd2', linewidth=3,
                    label=f'Target (Direct) [n={len(target_wsa_errors):,}]')
        
        if len(interpolated_wsa_errors_all) > 0:
            interp_wsa_sorted = np.sort(interpolated_wsa_errors_all)
            interp_wsa_p = np.arange(1, len(interp_wsa_sorted) + 1) / len(interp_wsa_sorted)
            ax3.plot(interp_wsa_sorted, interp_wsa_p * 100, color='#fdb863', linewidth=3, linestyle='--',
                    label=f'Target-Daily (Interpolated) [n={len(interpolated_wsa_errors_all):,}]')
        
        ax3.set_xlabel('Absolute WSA Percentage Error (%)', fontsize=14)
        ax3.set_ylabel('Cumulative Percentage (%)', fontsize=14)
        ax3.set_title('WSA Error CDF: Target vs Target-Daily', fontsize=14, fontweight='bold')
        ax3.set_xlim(0, 50)
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=11)
        
        # Panel 4: WSA Histogram Comparison (Density)
        if len(target_wsa_errors) > 0 and len(interpolated_wsa_errors_all) > 0:
            # Use same bins for both
            max_wsa_error = max(target_wsa_errors.max(), max(interpolated_wsa_errors_all))
            wsa_bins = np.linspace(0, min(max_wsa_error, 50), 51)
            
            ax4.hist(target_wsa_errors, bins=wsa_bins, alpha=0.7, density=True,
                    edgecolor='#b2abd2', color='#b2abd2',
                    label=f'Target (Direct) [n={len(target_wsa_errors):,}]')
            
            ax4.hist(interpolated_wsa_errors_all, bins=wsa_bins, alpha=0.7, density=True,
                    edgecolor='#fdb863', color='#fdb863',
                    label=f'Target-Daily (Interpolated) [n={len(interpolated_wsa_errors_all):,}]')
        
        ax4.set_xlabel('Absolute WSA Percentage Error (%)', fontsize=14)
        ax4.set_ylabel('Probability Density', fontsize=14)
        ax4.set_title('WSA Error Distribution: Target vs Target-Daily', fontsize=14, fontweight='bold')
        ax4.legend(fontsize=11)
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs('experiments/results/3.1', exist_ok=True)
    plt.savefig('experiments/results/3.1/target_vs_interpolated_comparison.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("Target vs Target-Daily comparison plot saved to: experiments/results/3.1/target_vs_interpolated_comparison.png")
    
    return fig

def create_interpolation_error_vs_gap_plot(df, wse_optimal_result):
    """Create scatter plot of interpolation error vs sampling gap"""
    
    print("\nCreating interpolation error vs sampling gap plot...")
    
    # Set up plotting style
    plt.rcParams.update({
        'font.size': 14,
        'text.color': 'black',
        'axes.labelcolor': 'black',
        'axes.edgecolor': 'black',
        'xtick.color': 'black',
        'ytick.color': 'black'
    })
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Collect interpolation errors and their corresponding sampling gaps
    interpolation_errors = []
    sampling_gaps = []
    
    for lake_id in df['swot_lake_id'].unique():
        lake_data = df[df['swot_lake_id'] == lake_id].copy()
        lake_data['date'] = pd.to_datetime(lake_data['date'])
        lake_data = lake_data.sort_values('date').reset_index(drop=True)
        
        if len(lake_data) < 3:
            continue
        
        if wse_optimal_result is not None and wse_optimal_result['success']:
            lake_data_target = lake_data[lake_data['swot_wse_abs_error'] <= wse_optimal_result['threshold']].copy()
            
            if len(lake_data_target) >= 3:
                # Use consolidated interpolation function
                interp_result = interpolate_swot_to_insitu_dates(lake_data_target, lake_data, variable='wse')
                
                if interp_result is not None:
                    interpolated_errors_vals = interp_result['interpolated_errors']
                    valid_dates = interp_result['valid_dates']
                    target_dates_with_data = interp_result['target_dates_with_data']
                    
                    # Calculate sampling gap for each interpolated point
                    for i, interp_date in enumerate(valid_dates):
                        # Find the closest SWOT observation before this interpolated date
                        swot_dates_before = target_dates_with_data[target_dates_with_data['date'] <= pd.to_datetime(interp_date)]
                        
                        if len(swot_dates_before) > 0:
                            last_swot_date = swot_dates_before['date'].max()
                            gap_days = (pd.to_datetime(interp_date) - last_swot_date).days
                            
                            # Only include gaps > 0 days and <= 50 days (actual interpolation)
                            if 0 < gap_days <= 50:
                                error_val = np.abs(interpolated_errors_vals[i])
                                # Debug: Check for unreasonable errors
                                if error_val > 10.0:  # WSE errors > 10m are suspicious
                                    print(f"Warning: Large interpolation error {error_val:.3f}m for lake {lake_id}, gap {gap_days} days")
                                interpolation_errors.append(error_val)
                                sampling_gaps.append(gap_days)
    
    if len(interpolation_errors) > 0 and len(sampling_gaps) > 0:
        # Create scatter plot
        ax.scatter(sampling_gaps, interpolation_errors, alpha=0.3, s=20, color='blue', edgecolors='none')
        
        # Add trend line (moving average in bins)
        max_gap = max(sampling_gaps)
        gap_bins = np.linspace(0, min(max_gap, 50), 21)  # Limit to 50 days for clarity
        bin_centers = []
        bin_means = []
        
        for i in range(len(gap_bins)-1):
            mask = (np.array(sampling_gaps) >= gap_bins[i]) & (np.array(sampling_gaps) < gap_bins[i+1])
            if mask.sum() >= 10:  # At least 10 points per bin
                bin_centers.append((gap_bins[i] + gap_bins[i+1]) / 2)
                bin_means.append(np.mean(np.array(interpolation_errors)[mask]))
        
        if len(bin_centers) > 2:
            ax.plot(bin_centers, bin_means, 'r-', linewidth=3, label='Binned Mean')
            ax.legend()
        
        ax.set_xlabel('Days Since Last SWOT Observation', fontsize=16)
        ax.set_ylabel('Interpolation Error (m)', fontsize=16)
        ax.set_title('WSE Interpolation Error vs Sampling Gap\n(Target Distribution)', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, min(max_gap, 50))
        
        # Set reasonable y-axis limit for WSE errors
        if len(interpolation_errors) > 0:
            max_error_to_show = min(max(interpolation_errors), 2.0)  # Cap at 2m for visibility
            ax.set_ylim(0, max_error_to_show)
        
        # Add sample size info
        ax.text(0.05, 0.95, f'n = {len(interpolation_errors):,} interpolated points', 
               transform=ax.transAxes, fontsize=12, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    else:
        ax.text(0.5, 0.5, 'No interpolation data available', ha='center', va='center', 
               transform=ax.transAxes, fontsize=14)
        ax.set_xlabel('Days Since Last SWOT Observation', fontsize=16)
        ax.set_ylabel('Interpolation Error (m)', fontsize=16)
        ax.set_title('WSE Interpolation Error vs Sampling Gap\n(Target Distribution)', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs('experiments/results/3.1', exist_ok=True)
    plt.savefig('experiments/results/3.1/interpolation_error_vs_gap.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return fig

def main():
    """Main execution function"""
    
    print("COMBINED COMPREHENSIVE ANALYSIS")
    print("=" * 50)
    
    # Configuration - easily toggle PID0 filter
    USE_PID0_ONLY = False  # Set to False to use all data versions
    
    # Load data
    print("Loading benchmark data...")
    df = load_benchmark_data_with_ice()
    
    # Filter to post 2023-07-21
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] > '2023-07-21'].copy()
    
    # Keep full dataset for temporal resolution calculations
    df_full_temporal = df.copy()
    
    # Apply PID0 filter if requested (for error calculations only)
    if USE_PID0_ONLY:
        print("Filtering to PID0 data only for error calculations...")
        df = df[df['swot_crid']=='PID0'].copy()
    
    # Filter lakes with <5 SWOT observations (apply to both datasets)
    df = filter_lakes_by_observation_count(df, min_observations=5)
    df_full_temporal = filter_lakes_by_observation_count(df_full_temporal, min_observations=5)
    
    if df.empty:
        print("No data remaining after filtering!")
        return
    
    # Calculate WSA percentage errors for both datasets
    df = calculate_wsa_percentage_errors(df)
    df_full_temporal = calculate_wsa_percentage_errors(df_full_temporal)
    
    # Load lake size mapping and add categories
    lake_size_dict = load_lake_size_mapping()
    df = add_lake_size_categories(df, lake_size_dict)
    
    # Filter to observations with known lake sizes
    df_with_size = df[df['lake_size_category'] != 'Unknown size'].copy()
    print(f"Final data shape: {len(df_with_size)} observations from {df_with_size['swot_lake_id'].nunique()} lakes")
    
    # Create comprehensive plot using the filtered dataset
    fig = create_combined_comprehensive_plot(df, df_with_size)
    
    # Calculate optimal WSE threshold for SNR analysis
    wse_optimal_result = find_threshold_for_target_std(df, target_std=0.1)
    
    # Calculate optimal WSA threshold for SNR analysis
    wsa_optimal_result = find_threshold_for_target_std_percentage(df, target_std=15.0)
    
    # Create interpolation error vs gap plot
    #gap_fig = create_interpolation_error_vs_gap_plot(df, wse_optimal_result)
    
    # Create Target vs Target-Daily comparison plot
    #comparison_fig = create_target_vs_interpolated_comparison_plot(df, wse_optimal_result, wsa_optimal_result)
    
    # Calculate SNR by lake (both WSE and WSA)
    #snr_df = calculate_snr_by_lake(df_with_size, wse_optimal_result)
    #wsa_snr_df = calculate_wsa_snr_by_lake(df_with_size, wsa_optimal_result)
    
    # Calculate NSE by lake (both WSE and WSA)
    nse_df = calculate_nse_by_lake(df_with_size, wse_optimal_result)
    wsa_nse_df = calculate_wsa_nse_by_lake(df_with_size, wsa_optimal_result)
    
    # Create SNR analysis plot (combined WSE and WSA)
    #snr_fig = create_snr_analysis_plot(snr_df, wsa_snr_df, df_with_size)
    
    # Export error metrics to CSV, passing both datasets
    wse_csv_path, wsa_csv_path = export_error_metrics_csv(df_with_size, df_full_temporal=df_full_temporal)
    
    # Export SNR metrics to CSV (both WSE and WSA)
    #snr_summary_csv, snr_lake_csv = export_snr_metrics_csv(snr_df)
    
    # Export WSA SNR metrics separately
    #wsa_snr_summary_csv, wsa_snr_lake_csv = export_wsa_snr_metrics_csv(wsa_snr_df)
    
    # Export NSE metrics to CSV (both WSE and WSA)
    nse_summary_csv, nse_lake_csv = export_nse_metrics_csv(nse_df)
    
    # Export WSA NSE metrics separately
    wsa_nse_summary_csv, wsa_nse_lake_csv = export_wsa_nse_metrics_csv(wsa_nse_df)
    '''
    print("\n=== ANALYSIS COMPLETE ===")
    print("Results saved to experiments/results/3.1/combined_comprehensive_analysis.png")
    print("SNR analysis saved to experiments/results/3.1/snr_analysis.png")
    print(f"WSE error metrics saved to: {wse_csv_path}")
    print(f"WSA error metrics saved to: {wsa_csv_path}")
    print(f"WSE SNR summary statistics saved to: {snr_summary_csv}")
    print(f"Per-lake WSE SNR data saved to: {snr_lake_csv}")
    print(f"WSA SNR summary statistics saved to: {wsa_snr_summary_csv}")
    print(f"Per-lake WSA SNR data saved to: {wsa_snr_lake_csv}")
    print(f"WSE NSE summary statistics saved to: {nse_summary_csv}")
    print(f"Per-lake WSE NSE data saved to: {nse_lake_csv}")
    print(f"WSA NSE summary statistics saved to: {wsa_nse_summary_csv}")
    print(f"Per-lake WSA NSE data saved to: {wsa_nse_lake_csv}")
    '''
if __name__ == "__main__":
    main()