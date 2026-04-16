#!/usr/bin/env python3
"""
Storage Uncertainty Attribution Analysis

Implements first-order error propagation to attribute storage uncertainty to:
1. WSE measurement errors 
2. Elevation-area relationship errors (from WSA uncertainty)
3. Comparison with observed storage errors

Uses analytical derivatives of the trapezoidal integration storage calculation.

Updated to run all uncertainty combinations from input_uncertainties.csv.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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



def calculate_storage_uncertainty_components_two_way(df, wse_std=0.1, wse_temporal_std=0.0, wse_filt_std=0.0, wsa_percent_error=15.0):
    """
    Calculate uncertainty components for storage estimates using two-component error propagation:
    1. WSE measurement uncertainty (base + temporal interpolation + filtering)
    2. WSA measurement uncertainty (affects bathymetry)
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Lake data with WSE and WSA columns
    wse_std : float
        Base WSE measurement uncertainty (meters)
    wse_temporal_std : float
        Temporal interpolation uncertainty (meters) - 0 for discrete, >0 for continuous
    wse_filt_std : float
        Filtering uncertainty (meters) - additional error from filtered data
    wsa_percent_error : float  
        WSA measurement uncertainty (percent)
        
    Returns:
    --------
    pandas.DataFrame with two-component uncertainty breakdown
    """
    
    results = []
    
    # Calculate combined WSE uncertainty (linear addition of all WSE components)
    wse_total_std = wse_std + wse_temporal_std + wse_filt_std
    
    # Process each observation
    for idx, row in df.iterrows():
        
        # Skip if missing essential data
        if pd.isna(row['swot_wse_anomaly']) or pd.isna(row['wsa']):
            continue
            
        # Basic parameters
        wse = row['stage_anomaly_swotdates']  # WSE above some reference
        current_area = row['wsa']  # Current area in km²
        height_above_ref = abs(wse)  # meters above reference
        
        # Convert WSA percentage error to absolute area uncertainty  
        sigma_area_km2 = current_area * (wsa_percent_error / 100.0)  # km²
        
        # For storage calculation S = ∫[ref to WSE] A(h) dh using trapezoidal rule
        # Analytical partial derivatives for two-component separation:
        
        # Component 1: Total WSE uncertainty (base + temporal)
        # ∂S/∂WSE = A(WSE) 
        partial_S_partial_WSE = current_area  # km²
        sigma_storage_wse_total = abs((partial_S_partial_WSE * wse_total_std) / 1000.0)  # km³
        
        # Component 2: WSA measurement uncertainty (bathymetry fitting)  
        # ∂S/∂A ≈ height_above_ref for changes in area estimates
        partial_S_partial_A = height_above_ref / 1000.0  # Convert m to km for units
        sigma_storage_area = abs(partial_S_partial_A * sigma_area_km2)  # km³
        
        # Total predicted uncertainty (assuming independence)
        sigma_storage_total = np.sqrt(sigma_storage_wse_total**2 + sigma_storage_area**2)
        
        # Attribution percentages (consistent with linear WSE addition)
        if sigma_storage_total > 0:
            # Calculate linear sum for proper percentage attribution when WSE components add linearly
            linear_total = sigma_storage_wse_total + sigma_storage_area
            
            wse_contribution_pct = (sigma_storage_wse_total / linear_total) * 100
            area_contribution_pct = (sigma_storage_area / linear_total) * 100
            
            # Break down WSE contribution into base, temporal, and filtering (linear contributions)
            if linear_total > 0:
                wse_base_contribution_pct = (abs(partial_S_partial_WSE * wse_std / 1000.0) / linear_total) * 100
                wse_temporal_contribution_pct = (abs(partial_S_partial_WSE * wse_temporal_std / 1000.0) / linear_total) * 100
                wse_filt_contribution_pct = (abs(partial_S_partial_WSE * wse_filt_std / 1000.0) / linear_total) * 100
            else:
                wse_base_contribution_pct = 0.0
                wse_temporal_contribution_pct = 0.0
                wse_filt_contribution_pct = 0.0
        else:
            wse_contribution_pct = np.nan
            wse_base_contribution_pct = np.nan
            wse_temporal_contribution_pct = np.nan
            wse_filt_contribution_pct = np.nan
            area_contribution_pct = np.nan
        
        # Calculate equal weighting (inverse height weighting)
        height_weight = 1.0 / max(abs(height_above_ref), 0.1)  # Avoid division by zero with minimum 0.1m
        
        # Store results
        result = {
            'date': row['date'],
            'swot_lake_id': row.get('swot_lake_id', ''),
            'wse_anomaly': wse,
            'area_km2': current_area,
            'height_above_ref': height_above_ref,
            'height_weight': height_weight,
            'sigma_area_km2': sigma_area_km2,
            'wse_std': wse_std,
            'wse_temporal_std': wse_temporal_std,
            'wse_filt_std': wse_filt_std,
            'wse_total_std': wse_total_std,
            
            # Two-component breakdown
            'sigma_storage_wse_total_km3': sigma_storage_wse_total,
            'sigma_storage_area_km3': sigma_storage_area,
            'sigma_storage_total_km3': sigma_storage_total,
            
            # Attribution percentages
            'wse_total_contribution_pct': wse_contribution_pct,
            'wse_base_contribution_pct': wse_base_contribution_pct,
            'wse_temporal_contribution_pct': wse_temporal_contribution_pct,
            'wse_filt_contribution_pct': wse_filt_contribution_pct,
            'area_contribution_pct': area_contribution_pct,
            
            # Partial derivatives for reference
            'partial_S_partial_WSE': partial_S_partial_WSE,
            'partial_S_partial_A': partial_S_partial_A,
        }
        results.append(result)
    
    return pd.DataFrame(results)

def get_observed_error_from_benchmark_summary(model_info):
    """Get observed error metrics from pre-calculated benchmark summary file
    
    Parameters:
    -----------
    model_info : dict
        Model information (model, swot_filter, temporal)
        
    Returns:
    --------
    dict : Observed error metrics from benchmark analysis
    """
    
    # Load benchmark summary file
    benchmark_file = PROJECT_ROOT / "analysis/storage_estimation_assessment/results/benchmark_storage_variants_summary_stats.csv"
    
    if not benchmark_file.exists():
        print(f"Warning: Benchmark file {benchmark_file} not found")
        return None
        
    benchmark_df = pd.read_csv(benchmark_file)
    
    # Map our naming to benchmark file naming
    model_map = {'swot': 'swot', 'swots2': 'swots2', 'static': 'static'}
    filter_map = {'optimal': 'opt', 'filtered': 'filt'}
    temporal_map = {'discrete': 'dis', 'daily': 'con'}
    
    model_name = model_map.get(model_info['model'])
    filter_name = filter_map.get(model_info['swot_filter'])
    temporal_name = temporal_map.get(model_info['temporal'])
    
    if not all([model_name, filter_name, temporal_name]):
        print(f"Warning: Could not map model info {model_info}")
        return None
    
    # Find matching row
    match = benchmark_df[
        (benchmark_df['model'] == model_name) &
        (benchmark_df['filter_type'] == filter_name) &
        (benchmark_df['temporal_type'] == temporal_name)
    ]
    
    if len(match) == 0:
        print(f"Warning: No benchmark data found for {model_name}-{filter_name}-{temporal_name}")
        return None
    
    if len(match) > 1:
        print(f"Warning: Multiple matches found for {model_name}-{filter_name}-{temporal_name}, using first")
    
    row = match.iloc[0]
    
    #print(f"  Found benchmark data: MAE={row['mae_median']:.4f} km³, n_lakes={row['n_lakes']}")
    
    return {
        'observed_mae': row['mae_median'],
        'observed_rmse': row['rmse_median'], 
        'observed_std': row['std_error_median'],
        'observed_nse_median': row['nse_median'],
        'observed_nse_mean': row['nse_mean'],
        'n_observations': row['n_observations']  # Changed from n_lakes to n_observations
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
        
        return round(weighted_mean, 4), round(weighted_median, 4)  # More precision for km³ values
    else:
        # Regular unweighted statistics
        return round(valid_data.mean(), 4), round(valid_data.median(), 4)

def create_uncertainty_attribution_plots_two_way(uncertainty_df, output_dir, analysis_type="discrete"):
    """Create plots showing two-component uncertainty attribution results"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter to valid data
    valid_data = uncertainty_df.dropna(subset=['sigma_storage_total_km3', 
                                               'wse_contribution_pct', 
                                               'area_contribution_pct'])
    
    if len(valid_data) == 0:
        print("No valid data for plotting")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Two-Component Storage Uncertainty Attribution Analysis ({analysis_type.title()})', fontsize=16, fontweight='bold')
    
    # 1. Uncertainty magnitude distributions
    axes[0, 0].hist(valid_data['sigma_storage_wse_total_km3'] * 1000, bins=30, alpha=0.7, 
                    label='Total WSE', color='blue')
    axes[0, 0].hist(valid_data['sigma_storage_area_km3'] * 1000, bins=30, alpha=0.7, 
                    label='WSA', color='red')  
    axes[0, 0].hist(valid_data['sigma_storage_total_km3'] * 1000, bins=30, alpha=0.7, 
                    label='Total', color='green')
    axes[0, 0].set_xlabel('Storage Uncertainty (m³ × 10⁶)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Two-Component Uncertainty Magnitudes')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Two-component attribution percentages
    axes[0, 1].hist(valid_data['wse_contribution_pct'], bins=30, alpha=0.7, 
                    label='WSE (Base + Temporal)', color='blue')
    axes[0, 1].hist(valid_data['area_contribution_pct'], bins=30, alpha=0.7, 
                    label='WSA', color='red')
    axes[0, 1].set_xlabel('Contribution Percentage (%)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Two-Component Attribution Percentages')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Stacked bar chart of mean contributions with WSE breakdown
    wse_base_mean = valid_data['wse_base_contribution_pct'].mean()
    wse_temporal_mean = valid_data['wse_temporal_contribution_pct'].mean()
    area_mean = valid_data['area_contribution_pct'].mean()
    
    component_means = [wse_base_mean, wse_temporal_mean, area_mean]
    component_labels = ['WSE Base\n(Measurement)', 'WSE Temporal\n(Interpolation)', 'WSA\n(Bathymetry)']
    colors = ['blue', 'lightblue', 'red']
    
    axes[0, 2].bar(['Storage Uncertainty'], [100], color='lightgray', alpha=0.3, label='Total')
    bottom = 0
    for i, (mean_val, label, color) in enumerate(zip(component_means, component_labels, colors)):
        axes[0, 2].bar(['Storage Uncertainty'], [mean_val], bottom=bottom, 
                       color=color, alpha=0.7, label=label)
        # Add percentage text
        axes[0, 2].text(0, bottom + mean_val/2, f'{mean_val:.1f}%', 
                       ha='center', va='center', fontweight='bold', color='white')
        bottom += mean_val
    
    axes[0, 2].set_ylabel('Contribution Percentage (%)')
    axes[0, 2].set_title('Mean Attribution Breakdown')
    axes[0, 2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Uncertainty vs lake size
    scatter_data = valid_data.dropna(subset=['area_km2'])
    if len(scatter_data) > 0:
        scatter = axes[1, 0].scatter(scatter_data['area_km2'], 
                                   scatter_data['sigma_storage_total_km3'] * 1000, 
                                   alpha=0.6, c=scatter_data['wse_contribution_pct'], 
                                   cmap='RdYlBu_r', s=30)
        axes[1, 0].set_xlabel('Lake Area (km²)')
        axes[1, 0].set_ylabel('Total Storage Uncertainty (m³ × 10⁶)')
        axes[1, 0].set_title('Uncertainty vs Lake Size')
        axes[1, 0].set_xscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=axes[1, 0])
        cbar.set_label('WSE Contribution (%)')
    
    # 5. Predicted vs Observed (if available)
    comparison_data = uncertainty_df.dropna(subset=['observed_storage_anomaly', 'benchmark_storage_anomaly', 
                                                    'sigma_storage_total_km3'])
    if len(comparison_data) > 10:
        comparison_data['observed_error_abs'] = np.abs(comparison_data['observed_storage_anomaly'] - 
                                                       comparison_data['benchmark_storage_anomaly'])
        
        axes[1, 1].scatter(comparison_data['sigma_storage_total_km3'] * 1000, 
                          comparison_data['observed_error_abs'] * 1000, alpha=0.6)
        
        # Add 1:1 line
        max_val = max(comparison_data['sigma_storage_total_km3'].max() * 1000,
                      comparison_data['observed_error_abs'].max() * 1000)
        axes[1, 1].plot([0, max_val], [0, max_val], 'r--', alpha=0.7, label='1:1 Line')
        
        axes[1, 1].set_xlabel('Predicted Uncertainty (m³ × 10⁶)')
        axes[1, 1].set_ylabel('Observed Error Magnitude (m³ × 10⁶)')
        axes[1, 1].set_title('Predicted vs Observed Uncertainty')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'Insufficient data\nfor comparison', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Predicted vs Observed Uncertainty')
    
    # 6. Component comparison scatter plot
    if len(valid_data) > 0:
        axes[1, 2].scatter(valid_data['wse_base_contribution_pct'], 
                          valid_data['area_contribution_pct'], 
                          alpha=0.6, c=valid_data['wse_temporal_contribution_pct'], 
                          cmap='viridis', s=30)
        axes[1, 2].set_xlabel('WSE Base Contribution (%)')
        axes[1, 2].set_ylabel('WSA Contribution (%)')
        axes[1, 2].set_title('Component Contribution Relationships')
        axes[1, 2].grid(True, alpha=0.3)
        cbar2 = plt.colorbar(axes[1, 2].collections[0], ax=axes[1, 2])
        cbar2.set_label('WSE Temporal Contribution (%)')
    
    plt.tight_layout()
    
    # Save plot
    output_file = output_dir / f"storage_uncertainty_attribution_two_way_{analysis_type}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved two-component uncertainty attribution plot to {output_file}")
    plt.show()
    
    return fig

def process_lake_file(csv_file, wse_std=0.1, wse_temporal_std=0.0, wse_filt_std=0.0, wsa_percent_error=15.0, model_info=None):
    """Process a single lake file and calculate uncertainty attribution"""
    
    try:
        # Read CSV with string dtypes for ID columns
        df = pd.read_csv(csv_file, dtype={'gage_id': str, 'swot_lake_id': str})
        
        if df.empty:
            return None
            
        lake_id = csv_file.stem.replace("_daily", "")
        
        # We only need WSE and area data for uncertainty calculations
        # Check for basic required columns
        if 'swot_wse_anomaly' not in df.columns or 'wsa' not in df.columns:
            print(f"  Skipping {lake_id}: Missing basic WSE or area columns")
            return None
            
        print(f"Processing {lake_id}...")
        
        # Filter to rows with valid WSE and area data
        df_filtered = df[(df['swot_wse_anomaly'].notna()) & (df['wsa'].notna())].copy()
        
        if len(df_filtered) < 2:
            print(f"  Insufficient data for {lake_id}")
            return None
            
        # Calculate uncertainty breakdown
        uncertainty_results = calculate_storage_uncertainty_components_two_way(
            df_filtered, wse_std=wse_std, wse_temporal_std=wse_temporal_std,
            wse_filt_std=wse_filt_std, wsa_percent_error=wsa_percent_error)
        
        uncertainty_results['lake_id'] = lake_id
        
        print(f"  Processed {len(uncertainty_results)} observations for {lake_id}")
        
        return uncertainty_results
        
    except Exception as e:
        print(f"Error processing {csv_file.name}: {e}")
        return None

def run_uncertainty_combination(wse_std, wse_temporal_std, wse_filt_std, wsa_percent_error, analysis_type, model_info, use_height_weighting=False):
    """Run storage uncertainty analysis for a single uncertainty combination
    
    Parameters:
    -----------
    wse_std : float
        Base WSE measurement uncertainty (meters)
    wse_temporal_std : float  
        Temporal interpolation uncertainty (meters)
    wse_filt_std : float
        Filtering uncertainty (meters)
    wsa_percent_error : float
        WSA measurement uncertainty (percent)
    analysis_type : str
        "discrete" or "daily" for labeling outputs
    model_info : dict
        Model information (model, swot_filter, temporal)
    
    Returns:
    --------
    dict : Summary statistics for this combination
    """
    
    # Define paths
    data_dir = PROJECT_ROOT / "data/benchmark_timeseries"
    
    # Get CSV files
    csv_files = list(data_dir.glob("*_daily.csv"))
    
    # Process all files
    all_results = []
    skipped_count = 0
    processed_count = 0
    
    for csv_file in csv_files:
        result = process_lake_file(csv_file, wse_std=wse_std, 
                                 wse_temporal_std=wse_temporal_std,
                                 wse_filt_std=wse_filt_std,
                                 wsa_percent_error=wsa_percent_error,
                                 model_info=model_info)
        if result is not None and len(result) > 0:
            all_results.append(result)
            processed_count += 1
        else:
            skipped_count += 1
    
    print(f"  Processed {processed_count} files, skipped {skipped_count} files")
    
    if len(all_results) == 0:
        return None
    
    # Combine all results
    combined_results = pd.concat(all_results, ignore_index=True)
    
    # Get observed errors from benchmark summary
    benchmark_stats = get_observed_error_from_benchmark_summary(model_info)
    
    # Calculate summary statistics
    summary_stats = {
        'model': model_info['model'],
        'swot_filter': model_info['swot_filter'],
        'temporal': model_info['temporal'],
        'propagated_wse': wse_std,
        'propagated_wse_temporal': wse_temporal_std,
        'propagated_wse_filt': wse_filt_std,
        'propagated_wsa': wsa_percent_error,
        'n_observations': len(combined_results),
        'n_lakes': combined_results['lake_id'].nunique(),
    }
    
    # Calculate statistics using weighted or unweighted approach
    wse_total_mean, wse_total_median = calculate_weighted_statistics(
        combined_results['wse_total_contribution_pct'], combined_results['height_weight'], use_height_weighting)
    summary_stats['wse_total_contribution_mean'] = wse_total_mean
    summary_stats['wse_total_contribution_median'] = wse_total_median
    
    wse_base_mean, wse_base_median = calculate_weighted_statistics(
        combined_results['wse_base_contribution_pct'], combined_results['height_weight'], use_height_weighting)
    summary_stats['wse_base_contribution_mean'] = wse_base_mean
    summary_stats['wse_base_contribution_median'] = wse_base_median
    
    wse_temp_mean, wse_temp_median = calculate_weighted_statistics(
        combined_results['wse_temporal_contribution_pct'], combined_results['height_weight'], use_height_weighting)
    summary_stats['wse_temporal_contribution_mean'] = wse_temp_mean
    summary_stats['wse_temporal_contribution_median'] = wse_temp_median
    
    wse_filt_mean, wse_filt_median = calculate_weighted_statistics(
        combined_results['wse_filt_contribution_pct'], combined_results['height_weight'], use_height_weighting)
    summary_stats['wse_filt_contribution_mean'] = wse_filt_mean
    summary_stats['wse_filt_contribution_median'] = wse_filt_median
    
    area_mean, area_median = calculate_weighted_statistics(
        combined_results['area_contribution_pct'], combined_results['height_weight'], use_height_weighting)
    summary_stats['area_contribution_mean'] = area_mean
    summary_stats['area_contribution_median'] = area_median
    
    total_mean, total_median = calculate_weighted_statistics(
        combined_results['sigma_storage_total_km3'], combined_results['height_weight'], use_height_weighting)
    summary_stats['total_uncertainty_mean'] = total_mean
    summary_stats['total_uncertainty_median'] = total_median
    
    summary_stats.update({
        'analysis_type': analysis_type
    })
    
    # Add benchmark statistics if available
    if benchmark_stats:
        predicted_uncertainty_mean = total_mean  # Use already calculated weighted/unweighted mean
        predicted_uncertainty_median = total_median  # Use already calculated weighted/unweighted median
        
        # Calculate ratios using MAE as primary metric
        ratio_mae = predicted_uncertainty_median / benchmark_stats['observed_mae'] if benchmark_stats['observed_mae'] > 0 else np.nan
        ratio_rmse = predicted_uncertainty_median / benchmark_stats['observed_rmse'] if benchmark_stats['observed_rmse'] > 0 else np.nan
        
        print(f"\n=== UNCERTAINTY COMPARISON ===")
        print(f"Model: {model_info['model']}, Filter: {model_info['swot_filter']}, Temporal: {model_info['temporal']}")
        print(f"Observed MAE: {benchmark_stats['observed_mae']:.6f} km³")
        print(f"Predicted uncertainty mean: {predicted_uncertainty_mean:.6f} km³")
        print(f"Predicted/Observed MAE ratio: {ratio_mae:.2f}")
        print(f"Predicted/Observed RMSE ratio: {ratio_rmse:.2f}")
        
        summary_stats.update({
            'observed_mae': benchmark_stats['observed_mae'],
            'observed_rmse': benchmark_stats['observed_rmse'],
            'observed_std': benchmark_stats['observed_std'],
            'observed_nse_median': benchmark_stats['observed_nse_median'],
            'observed_nse_mean': benchmark_stats['observed_nse_mean'],
            'predicted_mean': predicted_uncertainty_mean,
            'predicted_median': predicted_uncertainty_median,
            'ratio_mae': ratio_mae,
            'ratio_rmse': ratio_rmse#,
            #'benchmark_n_lakes': benchmark_stats['n_lakes']
        })
    
    return summary_stats


def main(use_height_weighting=False):
    """Main analysis function that reads all combinations from input_uncertainties.csv"""
    
    print("=== STORAGE UNCERTAINTY ATTRIBUTION ANALYSIS ===")
    if use_height_weighting:
        print("Using HEIGHT-WEIGHTED statistics (equal water level contribution)")
    else:
        print("Using UNWEIGHTED statistics (current approach)")
    print("Processing all uncertainty combinations from input_uncertainties.csv")
    
    # Define paths
    input_file = PROJECT_ROOT / "analysis/storage_uncertainty_attribution/results/input_uncertainties.csv"
    output_dir = PROJECT_ROOT / "analysis/storage_uncertainty_attribution/results"
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read input uncertainties
    if not input_file.exists():
        print(f"Error: Input file {input_file} not found")
        return
    
    input_df = pd.read_csv(input_file)
    print(f"Found {len(input_df)} uncertainty combinations to process")
    
    # Process each combination
    all_results = []
    
    for idx, row in input_df.iterrows():
        # Skip rows with missing WSE values 
        if pd.isna(row['wse']) or pd.isna(row['wse_temporal']) or pd.isna(row['wsa']):
            print(f"Skipping row {idx+1}: Missing uncertainty values")
            continue
            
        # Get wse_filt value (default to 0 if not present)
        wse_filt_val = row.get('wse_filt', 0.0) if 'wse_filt' in row else 0.0
        if pd.isna(wse_filt_val):
            wse_filt_val = 0.0
            
        print(f"\\nProcessing combination {idx+1}/{len(input_df)}: {row['model']} {row['swot filter']} {row['temporal']}")
        print(f"  WSE={row['wse']}, WSE_temporal={row['wse_temporal']}, WSE_filt={wse_filt_val}, WSA={row['wsa']}%")
        
        model_info = {
            'model': row['model'],
            'swot_filter': row['swot filter'],
            'temporal': row['temporal']
        }
        
        # Note: Using pre-calculated benchmark metrics instead of reading columns
        print(f"  Using benchmark metrics for: {model_info['model']}-{model_info['swot_filter']}-{model_info['temporal']}")
        
        # Run analysis for this combination
        result = run_uncertainty_combination(
            wse_std=row['wse'],
            wse_temporal_std=row['wse_temporal'],
            wse_filt_std=wse_filt_val,
            wsa_percent_error=row['wsa'],
            analysis_type=row['temporal'],
            model_info=model_info,
            use_height_weighting=use_height_weighting
        )
        
        if result is not None:
            all_results.append(result)
            print(f"  Completed: {result['n_observations']} observations from {result['n_lakes']} lakes")
        else:
            print(f"  Failed: No valid results")
    
    if len(all_results) == 0:
        print("No valid results found")
        return
    
    # Combine all results into output dataframe
    output_df = pd.DataFrame(all_results)
    
    # Save results
    suffix = "_weighted" if use_height_weighting else ""
    output_file = output_dir / f"storage_uncertainty_attribution{suffix}.csv"
    output_df.to_csv(output_file, index=False)
    print(f"\\nSaved results for all combinations to {output_file}")
    print(f"Processed {len(output_df)} combinations successfully")
    
    # Print summary
    print("\\n=== PROCESSING COMPLETE ===")
    print(f"Total combinations processed: {len(output_df)}")
    print(f"Average observations per combination: {output_df['n_observations'].mean():.0f}")
    print(f"Average lakes per combination: {output_df['n_lakes'].mean():.0f}")
    
    print("\\nResults saved to analysis/storage_uncertainty_attribution/results/storage_uncertainty_attribution.csv")


if __name__ == "__main__":
    # Configuration: Set to True to use height-weighted statistics for equal water level contribution
    # Set to False for standard unweighted statistics (current approach)
    USE_HEIGHT_WEIGHTING = False  # Change this to True to enable weighting
    
    main(use_height_weighting=USE_HEIGHT_WEIGHTING)
