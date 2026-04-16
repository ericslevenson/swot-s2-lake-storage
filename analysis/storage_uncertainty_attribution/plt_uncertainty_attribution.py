#!/usr/bin/env python3
"""
Create normalized stacked bar plots for FILTERED models only, decomposing uncertainty into:
- WSE Measurements (base uncertainty)  
- Area Measurements
- Filtering Errors (difference between filtered and optimal RMSE)
- Temporal Sampling Gaps (for daily models only)

All values are normalized as percentage of storage variability (1st to 99th percentile range).
Bar order: WSE, Area, Filtering, Temporal
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Set style
plt.style.use('default')
sns.set_palette("Set2")

def load_data():
    """Load the normalized uncertainty attribution data with benchmark values"""
    
    # Load normalized uncertainty attribution data
    uncertainty_file = PROJECT_ROOT / "analysis/storage_uncertainty_attribution/results/storage_uncertainty_attribution_normalized.csv"
    
    if not uncertainty_file.exists():
        print(f"Error: Normalized uncertainty file not found at {uncertainty_file}")
        print("Please run storage_uncertainty_attribution.py in storage_normalized first")
        return None
        
    df = pd.read_csv(uncertainty_file)
    
    print(f"Loaded {len(df)} model combinations")
    print("Data columns:", df.columns.tolist())
    
    return df

def calculate_filtering_decomposition(df):
    """Calculate normalized uncertainty components for filtered models with filtering errors decomposition"""
    
    results = []
    
    # Process filtered models only
    filtered_models = df[df['swot_filter'] == 'filtered'].copy()
    
    for _, filtered_row in filtered_models.iterrows():
        # Find corresponding optimal model
        optimal_mask = (df['model'] == filtered_row['model']) & \
                      (df['temporal'] == filtered_row['temporal']) & \
                      (df['swot_filter'] == 'optimal')
        
        if optimal_mask.sum() == 0:
            continue
            
        optimal_row = df[optimal_mask].iloc[0]
        
        # Get normalized RMSE values (already in %)
        filtered_rmse = filtered_row['observed_rmse']
        optimal_rmse = optimal_row['observed_rmse']
        
        # Calculate filtering error as the difference (in normalized %)

        
        # Calculate components based on filtered model contribution percentages
        # Each component is a percentage of the total observed RMSE for the filtered model
        if filtered_row['temporal'] == 'discrete':
            # For discrete: WSE base, Area, Filtering components
            wse_component_rmse = filtered_rmse * (filtered_row['wse_base_contribution_median'] / 100.0)
            area_component_rmse = filtered_rmse * (filtered_row['area_contribution_median'] / 100.0)
            filtering_component_rmse = filtered_rmse * (filtered_row.get('wse_filt_contribution_median', 0) / 100.0)
            temporal_component_rmse = 0.0
        else:
            # For continuous/daily: WSE base, Area, Filtering, and Temporal components
            wse_component_rmse = filtered_rmse * (filtered_row['wse_base_contribution_median'] / 100.0)
            area_component_rmse = filtered_rmse * (filtered_row['area_contribution_median'] / 100.0)
            filtering_component_rmse = filtered_rmse * (filtered_row.get('wse_filt_contribution_median', 0) / 100.0)
            temporal_component_rmse = filtered_rmse * (filtered_row.get('wse_temporal_contribution_median', 0) / 100.0)
        
        # Create result record (all values in normalized %)
        result = {
            'model': filtered_row['model'],
            'temporal': filtered_row['temporal'],
            'total_rmse': round(filtered_rmse, 1),
            'wse_component': round(wse_component_rmse, 1),
            'area_component': round(area_component_rmse, 1),
            'filtering_component': round(filtering_component_rmse, 1),
            'temporal_component': round(temporal_component_rmse, 1)
        }
        
        results.append(result)
    
    return pd.DataFrame(results)

def create_filtered_stacked_bars():
    """Create normalized stacked bar plots for filtered models with filtering error decomposition"""
    
    # Load and process data
    df = load_data()
    if df is None:
        return
        
    decomp_df = calculate_filtering_decomposition(df)
    
    if decomp_df.empty:
        print("No data available for creating stacked bar plots")
        return
    
    # Create figure with 2 panels (discrete left, continuous right)
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 4), sharey=True)
    
    # Define colors matching original scheme
    color_wse = '#d95f02'      # Orange for WSE (base measurements)
    color_area = '#7570b3'     # Purple for Area
    color_filtering = '#1b9e77' # Green for Filtering Errors
    color_temporal = '#e7298a'  # Pink for Temporal Sampling Gaps
    
    # Process discrete and continuous/daily separately
    for idx, (temporal_type, temporal_label) in enumerate([('discrete', 'Discrete'), ('daily', 'Continuous')]):
        ax = axes[idx]
        
        # Filter data for this temporal type (handle both 'continuous' and 'daily')
        if temporal_type == 'daily':
            # Check for both 'daily' and 'continuous' in case of different naming
            temporal_data = decomp_df[(decomp_df['temporal'] == 'daily') | (decomp_df['temporal'] == 'continuous')]
        else:
            temporal_data = decomp_df[decomp_df['temporal'] == temporal_type]
        
        if temporal_data.empty:
            ax.text(0.5, 0.5, f'No {temporal_label} data', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(temporal_label)
            continue
        
        # Model order matching original
        model_order = ['static', 'swot', 'swots2']
        model_labels = ['STATIC', 'SWOT', 'SWOTS2']
        
        # Filter to only models present in data
        available_models = []
        available_labels = []
        for i, model in enumerate(model_order):
            if model in temporal_data['model'].values:
                available_models.append(model)
                available_labels.append(model_labels[i])
        
        if not available_models:
            ax.text(0.5, 0.5, f'No models available for {temporal_label}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(temporal_label)
            continue
        
        # Prepare data for stacking
        n_models = len(available_models)
        x_pos = np.arange(n_models)
        
        # Initialize arrays for each component in model order
        wse_values = []
        area_values = []
        filtering_values = []
        temporal_values = []
        
        for model in available_models:
            model_data = temporal_data[temporal_data['model'] == model]
            if len(model_data) > 0:
                model_row = model_data.iloc[0]
                wse_values.append(model_row['wse_component'])
                area_values.append(model_row['area_component'])
                filtering_values.append(model_row['filtering_component'])
                temporal_values.append(model_row['temporal_component'])
            else:
                wse_values.append(0)
                area_values.append(0)
                filtering_values.append(0)
                temporal_values.append(0)
        
        # Create stacked bars matching original order (bottom to top: WSE, Filtering, Area, Temporal)
        bar_width = 0.6
        wse_values = np.array(wse_values)
        filtering_values = np.array(filtering_values)
        area_values = np.array(area_values)
        temporal_values = np.array(temporal_values)
        
        # Bottom layer: WSE (orange)
        bars1 = ax.bar(x_pos, wse_values, bar_width, 
                      label='WSE Measurement' if idx == 0 else None,
                      color=color_wse, edgecolor='black', linewidth=0.5)
        
        # Second layer: Filtering (green)
        bars2 = ax.bar(x_pos, filtering_values, bar_width,
                      bottom=wse_values,
                      label='Filtering Error' if idx == 0 else None,
                      color=color_filtering, edgecolor='black', linewidth=0.5)
        
        # Third layer: Area (purple) 
        bottom2 = wse_values + filtering_values
        bars3 = ax.bar(x_pos, area_values, bar_width,
                      bottom=bottom2,
                      label='Area Measurement' if idx == 0 else None,
                      color=color_area, edgecolor='black', linewidth=0.5)
        
        # Fourth layer: Temporal (pink - only for continuous/daily)
        if temporal_type == 'daily':
            bottom3 = bottom2 + area_values
            bars4 = ax.bar(x_pos, temporal_values, bar_width,
                          bottom=bottom3,
                          label='Temporal Sampling' if idx == 0 else None,
                          color=color_temporal, edgecolor='black', linewidth=0.5)
        
        # Customize axes
        ax.set_xlabel('Model', fontsize=11, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(available_labels, fontsize=10)
        ax.set_axisbelow(True)
        
        if idx == 0:
            ax.set_ylabel('Normalized RMSE (%)', fontsize=11, fontweight='bold')
        
        ax.set_title(temporal_label, fontsize=12, fontweight='bold')
        
        # Set y-axis limit
        ax.set_ylim(0, 14)  # Adjusted for normalized percentages
        
        # Add grid
        ax.grid(True, alpha=0.3, axis='y')
        '''
        # Add value labels on bars (optional - comment out if too cluttered)
        for i, model in enumerate(available_models):
            model_data = temporal_data[temporal_data['model'] == model]
            if len(model_data) > 0:
                model_row = model_data.iloc[0]
                total = model_row['total_rmse']
                ax.text(i, total + 0.5, f'{total:.1f}%', 
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Add legend (only once)
    if temporal_type == 'discrete':
        axes[0].legend(loc='upper left', fontsize=9, framealpha=0.9)
    else:
        # For continuous, include all 4 components
        handles, labels = axes[1].get_legend_handles_labels()
        axes[0].legend(loc='upper left', fontsize=9, framealpha=0.9)
    
    # Overall title
    fig.suptitle('Normalized Storage Uncertainty Decomposition - Filtered Models\n(% of Storage Variability)', 
                fontsize=13, fontweight='bold', y=1.02)
    '''
    plt.tight_layout()
    
    # Save figure
    output_dir = PROJECT_ROOT / "analysis/storage_estimation_assessment/results_normalized"
    output_file = output_dir / "filtered_uncertainty_stacked_bars_normalized.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved normalized stacked bar plot to {output_file}")
    
    plt.show()

def create_comparison_plot():
    """Create a comparison plot showing normalized vs non-normalized uncertainties"""
    
    # This would require loading both normalized and non-normalized data
    # For now, we'll focus on the normalized version
    pass

def main():
    """Main execution function"""
    
    print("Creating normalized filtered uncertainty stacked bar plots...")
    create_filtered_stacked_bars()
    print("Done!")

if __name__ == "__main__":
    main()