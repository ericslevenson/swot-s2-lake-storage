#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Centralized filter functions for SWOT analysis

Each filter function takes a DataFrame and returns a boolean mask.
All filters are applied to individual lake CSV files from benchmark_daily/.

Usage:
    from src.filter.filters import swot_custom_standard, s2_high_coverage
    
    # Apply filters
    df_filtered = df[swot_custom_standard(df) & date_range_swot_era(df)]
    df_s2 = df[s2_high_coverage(df)]
"""

import pandas as pd
import numpy as np


# =============================================================================
# DATE FILTERS
# =============================================================================

def date_range_swot_era(df):
    """
    Filter for SWOT science orbit era (after July 21, 2023)
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with 'date' column
        
    Returns:
    --------
    pandas.Series : Boolean mask
    """
    return df['date'] > '2023-07-21'


# =============================================================================
# SWOT FILTERS
# =============================================================================

def swot_quality_flag(df):
    """
    Official SWOT quality flag filter
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with 'swot_quality_f' column
        
    Returns:
    --------
    pandas.Series : Boolean mask
    """
    return df['swot_quality_f'] == 0


def swot_custom_standard(df):
    """
    Standard custom filter for most SWOT analyses
    
    This is the most commonly used filter across experiments.
    Uses base columns (not smoothed) to avoid dependency on add_median_filters.py
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with SWOT columns
        
    Returns:
    --------
    pandas.Series : Boolean mask
    """
    return ((df['swot_xovr_cal_q'] < 2) & 
            (df['swot_wse_u'] < 0.1) &
            (df['swot_wse_std'] < 0.4) &
            (df['swot_wse_anomaly'] > -50) &
            (df['swot_wse_anomaly'] < 50) &
            (df['swot_wse']>0))
            #(df['swot_ice_clim_f'] < 2))


def swot_strict(df):
    """
    Stricter SWOT filter for high-quality analysis
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with SWOT columns
        
    Returns:
    --------
    pandas.Series : Boolean mask
    """
    return ((df['swot_xovr_cal_q'] < 2) & 
            (df['swot_wse_u'] < 0.05) &
            (df['swot_wse_std'] < 0.2) &
            (df['swot_wse_anomaly'] > -25) &
            (df['swot_wse_anomaly'] < 25) &
            (df['swot_partial_f'] == 0))


def swot_relaxed(df):
    """
    More permissive SWOT filter for temporal resolution analysis
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with SWOT columns
        
    Returns:
    --------
    pandas.Series : Boolean mask
    """
    return ((df['swot_xovr_cal_q'] < 2) & 
            (df['swot_wse_u'] < 0.2) &
            (df['swot_wse_std'] < 0.5) &
            (df['swot_wse_anomaly'] > -100) &
            (df['swot_wse_anomaly'] < 100) &
            (df['swot_partial_f'] == 0))


def swot_no_partial(df):
    """
    Filter to exclude partial lake observations
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with 'swot_partial_f' column
        
    Returns:
    --------
    pandas.Series : Boolean mask
    """
    return df['swot_partial_f'] == 0


# =============================================================================
# SENTINEL-2 FILTERS
# =============================================================================

def s2_high_coverage(df):
    """
    High coverage Sentinel-2 filter (>99% coverage, no ice)
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with 's2_coverage' and 'ice' columns
        
    Returns:
    --------
    pandas.Series : Boolean mask
    """
    return ((df['s2_coverage'] > 99) & (df['ice'] == 0))


def s2_medium_coverage(df):
    """
    Medium coverage Sentinel-2 filter (>95% coverage, no ice)
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with 's2_coverage' and 'ice' columns
        
    Returns:
    --------
    pandas.Series : Boolean mask
    """
    return ((df['s2_coverage'] > 95) & (df['ice'] == 0))


def s2_no_ice(df):
    """
    Filter to exclude ice-covered observations
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with 'ice' column
        
    Returns:
    --------
    pandas.Series : Boolean mask
    """
    return df['ice'] == 0


# =============================================================================
# COMBINED FILTERS
# =============================================================================

def swot_wse_analysis(df):
    """
    Combined filter for SWOT WSE analysis (standard + date + no partial)
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with SWOT columns
        
    Returns:
    --------
    pandas.Series : Boolean mask
    """
    return (date_range_swot_era(df) & 
            swot_custom_standard(df) & 
            swot_no_partial(df))


def swot_wsa_analysis(df):
    """
    Combined filter for SWOT WSA analysis (standard + date + no partial)
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with SWOT columns
        
    Returns:
    --------
    pandas.Series : Boolean mask
    """
    return (date_range_swot_era(df) & 
            swot_custom_standard(df) & 
            swot_no_partial(df))


def elevation_area_relationship(df):
    """
    Filter for building elevation-area relationships
    Same as swot_wse_analysis but explicitly named for clarity
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with SWOT columns
        
    Returns:
    --------
    pandas.Series : Boolean mask
    """
    return swot_wse_analysis(df)


# =============================================================================
# ADAPTIVE FILTERS
# =============================================================================

def swot_adaptive_lakeSP(df, **kwargs):
    """
    Adaptive LakeSP filter (v9) by Jida Wang & Melanie Trudel
    
    This is a sophisticated filter that calibrates lake-specific thresholds
    and applies iterative low-pass filtering for optimal noise removal.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with SWOT columns for a single lake
    **kwargs : 
        Additional arguments (verbose=True for detailed output)
        
    Returns:
    --------
    pandas.Series : Boolean mask
    
    Note:
    -----
    This filter requires more computational time but provides
    superior noise removal compared to fixed-threshold filters.
    """
    try:
        from .adaptive_filter import swot_adaptive_lakeSP as adaptive_filter
        return adaptive_filter(df, **kwargs)
    except ImportError:
        # Fallback to standard filter if adaptive filter not available
        return swot_custom_standard(df)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def combine_filters(*filters):
    """
    Combine multiple filter functions with AND logic
    
    Parameters:
    -----------
    *filters : callable
        Filter functions that each return a boolean mask
        
    Returns:
    --------
    callable : Function that applies all filters
    
    Example:
    --------
    combined = combine_filters(swot_custom_standard, date_range_swot_era, s2_high_coverage)
    df_filtered = df[combined(df)]
    """
    def combined_filter(df):
        result = pd.Series(True, index=df.index)
        for filter_func in filters:
            result = result & filter_func(df)
        return result
    return combined_filter


def get_filter_info():
    """
    Return information about available filters
    
    Returns:
    --------
    dict : Dictionary with filter names and descriptions
    """
    return {
        'date_range_swot_era': 'SWOT science orbit era (after July 21, 2023)',
        'swot_quality_flag': 'Official SWOT quality flag filter',
        'swot_custom_standard': 'Standard custom filter for most analyses',
        'swot_strict': 'Stricter SWOT filter for high-quality analysis',
        'swot_relaxed': 'More permissive SWOT filter',
        'swot_no_partial': 'Exclude partial lake observations',
        's2_high_coverage': 'High coverage S2 (>99%, no ice)',
        's2_medium_coverage': 'Medium coverage S2 (>95%, no ice)',
        's2_no_ice': 'Exclude ice-covered observations',
        'swot_wse_analysis': 'Combined filter for WSE analysis',
        'swot_wsa_analysis': 'Combined filter for WSA analysis',
        'elevation_area_relationship': 'Filter for elevation-area relationships'
    }


if __name__ == "__main__":
    # Print available filters
    print("Available SWOT Analysis Filters:")
    print("=" * 50)
    for name, description in get_filter_info().items():
        print(f"{name:25} : {description}")