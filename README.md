# Assessing SWOT Capabilities for Water Storage Monitoring in Lakes

Code and data supporting:

> Levenson, E.S., Cooley, S.W., Wang, J., & Trudel, M. Assessing SWOT Capabilities for Water Storage Monitoring in Lakes.(submitted).

## Data

`data/benchmark_timeseries/` contains daily time series CSV files for 411 gauged lakes and reservoirs, with paired SWOT LakeSP, Sentinel-2, and in situ gauge observations from July 2023 to October 2025. Each file is named by its SWOT Prior Lake Database (PLD) identifier. See `data/column_guide.csv` for column descriptions and units.

`data/gage_swot_mapping.csv` maps each lake's PLD identifier to in situ gauge station identifiers and lake attributes.

The benchmark time series were constructed by merging in situ gauge data (USGS NWIS, USBR RISE, CDEC), SWOT LakeSP observations retrieved via the [Hydrocron API](https://podaac.github.io/hydrocron/), and Sentinel-2 water surface area classifications. Adaptive filtering (Wang et al., 2025) and storage anomaly calculations have already been applied to the provided data.

## Repository Structure

```
swot-s2-lake-storage/
├── data/
│   ├── benchmark_timeseries/   # 411 lake time series (one CSV per lake)
│   ├── column_guide.csv        # Column descriptions and units
│   └── gage_swot_mapping.csv   # Lake-gauge mapping table
├── analysis/                   # Scripts that produce manuscript results
│   ├── swot_measurement_accuracy/
│   ├── storage_estimation_assessment/
│   └── storage_uncertainty_attribution/
└── src/                        # Data processing code (for reproducibility)
    ├── calculate_storage_anomalies.py
    ├── filtering/
    └── Sentinel2/
```

## src/ — Dataset Processing

These scripts were used to construct and process the time series for each benchmark lake. They are provided for reproducibility and documentation of methods, but their outputs are already included in the released data, and do not need to be executed to re-run analyses.

- **`calculate_storage_anomalies.py`** — Builds elevation-area relationships and calculates the 16 storage anomaly columns (4 models x 2 filters x 2 temporal resolutions) described in Section 2.5 and Table 1.
- **`filtering/adaptive_filter.py`** — Adaptive SWOT LakeSP filter implementation (Wang et al., 2025), which produces the `adaptive_filter` column.
- **`filtering/filters.py`** — Filter definitions for the quality flag, simple, and idealized filtering schemes described in Section 2.4.
- **`Sentinel2/GEE_S2_timeseries.py`** — Google Earth Engine script for extracting Sentinel-2 water surface area time series using adaptive NDWI thresholding (Cooley et al., 2017; Levenson et al., 2025).
- **`Sentinel2/GEE_S2_ice_timeseries.py`** — Google Earth Engine script for Sentinel-2 ice classification.

## analysis/ — Manuscript Results

**Naming note:** The code uses `opt` (optimal) to refer to the idealized filter described in the manuscript. Column names like `swot_opt_dis` correspond to the "Idealized" filter in the paper. The `filt` suffix corresponds to the "Adaptive" (functional) filter.

### SWOT Measurement Accuracy (Section 3.1)

| Script | Produces |
|--------|----------|
| `idealized_measurement_accuracy_frequency.py` | WSE/WSA error distributions, sampling gap analysis, accuracy-frequency tradeoff (Figure 3); error metrics (Table 2, Extended Table 1); NSE scores |
| `wse_filter_evaluation.py` | Filter comparison across quality flag, simple, and adaptive schemes (Figure 4, Table 2) |

### Storage Estimation Assessment (Section 3.2)

| Script | Produces |
|--------|----------|
| `benchmark_storage_analysis_km3.py` | Storage anomaly error metrics in km³ across all model/filter/temporal variants |
| `benchmark_storage_analysis_norm.py` | Storage anomaly error metrics normalized by variable storage capacity (Figure 6); per-lake statistics |
| `storage_variability_detection_analysis.py` | Storage variability detection as a function of time scale and WSE amplitude (Figure 5) |
| `statistical_tests.py` | Paired Wilcoxon signed-rank tests for model comparisons (Section 3.3 p-values) |

### Storage Uncertainty Attribution (Sections 2.6, 3.3)

| Script | Produces |
|--------|----------|
| `storage_uncertainty_attribution.py` | First-order error propagation decomposing storage uncertainty into WSE, WSA, filtering, and sampling components (km³) |
| `storage_uncertainty_attribution_norm.py` | Normalized uncertainty attribution (Figure 7, Table 3) |
| `plt_uncertainty_attribution.py` | Stacked bar visualization of uncertainty contributions (Figure 7) |
| `analyze_pure_interpolation_uncertainty.py` | Temporal sampling uncertainty parameterization (Section 3.1, ~3 cm/day interpolation error rate) |

## Requirements

Python >= 3.10 with: `pandas`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`.

Sentinel-2 scripts require a [Google Earth Engine](https://earthengine.google.com/) account and the `ee` Python API.
