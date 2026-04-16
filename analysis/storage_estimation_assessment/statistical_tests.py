#!/usr/bin/env python3
"""
Statistical significance tests for storage estimation model comparisons.

Uses paired Wilcoxon signed-rank tests on per-lake metrics (RMSE, MAE, NSE)
to determine whether performance differences among models are statistically significant.
"""

import pandas as pd
import numpy as np
from scipy import stats
from itertools import combinations

# Load per-lake stats
results_dir = "results"
df = pd.read_csv(f"{results_dir}/benchmark_storage_variants_per_lake_stats_normalized.csv")

# Collect all test results for CSV output
test_rows = []

# Focus on the main model comparisons: swot, swots2, s2, static
# Use opt filter, dis temporal as the primary comparison (most interpretable)
FILTER_TYPE = "opt"
TEMPORAL_TYPE = "dis"
METRICS = ["rmse", "mae", "nse", "pearson_r"]

print("=" * 80)
print("PAIRED WILCOXON SIGNED-RANK TESTS: MODEL COMPARISONS")
print(f"Filter: {FILTER_TYPE}, Temporal: {TEMPORAL_TYPE}")
print("Errors normalized as % of storage variability (1st-99th percentile range)")
print("=" * 80)

# Filter to the variants of interest
subset = df[(df["filter_type"] == FILTER_TYPE) & (df["temporal_type"] == TEMPORAL_TYPE)].copy()

models = subset["model"].unique()
print(f"\nModels: {list(models)}")

# Pivot so each row is a lake, columns are models, for each metric
for metric in METRICS:
    print(f"\n{'─' * 80}")
    print(f"METRIC: {metric.upper()}")
    print(f"{'─' * 80}")

    pivot = subset.pivot_table(index="lake_id", columns="model", values=metric)
    # Drop lakes with missing values for any model
    pivot_clean = pivot.dropna()
    n_lakes = len(pivot_clean)
    print(f"Lakes with data for all models: {n_lakes}")

    # Summary stats
    print(f"\n  {'Model':<10} {'Median':>10} {'Mean':>10} {'Std':>10}")
    for model in sorted(pivot_clean.columns):
        vals = pivot_clean[model]
        print(f"  {model:<10} {vals.median():>10.3f} {vals.mean():>10.3f} {vals.std():>10.3f}")

    # Pairwise Wilcoxon signed-rank tests
    print(f"\n  Pairwise Wilcoxon signed-rank tests (two-sided):")
    print(f"  {'Comparison':<25} {'statistic':>12} {'p-value':>12} {'significant':>12}")

    pairs = list(combinations(sorted(pivot_clean.columns), 2))
    for m1, m2 in pairs:
        d = pivot_clean[m1] - pivot_clean[m2]
        # Skip if all differences are zero
        if (d == 0).all():
            print(f"  {m1} vs {m2:<15} {'N/A':>12} {'N/A':>12} {'identical':>12}")
            continue
        stat, p = stats.wilcoxon(pivot_clean[m1], pivot_clean[m2], alternative="two-sided")
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  {m1} vs {m2:<15} {stat:>12.1f} {p:>12.2e} {sig:>12}")
        test_rows.append({
            "comparison_type": "model",
            "metric": metric,
            "group_a": m1,
            "group_b": m2,
            "median_a": pivot_clean[m1].median(),
            "median_b": pivot_clean[m2].median(),
            "n_lakes": n_lakes,
            "W_statistic": stat,
            "p_value": p,
            "significance": sig,
        })

# Also compare filter types (opt vs filt) and temporal types (dis vs con) for SWOT
print(f"\n{'=' * 80}")
print("ADDITIONAL COMPARISONS: FILTER AND TEMPORAL EFFECTS (SWOT model only)")
print("=" * 80)

for metric in ["rmse", "nse"]:
    print(f"\n{'─' * 80}")
    print(f"METRIC: {metric.upper()}")

    # Opt vs Filt (discrete)
    opt_dis = df[(df["model"] == "swot") & (df["filter_type"] == "opt") & (df["temporal_type"] == "dis")]
    filt_dis = df[(df["model"] == "swot") & (df["filter_type"] == "filt") & (df["temporal_type"] == "dis")]
    merged = opt_dis[["lake_id", metric]].merge(filt_dis[["lake_id", metric]], on="lake_id", suffixes=("_opt", "_filt")).dropna()
    if len(merged) > 0:
        stat, p = stats.wilcoxon(merged[f"{metric}_opt"], merged[f"{metric}_filt"], alternative="two-sided")
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  opt vs filt (dis):  n={len(merged)}, W={stat:.1f}, p={p:.2e} {sig}")
        print(f"    opt median={merged[f'{metric}_opt'].median():.3f}, filt median={merged[f'{metric}_filt'].median():.3f}")
        test_rows.append({
            "comparison_type": "filter",
            "metric": metric,
            "group_a": "opt",
            "group_b": "filt",
            "median_a": merged[f"{metric}_opt"].median(),
            "median_b": merged[f"{metric}_filt"].median(),
            "n_lakes": len(merged),
            "W_statistic": stat,
            "p_value": p,
            "significance": sig,
        })

    # Dis vs Con (optimal)
    opt_dis2 = df[(df["model"] == "swot") & (df["filter_type"] == "opt") & (df["temporal_type"] == "dis")]
    opt_con = df[(df["model"] == "swot") & (df["filter_type"] == "opt") & (df["temporal_type"] == "con")]
    merged2 = opt_dis2[["lake_id", metric]].merge(opt_con[["lake_id", metric]], on="lake_id", suffixes=("_dis", "_con")).dropna()
    if len(merged2) > 0:
        stat, p = stats.wilcoxon(merged2[f"{metric}_dis"], merged2[f"{metric}_con"], alternative="two-sided")
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  dis vs con (opt):   n={len(merged2)}, W={stat:.1f}, p={p:.2e} {sig}")
        print(f"    dis median={merged2[f'{metric}_dis'].median():.3f}, con median={merged2[f'{metric}_con'].median():.3f}")
        test_rows.append({
            "comparison_type": "temporal",
            "metric": metric,
            "group_a": "dis",
            "group_b": "con",
            "median_a": merged2[f"{metric}_dis"].median(),
            "median_b": merged2[f"{metric}_con"].median(),
            "n_lakes": len(merged2),
            "W_statistic": stat,
            "p_value": p,
            "significance": sig,
        })

# Bonferroni correction note
n_comparisons = len(list(combinations(models, 2)))
print(f"\n{'=' * 80}")
print(f"NOTE: {n_comparisons} pairwise model comparisons per metric.")
print(f"Bonferroni-corrected significance threshold: α = {0.05/n_comparisons:.4f}")
print(f"All p-values reported are uncorrected; apply Bonferroni when interpreting.")
print("=" * 80)

# Save results to CSV
out_path = f"{results_dir}/wilcoxon_signed_rank_tests.csv"
results_df = pd.DataFrame(test_rows)
results_df.to_csv(out_path, index=False)
print(f"\nResults saved to {out_path}")
