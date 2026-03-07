"""
Compute Cohen's d effect sizes for CSD indicators (Tables 2 and 4).
Uses the same pipeline as run_cga_analysis.py and run_cga_cmv_analysis.py.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from config import (
    CGA_FEATURES_PATH, CGA_CONVERSATION_SUMMARY_PATH,
    CGA_CMV_FEATURES_PATH, CGA_CMV_CONVERSATION_SUMMARY_PATH,
    ROLLING_WINDOW, RANDOM_SEED
)
from utils.csd_indicators import rolling_variance, rolling_lag1_autocorrelation, kendall_tau_trend


def cohens_d(g1, g2):
    n1, n2 = len(g1), len(g2)
    var1, var2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(g1) - np.mean(g2)) / pooled_std


def compute_effect_sizes(features_path, summary_path, label_col, min_turns=10):
    utt_df = pd.read_parquet(features_path)
    conv_df = pd.read_parquet(summary_path)

    ge_columns = [c for c in utt_df.columns if c.startswith("ge_")]

    series_columns = ["q_da", "q_rm", "q_ce", "toxicity", "vader_compound"]
    all_series_columns = [c for c in series_columns if c in utt_df.columns]

    derail_taus = {col: {"ac1": [], "var": []} for col in all_series_columns}
    civil_taus = {col: {"ac1": [], "var": []} for col in all_series_columns}
    amd_derail_var_taus = []
    amd_civil_var_taus = []

    for conv_id, group in utt_df.groupby("convo_id"):
        if len(group) < min_turns:
            continue

        conv_info = conv_df[conv_df["convo_id"] == conv_id]
        if conv_info.empty:
            continue
        is_derailing = bool(conv_info.iloc[0][label_col])

        n = len(group)
        pre_breakdown_start = max(0, n - 5)
        pre_breakdown_end = n

        for col in all_series_columns:
            if col not in group.columns:
                continue
            series = group[col].values
            ac1 = rolling_lag1_autocorrelation(series, ROLLING_WINDOW)
            var = rolling_variance(series, ROLLING_WINDOW)

            offset = ROLLING_WINDOW - 1
            adj_start = max(0, pre_breakdown_start - offset)
            adj_end = max(0, pre_breakdown_end - offset)

            ac1_tau, _ = kendall_tau_trend(ac1[adj_start:adj_end])
            var_tau, _ = kendall_tau_trend(var[adj_start:adj_end])

            target = derail_taus if is_derailing else civil_taus
            if not np.isnan(ac1_tau):
                target[col]["ac1"].append(ac1_tau)
            if not np.isnan(var_tau):
                target[col]["var"].append(var_tau)

        if ge_columns:
            ge_matrix = group[ge_columns].values
            turn_variances = np.var(ge_matrix, axis=1)
            var_series = rolling_variance(turn_variances, ROLLING_WINDOW)
            offset = ROLLING_WINDOW - 1
            adj_start = max(0, pre_breakdown_start - offset)
            adj_end = max(0, pre_breakdown_end - offset)
            tau, _ = kendall_tau_trend(var_series[adj_start:adj_end])
            if not np.isnan(tau):
                if is_derailing:
                    amd_derail_var_taus.append(tau)
                else:
                    amd_civil_var_taus.append(tau)

    print(f"\n{'Indicator':<25} {'d (derail-civil)':>18} {'n_derail':>10} {'n_civil':>10}")
    print("-" * 65)

    for col in all_series_columns:
        for stat_type in ["ac1", "var"]:
            d_vals = derail_taus[col][stat_type]
            c_vals = civil_taus[col][stat_type]
            if d_vals and c_vals:
                d = cohens_d(d_vals, c_vals)
                print(f"{col} {stat_type.upper():<8} {d:>15.3f} {len(d_vals):>10} {len(c_vals):>10}")

    if amd_derail_var_taus and amd_civil_var_taus:
        d = cohens_d(amd_derail_var_taus, amd_civil_var_taus)
        print(f"{'AMD VAR':<25} {d:>15.3f} {len(amd_derail_var_taus):>10} {len(amd_civil_var_taus):>10}")


print("=" * 65)
print("CGA-Wiki Effect Sizes")
print("=" * 65)
compute_effect_sizes(CGA_FEATURES_PATH, CGA_CONVERSATION_SUMMARY_PATH, "derails")

print("\n" + "=" * 65)
print("CGA-CMV Effect Sizes")
print("=" * 65)
compute_effect_sizes(CGA_CMV_FEATURES_PATH, CGA_CMV_CONVERSATION_SUMMARY_PATH, "derails")
