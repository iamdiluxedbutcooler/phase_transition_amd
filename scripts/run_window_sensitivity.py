"""Window-size sensitivity for CSD indicators (Exp. 4) on CGA-Wiki."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    RANDOM_SEED,
    PERMUTATION_N,
    CGA_FEATURES_PATH,
    CGA_CONVERSATION_SUMMARY_PATH,
)
from utils.csd_indicators import rolling_variance, kendall_tau_trend
from utils.statistical_tests import vectorized_permutation_test


def cohens_d(g1, g2):
    g1, g2 = np.asarray(g1), np.asarray(g2)
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return np.nan
    var1, var2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(g1) - np.mean(g2)) / pooled_std


def load_data():
    utt_df = pd.read_parquet(CGA_FEATURES_PATH)

    conciliatory_cols = []
    for tag in ["aa", "bk", "br", "ba"]:
        col = f"da_prob_{tag}"
        if col in utt_df.columns:
            conciliatory_cols.append(col)
    if not conciliatory_cols:
        for lbl in ["LABEL_2", "LABEL_10", "LABEL_25", "LABEL_42"]:
            col = f"da_prob_{lbl}"
            if col in utt_df.columns:
                conciliatory_cols.append(col)
    if conciliatory_cols:
        utt_df["q_da"] = utt_df[conciliatory_cols].sum(axis=1)

    return utt_df


def run_exp4_for_window(utt_df, rolling_window):
    ge_columns = [c for c in utt_df.columns if c.startswith("ge_")]
    pre_breakdown_length = 5
    min_turns = rolling_window + pre_breakdown_length

    amd_derail, amd_civil = [], []
    qda_derail, qda_civil = [], []

    for convo_id, group in utt_df.groupby("convo_id"):
        group = group.sort_values("turn_idx")
        is_derailing = group["derails"].iloc[0]
        n_turns = len(group)

        if n_turns < min_turns:
            continue

        attack_turn = group["attack_turn"].iloc[0]
        if is_derailing and pd.notna(attack_turn):
            pre_breakdown_end = int(attack_turn)
            pre_breakdown_start = max(0, pre_breakdown_end - pre_breakdown_length)
        else:
            pre_breakdown_end = n_turns
            pre_breakdown_start = max(0, n_turns - pre_breakdown_length)

        offset = rolling_window - 1
        adjusted_start = max(0, pre_breakdown_start - offset)
        adjusted_end = max(0, pre_breakdown_end - offset)

        if "q_da" in group.columns:
            series = group["q_da"].values.astype(float)
            var = rolling_variance(series, rolling_window)
            var_window = var[adjusted_start:adjusted_end]
            tau, _ = kendall_tau_trend(var_window)
            if not np.isnan(tau):
                (qda_derail if is_derailing else qda_civil).append(tau)

        if len(ge_columns) > 0:
            ge_matrix = group[ge_columns].values
            turn_variances = np.var(ge_matrix, axis=1)
            var_series = rolling_variance(turn_variances, rolling_window)
            var_window = var_series[adjusted_start:adjusted_end]
            tau, _ = kendall_tau_trend(var_window)
            if not np.isnan(tau):
                (amd_derail if is_derailing else amd_civil).append(tau)

    if amd_derail and amd_civil:
        amd_perm = vectorized_permutation_test(
            amd_derail, amd_civil, n_permutations=PERMUTATION_N, seed=RANDOM_SEED
        )
        amd_p = amd_perm["p_value"]
        amd_d = cohens_d(amd_derail, amd_civil)
    else:
        amd_p, amd_d = np.nan, np.nan

    if qda_derail and qda_civil:
        qda_perm = vectorized_permutation_test(
            qda_derail, qda_civil, n_permutations=PERMUTATION_N, seed=RANDOM_SEED
        )
        qda_p = qda_perm["p_value"]
        qda_d = cohens_d(qda_derail, qda_civil)
    else:
        qda_p, qda_d = np.nan, np.nan

    return {
        "AMD_Var_p": amd_p,
        "AMD_Var_d": amd_d,
        "qDA_Var_p": qda_p,
        "qDA_Var_d": qda_d,
        "n_amd": len(amd_derail) + len(amd_civil),
        "n_qda": len(qda_derail) + len(qda_civil),
    }


def main():
    print("Loading CGA-Wiki features...")
    utt_df = load_data()
    print(f"  {len(utt_df)} utterances, {utt_df['convo_id'].nunique()} conversations")

    window_sizes = [3, 4, 5, 6, 7]
    rows = []

    for W in window_sizes:
        print(f"\nRunning Exp. 4 with W={W}...")
        result = run_exp4_for_window(utt_df, W)
        rows.append({"W": W, **result})

    print("\n" + "=" * 72)
    print("Window-Size Sensitivity: CSD Indicators on CGA-Wiki (Exp. 4)")
    print("=" * 72)
    print(f"{'W':>3}  {'AMD_Var_p':>10}  {'AMD_Var_d':>10}  {'qDA_Var_p':>10}  {'qDA_Var_d':>10}  {'N':>5}")
    print("-" * 62)
    for r in rows:
        print(
            f"{r['W']:>3}  "
            f"{r['AMD_Var_p']:>10.4f}  "
            f"{r['AMD_Var_d']:>10.3f}  "
            f"{r['qDA_Var_p']:>10.4f}  "
            f"{r['qDA_Var_d']:>10.3f}  "
            f"{r['n_amd']:>5}"
        )
    print("-" * 62)

    strongest_all = all(r["AMD_Var_p"] < r["qDA_Var_p"] for r in rows)
    print(f"\nAMD Variance is strongest across all W: {strongest_all}")


if __name__ == "__main__":
    main()
