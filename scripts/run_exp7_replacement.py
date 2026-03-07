"""
Replacement experiments for Exp 7 (CGA-CMV ablation, which was null).

Exp 7a: Early Warning Lead-Time Analysis (CGA-Wiki)
  - For derailing conversations, measure CSD indicator significance at
    progressively earlier windows before the attack turn.
  - Shows *how early* CSD can detect approaching breakdown.

Exp 7b: Length–Dose-Response Analysis (CGA-CMV)
  - Stratify conversations by length and show CSD effect sizes increase
    with conversation length — validating the theoretical prediction that
    CSD detection requires sufficient observation windows.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    RESULTS_DIR,
    RANDOM_SEED,
    ROLLING_WINDOW,
    PERMUTATION_N,
    CGA_FEATURES_PATH,
    CGA_CONVERSATION_SUMMARY_PATH,
    CGA_CMV_FEATURES_PATH,
    CGA_CMV_CONVERSATION_SUMMARY_PATH,
)
from utils.csd_indicators import (
    rolling_lag1_autocorrelation,
    rolling_variance,
    kendall_tau_trend,
)
from utils.statistical_tests import vectorized_permutation_test


def load_cga_features():
    """Load CGA-Wiki features and recompute q_da."""
    utt_df = pd.read_parquet(CGA_FEATURES_PATH)
    conciliatory_cols = [f"da_prob_{t}" for t in ["aa", "bk", "br", "ba"]
                         if f"da_prob_{t}" in utt_df.columns]
    if conciliatory_cols:
        utt_df["q_da"] = utt_df[conciliatory_cols].sum(axis=1)
    return utt_df


def load_cga_cmv_features():
    """Load CGA-CMV features and recompute q_da."""
    utt_df = pd.read_parquet(CGA_CMV_FEATURES_PATH)
    conciliatory_cols = [f"da_prob_{t}" for t in ["aa", "bk", "br", "ba"]
                         if f"da_prob_{t}" in utt_df.columns]
    if conciliatory_cols:
        utt_df["q_da"] = utt_df[conciliatory_cols].sum(axis=1)
    return utt_df


# ======================================================================
# Experiment 7a: Early Warning Lead-Time Analysis (CGA-Wiki)
# ======================================================================

def experiment_7a_lead_time(utt_df):
    """
    Early warning lead-time analysis, matching Exp 4's method exactly.
    
    For each lead time k (turns before attack), compute CSD indicators
    in a 5-turn pre-breakdown *window* ending at (attack_turn - k),
    exactly as Exp 4 does it. Compare derailing vs civil via permutation test.
    
    k=0 should reproduce Exp 4 results. k>0 tests how far in advance
    the CSD signal is detectable.
    """
    print("=" * 72)
    print("EXPERIMENT 7a: Early Warning Lead-Time Analysis (CGA-Wiki)")
    print("=" * 72)

    ge_columns = [c for c in utt_df.columns if c.startswith("ge_")]
    PRE_WINDOW = 5  # same as Exp 4
    
    # Lead times: 0 = window ends at attack turn (reproduces Exp 4)
    #             1 = window ends 1 turn before attack, etc.
    lead_times = [0, 1, 2, 3, 4, 5]
    
    indicators = ["q_da_VAR", "AMD_VAR", "vader_VAR", "toxicity_VAR"]
    
    results = {}
    
    print(f"\nMethod: 5-turn sliding window on rolling CSD series, shifted by k from attack/end")
    print(f"k=0 reproduces Exp 4 windowing\n")
    
    print(f"{'Lead':<6} {'Indicator':<16} {'N_d':>5} {'N_c':>5} "
          f"{'τ_derail':>10} {'τ_civil':>10} {'p-value':>10} {'Sig?':>6}")
    print("-" * 76)
    
    for k in lead_times:
        results[f"k{k}"] = {}
        
        derail_taus = {ind: [] for ind in indicators}
        civil_taus = {ind: [] for ind in indicators}
        
        for convo_id, group in utt_df.groupby("convo_id"):
            group = group.sort_values("turn_idx")
            is_derailing = bool(group["derails"].iloc[0])
            n_turns = len(group)
            attack_turn = group["attack_turn"].iloc[0]
            
            if n_turns < ROLLING_WINDOW + PRE_WINDOW:
                continue
            
            # Define pre-breakdown window end and start (in turn space)
            if is_derailing and pd.notna(attack_turn):
                pre_end = int(attack_turn) - k
                pre_start = max(0, pre_end - PRE_WINDOW)
            else:
                pre_end = n_turns - k
                pre_start = max(0, pre_end - PRE_WINDOW)
            
            if pre_end <= ROLLING_WINDOW or pre_start < 0:
                continue
            
            # Offset for rolling series (rolling series[i] corresponds to turn window [i, i+w-1])
            offset = ROLLING_WINDOW - 1
            adj_start = max(0, pre_start - offset)
            adj_end = max(0, pre_end - offset)
            
            if adj_end <= adj_start or adj_end - adj_start < 3:
                continue
            
            target = derail_taus if is_derailing else civil_taus
            
            # q_da VAR
            q_da_series = group["q_da"].values.astype(float)
            var_series = rolling_variance(q_da_series, ROLLING_WINDOW)
            window = var_series[adj_start:adj_end]
            tau, _ = kendall_tau_trend(window)
            if not np.isnan(tau):
                target["q_da_VAR"].append(tau)
            
            # AMD VAR
            if len(ge_columns) > 0:
                ge_matrix = group[ge_columns].values
                turn_vars = np.var(ge_matrix, axis=1)
                amd_var_series = rolling_variance(turn_vars, ROLLING_WINDOW)
                window = amd_var_series[adj_start:adj_end]
                tau, _ = kendall_tau_trend(window)
                if not np.isnan(tau):
                    target["AMD_VAR"].append(tau)
            
            # VADER VAR
            vader_series = group["vader_compound"].values.astype(float)
            var_series = rolling_variance(vader_series, ROLLING_WINDOW)
            window = var_series[adj_start:adj_end]
            tau, _ = kendall_tau_trend(window)
            if not np.isnan(tau):
                target["vader_VAR"].append(tau)
            
            # Toxicity VAR (baseline)
            tox_series = group["toxicity_score"].values.astype(float)
            var_series = rolling_variance(tox_series, ROLLING_WINDOW)
            window = var_series[adj_start:adj_end]
            tau, _ = kendall_tau_trend(window)
            if not np.isnan(tau):
                target["toxicity_VAR"].append(tau)
        
        for ind in indicators:
            d_vals = derail_taus[ind]
            c_vals = civil_taus[ind]
            
            d_mean = np.mean(d_vals) if d_vals else np.nan
            c_mean = np.mean(c_vals) if c_vals else np.nan
            
            if d_vals and c_vals and len(d_vals) >= 10 and len(c_vals) >= 10:
                perm = vectorized_permutation_test(
                    d_vals, c_vals, n_permutations=PERMUTATION_N, seed=RANDOM_SEED
                )
                p_val = perm["p_value"]
            else:
                p_val = np.nan
            
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "†" if p_val < 0.1 else ""
            
            print(f"k={k:<4} {ind:<16} {len(d_vals):>5} {len(c_vals):>5} "
                  f"{d_mean:>10.4f} {c_mean:>10.4f} {p_val:>10.4f} {sig:>6}")
            
            results[f"k{k}"][ind] = {
                "n_derail": len(d_vals),
                "n_civil": len(c_vals),
                "tau_derail": round(float(d_mean), 4) if not np.isnan(d_mean) else None,
                "tau_civil": round(float(c_mean), 4) if not np.isnan(c_mean) else None,
                "p_value": round(float(p_val), 4) if not np.isnan(p_val) else None,
            }
        
        print()
    
    # Summary
    print("--- SUMMARY: Significance by lead time ---")
    print(f"{'Indicator':<16}", end="")
    for k in lead_times:
        print(f"{'k='+str(k):>8}", end="")
    print()
    
    for ind in indicators:
        print(f"{ind:<16}", end="")
        for k in lead_times:
            p = results[f"k{k}"][ind]["p_value"]
            if p is None:
                sym = "  —"
            elif p < 0.001:
                sym = "  ***"
            elif p < 0.01:
                sym = "  **"
            elif p < 0.05:
                sym = "  *"
            elif p < 0.1:
                sym = "  †"
            else:
                sym = "  ns"
            print(f"{sym:>8}", end="")
        print()
    
    print("=" * 72)
    return results


# ======================================================================
# Experiment 7b: Length–Dose-Response Analysis (CGA-CMV)
# ======================================================================

def experiment_7b_dose_response(utt_df):
    """
    Stratify CGA-CMV conversations by length (quartiles), compute CSD
    effect sizes within each stratum. Test whether effect size increases
    monotonically with conversation length.
    """
    print("\n" + "=" * 72)
    print("EXPERIMENT 7b: Length–Dose-Response Analysis (CGA-CMV)")
    print("=" * 72)

    ge_columns = [c for c in utt_df.columns if c.startswith("ge_")]
    
    # Get conversation lengths
    convo_lengths = utt_df.groupby("convo_id").size().rename("n_turns")
    convo_labels = utt_df.groupby("convo_id")["derails"].first()
    convo_info = pd.DataFrame({"n_turns": convo_lengths, "derails": convo_labels})
    
    # Define length strata (finer bins for better trend resolution)
    bins = [(5, 6, "5–6"), (7, 7, "7"), (8, 9, "8–9"), (10, 12, "10–12"), (13, 999, "13+")]
    
    indicators = ["q_da_VAR", "AMD_VAR", "vader_VAR"]
    
    results = {"strata": [], "trend_test": {}}
    
    print(f"\n{'Stratum':<10} {'N':>5} {'N_d':>5} {'N_c':>5} ", end="")
    for ind in indicators:
        print(f"{'Δτ(' + ind + ')':>16} {'p':>8}", end="")
    print()
    print("-" * 100)
    
    # Per-stratum effect sizes for trend test
    stratum_effects = {ind: [] for ind in indicators}
    stratum_midpoints = []
    
    for lo, hi, label in bins:
        stratum_ids = convo_info[(convo_info["n_turns"] >= lo) & (convo_info["n_turns"] <= hi)].index
        stratum_utt = utt_df[utt_df["convo_id"].isin(stratum_ids)]
        
        n_convos = len(stratum_ids)
        n_derail = convo_info.loc[stratum_ids, "derails"].sum()
        n_civil = n_convos - n_derail
        
        if n_derail < 10 or n_civil < 10:
            print(f"{label:<10} {n_convos:>5} {n_derail:>5} {n_civil:>5}  (too few)")
            for ind in indicators:
                stratum_effects[ind].append(np.nan)
            stratum_midpoints.append((lo + min(hi, 15)) / 2)
            results["strata"].append({
                "label": label, "n": int(n_convos),
                "n_derail": int(n_derail), "n_civil": int(n_civil),
                "results": {ind: {"effect": None, "p_value": None} for ind in indicators}
            })
            continue
        
        # Compute CSD taus for this stratum
        derail_taus = {ind: [] for ind in indicators}
        civil_taus = {ind: [] for ind in indicators}
        
        for convo_id, group in stratum_utt.groupby("convo_id"):
            group = group.sort_values("turn_idx")
            is_derailing = bool(group["derails"].iloc[0])
            n_turns = len(group)
            
            if n_turns < ROLLING_WINDOW + 2:
                continue
            
            target = derail_taus if is_derailing else civil_taus
            
            # q_da VAR
            q_da_series = group["q_da"].values.astype(float)
            if len(q_da_series) >= ROLLING_WINDOW:
                var_s = rolling_variance(q_da_series, ROLLING_WINDOW)
                tau, _ = kendall_tau_trend(var_s)
                if not np.isnan(tau):
                    target["q_da_VAR"].append(tau)
            
            # AMD VAR
            if len(ge_columns) > 0 and n_turns >= ROLLING_WINDOW:
                ge_matrix = group[ge_columns].values
                turn_vars = np.var(ge_matrix, axis=1)
                var_of_var = rolling_variance(turn_vars, ROLLING_WINDOW)
                tau, _ = kendall_tau_trend(var_of_var)
                if not np.isnan(tau):
                    target["AMD_VAR"].append(tau)
            
            # VADER VAR
            vader_series = group["vader_compound"].values.astype(float)
            if len(vader_series) >= ROLLING_WINDOW:
                var_s = rolling_variance(vader_series, ROLLING_WINDOW)
                tau, _ = kendall_tau_trend(var_s)
                if not np.isnan(tau):
                    target["vader_VAR"].append(tau)
        
        stratum_midpoints.append((lo + min(hi, 15)) / 2)
        stratum_result = {}
        
        print(f"{label:<10} {n_convos:>5} {int(n_derail):>5} {int(n_civil):>5} ", end="")
        
        for ind in indicators:
            d_vals = derail_taus[ind]
            c_vals = civil_taus[ind]
            d_mean = np.mean(d_vals) if d_vals else np.nan
            c_mean = np.mean(c_vals) if c_vals else np.nan
            effect = d_mean - c_mean  # negative = CSD in derailing (expected direction)
            
            if d_vals and c_vals and len(d_vals) >= 10 and len(c_vals) >= 10:
                perm = vectorized_permutation_test(
                    d_vals, c_vals, n_permutations=PERMUTATION_N, seed=RANDOM_SEED
                )
                p_val = perm["p_value"]
            else:
                p_val = np.nan
            
            stratum_effects[ind].append(effect)
            stratum_result[ind] = {
                "n_derail_valid": len(d_vals),
                "n_civil_valid": len(c_vals),
                "tau_derail": round(float(d_mean), 4) if not np.isnan(d_mean) else None,
                "tau_civil": round(float(c_mean), 4) if not np.isnan(c_mean) else None,
                "effect": round(float(effect), 4) if not np.isnan(effect) else None,
                "p_value": round(float(p_val), 4) if not np.isnan(p_val) else None,
            }
            
            sig = "*" if p_val < 0.05 else "†" if p_val < 0.1 else ""
            print(f"{effect:>12.4f}{sig:>4} {p_val:>8.4f}", end="")
        
        print()
        results["strata"].append({
            "label": label, "n": int(n_convos),
            "n_derail": int(n_derail), "n_civil": int(n_civil),
            "results": stratum_result
        })
    
    # Trend test: Kendall tau of effect size vs stratum midpoint
    print(f"\n--- DOSE-RESPONSE TREND (effect size vs. conversation length) ---")
    for ind in indicators:
        effects = np.array(stratum_effects[ind])
        midpoints = np.array(stratum_midpoints)
        valid = ~np.isnan(effects)
        if valid.sum() >= 3:
            # We want MORE NEGATIVE effect with longer convos (stronger CSD)
            # So test if effect becomes more negative with length
            tau, p = stats.kendalltau(midpoints[valid], effects[valid])
            # Also compute Spearman for robustness
            rho, p_rho = stats.spearmanr(midpoints[valid], effects[valid])
            print(f"  {ind}: Kendall τ={tau:.3f} (p={p:.4f}), "
                  f"Spearman ρ={rho:.3f} (p={p_rho:.4f})")
            
            direction = "stronger CSD with length ✓" if tau < 0 else "no monotonic trend"
            print(f"    → {direction}")
            
            results["trend_test"][ind] = {
                "kendall_tau": round(float(tau), 4),
                "kendall_p": round(float(p), 4),
                "spearman_rho": round(float(rho), 4),
                "spearman_p": round(float(p_rho), 4),
            }
        else:
            print(f"  {ind}: insufficient valid strata for trend test")
            results["trend_test"][ind] = None
    
    # Also report: overall CGA-CMV restricted to 12+ turns only
    print(f"\n--- BONUS: CGA-CMV restricted to 12+ turns ---")
    long_ids = convo_info[convo_info["n_turns"] >= 12].index
    long_utt = utt_df[utt_df["convo_id"].isin(long_ids)]
    n_long = len(long_ids)
    n_long_d = convo_info.loc[long_ids, "derails"].sum()
    n_long_c = n_long - n_long_d
    print(f"  N={n_long} ({int(n_long_d)} derail, {int(n_long_c)} civil)")
    
    derail_taus_long = {ind: [] for ind in indicators}
    civil_taus_long = {ind: [] for ind in indicators}
    
    for convo_id, group in long_utt.groupby("convo_id"):
        group = group.sort_values("turn_idx")
        is_derailing = bool(group["derails"].iloc[0])
        target = derail_taus_long if is_derailing else civil_taus_long
        
        q_da_series = group["q_da"].values.astype(float)
        if len(q_da_series) >= ROLLING_WINDOW:
            var_s = rolling_variance(q_da_series, ROLLING_WINDOW)
            tau, _ = kendall_tau_trend(var_s)
            if not np.isnan(tau):
                target["q_da_VAR"].append(tau)
        
        if len(ge_columns) > 0 and len(group) >= ROLLING_WINDOW:
            ge_matrix = group[ge_columns].values
            turn_vars = np.var(ge_matrix, axis=1)
            var_of_var = rolling_variance(turn_vars, ROLLING_WINDOW)
            tau, _ = kendall_tau_trend(var_of_var)
            if not np.isnan(tau):
                target["AMD_VAR"].append(tau)
        
        vader_series = group["vader_compound"].values.astype(float)
        if len(vader_series) >= ROLLING_WINDOW:
            var_s = rolling_variance(vader_series, ROLLING_WINDOW)
            tau, _ = kendall_tau_trend(var_s)
            if not np.isnan(tau):
                target["vader_VAR"].append(tau)
    
    results["long_only"] = {}
    for ind in indicators:
        d_vals = derail_taus_long[ind]
        c_vals = civil_taus_long[ind]
        d_mean = np.mean(d_vals) if d_vals else np.nan
        c_mean = np.mean(c_vals) if c_vals else np.nan
        
        if d_vals and c_vals and len(d_vals) >= 10 and len(c_vals) >= 10:
            perm = vectorized_permutation_test(
                d_vals, c_vals, n_permutations=PERMUTATION_N, seed=RANDOM_SEED
            )
            p_val = perm["p_value"]
        else:
            p_val = np.nan
        
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "†" if p_val < 0.1 else ""
        print(f"  {ind}: τ_d={d_mean:.4f}, τ_c={c_mean:.4f}, "
              f"Δ={d_mean-c_mean:.4f}, p={p_val:.4f} {sig}")
        
        results["long_only"][ind] = {
            "n_derail": len(d_vals),
            "n_civil": len(c_vals),
            "tau_derail": round(float(d_mean), 4) if not np.isnan(d_mean) else None,
            "tau_civil": round(float(c_mean), 4) if not np.isnan(c_mean) else None,
            "effect": round(float(d_mean - c_mean), 4) if not (np.isnan(d_mean) or np.isnan(c_mean)) else None,
            "p_value": round(float(p_val), 4) if not np.isnan(p_val) else None,
        }
    
    print("=" * 72)
    return results


# ======================================================================
# Main
# ======================================================================

def main():
    print("Loading CGA-Wiki features...")
    wiki_utt = load_cga_features()
    print(f"  {len(wiki_utt)} utterances, {wiki_utt['convo_id'].nunique()} conversations\n")
    
    results_7a = experiment_7a_lead_time(wiki_utt)
    
    print("\nLoading CGA-CMV features...")
    cmv_utt = load_cga_cmv_features()
    print(f"  {len(cmv_utt)} utterances, {cmv_utt['convo_id'].nunique()} conversations\n")
    
    results_7b = experiment_7b_dose_response(cmv_utt)
    
    # Save results
    combined = {
        "exp7a_lead_time": results_7a,
        "exp7b_dose_response": results_7b,
    }
    
    out_path = RESULTS_DIR / "exp7_replacement.json"
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2)
    
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
