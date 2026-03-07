"""
CMV analysis script for Experiments 6 and 7.

Experiment 6: AMD Trajectory Divergence + CSD Indicators
  6a: In delta-awarded threads, D_cond should decrease (negative Kendall
      tau). In no-delta threads, D_cond stays flat or increases.
  6b: CSD indicators on CMV. In delta threads, AC1 of GoEmotions
      variance should be stable (no critical slowing down). In no-delta
      threads, AC1 may rise (approaching the bifurcation).

Experiment 7: Half-Window Convergence with Engagement Controls
  D_cond should drop from first to second half more in delta threads
  than in no-delta threads, after matching on turn count, mean word
  count, and total words to control for the engagement confounder.
"""

import json
import re
import sys
from collections import Counter, defaultdict
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
    CMV_FEATURES_PATH,
    CMV_CONVERSATION_SUMMARY_PATH,
)
from utils.amd import total_variation
from utils.csd_indicators import (
    rolling_lag1_autocorrelation,
    rolling_variance,
    kendall_tau_trend,
)
from utils.statistical_tests import vectorized_permutation_test, pearson_with_pvalue

WORD_PATTERN = re.compile(r"\b[a-z]{2,}\b")
try:
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words("english"))
except Exception:
    STOPWORDS = set()

MIN_ANCHOR_FREQ = 2
MIN_PER_CELL = 1


def load_cmv_data():
    utt_df = pd.read_parquet(CMV_FEATURES_PATH)
    summary_df = pd.read_parquet(CMV_CONVERSATION_SUMMARY_PATH)
    return utt_df, summary_df


def recompute_rolling_amd(utt_df, window_size=ROLLING_WINDOW):
    ge_columns = [c for c in utt_df.columns if c.startswith("ge_")]
    n_emo = len(ge_columns)

    results = []

    for convo_id, group in utt_df.groupby("convo_id"):
        group = group.sort_values("turn_idx")
        speakers = group["speaker"].unique()
        if len(speakers) < 2:
            continue

        is_op = group["is_op"].values
        spk1 = group[is_op]["speaker"].iloc[0] if is_op.any() else speakers[0]
        spk2 = speakers[0] if speakers[0] != spk1 else speakers[1]
        delta = group["delta"].iloc[0]

        records = []
        for _, row in group.iterrows():
            records.append({
                "speaker": row["speaker"],
                "text": str(row["text"]),
                "context": row["context"],
                "ge": row[ge_columns].values.astype(float),
            })

        n = len(records)
        if n < window_size:
            continue

        d_cond_series = []
        d_marg_series = []

        for start in range(n - window_size + 1):
            window = records[start:start + window_size]

            words_s1, words_s2 = [], []
            for r in window:
                cw = [w for w in WORD_PATTERN.findall(r["text"].lower()) if w not in STOPWORDS]
                if r["speaker"] == spk1:
                    words_s1.extend(cw)
                else:
                    words_s2.extend(cw)

            c1, c2 = Counter(words_s1), Counter(words_s2)
            shared = set(c1.keys()) & set(c2.keys())
            anchors = [w for w in shared if c1[w] + c2[w] >= MIN_ANCHOR_FREQ]

            if not anchors:
                d_marg_series.append(np.nan)
                d_cond_series.append(np.nan)
                continue

            anchor_map = defaultdict(lambda: defaultdict(list))
            for r in window:
                text_words = set(WORD_PATTERN.findall(r["text"].lower()))
                for anchor in anchors:
                    if anchor in text_words:
                        anchor_map[(anchor, r["speaker"])][r["context"]].append(r["ge"])

            d_marg_vals, d_cond_vals = [], []

            for anchor in anchors:
                g1 = anchor_map.get((anchor, spk1), {})
                g2 = anchor_map.get((anchor, spk2), {})

                ed1, ed2, cnt1, cnt2 = {}, {}, {}, {}
                for ctx, dists in g1.items():
                    if len(dists) >= MIN_PER_CELL:
                        ed1[ctx] = np.mean(dists, axis=0)
                        cnt1[ctx] = len(dists)
                for ctx, dists in g2.items():
                    if len(dists) >= MIN_PER_CELL:
                        ed2[ctx] = np.mean(dists, axis=0)
                        cnt2[ctx] = len(dists)

                if not ed1 or not ed2:
                    continue

                t1 = sum(cnt1.values())
                t2 = sum(cnt2.values())
                cw1 = {c: nn / t1 for c, nn in cnt1.items()}
                cw2 = {c: nn / t2 for c, nn in cnt2.items()}

                all_ctx = set(cw1.keys()) | set(cw2.keys())
                marg1 = sum(cw1.get(c, 0) * ed1.get(c, np.zeros(n_emo)) for c in all_ctx)
                marg2 = sum(cw2.get(c, 0) * ed2.get(c, np.zeros(n_emo)) for c in all_ctx)
                d_marg_vals.append(total_variation(marg1, marg2))

                shared_ctx = set(ed1.keys()) & set(ed2.keys())
                if shared_ctx:
                    wt_sum, tv_sum = 0.0, 0.0
                    for c in shared_ctx:
                        w = cnt1[c] + cnt2[c]
                        tv_sum += w * total_variation(ed1[c], ed2[c])
                        wt_sum += w
                    if wt_sum > 0:
                        d_cond_vals.append(tv_sum / wt_sum)

            d_marg_series.append(np.mean(d_marg_vals) if d_marg_vals else np.nan)
            d_cond_series.append(np.mean(d_cond_vals) if d_cond_vals else np.nan)

        d_cond_arr = np.array(d_cond_series, dtype=float)
        d_marg_arr = np.array(d_marg_series, dtype=float)
        valid_cond = d_cond_arr[~np.isnan(d_cond_arr)]
        valid_marg = d_marg_arr[~np.isnan(d_marg_arr)]

        if len(valid_cond) < 3:
            continue

        tau_cond, _ = stats.kendalltau(np.arange(len(valid_cond)), valid_cond)
        tau_marg, _ = stats.kendalltau(np.arange(len(valid_marg)), valid_marg)

        first_half = valid_cond[:len(valid_cond) // 2]
        second_half = valid_cond[len(valid_cond) // 2:]
        mean_first = np.mean(first_half) if len(first_half) > 0 else np.nan
        mean_second = np.mean(second_half) if len(second_half) > 0 else np.nan
        d_cond_change = mean_second - mean_first

        word_counts = [len(r["text"].split()) for r in records]

        ge_matrix = np.array([r["ge"] for r in records])
        turn_ge_var = np.var(ge_matrix, axis=1)
        ac1_ge = rolling_lag1_autocorrelation(turn_ge_var, ROLLING_WINDOW)
        var_ge = rolling_variance(turn_ge_var, ROLLING_WINDOW)
        v_ac1 = ac1_ge[~np.isnan(ac1_ge)]
        v_var = var_ge[~np.isnan(var_ge)]
        tau_ac1_ge = stats.kendalltau(np.arange(len(v_ac1)), v_ac1)[0] if len(v_ac1) >= 3 else np.nan
        tau_var_ge = stats.kendalltau(np.arange(len(v_var)), v_var)[0] if len(v_var) >= 3 else np.nan

        if len(valid_cond) >= ROLLING_WINDOW:
            ac1_dc = rolling_lag1_autocorrelation(valid_cond, ROLLING_WINDOW)
            var_dc = rolling_variance(valid_cond, ROLLING_WINDOW)
            v_ac1_dc = ac1_dc[~np.isnan(ac1_dc)]
            v_var_dc = var_dc[~np.isnan(var_dc)]
            tau_ac1_dc = stats.kendalltau(np.arange(len(v_ac1_dc)), v_ac1_dc)[0] if len(v_ac1_dc) >= 3 else np.nan
            tau_var_dc = stats.kendalltau(np.arange(len(v_var_dc)), v_var_dc)[0] if len(v_var_dc) >= 3 else np.nan
        else:
            tau_ac1_dc, tau_var_dc = np.nan, np.nan

        results.append({
            "convo_id": str(convo_id),
            "delta": bool(delta),
            "n_turns": n,
            "n_valid_cond": int(len(valid_cond)),
            "mean_d_cond": float(np.mean(valid_cond)),
            "mean_d_marg": float(np.mean(valid_marg)),
            "tau_d_cond": float(tau_cond),
            "tau_d_marg": float(tau_marg),
            "d_cond_first_half": float(mean_first),
            "d_cond_second_half": float(mean_second),
            "d_cond_change": float(d_cond_change),
            "mean_word_count": float(np.mean(word_counts)),
            "total_words": int(sum(word_counts)),
            "tau_ac1_ge_var": float(tau_ac1_ge) if not np.isnan(tau_ac1_ge) else None,
            "tau_var_ge_var": float(tau_var_ge) if not np.isnan(tau_var_ge) else None,
            "tau_ac1_dcond": float(tau_ac1_dc) if not np.isnan(tau_ac1_dc) else None,
            "tau_var_dcond": float(tau_var_dc) if not np.isnan(tau_var_dc) else None,
        })

    return pd.DataFrame(results)


def experiment_6a_trajectory(summary_df):
    print("=" * 72)
    print("EXPERIMENT 6a: AMD Trajectory Divergence (CMV)")
    print("=" * 72)

    valid = summary_df.dropna(subset=["tau_d_cond"])
    delta_df = valid[valid["delta"] == True]
    nodelta_df = valid[valid["delta"] == False]

    print(f"Conversations with valid AMD trajectory: {len(valid)}")
    print(f"  Delta-awarded: {len(delta_df)}")
    print(f"  No-delta: {len(nodelta_df)}")

    tau_delta = delta_df["tau_d_cond"].values
    tau_nodelta = nodelta_df["tau_d_cond"].values

    mean_tau_delta = np.mean(tau_delta)
    mean_tau_nodelta = np.mean(tau_nodelta)
    median_tau_delta = np.median(tau_delta)
    median_tau_nodelta = np.median(tau_nodelta)

    u_stat, u_p = stats.mannwhitneyu(tau_delta, tau_nodelta, alternative="less")

    perm = vectorized_permutation_test(
        tau_delta.tolist(), tau_nodelta.tolist(),
        n_permutations=PERMUTATION_N, seed=RANDOM_SEED,
    )
    perm_p = perm["p_value"]

    t_stat, t_p = stats.ttest_ind(tau_delta, tau_nodelta, alternative="less")

    neg_frac_delta = np.mean(tau_delta < 0)
    neg_frac_nodelta = np.mean(tau_nodelta < 0)

    change_delta = delta_df["d_cond_change"].dropna().values
    change_nodelta = nodelta_df["d_cond_change"].dropna().values
    if len(change_delta) >= 5 and len(change_nodelta) >= 5:
        u_change, u_change_p = stats.mannwhitneyu(
            change_delta, change_nodelta, alternative="less"
        )
    else:
        u_change, u_change_p = np.nan, np.nan

    mean_d_delta = delta_df["mean_d_cond"].mean()
    mean_d_nodelta = nodelta_df["mean_d_cond"].mean()

    results = {
        "n_delta": len(delta_df),
        "n_nodelta": len(nodelta_df),
        "mean_tau_delta": round(float(mean_tau_delta), 4),
        "mean_tau_nodelta": round(float(mean_tau_nodelta), 4),
        "median_tau_delta": round(float(median_tau_delta), 4),
        "median_tau_nodelta": round(float(median_tau_nodelta), 4),
        "mann_whitney_p": round(float(u_p), 4),
        "permutation_p": round(float(perm_p), 4),
        "t_test_p": round(float(t_p), 4),
        "neg_frac_delta": round(float(neg_frac_delta), 4),
        "neg_frac_nodelta": round(float(neg_frac_nodelta), 4),
        "mean_d_cond_delta": round(float(mean_d_delta), 4),
        "mean_d_cond_nodelta": round(float(mean_d_nodelta), 4),
        "mean_change_delta": round(float(np.mean(change_delta)), 4) if len(change_delta) > 0 else None,
        "mean_change_nodelta": round(float(np.mean(change_nodelta)), 4) if len(change_nodelta) > 0 else None,
        "change_mann_whitney_p": round(float(u_change_p), 4) if not np.isnan(u_change_p) else None,
    }

    print(f"\n{'Measure':<45} {'Delta':>10} {'No-Delta':>10}")
    print("-" * 67)
    print(f"{'Mean Kendall tau (D_cond)':<45} {mean_tau_delta:>10.4f} {mean_tau_nodelta:>10.4f}")
    print(f"{'Median Kendall tau (D_cond)':<45} {median_tau_delta:>10.4f} {median_tau_nodelta:>10.4f}")
    print(f"{'Fraction with negative tau':<45} {neg_frac_delta:>10.4f} {neg_frac_nodelta:>10.4f}")
    print(f"{'Mean D_cond':<45} {mean_d_delta:>10.4f} {mean_d_nodelta:>10.4f}")
    if len(change_delta) > 0 and len(change_nodelta) > 0:
        print(f"{'Mean D_cond change (2nd-1st half)':<45} {np.mean(change_delta):>10.4f} {np.mean(change_nodelta):>10.4f}")

    print(f"\n{'Test':<50} {'p-value':>10}")
    print("-" * 62)
    print(f"{'Mann-Whitney U (tau_delta < tau_nodelta)':<50} {u_p:>10.4f}")
    print(f"{'Permutation test':<50} {perm_p:>10.4f}")
    print(f"{'Welch t-test':<50} {t_p:>10.4f}")
    if not np.isnan(u_change_p):
        print(f"{'M-W U (change_delta < change_nodelta)':<50} {u_change_p:>10.4f}")
    print("=" * 72)

    return results


def experiment_6b_csd(summary_df):
    print("\n" + "=" * 72)
    print("EXPERIMENT 6b: CSD Indicators on CMV")
    print("=" * 72)

    csd_cols = ["tau_ac1_ge_var", "tau_var_ge_var", "tau_ac1_dcond", "tau_var_dcond"]
    existing = [c for c in csd_cols if c in summary_df.columns]

    if not existing:
        print("No CSD columns found in summary. Skipping.")
        return {}

    results = {}

    indicators = [
        ("tau_ac1_ge_var", "AC1 trend of GoEmotions variance"),
        ("tau_var_ge_var", "Variance trend of GoEmotions variance"),
        ("tau_ac1_dcond", "AC1 trend of D_cond trajectory"),
        ("tau_var_dcond", "Variance trend of D_cond trajectory"),
    ]

    print(f"\n{'Indicator':<45} {'Delta':>10} {'No-Delta':>10} {'M-W p':>10} {'Perm p':>10}")
    print("-" * 87)

    for col, label in indicators:
        if col not in summary_df.columns:
            continue

        valid = summary_df.dropna(subset=[col])
        delta_vals = valid[valid["delta"] == True][col].values
        nodelta_vals = valid[valid["delta"] == False][col].values

        if len(delta_vals) < 5 or len(nodelta_vals) < 5:
            print(f"{label:<45} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10}")
            continue

        mean_d = np.mean(delta_vals)
        mean_nd = np.mean(nodelta_vals)

        _, mw_p = stats.mannwhitneyu(delta_vals, nodelta_vals, alternative="two-sided")

        perm = vectorized_permutation_test(
            delta_vals.tolist(), nodelta_vals.tolist(),
            n_permutations=PERMUTATION_N, seed=RANDOM_SEED,
        )
        perm_p = perm["p_value"]

        results[f"{col}_delta"] = round(float(mean_d), 4)
        results[f"{col}_nodelta"] = round(float(mean_nd), 4)
        results[f"{col}_mw_p"] = round(float(mw_p), 4)
        results[f"{col}_perm_p"] = round(float(perm_p), 4)
        results[f"{col}_n_delta"] = len(delta_vals)
        results[f"{col}_n_nodelta"] = len(nodelta_vals)

        print(f"{label:<45} {mean_d:>10.4f} {mean_nd:>10.4f} {mw_p:>10.4f} {perm_p:>10.4f}")

    print("=" * 72)
    return results


def experiment_7_convergence(summary_df):
    print("\n" + "=" * 72)
    print("EXPERIMENT 7: Half-Window Convergence with Controls (CMV)")
    print("=" * 72)

    valid = summary_df.dropna(subset=["d_cond_first_half", "d_cond_second_half"])
    delta_df = valid[valid["delta"] == True].copy()
    nodelta_df = valid[valid["delta"] == False].copy()

    print(f"Conversations with valid half-window AMD: {len(valid)}")
    print(f"  Delta: {len(delta_df)}, No-delta: {len(nodelta_df)}")

    has_engagement = "mean_word_count" in valid.columns and "total_words" in valid.columns

    if has_engagement:
        print(f"\nEngagement stats before matching:")
        for tag, df in [("Delta", delta_df), ("No-delta", nodelta_df)]:
            print(f"  {tag}: n_turns={df['n_turns'].mean():.1f}, "
                  f"mean_wc={df['mean_word_count'].mean():.1f}, "
                  f"total_words={df['total_words'].mean():.0f}")

    results = _run_convergence_tests(delta_df, nodelta_df, "unmatched")

    if has_engagement:
        matched_delta, matched_nodelta = _propensity_match(delta_df, nodelta_df)
        if matched_delta is not None and len(matched_delta) >= 20:
            print(f"\nAfter matching on engagement (n_turns, mean_wc, total_words):")
            print(f"  Matched pairs: {len(matched_delta)}")
            for tag, df in [("Delta", matched_delta), ("No-delta", matched_nodelta)]:
                print(f"  {tag}: n_turns={df['n_turns'].mean():.1f}, "
                      f"mean_wc={df['mean_word_count'].mean():.1f}, "
                      f"total_words={df['total_words'].mean():.0f}")

            matched_results = _run_convergence_tests(matched_delta, matched_nodelta, "matched")
            results.update(matched_results)
        else:
            print("\nMatching produced too few pairs. Reporting unmatched only.")

        results.update(_partial_corr_change(valid))

    print("=" * 72)
    return results


def _propensity_match(delta_df, nodelta_df, caliper_std=0.5):
    from sklearn.preprocessing import StandardScaler

    match_vars = ["n_turns", "mean_word_count", "total_words"]
    d_feats = delta_df[match_vars].values.astype(float)
    nd_feats = nodelta_df[match_vars].values.astype(float)

    scaler = StandardScaler()
    all_feats = np.vstack([d_feats, nd_feats])
    scaler.fit(all_feats)
    d_scaled = scaler.transform(d_feats)
    nd_scaled = scaler.transform(nd_feats)

    matched_d_idx = []
    matched_nd_idx = []
    used_nd = set()

    for i in range(len(d_scaled)):
        dists = np.sqrt(np.sum((nd_scaled - d_scaled[i]) ** 2, axis=1))
        for j in np.argsort(dists):
            if j not in used_nd and dists[j] < caliper_std:
                matched_d_idx.append(i)
                matched_nd_idx.append(j)
                used_nd.add(j)
                break

    if len(matched_d_idx) < 10:
        return None, None

    return delta_df.iloc[matched_d_idx].copy(), nodelta_df.iloc[matched_nd_idx].copy()


def _run_convergence_tests(delta_df, nodelta_df, prefix):
    delta_first = delta_df["d_cond_first_half"].values
    delta_second = delta_df["d_cond_second_half"].values
    nodelta_first = nodelta_df["d_cond_first_half"].values
    nodelta_second = nodelta_df["d_cond_second_half"].values

    delta_change = delta_second - delta_first
    nodelta_change = nodelta_second - nodelta_first

    t_delta, p_delta = stats.ttest_rel(delta_second, delta_first, alternative="less")
    t_nodelta, p_nodelta = stats.ttest_rel(nodelta_second, nodelta_first, alternative="less")

    if len(delta_change) >= 10:
        w_delta, wp_delta = stats.wilcoxon(delta_change, alternative="less")
    else:
        wp_delta = np.nan
    if len(nodelta_change) >= 10:
        w_nodelta, wp_nodelta = stats.wilcoxon(nodelta_change, alternative="less")
    else:
        wp_nodelta = np.nan

    delta_drops = np.mean(delta_change < 0)
    nodelta_drops = np.mean(nodelta_change < 0)

    did_u, did_p = stats.mannwhitneyu(delta_change, nodelta_change, alternative="less")

    d_delta = np.mean(delta_change) / np.std(delta_change) if np.std(delta_change) > 0 else 0
    d_nodelta = np.mean(nodelta_change) / np.std(nodelta_change) if np.std(nodelta_change) > 0 else 0

    results = {
        f"{prefix}_delta_mean_first": round(float(np.mean(delta_first)), 4),
        f"{prefix}_delta_mean_second": round(float(np.mean(delta_second)), 4),
        f"{prefix}_nodelta_mean_first": round(float(np.mean(nodelta_first)), 4),
        f"{prefix}_nodelta_mean_second": round(float(np.mean(nodelta_second)), 4),
        f"{prefix}_delta_paired_t_p": round(float(p_delta), 4),
        f"{prefix}_nodelta_paired_t_p": round(float(p_nodelta), 4),
        f"{prefix}_delta_wilcoxon_p": round(float(wp_delta), 4) if not np.isnan(wp_delta) else None,
        f"{prefix}_nodelta_wilcoxon_p": round(float(wp_nodelta), 4) if not np.isnan(wp_nodelta) else None,
        f"{prefix}_delta_drop_frac": round(float(delta_drops), 4),
        f"{prefix}_nodelta_drop_frac": round(float(nodelta_drops), 4),
        f"{prefix}_did_mann_whitney_p": round(float(did_p), 4),
        f"{prefix}_cohen_d_delta": round(float(d_delta), 4),
        f"{prefix}_cohen_d_nodelta": round(float(d_nodelta), 4),
        f"{prefix}_n_delta": len(delta_df),
        f"{prefix}_n_nodelta": len(nodelta_df),
    }

    print(f"\n  [{prefix.upper()}]")
    print(f"  {'Measure':<45} {'Delta':>10} {'No-Delta':>10}")
    print(f"  " + "-" * 67)
    print(f"  {'D_cond first half':<45} {np.mean(delta_first):>10.4f} {np.mean(nodelta_first):>10.4f}")
    print(f"  {'D_cond second half':<45} {np.mean(delta_second):>10.4f} {np.mean(nodelta_second):>10.4f}")
    print(f"  {'Mean change (2nd - 1st)':<45} {np.mean(delta_change):>10.4f} {np.mean(nodelta_change):>10.4f}")
    print(f"  {'Fraction with decreasing D_cond':<45} {delta_drops:>10.4f} {nodelta_drops:>10.4f}")
    print(f"  {'Cohen d of change':<45} {d_delta:>10.4f} {d_nodelta:>10.4f}")

    print(f"\n  {'Test':<50} {'p-value':>10}")
    print(f"  " + "-" * 62)
    print(f"  {'Paired t (2nd < 1st) -- delta':<50} {p_delta:>10.4f}")
    print(f"  {'Paired t (2nd < 1st) -- no-delta':<50} {p_nodelta:>10.4f}")
    if not np.isnan(wp_delta):
        print(f"  {'Wilcoxon signed-rank -- delta':<50} {wp_delta:>10.4f}")
    if not np.isnan(wp_nodelta):
        print(f"  {'Wilcoxon signed-rank -- no-delta':<50} {wp_nodelta:>10.4f}")
    print(f"  {'DiD: delta change < nodelta change (M-W U)':<50} {did_p:>10.4f}")

    return results


def _partial_corr_change(valid):
    results = {}

    delta_num = valid["delta"].astype(float).values
    change = valid["d_cond_change"].values
    mask = ~np.isnan(change)

    if mask.sum() < 20:
        return results

    r_raw, p_raw = pearson_with_pvalue(delta_num[mask], change[mask])
    results["raw_corr_delta_change"] = round(float(r_raw), 4)
    results["raw_p_delta_change"] = round(float(p_raw), 4)

    from utils.statistical_tests import partial_correlation
    covariates = np.column_stack([
        valid["n_turns"].values[mask].astype(float),
        valid["mean_word_count"].values[mask].astype(float),
        valid["total_words"].values[mask].astype(float),
    ])

    pr, pp = partial_correlation(delta_num[mask], change[mask], covariates)
    results["partial_corr_delta_change"] = round(float(pr), 4)
    results["partial_p_delta_change"] = round(float(pp), 4)

    print(f"\n  Partial correlation: delta vs D_cond change")
    print(f"  {'Raw correlation':<50} r={r_raw:.4f}, p={p_raw:.4f}")
    print(f"  {'Controlling for n_turns, mean_wc, total_words':<50} r={pr:.4f}, p={pp:.4f}")

    return results


def main():
    np.random.seed(RANDOM_SEED)

    utt_df, summary_df = load_cmv_data()
    n_convos = utt_df["convo_id"].nunique()
    print(f"Loaded {len(utt_df)} utterances from {n_convos} conversations")

    if "tau_d_cond" not in summary_df.columns or summary_df["tau_d_cond"].isna().all():
        print("Recomputing rolling AMD trajectories from utterance features...")
        summary_df = recompute_rolling_amd(utt_df)

    n_valid = summary_df["tau_d_cond"].notna().sum()
    n_delta = summary_df[summary_df["delta"] == True].shape[0]
    n_nodelta = summary_df[summary_df["delta"] == False].shape[0]
    print(f"Conversations with valid trajectory: {n_valid}")
    print(f"  Delta: {n_delta}, No-delta: {n_nodelta}\n")

    exp6a_results = experiment_6a_trajectory(summary_df)
    exp6b_results = experiment_6b_csd(summary_df)
    exp7_results = experiment_7_convergence(summary_df)

    all_results = {
        "exp6a": exp6a_results,
        "exp6b": exp6b_results,
        "exp7": exp7_results,
    }

    output_path = RESULTS_DIR / "cmv_analysis.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nAll CMV results saved to {output_path}")


if __name__ == "__main__":
    main()
