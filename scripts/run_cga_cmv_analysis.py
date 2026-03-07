"""
CGA-CMV analysis script for Experiments 6 and 7 (cross-domain replication).

Replicates Exp 4 (CSD indicators) and Exp 5 (incremental ablation) from
CGA-Wiki on the CGA-CMV corpus (Reddit ChangeMyView personal attacks).

Same methodology, different domain — provides cross-domain validation.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    RESULTS_DIR,
    RANDOM_SEED,
    ROLLING_WINDOW,
    PERMUTATION_N,
    CV_FOLDS,
    CGA_CMV_FEATURES_PATH,
    CGA_CMV_CONVERSATION_SUMMARY_PATH,
)
from utils.csd_indicators import (
    rolling_lag1_autocorrelation,
    rolling_variance,
    kendall_tau_trend,
)
from utils.statistical_tests import vectorized_permutation_test


def load_cga_cmv_features():
    """Load pre-extracted CGA-CMV features from parquet files."""
    utt_df = pd.read_parquet(CGA_CMV_FEATURES_PATH)
    summary_df = pd.read_parquet(CGA_CMV_CONVERSATION_SUMMARY_PATH)

    # Recompute q_da from da_prob_* columns
    conciliatory_cols = []
    for tag in ["aa", "bk", "br", "ba"]:
        col = f"da_prob_{tag}"
        if col in utt_df.columns:
            conciliatory_cols.append(col)

    if not conciliatory_cols:
        label_mapping = {"LABEL_2": "aa", "LABEL_10": "bk", "LABEL_25": "br", "LABEL_42": "ba"}
        for lbl in label_mapping:
            col = f"da_prob_{lbl}"
            if col in utt_df.columns:
                conciliatory_cols.append(col)

    if conciliatory_cols:
        utt_df["q_da"] = utt_df[conciliatory_cols].sum(axis=1)
        print(f"  Recomputed q_da from {conciliatory_cols}")
        print(f"  q_da mean={utt_df['q_da'].mean():.4f}, std={utt_df['q_da'].std():.4f}")
    else:
        print("  WARNING: No da_prob columns found; q_da unchanged")

    return utt_df, summary_df


def compute_lexical_divergence(texts_speaker1, texts_speaker2):
    divergences = []
    min_len = min(len(texts_speaker1), len(texts_speaker2))
    for i in range(min_len):
        words1 = set(texts_speaker1[i].lower().split())
        words2 = set(texts_speaker2[i].lower().split())
        union = words1 | words2
        if len(union) == 0:
            divergences.append(0.0)
        else:
            divergences.append(1.0 - len(words1 & words2) / len(union))
    return np.mean(divergences) if divergences else 0.0


def experiment_6_csd_indicators(utt_df):
    """
    Experiment 6: CSD indicators on CGA-CMV (cross-domain replication of Exp 4).
    """
    print("=" * 72)
    print("EXPERIMENT 6: CSD Indicators (CGA-CMV Cross-Domain Replication)")
    print("=" * 72)

    ge_columns = [c for c in utt_df.columns if c.startswith("ge_")]

    proxy_columns = ["q_da", "q_rm", "q_ce"]
    baseline_columns = ["toxicity_score", "vader_compound", "nrc_vad"]
    all_series_columns = proxy_columns + baseline_columns

    derail_taus = {col: {"ac1": [], "var": []} for col in all_series_columns}
    civil_taus = {col: {"ac1": [], "var": []} for col in all_series_columns}

    amd_derail_var_taus = []
    amd_civil_var_taus = []

    for convo_id, group in utt_df.groupby("convo_id"):
        group = group.sort_values("turn_idx")
        is_derailing = group["derails"].iloc[0]
        n_turns = len(group)

        if n_turns < ROLLING_WINDOW + 5:
            continue

        # CGA-CMV has no attack_turn annotation; use last 5 turns as
        # the pre-breakdown window for derailing, last 5 for civil
        pre_breakdown_end = n_turns
        pre_breakdown_start = max(0, n_turns - 5)

        for col in all_series_columns:
            if col not in group.columns:
                continue

            series = group[col].values.astype(float)
            ac1 = rolling_lag1_autocorrelation(series, ROLLING_WINDOW)
            var = rolling_variance(series, ROLLING_WINDOW)

            offset = ROLLING_WINDOW - 1
            adjusted_start = max(0, pre_breakdown_start - offset)
            adjusted_end = max(0, pre_breakdown_end - offset)

            ac1_window = ac1[adjusted_start:adjusted_end]
            var_window = var[adjusted_start:adjusted_end]

            ac1_tau, _ = kendall_tau_trend(ac1_window)
            var_tau, _ = kendall_tau_trend(var_window)

            target = derail_taus if is_derailing else civil_taus

            if not np.isnan(ac1_tau):
                target[col]["ac1"].append(ac1_tau)
            if not np.isnan(var_tau):
                target[col]["var"].append(var_tau)

        if len(ge_columns) > 0:
            ge_matrix = group[ge_columns].values
            turn_variances = np.var(ge_matrix, axis=1)
            var_series = rolling_variance(turn_variances, ROLLING_WINDOW)

            offset = ROLLING_WINDOW - 1
            adjusted_start = max(0, pre_breakdown_start - offset)
            adjusted_end = max(0, pre_breakdown_end - offset)
            var_window = var_series[adjusted_start:adjusted_end]

            tau, _ = kendall_tau_trend(var_window)
            if not np.isnan(tau):
                if is_derailing:
                    amd_derail_var_taus.append(tau)
                else:
                    amd_civil_var_taus.append(tau)

    results = {}
    tau_index = 1

    print(f"\n{'Indicator':<30} {'Derail tau':>12} {'Civil tau':>12} {'p-value':>10}")
    print("-" * 72)

    for col in all_series_columns:
        for stat_type in ["ac1", "var"]:
            d_values = derail_taus[col][stat_type]
            c_values = civil_taus[col][stat_type]

            d_mean = np.mean(d_values) if d_values else np.nan
            c_mean = np.mean(c_values) if c_values else np.nan

            if d_values and c_values:
                perm = vectorized_permutation_test(
                    d_values, c_values, n_permutations=PERMUTATION_N, seed=RANDOM_SEED
                )
                p_val = perm["p_value"]
            else:
                p_val = np.nan

            key_d = f"t{tau_index}"
            key_c = f"t{tau_index + 1}"
            key_p = f"p{tau_index}"

            results[key_d] = round(float(d_mean), 4) if not np.isnan(d_mean) else None
            results[key_c] = round(float(c_mean), 4) if not np.isnan(c_mean) else None
            results[key_p] = round(float(p_val), 4) if not np.isnan(p_val) else None

            label = f"{col} {stat_type.upper()}"
            print(f"{label:<30} {d_mean:>12.4f} {c_mean:>12.4f} {p_val:>10.4f}")

            tau_index += 2

    d_amd_mean = np.mean(amd_derail_var_taus) if amd_derail_var_taus else np.nan
    c_amd_mean = np.mean(amd_civil_var_taus) if amd_civil_var_taus else np.nan

    if amd_derail_var_taus and amd_civil_var_taus:
        perm = vectorized_permutation_test(
            amd_derail_var_taus, amd_civil_var_taus,
            n_permutations=PERMUTATION_N, seed=RANDOM_SEED
        )
        p_amd = perm["p_value"]
    else:
        p_amd = np.nan

    results["t_amd_var_derail"] = round(float(d_amd_mean), 4) if not np.isnan(d_amd_mean) else None
    results["t_amd_var_civil"] = round(float(c_amd_mean), 4) if not np.isnan(c_amd_mean) else None
    results["p_amd_var"] = round(float(p_amd), 4) if not np.isnan(p_amd) else None

    print(f"{'AMD Var':<30} {d_amd_mean:>12.4f} {c_amd_mean:>12.4f} {p_amd:>10.4f}")
    print("=" * 72)

    return results


def experiment_7_incremental_regression(utt_df):
    """
    Experiment 7: Incremental ablation on CGA-CMV (cross-domain replication of Exp 5).
    """
    print("\n" + "=" * 72)
    print("EXPERIMENT 7: Incremental Ablation (CGA-CMV Cross-Domain Replication)")
    print("=" * 72)

    ge_columns = [c for c in utt_df.columns if c.startswith("ge_")]
    conversation_features = []

    MIN_EARLY_TURNS = ROLLING_WINDOW + 1

    for convo_id, group in utt_df.groupby("convo_id"):
        group = group.sort_values("turn_idx")
        n_turns = len(group)
        cutoff = int(0.6 * n_turns)

        if cutoff < MIN_EARLY_TURNS:
            continue

        early = group.iloc[:cutoff]
        is_derailing = int(group["derails"].iloc[0])

        turn_indices = np.arange(len(early))
        tox_values = early["toxicity_score"].values.astype(float)
        vader_values = early["vader_compound"].values.astype(float)

        if len(turn_indices) >= 2 and np.std(turn_indices) > 0:
            tox_slope = np.polyfit(turn_indices, tox_values, 1)[0]
            sent_slope = np.polyfit(turn_indices, vader_values, 1)[0]
        else:
            tox_slope = 0.0
            sent_slope = 0.0

        tox_mean = np.mean(tox_values)
        tox_max = np.max(tox_values)
        sent_mean = np.mean(vader_values)
        sent_var = np.var(vader_values)

        speakers = early["speaker_id"].unique()
        if len(speakers) >= 2:
            texts_by_speaker = {}
            for spk in speakers[:2]:
                texts_by_speaker[spk] = early[early["speaker_id"] == spk]["text"].tolist()
            lex_div = compute_lexical_divergence(
                texts_by_speaker[speakers[0]], texts_by_speaker[speakers[1]]
            )
        else:
            lex_div = 0.0

        if "q_da" in early.columns:
            q_da_series = early["q_da"].values.astype(float)
            if np.std(q_da_series) > 1e-10 and len(q_da_series) >= ROLLING_WINDOW:
                ac1_series = rolling_lag1_autocorrelation(q_da_series, ROLLING_WINDOW)
                var_series = rolling_variance(q_da_series, ROLLING_WINDOW)
                ac1_q_da_tau, _ = kendall_tau_trend(ac1_series)
                var_q_da_tau, _ = kendall_tau_trend(var_series)
                ac1_q_da_tau = ac1_q_da_tau if not np.isnan(ac1_q_da_tau) else 0.0
                var_q_da_tau = var_q_da_tau if not np.isnan(var_q_da_tau) else 0.0
            else:
                ac1_q_da_tau = 0.0
                var_q_da_tau = 0.0
        else:
            ac1_q_da_tau = 0.0
            var_q_da_tau = 0.0

        if len(ge_columns) > 0:
            ge_matrix = early[ge_columns].values
            turn_variances = np.var(ge_matrix, axis=1)
            if len(turn_variances) >= ROLLING_WINDOW:
                amd_var_series = rolling_variance(turn_variances, ROLLING_WINDOW)
                amd_var_tau, _ = kendall_tau_trend(amd_var_series)
                amd_var_tau = amd_var_tau if not np.isnan(amd_var_tau) else 0.0
            else:
                amd_var_tau = 0.0
            amd_var_mean = np.var(turn_variances) if len(turn_variances) > 1 else 0.0
        else:
            amd_var_tau = 0.0
            amd_var_mean = 0.0

        if len(vader_values) >= ROLLING_WINDOW:
            sent_ac1_series = rolling_lag1_autocorrelation(vader_values, ROLLING_WINDOW)
            sent_ac1_tau, _ = kendall_tau_trend(sent_ac1_series)
            sent_ac1_tau = sent_ac1_tau if not np.isnan(sent_ac1_tau) else 0.0
        else:
            sent_ac1_tau = 0.0

        conversation_features.append({
            "convo_id": convo_id,
            "derails": is_derailing,
            "toxicity_trend": tox_slope,
            "toxicity_mean": tox_mean,
            "toxicity_max": tox_max,
            "sentiment_trend": sent_slope,
            "sentiment_mean": sent_mean,
            "sentiment_var": sent_var,
            "lexical_divergence": lex_div,
            "ac1_q_da_tau": ac1_q_da_tau,
            "var_q_da_tau": var_q_da_tau,
            "amd_var_tau": amd_var_tau,
            "amd_var_mean": amd_var_mean,
            "sent_ac1_tau": sent_ac1_tau,
        })

    feat_df = pd.DataFrame(conversation_features)
    feat_df = feat_df.fillna(0.0)

    n_derail = int(feat_df["derails"].sum())
    n_civil = int((1 - feat_df["derails"]).sum())
    print(f"Conversations with valid features: {len(feat_df)}")
    print(f"Derailing: {n_derail}, Civil: {n_civil}")

    y = feat_df["derails"].values

    if len(feat_df) < 100 or n_derail < CV_FOLDS * 2:
        cv_strategy = LeaveOneOut()
        cv_label = "LOO"
        print(f"Using Leave-One-Out CV (small sample)")
    else:
        cv_strategy = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        cv_label = f"{CV_FOLDS}-fold"
        print(f"Using {CV_FOLDS}-fold stratified CV")

    feature_sets = [
        (["toxicity_trend", "toxicity_mean", "toxicity_max"],
         "Toxicity (trend+mean+max)"),
        (["toxicity_trend", "toxicity_mean", "toxicity_max",
          "sentiment_trend", "sentiment_mean", "sentiment_var"],
         "+ Sentiment"),
        (["toxicity_trend", "toxicity_mean", "toxicity_max",
          "sentiment_trend", "sentiment_mean", "sentiment_var",
          "lexical_divergence"],
         "+ Lexical divergence"),
        (["toxicity_trend", "toxicity_mean", "toxicity_max",
          "sentiment_trend", "sentiment_mean", "sentiment_var",
          "lexical_divergence", "sent_ac1_tau"],
         "+ Sent AC1 tau (baseline CSD)"),
        (["toxicity_trend", "toxicity_mean", "toxicity_max",
          "sentiment_trend", "sentiment_mean", "sentiment_var",
          "lexical_divergence", "sent_ac1_tau",
          "ac1_q_da_tau", "var_q_da_tau"],
         "+ q_DA CSD (novel)"),
        (["toxicity_trend", "toxicity_mean", "toxicity_max",
          "sentiment_trend", "sentiment_mean", "sentiment_var",
          "lexical_divergence", "sent_ac1_tau",
          "ac1_q_da_tau", "var_q_da_tau",
          "amd_var_tau", "amd_var_mean"],
         "+ AMD CSD (novel)"),
    ]

    results = {}
    previous_auc = 0.0

    print(f"\n{'Row':<5} {'Features':<35} {'AUC':>8} {'Delta':>8}")
    print("-" * 60)

    for row_idx, (feature_names, label) in enumerate(feature_sets):
        X = feat_df[feature_names].values

        all_y_true = []
        all_y_prob = []

        for train_idx, test_idx in cv_strategy.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            if len(np.unique(y_train)) < 2:
                continue

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            clf = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.05,
                min_samples_leaf=20,
                random_state=RANDOM_SEED,
            )
            clf.fit(X_train_s, y_train)

            y_prob = clf.predict_proba(X_test_s)[:, 1]
            all_y_true.extend(y_test.tolist())
            all_y_prob.extend(y_prob.tolist())

        if len(np.unique(all_y_true)) > 1:
            mean_auc = roc_auc_score(all_y_true, all_y_prob)
        else:
            mean_auc = 0.5

        delta = mean_auc - previous_auc if row_idx > 0 else 0.0

        auc_key = f"a{row_idx + 1}"
        results[auc_key] = round(float(mean_auc), 4)

        if row_idx > 0:
            delta_key = f"d{row_idx}"
            results[delta_key] = round(float(delta), 4)

        print(f"{row_idx + 1:<5} {label:<35} {mean_auc:>8.4f} {delta:>+8.4f}")
        previous_auc = mean_auc

    results["cv_method"] = cv_label
    results["n_conversations"] = len(feat_df)
    results["n_derailing"] = n_derail
    results["n_civil"] = n_civil

    print("=" * 72)

    return results


def main():
    np.random.seed(RANDOM_SEED)

    utt_df, summary_df = load_cga_cmv_features()
    print(f"Loaded {len(utt_df)} utterances from {utt_df['convo_id'].nunique()} conversations\n")

    exp6_results = experiment_6_csd_indicators(utt_df)
    exp7_results = experiment_7_incremental_regression(utt_df)

    all_results = {
        "exp6_csd": exp6_results,
        "exp7_ablation": exp7_results,
    }

    output_path = RESULTS_DIR / "cga_cmv_analysis.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nAll CGA-CMV results saved to {output_path}")


if __name__ == "__main__":
    main()
