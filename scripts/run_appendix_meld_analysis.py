"""
MELD analysis script for Experiments 6 and 7.
Reads pre-extracted features from notebook 02 and computes AMD-rapport
correlations (Exp 6) and anchor set sensitivity analysis (Exp 7).

Replaces the original IEMOCAP analysis (IEMOCAP access not available).
MELD provides multi-party dyadic conversations from Friends with
per-utterance emotion and sentiment labels.
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
    MIN_UTTERANCES_PER_CELL,
    MELD_FEATURES_PATH,
    MELD_CONVERSATION_SUMMARY_PATH,
)
from utils.amd import total_variation
from utils.statistical_tests import pearson_with_pvalue, partial_correlation


def load_meld_features():
    """Load pre-extracted MELD features from parquet files."""
    utt_df = pd.read_parquet(MELD_FEATURES_PATH)
    return utt_df


def recompute_meld_summaries(utt_df, min_anchor_freq=2, min_per_cell=1):
    """
    Recompute AMD and rapport summaries from raw utterance features.
    Uses relaxed thresholds appropriate for MELD's shorter dialogues
    (median ~10 turns vs CGA's longer conversations).
    """
    WORD_PATTERN = re.compile(r"\b[a-z]{2,}\b")

    try:
        from nltk.corpus import stopwords
        STOPWORDS = set(stopwords.words("english"))
    except Exception:
        STOPWORDS = set()

    ge_columns = [c for c in utt_df.columns if c.startswith("ge_")]
    n_emo = len(ge_columns)

    POSITIVE_EMOTIONS = {"joy", "surprise"}
    SENTIMENT_MAP = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}

    summaries = []

    for dialog_id, group in utt_df.groupby("dialog_id"):
        group = group.sort_values("turn_idx")
        speakers = group["speaker"].unique()
        if len(speakers) < 2:
            continue

        spk1, spk2 = speakers[0], speakers[1]

        # Extract content words per speaker
        words_s1, words_s2 = [], []
        for _, row in group[group["speaker"] == spk1].iterrows():
            words = WORD_PATTERN.findall(str(row["text"]).lower())
            words_s1.extend([w for w in words if w not in STOPWORDS])
        for _, row in group[group["speaker"] == spk2].iterrows():
            words = WORD_PATTERN.findall(str(row["text"]).lower())
            words_s2.extend([w for w in words if w not in STOPWORDS])

        c1, c2 = Counter(words_s1), Counter(words_s2)
        shared = set(c1.keys()) & set(c2.keys())
        anchors = [w for w in shared if c1[w] + c2[w] >= min_anchor_freq]

        if not anchors:
            continue

        # Build anchor-speaker-context map
        anchor_map = defaultdict(lambda: defaultdict(list))
        for _, row in group.iterrows():
            text_words = set(WORD_PATTERN.findall(str(row["text"]).lower()))
            for anchor in anchors:
                if anchor in text_words:
                    anchor_map[(anchor, row["speaker"])][row["context"]].append(
                        row[ge_columns].values.astype(float)
                    )

        d_marg_vals, d_cond_vals = [], []

        for anchor in anchors:
            g1 = anchor_map.get((anchor, spk1), {})
            g2 = anchor_map.get((anchor, spk2), {})

            ed1, ed2, cnt1, cnt2 = {}, {}, {}, {}
            for ctx, dists in g1.items():
                if len(dists) >= min_per_cell:
                    ed1[ctx] = np.mean(dists, axis=0)
                    cnt1[ctx] = len(dists)
            for ctx, dists in g2.items():
                if len(dists) >= min_per_cell:
                    ed2[ctx] = np.mean(dists, axis=0)
                    cnt2[ctx] = len(dists)

            if not ed1 or not ed2:
                continue

            t1 = sum(cnt1.values())
            t2 = sum(cnt2.values())
            cw1 = {c: n / t1 for c, n in cnt1.items()}
            cw2 = {c: n / t2 for c, n in cnt2.items()}

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

        if not d_marg_vals:
            continue

        # Rapport proxy: emotion agreement
        emos_s1 = group[group["speaker"] == spk1]["emotion"].tolist()
        emos_s2 = group[group["speaker"] == spk2]["emotion"].tolist()
        min_len = min(len(emos_s1), len(emos_s2))
        if min_len > 0:
            agreements = sum(1 for i in range(min_len) if emos_s1[i] == emos_s2[i])
            emotion_agreement = agreements / min_len
        else:
            emotion_agreement = np.nan

        # Shared positivity
        pos_s1 = sum(1 for e in emos_s1 if e in POSITIVE_EMOTIONS) / max(len(emos_s1), 1)
        pos_s2 = sum(1 for e in emos_s2 if e in POSITIVE_EMOTIONS) / max(len(emos_s2), 1)
        shared_positivity = (pos_s1 + pos_s2) / 2

        # Mean valence
        valences = [SENTIMENT_MAP.get(str(row["sentiment"]).lower(), 0.0)
                     for _, row in group.iterrows()]
        mean_valence = np.mean(valences)

        summaries.append({
            "dialog_id": str(dialog_id),
            "n_turns": len(group),
            "n_anchors": len(anchors),
            "mean_d_marg": np.mean(d_marg_vals),
            "mean_d_cond": np.mean(d_cond_vals) if d_cond_vals else np.nan,
            "rapport_proxy": emotion_agreement,
            "shared_positivity": shared_positivity,
            "mean_valence": mean_valence,
        })

    summary_df = pd.DataFrame(summaries)
    print(f"  Recomputed summaries: {len(summary_df)} dialogues "
          f"(min_anchor_freq={min_anchor_freq}, min_per_cell={min_per_cell})")
    n_valid_cond = summary_df["mean_d_cond"].notna().sum()
    print(f"  With valid D_cond: {n_valid_cond}")
    return summary_df


def experiment_6_rapport_correlations(summary_df):
    """
    Experiment 6: AMD-rapport correlations for MELD.

    Rapport proxy: inter-speaker emotion agreement rate
    (proportion of turn-aligned pairs where both speakers express the
    same categorical emotion).

    Computes:
    - r1: Pearson(D_marg, rapport_proxy)
    - r2: Pearson(D_cond, rapport_proxy)
    - r3: Pearson(D_cond, shared_positivity) — alternative rapport measure
    - r4: Pearson(D_cond slope proxy, rapport)
    - Partial correlations controlling for mean valence
    - Partial correlations controlling for dialogue length
    - Partial correlations controlling for lexical overlap
    """
    print("=" * 72)
    print("EXPERIMENT 6: AMD-Rapport Correlations (MELD)")
    print("=" * 72)

    valid = summary_df.dropna(subset=["mean_d_marg", "mean_d_cond", "rapport_proxy"])

    if len(valid) < 5:
        print(f"Insufficient data ({len(valid)} dialogues) for correlation analysis.")
        return {f"r{i}": None for i in range(1, 8)} | {f"pr{i}": None for i in range(1, 5)}

    print(f"Dialogues with valid AMD + rapport: {len(valid)}")

    d_marg = valid["mean_d_marg"].values
    d_cond = valid["mean_d_cond"].values
    rapport = valid["rapport_proxy"].values
    mean_valence = valid["mean_valence"].values
    shared_pos = valid["shared_positivity"].values
    n_turns = valid["n_turns"].values.astype(float)

    # Core correlations
    r1, p_r1 = pearson_with_pvalue(d_marg, rapport)
    r2, p_r2 = pearson_with_pvalue(d_cond, rapport)

    # D_cond vs shared positivity (alternative rapport measure)
    r3, p_r3 = pearson_with_pvalue(d_cond, shared_pos)

    # AMD "slope" proxy: use D_cond itself as the AMD divergence measure
    r4, p_r4 = pearson_with_pvalue(d_cond, rapport)

    # Partial correlations: control for mean valence
    pr1, p_pr1 = partial_correlation(d_marg, rapport, mean_valence)
    pr2, p_pr2 = partial_correlation(d_cond, rapport, mean_valence)

    # Partial correlation: control for dialogue length
    pr3, p_pr3 = partial_correlation(d_cond, rapport, n_turns)

    # Partial correlation: control for both valence and length
    covariates = np.column_stack([mean_valence, n_turns])
    pr4, p_pr4 = partial_correlation(d_cond, rapport, covariates)

    # Partial correlation: D_cond vs shared_positivity | valence
    r5, p_r5 = partial_correlation(d_cond, shared_pos, mean_valence)

    # D_marg vs shared_positivity
    r6, p_r6 = pearson_with_pvalue(d_marg, shared_pos)

    # N_anchors vs D_cond
    n_anchors = valid["n_anchors"].values.astype(float)
    r7, p_r7 = pearson_with_pvalue(n_anchors, d_cond)

    results = {
        "r1": round(float(r1), 4) if not np.isnan(r1) else None,
        "p_r1": round(float(p_r1), 4) if not np.isnan(p_r1) else None,
        "r2": round(float(r2), 4) if not np.isnan(r2) else None,
        "p_r2": round(float(p_r2), 4) if not np.isnan(p_r2) else None,
        "r3": round(float(r3), 4) if not np.isnan(r3) else None,
        "p_r3": round(float(p_r3), 4) if not np.isnan(p_r3) else None,
        "r4": round(float(r4), 4) if not np.isnan(r4) else None,
        "p_r4": round(float(p_r4), 4) if not np.isnan(p_r4) else None,
        "r5": round(float(r5), 4) if not np.isnan(r5) else None,
        "p_r5": round(float(p_r5), 4) if not np.isnan(p_r5) else None,
        "r6": round(float(r6), 4) if not np.isnan(r6) else None,
        "p_r6": round(float(p_r6), 4) if not np.isnan(p_r6) else None,
        "r7": round(float(r7), 4) if not np.isnan(r7) else None,
        "p_r7": round(float(p_r7), 4) if not np.isnan(p_r7) else None,
        "pr1": round(float(pr1), 4) if not np.isnan(pr1) else None,
        "p_pr1": round(float(p_pr1), 4) if not np.isnan(p_pr1) else None,
        "pr2": round(float(pr2), 4) if not np.isnan(pr2) else None,
        "p_pr2": round(float(p_pr2), 4) if not np.isnan(p_pr2) else None,
        "pr3": round(float(pr3), 4) if not np.isnan(pr3) else None,
        "p_pr3": round(float(p_pr3), 4) if not np.isnan(p_pr3) else None,
        "pr4": round(float(pr4), 4) if not np.isnan(pr4) else None,
        "p_pr4": round(float(p_pr4), 4) if not np.isnan(p_pr4) else None,
        "n_valid_dialogues": len(valid),
    }

    print(f"\n{'Measure':<45} {'r':>8} {'p':>10}")
    print("-" * 65)
    print(f"{'r1: D_marg vs rapport':<45} {r1:>8.4f} {p_r1:>10.4f}")
    print(f"{'r2: D_cond vs rapport':<45} {r2:>8.4f} {p_r2:>10.4f}")
    print(f"{'r3: D_cond vs shared positivity':<45} {r3:>8.4f} {p_r3:>10.4f}")
    print(f"{'r4: D_cond vs rapport (=r2)':<45} {r4:>8.4f} {p_r4:>10.4f}")
    print(f"{'r5: D_cond vs shared_pos | valence':<45} {r5:>8.4f} {p_r5:>10.4f}")
    print(f"{'r6: D_marg vs shared positivity':<45} {r6:>8.4f} {p_r6:>10.4f}")
    print(f"{'r7: N_anchors vs D_cond':<45} {r7:>8.4f} {p_r7:>10.4f}")
    print(f"\n{'Partial correlations:':<45}")
    print(f"{'pr1: D_marg vs rapport | valence':<45} {pr1:>8.4f} {p_pr1:>10.4f}")
    print(f"{'pr2: D_cond vs rapport | valence':<45} {pr2:>8.4f} {p_pr2:>10.4f}")
    print(f"{'pr3: D_cond vs rapport | length':<45} {pr3:>8.4f} {p_pr3:>10.4f}")
    print(f"{'pr4: D_cond vs rapport | valence+length':<45} {pr4:>8.4f} {p_pr4:>10.4f}")
    print("=" * 72)

    return results


def experiment_7_anchor_sensitivity(utt_df, summary_df):
    """
    Experiment 7: Anchor set sensitivity analysis on MELD.
    Varies anchor frequency thresholds and word type filters,
    re-computes D_cond, and checks correlation with rapport proxy.
    """
    print("\n" + "=" * 72)
    print("EXPERIMENT 7: Anchor Set Sensitivity (MELD)")
    print("=" * 72)

    WORD_PATTERN = re.compile(r"\b[a-z]{2,}\b")

    try:
        from nltk.corpus import stopwords
        STOPWORDS = set(stopwords.words("english"))
    except Exception:
        STOPWORDS = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                     "being", "have", "has", "had", "do", "does", "did", "will",
                     "would", "could", "should", "may", "might", "shall", "can",
                     "to", "of", "in", "for", "on", "with", "at", "by", "from",
                     "as", "into", "through", "during", "before", "after", "above",
                     "below", "between", "out", "off", "over", "under", "again",
                     "further", "then", "once", "here", "there", "when", "where",
                     "why", "how", "all", "both", "each", "few", "more", "most",
                     "other", "some", "such", "no", "nor", "not", "only", "own",
                     "same", "so", "than", "too", "very", "and", "but", "or",
                     "if", "because", "until", "while", "it", "its", "i", "me",
                     "my", "we", "our", "you", "your", "he", "him", "his", "she",
                     "her", "they", "them", "their", "this", "that", "these", "those"}

    EVALUATIVE_WORDS = {
        "good", "bad", "great", "terrible", "awful", "wonderful", "excellent",
        "horrible", "nice", "nasty", "beautiful", "ugly", "love", "hate",
        "happy", "sad", "angry", "scared", "afraid", "worried", "anxious",
        "excited", "bored", "tired", "frustrated", "annoyed", "pleased",
        "satisfied", "disappointed", "disgusted", "surprised", "shocked",
        "amazing", "fantastic", "perfect", "worst", "best", "better", "worse",
        "right", "wrong", "fair", "unfair", "kind", "cruel", "sweet", "bitter",
        "funny", "serious", "stupid", "smart", "brilliant", "dumb", "crazy",
        "silly", "weird", "strange", "normal", "fine", "okay",
    }

    ge_columns = [c for c in utt_df.columns if c.startswith("ge_")]

    conditions = [
        {"label": "freq>=3, content", "min_freq": 3, "content_only": True, "evaluative_only": False},
        {"label": "freq>=5, content", "min_freq": 5, "content_only": True, "evaluative_only": False},
        {"label": "freq>=10, content", "min_freq": 10, "content_only": True, "evaluative_only": False},
        {"label": "freq>=5, evaluative", "min_freq": 5, "content_only": True, "evaluative_only": True},
        {"label": "freq>=5, all words", "min_freq": 5, "content_only": False, "evaluative_only": False},
    ]

    results = {}

    print(f"\n{'Condition':<25} {'N anchors':>10} {'Mean D_cond':>12} {'r':>8} {'p':>10}")
    print("-" * 70)

    for cond_idx, cond in enumerate(conditions):
        all_d_cond = []
        all_rapport = []
        total_anchors = 0

        for dialog_id, group in utt_df.groupby("dialog_id"):
            speakers = group["speaker"].unique()
            if len(speakers) < 2:
                continue

            spk1, spk2 = speakers[0], speakers[1]

            def get_words(text, content_only, evaluative_only):
                words = WORD_PATTERN.findall(str(text).lower())
                if content_only:
                    words = [w for w in words if w not in STOPWORDS]
                if evaluative_only:
                    words = [w for w in words if w in EVALUATIVE_WORDS]
                return words

            texts_s1 = group[group["speaker"] == spk1]["text"].tolist()
            texts_s2 = group[group["speaker"] == spk2]["text"].tolist()

            words_s1, words_s2 = [], []
            for t in texts_s1:
                words_s1.extend(get_words(t, cond["content_only"], cond["evaluative_only"]))
            for t in texts_s2:
                words_s2.extend(get_words(t, cond["content_only"], cond["evaluative_only"]))

            c1, c2 = Counter(words_s1), Counter(words_s2)
            shared = set(c1.keys()) & set(c2.keys())
            anchors = [w for w in shared if c1[w] + c2[w] >= cond["min_freq"]]
            total_anchors += len(anchors)

            if not anchors or len(ge_columns) == 0:
                continue

            anchor_map = defaultdict(lambda: defaultdict(list))
            for _, row in group.iterrows():
                text_words = set(WORD_PATTERN.findall(str(row["text"]).lower()))
                for anchor in anchors:
                    if anchor in text_words:
                        anchor_map[(anchor, row["speaker"])][row["context"]].append(
                            row[ge_columns].values.astype(float)
                        )

            # Use min_per_cell=1 for MELD (short dialogues, median ~10 turns)
            meld_min_per_cell = 1

            d_cond_vals = []
            for anchor in anchors:
                g1 = anchor_map.get((anchor, spk1), {})
                g2 = anchor_map.get((anchor, spk2), {})

                ed1, ed2, cnt1, cnt2 = {}, {}, {}, {}
                for ctx, dists in g1.items():
                    if len(dists) >= meld_min_per_cell:
                        ed1[ctx] = np.mean(dists, axis=0)
                        cnt1[ctx] = len(dists)
                for ctx, dists in g2.items():
                    if len(dists) >= meld_min_per_cell:
                        ed2[ctx] = np.mean(dists, axis=0)
                        cnt2[ctx] = len(dists)

                shared_ctx = set(ed1.keys()) & set(ed2.keys())
                if not shared_ctx:
                    continue

                wt_sum, tv_sum = 0.0, 0.0
                for c in shared_ctx:
                    w = cnt1[c] + cnt2[c]
                    tv_sum += w * total_variation(ed1[c], ed2[c])
                    wt_sum += w

                if wt_sum > 0:
                    d_cond_vals.append(tv_sum / wt_sum)

            if d_cond_vals:
                dialog_summary = summary_df[summary_df["dialog_id"] == str(dialog_id)]
                if len(dialog_summary) > 0 and not np.isnan(dialog_summary.iloc[0]["rapport_proxy"]):
                    all_d_cond.append(np.mean(d_cond_vals))
                    all_rapport.append(dialog_summary.iloc[0]["rapport_proxy"])

        n_dialogs = utt_df["dialog_id"].nunique()
        mean_anchors = total_anchors / max(n_dialogs, 1)
        mean_d_cond = np.mean(all_d_cond) if all_d_cond else np.nan

        if len(all_d_cond) >= 5:
            r_val, p_val = pearson_with_pvalue(all_d_cond, all_rapport)
        else:
            r_val, p_val = np.nan, np.nan

        results[f"s{cond_idx + 1}_label"] = cond["label"]
        results[f"s{cond_idx + 1}_mean_anchors"] = round(float(mean_anchors), 2)
        results[f"s{cond_idx + 1}_mean_d_cond"] = round(float(mean_d_cond), 4) if not np.isnan(mean_d_cond) else None
        results[f"s{cond_idx + 1}_r"] = round(float(r_val), 4) if not np.isnan(r_val) else None
        results[f"s{cond_idx + 1}_p"] = round(float(p_val), 4) if not np.isnan(p_val) else None
        results[f"s{cond_idx + 1}_n"] = len(all_d_cond)

        r_str = f"{r_val:.4f}" if not np.isnan(r_val) else "N/A"
        p_str = f"{p_val:.4f}" if not np.isnan(p_val) else "N/A"

        print(f"{cond['label']:<25} {mean_anchors:>10.1f} {mean_d_cond:>12.4f} {r_str:>8} {p_str:>10}")

    print("=" * 72)
    return results


def experiment_6b_long_dialogues(summary_df, utt_df, min_turns=15):
    """
    Experiment 6b: AMD-rapport on longer MELD dialogues only.
    Filters to dialogues with >= min_turns to increase anchor coverage
    and statistical power.
    """
    print("\n" + "=" * 72)
    print(f"EXPERIMENT 6b: AMD-Rapport (dialogues >= {min_turns} turns)")
    print("=" * 72)

    long_df = summary_df[summary_df["n_turns"] >= min_turns].copy()
    valid = long_df.dropna(subset=["mean_d_marg", "mean_d_cond", "rapport_proxy"])

    if len(valid) < 5:
        print(f"Insufficient data ({len(valid)} dialogues).")
        return {"r2_long": None, "p_r2_long": None, "n_long": len(valid)}

    print(f"Long dialogues with valid AMD + rapport: {len(valid)}")
    print(f"  Mean turns: {valid['n_turns'].mean():.1f}")
    print(f"  Mean anchors: {valid['n_anchors'].mean():.1f}")

    d_marg = valid["mean_d_marg"].values
    d_cond = valid["mean_d_cond"].values
    rapport = valid["rapport_proxy"].values
    mean_valence = valid["mean_valence"].values
    n_turns = valid["n_turns"].values.astype(float)

    r1_l, p_r1_l = pearson_with_pvalue(d_marg, rapport)
    r2_l, p_r2_l = pearson_with_pvalue(d_cond, rapport)
    pr2_l, p_pr2_l = partial_correlation(d_cond, rapport, mean_valence)
    pr3_l, p_pr3_l = partial_correlation(d_cond, rapport, n_turns)

    covariates = np.column_stack([mean_valence, n_turns])
    pr4_l, p_pr4_l = partial_correlation(d_cond, rapport, covariates)

    results = {
        "r1_long": round(float(r1_l), 4) if not np.isnan(r1_l) else None,
        "p_r1_long": round(float(p_r1_l), 4) if not np.isnan(p_r1_l) else None,
        "r2_long": round(float(r2_l), 4) if not np.isnan(r2_l) else None,
        "p_r2_long": round(float(p_r2_l), 4) if not np.isnan(p_r2_l) else None,
        "pr2_long": round(float(pr2_l), 4) if not np.isnan(pr2_l) else None,
        "p_pr2_long": round(float(p_pr2_l), 4) if not np.isnan(p_pr2_l) else None,
        "pr3_long": round(float(pr3_l), 4) if not np.isnan(pr3_l) else None,
        "p_pr3_long": round(float(p_pr3_l), 4) if not np.isnan(p_pr3_l) else None,
        "pr4_long": round(float(pr4_l), 4) if not np.isnan(pr4_l) else None,
        "p_pr4_long": round(float(p_pr4_l), 4) if not np.isnan(p_pr4_l) else None,
        "n_long": len(valid),
    }

    print(f"\n{'Measure':<45} {'r':>8} {'p':>10}")
    print("-" * 65)
    print(f"{'r1: D_marg vs rapport':<45} {r1_l:>8.4f} {p_r1_l:>10.4f}")
    print(f"{'r2: D_cond vs rapport':<45} {r2_l:>8.4f} {p_r2_l:>10.4f}")
    print(f"{'pr2: D_cond vs rapport | valence':<45} {pr2_l:>8.4f} {p_pr2_l:>10.4f}")
    print(f"{'pr3: D_cond vs rapport | length':<45} {pr3_l:>8.4f} {p_pr3_l:>10.4f}")
    print(f"{'pr4: D_cond vs rapport | val+len':<45} {pr4_l:>8.4f} {p_pr4_l:>10.4f}")
    print("=" * 72)

    return results


def experiment_6c_cosine_rapport(summary_df, utt_df):
    """
    Experiment 6c: GoEmotions cosine similarity as continuous rapport proxy.

    Instead of categorical emotion agreement (0/1 per turn pair),
    compute mean cosine similarity between consecutive inter-speaker
    GoEmotions vectors. This gives a continuous, graded measure of
    emotional alignment.
    """
    print("\n" + "=" * 72)
    print("EXPERIMENT 6c: AMD vs GoEmotions Cosine Rapport")
    print("=" * 72)

    ge_columns = [c for c in utt_df.columns if c.startswith("ge_")]

    cosine_rapports = []

    for dialog_id, group in utt_df.groupby("dialog_id"):
        group = group.sort_values("turn_idx")
        speakers = group["speaker"].unique()
        if len(speakers) < 2:
            continue

        spk1, spk2 = speakers[0], speakers[1]
        vecs_s1 = group[group["speaker"] == spk1][ge_columns].values
        vecs_s2 = group[group["speaker"] == spk2][ge_columns].values

        min_len = min(len(vecs_s1), len(vecs_s2))
        if min_len < 1:
            continue

        # Cosine similarity between aligned turn pairs
        cosines = []
        for i in range(min_len):
            v1 = vecs_s1[i]
            v2 = vecs_s2[i]
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 > 1e-10 and norm2 > 1e-10:
                cosines.append(np.dot(v1, v2) / (norm1 * norm2))

        if cosines:
            cosine_rapports.append({
                "dialog_id": str(dialog_id),
                "cosine_rapport": np.mean(cosines),
            })

    cosine_df = pd.DataFrame(cosine_rapports)
    merged = summary_df.merge(cosine_df, on="dialog_id", how="inner")
    valid = merged.dropna(subset=["mean_d_marg", "mean_d_cond", "cosine_rapport"])

    if len(valid) < 5:
        print(f"Insufficient data ({len(valid)} dialogues).")
        return {"r_cos": None, "p_cos": None, "n_cosine": len(valid)}

    print(f"Dialogues with valid D_cond + cosine rapport: {len(valid)}")
    print(f"  Cosine rapport: mean={valid['cosine_rapport'].mean():.4f}, "
          f"std={valid['cosine_rapport'].std():.4f}")

    d_marg = valid["mean_d_marg"].values
    d_cond = valid["mean_d_cond"].values
    cos_rap = valid["cosine_rapport"].values
    mean_valence = valid["mean_valence"].values
    n_turns = valid["n_turns"].values.astype(float)

    # Core: D_cond vs cosine rapport
    r_cos, p_cos = pearson_with_pvalue(d_cond, cos_rap)
    r_cos_m, p_cos_m = pearson_with_pvalue(d_marg, cos_rap)

    # Partial: control for valence
    pr_cos_v, p_pr_cos_v = partial_correlation(d_cond, cos_rap, mean_valence)

    # Partial: control for length
    pr_cos_l, p_pr_cos_l = partial_correlation(d_cond, cos_rap, n_turns)

    # Partial: control for both
    covariates = np.column_stack([mean_valence, n_turns])
    pr_cos_vl, p_pr_cos_vl = partial_correlation(d_cond, cos_rap, covariates)

    # Also test on long dialogues only
    long = valid[valid["n_turns"] >= 15]
    if len(long) >= 5:
        r_cos_long, p_cos_long = pearson_with_pvalue(
            long["mean_d_cond"].values, long["cosine_rapport"].values
        )
    else:
        r_cos_long, p_cos_long = np.nan, np.nan

    # Compare with original emotion-agreement rapport
    if "rapport_proxy" in valid.columns:
        r_orig, p_orig = pearson_with_pvalue(d_cond, valid["rapport_proxy"].values)
    else:
        r_orig, p_orig = np.nan, np.nan

    results = {
        "r_cos_marg": round(float(r_cos_m), 4) if not np.isnan(r_cos_m) else None,
        "p_cos_marg": round(float(p_cos_m), 4) if not np.isnan(p_cos_m) else None,
        "r_cos": round(float(r_cos), 4) if not np.isnan(r_cos) else None,
        "p_cos": round(float(p_cos), 4) if not np.isnan(p_cos) else None,
        "pr_cos_valence": round(float(pr_cos_v), 4) if not np.isnan(pr_cos_v) else None,
        "p_pr_cos_valence": round(float(p_pr_cos_v), 4) if not np.isnan(p_pr_cos_v) else None,
        "pr_cos_length": round(float(pr_cos_l), 4) if not np.isnan(pr_cos_l) else None,
        "p_pr_cos_length": round(float(p_pr_cos_l), 4) if not np.isnan(p_pr_cos_l) else None,
        "pr_cos_val_len": round(float(pr_cos_vl), 4) if not np.isnan(pr_cos_vl) else None,
        "p_pr_cos_val_len": round(float(p_pr_cos_vl), 4) if not np.isnan(p_pr_cos_vl) else None,
        "r_cos_long": round(float(r_cos_long), 4) if not np.isnan(r_cos_long) else None,
        "p_cos_long": round(float(p_cos_long), 4) if not np.isnan(p_cos_long) else None,
        "n_cos_long": int(len(long)),
        "n_cosine": len(valid),
        "r_orig_emag": round(float(r_orig), 4) if not np.isnan(r_orig) else None,
        "p_orig_emag": round(float(p_orig), 4) if not np.isnan(p_orig) else None,
    }

    print(f"\n{'Measure':<50} {'r':>8} {'p':>10}")
    print("-" * 70)
    print(f"{'D_marg vs cosine rapport':<50} {r_cos_m:>8.4f} {p_cos_m:>10.4f}")
    print(f"{'D_cond vs cosine rapport':<50} {r_cos:>8.4f} {p_cos:>10.4f}")
    print(f"{'D_cond vs cosine rapport | valence':<50} {pr_cos_v:>8.4f} {p_pr_cos_v:>10.4f}")
    print(f"{'D_cond vs cosine rapport | length':<50} {pr_cos_l:>8.4f} {p_pr_cos_l:>10.4f}")
    print(f"{'D_cond vs cosine rapport | val+len':<50} {pr_cos_vl:>8.4f} {p_pr_cos_vl:>10.4f}")
    if not np.isnan(r_cos_long):
        print(f"{'D_cond vs cosine rapport (>=15 turns)':<50} {r_cos_long:>8.4f} {p_cos_long:>10.4f}")
    print(f"\n{'Comparison: D_cond vs emotion agreement':<50} {r_orig:>8.4f} {p_orig:>10.4f}")
    print("=" * 72)

    return results


def experiment_6d_independent_rapport(summary_df, utt_df):
    """
    Experiment 6d: AMD vs MELD ground-truth rapport proxies.

    These proxies use only MELD's human-annotated emotion/sentiment labels —
    completely independent of the GoEmotions model used to compute AMD.
    This avoids the shared-method-variance concern in Exp 6c.

    Proxies:
    - Sentiment agreement: proportion of turn-aligned pairs with matching sentiment
    - Emotion TV distance: total variation between per-speaker emotion distributions
    - Negative emotion gap: |neg_proportion_s1 - neg_proportion_s2|
    """
    print("\n" + "=" * 72)
    print("EXPERIMENT 6d: AMD vs Independent MELD Ground-Truth Rapport")
    print("=" * 72)

    MELD_EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
    NEG_EMOS = {"anger", "disgust", "fear", "sadness"}
    SENT_ORD = {"positive": 1, "neutral": 0, "negative": -1}

    rap_records = []

    for dialog_id, group in utt_df.groupby("dialog_id"):
        group = group.sort_values("turn_idx")
        speakers = group["speaker"].unique()
        if len(speakers) < 2:
            continue
        spk1, spk2 = speakers[0], speakers[1]

        emos_s1 = group[group["speaker"] == spk1]["emotion"].tolist()
        emos_s2 = group[group["speaker"] == spk2]["emotion"].tolist()
        sents_s1 = group[group["speaker"] == spk1]["sentiment"].tolist()
        sents_s2 = group[group["speaker"] == spk2]["sentiment"].tolist()
        min_e = min(len(emos_s1), len(emos_s2))
        min_s = min(len(sents_s1), len(sents_s2))

        if min_e < 1:
            continue

        # Sentiment agreement rate
        sent_agree = sum(1 for i in range(min_s) if sents_s1[i] == sents_s2[i]) / max(min_s, 1)

        # Emotion distribution TV distance
        def emo_dist(emos):
            c = Counter(emos)
            total = sum(c.values())
            return np.array([c.get(e, 0) / total for e in MELD_EMOTIONS])
        d1 = emo_dist(emos_s1)
        d2 = emo_dist(emos_s2)
        emo_tv = 0.5 * np.sum(np.abs(d1 - d2))

        # Negative emotion proportion gap
        neg_s1 = sum(1 for e in emos_s1 if e in NEG_EMOS) / max(len(emos_s1), 1)
        neg_s2 = sum(1 for e in emos_s2 if e in NEG_EMOS) / max(len(emos_s2), 1)
        neg_gap = abs(neg_s1 - neg_s2)

        # Sentiment ordinal correlation
        if min_s >= 3:
            s1_vals = [SENT_ORD.get(s, 0) for s in sents_s1[:min_s]]
            s2_vals = [SENT_ORD.get(s, 0) for s in sents_s2[:min_s]]
            if np.std(s1_vals) > 0 and np.std(s2_vals) > 0:
                sent_corr = np.corrcoef(s1_vals, s2_vals)[0, 1]
            else:
                sent_corr = np.nan
        else:
            sent_corr = np.nan

        rap_records.append({
            "dialog_id": str(dialog_id),
            "sent_agreement": sent_agree,
            "emo_tv": emo_tv,
            "neg_gap": neg_gap,
            "sent_corr": sent_corr,
        })

    rap_df = pd.DataFrame(rap_records)
    merged = summary_df.merge(rap_df, on="dialog_id", how="inner")
    valid = merged.dropna(subset=["mean_d_cond"])

    print(f"Dialogues with valid D_cond + independent rapport: {len(valid)}")

    d_marg = valid["mean_d_marg"].values
    d_cond = valid["mean_d_cond"].values
    mean_valence = valid["mean_valence"].values
    n_turns = valid["n_turns"].values.astype(float)

    results = {}

    proxies = [
        ("sent_agreement", "Sentiment agreement"),
        ("emo_tv", "Emotion TV distance (GT)"),
        ("neg_gap", "Negative emotion gap"),
        ("sent_corr", "Sentiment ordinal corr"),
    ]

    print(f"\n{'Measure':<55} {'r':>8} {'p':>10} {'n':>6}")
    print("-" * 80)

    for col, label in proxies:
        vals = valid[col].values
        mask = ~np.isnan(vals)
        n = int(mask.sum())
        if n < 10:
            print(f"D_cond vs {label:<45} {'N/A':>8} {'N/A':>10} {n:>6}")
            results[f"r_{col}"] = None
            results[f"p_{col}"] = None
            continue

        r, p = pearson_with_pvalue(d_cond[mask], vals[mask])
        r_m, p_m = pearson_with_pvalue(d_marg[mask], vals[mask])

        # Partial: control for valence + length
        cov = np.column_stack([mean_valence[mask], n_turns[mask]])
        pr, pp = partial_correlation(d_cond[mask], vals[mask], cov)

        results[f"r_cond_{col}"] = round(float(r), 4) if not np.isnan(r) else None
        results[f"p_cond_{col}"] = round(float(p), 4) if not np.isnan(p) else None
        results[f"r_marg_{col}"] = round(float(r_m), 4) if not np.isnan(r_m) else None
        results[f"p_marg_{col}"] = round(float(p_m), 4) if not np.isnan(p_m) else None
        results[f"pr_cond_{col}"] = round(float(pr), 4) if not np.isnan(pr) else None
        results[f"pp_cond_{col}"] = round(float(pp), 4) if not np.isnan(pp) else None
        results[f"n_{col}"] = n

        print(f"D_cond vs {label:<45} {r:>8.4f} {p:>10.4f} {n:>6}")
        print(f"D_marg vs {label:<45} {r_m:>8.4f} {p_m:>10.4f}")
        print(f"D_cond vs {label[:33]+'|val+len':<45} {pr:>8.4f} {pp:>10.4f}")

    results["n_independent"] = len(valid)
    print("=" * 72)

    return results


def main():
    """Run MELD analysis experiments and save results."""
    np.random.seed(RANDOM_SEED)

    utt_df = load_meld_features()
    print(f"Loaded {len(utt_df)} utterances from {utt_df['dialog_id'].nunique()} dialogues\n")

    # Recompute AMD summaries with relaxed thresholds for short MELD dialogues
    summary_df = recompute_meld_summaries(utt_df, min_anchor_freq=2, min_per_cell=1)

    n_valid = summary_df["mean_d_cond"].notna().sum()
    print(f"Dialogues with valid D_cond + rapport: {n_valid}\n")

    exp6_results = experiment_6_rapport_correlations(summary_df)

    # ── Exp 6b: Filter to longer dialogues (≥ 15 turns) ──
    exp6b_results = experiment_6b_long_dialogues(summary_df, utt_df, min_turns=15)

    # ── Exp 6c: GoEmotions cosine rapport proxy ──
    # NOTE: shares method variance with AMD (both use GoEmotions).
    # Included for completeness but Exp 6d is the clean test.
    exp6c_results = experiment_6c_cosine_rapport(summary_df, utt_df)

    # ── Exp 6d: Independent ground-truth rapport (no GoEmotions) ──
    exp6d_results = experiment_6d_independent_rapport(summary_df, utt_df)

    exp7_results = experiment_7_anchor_sensitivity(utt_df, summary_df)

    all_results = {}
    all_results.update(exp6_results)
    all_results.update(exp6b_results)
    all_results.update(exp6c_results)
    all_results.update(exp6d_results)
    all_results.update(exp7_results)

    output_path = RESULTS_DIR / "meld_analysis.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nAll MELD results saved to {output_path}")


if __name__ == "__main__":
    main()
