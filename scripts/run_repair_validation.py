"""
Appendix: Repair proxy validation.
Loads DA classifier test results, computes inter-proxy correlations,
and reports repair marker coverage on CGA data.
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
    CGA_FEATURES_PATH,
    DA_MODEL_SAVE_PATH,
)
from utils.statistical_tests import pearson_with_pvalue


def load_da_classifier_results():
    """Load DA classifier test accuracy from saved results."""
    results_path = RESULTS_DIR / "da_classifier_results.json"
    if results_path.exists():
        with open(results_path) as f:
            return json.load(f)

    metadata_path = DA_MODEL_SAVE_PATH / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            meta = json.load(f)
            return {"sw_acc": meta.get("sw_acc", None)}

    return {"sw_acc": None}


def compute_inter_proxy_correlations(utt_df):
    """Compute pairwise correlations between the three repair proxies."""
    q_da = utt_df["q_da"].values.astype(float)
    q_rm = utt_df["q_rm"].values.astype(float)
    q_ce = utt_df["q_ce"].values.astype(float)

    valid_da_rm = ~(np.isnan(q_da) | np.isnan(q_rm))
    valid_da_ce = ~(np.isnan(q_da) | np.isnan(q_ce))
    valid_rm_ce = ~(np.isnan(q_rm) | np.isnan(q_ce))

    if valid_da_rm.sum() >= 3:
        corr_da_rm, p_da_rm = stats.pearsonr(q_da[valid_da_rm], q_rm[valid_da_rm])
    else:
        corr_da_rm, p_da_rm = np.nan, np.nan

    if valid_da_ce.sum() >= 3:
        corr_da_ce, p_da_ce = stats.pearsonr(q_da[valid_da_ce], q_ce[valid_da_ce])
    else:
        corr_da_ce, p_da_ce = np.nan, np.nan

    if valid_rm_ce.sum() >= 3:
        corr_rm_ce, p_rm_ce = stats.pearsonr(q_rm[valid_rm_ce], q_ce[valid_rm_ce])
    else:
        corr_rm_ce, p_rm_ce = np.nan, np.nan

    return {
        "corr_da_rm": corr_da_rm,
        "p_da_rm": p_da_rm,
        "corr_da_ce": corr_da_ce,
        "p_da_ce": p_da_ce,
        "corr_rm_ce": corr_rm_ce,
        "p_rm_ce": p_rm_ce,
    }


def compute_repair_marker_coverage(utt_df):
    """Compute percentage of conversations containing at least one repair marker."""
    conversations_with_repair = 0
    total_conversations = 0

    for convo_id, group in utt_df.groupby("convo_id"):
        total_conversations += 1
        if (group["q_rm"] > 0).any():
            conversations_with_repair += 1

    if total_conversations > 0:
        return conversations_with_repair / total_conversations
    return 0.0


def main():
    """Run repair proxy validation and save results."""
    np.random.seed(RANDOM_SEED)

    print("=" * 72)
    print("APPENDIX: Repair Proxy Validation")
    print("=" * 72)

    da_results = load_da_classifier_results()
    sw_acc = da_results.get("sw_acc", None)
    print(f"\nDA Classifier Test Accuracy (sw_acc): {sw_acc}")

    print("\nManual annotation validation:")
    print("  NOTE: This step requires manual annotation of 200 CGA utterances.")
    print("  Placeholder values are used below. Replace with actual annotations.")
    kappa = None
    da_repair_corr = None

    if CGA_FEATURES_PATH.exists():
        utt_df = pd.read_parquet(CGA_FEATURES_PATH)

        # Recompute q_da from da_prob_* columns
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
            print(f"  Recomputed q_da from {conciliatory_cols}")
        print(f"\nLoaded {len(utt_df)} utterances from CGA features")

        proxy_corrs = compute_inter_proxy_correlations(utt_df)
        rm_coverage = compute_repair_marker_coverage(utt_df)

        print(f"\nInter-proxy correlations:")
        print(f"  corr(q_DA, q_RM): {proxy_corrs['corr_da_rm']:.4f} (p={proxy_corrs['p_da_rm']:.4f})")
        print(f"  corr(q_DA, q_CE): {proxy_corrs['corr_da_ce']:.4f} (p={proxy_corrs['p_da_ce']:.4f})")
        print(f"  corr(q_RM, q_CE): {proxy_corrs['corr_rm_ce']:.4f} (p={proxy_corrs['p_rm_ce']:.4f})")
        print(f"\nRepair marker coverage: {rm_coverage:.4f}")
    else:
        print(f"\nCGA features not found at {CGA_FEATURES_PATH}")
        print("Run notebook 01 first to generate features.")
        proxy_corrs = {
            "corr_da_rm": None, "p_da_rm": None,
            "corr_da_ce": None, "p_da_ce": None,
            "corr_rm_ce": None, "p_rm_ce": None,
        }
        rm_coverage = None

    def safe_round(val, digits=4):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        return round(float(val), digits)

    results = {
        "sw_acc": safe_round(sw_acc),
        "kappa": safe_round(kappa),
        "da_repair_corr": safe_round(da_repair_corr),
        "corr_da_rm": safe_round(proxy_corrs.get("corr_da_rm")),
        "p_corr_da_rm": safe_round(proxy_corrs.get("p_da_rm")),
        "corr_da_ce": safe_round(proxy_corrs.get("corr_da_ce")),
        "p_corr_da_ce": safe_round(proxy_corrs.get("p_da_ce")),
        "corr_rm_ce": safe_round(proxy_corrs.get("corr_rm_ce")),
        "p_corr_rm_ce": safe_round(proxy_corrs.get("p_rm_ce")),
        "rm_coverage": safe_round(rm_coverage),
    }

    output_path = RESULTS_DIR / "repair_validation.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'Metric':<30} {'Value':>10}")
    print("-" * 42)
    for key, val in results.items():
        display_val = f"{val:.4f}" if val is not None else "PENDING"
        print(f"{key:<30} {display_val:>10}")

    print("=" * 72)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
