"""
Synthetic experiments for bifurcation validation, confound stress test,
and estimator balancing (Experiments 1, 2, and 3).
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import expit, logit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    RESULTS_DIR,
    RANDOM_SEED,
    BIFURCATION_ALPHA_BETA_PAIRS,
    BIFURCATION_KAPPA_RANGE,
    BIFURCATION_KAPPA_STEP,
    BIFURCATION_T_ITER,
    BOOTSTRAP_REPS,
    BOOTSTRAP_N_SAMPLES,
)
from utils.amd import total_variation


def logit_best_response(q, alpha, beta, kappa):
    """Compute one step of the logit best-response map."""
    return expit(beta * (alpha * q - kappa))


def compute_bifurcation_thresholds(alpha, beta):
    """
    Compute closed-form bifurcation thresholds kappa_minus and kappa_plus.

    Returns None for each threshold if the discriminant is negative.
    """
    discriminant = 1.0 - 4.0 / (beta * alpha)

    if discriminant < 0:
        return None, None

    sqrt_disc = np.sqrt(discriminant)
    q_minus = 0.5 * (1.0 - sqrt_disc)
    q_plus = 0.5 * (1.0 + sqrt_disc)

    kappa_minus = alpha * q_minus - (1.0 / beta) * logit(q_minus)
    kappa_plus = alpha * q_plus - (1.0 / beta) * logit(q_plus)

    return float(kappa_minus), float(kappa_plus)


def run_bifurcation_sweep(alpha, beta, kappa_values, t_iter):
    """
    Run forward and backward bifurcation sweeps for given parameters.

    Returns arrays of equilibrium q values for forward (q0=0.001) and backward (q0=0.999).
    """
    forward_eq = np.zeros(len(kappa_values))
    backward_eq = np.zeros(len(kappa_values))

    for idx, kappa in enumerate(kappa_values):
        q_fwd = 0.001
        for _ in range(t_iter):
            q_fwd = logit_best_response(q_fwd, alpha, beta, kappa)
        forward_eq[idx] = q_fwd

        q_bwd = 0.999
        for _ in range(t_iter):
            q_bwd = logit_best_response(q_bwd, alpha, beta, kappa)
        backward_eq[idx] = q_bwd

    return forward_eq, backward_eq


def experiment_1_bifurcation():
    """
    Experiment 1: Bifurcation validation.
    Computes thresholds and sweep data for all (alpha, beta) pairs.
    Saves bifurcation diagram as PDF and results as JSON.
    """
    kappa_values = np.arange(
        BIFURCATION_KAPPA_RANGE[0],
        BIFURCATION_KAPPA_RANGE[1] + BIFURCATION_KAPPA_STEP,
        BIFURCATION_KAPPA_STEP,
    )

    results = {}
    fig, axes = plt.subplots(1, len(BIFURCATION_ALPHA_BETA_PAIRS),
                             figsize=(5 * len(BIFURCATION_ALPHA_BETA_PAIRS), 4),
                             sharey=True)

    if len(BIFURCATION_ALPHA_BETA_PAIRS) == 1:
        axes = [axes]

    print("=" * 72)
    print("EXPERIMENT 1: Bifurcation Validation (Table 1)")
    print("=" * 72)
    print(f"{'alpha':>6} {'beta':>6} {'kappa_minus':>14} {'kappa_plus':>14} {'regime':>10}")
    print("-" * 72)

    for panel_idx, (alpha, beta) in enumerate(BIFURCATION_ALPHA_BETA_PAIRS):
        kappa_minus, kappa_plus = compute_bifurcation_thresholds(alpha, beta)

        param_key = f"a{alpha}_b{beta}"

        if kappa_minus is not None:
            regime = "bistable"
            results[f"{param_key}_kappa_minus"] = round(kappa_minus, 6)
            results[f"{param_key}_kappa_plus"] = round(kappa_plus, 6)
            print(f"{alpha:>6} {beta:>6} {kappa_minus:>14.6f} {kappa_plus:>14.6f} {regime:>10}")
        else:
            regime = "monostable"
            results[f"{param_key}_kappa_minus"] = None
            results[f"{param_key}_kappa_plus"] = None
            print(f"{alpha:>6} {beta:>6} {'N/A':>14} {'N/A':>14} {regime:>10}")

        results[f"{param_key}_regime"] = regime

        forward_eq, backward_eq = run_bifurcation_sweep(
            alpha, beta, kappa_values, BIFURCATION_T_ITER
        )

        results[f"{param_key}_forward_eq"] = forward_eq.tolist()
        results[f"{param_key}_backward_eq"] = backward_eq.tolist()

        ax = axes[panel_idx]
        ax.plot(kappa_values, forward_eq, color="#2166ac", linewidth=0.5, label="Forward")
        ax.plot(kappa_values, backward_eq, color="#b2182b", linewidth=0.5, label="Backward")

        if kappa_minus is not None:
            ax.axvline(kappa_minus, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
            ax.axvline(kappa_plus, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

        ax.set_xlabel(r"$\kappa$")
        if panel_idx == 0:
            ax.set_ylabel(r"Equilibrium $q^*$")
        ax.set_title(rf"$\alpha={alpha},\ \beta={beta}$")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8)

    print("=" * 72)

    plt.tight_layout()
    figure_path = RESULTS_DIR / "bifurcation_diagram.pdf"
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nBifurcation diagram saved to {figure_path}")

    results["kappa_values"] = kappa_values.tolist()

    return results


def experiment_2_confound_stress():
    """
    Experiment 2: Confound stress test.
    Demonstrates that marginal AMD captures usage confound while conditional AMD does not.
    """
    print("\n" + "=" * 72)
    print("EXPERIMENT 2: Confound Stress Test")
    print("=" * 72)

    p_s1_given_A = np.array([0.1, 0.9])
    p_s1_given_B = np.array([0.9, 0.1])

    p_A_speaker1 = 0.9
    p_B_speaker1 = 0.1
    p_A_speaker2 = 0.1
    p_B_speaker2 = 0.9

    marginal_speaker1 = p_A_speaker1 * p_s1_given_A + p_B_speaker1 * p_s1_given_B
    marginal_speaker2 = p_A_speaker2 * p_s1_given_A + p_B_speaker2 * p_s1_given_B

    d_marg = total_variation(marginal_speaker1, marginal_speaker2)

    total_weight_A = 1.0
    total_weight_B = 1.0
    tv_A = total_variation(p_s1_given_A, p_s1_given_A)
    tv_B = total_variation(p_s1_given_B, p_s1_given_B)
    d_cond = (total_weight_A * tv_A + total_weight_B * tv_B) / (total_weight_A + total_weight_B)

    print(f"  D_marg  = {d_marg:.4f}  (expected: 0.6400)")
    print(f"  D_cond  = {d_cond:.4f}  (expected: 0.0000)")
    print("=" * 72)

    return {
        "exp2_D_marg": round(float(d_marg), 4),
        "exp2_D_cond": round(float(d_cond), 4),
    }


def experiment_3_estimator_balancing():
    """
    Experiment 3: Estimator with balancing via bootstrap.
    Tests two conditions: pure usage confound (C1) and true conditional AMD (C2).
    """
    print("\n" + "=" * 72)
    print("EXPERIMENT 3: Estimator with Balancing")
    print("=" * 72)

    rng = np.random.RandomState(RANDOM_SEED)
    n = BOOTSTRAP_N_SAMPLES

    def run_condition_c1(rng_state):
        """
        Condition C1: Pure usage confound.
        Both speakers have same P(s|x,c) but different P(c|x).
        Speaker 1: P(A|x)=0.725, Speaker 2: P(A|x)=0.275.
        Both: P(s=1|x,A)=0.9, P(s=1|x,B)=0.1.
        Target: D_marg=0.36 (from usage confound), D_cond approx 0.
        """
        contexts_s1 = rng_state.choice(["A", "B"], size=n, p=[0.725, 0.275])
        contexts_s2 = rng_state.choice(["A", "B"], size=n, p=[0.275, 0.725])

        def sample_emotion(context):
            if context == "A":
                return rng_state.choice([0, 1], p=[0.1, 0.9])
            return rng_state.choice([0, 1], p=[0.9, 0.1])

        emotions_s1 = np.array([sample_emotion(c) for c in contexts_s1])
        emotions_s2 = np.array([sample_emotion(c) for c in contexts_s2])

        def estimate_marginal(contexts, emotions):
            dist = np.zeros(2)
            for ctx_label in ["A", "B"]:
                mask = contexts == ctx_label
                if mask.sum() > 0:
                    ctx_weight = mask.sum() / len(contexts)
                    ctx_dist = np.array([
                        (emotions[mask] == 0).mean(),
                        (emotions[mask] == 1).mean(),
                    ])
                    dist += ctx_weight * ctx_dist
            return dist

        m1 = estimate_marginal(contexts_s1, emotions_s1)
        m2 = estimate_marginal(contexts_s2, emotions_s2)
        d_marg = total_variation(m1, m2)

        tv_sum = 0.0
        weight_sum = 0.0
        for ctx_label in ["A", "B"]:
            mask1 = contexts_s1 == ctx_label
            mask2 = contexts_s2 == ctx_label
            if mask1.sum() >= 3 and mask2.sum() >= 3:
                dist1 = np.array([
                    (emotions_s1[mask1] == 0).mean(),
                    (emotions_s1[mask1] == 1).mean(),
                ])
                dist2 = np.array([
                    (emotions_s2[mask2] == 0).mean(),
                    (emotions_s2[mask2] == 1).mean(),
                ])
                w = mask1.sum() + mask2.sum()
                tv_sum += w * total_variation(dist1, dist2)
                weight_sum += w

        d_cond = tv_sum / weight_sum if weight_sum > 0 else 0.0

        return d_marg, d_cond

    def run_condition_c2(rng_state):
        """
        Condition C2: True conditional AMD with symmetric marginals.
        Both speakers: P(A|x)=0.5, P(B|x)=0.5.
        Speaker 1: P(s=1|x,A)=0.9, P(s=1|x,B)=0.1.
        Speaker 2: P(s=1|x,A)=0.3, P(s=1|x,B)=0.7.
        Target: D_marg approx 0, D_cond=0.60.
        """
        contexts_s1 = rng_state.choice(["A", "B"], size=n, p=[0.5, 0.5])
        contexts_s2 = rng_state.choice(["A", "B"], size=n, p=[0.5, 0.5])

        def sample_s1(context):
            if context == "A":
                return rng_state.choice([0, 1], p=[0.1, 0.9])
            return rng_state.choice([0, 1], p=[0.9, 0.1])

        def sample_s2(context):
            if context == "A":
                return rng_state.choice([0, 1], p=[0.7, 0.3])
            return rng_state.choice([0, 1], p=[0.3, 0.7])

        emotions_s1 = np.array([sample_s1(c) for c in contexts_s1])
        emotions_s2 = np.array([sample_s2(c) for c in contexts_s2])

        def estimate_marginal(contexts, emotions):
            dist = np.zeros(2)
            for ctx_label in ["A", "B"]:
                mask = contexts == ctx_label
                if mask.sum() > 0:
                    ctx_weight = mask.sum() / len(contexts)
                    ctx_dist = np.array([
                        (emotions[mask] == 0).mean(),
                        (emotions[mask] == 1).mean(),
                    ])
                    dist += ctx_weight * ctx_dist
            return dist

        m1 = estimate_marginal(contexts_s1, emotions_s1)
        m2 = estimate_marginal(contexts_s2, emotions_s2)
        d_marg = total_variation(m1, m2)

        tv_sum = 0.0
        weight_sum = 0.0
        for ctx_label in ["A", "B"]:
            mask1 = contexts_s1 == ctx_label
            mask2 = contexts_s2 == ctx_label
            if mask1.sum() >= 3 and mask2.sum() >= 3:
                dist1 = np.array([
                    (emotions_s1[mask1] == 0).mean(),
                    (emotions_s1[mask1] == 1).mean(),
                ])
                dist2 = np.array([
                    (emotions_s2[mask2] == 0).mean(),
                    (emotions_s2[mask2] == 1).mean(),
                ])
                w = mask1.sum() + mask2.sum()
                tv_sum += w * total_variation(dist1, dist2)
                weight_sum += w

        d_cond = tv_sum / weight_sum if weight_sum > 0 else 0.0

        return d_marg, d_cond

    c1_marg_values = []
    c1_cond_values = []
    c2_marg_values = []
    c2_cond_values = []

    for rep in range(BOOTSTRAP_REPS):
        seed_offset = rep * 2
        rng_c1 = np.random.RandomState(RANDOM_SEED + seed_offset)
        rng_c2 = np.random.RandomState(RANDOM_SEED + seed_offset + 1)

        dm1, dc1 = run_condition_c1(rng_c1)
        dm2, dc2 = run_condition_c2(rng_c2)

        c1_marg_values.append(dm1)
        c1_cond_values.append(dc1)
        c2_marg_values.append(dm2)
        c2_cond_values.append(dc2)

    results = {
        "c1_d_marg_mean": round(float(np.mean(c1_marg_values)), 4),
        "c1_d_marg_std": round(float(np.std(c1_marg_values)), 4),
        "c1_d_cond_mean": round(float(np.mean(c1_cond_values)), 4),
        "c1_d_cond_std": round(float(np.std(c1_cond_values)), 4),
        "c2_d_marg_mean": round(float(np.mean(c2_marg_values)), 4),
        "c2_d_marg_std": round(float(np.std(c2_marg_values)), 4),
        "c2_d_cond_mean": round(float(np.mean(c2_cond_values)), 4),
        "c2_d_cond_std": round(float(np.std(c2_cond_values)), 4),
    }

    print(f"\n  Condition C1 (usage confound):")
    print(f"    D_marg: {results['c1_d_marg_mean']:.4f} +/- {results['c1_d_marg_std']:.4f}  (target: 0.36)")
    print(f"    D_cond: {results['c1_d_cond_mean']:.4f} +/- {results['c1_d_cond_std']:.4f}  (target: ~0.00)")
    print(f"\n  Condition C2 (true conditional AMD):")
    print(f"    D_marg: {results['c2_d_marg_mean']:.4f} +/- {results['c2_d_marg_std']:.4f}  (target: ~0.00)")
    print(f"    D_cond: {results['c2_d_cond_mean']:.4f} +/- {results['c2_d_cond_std']:.4f}  (target: 0.60)")
    print("=" * 72)

    return results


def main():
    """Run all synthetic experiments and save combined results."""
    np.random.seed(RANDOM_SEED)

    all_results = {}

    exp1_results = experiment_1_bifurcation()
    serializable_exp1 = {
        k: v for k, v in exp1_results.items()
        if k != "kappa_values" and not k.endswith("_forward_eq") and not k.endswith("_backward_eq")
    }
    all_results.update(serializable_exp1)

    exp2_results = experiment_2_confound_stress()
    all_results.update(exp2_results)

    exp3_results = experiment_3_estimator_balancing()
    all_results.update(exp3_results)

    output_path = RESULTS_DIR / "synthetic_experiments.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nAll synthetic results saved to {output_path}")


if __name__ == "__main__":
    main()
