"""
Affective Meaning Divergence computation.
Implements marginal AMD, conditional AMD, and context divergence
using total variation distance (0.5 * L1 norm) between probability distributions.
"""

import numpy as np


def total_variation(p, q):
    """Compute total variation distance between two discrete distributions."""
    return 0.5 * np.sum(np.abs(np.asarray(p) - np.asarray(q)))


def marginal_amd(emotion_distributions_speaker1, context_weights_speaker1,
                 emotion_distributions_speaker2, context_weights_speaker2):
    """
    Compute marginal AMD between two speakers for a single anchor.

    Parameters
    ----------
    emotion_distributions_speaker1 : dict
        Maps context label to mean emotion distribution (numpy array) for speaker 1.
    context_weights_speaker1 : dict
        Maps context label to P(c|x) for speaker 1.
    emotion_distributions_speaker2 : dict
        Maps context label to mean emotion distribution (numpy array) for speaker 2.
    context_weights_speaker2 : dict
        Maps context label to P(c|x) for speaker 2.

    Returns
    -------
    float
        Marginal AMD value D_marg(x).
    """
    all_contexts = set(context_weights_speaker1.keys()) | set(context_weights_speaker2.keys())
    num_emotions = None

    for ctx in all_contexts:
        if ctx in emotion_distributions_speaker1:
            num_emotions = len(emotion_distributions_speaker1[ctx])
            break
        if ctx in emotion_distributions_speaker2:
            num_emotions = len(emotion_distributions_speaker2[ctx])
            break

    if num_emotions is None:
        return 0.0

    marginal_speaker1 = np.zeros(num_emotions)
    marginal_speaker2 = np.zeros(num_emotions)

    for ctx in all_contexts:
        weight1 = context_weights_speaker1.get(ctx, 0.0)
        dist1 = emotion_distributions_speaker1.get(ctx, np.zeros(num_emotions))
        marginal_speaker1 += weight1 * np.asarray(dist1)

        weight2 = context_weights_speaker2.get(ctx, 0.0)
        dist2 = emotion_distributions_speaker2.get(ctx, np.zeros(num_emotions))
        marginal_speaker2 += weight2 * np.asarray(dist2)

    return total_variation(marginal_speaker1, marginal_speaker2)


def conditional_amd(emotion_distributions_speaker1, emotion_distributions_speaker2,
                    utterance_counts_speaker1, utterance_counts_speaker2):
    """
    Compute conditional AMD between two speakers for a single anchor.

    Parameters
    ----------
    emotion_distributions_speaker1 : dict
        Maps context label to mean emotion distribution for speaker 1.
    emotion_distributions_speaker2 : dict
        Maps context label to mean emotion distribution for speaker 2.
    utterance_counts_speaker1 : dict
        Maps context label to number of utterances by speaker 1.
    utterance_counts_speaker2 : dict
        Maps context label to number of utterances by speaker 2.

    Returns
    -------
    float
        Conditional AMD value D_cond(x).
    """
    shared_contexts = (set(emotion_distributions_speaker1.keys())
                       & set(emotion_distributions_speaker2.keys()))

    if not shared_contexts:
        return np.nan

    total_weight = 0.0
    weighted_tv_sum = 0.0

    for ctx in shared_contexts:
        count1 = utterance_counts_speaker1.get(ctx, 0)
        count2 = utterance_counts_speaker2.get(ctx, 0)
        weight = count1 + count2
        tv = total_variation(emotion_distributions_speaker1[ctx],
                             emotion_distributions_speaker2[ctx])
        weighted_tv_sum += weight * tv
        total_weight += weight

    if total_weight == 0:
        return np.nan

    return weighted_tv_sum / total_weight


def context_divergence(context_weights_speaker1, context_weights_speaker2):
    """
    Compute context divergence between two speakers for a single anchor.

    Parameters
    ----------
    context_weights_speaker1 : dict
        Maps context label to P(c|x) for speaker 1.
    context_weights_speaker2 : dict
        Maps context label to P(c|x) for speaker 2.

    Returns
    -------
    float
        Context divergence D_ctx(x).
    """
    all_contexts = set(context_weights_speaker1.keys()) | set(context_weights_speaker2.keys())

    p = np.array([context_weights_speaker1.get(ctx, 0.0) for ctx in all_contexts])
    q = np.array([context_weights_speaker2.get(ctx, 0.0) for ctx in all_contexts])

    p_sum = p.sum()
    q_sum = q.sum()
    if p_sum > 0:
        p = p / p_sum
    if q_sum > 0:
        q = q / q_sum

    return total_variation(p, q)


def compute_amd_for_anchor(utterances_speaker1, utterances_speaker2, min_per_cell=3):
    """
    Compute all three AMD variants for a single anchor given utterance-level data.

    Parameters
    ----------
    utterances_speaker1 : list of dict
        Each dict has keys 'context' and 'emotion_dist' (numpy array).
    utterances_speaker2 : list of dict
        Same format as utterances_speaker1.
    min_per_cell : int
        Minimum number of utterances per (speaker, context) cell.

    Returns
    -------
    dict
        Keys: 'd_marg', 'd_cond', 'd_ctx'. Values may be np.nan if insufficient data.
    """
    def aggregate_by_context(utterances):
        context_groups = {}
        for utt in utterances:
            ctx = utt["context"]
            if ctx not in context_groups:
                context_groups[ctx] = []
            context_groups[ctx].append(np.asarray(utt["emotion_dist"]))
        return context_groups

    groups1 = aggregate_by_context(utterances_speaker1)
    groups2 = aggregate_by_context(utterances_speaker2)

    emotion_dists1 = {}
    emotion_dists2 = {}
    counts1 = {}
    counts2 = {}
    ctx_weights1 = {}
    ctx_weights2 = {}

    for ctx, dists in groups1.items():
        if len(dists) >= min_per_cell:
            emotion_dists1[ctx] = np.mean(dists, axis=0)
            counts1[ctx] = len(dists)

    for ctx, dists in groups2.items():
        if len(dists) >= min_per_cell:
            emotion_dists2[ctx] = np.mean(dists, axis=0)
            counts2[ctx] = len(dists)

    total1 = sum(counts1.values())
    total2 = sum(counts2.values())

    if total1 > 0:
        ctx_weights1 = {ctx: cnt / total1 for ctx, cnt in counts1.items()}
    if total2 > 0:
        ctx_weights2 = {ctx: cnt / total2 for ctx, cnt in counts2.items()}

    if not emotion_dists1 or not emotion_dists2:
        return {"d_marg": np.nan, "d_cond": np.nan, "d_ctx": np.nan}

    d_marg = marginal_amd(emotion_dists1, ctx_weights1, emotion_dists2, ctx_weights2)
    d_cond = conditional_amd(emotion_dists1, emotion_dists2, counts1, counts2)
    d_ctx = context_divergence(ctx_weights1, ctx_weights2)

    return {"d_marg": d_marg, "d_cond": d_cond, "d_ctx": d_ctx}
