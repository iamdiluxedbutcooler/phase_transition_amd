"""
Repair proxy implementations for conversational breakdown detection.
Provides three complementary measures of conversational repair effort:
dialog-act based, regex-marker based, and civility-based.
"""

import re
import numpy as np


CONCILIATORY_DA_TAGS = {"aa", "bk", "br", "ba"}

REPAIR_PATTERNS = [
    r"\bwhat\b.*\?",
    r"\bhuh\b",
    r"\bsorry\b",
    r"\bpardon\b",
    r"\bexcuse me\b",
    r"\bI mean\b",
    r"\bactually\b",
    r"\bno\s*,",
    r"\bwait\b",
    r"\bhold on\b",
    r"\byeah\b",
    r"\bright\b",
    r"\bokay\b",
    r"\bmhm\b",
    r"\buh huh\b",
    r"\bI see\b",
]

COMPILED_REPAIR_PATTERNS = [re.compile(p, re.IGNORECASE) for p in REPAIR_PATTERNS]


def compute_q_da(da_probabilities, conciliatory_tags=None):
    """
    Compute dialog-act repair proxy as probability of conciliatory act.

    Parameters
    ----------
    da_probabilities : dict
        Maps dialog-act label to its predicted probability.
    conciliatory_tags : set or None
        Set of DA tags considered conciliatory. Uses default if None.

    Returns
    -------
    float
        Sum of probabilities assigned to conciliatory dialog acts.
    """
    if conciliatory_tags is None:
        conciliatory_tags = CONCILIATORY_DA_TAGS

    return sum(da_probabilities.get(tag, 0.0) for tag in conciliatory_tags)


def compute_q_rm_series(utterance_texts, alpha=0.3):
    """
    Compute repair-marker proxy as EMA-smoothed binary indicator series.

    Parameters
    ----------
    utterance_texts : list of str
        Ordered sequence of utterance texts in a conversation.
    alpha : float
        Exponential moving average smoothing parameter.

    Returns
    -------
    numpy.ndarray
        EMA-smoothed repair marker series, one value per utterance.
    """
    raw_indicators = np.zeros(len(utterance_texts))

    for idx, text in enumerate(utterance_texts):
        for pattern in COMPILED_REPAIR_PATTERNS:
            if pattern.search(text):
                raw_indicators[idx] = 1.0
                break

    smoothed = np.zeros(len(utterance_texts))
    if len(utterance_texts) == 0:
        return smoothed

    smoothed[0] = raw_indicators[0]
    for idx in range(1, len(utterance_texts)):
        smoothed[idx] = alpha * raw_indicators[idx] + (1 - alpha) * smoothed[idx - 1]

    return smoothed


def compute_q_ce(toxicity_probability):
    """
    Compute civility-based repair proxy as complement of toxicity probability.

    Parameters
    ----------
    toxicity_probability : float
        P(toxic | utterance) from toxicity classifier.

    Returns
    -------
    float
        1 - P(toxic | utterance).
    """
    return 1.0 - toxicity_probability


def compute_all_repair_proxies(utterance_texts, da_probabilities_list,
                               toxicity_probabilities, alpha=0.3):
    """
    Compute all three repair proxies for a full conversation.

    Parameters
    ----------
    utterance_texts : list of str
        Ordered utterance texts.
    da_probabilities_list : list of dict
        Per-utterance dialog-act probability distributions.
    toxicity_probabilities : list of float
        Per-utterance toxicity probabilities.
    alpha : float
        EMA smoothing parameter for repair marker proxy.

    Returns
    -------
    dict
        Keys 'q_da', 'q_rm', 'q_ce', each mapping to a numpy array.
    """
    q_da_values = np.array([compute_q_da(da_probs) for da_probs in da_probabilities_list])
    q_rm_values = compute_q_rm_series(utterance_texts, alpha=alpha)
    q_ce_values = np.array([compute_q_ce(tox) for tox in toxicity_probabilities])

    return {
        "q_da": q_da_values,
        "q_rm": q_rm_values,
        "q_ce": q_ce_values,
    }
