"""
Anchor extraction and filtering for affective meaning divergence analysis.
Identifies content words used by both speakers in a conversation,
applies frequency thresholds, and optionally filters by word type.
"""

import re
from collections import Counter

import numpy as np

try:
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words("english"))
except Exception:
    STOPWORDS = {
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
        "your", "yours", "yourself", "yourselves", "he", "him", "his",
        "himself", "she", "her", "hers", "herself", "it", "its", "itself",
        "they", "them", "their", "theirs", "themselves", "what", "which",
        "who", "whom", "this", "that", "these", "those", "am", "is", "are",
        "was", "were", "be", "been", "being", "have", "has", "had", "having",
        "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if",
        "or", "because", "as", "until", "while", "of", "at", "by", "for",
        "with", "about", "against", "between", "through", "during", "before",
        "after", "above", "below", "to", "from", "up", "down", "in", "out",
        "on", "off", "over", "under", "again", "further", "then", "once",
        "here", "there", "when", "where", "why", "how", "all", "both",
        "each", "few", "more", "most", "other", "some", "such", "no", "nor",
        "not", "only", "own", "same", "so", "than", "too", "very", "s", "t",
        "can", "will", "just", "don", "should", "now", "d", "ll", "m", "o",
        "re", "ve", "y", "ain", "aren", "couldn", "didn", "doesn", "hadn",
        "hasn", "haven", "isn", "ma", "mightn", "mustn", "needn", "shan",
        "shouldn", "wasn", "weren", "won", "wouldn",
    }

WORD_PATTERN = re.compile(r"\b[a-z]{2,}\b")


def extract_content_words(text):
    """
    Extract lowercase content words from text, excluding stopwords.

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    list of str
        Content words found in text.
    """
    words = WORD_PATTERN.findall(text.lower())
    return [w for w in words if w not in STOPWORDS]


def extract_all_words(text):
    """
    Extract all lowercase words from text (including function words).

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    list of str
        All words of length >= 2 found in text.
    """
    return WORD_PATTERN.findall(text.lower())


def identify_anchors(utterances_speaker1, utterances_speaker2,
                     min_frequency=3, content_only=True, evaluative_words=None):
    """
    Identify anchor words shared between two speakers meeting frequency threshold.

    Parameters
    ----------
    utterances_speaker1 : list of str
        All utterance texts from speaker 1.
    utterances_speaker2 : list of str
        All utterance texts from speaker 2.
    min_frequency : int
        Minimum total frequency across both speakers.
    content_only : bool
        If True, restrict to content words (non-stopwords).
    evaluative_words : set or None
        If provided, restrict anchors to this set of words.

    Returns
    -------
    list of str
        Sorted list of anchor words meeting all criteria.
    """
    extractor = extract_content_words if content_only else extract_all_words

    words1 = []
    for text in utterances_speaker1:
        words1.extend(extractor(text))

    words2 = []
    for text in utterances_speaker2:
        words2.extend(extractor(text))

    counter1 = Counter(words1)
    counter2 = Counter(words2)

    vocab1 = set(counter1.keys())
    vocab2 = set(counter2.keys())
    shared_vocab = vocab1 & vocab2

    anchors = []
    for word in shared_vocab:
        total_freq = counter1[word] + counter2[word]
        if total_freq >= min_frequency:
            if evaluative_words is not None and word not in evaluative_words:
                continue
            anchors.append(word)

    return sorted(anchors)


def assign_context(preceding_da, topic_cluster):
    """
    Create context variable from dialog act and topic cluster.

    Parameters
    ----------
    preceding_da : str
        Dialog act label of the preceding turn.
    topic_cluster : int
        Topic cluster assignment from k-means on TF-IDF.

    Returns
    -------
    str
        Context string in format "da_label__topic_k".
    """
    return f"{preceding_da}__topic_{topic_cluster}"


def build_anchor_utterance_map(utterances, anchors, speaker_id_col="speaker_id",
                               text_col="text", context_col="context",
                               emotion_dist_col="emotion_dist"):
    """
    Build a mapping from (anchor, speaker) to list of utterance records.

    Parameters
    ----------
    utterances : list of dict
        Each dict must have keys for speaker_id, text, context, and emotion_dist.
    anchors : list of str
        Anchor words to track.
    speaker_id_col : str
        Key name for speaker identifier.
    text_col : str
        Key name for utterance text.
    context_col : str
        Key name for context variable.
    emotion_dist_col : str
        Key name for emotion distribution array.

    Returns
    -------
    dict
        Maps (anchor, speaker_id) to list of dicts with 'context' and 'emotion_dist'.
    """
    anchor_set = set(anchors)
    result = {}

    for utt in utterances:
        text = utt[text_col].lower()
        speaker = utt[speaker_id_col]
        context = utt[context_col]
        emotion_dist = utt[emotion_dist_col]

        words_in_text = set(WORD_PATTERN.findall(text))
        matching_anchors = words_in_text & anchor_set

        for anchor in matching_anchors:
            key = (anchor, speaker)
            if key not in result:
                result[key] = []
            result[key].append({
                "context": context,
                "emotion_dist": np.asarray(emotion_dist),
            })

    return result
