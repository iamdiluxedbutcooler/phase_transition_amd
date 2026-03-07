"""
Central configuration for AMD phase-transition experiments.
Defines all paths, hyperparameters, model identifiers, and constants
used across scripts and notebooks.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

RESULTS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

RANDOM_SEED = 42

GOEMOTIONS_MODEL = "SamLowe/roberta-base-go_emotions"
TOXICITY_MODEL = "unitary/toxic-bert"
DA_CLASSIFIER_MODEL = "roberta-base"

GOEMOTIONS_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise", "neutral",
]
NUM_EMOTIONS = len(GOEMOTIONS_LABELS)

CONCILIATORY_DA_TAGS = {"aa", "bk", "br", "ba"}

TOPIC_CLUSTERS_K = 5
ROLLING_WINDOW = 5
EMA_ALPHA = 0.3
MIN_UTTERANCES_PER_CELL = 3
MIN_ANCHOR_FREQ_CGA = 3

PERMUTATION_N = 10_000

LOGISTIC_SOLVER = "lbfgs"
LOGISTIC_MAX_ITER = 1000
CV_FOLDS = 5

DA_FINETUNE_EPOCHS = 3
DA_FINETUNE_LR = 2e-5
DA_FINETUNE_BATCH_SIZE = 16

BIFURCATION_ALPHA_BETA_PAIRS = [(2, 2), (2, 3), (2, 4), (2, 5)]
BIFURCATION_KAPPA_RANGE = (0.0, 2.0)
BIFURCATION_KAPPA_STEP = 0.001
BIFURCATION_T_ITER = 5000

BOOTSTRAP_REPS = 100
BOOTSTRAP_N_SAMPLES = 1000

CGA_FEATURES_PATH = DATA_DIR / "cga_features.parquet"
CGA_CONVERSATION_SUMMARY_PATH = DATA_DIR / "cga_conversation_summary.parquet"
CGA_CMV_FEATURES_PATH = DATA_DIR / "cga_cmv_features.parquet"
CGA_CMV_CONVERSATION_SUMMARY_PATH = DATA_DIR / "cga_cmv_conversation_summary.parquet"
CMV_FEATURES_PATH = DATA_DIR / "cmv_features.parquet"
CMV_CONVERSATION_SUMMARY_PATH = DATA_DIR / "cmv_conversation_summary.parquet"
MELD_FEATURES_PATH = DATA_DIR / "meld_features.parquet"
MELD_CONVERSATION_SUMMARY_PATH = DATA_DIR / "meld_conversation_summary.parquet"
DA_MODEL_SAVE_PATH = MODELS_DIR / "da_classifier"

MIN_ANCHOR_FREQ_CMV = 3
MIN_CMV_TURNS = 6

# MELD emotion labels (7 categories)
MELD_EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
NUM_MELD_EMOTIONS = len(MELD_EMOTIONS)
MIN_ANCHOR_FREQ_MELD = 3

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
