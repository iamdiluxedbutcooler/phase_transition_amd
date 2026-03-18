# Phase Transitions in Affective Meaning Divergence

Code for the ACL 2026 submission: "Phase Transitions in Affective Meaning Divergence: The Hidden Drift Before the Break".

## Structure

```
config.py          # Paths, hyperparameters, constants
utils/             # Core library (AMD, CSD indicators, repair proxies, stats)
scripts/           # Experiment scripts (Exp 1-7, effect sizes, sensitivity)
notebooks/         # Feature extraction and DA classifier training (Colab)
data/              # Generated parquets (not tracked, see below)
```

## Requirements

Python 3.9+. Install:

```bash
pip install torch transformers scikit-learn pandas numpy scipy convokit vaderSentiment
```

## Reproducing

1. Run notebooks 01-04 on a GPU (Colab or local) to extract features into `data/`.
2. Run experiment scripts on CPU:

```bash
python scripts/run_synthetic.py
python scripts/run_cga_analysis.py
python scripts/run_cga_cmv_analysis.py
python scripts/run_exp7_replacement.py
python scripts/run_repair_validation.py
python scripts/compute_effect_sizes.py
python scripts/run_window_sensitivity.py  # Window-size sensitivity (W=3..7)
```

## License

MIT
