# Phase Transitions in Affective Meaning Divergence

Code for the ACL SRW 2026 paper: "Phase Transitions in Affective Meaning Divergence: The Hidden Drift Before the Break".

📄 [View the paper](https://arxiv.org/abs/2605.09043)

## About

Conversations don't break down suddenly; they drift. This paper formalizes **affective meaning divergence (AMD)**: the total-variation distance between interlocutors' anchor-conditioned affect distributions. When two people use the same word but attach different emotional meanings to it, that gap accumulates silently until repair coordination collapses.

We ground this in entropy-regularized game theory and prove that the resulting repair dynamics undergo a **saddle-node bifurcation**: below a critical AMD load, the dyad rests in a high-repair attractor; above it, repair collapses abruptly and hysteretically. Empirically, we detect **critical slowing down (CSD)** signatures (including rising variance in AMD, dialog-act repair, and lexical divergence) in the turns preceding conversational breakdown on Conversations Gone Awry (CGA-Wiki, N=652), with AMD providing a temporally distinct signal whose variance peaks at the bifurcation point while toxicity variance peaks earlier.

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

## Citation

If you use this code, please cite:
```
@inproceedings{litchiowong-2026-phase,
    title = "Phase Transitions in Affective Meaning Divergence: The Hidden Drift Before the Break",
    author = "Litchiowong, Napassorn",
    booktitle = "Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Student Research Workshop)",
    month = jul,
    year = "2026",
    address = "San Diego, California",
    publisher = "Association for Computational Linguistics",
}
```

## License

MIT
