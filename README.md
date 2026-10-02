# Music Genre Classification: Dilated CNN + Classical Algorithms

Reproduction (and extension) of **"Combining CNN and Classical Algorithms for Music Genre
Classification"** by Haojun Li, Siqi Xue and Jialun Zhang (Stanford, CS229).

**Idea:** train a small dilated CNN on MFCC features, then use its intermediate layer outputs as
compact features for classical classifiers (Logistic Regression, GDA, Random Forest, SVM), and
compare against the same classifiers on raw MFCCs.

## Dataset

[GTZAN](http://marsyasweb.appspot.com/download/data_sets/) (Tzanetakis & Cook): 30-second clips.
As in the paper, only 5 genres are used: **blues, classical, hiphop, metal, pop**
(100 clips each, 500 total). Fixed stratified split: 400 train / 100 test (seed 42).

The audio is **not included in this repo**. On Kaggle, attach the dataset
*"GTZAN Dataset - Music Genre Classification"* (by andradaolteanu) to your notebook.

## Repository structure

```
.
├── README.md
├── requirements.txt
├── src/
│   ├── data.py              # MFCC extraction (20 x 1290 per clip), caching, fixed train/test split
│   ├── baselines.py         # LR / GDA / RF / SVM on raw and PCA-50 features
│   ├── dcnn.py              # dilated CNN definition + training (validation-based epoch selection)
│   ├── extract_features.py  # CNN layer 1 / layer 2 features -> classical classifiers
│   └── cv.py                # 5-fold cross-validation + GDA shrinkage   (TODO: results)
├── results/                 # CSV tables, figures, saved model weights
└── report/                  # 2-page write-up (PDF)
```

## Setup and how to run

### On Kaggle (what we used)

1. Create a notebook and attach the GTZAN dataset (*Add Input*).
2. Clone this repo (or upload the `src/` folder), then run from inside `src/`:

```python
!git clone <repo-url>
%cd <repo-name>/src
!python data.py              # extract MFCCs, cache them, print dataset summary (~1-3 min)
!python baselines.py         # classical baselines on raw / PCA features
!python dcnn.py              # train the dilated CNN, save weights to results/
!python extract_features.py  # classical classifiers on CNN features (needs dcnn.py first)
!python cv.py                # 5-fold cross-validation (slowest, ~5-15 min on CPU)
```

Default paths point to the Kaggle input and `/kaggle/working`.

### Locally

```bash
pip install -r requirements.txt
export GTZAN_DIR=/path/to/genres_original   # folder containing blues/, classical/, ...
export CACHE_DIR=./cache
export RESULTS_DIR=./results
cd src && python data.py && python baselines.py && python dcnn.py && python extract_features.py
```

Run order matters: `data.py` builds the cache first, and `dcnn.py` must run before
`extract_features.py` because it saves the trained weights.

## Method summary

1. **Preprocessing:** `librosa` MFCCs, 20 coefficients x 1290 frames (25,800 features per clip).
2. **Baselines:** LR, GDA (LDA, shared covariance), RF (max depth 7), RBF-SVM on raw and PCA-50 features.
3. **Dilated CNN:** 2 x [dilated Conv1d -> ReLU -> AvgPool -> Dropout -> BatchNorm] -> Linear(5).
   Layer 1 output = 320 features, layer 2 output = 48 features. Adam, lr 1e-3, cross-entropy.
4. **CNN as feature extractor:** flatten layer 1 / layer 2 activations and train the four classical models on them.

## Results

> TODO: fill in from `results/*.csv` once all experiments are final.

| Test accuracy | LR | GDA | RF | SVM |
|---|---|---|---|---|
| Raw MFCC (25800) | | | | |
| DCNN layer 1 (320) | | | | |
| DCNN layer 2 (48) | | | | |

DCNN alone: _TODO_

## Our changes relative to the paper

- Best CNN epoch chosen on a held-out **validation** split (the paper tuned on the test set).
- Features standardised using **training statistics only** (no test leakage).
- SVM baseline: the paper's very low SVM accuracy is reproduced only with `gamma="auto"` on unscaled
  features; with `gamma="scale"` the same SVM reaches ~0.77 on raw MFCCs.
- _TODO: add cross-validation, GDA shrinkage, and any further improvements here._

## Reference

- Li, Xue, Zhang. *Combining CNN and Classical Algorithms for Music Genre Classification.* Stanford CS229.
  Original code: https://github.com/LithiumH/CS229-Music-Classification
- Tzanetakis & Cook (2002). *Musical genre classification of audio signals.* IEEE TSAP 10(5).

## Author
Ridhima Jain, PES1UG24CS373
