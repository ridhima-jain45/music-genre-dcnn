# Music Genre Classification: Dilated CNN + Classical Algorithms

Reproduction and extension of **"Combining CNN and Classical Algorithms for Music Genre
Classification"** by Haojun Li, Siqi Xue and Jialun Zhang (Stanford, CS229).

**Idea:** train a small dilated CNN on MFCC features, then use its intermediate layer outputs as
compact features for classical classifiers (Logistic Regression, GDA, Random Forest, SVM), and
compare against the same classifiers on raw MFCCs.

**Headline results (5-fold cross-validation, 500 clips, 5 genres):**

- Classical models on the CNN's layer-2 features reach about **0.87** accuracy, versus **0.79-0.82** on raw MFCCs.
- Training on 3-second segments and voting per clip raises the CNN to **0.876 ± 0.021** and the best classical models to about **0.88-0.89**.
- GDA shrinkage fixes GDA's failure on layer-1 features (0.58 to 0.81).
- Dilation and hyperparameter tuning gave **no clear benefit** within noise.

## Dataset

[GTZAN](http://marsyasweb.appspot.com/download/data_sets/) (Tzanetakis & Cook): 30-second clips.
As in the paper, only 5 genres are used: **blues, classical, hiphop, metal, pop**
(100 clips each, 500 total). The single-split experiments use a fixed stratified 400 train / 100 test
split (seed 42); the cross-validation experiments use 5 stratified folds (seed 42).

The audio is **not included in this repo**. On Kaggle, attach the dataset
*"GTZAN Dataset - Music Genre Classification"* (by andradaolteanu) to the notebook.

## Repository structure

```
.
├── README.md
├── requirements.txt
├── src/
│   ├── data.py              # MFCC extraction (20 x 1290 per clip), caching, fixed train/test split
│   ├── baselines.py         # LR / GDA / RF / SVM on raw and PCA-50 features
│   ├── dcnn.py              # dilated CNN definition + training (validation-based epoch selection)
│   ├── extract_features.py  # CNN layer 1 / layer 2 features -> classical classifiers (single split)
│   ├── cv.py                # leakage-free 5-fold CV + dilation ablation (--dil1/--dil2)
│   ├── tuning.py            # GDA shrinkage + hyperparameter tuning on CNN features (5-fold CV)
│   ├── segments.py          # 3-second segment model with clip-level voting (5-fold CV)
│   └── predict.py           # demo: audio file -> predicted genre
├── results/                 # CSV tables, confusion matrix, saved model weights
└── report/                  # 2-page write-up (PDF)
```

## Setup and how to run

### On Kaggle (what we used)

1. Create a notebook and attach the GTZAN dataset (*Add Input*).
2. Put the files from `src/` into `/kaggle/working` (clone the repo, or create each file with `%%writefile`).
3. Run from that folder:

```python
%cd /kaggle/working
!python data.py              # extract MFCCs, cache them, print dataset summary (~1-3 min)
!python baselines.py         # classical baselines on raw / PCA features
!python dcnn.py              # train the dilated CNN, save weights to results/dcnn.pt
!python extract_features.py  # classical classifiers on CNN features (needs dcnn.py first)

# cross-validation and dilation ablation (~5-15 min each)
!python cv.py --output /kaggle/working/results/cv_folds_dilated.csv
!python cv.py --dil1 1 --dil2 1 --skip-raw --output /kaggle/working/results/cv_folds_nodilation.csv

!python tuning.py            # GDA shrinkage + tuning (~3 min)
!python segments.py --epochs 20   # segment model with clip voting (~3-10 min)
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

### Notes

- **Run order matters.** `data.py` builds the MFCC cache first. `dcnn.py` must run before `extract_features.py` and `predict.py`, because it saves `results/dcnn.pt`.
- `tuning.py` and `segments.py` import `set_seed` from `cv.py`, so keep `cv.py` in the same folder.
- Output file names passed to `cv.py` must contain `_folds` (the summary file name is derived from it).
- `cv.py` runs on **CPU**. Do not enable a GPU accelerator for it.

## Demo

```bash
python predict.py path/to/song.wav
```

Loads the saved CNN (`results/dcnn.pt`), converts the audio to MFCCs, and prints the predicted genre
with the probability for each of the five genres. Nothing is trained at demo time.
The model only knows the five training genres, so any other genre will still receive one of them.

## Method summary

1. **Preprocessing:** `librosa` MFCCs, 20 coefficients x 1290 frames (25,800 features per clip).
2. **Baselines:** LR, GDA (LDA, shared covariance), RF (max depth 7, 200 trees), RBF-SVM on raw and PCA-50 features.
3. **Dilated CNN:** 2 x [dilated Conv1d -> ReLU -> AvgPool -> Dropout -> BatchNorm] -> Linear(5).
   Unit 1: 8 filters, kernel 8, dilation 4, pool 32 (320 features). Unit 2: 24 filters, kernel 4,
   dilation 2, pool 16 (48 features). Adam, lr 1e-3, cross-entropy, up to 100 epochs, best epoch chosen on a validation split.
4. **CNN as feature extractor:** flatten layer 1 / layer 2 activations and train the four classical models on them.
5. **Segment model:** each clip is cut into ten 129-frame (~3 s) segments, the CNN pooling is reduced
   (layer 1: 128 features, layer 2: 96), and segment predictions are combined per clip
   (CNN: mean softmax, classical models: majority vote).

## Results

All numbers are test accuracy. CSV files for every table are in `results/`.

### 1. Reproduction on the fixed 400/100 split (ours, with the paper's value in brackets)

Classical models, features used as-is (the paper's setting):

| Features | LR | GDA | RF | SVM |
|---|---|---|---|---|
| Raw MFCC (25800) | 0.70 (0.77) | 0.81 (0.77) | 0.79 (0.76) | 0.77 (0.28) |
| DCNN layer 1 (320) | 0.80 (0.87) | 0.65 (0.61) | 0.87 (0.86) | 0.86 (0.87) |
| DCNN layer 2 (48) | 0.86 (0.84) | 0.89 (0.80) | 0.86 (0.87) | 0.89 (0.84) |

**DCNN alone: 0.840** (paper: 0.84-0.87). Best epoch 26, validation accuracy 0.875.

DCNN confusion matrix (rows = true genre, columns = predicted; order: blues, classical, hiphop, metal, pop):

```
blues      [13  0  2  5  0]
classical  [ 0 19  1  0  0]
hiphop     [ 0  0 12  5  3]
metal      [ 0  0  0 20  0]
pop        [ 0  0  0  0 20]
```

Metal and pop are always recognised; blues and hip-hop clips are most often mistaken for metal.

### 2. Baseline findings

| Raw MFCC, test accuracy | LR | GDA | RF | SVM |
|---|---|---|---|---|
| Paper setting (no scaling) | 0.70 | 0.81 | 0.79 | 0.77 |
| Standardised features | 0.82 | 0.81 | 0.79 | 0.81 |
| SVM with `gamma="auto"` (no scaling) | | | | 0.32 |

- Scaling mainly helps LR (0.70 to 0.82, and 0.68 to 0.84 with PCA-50).
- **The paper's SVM collapse (0.28 raw, 0.21 with PCA) is reproduced only with `gamma="auto"` on unscaled features**
  (ours: 0.32 and 0.21). With scikit-learn's current default (`gamma="scale"`) the same SVM reaches 0.77-0.79.
  The paper's very low SVM baseline therefore comes from a hyperparameter setting, and a correctly configured raw-feature SVM is already near 0.8.

### 3. 5-fold cross-validation (mean ± std over folds, 500 clips)

The CNN is retrained inside every fold; the scaler, the MFCC normalisation and the best-epoch selection use only the training part of the fold.

| Features | LR | GDA | RF | SVM |
|---|---|---|---|---|
| Raw MFCC (25800) | 0.814 ± 0.032 | 0.800 ± 0.043 | 0.790 ± 0.037 | 0.824 ± 0.037 |
| DCNN layer 1 (320) | 0.846 ± 0.024 | 0.582 ± 0.036 | 0.842 ± 0.031 | 0.856 ± 0.030 |
| DCNN layer 2 (48) | 0.868 ± 0.029 | 0.840 ± 0.054 | 0.872 ± 0.036 | 0.866 ± 0.025 |

**DCNN alone: 0.846 ± 0.050.** Classical models on layer-2 features beat raw MFCCs by about 4-8 points,
but beat the CNN itself by only about 2 points, which is within the fold-to-fold noise.

### 4. Improvements

**a) GDA shrinkage** (`tuning.py`):

| Features | GDA | GDA with shrinkage | Folds improved |
|---|---|---|---|
| Layer 1 (320) | 0.582 ± 0.036 | **0.808 ± 0.026** | 5 of 5 |
| Layer 2 (48) | 0.840 ± 0.054 | 0.850 ± 0.046 | 3 of 5 |

With 320 features and 400 training clips the GDA covariance estimate is unstable; shrinkage fixes it.
With 48 features there is no clear gain.

**b) Dilation ablation** (`cv.py --dil1 1 --dil2 1`), dilated (4, 2) versus not dilated (1, 1):

| | Dilated | No dilation |
|---|---|---|
| DCNN alone | 0.846 ± 0.050 | 0.828 ± 0.035 |
| Layer 2, LR / RF / SVM | 0.868 / 0.872 / 0.866 | 0.864 / 0.864 / 0.854 |
| Layer 1, LR / RF / SVM | 0.846 / 0.842 / 0.856 | 0.834 / 0.842 / 0.848 |

Dilated networks score 0-2 points higher, which is smaller than the fold std (3-5 points).
**We found no clear benefit of dilation at this scale.**

**c) Hyperparameter tuning** (`tuning.py`; LR `C`, SVM `C`/`gamma`, RF depth/leaf size, chosen by an inner 3-fold CV on the training part only):

| Features | LR default / tuned | RF default / tuned | SVM default / tuned |
|---|---|---|---|
| Layer 1 | 0.846 / 0.858 | 0.842 / 0.848 | 0.856 / 0.854 |
| Layer 2 | 0.868 / 0.864 | 0.872 / 0.868 | 0.866 / 0.856 |

Changes are within ±1.2 points and mostly not consistent across folds. **Tuning gave no reliable improvement.**

**d) 3-second segments with clip-level voting** (`segments.py`; clip-level accuracy):

| Features | LR | GDA | RF | SVM |
|---|---|---|---|---|
| Layer 1 (128) | 0.886 ± 0.029 | 0.850 ± 0.024 | 0.858 ± 0.029 | 0.876 ± 0.030 |
| Layer 2 (96) | 0.882 ± 0.028 | 0.870 ± 0.025 | 0.884 ± 0.027 | 0.878 ± 0.036 |

**DCNN alone: 0.876 ± 0.021** at clip level (0.826 ± 0.027 per individual segment), compared with 0.846 ± 0.050 for the full-clip CNN.

- Folds are created by clip before cutting, so no song appears on both sides of a split.
- Most of the CNN's gain comes from one fold (fold 5: 0.76 to 0.86); the other folds change by 0-3 points. The main visible benefit is a lower variance (std 0.050 to 0.021).
- Plain GDA no longer fails on layer 1 (0.850) because there are about ten times more training samples.

### 5. Summary of findings

| Question | Result |
|---|---|
| Do CNN features beat raw MFCCs? | Yes, by about 4-8 points (consistent across experiments) |
| Do classical models beat the CNN itself? | Marginally (about 2 points), within noise |
| Does GDA shrinkage help? | Yes on layer 1 (+22 points); no clear gain on layer 2 |
| Does dilation help? | No clear effect |
| Does hyperparameter tuning help? | No |
| Do segments + voting help? | Modestly (+1 to +3 points on average) and they stabilise the CNN |

Best overall pipeline: segment-based CNN with LR or RF on its features, about **0.88** clip accuracy.

## Our changes relative to the paper

- The best CNN epoch is chosen on a held-out **validation** split (the paper tuned on the test set).
- Features are standardised with **training statistics only** (no test leakage).
- Results are reported with **5-fold cross-validation** (mean ± std), not just one 100-clip test set.
- The SVM baseline was diagnosed (the paper's collapse is reproduced only with `gamma="auto"`).
- GDA shrinkage, a dilation ablation, hyperparameter tuning and a segment-based model were added.

## Limitations

- Each fold uses one training seed per method, so part of the differences between methods can be training noise.
- The segment model also uses different pooling sizes, so its gain mixes "more data" with "different pooling".
  The raw-MFCC baselines were not segmented.
- The dilation ablation changes the time span seen by each filter as well as the tap spacing; no span-matched control was run.
- Unit-2 hyperparameters (24 filters, kernel 4, dilation 2, pool 16) are our choice, picked to match the paper's feature counts, because the paper does not fully specify them.
- With 100 clips per test fold, one clip is 1%, so differences of 2-3 points are not conclusive.
- GTZAN is small, and it has known issues (repeated artists and some mislabelled clips).

## Contributions

| Part | Author |
|---|---|
| Data pipeline, baselines, CNN, feature extraction, GDA shrinkage, tuning, segment model, demo, report | Ridhima Jain |
| 5-fold cross-validation (`cv.py`), dilation ablation | Sampath |

## References

- Li, Xue, Zhang. *Combining CNN and Classical Algorithms for Music Genre Classification.* Stanford CS229.
  Original code: https://github.com/LithiumH/CS229-Music-Classification
- Tzanetakis & Cook (2002). *Musical genre classification of audio signals.* IEEE TSAP 10(5).
