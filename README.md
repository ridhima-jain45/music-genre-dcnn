# Music Genre Classification using CNN and Classical Algorithms

Reproduction and extension of the paper **"Combining CNN and Classical Algorithms for Music Genre Classification"** by Haojun Li, Siqi Xue and Jialun Zhang (Stanford CS229).

## 1. Project Overview

This project classifies music into five genres using MFCC audio features, classical machine learning models, and a small Dilated Convolutional Neural Network (DCNN).

The project has two clearly separated parts:

1. **Original Paper Implementation** - reproduction of the reference methodology.
2. **Our Research Extensions** - two experiments added beyond the original work:
   - Five-fold stratified cross-validation
   - Dilation ablation study

The original implementation is preserved, and the new experiments are added on top of it.

---

## 2. Original Paper Implementation

### Dataset

The project uses the GTZAN dataset with five genres:

- blues
- classical
- hiphop
- metal
- pop

There are 100 clips per genre, giving 500 clips in total.

The audio files are not included in this repository.

### MFCC Features

Each 30-second clip is converted to:

- Sampling rate: 22050 Hz
- MFCC coefficients: 20
- Frames retained: 1290

Each clip therefore contains:

`20 x 1290 = 25,800` MFCC values.

### Classical Models

The original implementation evaluates:

- Logistic Regression (LR)
- Gaussian Discriminant Analysis (GDA)
- Random Forest (RF)
- Support Vector Machine (SVM)

The models are evaluated on raw MFCC features and PCA-50 features.

### Original Dilated CNN

The original CNN contains two convolutional units:

```text
Unit 1: kernel 8, dilation 4, pooling 32
Unit 2: kernel 4, dilation 2, pooling 16
```

Intermediate feature sizes:

```text
Layer 1 = 320 features
Layer 2 = 48 features
```

The CNN is also evaluated as a standalone classifier.

### CNN as a Feature Extractor

```text
MFCC
  |
  v
Dilated CNN
  |
  +--> Layer 1 (320) --> LR / GDA / RF / SVM
  |
  +--> Layer 2 (48)  --> LR / GDA / RF / SVM
```

---

## 3. Original Paper/Reproduction Results

The original implementation uses a fixed stratified 400/100 train-test split with seed 42.

### Single-split test accuracy

| Representation | LR | GDA | RF | SVM |
|---|---:|---:|---:|---:|
| Raw MFCC (25,800) | 0.70 | 0.81 | 0.79 | 0.77 |
| PCA-50 | 0.68 | 0.74 | 0.75 | 0.79 |
| Layer 1 (320) | 0.81 | 0.65 | 0.88 | 0.86 |
| Layer 2 (48) | 0.86 | 0.89 | 0.86 | 0.89 |

Standalone DCNN test accuracy: **0.85**

These results are the baseline before our extensions.

---

## 4. Our Research Extensions

The original evaluation depends on one 100-sample test split. We added two experiments beyond the original methodology.

### Extension 1 - Five-Fold Stratified Cross-Validation

#### Motivation

With only 100 test clips, one prediction changes accuracy by one percentage point. Five-fold cross-validation evaluates all 500 clips across five different held-out test folds.

We use:

```python
StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

Each fold contains 400 training clips and 100 test clips. Every clip is used as a test sample exactly once.

#### Leakage Prevention

For every fold:

1. A fresh CNN is trained from scratch.
2. CNN MFCC normalization uses training-fold statistics only.
3. CNN epoch selection uses a validation split from the training fold.
4. The held-out test fold is never used for model selection.
5. Layer 1 and Layer 2 features come from that fold's trained CNN.
6. Classical-model standardization uses training-fold features only.
7. The existing `dcnn.pt` checkpoint is not reused for CV.

#### Five-Fold Results

Results are reported as mean test accuracy +/- standard deviation.

| Representation | Algorithm | Mean +/- Std |
|---|---|---:|
| Raw MFCC | LR | 0.814 +/- 0.032 |
| Raw MFCC | GDA | 0.800 +/- 0.043 |
| Raw MFCC | RF | 0.790 +/- 0.037 |
| Raw MFCC | SVM | 0.824 +/- 0.037 |
| DCNN | DCNN | 0.846 +/- 0.050 |
| Layer 1 | LR | 0.850 +/- 0.032 |
| Layer 1 | GDA | 0.584 +/- 0.040 |
| Layer 1 | RF | 0.840 +/- 0.037 |
| Layer 1 | SVM | 0.856 +/- 0.031 |
| Layer 2 | LR | 0.868 +/- 0.029 |
| Layer 2 | GDA | 0.840 +/- 0.054 |
| Layer 2 | RF | 0.870 +/- 0.033 |
| Layer 2 | SVM | 0.866 +/- 0.025 |

Output files:

```text
results/cv_folds.csv
results/cv_summary.csv
```

### Extension 2 - Dilation Ablation Study

#### Research Question

The original CNN uses dilated convolutions. We added an ablation experiment to investigate whether dilation contributes to performance.

The two configurations are:

```text
Original dilated CNN: dilation = (4, 2)
No-dilation CNN:      dilation = (1, 1)
```

All other CNN architecture settings remain unchanged.

#### Configurable CNN

The original configuration is:

```python
DCNN()
```

which uses dilation `(4, 2)`.

The ablation configuration is:

```python
DCNN(dil1=1, dil2=1)
```

Both configurations continue to produce:

```text
Layer 1 = 320 features
Layer 2 = 48 features
```

#### Dilation Ablation Results

| Representation | Algorithm | Dilated (4,2) | No Dilation (1,1) |
|---|---|---:|---:|
| DCNN | DCNN | 0.846 +/- 0.050 | 0.830 +/- 0.035 |
| Layer 1 | LR | 0.850 +/- 0.032 | 0.836 +/- 0.037 |
| Layer 1 | GDA | 0.584 +/- 0.040 | 0.592 +/- 0.054 |
| Layer 1 | RF | 0.840 +/- 0.037 | 0.846 +/- 0.046 |
| Layer 1 | SVM | 0.856 +/- 0.031 | 0.848 +/- 0.015 |
| Layer 2 | LR | 0.868 +/- 0.029 | 0.866 +/- 0.029 |
| Layer 2 | GDA | 0.840 +/- 0.054 | 0.850 +/- 0.036 |
| Layer 2 | RF | 0.870 +/- 0.033 | 0.866 +/- 0.027 |
| Layer 2 | SVM | 0.866 +/- 0.025 | 0.854 +/- 0.021 |

The paired comparison across 45 fold/method combinations produced:

```text
Mean paired difference (Dilated - No Dilation) = 0.0036
```

The paired differences vary across methods and folds, so the experiment does not provide clear evidence of a consistent performance advantage from dilation at this experimental scale.

Output files:

```text
results/cv_nodilation_folds.csv
results/cv_nodilation_summary.csv
results/dilation_comparison.csv
```

---

## 5. Final Project Workflow

```text
REFERENCE PAPER
      |
      v
ORIGINAL PAPER IMPLEMENTATION
      |
      +-- MFCC features
      +-- Raw MFCC + Classical ML
      +-- PCA-50 + Classical ML
      +-- Dilated CNN
             |
             +-- CNN alone
             +-- Layer 1 + Classical ML
             +-- Layer 2 + Classical ML
      |
      v
OUR RESEARCH EXTENSIONS
      |
      +-- Extension 1: Five-Fold Cross-Validation
      |
      +-- Extension 2: Dilation Ablation
```

---

## 6. How to Run

### Windows dataset configuration

Replace the example paths with the actual dataset, cache, and results locations:

```cmd
set GTZAN_DIR=D:\path\to\genres_original
set CACHE_DIR=D:\path\to\cache
set RESULTS_DIR=D:\path\to\results
```

The dataset directory must contain:

```text
blues/
classical/
hiphop/
metal/
pop/
```

### Original Paper Implementation

Build the MFCC cache:

```cmd
python src\data.py
```

Run the classical baselines:

```cmd
python src\baselines.py
```

Train the original dilated CNN:

```cmd
python src\dccn.py
```

Extract CNN features and evaluate classical classifiers:

```cmd
python src\extract_features.py
```

### Our Extension 1 - Five-Fold Cross-Validation

```cmd
python src\cv.py --folds 5 --epochs 100 --output results\cv_folds.csv
```

### Our Extension 2 - No-Dilation Ablation

```cmd
python src\cv.py --folds 5 --epochs 100 --dil1 1 --dil2 1 --skip-raw --output results\cv_nodilation_folds.csv
```

---

## 7. Repository Structure

```text
.
├── README.md
├── requirements.txt
├── src/
│   ├── data.py
│   ├── baselines.py
│   ├── dccn.py
│   ├── extract_features.py
│   └── cv.py
└── results/
    ├── baselines.csv
    ├── cnn_features.csv
    ├── dcnn.pt
    ├── dcnn_confusion.npy
    ├── cv_folds.csv
    ├── cv_summary.csv
    ├── cv_nodilation_folds.csv
    ├── cv_nodilation_summary.csv
    └── dilation_comparison.csv
```

---

## 8. Reproducibility and Limitations

- Five-fold CV uses stratified folds with `random_state=42`.
- A fresh CNN is trained independently inside each fold.
- Test-fold data is not used for normalization, feature fitting, epoch selection, or model selection.
- The dilated and no-dilation CNNs use the same fold assignments and corresponding fold seeds.
- The primary dilation ablation uses one seed per fold.
- Five-fold cross-validation reduces dependence on one split but does not by itself establish statistical significance.
- Conclusions apply to this dataset and experimental setup.

---

## 9. Reference

Haojun Li, Siqi Xue and Jialun Zhang.

**Combining CNN and Classical Algorithms for Music Genre Classification.**

Stanford CS229.

Original code:
https://github.com/LithiumH/CS229-Music-Classification

GTZAN:
http://marsyasweb.appspot.com/download/data_sets/
