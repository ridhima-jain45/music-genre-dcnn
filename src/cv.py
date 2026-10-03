"""Leakage-free 5-fold cross-validation and dilation ablation."""


import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split

from data import build_dataset, GENRES, SEED
from baselines import run_classifiers
from dcnn import DCNN, train, accuracy, DEVICE


CLASSICAL_ALGORITHMS = ["LR", "GDA", "RF", "SVM"]


def set_seed(seed):
    """Set reproducible Python, NumPy, and PyTorch seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def evaluate_fold(
    X,
    y,
    train_idx,
    test_idx,
    fold,
    epochs,
    dil1=4,
    dil2=2,
    include_raw=True,
):
    """Evaluate all requested methods for one CV fold."""

    fold_seed = SEED + fold
    set_seed(fold_seed)

    X_train_raw = X[train_idx]
    y_train = y[train_idx]

    X_test_raw = X[test_idx]
    y_test = y[test_idx]

    rows = []

    # ---------------------------------------------------------------
    # 1. Raw MFCC + classical classifiers
    #
    # These baselines are independent of CNN dilation.
    # StandardScaler is fitted only on the training fold.
    # ---------------------------------------------------------------
    if include_raw:
        Xtr_flat = X_train_raw.reshape(len(X_train_raw), -1)
        Xte_flat = X_test_raw.reshape(len(X_test_raw), -1)

        raw_results = run_classifiers(
            Xtr_flat,
            y_train,
            Xte_flat,
            y_test,
            scale=True,
        )

        for algorithm in CLASSICAL_ALGORITHMS:
            rows.append({
                "fold": fold,
                "feature_representation": "raw_mfcc_25800",
                "algorithm": algorithm,
                "test_accuracy": raw_results[algorithm][1],
                "sample_count": len(test_idx),
                "seed": fold_seed,
                "dil1": 4,
                "dil2": 2,
            })

    # ---------------------------------------------------------------
    # 2. Fold-specific MFCC normalization for CNN
    #
    # Mean and std are calculated ONLY from the training fold.
    # ---------------------------------------------------------------
    mean = X_train_raw.mean(axis=(0, 2), keepdims=True)
    std = X_train_raw.std(axis=(0, 2), keepdims=True) + 1e-8

    X_train_norm = (X_train_raw - mean) / std
    X_test_norm = (X_test_raw - mean) / std

    X_train_tensor = torch.tensor(
        X_train_norm,
        dtype=torch.float32,
    )

    X_test_tensor = torch.tensor(
        X_test_norm,
        dtype=torch.float32,
    )

    y_train_tensor = torch.tensor(
        y_train,
        dtype=torch.long,
    )

    y_test_tensor = torch.tensor(
        y_test,
        dtype=torch.long,
    )

    # ---------------------------------------------------------------
    # 3. Validation split inside the training fold
    #
    # The CV test fold is never used for epoch selection.
    # ---------------------------------------------------------------
    inner_train_idx, val_idx = train_test_split(
        np.arange(len(y_train)),
        test_size=0.2,
        stratify=y_train,
        random_state=fold_seed,
    )

    # ---------------------------------------------------------------
    # 4. Train a fresh CNN inside this fold
    # ---------------------------------------------------------------
    set_seed(fold_seed)

    model = DCNN(
        dil1=dil1,
        dil2=dil2,
    ).to(DEVICE)

    best_val, best_ep = train(
        model,
        X_train_tensor[inner_train_idx],
        y_train_tensor[inner_train_idx],
        X_train_tensor[val_idx],
        y_train_tensor[val_idx],
        epochs=epochs,
    )

    # CNN test accuracy is measured only after the best validation
    # epoch has been selected.
    cnn_test_acc, _ = accuracy(
        model,
        X_test_tensor,
        y_test_tensor,
    )

    rows.append({
        "fold": fold,
        "feature_representation": "dcnn",
        "algorithm": "DCNN",
        "test_accuracy": cnn_test_acc,
        "sample_count": len(test_idx),
        "seed": fold_seed,
        "dil1": dil1,
        "dil2": dil2,
    })

    print(
        f"Fold {fold}: "
        f"best_epoch={best_ep}, "
        f"val={best_val:.3f}, "
        f"CNN_test={cnn_test_acc:.3f}"
    )

    # ---------------------------------------------------------------
    # 5. Extract Layer 1 and Layer 2 features
    #
    # Features are extracted from the CNN trained in THIS fold.
    # ---------------------------------------------------------------
    model.eval()

    with torch.no_grad():
        _, train_f1, train_f2 = model(
            X_train_tensor,
            return_features=True,
        )

        _, test_f1, test_f2 = model(
            X_test_tensor,
            return_features=True,
        )

    train_f1 = train_f1.cpu().numpy()
    test_f1 = test_f1.cpu().numpy()

    train_f2 = train_f2.cpu().numpy()
    test_f2 = test_f2.cpu().numpy()

    # Expected feature sizes from the original architecture.
    assert train_f1.shape[1] == 320
    assert test_f1.shape[1] == 320

    assert train_f2.shape[1] == 48
    assert test_f2.shape[1] == 48

    # ---------------------------------------------------------------
    # 6. Classical models on Layer 1 and Layer 2 features
    #
    # StandardScaler is fitted separately on each training-fold
    # feature representation only.
    # ---------------------------------------------------------------
    for feature_name, Xtr_feat, Xte_feat in [
        ("layer1_320", train_f1, test_f1),
        ("layer2_48", train_f2, test_f2),
    ]:
        results = run_classifiers(
            Xtr_feat,
            y_train,
            Xte_feat,
            y_test,
            scale=True,
        )

        for algorithm in CLASSICAL_ALGORITHMS:
            rows.append({
                "fold": fold,
                "feature_representation": feature_name,
                "algorithm": algorithm,
                "test_accuracy": results[algorithm][1],
                "sample_count": len(test_idx),
                "seed": fold_seed,
                "dil1": dil1,
                "dil2": dil2,
            })

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Leakage-free 5-fold CV and dilation ablation."
    )

    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of stratified CV folds.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum CNN training epochs per fold.",
    )

    parser.add_argument(
        "--dil1",
        type=int,
        default=4,
        help="Dilation of CNN unit 1.",
    )

    parser.add_argument(
        "--dil2",
        type=int,
        default=2,
        help="Dilation of CNN unit 2.",
    )

    parser.add_argument(
        "--skip-raw",
        action="store_true",
        help="Skip raw-MFCC baselines.",
    )

    parser.add_argument(
        "--output",
        default=os.path.join(
            os.environ.get("RESULTS_DIR", "results"),
            "cv_folds.csv",
        ),
        help="Output CSV for fold-level results.",
    )

    args = parser.parse_args()

    if args.folds < 2:
        raise ValueError("--folds must be at least 2.")

    X, y, paths = build_dataset()

    print("Dataset:", X.shape)
    print("Labels:", y.shape)
    print("Device:", DEVICE)
    print(
        f"Dilation: unit1={args.dil1}, "
        f"unit2={args.dil2}"
    )

    skf = StratifiedKFold(
        n_splits=args.folds,
        shuffle=True,
        random_state=SEED,
    )

    # Counts how many times each sample is used as a test sample.
    test_usage = np.zeros(len(y), dtype=int)

    all_rows = []

    for fold, (train_idx, test_idx) in enumerate(
        skf.split(X, y),
        start=1,
    ):
        # No overlap between train and test indices.
        assert len(set(train_idx) & set(test_idx)) == 0

        # Each test sample should eventually appear exactly once.
        test_usage[test_idx] += 1

        print("\n" + "=" * 70)
        print(f"FOLD {fold}/{args.folds}")
        print("=" * 70)

        print(f"Train samples: {len(train_idx)}")
        print(f"Test samples : {len(test_idx)}")

        train_class_counts = np.bincount(
            y[train_idx],
            minlength=len(GENRES),
        )

        test_class_counts = np.bincount(
            y[test_idx],
            minlength=len(GENRES),
        )

        print(
            "Train:",
            dict(zip(GENRES, train_class_counts)),
        )

        print(
            "Test :",
            dict(zip(GENRES, test_class_counts)),
        )

        fold_rows = evaluate_fold(
            X=X,
            y=y,
            train_idx=train_idx,
            test_idx=test_idx,
            fold=fold,
            epochs=args.epochs,
            dil1=args.dil1,
            dil2=args.dil2,
            include_raw=not args.skip_raw,
        )

        all_rows.extend(fold_rows)

    # ---------------------------------------------------------------
    # Global CV integrity checks
    # ---------------------------------------------------------------
    assert np.all(
        test_usage == 1
    ), "Each sample must appear as test exactly once."

    df = pd.DataFrame(all_rows)

    rows_per_fold = 9 if args.skip_raw else 13
    expected_rows = args.folds * rows_per_fold

    assert len(df) == expected_rows

    assert df["test_accuracy"].notna().all()

    assert np.isfinite(
        df["test_accuracy"]
    ).all()

    # Check all requested fold numbers are present.
    assert sorted(df["fold"].unique()) == list(
        range(1, args.folds + 1)
    )

    # ---------------------------------------------------------------
    # Save fold-level results
    # ---------------------------------------------------------------
    output_dir = os.path.dirname(args.output)

    if not output_dir:
        output_dir = "."

    os.makedirs(output_dir, exist_ok=True)

    df.to_csv(
        args.output,
        index=False,
    )

    # ---------------------------------------------------------------
    # Calculate mean +/- standard deviation
    # ---------------------------------------------------------------
    summary = (
        df.groupby(
            [
                "feature_representation",
                "algorithm",
                "dil1",
                "dil2",
            ]
        )["test_accuracy"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(
            columns={
                "count": "n_folds",
            }
        )
    )

    # Example:
    # results/cv_folds.csv
    # ->
    # results/cv_summary.csv
    summary_output = os.path.splitext(args.output)[0]
    summary_output = (
        summary_output.replace("_folds", "_summary")
        + ".csv"
    )

    summary.to_csv(
        summary_output,
        index=False,
    )

    print("\n" + "=" * 70)
    print("CV COMPLETE")
    print("=" * 70)

    print(
        f"Fold results saved to: {args.output}"
    )

    print(
        f"Summary saved to     : {summary_output}"
    )

    print("\nSummary:")

    print(
        summary.to_string(
            index=False,
            float_format=lambda v: f"{v:.4f}",
        )
    )


if __name__ == "__main__":
    main()
