"""GDA shrinkage + hyperparameter tuning on CNN features.
Same 5 folds/seeds as cv.py, so 'default' rows should roughly match cv_summary_dilated.csv.
Tuning uses an INNER 3-fold CV on the training part only; the test fold is never touched."""
import os
import io
import time
import warnings
import contextlib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from data import build_dataset, SEED
from dcnn import DCNN, train, DEVICE
from cv import set_seed

warnings.filterwarnings("ignore")
RESULTS_DIR = os.environ.get("RESULTS_DIR", "/kaggle/working/results")


def default_models():
    return {
        "LR": LogisticRegression(max_iter=2000),
        "GDA": LinearDiscriminantAnalysis(solver="svd"),
        "GDA-shrink": LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        "RF": RandomForestClassifier(max_depth=7, n_estimators=200, random_state=SEED),
        "SVM": SVC(kernel="rbf", gamma="scale", random_state=SEED),
    }


def tuned_searches(n_feat):
    g0 = 1.0 / n_feat        # = sklearn's gamma='scale' for standardised features
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

    def gs(clf, grid):
        return GridSearchCV(make_pipeline(StandardScaler(), clf), grid,
                            cv=inner, scoring="accuracy", n_jobs=-1)
    return {
        "LR": gs(LogisticRegression(max_iter=2000),
                 {"logisticregression__C": [0.01, 0.1, 1, 10]}),
        "SVM": gs(SVC(kernel="rbf"),
                  {"svc__C": [1, 10, 100], "svc__gamma": [g0 / 4, g0, g0 * 4]}),
        "RF": gs(RandomForestClassifier(n_estimators=200, random_state=SEED),
                 {"randomforestclassifier__max_depth": [3, 5, 7, None],
                  "randomforestclassifier__min_samples_leaf": [1, 3]}),
    }


def main():
    X, y, _ = build_dataset()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows, t0 = [], time.time()

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y), 1):
        print(f"--- fold {fold}/5 ({time.time() - t0:.0f}s elapsed) ---")
        fs = SEED + fold
        set_seed(fs)
        Xtr_raw, Xte_raw, ytr, yte = X[tr_idx], X[te_idx], y[tr_idx], y[te_idx]

        # same recipe as cv.py: train-fold normalisation, inner validation split, fresh CNN
        mean = Xtr_raw.mean(axis=(0, 2), keepdims=True)
        std = Xtr_raw.std(axis=(0, 2), keepdims=True) + 1e-8
        Xt = torch.tensor((Xtr_raw - mean) / std, dtype=torch.float32)
        Xe = torch.tensor((Xte_raw - mean) / std, dtype=torch.float32)
        yt = torch.tensor(ytr, dtype=torch.long)
        itr, iva = train_test_split(np.arange(len(ytr)), test_size=0.2,
                                    stratify=ytr, random_state=fs)
        set_seed(fs)
        model = DCNN().to(DEVICE)
        with contextlib.redirect_stdout(io.StringIO()):
            train(model, Xt[itr], yt[itr], Xt[iva], yt[iva])
        model.eval()
        with torch.no_grad():
            _, f1tr, f2tr = model(Xt.to(DEVICE), return_features=True)
            _, f1te, f2te = model(Xe.to(DEVICE), return_features=True)

        feats = {"layer1_320": (f1tr.cpu().numpy(), f1te.cpu().numpy()),
                 "layer2_48": (f2tr.cpu().numpy(), f2te.cpu().numpy())}

        for fname, (a, b) in feats.items():
            for name, clf in default_models().items():
                acc = make_pipeline(StandardScaler(), clf).fit(a, ytr).score(b, yte)
                rows.append(dict(fold=fold, features=fname, algorithm=name,
                                 variant="default", test_accuracy=acc, best_params=""))
            for name, search in tuned_searches(a.shape[1]).items():
                search.fit(a, ytr)                 # inner CV on the training part only
                rows.append(dict(fold=fold, features=fname, algorithm=name,
                                 variant="tuned", test_accuracy=search.score(b, yte),
                                 best_params=str(search.best_params_)))

    df = pd.DataFrame(rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(os.path.join(RESULTS_DIR, "tuning_folds.csv"), index=False)

    g = (df.groupby(["features", "algorithm", "variant"])["test_accuracy"]
           .agg(["mean", "std"]).reset_index())
    g["txt"] = g["mean"].map("{:.3f}".format) + " ± " + g["std"].map("{:.3f}".format)
    wide = g.pivot(index=["features", "algorithm"], columns="variant", values="txt").fillna("-")
    wide.to_csv(os.path.join(RESULTS_DIR, "tuning_summary.csv"))

    print("\n" + "=" * 66 + "\nTEST ACCURACY, mean ± std over 5 folds\n" + "=" * 66)
    print(wide.to_string())

    # paired per-fold gains (more informative than comparing two means)
    p = df.pivot_table(index=["features", "algorithm", "fold"], columns="variant",
                       values="test_accuracy").dropna()
    d = (p["tuned"] - p["default"]).groupby(level=["features", "algorithm"])
    print("\nTUNED minus DEFAULT (per-fold paired)")
    print(pd.DataFrame({"mean_gain": d.mean(),
                        "folds_improved_of_5": d.apply(lambda s: int((s > 0).sum()))}).round(3))

    q = df[df["variant"] == "default"].pivot_table(
        index=["features", "fold"], columns="algorithm", values="test_accuracy")
    d = (q["GDA-shrink"] - q["GDA"]).groupby(level="features")
    print("\nGDA-shrink minus plain GDA (per-fold paired)")
    print(pd.DataFrame({"mean_gain": d.mean(),
                        "folds_improved_of_5": d.apply(lambda s: int((s > 0).sum()))}).round(3))
    print(f"\nTotal {time.time() - t0:.0f}s. Saved tuning_folds.csv and tuning_summary.csv")


if __name__ == "__main__":
    main()
