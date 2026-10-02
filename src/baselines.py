"""Classical baselines (LR, GDA, RF, SVM) on raw flattened MFCCs and PCA-50.

paper    : features used as-is (reproduction of the paper)
improved : StandardScaler fitted on train only
Also tests SVM with gamma='auto' (old sklearn default) to see if it reproduces
the paper's SVM collapse (~0.2-0.3 test accuracy).
"""
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from data import load_split, SEED

warnings.filterwarnings("ignore", category=ConvergenceWarning)
RESULTS_DIR = os.environ.get("RESULTS_DIR", "/kaggle/working/results")

# Table V.3 test accuracies from the paper
PAPER_TEST = {
    "raw (25800)": {"LR": 0.77, "GDA": 0.77, "RF": 0.76, "SVM": 0.28},
    "PCA-50":      {"LR": 0.81, "GDA": 0.79, "RF": 0.77, "SVM": 0.21},
}


def make_classifiers():
    return {
        "LR": LogisticRegression(max_iter=2000),
        "GDA": LinearDiscriminantAnalysis(solver="svd"),   # shared covariance
        "RF": RandomForestClassifier(max_depth=7, n_estimators=200, random_state=SEED),
        "SVM": SVC(kernel="rbf", gamma="scale", random_state=SEED),
        "SVM (auto)": SVC(kernel="rbf", gamma="auto", random_state=SEED),
    }


def run_classifiers(Xtr, ytr, Xte, yte, scale=False):
    """Fit all classifiers; return {name: (train_acc, test_acc)}."""
    if scale:
        sc = StandardScaler().fit(Xtr)          # fit on train only -> no leakage
        Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    out = {}
    for name, clf in make_classifiers().items():
        clf.fit(Xtr, ytr)
        out[name] = (accuracy_score(ytr, clf.predict(Xtr)),
                     accuracy_score(yte, clf.predict(Xte)))
    return out


def to_rows(feature_name, mode, res):
    rows = []
    for split, k in (("train", 0), ("test", 1)):
        row = {"features": feature_name, "mode": mode, "split": split}
        row.update({alg: v[k] for alg, v in res.items()})
        rows.append(row)
    return rows


def fmt(df):
    return df.to_string(index=False, float_format=lambda v: f"{v:.3f}")


def main():
    s = load_split()
    Xtr = s["X_train"].reshape(len(s["X_train"]), -1)
    Xte = s["X_test"].reshape(len(s["X_test"]), -1)
    ytr, yte = s["y_train"], s["y_test"]
    print(f"Flattened shapes: train {Xtr.shape}, test {Xte.shape}")

    all_rows = []
    for mode, scale in (("paper", False), ("improved", True)):
        print(f"Running mode = {mode} ...")
        all_rows += to_rows("raw (25800)", mode,
                            run_classifiers(Xtr, ytr, Xte, yte, scale=scale))
        a, b = Xtr, Xte
        if scale:
            sc = StandardScaler().fit(Xtr)
            a, b = sc.transform(Xtr), sc.transform(Xte)
        pca = PCA(n_components=50, random_state=SEED).fit(a)   # train only
        all_rows += to_rows("PCA-50", mode,
                            run_classifiers(pca.transform(a), ytr, pca.transform(b), yte))

    df = pd.DataFrame(all_rows)

    titles = {"paper": "MODE: PAPER (no scaling)",
              "improved": "MODE: IMPROVED (standardised features)"}
    for mode in ("paper", "improved"):
        sub = df[df["mode"] == mode].drop(columns="mode")
        print("\n" + "=" * 70 + f"\n{titles[mode]}\n" + "=" * 70)
        print(fmt(sub))

    # side-by-side with the paper (paper mode, test accuracy)
    print("\n" + "=" * 70 + "\nREPRODUCTION CHECK: test accuracy vs paper Table V.3\n" + "=" * 70)
    rows = []
    t = df[(df["mode"] == "paper") & (df["split"] == "test")].set_index("features")
    for feat in PAPER_TEST:
        for alg in ["LR", "GDA", "RF", "SVM", "SVM (auto)"]:
            ref = PAPER_TEST[feat]["SVM" if alg == "SVM (auto)" else alg]
            rows.append({"features": feat, "algorithm": alg, "paper": ref,
                         "ours": t.loc[feat, alg], "diff": t.loc[feat, alg] - ref})
    print(fmt(pd.DataFrame(rows)))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(os.path.join(RESULTS_DIR, "baselines.csv"), index=False)
    print(f"\nSaved to {RESULTS_DIR}/baselines.csv")


if __name__ == "__main__":
    main()
