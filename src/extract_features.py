"""Use the trained DCNN as a feature extractor for the classical classifiers (paper Table V.3)."""
import os
import warnings
import numpy as np
import pandas as pd
import torch
from sklearn.exceptions import ConvergenceWarning

from data import load_split
from dcnn import DCNN, DEVICE
from baselines import run_classifiers, to_rows, fmt

warnings.filterwarnings("ignore", category=ConvergenceWarning)
RESULTS_DIR = os.environ.get("RESULTS_DIR", "/kaggle/working/results")
ALGS = ["LR", "GDA", "RF", "SVM"]

# Table V.3 test accuracies from the paper
PAPER_TEST = {
    "layer 1 (320)": {"LR": 0.87, "GDA": 0.61, "RF": 0.86, "SVM": 0.87},
    "layer 2 (48)":  {"LR": 0.84, "GDA": 0.80, "RF": 0.87, "SVM": 0.84},
}


@torch.no_grad()
def extract(model, X):
    model.eval()   # dropout off, batch-norm uses running stats
    logits, f1, f2 = model(X.to(DEVICE), return_features=True)
    return f1.cpu().numpy(), f2.cpu().numpy(), logits.argmax(1).cpu().numpy()


def main():
    s = load_split()
    ckpt = torch.load(os.path.join(RESULTS_DIR, "dcnn.pt"))
    model = DCNN().to(DEVICE)
    model.load_state_dict(ckpt["state"])
    mean, std = ckpt["mean"].numpy(), ckpt["std"].numpy()
    norm = lambda a: torch.tensor((a - mean) / std, dtype=torch.float32)

    tr1, tr2, _ = extract(model, norm(s["X_train"]))
    te1, te2, te_pred = extract(model, norm(s["X_test"]))
    ytr, yte = s["y_train"], s["y_test"]
    print(f"Feature shapes: layer 1 {tr1.shape}, layer 2 {tr2.shape}")
    print(f"DCNN's own test accuracy (for reference): {(te_pred == yte).mean():.3f}")

    feature_sets = {
        "layer 1 (320)": (tr1, te1),
        "layer 2 (48)": (tr2, te2),
        "layer 1+2 (368)": (np.hstack([tr1, tr2]), np.hstack([te1, te2])),
    }

    rows = []
    for mode, scale in (("paper", False), ("improved", True)):
        for name, (a, b) in feature_sets.items():
            res = run_classifiers(a, ytr, b, yte, scale=scale)
            res = {k: v for k, v in res.items() if k in ALGS}
            rows += to_rows(name, mode, res)
    df = pd.DataFrame(rows)

    for mode, title in (("paper", "MODE: PAPER (features as-is)"),
                        ("improved", "MODE: IMPROVED (standardised features)")):
        print("\n" + "=" * 60 + f"\n{title}\n" + "=" * 60)
        print(fmt(df[df["mode"] == mode].drop(columns="mode")))

    print("\n" + "=" * 60 + "\nREPRODUCTION CHECK: test accuracy vs paper Table V.3\n" + "=" * 60)
    t = df[(df["mode"] == "paper") & (df["split"] == "test")].set_index("features")
    rows = []
    for feat, ref in PAPER_TEST.items():
        for alg in ALGS:
            rows.append({"features": feat, "algorithm": alg, "paper": ref[alg],
                         "ours": t.loc[feat, alg], "diff": t.loc[feat, alg] - ref[alg]})
    print(fmt(pd.DataFrame(rows)))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(os.path.join(RESULTS_DIR, "cnn_features.csv"), index=False)
    print(f"\nSaved to {RESULTS_DIR}/cnn_features.csv")


if __name__ == "__main__":
    main()
