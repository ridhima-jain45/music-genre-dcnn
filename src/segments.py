"""3-second segments. Folds are made by CLIP first, then each clip is cut into 10 segments.
Segment predictions are combined per clip (CNN: mean softmax, classical: majority vote).
Same 5 folds/seeds as cv.py."""
import argparse
import io
import os
import time
import warnings
import contextlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from data import build_dataset, SEED
from dcnn import Unit, train, DEVICE
from cv import set_seed

warnings.filterwarnings("ignore")
RESULTS_DIR = os.environ.get("RESULTS_DIR", "/kaggle/working/results")
N_SEG, SEG, N_CLASSES = 10, 129, 5      # 10 x 129 = 1290 frames; 129 frames ~ 3 s


class SegDCNN(nn.Module):
    """Same design as DCNN, with pooling sized for 129-frame inputs."""
    def __init__(self, n_classes=N_CLASSES, n_mfcc=20, n_frames=SEG,
                 dil1=4, dil2=2, pool1=8, pool2=4):
        super().__init__()
        self.unit1 = Unit(n_mfcc, 8, 8, dil1, pool1)     # -> 8 x 16  = 128 features
        self.unit2 = Unit(8, 24, 4, dil2, pool2)         # -> 24 x 4  = 96 features
        with torch.no_grad():
            self.eval()
            flat = self.unit2(self.unit1(torch.zeros(2, n_mfcc, n_frames))).flatten(1).shape[1]
        self.fc = nn.Linear(flat, n_classes)

    def forward(self, x, return_features=False):
        f1 = self.unit1(x)
        f2 = self.unit2(f1)
        logits = self.fc(f2.flatten(1))
        if return_features:
            return logits, f1.flatten(1), f2.flatten(1)
        return logits


def make_segments(Xc):
    """(n, 20, 1290) -> (n*10, 20, 129). Segments stay grouped by clip (clip-major order)."""
    n = len(Xc)
    return Xc.reshape(n, 20, N_SEG, SEG).transpose(0, 2, 1, 3).reshape(n * N_SEG, 20, SEG)


def vote(seg_pred):
    p = seg_pred.reshape(-1, N_SEG)
    return np.array([np.bincount(r, minlength=N_CLASSES).argmax() for r in p])


@torch.no_grad()
def run_model(model, X):
    model.eval()
    logits, f1, f2 = model(X.to(DEVICE), return_features=True)
    return logits.cpu(), f1.cpu().numpy(), f2.cpu().numpy()


def classifiers():
    return {
        "LR": LogisticRegression(max_iter=2000),
        "GDA": LinearDiscriminantAnalysis(solver="svd"),
        "RF": RandomForestClassifier(max_depth=7, n_estimators=200, random_state=SEED),
        "SVM": SVC(kernel="rbf", gamma="scale", random_state=SEED),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--dil1", type=int, default=4)
    ap.add_argument("--dil2", type=int, default=2)
    args = ap.parse_args()

    X, y, _ = build_dataset()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)   # splits CLIPS
    seg_ids = lambda c: (c[:, None] * N_SEG + np.arange(N_SEG)).ravel()
    rows, t0 = [], time.time()

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y), 1):
        print(f"--- fold {fold}/5 ({time.time() - t0:.0f}s elapsed) ---")
        fs = SEED + fold
        set_seed(fs)
        Xtr_c, Xte_c, ytr_c, yte_c = X[tr_idx], X[te_idx], y[tr_idx], y[te_idx]

        mean = Xtr_c.mean(axis=(0, 2), keepdims=True)     # train-clip statistics only
        std = Xtr_c.std(axis=(0, 2), keepdims=True) + 1e-8
        Str = torch.tensor(make_segments((Xtr_c - mean) / std), dtype=torch.float32)
        Ste = torch.tensor(make_segments((Xte_c - mean) / std), dtype=torch.float32)
        ytr_s, yte_s = np.repeat(ytr_c, N_SEG), np.repeat(yte_c, N_SEG)
        ytr_t = torch.tensor(ytr_s, dtype=torch.long)

        # validation split by CLIP, so no song is in both train and validation
        in_clips, val_clips = train_test_split(np.arange(len(ytr_c)), test_size=0.2,
                                               stratify=ytr_c, random_state=fs)
        itr, iva = seg_ids(in_clips), seg_ids(val_clips)

        set_seed(fs)
        model = SegDCNN(dil1=args.dil1, dil2=args.dil2).to(DEVICE)
        with contextlib.redirect_stdout(io.StringIO()):
            best_val, best_ep = train(model, Str[itr], ytr_t[itr], Str[iva], ytr_t[iva],
                                      epochs=args.epochs)

        logits, _, f2te = run_model(model, Ste)
        _, f1te, _ = run_model(model, Ste)
        seg_acc = (logits.argmax(1).numpy() == yte_s).mean()
        probs = torch.softmax(logits, 1).numpy().reshape(-1, N_SEG, N_CLASSES).mean(1)
        clip_acc = (probs.argmax(1) == yte_c).mean()
        print(f"fold {fold}: best_epoch={best_ep} val={best_val:.3f} "
              f"segment_acc={seg_acc:.3f} CLIP_acc={clip_acc:.3f}")
        rows.append(dict(fold=fold, features="dcnn", algorithm="DCNN",
                         level="segment", test_accuracy=seg_acc))
        rows.append(dict(fold=fold, features="dcnn", algorithm="DCNN",
                         level="clip", test_accuracy=clip_acc))

        _, f1tr, f2tr = run_model(model, Str)
        for fname, (a, b) in {"layer1_128": (f1tr, f1te), "layer2_96": (f2tr, f2te)}.items():
            for name, clf in classifiers().items():
                pipe = make_pipeline(StandardScaler(), clf).fit(a, ytr_s)
                rows.append(dict(fold=fold, features=fname, algorithm=name, level="clip",
                                 test_accuracy=(vote(pipe.predict(b)) == yte_c).mean()))

    df = pd.DataFrame(rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(os.path.join(RESULTS_DIR, "segments_folds.csv"), index=False)
    g = (df.groupby(["features", "algorithm", "level"])["test_accuracy"]
           .agg(["mean", "std"]).reset_index())
    g.to_csv(os.path.join(RESULTS_DIR, "segments_summary.csv"), index=False)
    print("\n" + "=" * 66 + "\nSEGMENT MODEL: 5-fold CV, mean ± std\n" + "=" * 66)
    print(g.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nTotal {time.time() - t0:.0f}s. Saved segments_folds.csv and segments_summary.csv")


if __name__ == "__main__":
    main()
