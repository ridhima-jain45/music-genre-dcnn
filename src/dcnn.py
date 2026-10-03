"""Dilated CNN: 2 x [Conv1d(dilated) -> ReLU -> AvgPool -> Dropout -> BatchNorm] -> Linear(5)."""
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from data import load_split, GENRES, SEED

RESULTS_DIR = os.environ.get("RESULTS_DIR", "/kaggle/working/results")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Unit(nn.Module):
    def __init__(self, in_ch, filters, kernel, dilation, pool, p_drop=0.5):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, filters, kernel, dilation=dilation, padding="same")
        self.act = nn.ReLU()
        self.pool = nn.AvgPool1d(pool)
        self.drop = nn.Dropout(p_drop)
        self.bn = nn.BatchNorm1d(filters)

    def forward(self, x):
        return self.bn(self.drop(self.pool(self.act(self.conv(x)))))


class DCNN(nn.Module):
    def __init__(self, n_classes=5, n_mfcc=20, n_frames=1290):
        super().__init__()
        # unit 1: 8 filters, kernel 8, dilation 4, pool 32 -> 8 * 40 = 320 features
        self.unit1 = Unit(n_mfcc, 8, 8, 4, 32)
        # unit 2: 24 filters, kernel 4, dilation 2, pool 16 -> 24 * 2 = 48 features
        self.unit2 = Unit(8, 24, 4, 2, 16)
        with torch.no_grad():
            self.eval()
            flat = self.unit2(self.unit1(torch.zeros(2, n_mfcc, n_frames))).flatten(1).shape[1]
        self.fc = nn.Linear(flat, n_classes)

    def forward(self, x, return_features=False):
        f1 = self.unit1(x)
        f2 = self.unit2(f1)
        logits = self.fc(f2.flatten(1))   # softmax is inside CrossEntropyLoss
        if return_features:
            return logits, f1.flatten(1), f2.flatten(1)
        return logits


@torch.no_grad()
def accuracy(model, X, y):
    model.eval()
    pred = model(X.to(DEVICE)).argmax(1).cpu()
    return (pred == y).float().mean().item(), pred.numpy()


def train(model, Xtr, ytr, Xva, yva, epochs=100, lr=1e-3, bs=32):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    best_val, best_ep, best_state = -1.0, 0, None
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), bs):
            idx = perm[i:i + bs]
            xb, yb = Xtr[idx].to(DEVICE), ytr[idx].to(DEVICE)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        tr_acc, _ = accuracy(model, Xtr, ytr)
        va_acc, _ = accuracy(model, Xva, yva)
        if va_acc > best_val:   # pick the best epoch on VALIDATION, never on test
            best_val = va_acc
            best_ep = ep
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if ep % 10 == 0 or ep == 1:
            print(f"epoch {ep:3d}  train {tr_acc:.3f}  val {va_acc:.3f}")
    model.load_state_dict(best_state)
    return best_val, best_ep


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    s = load_split()

    # normalise each MFCC coefficient using TRAIN statistics only
    mean = s["X_train"].mean(axis=(0, 2), keepdims=True)
    std = s["X_train"].std(axis=(0, 2), keepdims=True) + 1e-8
    norm = lambda a: torch.tensor((a - mean) / std, dtype=torch.float32)
    X, y = norm(s["X_train"]), torch.tensor(s["y_train"])
    Xte, yte = norm(s["X_test"]), torch.tensor(s["y_test"])

    # hold out 20% of train as validation (the paper tuned on test, which leaks)
    tr, va = train_test_split(np.arange(len(y)), test_size=0.2,
                              stratify=y.numpy(), random_state=SEED)
    model = DCNN().to(DEVICE)
    print("device:", DEVICE, "| layer-2 feature size:", model.fc.in_features)
    best_val, best_ep = train(model, X[tr], y[tr], X[va], y[va])

    tr_acc, _ = accuracy(model, X[tr], y[tr])    # clips it trained on
    va_acc, _ = accuracy(model, X[va], y[va])    # validation clips
    te_acc, pred = accuracy(model, Xte, yte)
    print(f"\nbest epoch {best_ep} | train {tr_acc:.3f} | val {va_acc:.3f} | TEST {te_acc:.3f}")
    cm = confusion_matrix(yte.numpy(), pred)
    print("genres:", GENRES)
    print(cm)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    torch.save({"state": model.state_dict(),
                "mean": torch.tensor(mean), "std": torch.tensor(std)},
               os.path.join(RESULTS_DIR, "dcnn.pt"))
    np.save(os.path.join(RESULTS_DIR, "dcnn_confusion.npy"), cm)


if __name__ == "__main__":
    main()
