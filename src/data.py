%%writefile data.py
"""Data loading: GTZAN (5 genres) -> MFCC arrays (20 x 1290) -> fixed stratified 80/20 split.

Run once:  python src/data.py
Afterwards other scripts call load_split() which reads the cached .npz file.
"""
import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split

# ---- configuration -------------------------------------------------------
DATA_DIR = os.environ.get(
    "GTZAN_DIR",
    "/kaggle/input/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/Data/genres_original",
)
CACHE_DIR = os.environ.get("CACHE_DIR", "/kaggle/working/cache")
GENRES = ["blues", "classical", "hiphop", "metal", "pop"]
SR = 22050          # sampling rate
N_MFCC = 20         # MFCC coefficients per frame
N_FRAMES = 1290     # time frames kept per clip -> 20 * 1290 = 25,800 features
SEED = 42
TEST_SIZE = 0.2
CACHE_FILE = os.path.join(CACHE_DIR, "mfcc_5genres.npz")


def extract_mfcc(path):
    """Load one wav and return a (N_MFCC, N_FRAMES) MFCC matrix (cropped/padded)."""
    y, sr = librosa.load(path, sr=SR, duration=30)
    m = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    if m.shape[1] >= N_FRAMES:
        m = m[:, :N_FRAMES]
    else:
        m = np.pad(m, ((0, 0), (0, N_FRAMES - m.shape[1])), mode="edge")
    return m.astype(np.float32)


def build_dataset(force=False):
    """Extract MFCCs for all clips and cache them. Returns X (N,20,1290), y (N,), paths."""
    if os.path.exists(CACHE_FILE) and not force:
        d = np.load(CACHE_FILE, allow_pickle=True)
        return d["X"], d["y"], d["paths"]

    X, y, paths = [], [], []
    for label, genre in enumerate(GENRES):
        folder = os.path.join(DATA_DIR, genre)
        files = sorted(f for f in os.listdir(folder) if f.endswith(".wav"))
        for f in files:
            p = os.path.join(folder, f)
            try:
                X.append(extract_mfcc(p))
                y.append(label)
                paths.append(p)
            except Exception as e:  # skip unreadable files
                print(f"Skipping {p}: {e}")
        print(f"{genre}: {sum(1 for v in y if v == label)} clips")

    X, y, paths = np.stack(X), np.array(y), np.array(paths)
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(CACHE_FILE, X=X, y=y, paths=paths)
    print("Saved", CACHE_FILE, X.shape)
    return X, y, paths


def load_split():
    """Fixed stratified split (same for every script). Returns dict of arrays."""
    X, y, paths = build_dataset()
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=TEST_SIZE, stratify=y, random_state=SEED)
    return dict(
        X_train=X[tr], y_train=y[tr], X_test=X[te], y_test=y[te],
        paths_train=paths[tr], paths_test=paths[te],
    )


if __name__ == "__main__":
    import pandas as pd
    s = load_split()
    print("=" * 50)
    print("DATASET SUMMARY")
    print("=" * 50)
    print(f"Train: {s['X_train'].shape}   Test: {s['X_test'].shape}")
    print(f"MFCC grid per clip: {N_MFCC} coefficients x {N_FRAMES} frames "
          f"= {N_MFCC * N_FRAMES} features")
    counts = pd.DataFrame({
        "genre": GENRES,
        "train": np.bincount(s["y_train"]),
        "test": np.bincount(s["y_test"]),
    })
    counts["total"] = counts["train"] + counts["test"]
    print("\nClips per genre:")
    print(counts.to_string(index=False))
    print(f"\nCache file: {CACHE_FILE}")
