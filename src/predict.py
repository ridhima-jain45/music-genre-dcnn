"""Demo: predict the genre of an audio file with the trained dilated CNN.

Usage:  python predict.py path/to/song.wav [--weights path/to/dcnn.pt]
"""
import argparse
import os
import numpy as np
import torch

from data import extract_mfcc, GENRES
from dcnn import DCNN, DEVICE


def default_weights():
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (os.path.join(here, "..", "results", "dcnn.pt"),   # repo layout
              os.path.join(here, "results", "dcnn.pt"),
              "/kaggle/working/results/dcnn.pt"):               # Kaggle
        if os.path.exists(p):
            return p
    return "results/dcnn.pt"


def predict(path, weights):
    ckpt = torch.load(weights, map_location=DEVICE)
    model = DCNN().to(DEVICE)
    model.load_state_dict(ckpt["state"])
    model.eval()

    mfcc = extract_mfcc(path)                                   # (20, 1290), same as training
    x = (mfcc[None] - ckpt["mean"].numpy()) / ckpt["std"].numpy()   # training-set normalisation
    x = torch.tensor(x, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
    return probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", help="path to a .wav file (30 s works best)")
    ap.add_argument("--weights", default=default_weights())
    args = ap.parse_args()

    if not os.path.exists(args.audio):
        raise SystemExit(f"Audio file not found: {args.audio}")

    probs = predict(args.audio, args.weights)
    best = int(np.argmax(probs))

    print(f"\nFile: {os.path.basename(args.audio)}")
    print(f"Predicted genre: {GENRES[best].upper()}  ({probs[best]:.0%} confidence)\n")
    for i in np.argsort(-probs):
        print(f"  {GENRES[i]:10s} {probs[i]:6.1%}  {'#' * int(probs[i] * 40)}")


if __name__ == "__main__":
    main()
