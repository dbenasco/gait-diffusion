"""
Generalization / conditioning evaluation (paper Table 2 and confusion-matrix figure).

Random Forest classifier on mean-pooled 64-dim VAE latent features:
  - TRTR : Train-on-Real, Test-on-Real (10-fold CV)  → data ceiling
  - TRTS : Train-on-Real, Test-on-Synthetic          → conditioning correctness

Usage:
    python -m evaluation.generalization [--n_folds 10] [--gen_path PATH]
"""

import os
import sys
import argparse

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    EVAL_DATA_PATH, EVAL_LABELS_PATH, GEN_OUTPUT_PATH, EVAL_OUTPUT_DIR,
    VAE_MODEL_PATH, NORM_PARAMS_PATH, N_CHANNELS, LATENT_CHANNELS,
    UPDRS_CLASSES, DEVICE,
)
from training.vae_updrs import GaitVAE

CLS_NAMES = {0: "Normal", 1: "Mild", 2: "Moderate", 3: "Severe"}


@torch.no_grad()
def _latents(data_nct, vae, mean_t, std_t, batch=64):
    """(N, 263, T) → mean-pooled VAE μ (N, 64)."""
    m, s = mean_t.to(DEVICE), std_t.to(DEVICE)
    x = torch.from_numpy(data_nct.astype(np.float32))
    mus = [vae.encode(torch.clamp((x[i:i + batch].to(DEVICE) - m) / s, -4, 4))[1].cpu()
           for i in range(0, len(x), batch)]
    return torch.cat(mus, dim=0).mean(dim=-1).numpy()


def _rf():
    return RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)


def _report(tag, preds, labels):
    print(f"\n  {tag}")
    print(f"  {'Class':<16} {'Correct':>9} {'Total':>8} {'Acc %':>8}")
    print("  " + "-" * 44)
    for c in range(UPDRS_CLASSES):
        mask = labels == c
        total = int(mask.sum())
        if total == 0:
            continue
        correct = int((preds[mask] == c).sum())
        print(f"  UPDRS {c} {CLS_NAMES[c]:<8} {correct:>9} {total:>8} {100 * correct / total:>8.1f}")
    overall = 100 * float((preds == labels).mean())
    print("  " + "-" * 44)
    print(f"  {'Overall':<16} {'':>9} {'':>8} {overall:>8.1f}   (chance={100 / UPDRS_CLASSES:.1f}%)")
    return overall


def _confusion(preds, labels):
    return np.array([[int(((labels == t) & (preds == p)).sum())
                      for p in range(UPDRS_CLASSES)] for t in range(UPDRS_CLASSES)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_folds", type=int, default=10)
    parser.add_argument("--gen_path", type=str, default=None)
    args = parser.parse_args()

    gen_path = args.gen_path or GEN_OUTPUT_PATH

    state = torch.load(VAE_MODEL_PATH, map_location=DEVICE)
    vae = GaitVAE(N_CHANNELS, LATENT_CHANNELS, UPDRS_CLASSES,
                  use_prototypes="prototypes" in state).to(DEVICE)
    vae.load_state_dict(state)
    vae.eval()
    norm = torch.load(NORM_PARAMS_PATH, map_location="cpu")
    mean_t, std_t = norm["mean"].float(), norm["std"].float()

    real = np.load(EVAL_DATA_PATH)
    real_labels = np.load(EVAL_LABELS_PATH)
    synth = np.load(gen_path)
    synth_labels = np.load(gen_path.replace(".npy", "_labels.npy"))

    def _nct(a):
        return a if a.ndim == 3 and a.shape[1] == N_CHANNELS else a.transpose(0, 2, 1)

    real_lats = _latents(_nct(real), vae, mean_t, std_t)
    synth_lats = _latents(_nct(synth), vae, mean_t, std_t)

    print("=" * 60)
    print("  Generalization — Random Forest on VAE latent features")
    print("=" * 60)

    # TRTR — 10-fold cross-validation on real latents.
    trtr_preds = np.zeros(len(real_labels), dtype=int)
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    for train_idx, test_idx in skf.split(real_lats, real_labels):
        clf = _rf().fit(real_lats[train_idx], real_labels[train_idx])
        trtr_preds[test_idx] = clf.predict(real_lats[test_idx])
    trtr_overall = _report("TRTR — Train on Real, Test on Real (10-fold CV)",
                           trtr_preds, real_labels)

    # TRTS — train on real, test on synthetic.
    clf = _rf().fit(real_lats, real_labels)
    trts_preds = clf.predict(synth_lats)
    trts_overall = _report("TRTS — Train on Real, Test on Synthetic", trts_preds, synth_labels)

    print(f"\n  TRTR {trtr_overall:.1f}%  |  TRTS {trts_overall:.1f}%  "
          f"|  gap {trtr_overall - trts_overall:+.1f}pp")
    print("\n  TRTR confusion matrix (rows=true, cols=pred):")
    print(_confusion(trtr_preds, real_labels))
    print("  TRTS confusion matrix (rows=true, cols=pred):")
    print(_confusion(trts_preds, synth_labels))

    os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)
    np.save(os.path.join(EVAL_OUTPUT_DIR, "trtr_confusion.npy"), _confusion(trtr_preds, real_labels))
    np.save(os.path.join(EVAL_OUTPUT_DIR, "trts_confusion.npy"), _confusion(trts_preds, synth_labels))
    print(f"\n  Saved confusion matrices to {EVAL_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
