"""
GAITGen-style distance metrics (paper Table 3).

  - AVE  : Average Variance Error (per-joint temporal variance mismatch)
  - AAMD : Absolute Arm-swing Mean Difference
  - ASMD : Absolute Stooped-posture Mean Difference

Definitions follow GAITGen (arXiv:2503.22397, Appendix H), computed on
recovered full-body joint positions (metres).

Usage:
    python -m evaluation.gaitgen [--gen_path PATH]
"""

import os
import sys
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import EVAL_DATA_PATH, EVAL_LABELS_PATH, GEN_OUTPUT_PATH, N_CHANNELS
from h3d_bridge import ave, aamd, asmd

CLS_NAMES = {0: "Normal", 1: "Mild", 2: "Moderate", 3: "Severe"}


def _nct(a):
    return a if a.ndim == 3 and a.shape[1] == N_CHANNELS else a.transpose(0, 2, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_path", type=str, default=None)
    args = parser.parse_args()
    gen_path = args.gen_path or GEN_OUTPUT_PATH

    real = _nct(np.load(EVAL_DATA_PATH))
    real_labels = np.load(EVAL_LABELS_PATH)
    synth = _nct(np.load(gen_path))
    synth_labels = np.load(gen_path.replace(".npy", "_labels.npy"))

    real_by_cls = {int(c): real[real_labels == c] for c in np.unique(real_labels)}
    synth_by_cls = {int(c): synth[synth_labels == c] for c in np.unique(synth_labels)}

    print("=" * 56)
    print("  GAITGen distance metrics (lower is better)")
    print("=" * 56)
    print(f"  {'Class':<16} {'Real':>8} {'Synth':>8}")
    print("  " + "-" * 34)
    for c in sorted(set(real_by_cls) & set(synth_by_cls)):
        print(f"  UPDRS {c} {CLS_NAMES[c]:<8} {len(real_by_cls[c]):>8} {len(synth_by_cls[c]):>8}")

    ave_overall, _ = ave(real, synth)
    print(f"\n  AVE  (all 22 joints):  {ave_overall:.5f}")
    print(f"  AAMD (arm swing):      {aamd(real_by_cls, synth_by_cls):.5f}")
    print(f"  ASMD (stooped posture):{asmd(real_by_cls, synth_by_cls):.5f}")


if __name__ == "__main__":
    main()
