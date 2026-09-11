"""
Compute the per-UPDRS-class ROM targets (degrees) from real H3D train+eval data.

Prints a `TARGET_ROM_DEG` tensor for `config.py`, with left/right channels
averaged to symmetric targets (the ROM loss operates on the per-class batch
mean, which mixes left- and right-impaired windows).

Usage:
    python -m preprocessing.compute_rom_targets
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    TRAIN_DATA_PATH, TRAIN_LABELS_PATH, EVAL_DATA_PATH, EVAL_LABELS_PATH,
    UPDRS_CLASSES, UPDRS_JOINT_NAMES,
)
from h3d_bridge import h3d_to_angles

CLS_NAMES = {0: "Normal", 1: "Mild", 2: "Moderate", 3: "Severe"}


def main():
    data = np.concatenate([np.load(TRAIN_DATA_PATH), np.load(EVAL_DATA_PATH)], axis=0)
    labels = np.concatenate([np.load(TRAIN_LABELS_PATH).ravel(),
                             np.load(EVAL_LABELS_PATH).ravel()]).astype(int)

    feat = torch.as_tensor(data, dtype=torch.float32)
    leg_ang = h3d_to_angles(feat, sagittal_only=True)                      # (N, 6, T)
    rom = (leg_ang.max(dim=-1).values - leg_ang.min(dim=-1).values).numpy()  # (N, 6)

    print(f"{'':18}" + "".join(f"{'U' + str(c):>14}" for c in range(UPDRS_CLASSES)))
    per_class = []
    for c in range(UPDRS_CLASSES):
        per_class.append(rom[labels == c].mean(axis=0))

    for j, jname in enumerate(UPDRS_JOINT_NAMES):
        print(f"  {jname:<16}" + "".join(f"{per_class[c][j]:>14.2f}" for c in range(UPDRS_CLASSES)))

    print("\nTARGET_ROM_DEG = torch.tensor([")
    for c in range(UPDRS_CLASSES):
        hip = (per_class[c][0] + per_class[c][1]) / 2
        knee = (per_class[c][2] + per_class[c][3]) / 2
        ankle = (per_class[c][4] + per_class[c][5]) / 2
        sym = [hip, hip, knee, knee, ankle, ankle]
        print(f"    [{', '.join(f'{v:.2f}' for v in sym)}],  # UPDRS {c} ({CLS_NAMES[c]})")
    print("])")


if __name__ == "__main__":
    main()
