"""
compute_fullbody_targets.py
--------------------------------
Compute TARGET_ARM_SWING / TARGET_TRUNK_INCL_DEG per UPDRS class from real H3D
train+eval data (mirrors compute_rom_targets.py's approach for the 6 leg angles).

Requires config.py to point TRAIN/EVAL_DATA_PATH at the H3D
.npy files, and that preprocess_carepd_h3d.py has already been run.

Usage:
    python -m preprocessing.compute_fullbody_targets
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    TRAIN_DATA_PATH, TRAIN_LABELS_PATH,
    EVAL_DATA_PATH, EVAL_LABELS_PATH,
    USE_H3D, UPDRS_CLASSES, UPDRS_JOINT_NAMES,
)
from h3d_bridge import (
    h3d_to_angles, h3d_to_positions22, arm_swing_range, trunk_inclination,
)


def main():
    if not USE_H3D:
        print("ERROR: set USE_H3D=1 — this script only makes sense for the H3D data paths.")
        return

    data   = np.concatenate([np.load(TRAIN_DATA_PATH), np.load(EVAL_DATA_PATH)], axis=0)  # (N,263,T) raw
    labels = np.concatenate([np.load(TRAIN_LABELS_PATH).ravel(), np.load(EVAL_LABELS_PATH).ravel()]).astype(int)

    feat = torch.as_tensor(data, dtype=torch.float32)
    pos = h3d_to_positions22(feat).numpy()   # (N, T, 22, 3)

    arm = arm_swing_range(pos)        # (N,) meters
    trunk = trunk_inclination(pos)    # (N,) degrees

    # ROM targets (6 sagittal leg angles, same mechanism as training/train_dit.py)
    leg_ang   = h3d_to_angles(feat, sagittal_only=True)  # (N, 6, T)
    pred_rom  = leg_ang.max(dim=-1).values - leg_ang.min(dim=-1).values  # (N, 6)
    rom_means = pred_rom.numpy()  # (N, 6)

    cls_names = {0: "Normal", 1: "Mild", 2: "Moderate", 3: "Severe"}
    arm_means, trunk_means, rom_per_class = [], [], []
    arm_stds, trunk_stds, rom_stds = [], [], []
    print("=" * 60)
    print(f"{'':18}" + "".join(f"{'U' + str(c):>14}" for c in range(UPDRS_CLASSES)))
    for c in range(UPDRS_CLASSES):
        m = labels == c
        arm_means.append(float(arm[m].mean()))
        trunk_means.append(float(trunk[m].mean()))
        arm_stds.append(float(arm[m].std()))
        trunk_stds.append(float(trunk[m].std()))
        rom_per_class.append(rom_means[m].mean(axis=0))
        rom_stds.append(rom_means[m].std(axis=0))

    print("Arm swing (m):    " + "".join(f"{v:>14.4f}" for v in arm_means))
    print("  std:            " + "".join(f"{v:>14.4f}" for v in arm_stds))
    print("Trunk incl (deg): " + "".join(f"{v:>14.2f}" for v in trunk_means))
    print("  std:            " + "".join(f"{v:>14.2f}" for v in trunk_stds))

    print("\nROM (deg) per channel:")
    header_rom = f"{'':18}" + "".join(f"{'U' + str(c):>14}" for c in range(UPDRS_CLASSES))
    print(header_rom)
    for j, jname in enumerate(UPDRS_JOINT_NAMES):
        row = f"  {jname:<16}" + "".join(f"{rom_per_class[c][j]:>14.2f}" for c in range(UPDRS_CLASSES))
        print(row)

    # L/R symmetric targets
    sym = np.zeros((UPDRS_CLASSES, 6))
    for c in range(UPDRS_CLASSES):
        sym[c, 0] = sym[c, 1] = (rom_per_class[c][0] + rom_per_class[c][1]) / 2
        sym[c, 2] = sym[c, 3] = (rom_per_class[c][2] + rom_per_class[c][3]) / 2
        sym[c, 4] = sym[c, 5] = (rom_per_class[c][4] + rom_per_class[c][5]) / 2

    print("\n" + "=" * 60)
    print("Paste into config.py:")
    print("=" * 60)
    print("TARGET_ARM_SWING = torch.tensor([" + ", ".join(f"{v:.4f}" for v in arm_means) + "])")
    print("TARGET_TRUNK_INCL_DEG = torch.tensor([" + ", ".join(f"{v:.2f}" for v in trunk_means) + "])")
    print("TARGET_ROM_DEG = torch.tensor([")
    for c in range(UPDRS_CLASSES):
        vals = ", ".join(f"{sym[c, j]:.2f}" for j in range(6))
        print(f"    [{vals}],  # UPDRS {c} ({cls_names[c]})")
    print("])")


if __name__ == "__main__":
    main()
