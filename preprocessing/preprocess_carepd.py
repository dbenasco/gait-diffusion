"""
Stage 2: angles.pkl → training-ready arrays.

Reads the geometric joint angles produced by smpl_to_angles.py, applies
subject-level train/eval splits, sliding-window segmentation, mirror
augmentation, and normalization.

Output:
  data/carepd/train_gait_angles.npy  (N_train, 8, SEQ_LEN)
  data/carepd/eval_gait_angles.npy   (N_eval,  8, SEQ_LEN)
  data/carepd/norm_params.pt         per-channel mean/std from train split

Run with:
  python -m preprocessing.preprocess_carepd
"""

import os
import sys
import pickle

import numpy as np
import torch
# Note: scipy not used — np.interp handles resampling

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    CAREPD_LABELED_COHORTS, ANGLES_PATH,
    PROCESSED_DATA_PATH, PROCESSED_LABELS_PATH,
    TRAIN_DATA_PATH, TRAIN_LABELS_PATH, TRAIN_LATERALITY_PATH,
    EVAL_DATA_PATH, EVAL_LABELS_PATH, EVAL_LATERALITY_PATH,
    NORM_PARAMS_PATH, FOLDS_DIR,
    SEQ_LEN, WINDOW_STRIDE, UPDRS_CLASSES, UPDRS_MAX_SCORE, N_CHANNELS,
)


# ── Train/eval splits ─────────────────────────────────────────────────────────

def load_fixed_splits() -> dict:
    SPLIT_FILES = {
        "BMCLab":   "BMCLab_fixed.pkl",
        "PD-GaM":   "PD-GaM_authors_fixed.pkl",
        "3DGait":   "3DGait_fixed.pkl",
        "T-SDU-PD": "T-SDU-PD_PD_fixed.pkl",
    }
    splits = {}
    for cohort in CAREPD_LABELED_COHORTS:
        with open(os.path.join(FOLDS_DIR, SPLIT_FILES[cohort]), 'rb') as f:
            fold = pickle.load(f)[1]
        splits[cohort] = {
            'train': set(str(s) for s in fold['train']),
            'eval':  set(str(s) for s in fold['eval']),
        }
    return splits


# ── Laterality ────────────────────────────────────────────────────────────────

def detect_laterality(angles: np.ndarray, updrs: int) -> int:
    """
    Uses knee flex channels (4=L_Knee, 5=R_Knee).
    0 = Left impaired, 1 = Right impaired, 2 = Symmetric.
    """
    var_l = np.var(angles[:, 4])
    var_r = np.var(angles[:, 5])
    var_max = max(var_l, var_r) + 1e-8
    if updrs == 0 or abs(var_l - var_r) / var_max < 0.20:
        return 2
    return 1 if var_l > var_r else 0


def _flip_lat(side: int) -> int:
    return {0: 1, 1: 0, 2: 2}[side]


# ── Mirror augmentation ───────────────────────────────────────────────────────

def mirror_window(angles: np.ndarray) -> np.ndarray:
    """
    Swap L↔R channels.
    Layout: [L_Hip_Flex, L_Hip_Abd, R_Hip_Flex, R_Hip_Abd,
             L_Knee, R_Knee, L_Ankle, R_Ankle]
    """
    m = angles.copy()
    m[:, 0:2], m[:, 2:4] = angles[:, 2:4].copy(), angles[:, 0:2].copy()
    m[:, 4],   m[:, 5]   = angles[:, 5].copy(),   angles[:, 4].copy()
    m[:, 6],   m[:, 7]   = angles[:, 7].copy(),   angles[:, 6].copy()
    return m


# ── Sliding window ────────────────────────────────────────────────────────────

def to_windows(angles: np.ndarray, updrs: int):
    """
    Slice (T, 8) angles into overlapping (SEQ_LEN, 8) windows.
    Returns list of windows and the laterality label for the walk.
    """
    T = angles.shape[0]
    if T < SEQ_LEN:
        return [], None

    lat = detect_laterality(angles, updrs)
    windows, start = [], 0
    while start + SEQ_LEN <= T:
        windows.append(angles[start:start + SEQ_LEN])
        start += WINDOW_STRIDE
    return windows, lat


# ── Main preprocessing ────────────────────────────────────────────────────────

def preprocess():
    if not os.path.exists(ANGLES_PATH):
        print(f"ERROR: {ANGLES_PATH} not found.")
        print("Run smpl_to_angles.py first.")
        return

    print("=" * 60)
    print("CARE-PD → UPDRS-DiT  Stage 2: windowing + normalization")
    print(f"  Source  : {ANGLES_PATH}")
    print(f"  Window  : {SEQ_LEN} frames  stride: {WINDOW_STRIDE}")
    print(f"  Channels: {N_CHANNELS}")
    print("=" * 60)

    with open(ANGLES_PATH, 'rb') as f:
        all_angles = pickle.load(f)

    splits = load_fixed_splits()

    train_wins, train_labels, train_lat = [], [], []
    eval_wins,  eval_labels,  eval_lat  = [], [], []
    n_short = 0

    for cohort in CAREPD_LABELED_COHORTS:
        if cohort not in all_angles:
            print(f"  {cohort}: not in angles.pkl, skipping.")
            continue

        train_ids = splits[cohort]['train']
        eval_ids  = splits[cohort]['eval']
        n_train, n_eval = 0, 0

        for sid, subject_walks in all_angles[cohort].items():
            if sid in train_ids:
                win_buf, lab_buf, lat_buf = train_wins, train_labels, train_lat
                is_train = True
            elif sid in eval_ids:
                win_buf, lab_buf, lat_buf = eval_wins, eval_labels, eval_lat
                is_train = False
            else:
                continue

            for wid, walk_data in subject_walks.items():
                angles = walk_data['angles']   # (T, 8)
                updrs  = walk_data['updrs']

                windows, lat = to_windows(angles, updrs)
                if not windows:
                    n_short += 1
                    continue

                for w in windows:
                    win_buf.append(w.T)                    # (8, SEQ_LEN)
                    lab_buf.append(updrs)
                    lat_buf.append(lat)

                    win_buf.append(mirror_window(w).T)
                    lab_buf.append(updrs)
                    lat_buf.append(_flip_lat(lat))

                if is_train: n_train += 1
                else:        n_eval  += 1

        print(f"  {cohort}: {n_train} train walks, {n_eval} eval walks")

    if n_short:
        print(f"  Skipped {n_short} walks shorter than {SEQ_LEN} frames.")

    train_wins   = np.array(train_wins,   dtype=np.float32)
    train_labels = np.array(train_labels, dtype=np.int64)
    train_lat    = np.array(train_lat,    dtype=np.int64)
    eval_wins    = np.array(eval_wins,    dtype=np.float32)
    eval_labels  = np.array(eval_labels,  dtype=np.int64)
    eval_lat     = np.array(eval_lat,     dtype=np.int64)

    all_wins   = np.concatenate([train_wins, eval_wins])
    all_labels = np.concatenate([train_labels, eval_labels])
    all_lat    = np.concatenate([train_lat, eval_lat])

    lat_names = {0: "Left impaired", 1: "Right impaired", 2: "Symmetric"}
    for name, wins, labs, lats in [
        ("Train", train_wins, train_labels, train_lat),
        ("Eval",  eval_wins,  eval_labels,  eval_lat),
        ("Total", all_wins,   all_labels,   all_lat),
    ]:
        print(f"\n{name}: {len(wins)} windows  {wins.shape}")
        for cls in range(UPDRS_CLASSES):
            n = (labs == cls).sum()
            print(f"  UPDRS {cls}: {n} ({100*n/len(labs):.1f}%)")
        for side, sname in lat_names.items():
            n = (lats == side).sum()
            print(f"  {sname}: {n} ({100*n/len(lats):.1f}%)")

    # Save arrays
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    np.save(PROCESSED_DATA_PATH,   all_wins)
    np.save(PROCESSED_LABELS_PATH, all_labels)
    np.save(TRAIN_DATA_PATH,       train_wins)
    np.save(TRAIN_LABELS_PATH,     train_labels)
    np.save(TRAIN_LATERALITY_PATH, train_lat)
    np.save(EVAL_DATA_PATH,        eval_wins)
    np.save(EVAL_LABELS_PATH,      eval_labels)
    np.save(EVAL_LATERALITY_PATH,  eval_lat)

    # Normalization params from train split only
    train_tensor = torch.from_numpy(train_wins).float()
    avg_mean = train_tensor.mean(dim=(0, 2)).view(-1, 1)
    avg_std  = train_tensor.std(dim=(0, 2)).view(-1, 1)
    avg_std[avg_std < 1e-6] = 1.0
    torch.save({'mean': avg_mean, 'std': avg_std}, NORM_PARAMS_PATH)

    print(f"\nSaved train : {TRAIN_DATA_PATH}  {train_wins.shape}")
    print(f"Saved eval  : {EVAL_DATA_PATH}  {eval_wins.shape}")
    print(f"Saved norms : {NORM_PARAMS_PATH}")
    print("\nDone — ready to train.")


if __name__ == "__main__":
    preprocess()
