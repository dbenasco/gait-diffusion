"""
CARE-PD SMPL (pose + trans + beta) → HumanML3D (263-dim) training arrays.

Reads the raw CARE-PD cohort PKLs directly — the 263-dim vector needs SMPL
forward kinematics *with translation*. Per walk:
  pose(T,72) + trans(T,3) + beta(10), fps
    → resample to 30 fps → SMPL FK with translation → 22-joint world positions
    → slide 97-frame position clips (stride 48)
    → process_file each clip → 96-frame HumanML3D window (263-dim)
    → mirror augmentation in position space (X-negate + L/R swap, then process_file)

263 layout = root(4) | ric_pos(63) | rot6d(126) | local_vel(66) | foot_contact(4)

Output (channels-first, like the model input):
  data/carepd/train_gait_h3d.npy   (N_train, 263, 96)
  data/carepd/eval_gait_h3d.npy    (N_eval,  263, 96)
  data/carepd/norm_params_h3d.pt   per-channel mean/std (train split, std-floored)
  + label and pooled arrays

Usage:
    python -m preprocessing.preprocess_carepd_h3d
"""

import os
import sys
import pickle
import warnings

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import smpl_to_angles as s2a
from h3d.h3d_convert import smpl_to_positions22, positions_to_h3d, mirror_positions22
from preprocessing.splits import load_fixed_splits
from config import (
    CAREPD_RAW_DIR, CAREPD_LABELED_COHORTS, CAREPD_TARGET_FPS,
    SEQ_LEN, WINDOW_STRIDE, UPDRS_CLASSES, UPDRS_MAX_SCORE,
    H3D_STD_FLOOR,
    H3D_TRAIN_DATA_PATH, H3D_TRAIN_LABELS_PATH,
    H3D_EVAL_DATA_PATH, H3D_EVAL_LABELS_PATH,
    H3D_NORM_PARAMS_PATH,
)

CLIP_LEN = SEQ_LEN + 1  # process_file drops one frame (velocity diff)


def walk_to_windows(pose, trans, beta, fps):
    """One walk → list of (263, SEQ_LEN) windows, including mirrored copies."""
    pose = s2a.resample(np.asarray(pose, dtype=np.float32), fps, CAREPD_TARGET_FPS)
    trans = s2a.resample(np.asarray(trans, dtype=np.float32), fps, CAREPD_TARGET_FPS)
    positions = smpl_to_positions22(pose, trans, np.asarray(beta, dtype=np.float32).reshape(-1))
    T = positions.shape[0]
    if T < CLIP_LEN:
        return []

    windows = []
    start = 0
    while start + CLIP_LEN <= T:
        clip = positions[start:start + CLIP_LEN]
        feat = positions_to_h3d(clip)[:SEQ_LEN]
        if feat.shape[0] == SEQ_LEN and not np.isnan(feat).any():
            windows.append(feat.T.astype(np.float32))
            mfeat = positions_to_h3d(mirror_positions22(clip))[:SEQ_LEN]
            if mfeat.shape[0] == SEQ_LEN and not np.isnan(mfeat).any():
                windows.append(mfeat.T.astype(np.float32))
        start += WINDOW_STRIDE
    return windows


def _load_cohort(cohort):
    pkl_path = os.path.join(CAREPD_RAW_DIR, f"{cohort}.pkl")
    if not os.path.exists(pkl_path):
        print(f"  {cohort}: PKL not found, skipping.")
        return {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(pkl_path, "rb") as f:
            return pickle.load(f, encoding="latin1")


def preprocess():
    print("=" * 60)
    print("CARE-PD → HumanML3D (263-dim)  windowing + normalization")
    print(f"  Raw dir : {CAREPD_RAW_DIR}")
    print(f"  Window  : {SEQ_LEN} frames (clip {CLIP_LEN})  stride: {WINDOW_STRIDE}")
    print("=" * 60)

    splits = load_fixed_splits()
    train_wins, train_labels = [], []
    eval_wins, eval_labels = [], []
    n_short = n_skip = n_fail = 0

    for cohort in CAREPD_LABELED_COHORTS:
        data = _load_cohort(cohort)
        if not data:
            continue
        for subject_id, subject_walks in data.items():
            sid = str(subject_id)
            if sid in splits[cohort]["train"]:
                win_buf, lab_buf = train_wins, train_labels
            elif sid in splits[cohort]["eval"]:
                win_buf, lab_buf = eval_wins, eval_labels
            else:
                continue
            for walk in subject_walks.values():
                updrs = walk.get("UPDRS_GAIT")
                if updrs is None or int(updrs) > UPDRS_MAX_SCORE:
                    n_skip += 1
                    continue
                updrs = int(updrs)
                trans = walk.get("trans")
                if trans is None:
                    trans = np.zeros((np.asarray(walk["pose"]).shape[0], 3), dtype=np.float32)
                try:
                    windows = walk_to_windows(walk["pose"], trans, walk["beta"],
                                              float(walk.get("fps", CAREPD_TARGET_FPS)))
                except Exception as e:
                    print(f"  FAIL [{sid}]: {e}")
                    n_fail += 1
                    continue
                if not windows:
                    n_short += 1
                    continue
                for win in windows:
                    win_buf.append(win)
                    lab_buf.append(updrs)

    print(f"\n  Skipped {n_short} short walks, {n_skip} (no/too-high UPDRS), {n_fail} failed.")
    train_wins = np.asarray(train_wins, dtype=np.float32)
    train_labels = np.asarray(train_labels, dtype=np.int64)
    eval_wins = np.asarray(eval_wins, dtype=np.float32)
    eval_labels = np.asarray(eval_labels, dtype=np.int64)
    if len(train_wins) == 0:
        print("ERROR: no training windows produced — check raw data paths.")
        return

    all_wins = np.concatenate([train_wins, eval_wins])
    all_labels = np.concatenate([train_labels, eval_labels])
    for name, wins, labs in [("Train", train_wins, train_labels),
                             ("Eval", eval_wins, eval_labels),
                             ("Total", all_wins, all_labels)]:
        print(f"\n{name}: {len(wins)} windows  {wins.shape}")
        for cls in range(UPDRS_CLASSES):
            n = int((labs == cls).sum())
            print(f"  UPDRS {cls}: {n} ({100 * n / len(labs):.1f}%)")

    os.makedirs(os.path.dirname(H3D_TRAIN_DATA_PATH), exist_ok=True)
    np.save(H3D_TRAIN_DATA_PATH, train_wins)
    np.save(H3D_TRAIN_LABELS_PATH, train_labels)
    np.save(H3D_EVAL_DATA_PATH, eval_wins)
    np.save(H3D_EVAL_LABELS_PATH, eval_labels)

    train_tensor = torch.from_numpy(train_wins).float()
    mean = train_tensor.mean(dim=(0, 2)).view(-1, 1)
    std = torch.clamp(train_tensor.std(dim=(0, 2)).view(-1, 1), min=H3D_STD_FLOOR)
    torch.save({"mean": mean, "std": std}, H3D_NORM_PARAMS_PATH)

    print(f"\nSaved train : {H3D_TRAIN_DATA_PATH}  {train_wins.shape}")
    print(f"Saved eval  : {H3D_EVAL_DATA_PATH}  {eval_wins.shape}")
    print(f"Saved norms : {H3D_NORM_PARAMS_PATH}")
    print("\nDone — run training.train_vae next.")


if __name__ == "__main__":
    preprocess()
