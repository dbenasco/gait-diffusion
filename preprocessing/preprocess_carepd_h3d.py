"""
CARE-PD SMPL (pose+trans+beta) → HumanML3D (263-dim) training arrays.

Parallel to preprocess_carepd.py, but for the full-body HumanML3D representation
(USE_H3D pipeline). Reads the raw CARE-PD cohort PKLs directly — no angles.pkl
intermediate — because the 263-dim vector needs SMPL FK *with translation*.

Per walk:
  pose(T,72) + trans(T,3) + beta(10) , fps
    → resample to 30 fps → SMPL FK with translation → 22-joint world positions
    → slide 97-frame position clips (stride 48)
    → process_file each clip → 96-frame HumanML3D window (263-dim)
    → mirror augmentation in *position* space (X-negate + L/R swap, then process_file)

263 layout = root(4) | ric_pos(63) | rot6d(126) | local_vel(66) | foot_contact(4)

Per-window canonicalization: each 97-frame clip is processed independently, so
every stored window is a standalone canonical HumanML3D clip (root re-faced +Z,
floor at y=0) — matching how the VAE/DiT outputs are interpreted at generation.

Output (data layout (N, 263, 96) — channels-first, like the 6-ch arrays):
  data/carepd/train_gait_h3d.npy   (N_train, 263, 96)
  data/carepd/eval_gait_h3d.npy    (N_eval,  263, 96)
  data/carepd/norm_params_h3d.pt   per-channel mean/std (train split, std-floored)
  + label / laterality arrays

Run with:
  python -m preprocessing.preprocess_carepd_h3d [--test]
  python -m preprocessing.preprocess_carepd_h3d --gaitgen_split
    (train=PD-GaM only, eval=other cohorts, matching GAITGen protocol)
"""

import os
import sys
import pickle
import argparse
import warnings

import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importing smpl_to_angles installs the chumpy stub needed to unpickle CARE-PD walks.
import smpl_to_angles as s2a
from h3d.h3d_convert import (
    smpl_to_positions22, positions_to_h3d, mirror_positions22, N_JOINTS,
)
from preprocessing.preprocess_carepd import load_fixed_splits, _flip_lat
from config import (
    CAREPD_RAW_DIR, CAREPD_LABELED_COHORTS, CAREPD_TARGET_FPS,
    SEQ_LEN, WINDOW_STRIDE, UPDRS_CLASSES, UPDRS_MAX_SCORE,
    H3D_FEATURE_DIM, H3D_STD_FLOOR,
    H3D_PROCESSED_DATA_PATH, H3D_PROCESSED_LABELS_PATH,
    H3D_TRAIN_DATA_PATH, H3D_TRAIN_LABELS_PATH, H3D_TRAIN_LATERALITY_PATH,
    H3D_EVAL_DATA_PATH, H3D_EVAL_LABELS_PATH, H3D_EVAL_LATERALITY_PATH,
    H3D_NORM_PARAMS_PATH,
    H3D_TRAIN_SUBJECTS_PATH, H3D_EVAL_SUBJECTS_PATH,
)

# Input clip length: process_file drops one frame (velocity diff), so a (SEQ_LEN+1)
# position clip yields a SEQ_LEN-frame HumanML3D window.
CLIP_LEN = SEQ_LEN + 1


# ── Laterality (knee-flex variance, same convention as the 6-ch pipeline) ──────

def detect_laterality_pos(positions: np.ndarray, updrs: int) -> int:
    """
    From 22-joint positions: compare L vs R knee-flexion variance.
    0 = Left impaired, 1 = Right impaired, 2 = Symmetric. Knee flexion is
    frame-invariant, so this matches detect_laterality() in preprocess_carepd.py.
    """
    j = torch.from_numpy(np.asarray(positions, dtype=np.float32))
    l_knee = s2a._flex_angle(j[:, s2a._L_HIP], j[:, s2a._L_KNEE], j[:, s2a._L_ANKLE])
    r_knee = s2a._flex_angle(j[:, s2a._R_HIP], j[:, s2a._R_KNEE], j[:, s2a._R_ANKLE])
    var_l, var_r = float(l_knee.var()), float(r_knee.var())
    var_max = max(var_l, var_r) + 1e-8
    if updrs == 0 or abs(var_l - var_r) / var_max < 0.20:
        return 2
    return 1 if var_l > var_r else 0


# ── Per-walk conversion ────────────────────────────────────────────────────────

def walk_to_windows(pose: np.ndarray, trans: np.ndarray, beta: np.ndarray,
                    fps: float, updrs: int):
    """
    One walk → list of (window(263,SEQ_LEN), laterality) pairs including mirrors.
    Returns ([], None) if the walk is shorter than CLIP_LEN frames after resampling.
    """
    pose = s2a.resample(np.asarray(pose, dtype=np.float32), fps, CAREPD_TARGET_FPS)
    trans = s2a.resample(np.asarray(trans, dtype=np.float32), fps, CAREPD_TARGET_FPS)
    beta = np.asarray(beta, dtype=np.float32).reshape(-1)

    positions = smpl_to_positions22(pose, trans, beta)   # (T, 22, 3)
    T = positions.shape[0]
    if T < CLIP_LEN:
        return [], None

    lat = detect_laterality_pos(positions, updrs)
    out = []
    start = 0
    while start + CLIP_LEN <= T:
        clip = positions[start:start + CLIP_LEN]
        feat = positions_to_h3d(clip)[:SEQ_LEN]                 # (SEQ_LEN, 263)
        if feat.shape[0] == SEQ_LEN and not np.isnan(feat).any():
            out.append((feat.T.astype(np.float32), lat))       # (263, SEQ_LEN)
            mfeat = positions_to_h3d(mirror_positions22(clip))[:SEQ_LEN]
            if mfeat.shape[0] == SEQ_LEN and not np.isnan(mfeat).any():
                out.append((mfeat.T.astype(np.float32), _flip_lat(lat)))
        start += WINDOW_STRIDE
    return out, lat


def _load_cohort(cohort: str) -> dict:
    pkl_path = os.path.join(CAREPD_RAW_DIR, f"{cohort}.pkl")
    if not os.path.exists(pkl_path):
        print(f"  {cohort}: PKL not found, skipping.")
        return {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(pkl_path, 'rb') as f:
            return pickle.load(f, encoding='latin1')


# ── Main ────────────────────────────────────────────────────────────────────────

def preprocess(max_walks_per_cohort: int = None, subjects_only: bool = False,
               gaitgen_split: bool = False, eval_cohorts: list = None,
               exclude_cohorts: list = None):
    print("=" * 60)
    print("CARE-PD → HumanML3D (263-dim)  windowing + normalization")
    print(f"  Raw dir : {CAREPD_RAW_DIR}")
    print(f"  Window  : {SEQ_LEN} frames (clip {CLIP_LEN})  stride: {WINDOW_STRIDE}")
    print(f"  Feature : {H3D_FEATURE_DIM}-dim  target fps: {CAREPD_TARGET_FPS}")
    print("=" * 60)

    splits = load_fixed_splits()

    if gaitgen_split:
        # GAITGen protocol: train on PD-GaM only, eval on other cohorts
        print("\nUsing GAITGen-style split:")
        for cohort in splits:
            all_ids = set(splits[cohort]['train']) | set(splits[cohort]['eval'])
            if cohort == "PD-GaM":
                splits[cohort] = {'train': all_ids, 'eval': set()}
                print(f"  {cohort}: ALL {len(all_ids)} subjects → TRAIN")
            else:
                splits[cohort] = {'train': set(), 'eval': all_ids}
                print(f"  {cohort}: ALL {len(all_ids)} subjects → EVAL")

    if eval_cohorts:
        # Restrict eval set to specific cohorts; all other cohorts' subjects → train
        print(f"\nRestricting eval to cohorts: {', '.join(eval_cohorts)}")
        for cohort in splits:
            if cohort not in eval_cohorts:
                all_ids = set(splits[cohort]['train']) | set(splits[cohort]['eval'])
                splits[cohort] = {'train': all_ids, 'eval': set()}
                print(f"  {cohort}: ALL → TRAIN (excluded from eval)")

    if exclude_cohorts:
        print(f"\nExcluding cohorts: {', '.join(exclude_cohorts)}")

    train_wins, train_labels, train_lat, train_subj = [], [], [], []
    eval_wins,  eval_labels,  eval_lat,  eval_subj  = [], [], [], []
    n_short, n_skip, n_fail = 0, 0, 0

    for cohort in CAREPD_LABELED_COHORTS:
        if exclude_cohorts and cohort in exclude_cohorts:
            print(f"  {cohort}: SKIPPED (excluded)")
            continue
        data = _load_cohort(cohort)
        if not data:
            continue
        train_ids = splits[cohort]['train']
        eval_ids  = splits[cohort]['eval']
        n_train, n_eval, n_done = 0, 0, 0

        for subject_id, subject_walks in data.items():
            sid = str(subject_id)
            if sid in train_ids:
                win_buf, lab_buf, lat_buf, subj_buf, is_train = train_wins, train_labels, train_lat, train_subj, True
            elif sid in eval_ids:
                win_buf, lab_buf, lat_buf, subj_buf, is_train = eval_wins, eval_labels, eval_lat, eval_subj, False
            else:
                continue

            for walk_id, walk in subject_walks.items():
                updrs = walk.get('UPDRS_GAIT')
                if updrs is None or int(updrs) > UPDRS_MAX_SCORE:
                    n_skip += 1
                    continue
                updrs = int(updrs)

                if max_walks_per_cohort is not None and n_done >= max_walks_per_cohort:
                    break

                trans = walk.get('trans')
                if trans is None:
                    trans = np.zeros((np.asarray(walk['pose']).shape[0], 3), dtype=np.float32)

                try:
                    windows, lat = walk_to_windows(
                        walk['pose'], trans, walk['beta'],
                        float(walk.get('fps', CAREPD_TARGET_FPS)), updrs,
                    )
                except Exception as e:
                    print(f"  FAIL [{sid}] {walk_id}: {e}")
                    n_fail += 1
                    continue

                if not windows:
                    n_short += 1
                    continue

                for win, wlat in windows:
                    win_buf.append(win)
                    lab_buf.append(updrs)
                    lat_buf.append(wlat)
                    subj_buf.append(sid)

                n_done += 1
                if is_train: n_train += 1
                else:        n_eval += 1
                if n_done % 50 == 0:
                    print(f"  [{cohort}] {n_done} walks processed...")

            if max_walks_per_cohort is not None and n_done >= max_walks_per_cohort:
                break

        print(f"  {cohort}: {n_train} train walks, {n_eval} eval walks")

    if n_short:
        print(f"  Skipped {n_short} walks shorter than {CLIP_LEN} frames.")
    if n_skip:
        print(f"  Skipped {n_skip} walks (no/too-high UPDRS).")
    if n_fail:
        print(f"  Failed  {n_fail} walks (conversion error).")

    train_wins   = np.asarray(train_wins,   dtype=np.float32)
    train_labels = np.asarray(train_labels, dtype=np.int64)
    train_lat    = np.asarray(train_lat,    dtype=np.int64)
    eval_wins    = np.asarray(eval_wins,    dtype=np.float32)
    eval_labels  = np.asarray(eval_labels,  dtype=np.int64)
    eval_lat     = np.asarray(eval_lat,     dtype=np.int64)

    if len(train_wins) == 0:
        print("ERROR: no training windows produced — check raw data paths.")
        return

    all_wins   = np.concatenate([train_wins, eval_wins])
    all_labels = np.concatenate([train_labels, eval_labels])
    all_lat    = np.concatenate([train_lat, eval_lat])

    lat_names = {0: "Left impaired", 1: "Right impaired", 2: "Symmetric"}
    for name, wins, labs, lats in [
        ("Train", train_wins, train_labels, train_lat),
        ("Eval",  eval_wins,  eval_labels,  eval_lat),
        ("Total", all_wins,   all_labels,   all_lat),
    ]:
        if len(wins) == 0:
            continue
        print(f"\n{name}: {len(wins)} windows  {wins.shape}")
        for cls in range(UPDRS_CLASSES):
            n = int((labs == cls).sum())
            print(f"  UPDRS {cls}: {n} ({100*n/len(labs):.1f}%)")
        for side, sname in lat_names.items():
            n = int((lats == side).sum())
            print(f"  {sname}: {n} ({100*n/len(lats):.1f}%)")

    os.makedirs(os.path.dirname(H3D_PROCESSED_DATA_PATH), exist_ok=True)

    if subjects_only:
        np.save(H3D_TRAIN_SUBJECTS_PATH, np.asarray(train_subj, dtype=object))
        np.save(H3D_EVAL_SUBJECTS_PATH,  np.asarray(eval_subj,  dtype=object))
        print(f"\nSaved subject IDs only (existing data/labels/norms untouched):")
        print(f"  {H3D_TRAIN_SUBJECTS_PATH}  ({len(train_subj)} windows)")
        print(f"  {H3D_EVAL_SUBJECTS_PATH}   ({len(eval_subj)} windows)")
        print("\nDone — subject ID arrays ready for bootstrap CIs.")
        return

    # Apply filename suffix based on split mode
    if gaitgen_split and exclude_cohorts:
        _s = "_gaitgen_excl_" + "_".join(sorted(exclude_cohorts)).lower()
    elif gaitgen_split:
        _s = "_gaitgen"
    elif eval_cohorts:
        _s = "_eval_" + "_".join(sorted(eval_cohorts)).lower()
    else:
        _s = ""
    def _gg(path):
        p = Path(path)
        return str(p.with_name(p.stem + _s + p.suffix))

    all_wins_path    = _gg(H3D_PROCESSED_DATA_PATH)
    all_labels_path  = _gg(H3D_PROCESSED_LABELS_PATH)
    train_data_path  = _gg(H3D_TRAIN_DATA_PATH)
    train_labels_path = _gg(H3D_TRAIN_LABELS_PATH)
    train_lat_path   = _gg(H3D_TRAIN_LATERALITY_PATH)
    train_subj_path  = _gg(H3D_TRAIN_SUBJECTS_PATH)
    eval_data_path   = _gg(H3D_EVAL_DATA_PATH)
    eval_labels_path = _gg(H3D_EVAL_LABELS_PATH)
    eval_lat_path    = _gg(H3D_EVAL_LATERALITY_PATH)
    eval_subj_path   = _gg(H3D_EVAL_SUBJECTS_PATH)
    norm_path        = _gg(H3D_NORM_PARAMS_PATH) if (_s) else H3D_NORM_PARAMS_PATH

    np.save(all_wins_path,   all_wins)
    np.save(all_labels_path, all_labels)
    np.save(train_data_path,       train_wins)
    np.save(train_labels_path,     train_labels)
    np.save(train_lat_path,        train_lat)
    np.save(train_subj_path,       np.asarray(train_subj, dtype=object))
    np.save(eval_data_path,        eval_wins)
    np.save(eval_labels_path,      eval_labels)
    np.save(eval_lat_path,         eval_lat)
    np.save(eval_subj_path,        np.asarray(eval_subj, dtype=object))

    # Per-channel normalization from train split only (PD-GaM, when gaitgen_split)
    # Use the same norm_params_h3d.pt if not gaitgen, or a fresh one if gaitgen
    train_tensor = torch.from_numpy(train_wins).float()
    mean = train_tensor.mean(dim=(0, 2)).view(-1, 1)
    std  = train_tensor.std(dim=(0, 2)).view(-1, 1)
    std  = torch.clamp(std, min=H3D_STD_FLOOR)
    torch.save({'mean': mean, 'std': std}, norm_path)

    desc = ""
    if gaitgen_split and exclude_cohorts:
        desc = f" (train=PD-GaM, eval=T-SDU-PD+BMCLab, excluded={', '.join(exclude_cohorts)})"
    elif gaitgen_split:
        desc = " (_gaitgen split)"
    elif eval_cohorts:
        desc = f" (eval only: {', '.join(eval_cohorts)})"
    print(f"\nSaved train : {train_data_path}  {train_wins.shape}{desc}")
    print(f"Saved eval  : {eval_data_path}  {eval_wins.shape}{desc}")
    print(f"Saved norms : {norm_path}  (std floor {H3D_STD_FLOOR}){desc}")
    if gaitgen_split:
        print(f"\nNext: train DiT with env vars pointing to these files:")
        print(f"  UPDRS_TRAIN_DATA={train_data_path}")
        print(f"  UPDRS_EVAL_DATA={eval_data_path}")
        print(f"  UPDRS_TRAIN_LABELS={train_labels_path}")
        print(f"  UPDRS_EVAL_LABELS={eval_labels_path}")
        print(f"  UPDRS_TRAIN_LAT={train_lat_path}")
        print(f"  UPDRS_EVAL_LAT={eval_lat_path}")
        if exclude_cohorts:
            print(f"\n  Training set = PD-GaM only  ({len(train_wins)} windows)")
            print(f"  Eval set     = T-SDU-PD + BMCLab  ({len(eval_wins)} windows)")
            print(f"  Excluded     = {', '.join(exclude_cohorts)}")
    elif eval_cohorts:
        print(f"\nTo evaluate on these cohorts, set env vars:")
        print(f"  UPDRS_EVAL_DATA={eval_data_path}")
        print(f"  UPDRS_EVAL_LABELS={eval_labels_path}")
        print(f"  UPDRS_EVAL_LAT={eval_lat_path}")
        print(f"  UPDRS_EVAL_SUBJECTS={eval_subj_path}")
    print("\nDone — run training.train_vae next.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CARE-PD SMPL → HumanML3D windows")
    parser.add_argument("--test", action="store_true",
                        help="Process only first 10 walks per cohort")
    parser.add_argument("--subjects_only", action="store_true",
                        help="Only save subject ID arrays; leave existing data/labels/norms untouched")
    parser.add_argument("--gaitgen_split", action="store_true",
                        help="GAITGen-style split: PD-GaM → train, other cohorts → eval")
    parser.add_argument("--eval_cohorts", nargs="+", default=None,
                        help="Restrict eval set to specific cohorts (e.g. T-SDU-PD BMCLab)")
    parser.add_argument("--exclude_cohorts", nargs="+", default=None,
                        help="Skip these cohorts entirely (e.g. 3DGait)")
    args = parser.parse_args()
    preprocess(max_walks_per_cohort=10 if args.test else None,
               subjects_only=args.subjects_only,
               gaitgen_split=args.gaitgen_split,
               eval_cohorts=args.eval_cohorts,
               exclude_cohorts=args.exclude_cohorts)
