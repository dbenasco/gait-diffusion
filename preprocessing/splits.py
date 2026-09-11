"""Fixed CARE-PD subject-level train/eval splits."""

import os
import pickle

from config import CAREPD_LABELED_COHORTS, FOLDS_DIR

SPLIT_FILES = {
    "BMCLab":   "BMCLab_fixed.pkl",
    "PD-GaM":   "PD-GaM_authors_fixed.pkl",
    "3DGait":   "3DGait_fixed.pkl",
    "T-SDU-PD": "T-SDU-PD_PD_fixed.pkl",
}


def load_fixed_splits() -> dict:
    """Return {cohort: {'train': set(subject_ids), 'eval': set(subject_ids)}}."""
    splits = {}
    for cohort in CAREPD_LABELED_COHORTS:
        with open(os.path.join(FOLDS_DIR, SPLIT_FILES[cohort]), "rb") as f:
            fold = pickle.load(f)[1]
        splits[cohort] = {
            "train": set(str(s) for s in fold["train"]),
            "eval": set(str(s) for s in fold["eval"]),
        }
    return splits
