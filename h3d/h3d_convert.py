"""
SMPL → HumanML3D (263-dim) conversion driver.

The vendored HumanML3D toolkit (`motion_process.py`) is the *library* half: its
`process_file` / `extract_features` reference module-level globals
(`n_raw_offsets`, `kinematic_chain`, `face_joint_indx`, `fid_l`, `fid_r`,
`l_idx1`, `l_idx2`, `tgt_offsets`) that the canonical driver (a notebook) sets up
but `paramUtil.py` does not define. This module is that missing driver, specialised
for the CARE-PD SMPL → 22-joint T2M skeleton path.

Pipeline per walk:
    pose(T,72) axis-angle + trans(T,3) + beta(10) -> SMPL FK *with translation*
      -> 24-joint world positions -> first 22 (T2M skeleton order)
      -> motion_process.process_file -> HumanML3D features (T-1, 263)

263 layout = root(4) | ric_pos(63) | rot6d(126) | local_vel(66) | foot_contact(4)

tgt_offsets (retarget target skeleton): the canonical HumanML3D pipeline retargets
every sequence onto a fixed reference skeleton (subject '000021'). We don't have that
reference, so we use the **neutral SMPL skeleton** (beta=0, zero pose) as the target.
This is deterministic, reproducible, and shares the data's metric scale. If exact
GaitGen scale comparability is later required, swap `_compute_tgt_offsets` to load the
000021 reference offsets.
"""

import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
# Put the vendored package root on sys.path so motion_process's
# `from common.skeleton import ...` / `from utils.paramUtil import ...` resolve.
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import motion_process as mp                       # noqa: E402
from common.skeleton import Skeleton              # noqa: E402
from utils.paramUtil import (                     # noqa: E402
    t2m_raw_offsets, t2m_kinematic_chain,
)

# smpl_to_angles installs the chumpy stub on import and exposes the SMPL singleton.
sys.path.insert(0, os.path.dirname(_HERE))        # repo root on path
import smpl_to_angles as s2a                       # noqa: E402

# ── T2M / HumanML3D skeleton constants (SMPL 22-joint order) ──────────────────
N_JOINTS = 22
FEATURE_DIM = 263
FEET_THRE = 0.002

# face_joint_indx order = [right hip, left hip, right shoulder, left shoulder]
FACE_JOINT_INDX = [2, 1, 17, 16]
FID_L = [7, 10]   # L_ankle, L_foot
FID_R = [8, 11]   # R_ankle, R_foot
L_IDX1, L_IDX2 = 5, 8   # leg joints used for the uniform-skeleton scale ratio

_N_RAW_OFFSETS = torch.from_numpy(t2m_raw_offsets)

_tgt_offsets_cache = None
_globals_installed = False


def _compute_tgt_offsets() -> torch.Tensor:
    """Bone offsets of the neutral SMPL skeleton (beta=0, zero pose), 22 joints."""
    model = s2a._get_smpl(1)
    with torch.no_grad():
        out = model(
            betas=torch.zeros(1, 10),
            global_orient=torch.zeros(1, 3),
            body_pose=torch.zeros(1, 69),
            transl=torch.zeros(1, 3),
        )
    rest = out.joints[:, :N_JOINTS, :].cpu().float()[0]   # (22, 3)
    skel = Skeleton(_N_RAW_OFFSETS, t2m_kinematic_chain, "cpu")
    return skel.get_offsets_joints(rest)                   # (22, 3)


def _install_globals():
    """Inject the globals process_file/extract_features read from their namespace."""
    global _globals_installed, _tgt_offsets_cache
    if _globals_installed:
        return
    if _tgt_offsets_cache is None:
        _tgt_offsets_cache = _compute_tgt_offsets()
    mp.n_raw_offsets = _N_RAW_OFFSETS
    mp.kinematic_chain = t2m_kinematic_chain
    mp.face_joint_indx = FACE_JOINT_INDX
    mp.fid_l = FID_L
    mp.fid_r = FID_R
    mp.l_idx1 = L_IDX1
    mp.l_idx2 = L_IDX2
    mp.tgt_offsets = _tgt_offsets_cache
    _globals_installed = True


# SMPL 22-joint indices used only for the up/down sanity check below.
_J_NECK, _J_HEAD = 12, 15
_J_L_ANKLE, _J_R_ANKLE, _J_L_FOOT, _J_R_FOOT = 7, 8, 10, 11


def _fix_updown(positions: np.ndarray) -> np.ndarray:
    """
    Some CARE-PD cohorts (PD-GaM, 3DGait, T-SDU-PD) store SMPL pose params with
    an extra ~180deg rotation relative to BMCLab's convention, so naive FK comes
    out upside-down (feet above head, confirmed on real data: PD-GaM/3DGait/
    T-SDU-PD walks are upside-down in 100% of sampled walks, BMCLab in 0%).

    Detected via mean foot/ankle Y vs neck/head Y over the whole walk, corrected
    with a 180deg rotation about the X-axis (negate Y, Z) -- a proper rotation
    (det=+1, preserves left/right and chirality), so it's a no-op for every
    rotation-invariant downstream metric (leg ROM, DTW, arm-swing range) and
    only fixes metrics that reference the world vertical axis directly (trunk
    inclination).
    """
    feet_y = positions[:, [_J_L_ANKLE, _J_R_ANKLE, _J_L_FOOT, _J_R_FOOT], 1].mean()
    head_y = positions[:, [_J_NECK, _J_HEAD], 1].mean()
    if feet_y > head_y:
        positions = positions.copy()
        positions[:, :, 1] *= -1.0
        positions[:, :, 2] *= -1.0
    return positions


def smpl_to_positions22(pose_72: np.ndarray, trans: np.ndarray,
                        beta_10: np.ndarray) -> np.ndarray:
    """
    (T,72) axis-angle + (T,3) trans + (10,) beta -> (T, 22, 3) world joint positions.

    Includes global_orient (heading) and translation so the HumanML3D root channels
    carry real locomotion (gait speed/turning). Assumes SMPL's native y-up frame;
    process_file re-canonicalises heading to +Z and the floor to y=0 per sequence.
    `_fix_updown` corrects cohorts whose pose params violate that y-up assumption.
    """
    T = pose_72.shape[0]
    model = s2a._get_smpl(T)

    beta = torch.as_tensor(beta_10, dtype=torch.float32).reshape(-1)[:10]
    beta = beta.unsqueeze(0).expand(T, -1)

    with torch.no_grad():
        out = model(
            betas=beta,
            global_orient=torch.as_tensor(pose_72[:, :3], dtype=torch.float32),
            body_pose=torch.as_tensor(pose_72[:, 3:], dtype=torch.float32),
            transl=torch.as_tensor(trans, dtype=torch.float32),
        )
    positions = out.joints[:, :N_JOINTS, :].cpu().numpy().astype(np.float32)
    return _fix_updown(positions)


def positions_to_h3d(positions: np.ndarray) -> np.ndarray:
    """(T, 22, 3) world joint positions -> HumanML3D features (T-1, 263)."""
    _install_globals()
    data = mp.process_file(positions.astype(np.float32), FEET_THRE)
    if isinstance(data, tuple):           # process_file returns (data, glob, pos, lvel)
        data = data[0]
    return np.asarray(data, dtype=np.float32)                  # (T-1, 263)


# ── Mirror augmentation (position space) ──────────────────────────────────────
# L/R joint index pairs in the SMPL 22-joint order. Midline joints
# (0 pelvis, 3 spine1, 6 spine2, 9 spine3, 12 neck, 15 head) are never swapped.
POSITION_MIRROR_PAIRS = [
    (1, 2),    # hip
    (4, 5),    # knee
    (7, 8),    # ankle
    (10, 11),  # foot
    (13, 14),  # collar
    (16, 17),  # shoulder
    (18, 19),  # elbow
    (20, 21),  # wrist
]


def mirror_positions22(positions: np.ndarray) -> np.ndarray:
    """
    Sagittal mirror of a (T, 22, 3) joint-position sequence: negate the X axis and
    swap left/right joints. Yields a valid right-handed pose of the mirrored body.

    Mirroring in *position* space and re-running process_file is provably correct —
    it sidesteps the error-prone hand-derived mirror of the 263-dim vector (which
    must flip root angular/lateral velocity, the rot6d block, and per-joint local
    velocities consistently). process_file recomputes all of that from the mirrored
    skeleton.
    """
    m = positions.copy()
    m[..., 0] *= -1.0
    for a, b in POSITION_MIRROR_PAIRS:
        m[:, [a, b]] = m[:, [b, a]]
    return m
