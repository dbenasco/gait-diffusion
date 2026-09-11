"""
Bridge from the HumanML3D (263-dim) representation to anatomical angles and
clinical metrics, used by both the training losses and the evaluation scripts.

Provided metrics:
  - `h3d_to_angles`: 263-dim window → 6 sagittal leg angles (degrees), pure
    torch and differentiable, so it also serves as the ROM-loss angle source.
  - `h3d_to_positions22`: 263-dim window → (T, 22, 3) local joint positions.
  - `root_motion_metrics`: gait speed / step length / cadence from the root
    velocity and foot-contact blocks.
  - `ave`, `aamd`, `asmd`: GAITGen distance metrics on recovered joint positions.

263 layout = root(4) | ric_pos(63) | rot6d(126) | local_vel(66) | foot(4)
             root = [r_ang_vel(1), r_lin_vel_xz(2), root_height(1)]
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import smpl_to_angles as s2a
from config import (
    H3D_BLOCKS, H3D_N_JOINTS, CAREPD_TARGET_FPS, SAG_IDX,
)

_RIC_LO, _RIC_HI = H3D_BLOCKS["ric"]
_ROOT_HEIGHT_CH = 3
_FOOT_LO, _FOOT_HI = H3D_BLOCKS["foot"]
_RVEL_CH = 0
_LVEL_LO, _LVEL_HI = 1, 3

_J_PELVIS, _J_NECK, _J_HEAD = 0, 12, 15
_J_L_SHOULDER, _J_R_SHOULDER = 16, 17
_J_L_ELBOW,    _J_R_ELBOW    = 18, 19
_J_L_WRIST,    _J_R_WRIST    = 20, 21

ARM_JOINT_NAMES = ["L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist"]
_ARM_JOINTS     = [_J_L_SHOULDER, _J_R_SHOULDER, _J_L_ELBOW, _J_R_ELBOW, _J_L_WRIST, _J_R_WRIST]


def denormalize(feat_norm, mean, std):
    """(B,263,T) normalized → denormalized. mean/std are (263,1) tensors."""
    if not torch.is_tensor(feat_norm):
        feat_norm = torch.as_tensor(feat_norm, dtype=torch.float32)
    mean = torch.as_tensor(mean, dtype=feat_norm.dtype, device=feat_norm.device).view(1, -1, 1)
    std  = torch.as_tensor(std,  dtype=feat_norm.dtype, device=feat_norm.device).view(1, -1, 1)
    return feat_norm * std + mean


def h3d_to_positions22(feat_denorm: torch.Tensor) -> torch.Tensor:
    """
    Denormalized (B,263,T) → (B, T, 22, 3) local joint positions.

    Reconstructs the canonical per-frame skeleton: joint 0 (pelvis/root) at
    (0, root_height, 0); joints 1..21 from the ric block. Differentiable.
    """
    B, C, T = feat_denorm.shape
    root_y = feat_denorm[:, _ROOT_HEIGHT_CH:_ROOT_HEIGHT_CH + 1, :]
    ric = feat_denorm[:, _RIC_LO:_RIC_HI, :].reshape(B, H3D_N_JOINTS - 1, 3, T)
    zeros = torch.zeros_like(root_y)
    root = torch.stack([zeros, root_y, zeros], dim=2)
    pos = torch.cat([root, ric], dim=1)
    return pos.permute(0, 3, 1, 2).contiguous()


def _angles_from_joints(joints: torch.Tensor) -> torch.Tensor:
    """
    (N, 22, 3) joints (SMPL order) → (N, 8) angles in degrees, identical channel
    order to smpl_to_angles.extract_angles:
      [L_Hip_Flex, L_Hip_Abd, R_Hip_Flex, R_Hip_Abd,
       L_Knee, R_Knee, L_Ankle, R_Ankle]
    """
    l_hip = s2a._hip_angles(joints, s2a._L_HIP, s2a._L_KNEE)
    r_hip = s2a._hip_angles(joints, s2a._R_HIP, s2a._R_KNEE)
    l_knee = s2a._flex_angle(joints[:, s2a._L_HIP],  joints[:, s2a._L_KNEE],  joints[:, s2a._L_ANKLE])
    r_knee = s2a._flex_angle(joints[:, s2a._R_HIP],  joints[:, s2a._R_KNEE],  joints[:, s2a._R_ANKLE])
    l_ank  = s2a._flex_angle(joints[:, s2a._L_KNEE], joints[:, s2a._L_ANKLE], joints[:, s2a._L_FOOT]) - 90.0
    r_ank  = s2a._flex_angle(joints[:, s2a._R_KNEE], joints[:, s2a._R_ANKLE], joints[:, s2a._R_FOOT]) - 90.0
    return torch.cat([
        l_hip, r_hip,
        l_knee.unsqueeze(1), r_knee.unsqueeze(1),
        l_ank.unsqueeze(1),  r_ank.unsqueeze(1),
    ], dim=1)


def h3d_to_angles(feat, mean=None, std=None, sagittal_only=True) -> torch.Tensor:
    """
    HumanML3D features → anatomical angles (degrees), channels-first (B, A, T).

    feat: (B,263,T). If mean/std are given, feat is treated as normalized and
    denormalized first (use this in the ROM loss, passing the decoded VAE output
    and the H3D norm params). A = 6 (sagittal, SAG_IDX) if sagittal_only else 8.
    Pure torch and differentiable.
    """
    if not torch.is_tensor(feat):
        feat = torch.as_tensor(feat, dtype=torch.float32)
    if feat.dim() == 2:
        feat = feat.unsqueeze(0)
    feat = denormalize(feat, mean, std) if mean is not None else feat

    B, _, T = feat.shape
    pos = h3d_to_positions22(feat)
    joints = pos.reshape(B * T, H3D_N_JOINTS, 3)
    ang = _angles_from_joints(joints).reshape(B, T, 8)
    ang = ang.permute(0, 2, 1).contiguous()
    if sagittal_only:
        ang = ang[:, SAG_IDX, :]
    return ang


def h3d_to_arm_timeseries(feat_denorm: torch.Tensor) -> torch.Tensor:
    """
    (B, 263, T) denormalized → (B, 6, T) sagittal (Z) displacement of arm joints
    relative to the pelvis, in metres.
    Channels: [L_Shoulder, R_Shoulder, L_Elbow, R_Elbow, L_Wrist, R_Wrist].
    ROM of this signal = arm excursion (same definition as arm_swing_range_t but
    extended to shoulder and elbow, not just wrist).
    """
    if not torch.is_tensor(feat_denorm):
        feat_denorm = torch.as_tensor(feat_denorm, dtype=torch.float32)
    pos      = h3d_to_positions22(feat_denorm)
    pelvis_z = pos[:, :, _J_PELVIS, 2]
    channels = [pos[:, :, j, 2] - pelvis_z for j in _ARM_JOINTS]
    return torch.stack(channels, dim=1)


def root_motion_metrics(feat_denorm, fps: float = CAREPD_TARGET_FPS) -> dict:
    """
    Per-window gait speed / step length / cadence from the root + foot blocks.
    feat_denorm: (B,263,T) denormalized. Returns dict of (B,) numpy arrays.

      distance  = Σ_t ‖root_lin_vel_t‖           (m, per window)
      speed     = distance · fps / T              (m/s)
      cadence   = n_steps / (T/fps) · 60          (steps/min)
      step_len  = distance / max(n_steps, 1)      (m/step)

    n_steps = rising-edge count of per-foot contact (OR of each foot's channels),
    summed over left and right feet.
    """
    x = feat_denorm.detach().cpu().numpy() if torch.is_tensor(feat_denorm) else np.asarray(feat_denorm)
    B, _, T = x.shape
    lin_vel = x[:, _LVEL_LO:_LVEL_HI, :]
    distance = np.linalg.norm(lin_vel, axis=1).sum(axis=1)
    speed = distance * fps / T

    foot = x[:, _FOOT_LO:_FOOT_HI, :] > 0.5
    left  = foot[:, 0:2, :].any(axis=1)
    right = foot[:, 2:4, :].any(axis=1)
    def _onsets(contact):
        rises = (~contact[:, :-1]) & contact[:, 1:]
        return rises.sum(axis=1)
    n_steps = _onsets(left) + _onsets(right)
    duration = T / fps
    cadence = n_steps / duration * 60.0
    step_len = distance / np.maximum(n_steps, 1)
    return {"speed": speed, "cadence": cadence,
            "step_length": step_len, "distance": distance}


def _to_positions_np(feat_denorm) -> np.ndarray:
    """(B,263,T) → (B,T,22,3) numpy via the same local reconstruction."""
    if not torch.is_tensor(feat_denorm):
        feat_denorm = torch.as_tensor(feat_denorm, dtype=torch.float32)
    return h3d_to_positions22(feat_denorm).detach().cpu().numpy()


def per_joint_temporal_std(positions: np.ndarray) -> np.ndarray:
    """(B,T,22,3) → (B,22) per-joint temporal std (L2 over xyz of per-axis std)."""
    std_axis = positions.std(axis=1)
    return np.linalg.norm(std_axis, axis=-1)


def ave(real_feat, synth_feat):
    """
    Average Variance Error (GaitGen). Set-level interpretation: per-joint temporal
    std averaged over each set, L2 difference per joint, averaged over joints.
    Returns (overall_scalar, per_joint (22,)).
    """
    r = per_joint_temporal_std(_to_positions_np(real_feat)).mean(axis=0)
    s = per_joint_temporal_std(_to_positions_np(synth_feat)).mean(axis=0)
    per_joint = np.abs(r - s)
    return float(per_joint.mean()), per_joint


def arm_swing_range_t(positions: torch.Tensor) -> torch.Tensor:
    """
    (B,T,22,3) torch, differentiable → (B,) arm-swing range.
    GAITGen definition (arXiv:2503.22397): Euclidean distance between wrist and
    shoulder joints at each time step, max−min over the sequence, min of the two
    arms. Normalisation by leg length is applied by the caller (eval) or is not
    needed for the L1 training loss (which matches raw range).
    """
    out = []
    for w, s in ((_J_L_WRIST, _J_L_SHOULDER), (_J_R_WRIST, _J_R_SHOULDER)):
        dist = torch.linalg.norm(positions[:, :, w, :] - positions[:, :, s, :],
                                 dim=-1)
        out.append(dist.max(dim=1).values - dist.min(dim=1).values)
    stacked = torch.stack(out, dim=0)
    return stacked.min(dim=0).values


def trunk_inclination_t(positions: torch.Tensor) -> torch.Tensor:
    """
    (B,T,22,3) torch, differentiable → (B,) mean torso inclination (deg) from
    vertical: angle between the pelvis→neck vector and the +Y axis, averaged over
    the window. Same definition as the GaitGen `asmd` metric.
    """
    v = positions[:, :, _J_NECK, :] - positions[:, :, _J_PELVIS, :]
    horiz = torch.linalg.norm(v[..., [0, 2]], dim=-1)
    vert = v[..., 1]
    incl = torch.rad2deg(torch.atan2(horiz, vert))
    return incl.mean(dim=1)


def arm_swing_range(positions: np.ndarray) -> np.ndarray:
    """Numpy shim over `arm_swing_range_t` — used by the eval-only Tier-3 metrics."""
    t = torch.as_tensor(positions, dtype=torch.float32)
    return arm_swing_range_t(t).numpy()


def trunk_inclination(positions: np.ndarray) -> np.ndarray:
    """Numpy shim over `trunk_inclination_t` — used by the eval-only Tier-3 metrics."""
    t = torch.as_tensor(positions, dtype=torch.float32)
    return trunk_inclination_t(t).numpy()


def _abs_class_mean_diff(real_by_class: dict, synth_by_class: dict) -> float:
    """(1/C) Σ_c |mean(real_c) − mean(synth_c)| over shared classes."""
    classes = sorted(set(real_by_class) & set(synth_by_class))
    diffs = [abs(float(np.mean(real_by_class[c])) - float(np.mean(synth_by_class[c])))
             for c in classes]
    return float(np.mean(diffs)) if diffs else float("nan")


def aamd(real_feat_by_class: dict, synth_feat_by_class: dict) -> float:
    """Abs Arm-swing Mean Diff: |AS_gen − AS_gt| per class, averaged over classes."""
    real_as  = {c: arm_swing_range(_to_positions_np(f)) for c, f in real_feat_by_class.items()}
    synth_as = {c: arm_swing_range(_to_positions_np(f)) for c, f in synth_feat_by_class.items()}
    return _abs_class_mean_diff(real_as, synth_as)


def asmd(real_feat_by_class: dict, synth_feat_by_class: dict) -> float:
    """Abs Stooped-posture Mean Diff: |incl_gen − incl_gt| per class, averaged."""
    real_in  = {c: trunk_inclination(_to_positions_np(f)) for c, f in real_feat_by_class.items()}
    synth_in = {c: trunk_inclination(_to_positions_np(f)) for c, f in synth_feat_by_class.items()}
    return _abs_class_mean_diff(real_in, synth_in)
