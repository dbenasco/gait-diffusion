"""
SMPL forward kinematics and geometric angle helpers.

Provides the SMPL joint-position FK used by the H3D pipeline
(`smpl_joint_positions`), the anatomical angle primitives used by `h3d_bridge`
(`_flex_angle`, `_hip_angles`), FPS resampling, and the chumpy unpickling stub
needed to read the CARE-PD PKLs and the SMPL model.

Requires:
  pip install smplx
  data/smpl_models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl
"""

import os
import sys
import pickle
import copyreg
import types

import math

import numpy as np
import torch
import torch.nn.functional as F


def _install_chumpy_stub_now():
    """Replace any broken/missing chumpy with a minimal np.ndarray-based stub."""
    for key in list(sys.modules):
        if key == "chumpy" or key.startswith("chumpy."):
            del sys.modules[key]

    class _Ch(np.ndarray):
        """
        Minimal chumpy.Ch stub.  Subclasses np.ndarray so shape/dtype work.
        __new__    : always creates an empty 1-D array (will be filled by __setstate__)
        __setstate__: handles both ndarray tuple-states AND chumpy dict-states
                      (chumpy stores {'x': <data>, ...})
        """
        def __new__(cls, *args, **kwargs):
            return np.ndarray.__new__(cls, (0,))

        def __array_finalize__(self, obj):
            pass

        def __setstate__(self, state):
            if isinstance(state, dict):
                x = state.get('x')
                if x is not None:
                    try:
                        x = np.ascontiguousarray(np.asarray(x, dtype=float))
                        np.ndarray.__setstate__(
                            self,
                            (1, x.shape, x.dtype, False, x.tobytes())
                        )
                        return
                    except Exception:
                        pass
            else:
                try:
                    np.ndarray.__setstate__(self, state)
                except Exception:
                    pass

        def __reduce__(self):
            return (np.zeros, ((0,),))

    chumpy_mod    = types.ModuleType("chumpy")
    chumpy_ch_mod = types.ModuleType("chumpy.ch")
    chumpy_ch_mod.Ch  = _Ch
    chumpy_mod.ch     = chumpy_ch_mod
    chumpy_mod.Ch     = _Ch
    chumpy_mod.array  = np.array
    chumpy_mod.zeros  = np.zeros
    sys.modules["chumpy"]    = chumpy_mod
    sys.modules["chumpy.ch"] = chumpy_ch_mod
    return _Ch

_Ch_cls = _install_chumpy_stub_now()

_copyreg_orig = copyreg._reconstructor
def _copyreg_patched(cls, base, state):
    if isinstance(cls, type) and cls.__name__ == "_Ch":
        return _Ch_cls.__new__(_Ch_cls)
    return _copyreg_orig(cls, base, state)
copyreg._reconstructor = _copyreg_patched

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import SMPL_MODEL_PATH

DEVICE = torch.device("cpu")

_L_HIP = 1;  _R_HIP = 2
_L_KNEE = 4; _R_KNEE = 5
_L_ANKLE = 7; _R_ANKLE = 8
_L_FOOT = 10; _R_FOOT = 11


def _load_pkl_with_stub(path: str) -> dict:
    """
    Load a .pkl file that may contain chumpy objects (Python 2 pickle).
    Converts all ndarray subclasses to plain np.ndarrays.
    """
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with open(path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')

    def _to_np(v):
        if isinstance(v, np.ndarray) and type(v) is not np.ndarray:
            return np.array(v)
        return v

    if isinstance(data, dict):
        return {k: _to_np(v) for k, v in data.items()}
    return data


_smpl_cache: dict = {}
_model_data_cache = None


def _get_model_data() -> dict:
    """
    The 247MB SMPL pkl, loaded once and reused for every batch_size. Without
    this, _get_smpl re-parses the whole pickle (chumpy stub included) on every
    cache miss, which is every distinct walk length — on a full CARE-PD run
    (thousands of walks of varying length) that re-read makes preprocessing
    effectively hang.
    """
    global _model_data_cache
    if _model_data_cache is None:
        _model_data_cache = _load_pkl_with_stub(SMPL_MODEL_PATH)
    return _model_data_cache


def _get_smpl(batch_size: int):
    if batch_size not in _smpl_cache:
        from smplx import SMPL
        from smplx.utils import Struct
        data_struct = Struct(**_get_model_data())
        _smpl_cache[batch_size] = SMPL(
            model_path=SMPL_MODEL_PATH,
            data_struct=data_struct,
            batch_size=batch_size,
        ).to(DEVICE)
    return _smpl_cache[batch_size]


def smpl_joint_positions(pose_72: np.ndarray, beta_10: np.ndarray) -> np.ndarray:
    """(T, 72) + (10,) → (T, 24, 3) joint positions (no root translation)."""
    T = pose_72.shape[0]
    model = _get_smpl(T)

    beta = torch.tensor(beta_10, dtype=torch.float32, device=DEVICE)
    if beta.dim() == 1:
        beta = beta.unsqueeze(0)
    beta = beta.expand(T, -1)

    with torch.no_grad():
        out = model(
            betas=beta,
            global_orient=torch.tensor(pose_72[:, :3], dtype=torch.float32, device=DEVICE),
            body_pose=torch.tensor(pose_72[:, 3:],  dtype=torch.float32, device=DEVICE),
            transl=torch.zeros(T, 3, device=DEVICE),
        )
    return out.joints[:, :24].cpu().numpy()


_RAD2DEG = 180.0 / math.pi

_PELVIS  = 0
_SPINE1  = 3


def _flex_angle(p_prox: torch.Tensor, p_mid: torch.Tensor, p_dist: torch.Tensor) -> torch.Tensor:
    """
    Flexion angle at p_mid (degrees). 0° = extended.
    Uses atan2(|v1×v2|, v1·v2) — no arccos, numerically stable at all angles.
    """
    v1 = F.normalize(p_mid  - p_prox, dim=-1)
    v2 = F.normalize(p_dist - p_mid,  dim=-1)
    cross = torch.linalg.cross(v1, v2)
    return torch.atan2(cross.norm(dim=-1), (v1 * v2).sum(dim=-1)) * _RAD2DEG


def _pelvis_frame(joints: torch.Tensor) -> torch.Tensor:
    """
    Anatomical pelvis frame from joint positions — no rotation parameters needed.

    Right axis : L_hip → R_hip
    Forward    : cross(right, pelvis→spine)   (then re-orthogonalised)
    Up         : cross(forward, right)

    Returns (T, 3, 3) where columns are [right, up, forward].
    """
    right   = F.normalize(joints[:, _R_HIP]  - joints[:, _L_HIP],  dim=-1)
    up_raw  = F.normalize(joints[:, _SPINE1] - joints[:, _PELVIS], dim=-1)
    forward = F.normalize(torch.linalg.cross(right, up_raw), dim=-1)
    up      = torch.linalg.cross(forward, right)
    return torch.stack([right, up, forward], dim=-1)


def _hip_angles(joints: torch.Tensor, hip_idx: int, knee_idx: int) -> torch.Tensor:
    """
    (T, 2) — [flexion, abduction] in degrees, expressed in the pelvis frame.
      flex > 0 : forward flexion
      abd  > 0 : abduction (away from midline)
    """
    R = _pelvis_frame(joints)
    thigh = F.normalize(joints[:, knee_idx] - joints[:, hip_idx], dim=-1)
    local = torch.bmm(R.transpose(1, 2), thigh.unsqueeze(-1)).squeeze(-1)

    flex = torch.atan2( local[:, 2], -local[:, 1]) * _RAD2DEG
    abd  = torch.atan2( local[:, 0], -local[:, 1]) * _RAD2DEG
    return torch.stack([flex, abd], dim=-1)


def resample(pose: np.ndarray, src_fps: float, tgt_fps: float) -> np.ndarray:
    if src_fps == tgt_fps:
        return pose
    T_src, D = pose.shape
    T_tgt = int(round(T_src / src_fps * tgt_fps))
    if T_tgt < 2:
        return pose
    x_old = np.linspace(0, 1, T_src)
    x_new = np.linspace(0, 1, T_tgt)
    out = np.zeros((T_tgt, D), dtype=pose.dtype)
    for d in range(D):
        out[:, d] = np.interp(x_new, x_old, pose[:, d])
    return out

