"""
Stage 1: CARE-PD SMPL sequences → geometric joint angles.

For each walk in the CARE-PD PKL files:
  pose (T, 72) + beta (10,) → SMPL FK → 3D joint positions (T, 24, 3)
                                       → geometric angles  (T_30fps, 8)

Angle channels (degrees):
  0  L_Hip_Flex    1  L_Hip_Abd
  2  R_Hip_Flex    3  R_Hip_Abd
  4  L_Knee_Flex   5  R_Knee_Flex
  6  L_Ankle_Flex  7  R_Ankle_Flex

Hip flex/abd : thigh direction projected onto anatomical pelvis frame (built from joint positions)
Knee flex    : atan2(|v1×v2|, v1·v2) of thigh/shank vectors — 0° = extended
Ankle        : same formula for shank/foot vectors − 90°    — 0° = neutral, + = dorsiflexion

Output: data/carepd/angles.pkl
  {cohort: {subject_id: {walk_id: {'angles': (T, 8), 'updrs': int}}}}

Run with:
  python smpl_to_angles.py [--cohort BMCLab] [--test]

Requires:
  pip install smplx
  data/smpl_models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl
"""

import os
import sys
import pickle
import argparse
import copyreg
import types

import math

import numpy as np
import torch
import torch.nn.functional as F

# ── Install the chumpy stub BEFORE any other imports that might trigger chumpy ──
# The SMPL pkl and the CARE-PD data pkls both contain chumpy.Ch objects.  We
# replace any installed (possibly broken) chumpy with a minimal stub where Ch
# is an np.ndarray subclass, so every pickle reconstruction path returns a
# valid array with a .shape attribute.

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
                # Chumpy stores the numeric data under the key 'x'
                x = state.get('x')
                if x is not None:
                    try:
                        x = np.ascontiguousarray(np.asarray(x, dtype=float))
                        # Resize this object in-place by re-using the ndarray __setstate__
                        np.ndarray.__setstate__(
                            self,
                            (1, x.shape, x.dtype, False, x.tobytes())
                        )
                        return
                    except Exception:
                        pass
                # No usable data — leave as empty array; ndarray.__setstate__ with a dict
                # would crash, so just skip.
            else:
                # Normal ndarray tuple state
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

# Patch copyreg._reconstructor so that when pickle reconstructs a _Ch object
# it gets our stub class instead of trying object.__new__(_Ch).
# NOTE: in chumpy pkls, the call is _reconstructor(cls=_Ch, base=object, state)
#       — we must check `cls`, not `base`.
_copyreg_orig = copyreg._reconstructor
def _copyreg_patched(cls, base, state):
    if isinstance(cls, type) and cls.__name__ == "_Ch":
        # Return an instance of our _Ch stub; pickle will then call __setstate__
        # on it with the chumpy dict state.
        return _Ch_cls.__new__(_Ch_cls)
    return _copyreg_orig(cls, base, state)
copyreg._reconstructor = _copyreg_patched
# ─────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    CAREPD_RAW_DIR, CAREPD_LABELED_COHORTS, CAREPD_TARGET_FPS,
    SMPL_MODEL_PATH, ANGLES_PATH, UPDRS_MAX_SCORE,
)

DEVICE = torch.device("cpu")

_L_HIP = 1;  _R_HIP = 2
_L_KNEE = 4; _R_KNEE = 5
_L_ANKLE = 7; _R_ANKLE = 8
_L_FOOT = 10; _R_FOOT = 11



# ── chumpy stub functions (stub already installed at module load time) ──────

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


# ── SMPL model singleton ──────────────────────────────────────────────────────

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
        # data_struct built fresh per batch_size (smplx may consume/convert
        # entries in place), but the underlying pkl load is cached above.
        data_struct = Struct(**_get_model_data())
        _smpl_cache[batch_size] = SMPL(
            model_path=SMPL_MODEL_PATH,
            data_struct=data_struct,
            batch_size=batch_size,
        ).to(DEVICE)
    return _smpl_cache[batch_size]


# ── SMPL FK ───────────────────────────────────────────────────────────────────

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


# ── Geometric helpers (all in torch, no manual arccos) ────────────────────────

_RAD2DEG = 180.0 / math.pi

# SMPL joints used to build the pelvis anatomical frame
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
    return torch.stack([right, up, forward], dim=-1)   # (T, 3, 3)


def _hip_angles(joints: torch.Tensor, hip_idx: int, knee_idx: int) -> torch.Tensor:
    """
    (T, 2) — [flexion, abduction] in degrees, expressed in the pelvis frame.
      flex > 0 : forward flexion
      abd  > 0 : abduction (away from midline)
    """
    R = _pelvis_frame(joints)                                          # (T, 3, 3)
    thigh = F.normalize(joints[:, knee_idx] - joints[:, hip_idx], dim=-1)  # (T, 3)
    local = torch.bmm(R.transpose(1, 2), thigh.unsqueeze(-1)).squeeze(-1)  # R^T @ t

    flex = torch.atan2( local[:, 2], -local[:, 1]) * _RAD2DEG
    abd  = torch.atan2( local[:, 0], -local[:, 1]) * _RAD2DEG
    return torch.stack([flex, abd], dim=-1)


def extract_angles(pose_72: np.ndarray, beta_10: np.ndarray) -> np.ndarray:
    """
    SMPL FK → 8 geometric joint angles for one walk.

    Returns (T, 8) float32 array, degrees:
      [L_Hip_Flex, L_Hip_Abd, R_Hip_Flex, R_Hip_Abd,
       L_Knee_Flex, R_Knee_Flex, L_Ankle_Flex, R_Ankle_Flex]
    """
    joints_np = smpl_joint_positions(pose_72, beta_10)   # (T, 24, 3)
    joints    = torch.from_numpy(joints_np).float()       # keep in torch — avoids FakeCh contamination

    l_hip  = _hip_angles(joints, _L_HIP,  _L_KNEE)       # (T, 2)
    r_hip  = _hip_angles(joints, _R_HIP,  _R_KNEE)       # (T, 2)

    l_knee = _flex_angle(joints[:, _L_HIP],   joints[:, _L_KNEE],  joints[:, _L_ANKLE])   # (T,)
    r_knee = _flex_angle(joints[:, _R_HIP],   joints[:, _R_KNEE],  joints[:, _R_ANKLE])
    l_ank  = _flex_angle(joints[:, _L_KNEE],  joints[:, _L_ANKLE], joints[:, _L_FOOT]) - 90.0
    r_ank  = _flex_angle(joints[:, _R_KNEE],  joints[:, _R_ANKLE], joints[:, _R_FOOT]) - 90.0

    out = torch.cat([
        l_hip, r_hip,
        l_knee.unsqueeze(1), r_knee.unsqueeze(1),
        l_ank.unsqueeze(1),  r_ank.unsqueeze(1),
    ], dim=1)                                              # (T, 8)
    return out.numpy().astype(np.float32)


# ── FPS resampling ────────────────────────────────────────────────────────────

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


# ── Main ──────────────────────────────────────────────────────────────────────

def process_cohort(cohort: str, max_walks: int = None) -> dict:
    """
    Process all walks in a cohort PKL.
    Returns {subject_id: {walk_id: {'angles': (T, 8), 'updrs': int}}}.
    """
    pkl_path = os.path.join(CAREPD_RAW_DIR, f"{cohort}.pkl")
    if not os.path.exists(pkl_path):
        print(f"  {cohort}: PKL not found, skipping.")
        return {}

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')

    result = {}
    n_ok, n_skip, n_fail = 0, 0, 0
    n_processed = 0

    for subject_id, subject_walks in data.items():
        sid = str(subject_id)
        result[sid] = {}

        for walk_id, walk in subject_walks.items():
            wid = str(walk_id)

            updrs = walk.get('UPDRS_GAIT')
            if updrs is None:
                n_skip += 1
                continue

            updrs = int(updrs)
            if updrs > UPDRS_MAX_SCORE:
                n_skip += 1
                continue

            if max_walks is not None and n_processed >= max_walks:
                break

            # Resample pose to target FPS before FK
            pose = resample(walk['pose'], walk['fps'], CAREPD_TARGET_FPS)

            try:
                angles = extract_angles(pose, walk['beta'])
            except Exception as e:
                print(f"  FAIL [{sid}] {wid}: {e}")
                n_fail += 1
                continue

            result[sid][wid] = {'angles': angles, 'updrs': updrs}
            n_ok += 1
            n_processed += 1

            if n_ok % 50 == 0:
                print(f"  [{cohort}] {n_ok} walks done...")

        if max_walks is not None and n_processed >= max_walks:
            break

    print(f"  [{cohort}] done: {n_ok} ok, {n_skip} skipped, {n_fail} failed")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="CARE-PD SMPL → geometric joint angles (Stage 1)"
    )
    parser.add_argument("--cohort", type=str, default=None,
                        help="Restrict to one cohort (e.g. BMCLab)")
    parser.add_argument("--test",   action="store_true",
                        help="Process only first 10 walks per cohort")
    args = parser.parse_args()

    if not os.path.exists(SMPL_MODEL_PATH):
        print(f"ERROR: SMPL model not found at {SMPL_MODEL_PATH}")
        print("Download from https://smpl.is.tue.mpg.de/")
        sys.exit(1)

    cohorts   = [args.cohort] if args.cohort else list(CAREPD_LABELED_COHORTS)
    max_walks = 10 if args.test else None

    print("=" * 60)
    print("CARE-PD SMPL → geometric angles  (Stage 1)")
    print(f"  Cohorts   : {cohorts}")
    print(f"  Target FPS: {CAREPD_TARGET_FPS}")
    print(f"  Max walks : {max_walks or 'all'}")
    print("=" * 60)

    # Load existing output if present (to resume / merge)
    if os.path.exists(ANGLES_PATH):
        print(f"\nLoading existing angles from {ANGLES_PATH} (will merge)...")
        with open(ANGLES_PATH, 'rb') as f:
            all_angles = pickle.load(f)
    else:
        all_angles = {}

    for cohort in cohorts:
        print(f"\n── {cohort} ──")
        cohort_result = process_cohort(cohort, max_walks)
        all_angles[cohort] = cohort_result

    os.makedirs(os.path.dirname(ANGLES_PATH), exist_ok=True)

    def _sanitize(obj):
        """Recursively convert ndarray subclasses (e.g. _Ch) and mismatched ndarrays to plain np.ndarray."""
        # Check if it's an ndarray, a _Ch instance, or anything that thinks it's an ndarray
        if (isinstance(obj, np.ndarray) or type(obj).__name__ in ["ndarray", "_Ch"]) and type(obj) is not np.ndarray:
            return np.array(obj)
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(_sanitize(v) for v in obj)
        return obj

    # Restore original copyreg before saving so ndarray pickling works correctly.
    copyreg._reconstructor = _copyreg_orig
    with open(ANGLES_PATH, 'wb') as f:
        pickle.dump(_sanitize(all_angles), f)
    copyreg._reconstructor = _copyreg_patched


    # Summary
    total = sum(
        len(walks)
        for cohort_data in all_angles.values()
        for walks in cohort_data.values()
    )
    print(f"\nSaved {total} walks → {ANGLES_PATH}")
    print("=" * 60)
    print("Run preprocess_carepd.py next to build training arrays.")


if __name__ == "__main__":
    main()
