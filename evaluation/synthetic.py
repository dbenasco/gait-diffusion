"""
BIOMECHANICS EVALUATION TOOL
----------------------------
This script evaluates synthetic gait data against real-world biomechanical standards and reference data.

METRICS DESCRIPTION:
1. Skating Score: Measures foot plant stability. Penalizes foot velocity when the foot is on 
   the ground. (Score 1.0 = No sliding/skating).
2. Floating_Score: Penalizes 'hovering'. Measured by the distance of the lowest foot to 
   the ground level during stance. (Score 1.0 = No floating).
3. Penetration_Score: Penalizes floor clipping. Measured by how much the foot goes below 
   the Z=0 plane. (Score 1.0 = No penetration).
4. Mean_Correlation: Pearson Correlation Coefficient (R) averaged across lower 
   body joints. Measures if the timing/trend of the movement is correct.
5. Mean_RMSE: Root Mean Square Error averaged across joints. Measures 
   absolute distance in meters/radians. (Lower is better).
6. Mean_NRMSE: Normalized RMSE. Measures error relative to the total range 
   of motion. (0.1 = 10% error, 1.0 = 100% error).

JOINT DEFINITIONS (H1 Robot - Focus on Lower Body):
- Root_X/Y/Z: Global position of the robot pelvis (meters).
- Root_Qw/x/y/z: Global orientation of the pelvis (Quaternion).
- Hip_Yaw: Internal/External rotation of the leg.
- Hip_Roll: Abduction/Adduction of the leg.
- Hip_Pitch: Flexion/Extension of the hip.
- Knee: Flexion of the knee joint.
- Ankle: Dorsiflexion/Plantarflexion of the foot.
- Torso: Rotation of the upper body relative to the pelvis.
"""
import numpy as np
import torch
import os
import sys
import matplotlib.pyplot as plt

try:
    import mujoco
    _MUJOCO_AVAILABLE = True
except ImportError:
    _MUJOCO_AVAILABLE = False
    mujoco = None

# --- CONFIGURATION ---
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--updrs", action="store_true", default=True, help="Use UPDRS mode")
parser.add_argument("--n_samples", type=int, default=300,
                    help="Max synthetic samples per class (default: 300)")
parser.add_argument("--gen_path", type=str, default=None,
                    help="Path to generated .npy file (overrides UPDRS_GEN_OUTPUT_PATH env var)")
parser.add_argument("--vae_path", type=str, default=None,
                    help="Path to VAE checkpoint (overrides UPDRS_VAE_MODEL_PATH env var)")
args, _ = parser.parse_known_args()

if args.gen_path:
    os.environ["UPDRS_GEN_OUTPUT_PATH"] = args.gen_path
if args.vae_path:
    os.environ["UPDRS_VAE_MODEL_PATH"] = args.vae_path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    EVAL_DATA_PATH as REAL_DATA_PATH,
    EVAL_OUTPUT_DIR,
    GEN_OUTPUT_PATH,
    EVAL_LABELS_PATH as REAL_LABELS_PATH,
    UPDRS_JOINT_NAMES as JOINT_NAMES,
    TRAIN_LATERALITY_PATH,
    EVAL_LATERALITY_PATH,
    VAE_MODEL_PATH,
    NORM_PARAMS_PATH,
    N_CHANNELS,
    LATENT_CHANNELS,
    UPDRS_CLASSES as N_UPDRS_CLASSES,
)
from training.vae_updrs import GaitVAE
from h3d_bridge import (
    h3d_to_angles as _h3d_to_angles,
    h3d_to_arm_timeseries as _h3d_to_arm,
    h3d_to_positions22 as _h3d_to_positions22,
    ARM_JOINT_NAMES as _ARM_JOINT_NAMES,
)
USE_MAGNITUDE_COND = False
USE_HYBRID_TRAINING = False
USE_DPPO = False
USE_DIT = True
ROBOT_XML = None
FPS = 30.0
SYNTETHIC_DATA_PATH = GEN_OUTPUT_PATH
OUTPUT_DIR = EVAL_OUTPUT_DIR

print("\n" + "="*60)
print("📊 GAIT EVALUATION CONFIGURATION")
print("="*60)
print(f"🔹 Syntethic Data:   {SYNTETHIC_DATA_PATH}")
print(f"🔹 Real Reference:   {REAL_DATA_PATH}  (eval split only)")
print(f"🔹 Output Dir:       {OUTPUT_DIR}")
print("-" * 60)
print(f"🔸 Magnitude Cond:   {'✅ ON' if USE_MAGNITUDE_COND else '❌ OFF'}")
print(f"🔸 DiT CFG Mode:     {'✅ ON' if USE_DIT else '❌ OFF'}")
print(f"🔸 Hybrid Training:  {'✅ ON' if USE_HYBRID_TRAINING else '❌ OFF'}")
print(f"🔸 DPPO Mode:        {'✅ ON' if USE_DPPO else '❌ OFF'}")
print("="*60 + "\n")

try:
    from numba import njit as _njit

    @_njit(cache=True)
    def _dtw_1d_kernel(a, b, window):
        n, m = len(a), len(b)
        if window < 0:          # sentinel: unconstrained
            window = n + m
        if window < abs(n - m):
            window = abs(n - m)
        cost = np.full((n + 1, m + 1), np.inf)
        cost[0, 0] = 0.0
        for i in range(1, n + 1):
            j_lo = max(1, i - window)
            j_hi = min(m, i + window) + 1
            for j in range(j_lo, j_hi):
                c = abs(a[i - 1] - b[j - 1])
                best = cost[i-1, j]
                if cost[i, j-1] < best:
                    best = cost[i, j-1]
                if cost[i-1, j-1] < best:
                    best = cost[i-1, j-1]
                cost[i, j] = c + best
        return cost[n, m] / (n + m)

    _dtw_1d_kernel(np.zeros(4, dtype=np.float64), np.zeros(4, dtype=np.float64), 2)

    def _dtw_1d(a: np.ndarray, b: np.ndarray, window: int = None) -> float:
        w = window if window is not None else -1
        return float(_dtw_1d_kernel(a.astype(np.float64), b.astype(np.float64), w))

    print("⚡ numba JIT enabled for DTW")

except ImportError:
    def _dtw_1d(a: np.ndarray, b: np.ndarray, window: int = None) -> float:
        """DTW distance between two 1D sequences, normalized by path length."""
        n, m = len(a), len(b)
        if window is None:
            window = max(n, m)
        window = max(window, abs(n - m))
        cost = np.full((n + 1, m + 1), np.inf)
        cost[0, 0] = 0.0
        for i in range(1, n + 1):
            for j in range(max(1, i - window), min(m, i + window) + 1):
                c = abs(float(a[i - 1]) - float(b[j - 1]))
                cost[i, j] = c + min(cost[i-1, j], cost[i, j-1], cost[i-1, j-1])
        return float(cost[n, m]) / (n + m)

    print("⚠️  numba not found — using pure-Python DTW (slow). Run: pip install numba")


# Left/Right sagittal joint pairs for 6-channel UPDRS layout
# (L_Hip_Flex, R_Hip_Flex), (L_Knee_Flex, R_Knee_Flex), (L_Ankle_Flex, R_Ankle_Flex)
_LR_PAIRS = [(0, 1), (2, 3), (4, 5)]


def _local_peaks(signal):
    d = np.diff(signal)
    idx = np.where((d[:-1] > 0) & (d[1:] <= 0))[0] + 1
    return signal[idx], idx


def _cov_interval(signal):
    """CoV of peak-to-peak intervals — stride timing variability proxy.

    Measures how regularly spaced the peaks are in time (frames).
    UPDRS 2 patients have less regular cadence → higher CoV.
    Returns NaN when fewer than 3 peaks exist (not enough intervals to measure).
    """
    _, peak_idx = _local_peaks(signal)
    if len(peak_idx) < 3:   # need ≥2 intervals
        return np.nan
    intervals = np.diff(peak_idx).astype(float)
    m = intervals.mean()
    return float(intervals.std() / m) if m > 1e-6 else np.nan


def _symmetry_index(rom_l, rom_r):
    """SI = 2|L-R|/(L+R)*100. 0 = perfect symmetry."""
    denom = rom_l + rom_r
    return float(2 * abs(rom_l - rom_r) / denom * 100) if denom > 1e-6 else 0.0


def _si_laterality_aware(data_nct, lat_labels):
    """
    Compute SI per laterality subgroup so the L/R mirroring augmentation
    doesn't cancel out asymmetry when mixing all samples.

    For lat=0 (Left-impaired)  the impaired side is L → expect ROM_L < ROM_R.
    For lat=1 (Right-impaired) the impaired side is R → expect ROM_R < ROM_L.
    For lat=2 (Symmetric) SI is expected to be near 0.

    Returns a dict keyed by laterality id:
        {'si': float,           # unsigned SI averaged over Hip/Knee/Ankle pairs
         'correct_dir': float}  # fraction of pairs where impaired side has lower ROM
                                # (None for symmetric group)
    """
    _LR = [(0, 1), (2, 3), (4, 5)]   # (L_ch, R_ch) for Hip, Knee, Ankle (6-channel layout)
    results = {}
    for lat in np.unique(lat_labels):
        subset = data_nct[lat_labels == lat]
        if len(subset) == 0:
            continue
        mean_rom = (subset.max(axis=2) - subset.min(axis=2)).mean(axis=0)  # (C,)
        si_vals, dirs = [], []
        for l_ch, r_ch in _LR:
            if l_ch >= data_nct.shape[1] or r_ch >= data_nct.shape[1]:
                continue
            rl, rr = mean_rom[l_ch], mean_rom[r_ch]
            denom = rl + rr
            si_vals.append(float(2 * abs(rl - rr) / denom * 100) if denom > 1e-6 else 0.0)
            if lat == 0:    # left impaired → left ROM should be lower
                dirs.append(rl < rr)
            elif lat == 1:  # right impaired → right ROM should be lower
                dirs.append(rr < rl)
        results[int(lat)] = {
            'si':           float(np.mean(si_vals)) if si_vals else 0.0,
            'correct_dir':  float(np.mean(dirs))    if dirs    else None,
        }
    return results


def _extract_3d_features(h3d_nct):
    """(N, 263, T) H3D → (N, 132) 3D joint features.
    Per-joint temporal mean (66) + std (66) over all 22 SMPL joints in metres.
    Uniform units, all joints contribute equally — arms, trunk, root included."""
    pos = _h3d_to_positions22(
        torch.from_numpy(np.ascontiguousarray(h3d_nct).astype(np.float32))
    ).numpy()                                        # (N, T, 22, 3)
    mean3d = pos.mean(axis=1).reshape(len(h3d_nct), -1)   # (N, 66)
    std3d  = pos.std(axis=1).reshape(len(h3d_nct), -1)    # (N, 66)
    return np.concatenate([mean3d, std3d], axis=1).astype(np.float64)  # (N, 132)


def _r_mean_laterality_aware(synth_nct, real_nct, s_lat, r_lat):
    """R(mean-vs-mean) per laterality group, then averaged.
    Avoids the flat-mean problem when L/R-impaired subjects are mixed.
    Inputs: (N, C, T) arrays and matching laterality label vectors."""
    n_ch = synth_nct.shape[1]
    lat_ids = np.unique(np.concatenate([s_lat, r_lat]))
    r_per_lat = []
    for lat in lat_ids:
        sm = synth_nct[s_lat == lat]
        rm = real_nct[r_lat == lat]
        if len(sm) == 0 or len(rm) == 0:
            continue
        s_mean = sm.mean(axis=0)  # (C, T)
        r_mean = rm.mean(axis=0)  # (C, T)
        r_ch = np.zeros(n_ch)
        for c in range(n_ch):
            if np.std(s_mean[c]) > 1e-6 and np.std(r_mean[c]) > 1e-6:
                r_ch[c] = float(np.corrcoef(s_mean[c], r_mean[c])[0, 1])
        r_per_lat.append(r_ch)
    return np.mean(r_per_lat, axis=0) if r_per_lat else np.zeros(n_ch)


def _dtw_laterality_aware(synth_nct, real_nct, s_lat, r_lat, band: float = 0.10):
    """DTW(mean-vs-mean) per laterality group, then averaged.
    Same approach as _r_mean_laterality_aware — avoids the mirror-augmentation
    bias where L/R means cancel each other out when mixed together.
    Inputs: (N, C, T) arrays and matching laterality label vectors.
    Returns dtw_mean (C,) and diversity (C,) arrays."""
    n_ch = synth_nct.shape[1]
    T    = synth_nct.shape[2]
    window = max(1, int(band * T))

    lat_ids = np.unique(np.concatenate([s_lat, r_lat]))
    dtw_per_lat = []
    div_per_lat = []

    for lat in lat_ids:
        sm = synth_nct[s_lat == lat]   # (N_s, C, T)
        rm = real_nct[r_lat == lat]    # (N_r, C, T)
        if len(sm) == 0 or len(rm) == 0:
            continue

        s_mean = sm.mean(axis=0)   # (C, T)
        r_mean = rm.mean(axis=0)

        dtw_ch = np.array([_dtw_1d(s_mean[c], r_mean[c], window=window)
                           for c in range(n_ch)])
        dtw_per_lat.append(dtw_ch)

        # Diversity: mean pairwise DTW within this laterality subgroup of synth
        N   = len(sm)
        div = np.zeros(n_ch)
        n_pairs = 0
        for i in range(N):
            for j in range(i + 1, N):
                for c in range(n_ch):
                    div[c] += _dtw_1d(sm[i, c], sm[j, c], window=window)
                n_pairs += 1
        if n_pairs > 0:
            div /= n_pairs
        div_per_lat.append(div)

    dtw_mean = np.mean(dtw_per_lat, axis=0) if dtw_per_lat else np.zeros(n_ch)
    diversity = np.mean(div_per_lat, axis=0) if div_per_lat else np.zeros(n_ch)
    return dtw_mean, diversity


def _train_knn_classifier(real_h3d, real_labels, k=5, seed=42):
    """real_h3d: (N, 263, T). Returns (train_feats, train_labels, min_n) for KNN lookup.
    Features: 3D joint mean+std (132-dim, all 22 joints, metres). Subsampled to min class count."""
    rng     = np.random.default_rng(seed)
    classes = np.unique(real_labels)
    min_n   = int(min((real_labels == c).sum() for c in classes))
    idx     = np.concatenate([
        rng.choice(np.where(real_labels == c)[0], size=min_n, replace=False)
        for c in classes
    ])
    feats = _extract_3d_features(real_h3d[idx])
    return feats, real_labels[idx].copy(), min_n


def _classify_knn(knn_data, synth_h3d, k=5):
    """Majority-vote KNN (k=5) in 3D joint position feature space (all 22 joints)."""
    train_feats, train_labels, _ = knn_data
    feats = _extract_3d_features(synth_h3d)
    # (N_synth, N_train) distance matrix
    dists = np.linalg.norm(feats[:, None, :] - train_feats[None, :, :], axis=2)
    nn_idx = np.argpartition(dists, k, axis=1)[:, :k]
    nn_labels = train_labels[nn_idx]
    preds = np.array([np.bincount(row, minlength=int(train_labels.max()) + 1).argmax()
                      for row in nn_labels])
    return preds


class BioMechanicsEvaluator:
    def __init__(self, motion_data, fps=60.0, robot_xml=None):
        self.raw_data = motion_data
        self.fps = fps
        self.dt = 1.0 / fps
        self.model = None
        self.data = None
        
        # Load MuJoCo Model for FK if XML provided
        if robot_xml and os.path.exists(robot_xml) and _MUJOCO_AVAILABLE:
            print(f"🤖 Loading MuJoCo Model: {robot_xml}")
            try:
                self.model = mujoco.MjModel.from_xml_path(robot_xml)
                self.data = mujoco.MjData(self.model)
                self.id_l_foot = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'left_ankle_link')
                self.id_r_foot = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'right_ankle_link')
            except Exception as e:
                print(f"⚠️ Warning: FK Setup failed ({e}). Physics metrics may be skipped.")

        self.H_GROUND = 0.05
        self.H_PENETRATE = 0.00
        self.VEL_THRESH = 0.02

    def _format_batch(self, data):
        """ Standardize to (Batch, Time, Channels) and Clean NaNs """
        if data.ndim == 2: data = data[np.newaxis, ...]
        elif data.ndim == 3 and data.shape[1] > data.shape[2]: data = data.transpose(0, 2, 1)
        elif data.ndim == 3 and data.shape[1] < data.shape[2]: data = data.transpose(0, 2, 1)
        
        if np.isnan(data).any():
            print(f"🩹 Detected NaNs. Imputing Mean...")
            data = self._clean_nans(data)
        return data

    def _clean_nans(self, batch_data):
        """ Fills NaNs with the mean value of that specific channel. """
        for i in range(batch_data.shape[0]):
            sample = batch_data[i]
            col_mean = np.nanmean(sample, axis=0) 
            inds = np.where(np.isnan(sample))
            if len(inds[0]) > 0: sample[inds] = np.take(col_mean, inds[1])
            if np.isnan(sample).any(): sample = np.nan_to_num(sample, nan=0.0)
            batch_data[i] = sample
        return batch_data

    def _compute_fk(self, qpos_sequence):
        if self.model is None: return None
        T = qpos_sequence.shape[0]
        feet_pos = np.zeros((T, 2, 3)) 
        
        # URJC Mapping Indices (Approximate for H1)
        # URJC: 0-2 Pelvis, 3-5 L_Hip, 6-8 L_Knee, 9-11 L_Ankle, 12-14 R_Hip...
        # H1 qpos: 0-6 Root, 7 L_Hip_Yaw, 8 L_Hip_Roll, 9 L_Hip_Pitch, 10 L_Knee, 11 L_Ankle
        #          12 R_Hip_Yaw, 13 R_Hip_Roll, 14 R_Hip_Pitch, 15 R_Knee, 16 R_Ankle
        
        for t in range(T):
            current_q = qpos_sequence[t]
            full_qpos = np.zeros(self.model.nq)
            
            # 1. Root (Fixed Height Walking)
            full_qpos[2] = 0.98 # Z Height
            full_qpos[3] = 1.0  # Qw (Identity)
            
            # 2. Map Joints (URJC has 21 channels)
            
            # Convert degrees to radians for MuJoCo
            current_q_rad = current_q * (np.pi / 180.0)
            
            if len(current_q) == 21:
                # Left Leg
                full_qpos[7] = current_q_rad[5] * 0.5 # L_Hip_Yaw (Rot) - Scaled?
                full_qpos[8] = current_q_rad[4]       # L_Hip_Roll (Abd)
                full_qpos[9] = current_q_rad[3]       # L_Hip_Pitch (Flex)
                full_qpos[10] = current_q_rad[6]      # L_Knee (Flex)
                full_qpos[11] = current_q_rad[9]      # L_Ankle (Flex)
                
                # Right Leg
                full_qpos[12] = current_q_rad[14] * 0.5 # R_Hip_Yaw
                full_qpos[13] = current_q_rad[13]       # R_Hip_Roll
                full_qpos[14] = current_q_rad[12]       # R_Hip_Pitch
                full_qpos[15] = current_q_rad[15]       # R_Knee
                full_qpos[16] = current_q_rad[18]       # R_Ankle
                
            elif len(current_q) == 9:
                # 9 channels: [0:Pelvis, 1:List, 2:Rot, 3:L_Hip, 4:L_Knee, 5:L_Ankle, 6:R_Hip, 7:R_Knee, 8:R_Ankle]
                full_qpos[9] = current_q_rad[3]       # L_Hip_Pitch
                full_qpos[10] = current_q_rad[4]      # L_Knee
                full_qpos[11] = current_q_rad[5]      # L_Ankle
                full_qpos[14] = current_q_rad[6]      # R_Hip_Pitch
                full_qpos[15] = current_q_rad[7]      # R_Knee
                full_qpos[16] = current_q_rad[8]      # R_Ankle
            
            elif len(current_q) != self.model.nq:
                 if len(current_q) < self.model.nq: full_qpos = np.pad(current_q, (0, self.model.nq - len(current_q)))
                 else: full_qpos = current_q[:self.model.nq]
            else:
                full_qpos = current_q

            self.data.qpos[:] = full_qpos
            mujoco.mj_kinematics(self.model, self.data)
            feet_pos[t, 0] = self.data.xpos[self.id_l_foot]
            feet_pos[t, 1] = self.data.xpos[self.id_r_foot]
        return feet_pos

    def compute_physics_scores(self):
        data_batch = self._format_batch(self.raw_data)
        num_samples = data_batch.shape[0]
        batch_scores = {"Skating_Score": [], "Floating_Score": [], "Penetration_Score": []}
        
        for s_idx in range(num_samples):
            sample = data_batch[s_idx]
            feet_data = None
            
            if sample.shape[1] == 33:
                try: feet_data = sample.reshape(-1, 11, 3)[:, [3, 8], :]
                except: pass
            elif self.model is not None:
                feet_data = self._compute_fk(sample)

            if feet_data is None: continue

            # Auto-Grounding
            feet_data[:, :, 2] -= np.min(feet_data[:, :, 2])
            vel = np.zeros_like(feet_data)
            vel[1:] = (feet_data[1:] - feet_data[:-1]) / self.dt
            
            s_skate, s_float, s_pen = [], [], []
            for t in range(len(feet_data)):
                l_pos, r_pos = feet_data[t, 0], feet_data[t, 1]
                l_vel, r_vel = vel[t, 0], vel[t, 1]
                
                # Skating
                skate_err = 0.0
                for pos, v in zip([l_pos, r_pos], [l_vel, r_vel]):
                    if pos[2] < self.H_GROUND and np.linalg.norm(v) > self.VEL_THRESH:
                        skate_err += np.linalg.norm(v)**2
                s_skate.append(np.exp(-skate_err))
                
                # Penetration
                min_h = min(l_pos[2], r_pos[2])
                pen_err = 0.0
                if min_h < self.H_PENETRATE - 0.005: 
                    pen_err = (self.H_PENETRATE - min_h)**2
                s_pen.append(np.exp(-pen_err))
                
                # Floating
                float_err = 0.0
                if min_h > self.H_GROUND: float_err = (min_h - self.H_GROUND)**2
                s_float.append(np.exp(-float_err))

            batch_scores["Skating_Score"].append(np.nanmean(s_skate))
            batch_scores["Penetration_Score"].append(np.nanmean(s_pen))
            batch_scores["Floating_Score"].append(np.nanmean(s_float))

        if not batch_scores["Skating_Score"]: return {}
        return {k: np.nanmean(v) for k,v in batch_scores.items()}

    def compute_kinematics(self):
        """ Returns the raw distribution of metrics for all samples in the batch. """
        data = self._format_batch(self.raw_data)
        metrics = {"Speed": [], "Jerk": []}
        for i in range(data.shape[0]):
            sample = data[i]
            feet_data = None
            if sample.shape[1] == 33: feet_data = sample.reshape(-1, 11, 3)[:, [3, 8], :]
            elif self.model is not None: feet_data = self._compute_fk(sample)
                
            if feet_data is None: continue
            
            vel = np.diff(feet_data, axis=0) / self.dt
            acc = np.diff(vel, axis=0) / self.dt
            jerk = np.diff(acc, axis=0) / self.dt
            metrics["Speed"].append(np.mean(np.linalg.norm(vel, axis=2)))
            metrics["Jerk"].append(np.mean(np.linalg.norm(jerk, axis=2)))
        
        return metrics

    def compute_step_lengths(self):
        """ 
        Calculates step length for each sample using MuJoCo FK.
        Each sample = 1 step (t=0..100), so excursion IS step length.
        Returns: np.array of shape (N,) in cm.
        """
        data = self._format_batch(self.raw_data)
        lengths = []
        
        if self.model is None:
            print("⚠️ FK Model missing. Using 50cm fallback for lengths.")
            return np.full(data.shape[0], 50.0)

        for i in range(data.shape[0]):
            sample = data[i]
            feet_pos = self._compute_fk(sample) # (T, 2, 3)
                
            if feet_pos is None:
                lengths.append(50.0)
                continue
            
            # Use X excursion of both feet to be robust
            xl_traj = feet_pos[:, 0, 0] # Left foot X
            xr_traj = feet_pos[:, 1, 0] # Right foot X
            
            # Step length = max X excursion (root is fixed in FK)
            l_excursion = np.max(xl_traj) - np.min(xl_traj)
            r_excursion = np.max(xr_traj) - np.min(xr_traj)
            
            # Step Length = avg excursion (m to cm)
            avg_step = ((l_excursion + r_excursion) / 2.0) * 100.0
            lengths.append(avg_step)
        
        return np.array(lengths)

    def compute_correlation(self, real_data_raw):
        print("Computing Per-Sample Metrics...")
        synth = self._format_batch(self.raw_data)
        real = self._format_batch(real_data_raw)
        
        # Use all available channels (clip to the minimum of both just in case)
        n_channels = min(synth.shape[2], real.shape[2])
        synth = synth[:, :, :n_channels]
        real = real[:, :, :n_channels]
        
        # Real Mean (The reference template)
        mean_real = np.mean(real, axis=0) 
        T_synth = synth.shape[1]
        T_real = mean_real.shape[0]
        
        # Align Time
        if T_synth != T_real:
            from scipy.ndimage import zoom
            mean_real = zoom(mean_real, (T_synth / T_real, 1), order=1)
        
        # Initialize per-joint accumulators for the batch
        all_corrs = [] # (N, n_channels)
        all_rmses = []
        all_nrmses = []
        
        for i in range(synth.shape[0]):
            sample = synth[i]
            s_corrs, s_rmses, s_nrmses = [], [], []
            for c in range(n_channels):
                s, r = sample[:, c], mean_real[:, c]
                std_s, std_r = np.std(s), np.std(r)
                
                # Correlation
                if std_s < 1e-6 and std_r < 1e-6: corr = 1.0
                elif std_s < 1e-6 or std_r < 1e-6: corr = 0.0
                else: corr = np.corrcoef(s, r)[0, 1]
                
                rmse = np.sqrt(np.mean((s - r)**2))
                # Robust NRMSE: Normalize by standard deviation of the reference
                # This is more stable than Range for low-ROM joints
                nrmse = rmse / (std_r + 1e-4)
                
                s_corrs.append(corr)
                s_rmses.append(rmse)
                s_nrmses.append(nrmse)
            
            all_corrs.append(s_corrs)
            all_rmses.append(s_rmses)
            all_nrmses.append(s_nrmses)
            
        # Per-Joint Averages
        avg_corrs = np.nanmean(all_corrs, axis=0)
        avg_rmses = np.nanmean(all_rmses, axis=0)
        avg_nrmses = np.nanmean(all_nrmses, axis=0)
        
        return (np.mean(avg_corrs), np.mean(avg_rmses), np.mean(avg_nrmses),
                avg_corrs, avg_rmses, avg_nrmses, synth[:10], mean_real)

    def compute_class_metrics(self, real_data_raw):
        """
        UPDRS evaluation metrics vs a real-class reference.

        Kept  : ROM, R_mean_vs_mean, SI, CoV
        Dropped: R_sample_vs_mean, envelope overlap
        """
        synth = self._format_batch(self.raw_data)   # (N, T, C)
        real  = self._format_batch(real_data_raw)

        n_channels = min(synth.shape[2], real.shape[2])
        synth = synth[:, :, :n_channels]
        real  = real[:,  :, :n_channels]

        s_mean = np.mean(synth, axis=0)
        r_mean = np.mean(real,  axis=0)
        s_std  = np.std(synth,  axis=0)
        r_std  = np.std(real,   axis=0)

        T_s, T_r = s_mean.shape[0], r_mean.shape[0]
        if T_s != T_r:
            from scipy.ndimage import zoom
            scale  = T_s / T_r
            r_mean = zoom(r_mean, (scale, 1), order=1)
            r_std  = zoom(r_std,  (scale, 1), order=1)

        # 1. ROM: per-sample then averaged
        synth_nct = synth.transpose(0, 2, 1)   # (N, C, T)
        real_nct  = real.transpose(0, 2, 1)
        rom_synth = np.mean(synth_nct.max(axis=2) - synth_nct.min(axis=2), axis=0)
        rom_real  = np.mean(real_nct.max(axis=2)  - real_nct.min(axis=2),  axis=0)

        # 2. Shape fidelity — mean waveform correlation
        r_mean_vs_mean = np.zeros(n_channels)
        for c in range(n_channels):
            sc, rc = s_mean[:, c], r_mean[:, c]
            if np.std(sc) > 1e-6 and np.std(rc) > 1e-6:
                r_mean_vs_mean[c] = float(np.corrcoef(sc, rc)[0, 1])

        # 3. Symmetry Index (mean across L/R pairs)
        si_synth = np.mean([_symmetry_index(rom_synth[l], rom_synth[r])
                            for l, r in _LR_PAIRS if l < n_channels and r < n_channels])
        si_real  = np.mean([_symmetry_index(rom_real[l],  rom_real[r])
                            for l, r in _LR_PAIRS if l < n_channels and r < n_channels])

        # 4. CoV of peak-to-peak intervals (stride timing variability)
        # nanmean ignores windows with fewer than 3 peaks instead of inflating with zeros.
        cov_synth = np.nanmean([[_cov_interval(synth_nct[i, c, :]) for c in range(n_channels)]
                                for i in range(len(synth_nct))], axis=0)
        cov_real  = np.nanmean([[_cov_interval(real_nct[i, c, :])  for c in range(n_channels)]
                                for i in range(len(real_nct))],  axis=0)
        # Replace any remaining NaN (channel with no valid windows) with 0
        cov_synth = np.where(np.isfinite(cov_synth), cov_synth, 0.0)
        cov_real  = np.where(np.isfinite(cov_real),  cov_real,  0.0)

        # 5. Average Velocity Error — MAE between mean angular velocity profiles.
        # Velocity = frame-to-frame angle difference (°/frame at 30 fps).
        # Measures whether temporal dynamics match, not just amplitude.
        vel_synth = np.diff(synth_nct, axis=2)   # (N, C, T-1)
        vel_real  = np.diff(real_nct,  axis=2)   # (N, C, T-1)
        avg_vel_error = np.mean(
            np.abs(vel_synth.mean(axis=0) - vel_real.mean(axis=0)), axis=1
        )   # (C,) — per channel, lower is better

        return {
            'rom_synth':      rom_synth,
            'rom_real':       rom_real,
            'r_mean_vs_mean': r_mean_vs_mean,
            's_mean':         s_mean,
            'r_mean':         r_mean,
            's_std':          s_std,
            'r_std':          r_std,
            'si_synth':       si_synth,
            'si_real':        si_real,
            'cov_synth':      cov_synth,
            'cov_real':       cov_real,
            'avg_vel_error':  avg_vel_error,
            'synth_nct':      synth_nct,
            'real_nct':       real_nct,
        }

    def compute_dtw_metrics(self, real_data_raw, band: float = 0.10):
        """
        DTW-based metrics (phase-robust, Sakoe-Chiba band = 10% of seq).

        dtw_mean  — DTW(synth_mean, real_mean) per channel. Lower = better.
        diversity — mean pairwise DTW within generated samples. Higher = less mode collapse.
        """
        synth = self._format_batch(self.raw_data)
        real  = self._format_batch(real_data_raw)

        n_ch  = min(synth.shape[2], real.shape[2])
        synth = synth[:, :, :n_ch]
        real  = real[:,  :, :n_ch]

        T      = synth.shape[1]
        window = max(1, int(band * T))

        s_mean = np.mean(synth, axis=0)
        r_mean = np.mean(real,  axis=0)

        dtw_mean = np.array([
            _dtw_1d(s_mean[:, c], r_mean[:, c], window=window)
            for c in range(n_ch)
        ])

        N = synth.shape[0]
        diversity = np.zeros(n_ch)
        n_pairs = 0
        for i in range(N):
            for j in range(i + 1, N):
                for c in range(n_ch):
                    diversity[c] += _dtw_1d(synth[i, :, c], synth[j, :, c], window=window)
                n_pairs += 1
        if n_pairs > 0:
            diversity /= n_pairs

        return {
            'dtw_mean':  dtw_mean,
            'diversity': diversity,
            'window':    window,
        }

# --- PLOTTING ---
def save_graphs(scores, kin_real, kin_synth, sample_synth, m_real, corrs, rmses, nrmses, out_dir=OUTPUT_DIR, step_lengths=None):
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    
    # 1. Comprehensive Evaluation Report (All Metrics)
    if scores:
        metrics = list(scores.keys())
        values = list(scores.values())
        
        fig, ax = plt.subplots(figsize=(10, 6))
        # Color logic: Green for good correlation/physics, Blue for kinematics, Red for poor scores
        colors = []
        for m, v in zip(metrics, values):
            if "Score" in m or "Correlation" in m:
                colors.append('forestgreen' if v > 0.7 else 'crimson')
            else:
                colors.append('royalblue') # Kinematics like Speed/Jerk
        
        bars = ax.barh(metrics, values, color=colors, alpha=0.8)
        ax.set_xlim(0, 15)
        ax.set_title(f"Gait Bio-Fidelity Metrics")
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        for bar in bars:
            ax.text(bar.get_width()+0.01, bar.get_y()+bar.get_height()/2, f'{bar.get_width():.3f}', 
                    va='center', fontweight='bold')
            
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "evaluation_report.png"))
        plt.close()

    # (Removed: real_vs_synth_comparison box plot)

    # 3. Correlation Report (Still using Means for Overview)
    if m_real is not None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        m_synth_avg = np.mean(sample_synth, axis=0) # Calculate average for this plot
        sorted_idx = np.argsort(corrs)
        indices = [sorted_idx[0], sorted_idx[len(sorted_idx)//2], sorted_idx[-1]]
        titles = ["Worst Match", "Median Match", "Best Match"]
        
        for ax, idx, title in zip(axes, indices, titles):
            name = JOINT_NAMES[idx] if idx < len(JOINT_NAMES) else f"Ch {idx}"
            ax.plot(m_real[:, idx], 'b--', label='Real Mean', alpha=0.7, linewidth=2)
            ax.plot(m_synth_avg[:, idx], 'r-', label='Synth Mean', alpha=0.7, linewidth=2)
            ax.set_title(f"{title}: {name}\nAvg R={corrs[idx]:.3f} | Avg RMSE={rmses[idx]:.3f}", fontsize=11)
            ax.legend()
        plt.suptitle("Batch Average Comparison", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "correlation_report.png"))
        plt.close()

    # 4. Individual Trajectories (To show Variance/Style)
    if sample_synth is not None and m_real is not None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        sorted_idx = np.argsort(corrs)
        indices = [sorted_idx[0], sorted_idx[len(sorted_idx)//2], sorted_idx[-1]]
        
        for ax, idx, title in zip(axes, indices, titles):
            name = JOINT_NAMES[idx] if idx < len(JOINT_NAMES) else f"Ch {idx}"
            ax.plot(m_real[:, idx], 'k--', label='Real Mean', linewidth=3, alpha=0.8)
            for s_idx in range(min(5, len(sample_synth))):
                ax.plot(sample_synth[s_idx, :, idx], color='red', alpha=0.2, linewidth=1)
            ax.plot(np.mean(sample_synth, axis=0)[:, idx], 'r-', label='Synth Mean', linewidth=2)
            ax.set_title(f"Individual Variation: {name}", fontsize=12)
            ax.legend()
        plt.suptitle("Individual Synthetic Samples vs Real Reference", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "individual_trajectories.png"))
        plt.close()

    # 4b. PLOT ALL ANGLES
    if sample_synth is not None and m_real is not None:
        print("Generating individual plots for ALL angles...")
        all_plots_dir = os.path.join(out_dir, "all_joints")
        os.makedirs(all_plots_dir, exist_ok=True)
        
        for idx in range(min(sample_synth.shape[2], m_real.shape[1])):
            name = JOINT_NAMES[idx] if idx < len(JOINT_NAMES) else f"Ch_{idx}"
            safe_name = name.replace(" ", "_").replace("/", "-")
            
            plt.figure(figsize=(8, 5))
            plt.plot(m_real[:, idx], 'k--', label='Real Mean', linewidth=3, alpha=0.8)
            for s_idx in range(min(10, len(sample_synth))):
                plt.plot(sample_synth[s_idx, :, idx], color='red', alpha=0.15, linewidth=1)
            plt.plot(np.mean(sample_synth, axis=0)[:, idx], 'r-', label='Synth Mean', linewidth=2)
            
            plt.title(f"Trajectory Analysis: {name}\nR={corrs[idx]:.3f} | RMSE={rmses[idx]:.3f}")
            plt.xlabel("Time (%)")
            plt.ylabel("Angle (deg)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(all_plots_dir, f"joint_{idx}_{safe_name}.png"))
            plt.close()
    if m_real is not None:
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        x = np.arange(len(corrs))
        names = JOINT_NAMES[:len(corrs)] if len(corrs) <= len(JOINT_NAMES) else [f"Ch{i}" for i in range(len(corrs))]
        
        # Correlation Bar
        axes[0].bar(x, corrs, color='forestgreen', alpha=0.7)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(names, rotation=45, ha='right')
        axes[0].set_title("Average Correlation (R) per Joint")
        axes[0].axhline(0.7, color='red', linestyle='--', label='Good Threshold')
        axes[0].set_ylim(0, 1.1)
        
        # NRMSE Bar
        axes[1].bar(x, nrmses, color='orange', alpha=0.7)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(names, rotation=45, ha='right')
        axes[1].set_title("Relative Error (NRMSE) per Joint")
        axes[1].set_ylabel("Error % (1.0 = 100%)")
        axes[1].axhline(0.2, color='red', linestyle='--', label='High Error Threshold')
        axes[1].set_ylim(0, 15)
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "joint_metrics_comparison.png"))
        plt.close()

    # 4. Text Report
    # Define "good joints" - the channels with valid synthetic data
    if len(corrs) == 10:
        GOOD_JOINTS = list(range(10))
    else:
        GOOD_JOINTS = [0, 1, 2, 3, 6, 9, 12, 15, 18]  # 3 Pelvis + 6 Flexion
    
    with open(os.path.join(out_dir, "report.txt"), "w") as f:
        f.write("--- SUMMARY SCORES ---\n")
        for k, v in sorted(scores.items()):
            f.write(f"  {k:<20} | {v:.4f}\n")
        
        if step_lengths is not None and len(step_lengths) > 0:
            f.write("\n--- STEP LENGTH (MuJoCo FK) ---\n")
            f.write(f"  Mean:   {np.mean(step_lengths):.2f} cm\n")
            f.write(f"  Std:    {np.std(step_lengths):.2f} cm\n")
            f.write(f"  Min:    {np.min(step_lengths):.2f} cm\n")
            f.write(f"  Max:    {np.max(step_lengths):.2f} cm\n")
            f.write(f"  N:      {len(step_lengths)}\n")
        
        f.write("\n--- PER-JOINT METRICS (Good Joints Only) ---\n")
        for j in GOOD_JOINTS:
            if j < len(corrs):
                name = JOINT_NAMES[j] if j < len(JOINT_NAMES) else f"Ch {j}"
                f.write(f"  {name:<20} | R={corrs[j]:.3f}  RMSE={rmses[j]:.3f}  NRMSE={nrmses[j]:.3f}\n")
        f.write("\n--- ALL JOINTS ---\n")
        for j in range(len(corrs)):
            name = JOINT_NAMES[j] if j < len(JOINT_NAMES) else f"Ch {j}"
            f.write(f"  {name:<20} | R={corrs[j]:.3f}  RMSE={rmses[j]:.3f}  NRMSE={nrmses[j]:.3f}\n")
    print(f"📄 Report saved to {os.path.join(out_dir, 'report.txt')}")

def _save_updrs_class_plots(cls_label, metrics, dtw, out_dir, joint_names, waveform_note=""):
    """Saves ROM, R_mean, DTW, Diversity, SI, and CoV bar charts for one UPDRS class."""
    os.makedirs(out_dir, exist_ok=True)
    n = len(metrics['r_mean_vs_mean'])
    names = joint_names[:n] if n <= len(joint_names) else [f"Ch{i}" for i in range(n)]
    x = np.arange(n)

    fig, axes = plt.subplots(6, 1, figsize=(12, 24))

    # ROM comparison
    w = 0.35
    axes[0].bar(x - w/2, metrics['rom_real'],  w, label='Real',  color='steelblue', alpha=0.8)
    axes[0].bar(x + w/2, metrics['rom_synth'], w, label='Synth', color='tomato',    alpha=0.8)
    axes[0].set_xticks(x); axes[0].set_xticklabels(names, rotation=45, ha='right')
    axes[0].set_title(f"UPDRS {cls_label} — Per-Joint ROM (°)")
    axes[0].legend(); axes[0].grid(axis='y', alpha=0.3)

    # Shape fidelity: mean vs mean R
    colors_m = ['forestgreen' if v >= 0.7 else 'crimson' for v in metrics['r_mean_vs_mean']]
    axes[1].bar(x, metrics['r_mean_vs_mean'], color=colors_m, alpha=0.8)
    axes[1].axhline(0.7, color='k', linestyle='--', linewidth=1, label='R=0.7 threshold')
    axes[1].set_xticks(x); axes[1].set_xticklabels(names, rotation=45, ha='right')
    axes[1].set_title(f"UPDRS {cls_label} — R(mean-vs-mean): generated mean vs real mean")
    axes[1].set_ylim(-0.1, 1.1); axes[1].legend(); axes[1].grid(axis='y', alpha=0.3)

    # DTW mean-vs-mean (lower = better)
    axes[2].bar(x, dtw['dtw_mean'], color='darkorange', alpha=0.8)
    axes[2].set_xticks(x); axes[2].set_xticklabels(names, rotation=45, ha='right')
    axes[2].set_title(f"UPDRS {cls_label} — DTW(mean vs mean)  [lower = better, phase-robust]")
    axes[2].grid(axis='y', alpha=0.3)

    # Diversity (higher = less mode collapse)
    axes[3].bar(x, dtw['diversity'], color='mediumseagreen', alpha=0.8)
    axes[3].set_xticks(x); axes[3].set_xticklabels(names, rotation=45, ha='right')
    axes[3].set_title(f"UPDRS {cls_label} — Diversity (mean pairwise DTW, higher = less collapse)")
    axes[3].grid(axis='y', alpha=0.3)

    # CoV of peak amplitudes (stride-to-stride variability, per channel)
    axes[4].bar(x - w/2, metrics['cov_real'],  w, label='Real',  color='steelblue', alpha=0.8)
    axes[4].bar(x + w/2, metrics['cov_synth'], w, label='Synth', color='tomato',    alpha=0.8)
    axes[4].set_xticks(x); axes[4].set_xticklabels(names, rotation=45, ha='right')
    axes[4].set_title(f"UPDRS {cls_label} — Stride CoV (peak amplitude variability, higher = more irregular)")
    axes[4].legend(); axes[4].grid(axis='y', alpha=0.3)

    # Symmetry Index — single pair of bars (real vs synth, mean over L/R pairs)
    ax5 = axes[5]
    ax5.bar([0, 1], [metrics['si_real'], metrics['si_synth']],
            color=['steelblue', 'tomato'], alpha=0.8)
    ax5.set_xticks([0, 1]); ax5.set_xticklabels(['Real', 'Synth'])
    ax5.set_title(f"UPDRS {cls_label} — Symmetry Index (SI, %, mean over Hip/Knee/Ankle L-R pairs)")
    ax5.set_ylabel("SI (%)"); ax5.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"updrs{cls_label}_metrics.png"))
    plt.close()

    # Mean waveform overlays per channel — phase-aligned to L_Knee_Flex peak
    wave_dir = os.path.join(out_dir, "waveforms")
    os.makedirs(wave_dir, exist_ok=True)
    T = metrics['s_mean'].shape[0]
    t = np.linspace(0, 100, T)

    REF_CH     = 4        # L_Knee_Flex — most reliable peak reference
    TARGET_POS = T // 4   # pin the peak to 25% of the window

    s_mean_full = metrics['s_mean']   # (T, C)
    r_mean_full = metrics['r_mean']   # (T, C)

    # Compute per-source shift from the reference channel's peak
    if s_mean_full.shape[1] > REF_CH:
        s_shift = TARGET_POS - int(np.argmax(s_mean_full[:, REF_CH]))
    else:
        s_shift = 0
    if r_mean_full.shape[1] > REF_CH:
        r_shift = TARGET_POS - int(np.argmax(r_mean_full[:, REF_CH]))
    else:
        r_shift = 0

    s_mean_aligned = np.roll(s_mean_full, s_shift, axis=0)
    r_mean_aligned = np.roll(r_mean_full, r_shift, axis=0)

    for c in range(n):
        jname = names[c].replace(" ", "_").replace("/", "-")
        sm = s_mean_aligned[:, c]
        rm = r_mean_aligned[:, c]
        dtw_val = dtw['dtw_mean'][c] if c < len(dtw['dtw_mean']) else float('nan')
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(t, rm, 'steelblue', linewidth=2, label='Real mean')
        ax.plot(t, sm, 'tomato',    linewidth=2, label='Synth mean')
        ax.axvline(100 * TARGET_POS / T, color='k', linewidth=0.8,
                   linestyle=':', alpha=0.5, label='peak pin')
        ax.set_title(
            f"UPDRS {cls_label} — {names[c]}{waveform_note}  [phase-aligned]\n"
            f"DTW={dtw_val:.4f}  "
            f"CoV(synth)={metrics['cov_synth'][c]:.3f}  CoV(real)={metrics['cov_real'][c]:.3f}"
        )
        ax.set_xlabel("Time (frames, peak-aligned)"); ax.set_ylabel("Angle (°)")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(wave_dir, f"ch{c:02d}_{jname}.png"))
        plt.close()


def run_full_evaluation(synth_data, real_data, synth_labels=None, real_labels=None,
                        synth_laterality=None, real_laterality=None):
    """Runs the complete evaluation pipeline, with separate paths for UPDRS and URJC modes."""

    # ── UPDRS mode (no MuJoCo robot) ──────────────────────────────────────────
    if ROBOT_XML is None:
        _run_updrs_evaluation(synth_data, real_data, synth_labels, real_labels,
                              synth_laterality=synth_laterality, real_laterality=real_laterality)
        return

    # ── URJC / physics-based mode ─────────────────────────────────────────────
    # --- REAL DATA CLASSIFICATION ---
    if real_labels is not None:
        if isinstance(real_labels, dict) and 'step_len' in real_labels:
            r_classes = real_labels['step_len']
        else:
            r_classes = np.array(real_labels)
    else:
        print("🔍 POST-HOC: Real labels missing. Measuring reference data for classification...")
        evaluator_real_full = BioMechanicsEvaluator(real_data, fps=FPS, robot_xml=ROBOT_XML)
        real_lengths = evaluator_real_full.compute_step_lengths()
        r_classes = np.where(real_lengths < 55, 0, 1)
        print(f"   Measured Real: {np.sum(r_classes == 0)} Short, {np.sum(r_classes == 1)} Medium")

    # --- SYNTHETIC DATA CLASSIFICATION ---
    if synth_labels is not None:
        s_classes = synth_labels if synth_labels.ndim == 1 else synth_labels[:, 1]
    else:
        print("🔍 POST-HOC: Synthetic labels missing. Measuring samples for automatic classification...")
        evaluator_synth_full = BioMechanicsEvaluator(synth_data, fps=FPS, robot_xml=ROBOT_XML)
        measured_lengths = evaluator_synth_full.compute_step_lengths()
        s_classes = np.where(measured_lengths < 55, 0, 1)
        print(f"   Split Synth: {np.sum(s_classes == 0)} Short, {np.sum(s_classes == 1)} Medium")
        print(f"   Threshold: 55 cm (Measured mean: {np.mean(measured_lengths):.1f} cm)")

    all_classes = sorted(np.union1d(np.unique(s_classes), np.unique(r_classes)).tolist())
    print(f"📋 Classes to evaluate: {all_classes}")

    for cls in all_classes:
        cls_name = f"class_{int(cls)}"
        print(f"\n--- EVALUATING: {cls_name} ---")

        s_mask = (s_classes == cls)
        r_mask = (r_classes == cls)
        sub_synth = synth_data[s_mask]
        sub_real  = real_data[r_mask] if np.any(r_mask) else real_data

        if len(sub_synth) == 0:
            print(f"⏩ Skipping {cls_name}: No synthetic samples.")
            continue

        class_out_dir = os.path.join(OUTPUT_DIR, cls_name)
        os.makedirs(class_out_dir, exist_ok=True)

        evaluator = BioMechanicsEvaluator(sub_synth, fps=FPS, robot_xml=ROBOT_XML)

        print(f"[{cls_name}] Computing Step Lengths...")
        cls_step_lengths = evaluator.compute_step_lengths()
        print(f"   Step Length: mean={np.mean(cls_step_lengths):.1f} cm, std={np.std(cls_step_lengths):.1f} cm")

        print(f"[{cls_name}] Computing Physics Scores...")
        scores = evaluator.compute_physics_scores()
        scores["Step_Length_Mean_cm"] = np.mean(cls_step_lengths)
        scores["Step_Length_Std_cm"]  = np.std(cls_step_lengths)

        print(f"[{cls_name}] Computing Kinematics...")
        k_synth = evaluator.compute_kinematics()
        k_real  = BioMechanicsEvaluator(sub_real, fps=FPS, robot_xml=ROBOT_XML).compute_kinematics()

        avg_corr, avg_rmse, avg_nrmse, corrs, s_rmses, s_nrmses, sample_synth, m_real = \
            evaluator.compute_correlation(sub_real)
        scores["Mean_Correlation"] = avg_corr
        scores["Mean_RMSE"]        = avg_rmse
        scores["Mean_NRMSE"]       = avg_nrmse

        save_graphs(scores, k_real, k_synth, sample_synth, m_real, corrs, s_rmses, s_nrmses,
                    out_dir=class_out_dir, step_lengths=cls_step_lengths)

        print(f"✅ {cls_name.upper()} REPORT:")
        for k, v in sorted(scores.items()):
            print(f"   {k:<20} | {v:.4f}")


@torch.no_grad()
def _vae_roundtrip(real_nct, vae_model, mean_t, std_t):
    """Encode real (N, C, T) through VAE using μ (deterministic) then decode back.
    Returns (N, 6, T) sagittal angles in degrees."""
    x_t = torch.from_numpy(real_nct.astype(np.float32))
    x_n = torch.clamp((x_t - mean_t) / std_t, -4, 4)
    chunks = []
    for i in range(0, len(x_n), 64):
        _, mu, _ = vae_model.encode(x_n[i:i + 64])
        chunks.append(vae_model.decode(mu))
    recon_n   = torch.cat(chunks, dim=0)
    recon_deg = recon_n * std_t + mean_t
    return _h3d_to_angles(recon_deg, sagittal_only=True).numpy()  # (N, 6, T)



def _run_updrs_evaluation(synth_data, real_data, synth_labels, real_labels,
                          synth_laterality=None, real_laterality=None):
    """
    UPDRS-specific evaluation (no MuJoCo FK):
      1. Per-joint ROM vs real class reference
      2. Mean waveform correlation per channel (laterality-aware when labels available)
      3. Cross-class ROM separation test
    """
    # ── Class assignment ──────────────────────────────────────────────────────
    if synth_labels is not None:
        s_classes = synth_labels if synth_labels.ndim == 1 else synth_labels[:, 0]
    else:
        print("⚠️  No synthetic labels — cannot split by UPDRS class.")
        s_classes = np.zeros(len(synth_data), dtype=int)

    if real_labels is not None:
        r_classes = np.array(real_labels) if not isinstance(real_labels, np.ndarray) else real_labels
    else:
        print("⚠️  No real labels — using all real data as reference for every class.")
        r_classes = np.zeros(len(real_data), dtype=int)

    has_lat = synth_laterality is not None and real_laterality is not None
    if has_lat:
        s_lat_all = np.array(synth_laterality)
        r_lat_all = np.array(real_laterality)
        print("✅ Laterality labels loaded — R(mean) will be computed per-laterality subgroup.")
    else:
        print("⚠️  No laterality labels — R(mean) computed over all samples (may be flat for UPDRS 1/2).")

    cls_display = {0: "Normal", 1: "Mild", 2: "Moderate", 3: "Severe"}
    all_classes = sorted(np.union1d(np.unique(s_classes), np.unique(r_classes)).tolist())

    # H3D: bridge 263-dim features → 6 sagittal angles for all biomechanical metrics.
    # Original H3D arrays are kept as _synth_h3d / _real_h3d for VAE roundtrip input
    # (_vae_roundtrip bridges its own output, so it needs the raw H3D, not the sagittal slice).
    _to_t = lambda x: torch.from_numpy(np.ascontiguousarray(x).astype(np.float32))
    _synth_sag = _h3d_to_angles(_to_t(synth_data), sagittal_only=True).numpy()  # (N, 6, T)
    _real_sag  = _h3d_to_angles(_to_t(real_data),  sagittal_only=True).numpy()  # (N, 6, T)
    _synth_arm = _h3d_to_arm(_to_t(synth_data)).numpy()                          # (N, 6, T) metres
    _real_arm  = _h3d_to_arm(_to_t(real_data)).numpy()                           # (N, 6, T) metres

    # 22-joint 3D positions for full-body DTW (N, T, 22, 3)
    _synth_pos22 = _h3d_to_positions22(_to_t(synth_data)).numpy()
    _real_pos22  = _h3d_to_positions22(_to_t(real_data)).numpy()

    key_joints = list(range(_synth_sag.shape[1] if _synth_sag.ndim == 3 else _synth_sag.shape[2]))

    print(f"📋 UPDRS classes to evaluate: "
          + ", ".join(f"{c} ({cls_display.get(c,'?')})" for c in all_classes))

    # ── Load VAE for roundtrip reference ─────────────────────────────────────
    _vae_model = None
    _vae_mean_t = _vae_std_t = None
    if os.path.exists(VAE_MODEL_PATH) and os.path.exists(NORM_PARAMS_PATH):
        _norm       = torch.load(NORM_PARAMS_PATH, map_location='cpu')
        _vae_mean_t = _norm['mean'].float()
        _vae_std_t  = _norm['std'].float()
        _vae_state_dict = torch.load(VAE_MODEL_PATH, map_location='cpu')
        _vae_model  = GaitVAE(N_CHANNELS, LATENT_CHANNELS, N_UPDRS_CLASSES,
                              use_prototypes="prototypes" in _vae_state_dict)
        _vae_model.load_state_dict(_vae_state_dict)
        _vae_model.eval()
        print("✅ VAE loaded — waveform reference uses VAE roundtrip (encode→decode real data).")
    else:
        print("⚠️  VAE not found — falling back to raw real mean for waveform reference.")

    # Train KNN classifier on class-balanced real data (subsampled to min class count)
    n_classes = max(int(max(all_classes)) + 1, 3)
    knn_data = _train_knn_classifier(real_data, r_classes, k=5)

    per_class_metrics = {}
    W  = 10
    DW = 12

    for cls in all_classes:
        label  = cls_display.get(int(cls), f"UPDRS_{int(cls)}")
        s_mask = (s_classes == cls)
        r_mask = (r_classes == cls)

        # Sagittal slices for all metric computations
        sub_synth = _synth_sag[s_mask]                                              # (N, 6, T)
        sub_real  = _real_sag[r_mask] if np.any(r_mask) else _real_sag             # (N, 6, T)

        # Original H3D slice for VAE roundtrip input (encoder was trained on 263-dim H3D).
        sub_real_h3d = real_data[r_mask] if np.any(r_mask) else real_data

        # Arm joint Z-displacement slices (metres, pelvis-relative).
        sub_synth_arm = _synth_arm[s_mask]
        sub_real_arm  = _real_arm[r_mask] if np.any(r_mask) else _real_arm

        # 22-joint 3D positions → (N, T, 66) for full-body DTW
        _ns = len(sub_synth)
        sub_synth_pos66 = _synth_pos22[s_mask].reshape(_ns, 96, 66)
        _nr = len(sub_real)
        sub_real_pos66  = _real_pos22[r_mask].reshape(_nr, 96, 66) if np.any(r_mask) \
                          else _real_pos22.reshape(len(_real_pos22), 96, 66)

        if len(sub_synth) == 0:
            print(f"⏩ Skipping UPDRS {cls} ({label}): No synthetic samples.")
            continue

        print(f"\n{'='*68}")
        print(f"  UPDRS {cls} — {label}  ({len(sub_synth)} generated / {len(sub_real)} real)")
        print('='*68)

        # VAE roundtrip: encode real H3D → decode → bridge to sagittal (done inside _vae_roundtrip).
        # Both the reference and the generated samples have passed through the VAE decoder,
        # so any difference reflects generation quality, not raw inter-subject variability.
        if _vae_model is not None:
            sub_real_ref = _vae_roundtrip(sub_real_h3d, _vae_model, _vae_mean_t, _vae_std_t)
        else:
            sub_real_ref = sub_real   # fallback: sagittal real (N, 6, T)

        evaluator = BioMechanicsEvaluator(sub_synth, fps=FPS, robot_xml=None)
        m   = evaluator.compute_class_metrics(sub_real_ref)
        dtw = evaluator.compute_dtw_metrics(sub_real_ref)

        # 22-joint full-body DTW (mean over all 66 joint-position channels)
        evaluator_22j = BioMechanicsEvaluator(sub_synth_pos66, fps=FPS, robot_xml=None)
        dtw_22j = evaluator_22j.compute_dtw_metrics(sub_real_pos66)

        # Override R(mean) + waveform means with per-laterality computation when labels available
        waveform_note = "" if _vae_model is None else " [VAE roundtrip ref]"
        if has_lat:
            sub_s_lat = s_lat_all[s_mask]
            sub_r_lat = r_lat_all[r_mask] if np.any(r_mask) else r_lat_all
            # Both ss_nct and sr_nct are (N, 6, T) sagittal — no transpose needed
            ss_nct = sub_synth       # (N, 6, T)
            sr_nct = sub_real_ref    # already (N, 6, T) — VAE roundtrip or sagittal fallback
            m['r_mean_vs_mean'] = _r_mean_laterality_aware(ss_nct, sr_nct, sub_s_lat, sub_r_lat)

            # Override DTW with per-laterality version for the same reason as R(mean)
            dtw_mean_lat, diversity_lat = _dtw_laterality_aware(
                ss_nct, sr_nct, sub_s_lat, sub_r_lat)
            dtw['dtw_mean']  = dtw_mean_lat
            dtw['diversity'] = diversity_lat

            # Waveform plots: use a single laterality group so means are not flat
            # Pick lowest-id laterality present in synth (0=Left, 1=Right, 2=Sym)
            lat_for_plot = int(sorted(np.unique(sub_s_lat))[0])
            lat_names_map = {0: "Left-impaired", 1: "Right-impaired", 2: "Symmetric"}
            lat_tag = lat_names_map.get(lat_for_plot, f'lat={lat_for_plot}')
            waveform_note = f" [VAE roundtrip · {lat_tag}]" if _vae_model is not None \
                            else f" [{lat_tag}]"
            sm_plot = ss_nct[sub_s_lat == lat_for_plot]          # (N, C, T)
            rm_plot_mask = sub_r_lat == lat_for_plot
            rm_plot = sr_nct[rm_plot_mask] if np.any(rm_plot_mask) else sr_nct
            if len(sm_plot) > 0 and len(rm_plot) > 0:
                m['s_mean'] = sm_plot.transpose(0, 2, 1).mean(axis=0)  # (T, C)
                m['r_mean'] = rm_plot.transpose(0, 2, 1).mean(axis=0)
                m['s_std']  = sm_plot.transpose(0, 2, 1).std(axis=0)
                m['r_std']  = rm_plot.transpose(0, 2, 1).std(axis=0)

            # Override SI with per-laterality version.
            # Mixed-data SI is always ~0 because every window has a mirrored twin in the dataset.
            si_s = _si_laterality_aware(ss_nct, sub_s_lat)
            si_r = _si_laterality_aware(sr_nct, sub_r_lat)
            impaired_lats = [k for k in si_s if k != 2]
            if impaired_lats:
                m['si_synth']    = float(np.mean([si_s[k]['si'] for k in impaired_lats]))
                m['si_real']     = float(np.mean([si_r[k]['si'] for k in si_r if k != 2])) \
                                   if any(k != 2 for k in si_r) else 0.0
                m['si_lat_synth'] = si_s
                m['si_lat_real']  = si_r

        per_class_metrics[cls] = {**m, **{f'dtw_{k}': v for k, v in dtw.items()}}

        n_ch   = len(m['r_mean_vs_mean'])
        kj_idx = [j for j in key_joints if j < n_ch]

        # ROM + shape fidelity
        header    = f"  {'Joint':<20} {'Real ROM':>{W}} {'Synth ROM':>{W}} {'R(mean)':>{W}}"
        divider_w = 20 + 3 * (W + 1)
        print(header)
        print("  " + "─" * divider_w)
        for j in range(n_ch):
            name = JOINT_NAMES[j] if j < len(JOINT_NAMES) else f"Ch {j}"
            print(f"  {name:<20} "
                  f"{m['rom_real'][j]:>{W}.2f} "
                  f"{m['rom_synth'][j]:>{W}.2f} "
                  f"{m['r_mean_vs_mean'][j]:>{W}.3f}")
        print("  " + "─" * divider_w)
        print(f"  {'KEY JOINTS MEAN':<20} "
              f"{np.mean(m['rom_real'][kj_idx]):>{W}.2f} "
              f"{np.mean(m['rom_synth'][kj_idx]):>{W}.2f} "
              f"{np.mean(m['r_mean_vs_mean'][kj_idx]):>{W}.3f}")

        # DTW + Diversity + Avg Velocity Error
        VW = 12
        print(f"\n  DTW (Sakoe-Chiba band={dtw['window']}fr) | Div=intra-class pairwise DTW | AvgVelErr=°/frame")
        print(f"  {'Joint':<20} {'DTW(mean)':>{DW}} {'Diversity':>{DW}} {'AvgVelErr':>{VW}}")
        print("  " + "─" * (20 + 2 * (DW + 1) + VW + 1))
        for j in range(n_ch):
            name = JOINT_NAMES[j] if j < len(JOINT_NAMES) else f"Ch {j}"
            print(f"  {name:<20} "
                  f"{dtw['dtw_mean'][j]:>{DW}.4f} "
                  f"{dtw['diversity'][j]:>{DW}.4f} "
                  f"{m['avg_vel_error'][j]:>{VW}.4f}")
        print(f"  {'KEY JOINTS MEAN':<20} "
              f"{np.mean(dtw['dtw_mean'][kj_idx]):>{DW}.4f} "
              f"{np.mean(dtw['diversity'][kj_idx]):>{DW}.4f} "
              f"{np.mean(m['avg_vel_error'][kj_idx]):>{VW}.4f}")

        # 22-joint full-body DTW (mean over 22×3=66 position channels)
        dtw22_mean = float(np.mean(dtw_22j['dtw_mean']))
        print(f"\n  22-JOINT FULL-BODY DTW (mean over 22×3=66 channels): {dtw22_mean:.4f}")

        # Symmetry Index + CoV
        lat_names_print = {0: "Left-impaired", 1: "Right-impaired", 2: "Symmetric"}
        if 'si_lat_synth' in m:
            print(f"\n  Symmetry Index (SI, per-laterality group):")
            all_lats = sorted(set(m['si_lat_synth']) | set(m['si_lat_real']))
            for lat_id in all_lats:
                lname = lat_names_print.get(lat_id, f"lat={lat_id}")
                s_si  = m['si_lat_synth'].get(lat_id, {}).get('si', float('nan'))
                r_si  = m['si_lat_real'].get(lat_id,  {}).get('si', float('nan'))
                s_dir = m['si_lat_synth'].get(lat_id, {}).get('correct_dir')
                dir_str = "" if s_dir is None else \
                          f"  dir={s_dir*100:.0f}% correct {'✅' if s_dir >= 0.5 else '❌'}"
                print(f"    {lname:<18} real={r_si:5.1f}%  synth={s_si:5.1f}%{dir_str}")
        else:
            print(f"\n  Symmetry Index (SI): real={m['si_real']:.1f}%  synth={m['si_synth']:.1f}%")
        knee_ch = [c for c in [4, 5] if c < n_ch]
        cov_s = float(np.mean(m['cov_synth'][knee_ch])) if knee_ch else 0.0
        cov_r = float(np.mean(m['cov_real'][knee_ch]))  if knee_ch else 0.0
        print(f"  Stride CoV (knee):   real={cov_r:.3f}  synth={cov_s:.3f}")

        # Save plots and text report
        class_out_dir = os.path.join(OUTPUT_DIR, f"updrs_{int(cls)}_{label.lower()}")
        _save_updrs_class_plots(f"{int(cls)}_{label}", m, dtw, class_out_dir, JOINT_NAMES,
                                waveform_note=waveform_note)
        os.makedirs(class_out_dir, exist_ok=True)
        with open(os.path.join(class_out_dir, "report.txt"), "w") as f:
            f.write(f"UPDRS {cls} — {label}\n")
            f.write(f"  Generated: {len(sub_synth)}  Real: {len(sub_real)}\n\n")

            # ── Per-joint table ───────────────────────────────────────────────
            f.write(f"  {'Joint':<20} {'Real ROM':>10} {'Synth ROM':>10} {'R(mean)':>10} "
                    f"{'DTW':>10} {'Diversity':>10} {'AvgVelErr':>10}\n")
            f.write("  " + "─" * 76 + "\n")
            for j in range(n_ch):
                name = JOINT_NAMES[j] if j < len(JOINT_NAMES) else f"Ch {j}"
                f.write(f"  {name:<20} "
                        f"{m['rom_real'][j]:>10.2f} "
                        f"{m['rom_synth'][j]:>10.2f} "
                        f"{m['r_mean_vs_mean'][j]:>10.3f} "
                        f"{dtw['dtw_mean'][j]:>10.4f} "
                        f"{dtw['diversity'][j]:>10.4f} "
                        f"{m['avg_vel_error'][j]:>10.4f}\n")

            # ── Means across all joints ───────────────────────────────────────
            f.write("  " + "─" * 76 + "\n")
            f.write(f"  {'MEAN (all joints)':<20} "
                    f"{np.mean(m['rom_real']):>10.2f} "
                    f"{np.mean(m['rom_synth']):>10.2f} "
                    f"{np.mean(m['r_mean_vs_mean']):>10.3f} "
                    f"{np.mean(dtw['dtw_mean']):>10.4f} "
                    f"{np.mean(dtw['diversity']):>10.4f} "
                    f"{np.mean(m['avg_vel_error']):>10.4f}\n")
            f.write(f"  {'MEAN (key joints)':<20} "
                    f"{np.mean(m['rom_real'][kj_idx]):>10.2f} "
                    f"{np.mean(m['rom_synth'][kj_idx]):>10.2f} "
                    f"{np.mean(m['r_mean_vs_mean'][kj_idx]):>10.3f} "
                    f"{np.mean(dtw['dtw_mean'][kj_idx]):>10.4f} "
                    f"{np.mean(dtw['diversity'][kj_idx]):>10.4f} "
                    f"{np.mean(m['avg_vel_error'][kj_idx]):>10.4f}\n")

            # ── Scalar metrics ────────────────────────────────────────────────
            f.write(f"\n  SI synth={m['si_synth']:.1f}%  SI real={m['si_real']:.1f}%\n")
            f.write(f"  Stride CoV knee: synth={cov_s:.3f}  real={cov_r:.3f}\n")
            f.write(f"  CoV (mean all joints): synth={np.mean(m['cov_synth']):.3f}  "
                    f"real={np.mean(m['cov_real']):.3f}\n")
        print(f"  Report: {class_out_dir}/report.txt")

        # ── Arm joint metrics ─────────────────────────────────────────────────
        arm_evaluator = BioMechanicsEvaluator(sub_synth_arm, fps=FPS, robot_xml=None)
        arm_m   = arm_evaluator.compute_class_metrics(sub_real_arm)
        arm_dtw = arm_evaluator.compute_dtw_metrics(sub_real_arm)
        wrist_idx = [4, 5]
        print(f"\n  Arm joints (sagittal Z-displacement vs pelvis, metres):")
        print(f"  {'Joint':<20} {'Real ROM':>{W}} {'Synth ROM':>{W}} {'R(mean)':>{W}}")
        print("  " + "─" * (20 + 3 * (W + 1)))
        for j, jname in enumerate(_ARM_JOINT_NAMES):
            print(f"  {jname:<20} "
                  f"{arm_m['rom_real'][j]:>{W}.3f} "
                  f"{arm_m['rom_synth'][j]:>{W}.3f} "
                  f"{arm_m['r_mean_vs_mean'][j]:>{W}.3f}")
        print("  " + "─" * (20 + 3 * (W + 1)))
        print(f"  {'WRISTS MEAN':<20} "
              f"{np.mean(arm_m['rom_real'][wrist_idx]):>{W}.3f} "
              f"{np.mean(arm_m['rom_synth'][wrist_idx]):>{W}.3f} "
              f"{np.mean(arm_m['r_mean_vs_mean'][wrist_idx]):>{W}.3f}")
        print(f"\n  Arm DTW (Sakoe-Chiba band={arm_dtw['window']}fr):")
        print(f"  {'Joint':<20} {'DTW(mean)':>{DW}} {'Diversity':>{DW}}")
        print("  " + "─" * (20 + 2 * (DW + 1)))
        for j, jname in enumerate(_ARM_JOINT_NAMES):
            print(f"  {jname:<20} "
                  f"{arm_dtw['dtw_mean'][j]:>{DW}.4f} "
                  f"{arm_dtw['diversity'][j]:>{DW}.4f}")
        print(f"  {'WRISTS MEAN':<20} "
              f"{np.mean(arm_dtw['dtw_mean'][wrist_idx]):>{DW}.4f} "
              f"{np.mean(arm_dtw['diversity'][wrist_idx]):>{DW}.4f}")

        per_class_metrics[cls]['arm_swing_synth'] = float(np.mean(arm_m['rom_synth'][wrist_idx]))
        per_class_metrics[cls]['arm_swing_real']  = float(np.mean(arm_m['rom_real'][wrist_idx]))

    # Cross-class separation tests
    if len(per_class_metrics) >= 2:
        sorted_classes = sorted(per_class_metrics.keys())
        kj_idx = [j for j in key_joints
                  if j < len(next(iter(per_class_metrics.values()))['r_mean_vs_mean'])]

        print(f"\n{'='*64}")
        print(f"  CROSS-CLASS SEPARATION TESTS")
        print('='*64)

        # ROM
        key_roms = {cls: np.mean(per_class_metrics[cls]['rom_synth'][kj_idx])
                    for cls in sorted_classes}
        rom_ordered = all(key_roms[sorted_classes[i]] > key_roms[sorted_classes[i+1]]
                          for i in range(len(sorted_classes)-1))
        rom_vals   = "  ".join(f"UPDRS{c}={key_roms[c]:.2f}°" for c in sorted_classes)
        print(f"  ROM:  {rom_vals}  → {'✅ PASS' if rom_ordered else '❌ FAIL'}")

        # SI (UPDRS 0 should have lowest SI)
        key_si = {cls: per_class_metrics[cls]['si_synth'] for cls in sorted_classes}
        si_ok  = key_si[sorted_classes[0]] < key_si[sorted_classes[-1]]
        si_vals = "  ".join(f"UPDRS{c}={key_si[c]:.1f}%" for c in sorted_classes)
        print(f"  SI:   {si_vals}  → {'✅ normal most symmetric' if si_ok else '❌ FAIL'}")

        # CoV (UPDRS 2 should have highest CoV)
        knee_ch = [c for c in [4, 5] if c < len(next(iter(per_class_metrics.values()))['cov_synth'])]
        key_cov = {cls: float(np.mean(per_class_metrics[cls]['cov_synth'][knee_ch]))
                   for cls in sorted_classes}
        cov_ok  = key_cov[sorted_classes[0]] < key_cov[sorted_classes[-1]]
        cov_vals = "  ".join(f"UPDRS{c}={key_cov[c]:.3f}" for c in sorted_classes)
        print(f"  CoV:  {cov_vals}  → {'✅ normal most regular' if cov_ok else '❌ FAIL'}")

        # Arm swing (UPDRS 0 should have highest, U0>U2)
        if all('arm_swing_synth' in per_class_metrics[c] for c in sorted_classes):
            key_arm = {cls: per_class_metrics[cls]['arm_swing_synth'] for cls in sorted_classes}
            arm_ok  = key_arm[sorted_classes[0]] > key_arm[sorted_classes[-1]]
            arm_vals = "  ".join(f"UPDRS{c}={key_arm[c]:.3f}m" for c in sorted_classes)
            print(f"  Arm:  {arm_vals}  → {'✅ U0 largest arm swing' if arm_ok else '❌ FAIL'}")

        # UPDRS classifier test
        knn_min_n = knn_data[2]
        print(f"\n  UPDRS CLASSIFIER TEST (kNN k=5, balanced {knn_min_n}/class on real data):")
        clf_preds = _classify_knn(knn_data, synth_data, k=5)
        overall_clf = 100 * int((clf_preds == s_classes).sum()) / len(s_classes)
        for cls in sorted_classes:
            mask    = s_classes == cls
            correct = int((clf_preds[mask] == cls).sum())
            total   = int(mask.sum())
            acc     = 100 * correct / total if total > 0 else 0.0
            label   = cls_display.get(int(cls), f"UPDRS_{int(cls)}")
            print(f"  UPDRS {cls} ({label}): {correct}/{total} ({acc:.1f}%)  "
                  f"{'✅' if acc >= 50 else '⚠️'}")
        print(f"  Overall: {overall_clf:.1f}%  (chance=33.3%)")

        # Cross-class ROM bar chart
        fig, ax = plt.subplots(figsize=(8, 5))
        cls_labels_plot = [f"UPDRS {c}\n({cls_display.get(int(c),'?')})" for c in sorted_classes]
        real_roms  = [np.mean(per_class_metrics[c]['rom_real'][kj_idx])  for c in sorted_classes]
        synth_roms = [np.mean(per_class_metrics[c]['rom_synth'][kj_idx]) for c in sorted_classes]
        xp = np.arange(len(sorted_classes))
        w  = 0.35
        ax.bar(xp - w/2, real_roms,  w, label='Real',  color='steelblue', alpha=0.8)
        ax.bar(xp + w/2, synth_roms, w, label='Synth', color='tomato',    alpha=0.8)
        ax.set_xticks(xp); ax.set_xticklabels(cls_labels_plot)
        ax.set_title("Cross-Class ROM Separation (key sagittal joints)")
        ax.set_ylabel("Mean ROM (°)")
        ax.legend(); ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        sep_path = os.path.join(OUTPUT_DIR, "cross_class_separation.png")
        plt.savefig(sep_path); plt.close()
        print(f"\n  Cross-class plot: {sep_path}")

if __name__ == "__main__":
    print(f"--- Biomechanics Evaluator (Class-Specific) ---")
    synth_raw = np.load(SYNTETHIC_DATA_PATH) if os.path.exists(SYNTETHIC_DATA_PATH) else None
    if synth_raw is None:
        print(f"❌ Error: Synthetic data not found at {SYNTETHIC_DATA_PATH}")
        sys.exit(1)
    
    real_raw = np.load(REAL_DATA_PATH, allow_pickle=True) if os.path.exists(REAL_DATA_PATH) else None
    if isinstance(real_raw, dict): real_raw = real_raw['data']
    elif real_raw is not None and real_raw.shape == () and isinstance(real_raw.item(), dict): real_raw = real_raw.item()['data']
    
    # Load Labels
    real_labels = None
    if os.path.exists(REAL_LABELS_PATH):
        rl_raw = np.load(REAL_LABELS_PATH, allow_pickle=True)
        real_labels = rl_raw.item() if rl_raw.shape == () else rl_raw
    
    synth_labels_path = SYNTETHIC_DATA_PATH.replace(".npy", "_labels.npy")
    synth_labels = np.load(synth_labels_path) if os.path.exists(synth_labels_path) else None

    synth_lat_path = SYNTETHIC_DATA_PATH.replace(".npy", "_laterality.npy")
    synth_laterality = np.load(synth_lat_path) if os.path.exists(synth_lat_path) else None

    # Subsample synthetic data to n_samples per class
    if synth_labels is not None:
        rng = np.random.default_rng(42)
        keep = []
        for cls in np.unique(synth_labels):
            idx = np.where(synth_labels == cls)[0]
            keep.append(rng.choice(idx, size=min(args.n_samples, len(idx)), replace=False))
        keep = np.concatenate(keep)
        synth_raw        = synth_raw[keep]
        synth_labels     = synth_labels[keep]
        if synth_laterality is not None:
            synth_laterality = synth_laterality[keep]
        n_cls_kept = len(np.unique(synth_labels))
        actual_per_class = len(keep) // n_cls_kept if n_cls_kept > 0 else len(keep)
        print(f"⚡ Using {actual_per_class} synthetic samples per class ({len(keep)} total)")

    # Real laterality from eval split only (matches REAL_DATA_PATH)
    real_laterality = None
    if 'EVAL_LATERALITY_PATH' in dir() and os.path.exists(EVAL_LATERALITY_PATH):
        real_laterality = np.load(EVAL_LATERALITY_PATH)

    run_full_evaluation(synth_raw, real_raw, synth_labels=synth_labels, real_labels=real_labels,
                        synth_laterality=synth_laterality, real_laterality=real_laterality)
    
    print(f"\n✅ All reports generated in {OUTPUT_DIR}")