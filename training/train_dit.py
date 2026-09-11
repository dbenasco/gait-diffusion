import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.model_updrs_dit import DiffusionTransformerUPDRS
from training.vae_updrs import GaitVAE
from config import (
    TRAIN_DATA_PATH, TRAIN_LABELS_PATH, TRAIN_LATERALITY_PATH,
    EVAL_DATA_PATH, EVAL_LABELS_PATH, EVAL_LATERALITY_PATH,
    NORM_PARAMS_PATH, MODEL_PATH, STATS_PATH, EVAL_OUTPUT_DIR,
    VAE_MODEL_PATH,
    N_CHANNELS, SEQ_LEN, LATENT_CHANNELS, LATENT_TIME,
    EMBED_DIM, N_HEADS, N_LAYERS, DROPOUT,
    UPDRS_CLASSES,
    BATCH_SIZE, TIMESTEPS, LEARNING_RATE, EPOCHS, DEVICE,
    CFG_DROPOUT_PROB, GUIDANCE_SCALE,
    MSE_WEIGHT, VELOCITY_WEIGHT,
    ROM_WEIGHT, TARGET_ROM_DEG,
    ARM_SWING_WEIGHT, TRUNK_INCL_WEIGHT, TARGET_ARM_SWING, TARGET_TRUNK_INCL_DEG,
    RESUME_TRAINING, START_EPOCH, UPDRS_JOINT_NAMES,
    GEN_N_PER_CLASS, CLASS_WEIGHTED_LOSS,
    STEPSIZE_BINS, STEPSIZE_BIN_EDGES,
    TRAIN_STEPSIZE_PATH, EVAL_STEPSIZE_PATH,
)
from h3d_bridge import (
    h3d_to_angles, h3d_to_positions22, arm_swing_range_t, trunk_inclination_t,
    h3d_to_arm_timeseries, ARM_JOINT_NAMES,
)


# ============================================================
# Noise schedule (same as existing pipeline)
# ============================================================

def get_noise_schedule(beta_start=1e-4, beta_end=0.02, n_steps=500):
    return torch.linspace(beta_start, beta_end, n_steps)


# ============================================================
# Full-body ROM supervision (per-class batch-mean L1 vs real targets)
# ============================================================

def compute_fullbody_rom_loss(pred_z0, updrs_cls, cond_mask, vae, dataset, device):
    """
    Decode pred_z0 -> denormalized angle space -> per-UPDRS-class batch-mean ROM
    vs TARGET_ROM_DEG (L1), same mechanism as the pre-H3D ROM_WEIGHT loss. When
    USE_H3D, also supervises arm-swing range and trunk inclination (the two
    GaitGen full-body signals the 6-ch sagittal-leg representation didn't carry)
    against TARGET_ARM_SWING / TARGET_TRUNK_INCL_DEG.

    Returns (weighted_loss, rom_loss, arm_loss, trunk_loss) — all scalar tensors,
    0 for any term whose weight is 0 (skips the decode entirely if all three are).
    """
    zero = torch.tensor(0.0, device=device)
    if ROM_WEIGHT == 0.0 and ARM_SWING_WEIGHT == 0.0 and TRUNK_INCL_WEIGHT == 0.0:
        return zero, zero, zero, zero

    x_pred = vae.decode(pred_z0)
    avg_std_d  = dataset.avg_std.to(device).view(1, -1, 1)
    avg_mean_d = dataset.avg_mean.to(device).view(1, -1, 1)
    x_pred_deg = x_pred * avg_std_d + avg_mean_d

    leg_ang    = h3d_to_angles(x_pred_deg, sagittal_only=True)   # (B, 6, T)
    positions  = h3d_to_positions22(x_pred_deg)                  # (B, T, 22, 3)
    arm_swing  = arm_swing_range_t(positions)                    # (B,)
    trunk_incl = trunk_inclination_t(positions)                   # (B,)

    pred_rom = leg_ang.max(dim=-1).values - leg_ang.min(dim=-1).values  # (B, 6)

    rom_loss, arm_loss, trunk_loss = zero, zero, zero
    n_cls = 0
    for cls_idx in range(UPDRS_CLASSES):
        mask = (updrs_cls == cls_idx) & cond_mask
        if mask.sum() == 0:
            continue
        n_cls += 1
        rom_loss = rom_loss + F.l1_loss(pred_rom[mask].mean(dim=0), TARGET_ROM_DEG[cls_idx].to(device))
        arm_loss   = arm_loss   + F.l1_loss(arm_swing[mask].mean(), TARGET_ARM_SWING[cls_idx].to(device))
        trunk_loss = trunk_loss + F.l1_loss(trunk_incl[mask].mean(), TARGET_TRUNK_INCL_DEG[cls_idx].to(device))
    if n_cls > 0:
        rom_loss = rom_loss / n_cls
        arm_loss = arm_loss / n_cls
        trunk_loss = trunk_loss / n_cls

    weighted = ROM_WEIGHT * rom_loss + ARM_SWING_WEIGHT * arm_loss + TRUNK_INCL_WEIGHT * trunk_loss
    return weighted, rom_loss, arm_loss, trunk_loss


# ============================================================
# Dataset
# ============================================================

class GaitDatasetUPDRS(Dataset):
    def __init__(self, tensor_data, labels, laterality=None, norm_stats=None,
                 stepsizes=None, stepsize_bin_edges=None):
        """
        Args:
            tensor_data: (N, C, T) tensor of gait angle data
            labels:      (N,) tensor of UPDRS labels ∈ {0,1,2}
            laterality:  (N,) tensor of laterality labels (unused — laterality conditioning was
                         removed; kept as optional kwarg for backward compat with legacy loaders)
            norm_stats:  optional (mean, std) from train set — if None, computed from this data
            stepsizes:   (N,) float array of step lengths (cm) — if None, stepsize conditioning
                         is skipped and phys_labels won't contain 'stepsize'
            stepsize_bin_edges: boundaries for quantizing stepsizes into discrete bins.
                         None or empty → no stepsize conditioning.
        """
        self.data       = tensor_data
        self.updrs      = labels
        self.laterality = laterality
        self.stepsize_bins = None
        if stepsizes is not None and stepsize_bin_edges is not None:
            edges = np.asarray(stepsize_bin_edges, dtype=stepsizes.dtype)
            self.stepsize_bins = np.digitize(stepsizes, edges[1:-1]).astype(np.int64)

        if norm_stats is not None:
            self.avg_mean, self.avg_std = norm_stats
        else:
            self.avg_std = tensor_data.std(dim=(0, 2)).view(-1, 1)
            self.avg_std[self.avg_std < 1e-6] = 1.0
            self.avg_mean = tensor_data.mean(dim=(0, 2)).view(-1, 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]  # (C, T)

        global_mean = self.avg_mean.to(sample.device)
        global_std  = self.avg_std.to(sample.device)
        normalized  = (sample - global_mean) / global_std
        normalized  = torch.clamp(normalized, -4, 4)

        phys_labels = {
            'updrs':      self.updrs[idx],
        }
        if self.stepsize_bins is not None:
            phys_labels['stepsize'] = torch.tensor(self.stepsize_bins[idx], dtype=torch.long)
        return normalized, phys_labels


# ============================================================
# Generation (for evaluation checkpoints)
# ============================================================

@torch.no_grad()
def generate_batch_updrs(model, vae, n_samples, device, phys_cond, scale=2.5,
                          lat_mean=None, lat_std=None, temperature=1.0):
    """Generate gait windows via latent diffusion. Returns denormalized angles (B, 8, SEQ_LEN)."""
    model.eval()
    vae.eval()
    betas = get_noise_schedule(n_steps=TIMESTEPS).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    # Start from noise in latent space
    z = torch.randn(n_samples, LATENT_CHANNELS, LATENT_TIME).to(device)

    for i in reversed(range(TIMESTEPS)):
        t = torch.full((n_samples,), i, device=device, dtype=torch.long)
        alpha_bar = alphas_cumprod[i]

        if scale > 1.0 and phys_cond is not None:
            pred_cond   = model(z, t, phys_cond=phys_cond, drop_phys=False)
            pred_uncond = model(z, t, phys_cond=phys_cond, drop_phys=True)
            pred_z0 = pred_uncond + scale * (pred_cond - pred_uncond)
        else:
            pred_z0 = model(z, t, phys_cond=phys_cond, drop_phys=False)

        if i > 0:
            alpha_bar_prev = alphas_cumprod[i - 1]
            coef1 = torch.sqrt(alpha_bar_prev) * betas[i] / (1 - alpha_bar)
            coef2 = torch.sqrt(alphas[i]) * (1 - alpha_bar_prev) / (1 - alpha_bar)
            posterior_mean = coef1 * pred_z0 + coef2 * z
            posterior_var = betas[i] * (1 - alpha_bar_prev) / (1 - alpha_bar)
            z = posterior_mean + temperature * torch.sqrt(posterior_var) * torch.randn_like(z)
        else:
            z = pred_z0

    # Decode latent → angle space (still normalized)
    if lat_std is not None:
        z = z * lat_std
        if lat_mean is not None:
            z = z + lat_mean
    return vae.decode(z)


# ============================================================
# Plotting
# ============================================================

def save_checkpoint_plots(epoch, synth_np, real_np, corrs, output_dir, updrs_labels_np=None):
    os.makedirs(output_dir, exist_ok=True)
    n_joints = min(len(corrs), len(UPDRS_JOINT_NAMES))
    joint_names = UPDRS_JOINT_NAMES[:n_joints]

    mean_real = np.mean(real_np, axis=0).T  # (T, C)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sorted_idx = np.argsort(corrs)
    indices = [sorted_idx[0], sorted_idx[len(sorted_idx) // 2], sorted_idx[-1]]
    titles = ["Worst Match", "Median Match", "Best Match"]

    updrs_names = {0: "Normal", 1: "Mild", 2: "Moderate", 3: "Severe"}
    colors = {0: 'green', 1: 'blue', 2: 'orange', 3: 'red'}

    for ax, idx, title in zip(axes, indices, titles):
        name = joint_names[idx] if idx < len(joint_names) else f"Ch {idx}"
        ax.plot(mean_real[:, idx], 'k--', label='Real Mean', linewidth=3, alpha=0.8)

        if updrs_labels_np is not None:
            for cls in sorted(np.unique(updrs_labels_np)):
                mask = updrs_labels_np == cls
                if np.any(mask):
                    mean_cls = np.mean(synth_np[mask], axis=0).T
                    ax.plot(mean_cls[:, idx], color=colors.get(cls, 'gray'),
                            label=f'UPDRS {cls} ({updrs_names.get(cls, "?")})', linewidth=2)
        else:
            mean_synth = np.mean(synth_np, axis=0).T
            ax.plot(mean_synth[:, idx], 'r-', label='Synth Mean', linewidth=2)

        ax.set_title(f"{title}: {name}\nR={corrs[idx]:.3f}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Epoch {epoch + 1} - UPDRS-Conditioned Generation")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"checkpoint_epoch{epoch + 1}.png"))
    plt.close()


# ============================================================
# DTW helper
# ============================================================

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


# ============================================================
# Evaluation checkpoint
# ============================================================

# Left/Right joint index pairs for symmetry index (6-channel layout)
# (L_Hip_Flex, R_Hip_Flex), (L_Knee_Flex, R_Knee_Flex), (L_Ankle_Flex, R_Ankle_Flex)
LR_PAIRS = [(0, 1), (2, 3), (4, 5)]


def _local_peaks(signal):
    """Local maxima of a 1D signal via sign change of diff."""
    d = np.diff(signal)
    idx = np.where((d[:-1] > 0) & (d[1:] <= 0))[0] + 1
    return signal[idx], idx


def _cov_peaks(signal):
    """CoV of peak-to-peak intervals — stride timing variability (NaN if <3 peaks)."""
    _, peak_idx = _local_peaks(signal)
    if len(peak_idx) < 3:
        return np.nan
    intervals = np.diff(peak_idx).astype(float)
    m = intervals.mean()
    return float(intervals.std() / m) if m > 1e-6 else np.nan


def _symmetry_index(rom_l, rom_r):
    """SI = 2|L-R|/(L+R)*100. 0=perfect symmetry, higher=more asymmetric."""
    denom = rom_l + rom_r
    if denom < 1e-6:
        return 0.0
    return float(2 * abs(rom_l - rom_r) / denom * 100)


def _extract_features(data):
    """data: (N, C, T) → (N, 3C) feature vector for centroid classifier."""
    rom  = data.max(axis=2) - data.min(axis=2)
    mean = data.mean(axis=2)
    std  = data.std(axis=2)
    return np.concatenate([rom, mean, std], axis=1)


def train_centroid_classifier(real_np, real_labels_np):
    """Nearest-centroid classifier trained on real data. Returns centroids (n_classes, F)."""
    feats = _extract_features(real_np)
    centroids = np.array([
        feats[real_labels_np == cls].mean(axis=0)
        for cls in range(UPDRS_CLASSES)
    ])
    return centroids


def classify_centroid(centroids, data):
    """Classify data (N, C, T) using nearest centroid. Returns predicted labels (N,)."""
    feats = _extract_features(data)
    dists = np.linalg.norm(feats[:, None, :] - centroids[None, :, :], axis=2)
    return dists.argmin(axis=1)


def _class_metrics(synth_cls, real_cls, n_channels):
    """
    Metrics per UPDRS class.

    Kept  : ROM, R_mean_vs_mean, DTW, Diversity, SI, CoV
    Dropped: R_sample_vs_mean, envelope overlap, acceleration error
    """
    # (N, C, T) → (N, T, C)
    s = synth_cls.transpose(0, 2, 1)
    r = real_cls.transpose(0, 2, 1)

    s_mean = np.mean(s, axis=0)   # (T, C)
    r_mean = np.mean(r, axis=0)

    # 1. ROM: per-sample then averaged (avoids phase cancellation)
    rom_synth = np.mean(np.max(s, axis=1) - np.min(s, axis=1), axis=0)  # (C,)
    rom_real  = np.mean(np.max(r, axis=1) - np.min(r, axis=1), axis=0)

    # 2. Shape fidelity — mean waveform vs mean waveform
    r_mean_vs_mean = np.zeros(n_channels)
    for c in range(n_channels):
        sc, rc = s_mean[:, c], r_mean[:, c]
        if np.std(sc) > 1e-6 and np.std(rc) > 1e-6:
            r_mean_vs_mean[c] = float(np.corrcoef(sc, rc)[0, 1])

    # 3. DTW (phase-robust, Sakoe-Chiba band = 10%)
    T      = s_mean.shape[0]
    window = max(1, int(0.10 * T))
    dtw_mean = np.array([
        _dtw_1d(s_mean[:, c], r_mean[:, c], window=window)
        for c in range(n_channels)
    ])

    # 4. Diversity — mean pairwise DTW within generated class
    N = len(s)
    diversity  = np.zeros(n_channels)
    n_pairs    = 0
    for i in range(N):
        for j in range(i + 1, N):
            for c in range(n_channels):
                diversity[c] += _dtw_1d(s[i, :, c], s[j, :, c], window=window)
            n_pairs += 1
    if n_pairs > 0:
        diversity /= n_pairs

    # 5. Symmetry Index — mean SI across L/R joint pairs (synth vs real)
    si_synth = np.mean([_symmetry_index(rom_synth[l], rom_synth[r_]) for l, r_ in LR_PAIRS])
    si_real  = np.mean([_symmetry_index(rom_real[l],  rom_real[r_])  for l, r_ in LR_PAIRS])

    # 6. CoV of peak-to-peak intervals — stride timing variability (nanmean skips windows with <3 peaks)
    cov_synth = np.nanmean([[_cov_peaks(s[i, :, c]) for c in range(n_channels)] for i in range(N)], axis=0)
    cov_real  = np.nanmean([[_cov_peaks(r[i, :, c]) for c in range(n_channels)] for i in range(len(r))], axis=0)
    cov_synth = np.where(np.isfinite(cov_synth), cov_synth, 0.0)
    cov_real  = np.where(np.isfinite(cov_real),  cov_real,  0.0)

    return {
        'rom_synth':      rom_synth,
        'rom_real':       rom_real,
        'r_mean_vs_mean': r_mean_vs_mean,
        's_mean':         s_mean,
        'r_mean':         r_mean,
        'dtw_mean':       dtw_mean,
        'diversity':      diversity,
        'dtw_window':     window,
        'si_synth':       si_synth,
        'si_real':        si_real,
        'cov_synth':      cov_synth,
        'cov_real':       cov_real,
    }


def run_evaluation_checkpoint(model, vae, raw_tensor, dataset, device, epoch,
                               eval_per_class=25, lat_mean=None, lat_std=None):
    print(f"\n📊 EVALUATION CHECKPOINT (Epoch {epoch + 1})")
    seq_len    = raw_tensor.shape[2]
    n_channels = raw_tensor.shape[1]

    updrs_labels = torch.cat([
        torch.full((eval_per_class,), cls, dtype=torch.long) for cls in range(UPDRS_CLASSES)
    ]).to(device)

    synth_data = generate_batch_updrs(
        model, vae, eval_per_class * UPDRS_CLASSES,
        device, {'updrs': updrs_labels}, scale=GUIDANCE_SCALE,
        lat_mean=lat_mean, lat_std=lat_std,
    )

    # synth_data is normalized — denormalize via the dataset stats.
    synth_h3d = synth_data * dataset.avg_std.to(device) + dataset.avg_mean.to(device)
    raw_h3d   = raw_tensor.to(device)

    # Bridge to 6 sagittal angles for leg metrics.
    synth_denorm = h3d_to_angles(synth_h3d, sagittal_only=True)
    raw_sag      = h3d_to_angles(raw_h3d, sagittal_only=True)
    n_channels   = synth_denorm.shape[1]
    synth_np     = synth_denorm.detach().cpu().numpy()   # (N, 6, T)
    real_np      = raw_sag.detach().cpu().numpy()        # (M, 6, T)

    # Arm joint Z-displacement time series for arm metrics (metres, pelvis-relative).
    arm_synth_np = h3d_to_arm_timeseries(synth_h3d).detach().cpu().numpy()  # (N, 6, T)
    arm_real_np  = h3d_to_arm_timeseries(raw_h3d).detach().cpu().numpy()    # (M, 6, T)
    updrs_np      = updrs_labels.cpu().numpy()
    real_labels_np = dataset.updrs.cpu().numpy()

    cls_names  = {0: "Normal", 1: "Mild", 2: "Moderate", 3: "Severe"}
    key_joints = list(range(n_channels))   # all channels are sagittal

    per_class_results = {}

    # Train centroid classifier on real eval data (independent of VAE/DiT)
    centroids = train_centroid_classifier(real_np, real_labels_np)

    W  = 9
    DW = 11

    for cls in range(UPDRS_CLASSES):
        mask_s    = updrs_np == cls
        mask_r    = real_labels_np == cls
        synth_cls = synth_np[mask_s]
        real_cls  = real_np[mask_r] if np.any(mask_r) else real_np

        if len(synth_cls) == 0:
            continue

        m = _class_metrics(synth_cls, real_cls, n_channels)
        per_class_results[cls] = m

        n_s, n_r = np.sum(mask_s), np.sum(mask_r)
        print(f"\n  UPDRS {cls} — {cls_names.get(cls, '?')}  ({n_s} generated / {n_r} real)")

        # ROM + shape fidelity table
        header = (f"  {'Joint':<20} {'Real ROM':>{W}} {'Synth ROM':>{W}} {'R(mean)':>{W}}")
        divider_w = 20 + 3 * (W + 1)
        print(header)
        print("  " + "─" * divider_w)
        for j in key_joints:
            name = UPDRS_JOINT_NAMES[j] if j < len(UPDRS_JOINT_NAMES) else f"Ch {j}"
            print(f"  {name:<20} "
                  f"{m['rom_real'][j]:>{W}.2f} "
                  f"{m['rom_synth'][j]:>{W}.2f} "
                  f"{m['r_mean_vs_mean'][j]:>{W}.3f}")
        kj = key_joints
        print("  " + "─" * divider_w)
        print(f"  {'KEY JOINTS MEAN':<20} "
              f"{np.mean(m['rom_real'][kj]):>{W}.2f} "
              f"{np.mean(m['rom_synth'][kj]):>{W}.2f} "
              f"{np.mean(m['r_mean_vs_mean'][kj]):>{W}.3f}")

        # DTW + Diversity table
        print(f"\n  DTW (Sakoe-Chiba band={m['dtw_window']}fr) | Div=intra-class pairwise DTW")
        print(f"  {'Joint':<20} {'DTW(mean)':>{DW}} {'Diversity':>{DW}}")
        print("  " + "─" * (20 + 2 * (DW + 1)))
        for j in key_joints:
            jname = UPDRS_JOINT_NAMES[j] if j < len(UPDRS_JOINT_NAMES) else f"Ch {j}"
            print(f"  {jname:<20} "
                  f"{m['dtw_mean'][j]:>{DW}.4f} "
                  f"{m['diversity'][j]:>{DW}.4f}")
        print(f"  {'KEY JOINTS MEAN':<20} "
              f"{np.mean(m['dtw_mean'][kj]):>{DW}.4f} "
              f"{np.mean(m['diversity'][kj]):>{DW}.4f}")

        # Symmetry Index + CoV
        print(f"\n  Symmetry Index (SI): real={m['si_real']:.1f}%  synth={m['si_synth']:.1f}%"
              f"  {'✅ impaired>normal' if cls > 0 and m['si_synth'] > 5 else ''}")
        cov_key = np.mean(m['cov_synth'][[4, 5]])   # knee channels most informative
        cov_key_r = np.mean(m['cov_real'][[4, 5]])
        print(f"  Stride CoV (knee):  real={cov_key_r:.3f}  synth={cov_key:.3f}")

        # Arm joint metrics (sagittal Z-displacement relative to pelvis, metres)
        arm_s = arm_synth_np[mask_s]
        arm_r = arm_real_np[mask_r] if np.any(mask_r) else arm_real_np
        arm_m = _class_metrics(arm_s, arm_r, len(ARM_JOINT_NAMES))
        print(f"\n  Arm joints (sagittal Z-displacement vs pelvis, m):")
        print(f"  {'Joint':<20} {'Real ROM':>{W}} {'Synth ROM':>{W}} {'R(mean)':>{W}}")
        print("  " + "─" * (20 + 3 * (W + 1)))
        for j, jname in enumerate(ARM_JOINT_NAMES):
            print(f"  {jname:<20} "
                  f"{arm_m['rom_real'][j]:>{W}.3f} "
                  f"{arm_m['rom_synth'][j]:>{W}.3f} "
                  f"{arm_m['r_mean_vs_mean'][j]:>{W}.3f}")
        wrist_idx = [4, 5]
        print("  " + "─" * (20 + 3 * (W + 1)))
        print(f"  {'WRISTS MEAN':<20} "
              f"{np.mean(arm_m['rom_real'][wrist_idx]):>{W}.3f} "
              f"{np.mean(arm_m['rom_synth'][wrist_idx]):>{W}.3f} "
              f"{np.mean(arm_m['r_mean_vs_mean'][wrist_idx]):>{W}.3f}")
        print(f"\n  Arm DTW (Sakoe-Chiba band={arm_m['dtw_window']}fr):")
        print(f"  {'Joint':<20} {'DTW(mean)':>{DW}} {'Diversity':>{DW}}")
        print("  " + "─" * (20 + 2 * (DW + 1)))
        for j, jname in enumerate(ARM_JOINT_NAMES):
            print(f"  {jname:<20} "
                  f"{arm_m['dtw_mean'][j]:>{DW}.4f} "
                  f"{arm_m['diversity'][j]:>{DW}.4f}")
        print(f"  {'WRISTS MEAN':<20} "
              f"{np.mean(arm_m['dtw_mean'][wrist_idx]):>{DW}.4f} "
              f"{np.mean(arm_m['diversity'][wrist_idx]):>{DW}.4f}")

    # Cross-class ROM separation
    if len(per_class_results) == UPDRS_CLASSES:
        print(f"\n  CROSS-CLASS ROM SEPARATION (key joints, generated):")
        key_roms = {cls: np.mean(per_class_results[cls]['rom_synth'][key_joints])
                    for cls in range(UPDRS_CLASSES)}
        ordered = all(key_roms[c] > key_roms[c + 1] for c in range(UPDRS_CLASSES - 1))
        vals = "  ".join(f"UPDRS{c}={key_roms[c]:.2f}°" for c in range(UPDRS_CLASSES))
        status = "✅ PASS" if ordered else "❌ FAIL (collapsed to global mean)"
        print(f"  {vals}  →  {status}")

        # Cross-class SI separation (UPDRS 0 should have lowest SI)
        print(f"\n  CROSS-CLASS SYMMETRY INDEX (generated):")
        key_si = {cls: per_class_results[cls]['si_synth'] for cls in range(UPDRS_CLASSES)}
        si_ordered = key_si[0] < key_si[1] and key_si[0] < key_si[2]
        si_vals = "  ".join(f"UPDRS{c}={key_si[c]:.1f}%" for c in range(UPDRS_CLASSES))
        si_status = "✅ PASS" if si_ordered else "❌ FAIL (normal not most symmetric)"
        print(f"  {si_vals}  →  {si_status}")

        # Cross-class CoV separation (UPDRS 2 should have highest CoV)
        print(f"\n  CROSS-CLASS STRIDE CoV (knee, generated):")
        key_cov = {cls: float(np.mean(per_class_results[cls]['cov_synth'][[4, 5]]))
                   for cls in range(UPDRS_CLASSES)}
        cov_ordered = all(key_cov[c] < key_cov[c + 1] for c in range(UPDRS_CLASSES - 1))
        cov_vals = "  ".join(f"UPDRS{c}={key_cov[c]:.3f}" for c in range(UPDRS_CLASSES))
        cov_status = "✅ PASS" if cov_ordered else "❌ FAIL"
        print(f"  {cov_vals}  →  {cov_status}")

    # UPDRS classifier test
    print(f"\n  UPDRS CLASSIFIER TEST (centroid classifier trained on real eval data):")
    clf_preds = classify_centroid(centroids, synth_np)
    for cls in range(UPDRS_CLASSES):
        mask = updrs_np == cls
        if not np.any(mask):
            continue
        correct = int((clf_preds[mask] == cls).sum())
        total   = int(mask.sum())
        acc     = 100 * correct / total
        status  = "✅" if acc >= 50 else "⚠️"
        print(f"  UPDRS {cls} ({cls_names[cls]}): {correct}/{total} correctly classified ({acc:.1f}%)  {status}")
    overall_clf = 100 * int((clf_preds == updrs_np).sum()) / len(updrs_np)
    print(f"  Overall: {overall_clf:.1f}%  (chance=33.3%)")

    # Overall mean-vs-mean correlation (all classes combined) for checkpoint plot
    all_corrs = np.zeros(n_channels)
    sm = np.mean(synth_np.transpose(0, 2, 1), axis=0)   # (T, C)
    rm = np.mean(real_np.transpose(0, 2, 1),  axis=0)
    for c in range(n_channels):
        if np.std(sm[:, c]) > 1e-6 and np.std(rm[:, c]) > 1e-6:
            all_corrs[c] = float(np.corrcoef(sm[:, c], rm[:, c])[0, 1])

    save_checkpoint_plots(
        epoch, synth_np, real_np, all_corrs,
        os.path.join(EVAL_OUTPUT_DIR, "checkpoints"), updrs_labels_np=updrs_np
    )

    avg_corr = float(np.mean(all_corrs))
    return avg_corr, all_corrs


# ============================================================
# Training loop
# ============================================================

def train():
    print("=" * 60)
    print("UPDRS Latent DiT — Stage 2 Training")
    print(f"  Input space : latent (B, {LATENT_CHANNELS}, {LATENT_TIME})")
    print(f"  Angle space : (B, {N_CHANNELS}, {SEQ_LEN})")
    print("=" * 60)

    # 1. Load processed data
    if not os.path.exists(TRAIN_DATA_PATH):
        print(f"ERROR: Train data not found at {TRAIN_DATA_PATH}")
        print("Run preprocess_carepd.py first.")
        return
    if not os.path.exists(VAE_MODEL_PATH):
        print(f"ERROR: VAE model not found at {VAE_MODEL_PATH}")
        print("Run train_vae_updrs.py first.")
        return

    def _load_channels(path):
        return np.load(path)

    train_numpy      = _load_channels(TRAIN_DATA_PATH)
    train_labels     = torch.from_numpy(np.load(TRAIN_LABELS_PATH)).long()
    train_tensor     = torch.from_numpy(train_numpy).float()

    eval_numpy      = _load_channels(EVAL_DATA_PATH)
    eval_labels     = torch.from_numpy(np.load(EVAL_LABELS_PATH)).long()
    eval_tensor     = torch.from_numpy(eval_numpy).float()

    print(f"Train: {train_tensor.shape}, Eval: {eval_tensor.shape}")
    for cls in range(UPDRS_CLASSES):
        print(f"  UPDRS {cls}: train={int((train_labels == cls).sum())}, eval={int((eval_labels == cls).sum())}")

    # Inverse-frequency class weights: w_c = N_total / (n_classes * N_c)
    # E[w] = 1 so loss magnitude is preserved; under-represented classes get w > 1.
    train_counts = torch.tensor(
        [float((train_labels == cls).sum()) for cls in range(UPDRS_CLASSES)],
        dtype=torch.float32,
    )
    N_total = train_counts.sum()
    class_weights = (N_total / (UPDRS_CLASSES * train_counts)).to(DEVICE)

    # 2. Dataset & loaders
    # Stepsize conditioning: load pre-computed step lengths and bin into discrete classes.
    _stepsize_bins = STEPSIZE_BINS   # local copy — may be overridden below if labels missing
    train_stepsizes = None
    eval_stepsizes = None
    stepsize_edges = None
    if _stepsize_bins > 0:
        if os.path.exists(TRAIN_STEPSIZE_PATH) and os.path.exists(EVAL_STEPSIZE_PATH):
            train_stepsizes = np.load(TRAIN_STEPSIZE_PATH)
            eval_stepsizes  = np.load(EVAL_STEPSIZE_PATH)
            assert len(train_stepsizes) == len(train_labels), \
                f"Stepsize count {len(train_stepsizes)} != label count {len(train_labels)}"
            assert len(eval_stepsizes) == len(eval_labels)
            if STEPSIZE_BIN_EDGES is None:
                stepsize_edges = np.percentile(train_stepsizes,
                                               np.linspace(0, 100, _stepsize_bins + 1))
                stepsize_edges[0] = 0.0
            else:
                stepsize_edges = np.asarray(STEPSIZE_BIN_EDGES, dtype=np.float64)
            print(f"\n--- Stepsize Conditioning ({_stepsize_bins} bins) ---")
            print(f"  Train stepsizes: {train_stepsizes.shape}  cm")
            print(f"  Bin edges (cm):   {np.round(stepsize_edges, 1).tolist()}")
            for b in range(_stepsize_bins):
                lo, hi = stepsize_edges[b], stepsize_edges[b+1]
                mask = (train_stepsizes >= lo) & (train_stepsizes < hi) \
                    if b < _stepsize_bins - 1 else (train_stepsizes >= lo)
                print(f"    Bin {b} [{lo:.1f}, {hi:.1f}): n={mask.sum():>5d}")
        else:
            print(f"WARNING: STEPSIZE_BINS={_stepsize_bins} but stepsize labels not found; "
                  f"run compute_stepsize_labels.py first. Disabling stepsize conditioning.")
            _stepsize_bins = 0

    if _stepsize_bins > 0 and train_stepsizes is not None:
        train_set = GaitDatasetUPDRS(train_tensor, train_labels,
                                     stepsizes=train_stepsizes, stepsize_bin_edges=stepsize_edges)
        val_set   = GaitDatasetUPDRS(eval_tensor, eval_labels,
                                     stepsizes=eval_stepsizes, stepsize_bin_edges=stepsize_edges,
                                     norm_stats=(train_set.avg_mean, train_set.avg_std))
    else:
        train_set = GaitDatasetUPDRS(train_tensor, train_labels)
        val_set   = GaitDatasetUPDRS(eval_tensor, eval_labels,
                                     norm_stats=(train_set.avg_mean, train_set.avg_std))
    dataset = train_set  # reference for normalization stats

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    # Save normalization params
    norm_params = {'std': dataset.avg_std, 'mean': dataset.avg_mean}
    if _stepsize_bins > 0 and stepsize_edges is not None:
        norm_params['stepsize_bin_edges'] = torch.from_numpy(stepsize_edges.astype(np.float32))
        norm_params['stepsize_bins'] = _stepsize_bins
    os.makedirs(os.path.dirname(STATS_PATH), exist_ok=True)
    torch.save(norm_params, STATS_PATH)
    torch.save(norm_params, NORM_PARAMS_PATH)
    print(f"📈 Normalization params saved to {STATS_PATH}")

    # 3. Load frozen VAE
    print(f"\n🔒 Loading frozen VAE from {VAE_MODEL_PATH}")
    vae_state_dict = torch.load(VAE_MODEL_PATH, map_location=DEVICE)
    vae = GaitVAE(
        in_channels=N_CHANNELS,
        latent_channels=LATENT_CHANNELS,
        updrs_classes=UPDRS_CLASSES,
        use_prototypes="prototypes" in vae_state_dict,
    ).to(DEVICE)
    vae.load_state_dict(vae_state_dict)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    print("✅ VAE frozen — encoder and decoder will not be updated")

    # 4. DiT model — operates on latent (LATENT_CHANNELS, LATENT_TIME)
    print(f"\n Initializing Latent DiT (Dim: {EMBED_DIM}, Layers: {N_LAYERS}, Heads: {N_HEADS})")
    torch.cuda.empty_cache()

    model = DiffusionTransformerUPDRS(
        n_channels=LATENT_CHANNELS,
        seq_len=LATENT_TIME,
        embed_dim=EMBED_DIM,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        dropout=DROPOUT,
        updrs_classes=UPDRS_CLASSES,
        stepsize_bins=_stepsize_bins,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model on {DEVICE}. Total Params: {total_params / 1e6:.2f}M")

    # Resume
    start_epoch = 0
    if RESUME_TRAINING and os.path.exists(MODEL_PATH):
        print(f"🔄 Resuming from: {MODEL_PATH}")
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
        strict = _stepsize_bins == 0  # allow missing stepsize_emb when loading old checkpoints
        missing_keys, unexpected_keys = model.load_state_dict(ckpt, strict=strict)
        if not strict and missing_keys:
            print(f"⚠️  Missing keys (fresh init): {missing_keys}")
        if unexpected_keys:
            print(f"⚠️  Unexpected keys (ignored): {unexpected_keys}")
        start_epoch = START_EPOCH
        print(f"✅ Loaded. Starting from epoch {start_epoch}")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    betas = get_noise_schedule(n_steps=TIMESTEPS).to(DEVICE)
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)

    print("\n" + "=" * 60)
    print("LOSS WEIGHTS")
    print("=" * 60)
    print(f"  MSE_WEIGHT           : {MSE_WEIGHT}")
    print(f"  VELOCITY_WEIGHT      : {VELOCITY_WEIGHT}")
    print(f"  ROM_WEIGHT           : {ROM_WEIGHT}")
    print(f"  ARM_SWING_WEIGHT     : {ARM_SWING_WEIGHT}")
    print(f"  TRUNK_INCL_WEIGHT    : {TRUNK_INCL_WEIGHT}")
    print(f"  CLASS_WEIGHTED_LOSS  : {CLASS_WEIGHTED_LOSS}")
    if CLASS_WEIGHTED_LOSS:
        for cls in range(UPDRS_CLASSES):
            print(f"    UPDRS {cls} : n={int(train_counts[cls])}  w={class_weights[cls].item():.3f}")
    if _stepsize_bins > 0 and stepsize_edges is not None:
        print(f"\n  STEPSIZE_BINS = {_stepsize_bins}")
        for b in range(_stepsize_bins):
            lo, hi = stepsize_edges[b], stepsize_edges[b+1]
            print(f"    Bin {b} [{lo:.1f}, {hi:.1f}) cm")
    print("=" * 60 + "\n")

    # 4. Training loop
    best_val_loss = float('inf')
    best_epoch    = start_epoch

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        batch_losses = []
        batch_rom_losses, batch_arm_losses, batch_trunk_losses = [], [], []

        for batch in train_loader:
            x0, phys_labels = batch
            x0 = x0.to(DEVICE)
            for k in phys_labels:
                phys_labels[k] = phys_labels[k].to(DEVICE)

            curr_batch = x0.shape[0]
            drop_phys = torch.rand(curr_batch, device=DEVICE) < CFG_DROPOUT_PROB

            # Encode to latent space with frozen VAE (sample z for regularization benefit)
            with torch.no_grad():
                z0, _, _ = vae.encode(x0)

            t = torch.randint(0, TIMESTEPS, (curr_batch,), device=DEVICE).long()
            noise = torch.randn_like(z0)
            alpha_bar = alphas_cumprod[t].view(-1, 1, 1)
            zt = torch.sqrt(alpha_bar) * z0 + torch.sqrt(1 - alpha_bar) * noise

            pred_z0 = model(zt, t, phys_cond=phys_labels, drop_phys=drop_phys)

            updrs_cls = phys_labels['updrs']
            cond_mask = ~drop_phys

            per_sample_mse = ((pred_z0 - z0) ** 2).mean(dim=(1, 2))
            pred_vel = pred_z0[:, :, 1:] - pred_z0[:, :, :-1]
            real_vel = z0[:, :, 1:] - z0[:, :, :-1]
            per_sample_vel = ((pred_vel - real_vel) ** 2).mean(dim=(1, 2))

            if CLASS_WEIGHTED_LOSS:
                sample_w = class_weights[updrs_cls]   # (B,) — w_c = N_total / (n_classes * N_c)
                mse_loss = (sample_w * per_sample_mse).mean()
                vel_loss = (sample_w * per_sample_vel).mean()
            else:
                mse_loss = per_sample_mse.mean()
                vel_loss = per_sample_vel.mean()

            fb_loss, rom_loss, arm_loss, trunk_loss = compute_fullbody_rom_loss(
                pred_z0, updrs_cls, cond_mask, vae, dataset, DEVICE)

            total_loss = MSE_WEIGHT * mse_loss + VELOCITY_WEIGHT * vel_loss + fb_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_losses.append(total_loss.item())
            batch_rom_losses.append(rom_loss.item())
            batch_arm_losses.append(arm_loss.item())
            batch_trunk_losses.append(trunk_loss.item())

        scheduler.step()
        curr_lr = optimizer.param_groups[0]['lr']

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_v in val_loader:
                x0_v, phys_labels_v = batch_v
                x0_v = x0_v.to(DEVICE)
                for k in phys_labels_v:
                    phys_labels_v[k] = phys_labels_v[k].to(DEVICE)

                curr_batch_v = x0_v.shape[0]
                z0_v, _, _ = vae.encode(x0_v)

                t = torch.randint(0, TIMESTEPS, (curr_batch_v,), device=DEVICE).long()
                alpha_bar = alphas_cumprod[t].view(-1, 1, 1)
                zt_v = torch.sqrt(alpha_bar) * z0_v + torch.sqrt(1 - alpha_bar) * torch.randn_like(z0_v)

                pred_v = model(zt_v, t, phys_cond=phys_labels_v, drop_phys=False)

                per_sample_mse_v = ((pred_v - z0_v) ** 2).mean(dim=(1, 2))
                pred_vel_v = pred_v[:, :, 1:] - pred_v[:, :, :-1]
                real_vel_v = z0_v[:, :, 1:] - z0_v[:, :, :-1]
                per_sample_vel_v = ((pred_vel_v - real_vel_v) ** 2).mean(dim=(1, 2))

                if CLASS_WEIGHTED_LOSS:
                    sample_w_v = class_weights[phys_labels_v['updrs']]
                    mse_v = (sample_w_v * per_sample_mse_v).mean()
                    vel_v = (sample_w_v * per_sample_vel_v).mean()
                else:
                    mse_v = per_sample_mse_v.mean()
                    vel_v = per_sample_vel_v.mean()

                cond_mask_v = torch.ones(curr_batch_v, dtype=torch.bool, device=DEVICE)
                fb_v, _, _, _ = compute_fullbody_rom_loss(
                    pred_v, phys_labels_v['updrs'], cond_mask_v, vae, dataset, DEVICE)

                total_v = MSE_WEIGHT * mse_v + VELOCITY_WEIGHT * vel_v + fb_v
                val_losses.append(total_v.item())

        epoch_val = np.mean(val_losses)
        if epoch_val < best_val_loss:
            best_val_loss = epoch_val
            best_epoch    = epoch + 1
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{EPOCHS} | Train: {np.mean(batch_losses):.6f} | Val: {epoch_val:.6f} | LR: {curr_lr:.2e} | Best: ep{best_epoch} ({best_val_loss:.6f})")

        if (epoch + 1) % 100 == 0:
            if (epoch + 1) % 500 == 0:
                _p = Path(MODEL_PATH)
                ckpt_path = str(_p.with_name(_p.stem + f'_ep{epoch + 1}' + _p.suffix))
                torch.save(model.state_dict(), ckpt_path)
                print(f"📌 Numbered checkpoint: {ckpt_path}")
            avg_c, per_joint_c = run_evaluation_checkpoint(
                model, vae, eval_tensor, val_set, DEVICE, epoch,
                eval_per_class=50, lat_mean=None, lat_std=None,
            )

            print("\n🔍 PER-JOINT CORRELATION REPORT:")
            print("-" * 40)
            for j_idx, j_name in enumerate(UPDRS_JOINT_NAMES):
                if j_idx < len(per_joint_c):
                    print(f"   {j_name:<20} | {per_joint_c[j_idx]:.4f}")
            print("-" * 40 + "\n")

    print(f"\n✅ Training complete. Best epoch: {best_epoch}  val_loss: {best_val_loss:.6f}")
    print(f"   Best model already saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()
