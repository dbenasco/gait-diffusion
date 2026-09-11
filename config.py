import os
import torch

# Base directory for the repository root
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# SMPL model (required for biomechanical angle extraction via FK)
SMPL_MODEL_PATH = os.path.join(REPO_ROOT, "data/smpl_models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl")

# --- DATA PATHS ---
# Raw CARE-PD / GAITGen data (set this to wherever you download the dataset)
CAREPD_RAW_DIR = os.path.join(REPO_ROOT, "data/carepd/doi-10.5683-sp3-twikmk")
CAREPD_LABELED_COHORTS = ["BMCLab", "PD-GaM", "3DGait", "T-SDU-PD"]
CAREPD_TARGET_FPS = 30

# Intermediate output from smpl_to_angles.py (raw angles before windowing)
ANGLES_PATH = os.path.join(REPO_ROOT, "data/carepd/angles.pkl")

# Processed outputs from preprocess_carepd.py
PROCESSED_DATA_PATH = os.path.join(REPO_ROOT, "data/carepd/processed_gait_angles.npy")
PROCESSED_LABELS_PATH = os.path.join(REPO_ROOT, "data/carepd/updrs_labels.npy")
TRAIN_DATA_PATH = os.path.join(REPO_ROOT, "data/carepd/train_gait_angles.npy")
TRAIN_LABELS_PATH = os.path.join(REPO_ROOT, "data/carepd/train_updrs_labels.npy")
TRAIN_LATERALITY_PATH = os.path.join(REPO_ROOT, "data/carepd/train_laterality_labels.npy")
TRAIN_SUBJECTS_PATH = os.path.join(REPO_ROOT, "data/carepd/train_subject_ids.npy")
EVAL_DATA_PATH = os.path.join(REPO_ROOT, "data/carepd/eval_gait_angles.npy")
EVAL_LABELS_PATH = os.path.join(REPO_ROOT, "data/carepd/eval_updrs_labels.npy")
EVAL_LATERALITY_PATH = os.path.join(REPO_ROOT, "data/carepd/eval_laterality_labels.npy")
EVAL_SUBJECTS_PATH = os.path.join(REPO_ROOT, "data/carepd/eval_subject_ids.npy")
NORM_PARAMS_PATH = os.path.join(REPO_ROOT, "data/carepd/norm_params.pt")
LATENT_NORM_PARAMS_PATH = os.path.join(REPO_ROOT, "data/carepd/latent_norm_params.pt")
FOLDS_DIR = os.path.join(REPO_ROOT, "data/carepd/doi-10.5683-sp3-twikmk/folds/UPDRS_Datasets")

# --- MODEL PATHS ---
MODEL_PATH = os.environ.get(
    "UPDRS_MODEL_PATH",
    os.path.join(REPO_ROOT, "models/gait_updrs_dit_lam03_rom0_weighted.pth"),
)
STATS_PATH = os.path.join(REPO_ROOT, "models/gait_updrs_dit_stats.pt")
VAE_MODEL_PATH = os.environ.get(
    "UPDRS_VAE_MODEL_PATH",
    os.path.join(REPO_ROOT, "models/gait_vae_updrs.pth"),
)

# --- GENERATION OUTPUT ---
GEN_OUTPUT_PATH = os.environ.get(
    "UPDRS_GEN_OUTPUT_PATH",
    os.path.join(REPO_ROOT, "generated_data/generated_gait_updrs_dit.npy"),
)
EVAL_OUTPUT_DIR = os.path.join(REPO_ROOT, "graphs/graphs_evaluation/UPDRS_DiT")

# --- ARCHITECTURE ---
# Sagittal-plane flexion channels only (abduction channels dropped)
# Maps indices in the original 8-channel .npy files to the 6-channel model input
SAG_IDX = [0, 2, 4, 5, 6, 7]   # L_HipFlex, R_HipFlex, L_Knee, R_Knee, L_Ankle, R_Ankle
N_CHANNELS = 6          # 6 sagittal flexion joints (L/R hip flex, L/R knee, L/R ankle)
SEQ_LEN = 96            # Sliding window length (frames at 30fps ≈ 3.2s — captures ~2-4 strides)
WINDOW_STRIDE = 48      # Hop between consecutive windows (50% overlap)
EMBED_DIM = 256
N_HEADS = 4
N_LAYERS = 4
DROPOUT = 0.1
UPDRS_CLASSES = 4       # UPDRS-gait scores: 0 (normal), 1 (mild), 2 (moderate), 3 (severe)
UPDRS_MAX_SCORE = 3     # Filter out walks with UPDRS > this (set to 3 to include severe)

# Laterality conditioning: which leg is the impaired one
# 0 = Left impaired, 1 = Right impaired, 2 = Symmetric (UPDRS-0 / indeterminate)
LATERALITY_CLASSES = 3

# --- Stepsize (step length) conditioning ---
# Number of discrete step length bins for conditioning the DiT. 0 = disabled (backward
# compat — model, training, and generation behave identically to current pipeline).
# When >0, the model learns a stepsize_emb and generation accepts --stepsize.
STEPSIZE_BINS = int(os.environ.get("UPDRS_STEPSIZE_BINS", "0"))
# Bin edges in cm. None → computed as equal-frequency quantiles from train data.
STEPSIZE_BIN_EDGES = os.environ.get("UPDRS_STEPSIZE_BIN_EDGES")
if STEPSIZE_BIN_EDGES:
    STEPSIZE_BIN_EDGES = [float(x) for x in STEPSIZE_BIN_EDGES.split(",")]

# --- VAE / LATENT DIFFUSION ---
LATENT_CHANNELS = 6     # VAE latent channels (same as input channels)
LATENT_TIME = 24        # SEQ_LEN / 4 (compression ratio r=4 in time)
VAE_BETA = 0.001        # KL weight — kept low to preserve subtle pathological features
VAE_LAMBDA = float(os.environ.get("VAE_LAMBDA", "0.1"))  # Surrogate UPDRS loss weight
VAE_EPOCHS = 2000
VAE_LR = 1e-3

# --- Class-prototype latent loss (MAD-VAE style, opt-in via train_vae_updrs.py) ---
# See docs/superpowers/specs/2026-07-02-vae-class-prototype-loss-design.md.
# PROTOTYPE_PROXIMITY_WEIGHT weights a class-conditional KL divergence (not a
# bare MSE) on the pooled latent toward N(prototype, I) — because it has a
# learned variance, each class is regularized toward an actual Gaussian cloud
# around its prototype "slot" rather than being able to collapse onto a single
# discrete point. The original per-timestep KL(q||N(0,I)) is untouched, so
# reconstruction/temporal dynamics aren't affected. PROTOTYPE_DISTANCE_WEIGHT
# controls how hard the prototypes are pushed apart from each other.
# PROTOTYPE_MARGIN default is a placeholder — Task 6 of the implementation plan
# measures actual intra/inter-class spread on the current checkpoint and sets a
# real value (~3x max intra-class spread) before training starts.
PROTOTYPE_PROXIMITY_WEIGHT = float(os.environ.get("UPDRS_PROTOTYPE_PROX_WEIGHT", "0.01"))
PROTOTYPE_DISTANCE_WEIGHT  = float(os.environ.get("UPDRS_PROTOTYPE_DIST_WEIGHT", "0.01"))
PROTOTYPE_MARGIN          = float(os.environ.get("UPDRS_PROTOTYPE_MARGIN", "1.0"))

# --- TRAINING ---
BATCH_SIZE = 32
TIMESTEPS = 500
LEARNING_RATE = 2e-5
EPOCHS = 3000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_WEIGHTED_LOSS = True  # Inverse-frequency weighting on MSE/vel losses to counter class imbalance (U0 < U1 < U2 sample count)

# --- CFG ---
CFG_DROPOUT_PROB = float(os.environ.get("UPDRS_CFG_DROPOUT_PROB", "0.25"))
GUIDANCE_SCALE = 1.5   # selected via guidance scale sweep (ep3000)
TEMPERATURE    = 1.0   # DDPM noise temperature; >1 prevents mode collapse (sweep: T=1.2 best DTW)

# --- LOSS WEIGHTS ---
MSE_WEIGHT = 2.0
VELOCITY_WEIGHT = 0.1
ROM_WEIGHT = float(os.environ.get("UPDRS_ROM_WEIGHT", "0.1"))

# Target ROM per UPDRS class (degrees).
# Source: mean ROM per channel per UPDRS class, computed from raw windowed data (train+eval),
#         compute_rom_targets.py, 2026-06-15. Replaces VAE-reconstructed means from 2026-05-21.
# Key change: U1 knee 56.01° → 53.32° (was 2.69° too high due to VAE smoothing).
# L/R targets are symmetric (collapsed across laterality) — the ROM loss operates on the
# per-UPDRS-class batch mean, which averages Left- and Right-impaired samples together.
TARGET_ROM_DEG = torch.tensor([
    [51.94, 51.94, 57.87, 57.87, 24.21, 24.21],  # UPDRS 0 (Normal)
    [41.73, 41.73, 53.17, 53.17, 21.47, 21.47],  # UPDRS 1 (Mild)
    [26.54, 26.54, 39.82, 39.82, 14.58, 14.58],  # UPDRS 2 (Moderate)
    [13.71, 13.71, 23.98, 23.98, 7.92, 7.92],  # UPDRS 3 (Severe)
])  # shape: (4, 6) — sagittal channels on  ly: [L_HipFlex, R_HipFlex, L_Knee, R_Knee, L_Ankle, R_Ankle]

# --- H3D full-body ROM supervision (arm swing + trunk inclination) ---
# Same mechanism as ROM_WEIGHT/TARGET_ROM_DEG above (per-class batch-mean L1 vs a
# real-data target), extended to the two GaitGen full-body signals the bridge
# already defines (h3d_bridge.arm_swing_range_t / trunk_inclination_t), since H3D
# carries arms/trunk that the old 6-ch sagittal-leg representation didn't.
ARM_SWING_WEIGHT  = float(os.environ.get("UPDRS_ARM_SWING_WEIGHT",  "0.1"))
TRUNK_INCL_WEIGHT = float(os.environ.get("UPDRS_TRUNK_INCL_WEIGHT", "0.1"))

# Computed by compute_h3d_fullbody_targets.py, 2026-06-24 (train+eval, H3D,
# post-fix). Both signals now match the expected PD pattern:
#   - Arm swing: monotonic decrease with severity (reduced arm swing in PD).
#   - Trunk inclination: monotonic increase with severity (more stooped/forward
#     lean), plausible 0-30deg magnitude (was previously ~150deg before the
#     PD-GaM/3DGait/T-SDU-PD upside-down-skeleton fix in h3d_convert.py —
#     see _fix_updown; values below supersede that earlier, incorrect run).
TARGET_ARM_SWING = torch.tensor([0.0400, 0.0240, 0.0169, 0.0152])
TARGET_TRUNK_INCL_DEG = torch.tensor([10.97, 11.15, 15.26, 17.01])     # degrees, per UPDRS class

# --- GENERATION SAMPLE COUNTS ---
GEN_N_PER_CLASS = 300   # Samples to generate per UPDRS class

# --- EARLY STOPPING ---
PATIENCE = 600

# --- RESUME ---
RESUME_TRAINING = False
START_EPOCH = 0

# --- JOINT NAMES (CARE-PD lower-limb, sagittal plane only) ---
UPDRS_JOINT_NAMES = [
    "L_Hip_Flex",
    "R_Hip_Flex",
    "L_Knee_Flex",
    "R_Knee_Flex",
    "L_Ankle_Flex",
    "R_Ankle_Flex",
]


# ============================================================================
# HumanML3D (263-dim full-body) representation — parallel pipeline (USE_H3D)
# ----------------------------------------------------------------------------
# Swaps the 6-channel sagittal-angle representation for the 263-dim HumanML3D
# vector produced by preprocess_carepd_h3d.py (see
# docs/superpowers/specs/2026-06-22-humanml3d-representation-design.md).
# Same conv-VAE + latent-DiT architecture, resized to the new dims.
# ============================================================================

# Feature flag. When set, the shared scripts (VAE/DiT training, generation,
# evaluation) read the H3D artifacts and dims instead of the 6-channel angles.
# Env-overridable so remote runs toggle without editing the file.
USE_H3D = True

# --- H3D dimensions ---
H3D_FEATURE_DIM = 263      # root(4) | ric_pos(63) | rot6d(126) | local_vel(66) | foot(4)
H3D_N_JOINTS = 22          # T2M / SMPL skeleton (first 22 SMPL joints)
H3D_FEET_THRE = 0.002      # foot-contact velocity threshold (validated on CARE-PD cohorts)

# Channel-block boundaries inside the 263 vector (for slicing / per-block metrics).
H3D_BLOCKS = {
    "root":  (0, 4),       # r_ang_vel(1), r_lin_vel_xz(2), root_height(1)
    "ric":   (4, 67),      # (J-1)*3 = 63 local joint positions  ← metric bridge reads this
    "rot6d": (67, 193),    # (J-1)*6 = 126 continuous-6D rotations
    "vel":   (193, 259),   # J*3 = 66 local velocities
    "foot":  (259, 263),   # 4 foot-contact flags
}
# Convenience slice for the ric (local joint position) block used by the metric
# bridge (h3d_to_angles) and recover_from_ric.
H3D_RIC_SLICE = H3D_BLOCKS["ric"]   # (4, 67)

# Std-floor for per-channel normalization. Several H3D channels (e.g. foot
# contacts that saturate on glide-foot cohorts) are near-constant; without a
# floor their std≈0 gives ÷0 → NaN. Applied as std = max(std, floor); truly
# constant channels then normalize to ~0 regardless of the floor value.
H3D_STD_FLOOR = 1e-3

# --- Stepsize (step length) label arrays (float cm per window) ---
H3D_TRAIN_STEPSIZE_PATH = os.path.join(REPO_ROOT, "data/carepd/train_stepsize_h3d.npy")
H3D_EVAL_STEPSIZE_PATH  = os.path.join(REPO_ROOT, "data/carepd/eval_stepsize_h3d.npy")

# --- H3D latent sizing (VAE bottleneck; time compression r=4 unchanged) ---
# latent_channels is the VAE bottleneck width, decoupled from input channels by
# the encoder's 1x1 conv — so it need not equal H3D_FEATURE_DIM. The encoder's
# inner widths are widened to 256/256 for the 263-ch input (see vae_updrs.py).
# 64 channels → 64×24 = 1536-dim latent (~16× compression of the 263×96 input);
# the 6-ch VAE used 4× compression, but full-body motion is far more redundant.
# Validate by reconstruction quality: too small → poor reconstruction
# (arm/trunk/root), too large → diluted UPDRS surrogate class structure.
H3D_LATENT_CHANNELS = 64
H3D_LATENT_TIME = 24       # SEQ_LEN / 4 — same time compression as the 6-ch VAE

# --- H3D data paths (parallel to the 6-channel artifacts) ---
H3D_PROCESSED_DATA_PATH     = os.path.join(REPO_ROOT, "data/carepd/processed_gait_h3d.npy")
H3D_PROCESSED_LABELS_PATH   = os.path.join(REPO_ROOT, "data/carepd/updrs_labels_h3d.npy")
H3D_TRAIN_DATA_PATH         = os.environ.get("UPDRS_TRAIN_DATA",
    os.path.join(REPO_ROOT, "data/carepd/train_gait_h3d.npy"))
H3D_TRAIN_LABELS_PATH       = os.environ.get("UPDRS_TRAIN_LABELS",
    os.path.join(REPO_ROOT, "data/carepd/train_updrs_labels_h3d.npy"))
H3D_TRAIN_LATERALITY_PATH   = os.environ.get("UPDRS_TRAIN_LAT",
    os.path.join(REPO_ROOT, "data/carepd/train_laterality_labels_h3d.npy"))
H3D_TRAIN_SUBJECTS_PATH     = os.environ.get("UPDRS_TRAIN_SUBJECTS",
    os.path.join(REPO_ROOT, "data/carepd/train_subject_ids_h3d.npy"))
H3D_EVAL_DATA_PATH          = os.environ.get("UPDRS_EVAL_DATA",
    os.path.join(REPO_ROOT, "data/carepd/eval_gait_h3d.npy"))
H3D_EVAL_LABELS_PATH        = os.environ.get("UPDRS_EVAL_LABELS",
    os.path.join(REPO_ROOT, "data/carepd/eval_updrs_labels_h3d.npy"))
H3D_EVAL_LATERALITY_PATH    = os.environ.get("UPDRS_EVAL_LAT",
    os.path.join(REPO_ROOT, "data/carepd/eval_laterality_labels_h3d.npy"))
H3D_EVAL_SUBJECTS_PATH      = os.environ.get("UPDRS_EVAL_SUBJECTS",
    os.path.join(REPO_ROOT, "data/carepd/eval_subject_ids_h3d.npy"))
H3D_EVAL_LATERALITY_PATH    = os.path.join(REPO_ROOT, "data/carepd/eval_laterality_labels_h3d.npy")
H3D_EVAL_SUBJECTS_PATH      = os.path.join(REPO_ROOT, "data/carepd/eval_subject_ids_h3d.npy")
H3D_NORM_PARAMS_PATH        = os.environ.get("UPDRS_NORM_PARAMS",
    os.path.join(REPO_ROOT, "data/carepd/norm_params_h3d.pt"))
H3D_LATENT_NORM_PARAMS_PATH = os.environ.get("UPDRS_LATENT_NORM_PARAMS",
    os.path.join(REPO_ROOT, "data/carepd/latent_norm_params_h3d.pt"))

# --- H3D model / stats / generation paths ---
H3D_MODEL_PATH      = os.path.join(REPO_ROOT, "models/gait_updrs_dit_h3d.pth")
H3D_STATS_PATH      = os.environ.get("UPDRS_STATS_PATH",
    os.path.join(REPO_ROOT, "models/gait_updrs_dit_h3d_stats.pt"))
H3D_VAE_MODEL_PATH  = os.path.join(REPO_ROOT, "models/gait_vae_updrs_h3d.pth")
H3D_GEN_OUTPUT_PATH = os.path.join(REPO_ROOT, "generated_data/generated_gait_updrs_dit_h3d.npy")

# ----------------------------------------------------------------------------
# Flag-driven active-config override.
# With USE_H3D set, the names the shared scripts read (channels, latent size,
# data/model/stats paths) point at the H3D artifacts — mirroring the flag-derived
# path pattern in diffusion/config.py. TARGET_ROM_DEG is unchanged: the ROM loss
# still targets the same 6 sagittal angles, recovered via the metric bridge.
# ----------------------------------------------------------------------------
if USE_H3D:
    N_CHANNELS = H3D_FEATURE_DIM
    LATENT_CHANNELS = H3D_LATENT_CHANNELS
    LATENT_TIME = H3D_LATENT_TIME

    PROCESSED_DATA_PATH     = H3D_PROCESSED_DATA_PATH
    PROCESSED_LABELS_PATH   = H3D_PROCESSED_LABELS_PATH
    TRAIN_DATA_PATH         = H3D_TRAIN_DATA_PATH
    TRAIN_LABELS_PATH       = H3D_TRAIN_LABELS_PATH
    TRAIN_LATERALITY_PATH   = H3D_TRAIN_LATERALITY_PATH
    TRAIN_SUBJECTS_PATH     = H3D_TRAIN_SUBJECTS_PATH
    EVAL_DATA_PATH          = H3D_EVAL_DATA_PATH
    EVAL_LABELS_PATH        = H3D_EVAL_LABELS_PATH
    EVAL_LATERALITY_PATH    = H3D_EVAL_LATERALITY_PATH
    EVAL_SUBJECTS_PATH      = H3D_EVAL_SUBJECTS_PATH
    TRAIN_STEPSIZE_PATH     = H3D_TRAIN_STEPSIZE_PATH
    EVAL_STEPSIZE_PATH      = H3D_EVAL_STEPSIZE_PATH
    NORM_PARAMS_PATH        = H3D_NORM_PARAMS_PATH
    LATENT_NORM_PARAMS_PATH = H3D_LATENT_NORM_PARAMS_PATH

    # Respects UPDRS_MODEL_PATH if set (see the os.environ.get default above) so a
    # new training run can be saved under a different name without overwriting the
    # current reference checkpoint at H3D_MODEL_PATH.
    _model_env = os.environ.get("UPDRS_MODEL_PATH")
    _gen_env   = os.environ.get("UPDRS_GEN_OUTPUT_PATH")
    MODEL_PATH      = _model_env or H3D_MODEL_PATH
    STATS_PATH      = H3D_STATS_PATH
    VAE_MODEL_PATH  = os.environ.get("UPDRS_VAE_MODEL_PATH", H3D_VAE_MODEL_PATH)
    GEN_OUTPUT_PATH = _gen_env or H3D_GEN_OUTPUT_PATH

    # Stepsize suffix: when STEPSIZE_BINS > 0, save/load to suffixed files so
    # stepsize-conditioned checkpoints don't overwrite the non-stepsize ones.
    # Only the config *defaults* get the suffix. Paths provided explicitly via
    # env (UPDRS_MODEL_PATH / UPDRS_GEN_OUTPUT_PATH) are authoritative and are
    # used verbatim — otherwise compare_checkpoints' --gen_path pointing at
    # ..._stepsize_ss1.npy would get re-suffixed into ..._ss1_stepsize.npy.
    if STEPSIZE_BINS > 0:
        if not _model_env:
            _stem, _ext = os.path.splitext(MODEL_PATH)
            MODEL_PATH = _stem + "_stepsize" + _ext
        _stem_s, _ext_s = os.path.splitext(STATS_PATH)
        STATS_PATH = _stem_s + "_stepsize" + _ext_s
        if not _gen_env:
            _stem_g, _ext_g = os.path.splitext(GEN_OUTPUT_PATH)
            GEN_OUTPUT_PATH = _stem_g + "_stepsize" + _ext_g


