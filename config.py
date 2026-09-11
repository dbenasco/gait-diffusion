"""
Central configuration for the GaitDiffusion H3D (HumanML3D 263-dim) pipeline.

Every value is overridable via environment variables where noted, so remote
runs can change paths/hyperparameters without editing this file.
"""

import os
import torch

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# --- SMPL model (required for forward kinematics) ---
SMPL_MODEL_PATH = os.path.join(
    REPO_ROOT, "data/smpl_models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl"
)

# --- Raw CARE-PD data ---
CAREPD_RAW_DIR = os.path.join(REPO_ROOT, "data/carepd/doi-10.5683-sp3-twikmk")
CAREPD_LABELED_COHORTS = ["BMCLab", "PD-GaM", "3DGait", "T-SDU-PD"]
CAREPD_TARGET_FPS = 30
FOLDS_DIR = os.path.join(CAREPD_RAW_DIR, "folds/UPDRS_Datasets")

# --- Representation / windowing ---
SEQ_LEN = 96            # Sliding window length (frames at 30 fps ≈ 3.2 s)
WINDOW_STRIDE = 48      # Hop between consecutive windows (50% overlap)

H3D_FEATURE_DIM = 263   # root(4) | ric_pos(63) | rot6d(126) | local_vel(66) | foot(4)
N_CHANNELS = H3D_FEATURE_DIM
H3D_N_JOINTS = 22       # T2M / SMPL skeleton (first 22 SMPL joints)
H3D_STD_FLOOR = 1e-3    # per-channel normalization std floor (avoids ÷0 on constant channels)

# Channel-block boundaries inside the 263 vector (slicing / per-block metrics).
H3D_BLOCKS = {
    "root":  (0, 4),       # r_ang_vel(1), r_lin_vel_xz(2), root_height(1)
    "ric":   (4, 67),      # (J-1)*3 = 63 local joint positions
    "rot6d": (67, 193),    # (J-1)*6 = 126 continuous-6D rotations
    "vel":   (193, 259),   # J*3 = 66 local velocities
    "foot":  (259, 263),   # 4 foot-contact flags
}

# Sagittal-plane leg channels used by the ROM loss (indices into the 8-angle set).
SAG_IDX = [0, 2, 4, 5, 6, 7]   # L/R hip flex, L/R knee, L/R ankle

UPDRS_CLASSES = 4       # 0 (normal), 1 (mild), 2 (moderate), 3 (severe)
UPDRS_MAX_SCORE = 3     # Drop walks with UPDRS > this

UPDRS_JOINT_NAMES = [
    "L_Hip_Flex",
    "R_Hip_Flex",
    "L_Knee_Flex",
    "R_Knee_Flex",
    "L_Ankle_Flex",
    "R_Ankle_Flex",
]

# --- Data artifact paths ---
H3D_TRAIN_DATA_PATH = os.environ.get(
    "UPDRS_TRAIN_DATA", os.path.join(REPO_ROOT, "data/carepd/train_gait_h3d.npy"))
H3D_TRAIN_LABELS_PATH = os.environ.get(
    "UPDRS_TRAIN_LABELS", os.path.join(REPO_ROOT, "data/carepd/train_updrs_labels_h3d.npy"))

H3D_EVAL_DATA_PATH = os.environ.get(
    "UPDRS_EVAL_DATA", os.path.join(REPO_ROOT, "data/carepd/eval_gait_h3d.npy"))
H3D_EVAL_LABELS_PATH = os.environ.get(
    "UPDRS_EVAL_LABELS", os.path.join(REPO_ROOT, "data/carepd/eval_updrs_labels_h3d.npy"))

H3D_NORM_PARAMS_PATH = os.environ.get(
    "UPDRS_NORM_PARAMS", os.path.join(REPO_ROOT, "data/carepd/norm_params_h3d.pt"))
LATENT_NORM_PARAMS_PATH = os.environ.get(
    "UPDRS_LATENT_NORM_PARAMS", os.path.join(REPO_ROOT, "data/carepd/latent_norm_params_h3d.pt"))

# Generic aliases read by the training/eval scripts.
TRAIN_DATA_PATH         = H3D_TRAIN_DATA_PATH
TRAIN_LABELS_PATH       = H3D_TRAIN_LABELS_PATH
EVAL_DATA_PATH          = H3D_EVAL_DATA_PATH
EVAL_LABELS_PATH        = H3D_EVAL_LABELS_PATH
NORM_PARAMS_PATH        = H3D_NORM_PARAMS_PATH

# --- Model paths ---
MODEL_PATH = os.environ.get("UPDRS_MODEL_PATH", os.path.join(REPO_ROOT, "models/dit.pth"))
STATS_PATH = os.environ.get("UPDRS_STATS_PATH", os.path.join(REPO_ROOT, "models/stats.pth"))
VAE_MODEL_PATH = os.environ.get("UPDRS_VAE_MODEL_PATH", os.path.join(REPO_ROOT, "models/vae.pth"))

# --- Generation output ---
GEN_OUTPUT_PATH = os.environ.get(
    "UPDRS_GEN_OUTPUT_PATH",
    os.path.join(REPO_ROOT, "generated_data/generated_gait_updrs_dit.npy"))
EVAL_OUTPUT_DIR = os.path.join(REPO_ROOT, "graphs/graphs_evaluation/UPDRS_DiT")

# --- Architecture ---
EMBED_DIM = 256
N_HEADS = 4
N_LAYERS = 4
DROPOUT = 0.1
LATENT_CHANNELS = 64    # VAE bottleneck width
LATENT_TIME = 24        # SEQ_LEN / 4 (temporal compression r=4)

# --- VAE ---
VAE_BETA = 0.001        # KL weight
VAE_LAMBDA = float(os.environ.get("VAE_LAMBDA", "0.1"))   # surrogate UPDRS loss weight
VAE_EPOCHS = 2000
VAE_LR = 1e-3

# Class-prototype loss (MAD-VAE style). PROXIMITY pulls each sample toward its
# class prototype N(c_k, I); DISTANCE pushes prototypes apart.
PROTOTYPE_PROXIMITY_WEIGHT = float(os.environ.get("UPDRS_PROTOTYPE_PROX_WEIGHT", "0.01"))
PROTOTYPE_DISTANCE_WEIGHT  = float(os.environ.get("UPDRS_PROTOTYPE_DIST_WEIGHT", "0.01"))
PROTOTYPE_MARGIN           = float(os.environ.get("UPDRS_PROTOTYPE_MARGIN", "1.0"))

# --- DiT training ---
BATCH_SIZE = 32
TIMESTEPS = 500
LEARNING_RATE = 2e-5
EPOCHS = 3000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_WEIGHTED_LOSS = True   # inverse-frequency weighting on MSE/velocity losses
RESUME_TRAINING = False
START_EPOCH = 0

# --- Classifier-free guidance / generation ---
CFG_DROPOUT_PROB = float(os.environ.get("UPDRS_CFG_DROPOUT_PROB", "0.25"))
GUIDANCE_SCALE = 1.5
TEMPERATURE = 1.0
GEN_N_PER_CLASS = 300

# --- Loss weights ---
MSE_WEIGHT = 2.0
VELOCITY_WEIGHT = 0.1
ROM_WEIGHT = float(os.environ.get("UPDRS_ROM_WEIGHT", "0.1"))

# Per-class ROM targets (degrees) computed from real training data.
TARGET_ROM_DEG = torch.tensor([
    [51.94, 51.94, 57.87, 57.87, 24.21, 24.21],  # UPDRS 0 (Normal)
    [41.73, 41.73, 53.17, 53.17, 21.47, 21.47],  # UPDRS 1 (Mild)
    [26.54, 26.54, 39.82, 39.82, 14.58, 14.58],  # UPDRS 2 (Moderate)
    [13.71, 13.71, 23.98, 23.98, 7.92, 7.92],    # UPDRS 3 (Severe)
])  # (4, 6): [L_HipFlex, R_HipFlex, L_Knee, R_Knee, L_Ankle, R_Ankle]
