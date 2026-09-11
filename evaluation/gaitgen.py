"""
evaluation/gaitgen.py — AVE, AAMD, ASMD, MPJPE, PA-MPJPE, ACCL
as defined in GAITGen (arXiv:2503.22397, Appendix H + Table Sx).

  AVE  — Average Variance Error: how closely generated joint-position variance matches real.
  AAMD — Absolute Arm Swing Mean Difference: per-class |arm_swing_synth - arm_swing_real|.
  ASMD — Absolute Stooped Posture Mean Difference: per-class |stooped_posture_synth - real|.
  MPJPE/PA-MPJPE/ACCL — VAE *reconstruction* quality on real eval data
                        (encode → decode, compare input vs. reconstruction in joint space).

AAMD and ASMD are normalised by leg length; AVE is in raw position units
(metres, variance scale) following the GAITGen supplementary definition.
Reconstruction metrics (MPJPE/PA-MPJPE/ACCL) are in mm.

Usage:
    python -m evaluation.gaitgen [--n_samples 300] [--vae_path PATH]
"""

import argparse
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    EVAL_DATA_PATH, EVAL_LABELS_PATH,
    GEN_OUTPUT_PATH, GEN_N_PER_CLASS, UPDRS_CLASSES,
    VAE_MODEL_PATH, NORM_PARAMS_PATH, N_CHANNELS, LATENT_CHANNELS, DEVICE,
)
from h3d_bridge import h3d_to_positions22
from training.vae_updrs import GaitVAE


# ── SMPL 22-joint indices ────────────────────────────────────────────────────
PELVIS      = 0
L_ANKLE     = 7
R_ANKLE     = 8
NECK        = 12
L_SHOULDER  = 16
R_SHOULDER  = 17
L_WRIST     = 20
R_WRIST     = 21

JOINT_NAMES = [
    "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee",
    "Spine2", "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot",
    "Neck", "L_Collar", "R_Collar", "Head",
    "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist",
]

Y_AXIS = 1   # Y-up in HumanML3D canonical frame
BATCH  = 512 # chunk size for h3d_to_positions22 to keep memory manageable


# ── Utilities ────────────────────────────────────────────────────────────────

def to_pos(h3d_nct: np.ndarray) -> np.ndarray:
    """(N, 263, T) → (N, T, 22, 3) pelvis-relative joint positions in metres."""
    chunks = []
    for start in range(0, len(h3d_nct), BATCH):
        chunk = h3d_nct[start : start + BATCH]
        t = torch.from_numpy(np.ascontiguousarray(chunk).astype(np.float32))
        chunks.append(h3d_to_positions22(t).numpy())
    return np.concatenate(chunks, axis=0)


def _leg_length(pos: np.ndarray) -> float:
    """Mean distance pelvis→ankle (both sides, all frames) in metres."""
    # Pelvis is at origin in pelvis-relative frame, so ||ankle_pos||₂ = leg length.
    l = np.linalg.norm(pos[:, :, L_ANKLE], axis=-1).mean()
    r = np.linalg.norm(pos[:, :, R_ANKLE], axis=-1).mean()
    return float((l + r) / 2.0)


def _arm_swing(pos: np.ndarray, leg_len: float) -> np.ndarray:
    """
    Per-sample arm swing range, normalised by leg length. Returns (N,).
    GAITGen definition: max−min of wrist-to-shoulder Euclidean distance
    over T, min of L/R arms, divided by leg length.
    """
    l_dist = np.linalg.norm(pos[:, :, L_WRIST] - pos[:, :, L_SHOULDER], axis=-1)  # (N, T)
    r_dist = np.linalg.norm(pos[:, :, R_WRIST] - pos[:, :, R_SHOULDER], axis=-1)
    l_range = l_dist.max(axis=1) - l_dist.min(axis=1)   # (N,)
    r_range = r_dist.max(axis=1) - r_dist.min(axis=1)
    return np.minimum(l_range, r_range) / leg_len


def _stooped_posture(pos: np.ndarray, leg_len: float) -> np.ndarray:
    """
    Per-sample mean vertical neck height above pelvis, normalised by leg length.
    Returns (N,). Lower = more stooped.
    """
    neck_y = pos[:, :, NECK, Y_AXIS]          # (N, T)
    return neck_y.mean(axis=1) / leg_len       # (N,)


# ── AVE ──────────────────────────────────────────────────────────────────────

def compute_ave(real_h3d: np.ndarray, synth_h3d: np.ndarray):
    """
    Average Variance Error (GAITGen, Appendix H).

    For each joint j:
        σ[j]  = (1/(T-1)) Σ_t (P_t[j] − P̄[j])²   (temporal variance, 3D per sample)
        σ̂[j]  = same for generated samples
    Aggregate to set-level variance by averaging σ[j] over the real set and
    σ̂[j] over the generated set, then:
        AVE[j] = ||mean(σ_real[j]) − mean(σ_synth[j])||₂
    AVE = mean over all 22 joints.

    Returns (ave_per_joint (22,), ave_scalar).
    """
    print("  Computing positions for AVE...")
    real_pos  = to_pos(real_h3d)    # (N, T, 22, 3)
    synth_pos = to_pos(synth_h3d)

    # ddof=1 matches GAITGen's 1/(T-1) temporal variance definition.
    var_real  = real_pos.var(axis=1, ddof=1).mean(axis=0)    # (22, 3)
    var_synth = synth_pos.var(axis=1, ddof=1).mean(axis=0)

    ave_per_joint = np.linalg.norm(var_real - var_synth, axis=-1)   # (22,)
    return ave_per_joint, float(ave_per_joint.mean())


# ── AAMD ─────────────────────────────────────────────────────────────────────

def compute_aamd(real_by_cls: dict, synth_by_cls: dict):
    """
    AAMD = (1/C) Σ_c | mean_arm_swing_synth^(c) − mean_arm_swing_real^(c) |
    Returns (aamd_scalar, list of (cls, synth_val, real_val, abs_diff)).
    """
    details = []
    for cls in sorted(real_by_cls):
        if len(real_by_cls[cls]) == 0 or len(synth_by_cls[cls]) == 0:
            continue
        r_pos = to_pos(real_by_cls[cls])
        s_pos = to_pos(synth_by_cls[cls])
        ll    = _leg_length(np.concatenate([r_pos, s_pos], axis=0))
        as_r  = float(_arm_swing(r_pos, ll).mean())
        as_s  = float(_arm_swing(s_pos, ll).mean())
        details.append((cls, as_s, as_r, abs(as_s - as_r)))
    aamd = float(np.mean([d[3] for d in details]))
    return aamd, details


# ── ASMD ─────────────────────────────────────────────────────────────────────

def compute_asmd(real_by_cls: dict, synth_by_cls: dict):
    """
    ASMD = (1/C) Σ_c | mean_stooped_synth^(c) − mean_stooped_real^(c) |
    Returns (asmd_scalar, list of (cls, synth_val, real_val, abs_diff)).
    """
    details = []
    for cls in sorted(real_by_cls):
        if len(real_by_cls[cls]) == 0 or len(synth_by_cls[cls]) == 0:
            continue
        r_pos = to_pos(real_by_cls[cls])
        s_pos = to_pos(synth_by_cls[cls])
        ll    = _leg_length(np.concatenate([r_pos, s_pos], axis=0))
        sp_r  = float(_stooped_posture(r_pos, ll).mean())
        sp_s  = float(_stooped_posture(s_pos, ll).mean())
        details.append((cls, sp_s, sp_r, abs(sp_s - sp_r)))
    asmd = float(np.mean([d[3] for d in details]))
    return asmd, details


# ── Reconstruction MPJPE / PA-MPJPE / ACCL ───────────────────────────────────

RECON_BS = 64  # batch size for VAE forward pass


def _procrustes(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Procrustes-aligned X to Y (rigid rotation + uniform scale).
    X, Y are (J, 3). Returns aligned X.
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    XX = np.dot(Xc.T, Xc)
    U, _, Vt = np.linalg.svd(np.dot(Xc.T, Yc))
    R = np.dot(U, Vt)
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = np.dot(U, Vt)
    s = np.trace(np.dot(np.dot(Yc.T, Xc), R)) / max(np.trace(XX), 1e-12)
    s = max(s, 1e-6)
    return s * np.dot(Xc, R) + Y.mean(axis=0, keepdims=True)


def compute_recon_mpjpe(real_by_cls: dict, vae, mean_t, std_t, device,
                        procrustes: bool = False):
    """
    VAE reconstruction MPJPE / PA-MPJPE / ACCL on real eval data.

    For each class: normalise → encode → decode → denormalise → convert
    to 22-joint positions → per-sample MPJPE and ACCL.  If procrustes=True,
    applies per-frame Procrustes rigid alignment before computing error.

    Returns (class_mpjpe dict, scalar_mpjpe_mm, scalar_accl_mm/frame).
    """
    class_mpjpe = {}
    class_accl  = {}
    MEAN = mean_t.to(device)
    STD  = std_t.to(device)

    for cls in sorted(real_by_cls):
        arr = real_by_cls[cls]                                   # (N, 263, T)
        N = arr.shape[0]
        mpjpe_sum = 0.0
        accl_sum  = 0.0

        for start in range(0, N, RECON_BS):
            end = min(start + RECON_BS, N)
            x_raw = torch.from_numpy(arr[start:end]).float().to(device)   # (B, 263, T)
            x_norm = torch.clamp((x_raw - MEAN) / STD, -4, 4)

            with torch.no_grad():
                z, _, _ = vae.encode(x_norm)                  # (B, 64, 24)
                recon_norm = vae.decode(z)

            recon_raw = recon_norm * STD + MEAN

            real_pos = h3d_to_positions22(x_raw).cpu().numpy()       # (B, T, 22, 3)
            recon_pos = h3d_to_positions22(recon_raw).cpu().numpy()  # (B, T, 22, 3)

            if procrustes:
                for b in range(recon_pos.shape[0]):
                    for t in range(recon_pos.shape[1]):
                        recon_pos[b, t] = _procrustes(
                            recon_pos[b, t], real_pos[b, t]          # (22,3) per frame
                        )

            diff = recon_pos - real_pos                             # (B, T, 22, 3)
            mpjpe_batch = np.linalg.norm(diff, axis=-1).mean(axis=(-1, -2))   # (B,)
            mpjpe_sum += float(mpjpe_batch.sum())

            if not procrustes:
                vel_real  = real_pos[:, 1:] - real_pos[:, :-1]          # (B, T-1, 22, 3)
                vel_recon = recon_pos[:, 1:] - recon_pos[:, :-1]
                accl_real  = vel_real[:, 1:] - vel_real[:, :-1]         # (B, T-2, 22, 3)
                accl_recon = vel_recon[:, 1:] - vel_recon[:, :-1]
                accl_diff = np.linalg.norm(accl_recon - accl_real, axis=-1)
                accl_batch = accl_diff.mean(axis=(-1, -2))               # (B,)
                accl_sum += float(accl_batch.sum())

        class_mpjpe[cls] = mpjpe_sum / N * 1000  # m → mm
        class_accl[cls]  = accl_sum / N * 1000 if not procrustes else 0.0

    mpjpe = float(np.mean(list(class_mpjpe.values())))
    accl  = float(np.mean(list(class_accl.values())))
    return class_mpjpe, mpjpe, accl


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=300,
                        help="Max synth samples per class (default 300)")
    parser.add_argument("--vae_path", type=str, default=None,
                        help="VAE checkpoint (default from config: %s)" % VAE_MODEL_PATH)
    parser.add_argument("--gen_output", type=str, default=None,
                        help="Override generated data path (default from config)")
    args = parser.parse_args()

    CLS_NAMES = {0: "Normal", 1: "Mild", 2: "Moderate", 3: "Severe"}
    rng = np.random.default_rng(42)

    print("=" * 60)
    print("GAITGen Metrics  (arXiv:2503.22397, Appendix H)")
    print("=" * 60)

    # ── Load VAE for reconstruction metrics ────────────────────────────────────
    vae_path = args.vae_path or VAE_MODEL_PATH
    print(f"\nLoading VAE from {vae_path} ...")
    if not os.path.exists(vae_path):
        print(f"  WARNING: VAE not found at {vae_path} — reconstruction metrics skipped")
        skip_recon = True
        vae = mean_t = std_t = None
    else:
        ckpt = torch.load(vae_path, map_location=DEVICE)
        use_protos = "prototypes" in ckpt
        vae = GaitVAE(N_CHANNELS, LATENT_CHANNELS, UPDRS_CLASSES,
                      use_prototypes=use_protos).to(DEVICE)
        vae.load_state_dict(ckpt)
        vae.eval()
        print(f"  Loaded VAE (use_prototypes={use_protos})")

        norm = torch.load(NORM_PARAMS_PATH, map_location=DEVICE)
        mean_t = norm['mean'].to(DEVICE)   # (263, 1)
        std_t  = norm['std'].to(DEVICE)
        skip_recon = False

    # ── Load data ──────────────────────────────────────────────────────────────
    print("\nLoading data...")
    real_all  = np.load(EVAL_DATA_PATH)    # (N, 263, T)
    real_lbls = np.load(EVAL_LABELS_PATH)  # (N,)
    gen_path  = args.gen_output or GEN_OUTPUT_PATH
    synth_all = np.load(gen_path)          # (M, 263, T)

    # Synth labels: GEN_N_PER_CLASS per class in order 0, 1, 2
    synth_lbls = np.repeat(np.arange(UPDRS_CLASSES), GEN_N_PER_CLASS)
    if len(synth_lbls) != len(synth_all):
        per = len(synth_all) // UPDRS_CLASSES
        synth_lbls = np.repeat(np.arange(UPDRS_CLASSES), per)
        synth_all  = synth_all[:len(synth_lbls)]

    print(f"  Real:  {real_all.shape}   Synth: {synth_all.shape}")

    real_by_cls  = {}
    synth_by_cls = {}
    for cls in range(UPDRS_CLASSES):
        r_idx = np.where(real_lbls == cls)[0]
        s_idx = np.where(synth_lbls == cls)[0]
        real_by_cls[cls]  = real_all[r_idx]
        n = min(args.n_samples, len(s_idx))
        synth_by_cls[cls] = synth_all[rng.choice(s_idx, size=n, replace=False)]
        print(f"  UPDRS {cls}: {len(r_idx)} real / {n} synth")

    # ── AVE ──────────────────────────────────────────────────────────────────
    print("\n── AVE (Average Variance Error) — lower is better ──────────────")
    ave_per_joint, ave = compute_ave(real_all, synth_all)
    worst5 = np.argsort(ave_per_joint)[::-1][:5]
    print("  Per-joint (top 5 worst):")
    for j in worst5:
        print(f"    {JOINT_NAMES[j]:<14}  {ave_per_joint[j]:.5f}")
    print(f"  AVE (all 22 joints mean): {ave:.5f}")

    # ── AAMD ─────────────────────────────────────────────────────────────────
    print("\n── AAMD (Absolute Arm Swing Mean Difference) — lower is better ─")
    print("  (arm swing = wrist-to-shoulder range over time, normalised by leg length)")
    aamd, aamd_det = compute_aamd(real_by_cls, synth_by_cls)
    print(f"\n  {'Class':<18} {'Synth':>8} {'Real':>8} {'|diff|':>8}")
    print("  " + "─" * 46)
    for cls, s, r, d in aamd_det:
        print(f"  UPDRS {cls} {CLS_NAMES[cls]:<10}  {s:>8.4f}  {r:>8.4f}  {d:>8.4f}")
    print(f"  AAMD: {aamd:.5f}")

    # ── ASMD ─────────────────────────────────────────────────────────────────
    print("\n── ASMD (Absolute Stooped Posture Mean Difference) — lower is better ─")
    print("  (stooped posture = mean vertical neck height above pelvis, normalised by leg length)")
    asmd, asmd_det = compute_asmd(real_by_cls, synth_by_cls)
    print(f"\n  {'Class':<18} {'Synth':>8} {'Real':>8} {'|diff|':>8}")
    print("  " + "─" * 46)
    for cls, s, r, d in asmd_det:
        print(f"  UPDRS {cls} {CLS_NAMES[cls]:<10}  {s:>8.4f}  {r:>8.4f}  {d:>8.4f}")
    print(f"  ASMD: {asmd:.5f}")

    # ── VAE Reconstruction: MPJPE / PA-MPJPE / ACCL ────────────────────────────
    if skip_recon:
        mpjpe = pampjpe = accl = 0.0
        print("\n── VAE reconstruction metrics: SKIPPED (VAE not found) ────")
    else:
        print("\n── VAE Reconstruction MPJPE — lower is better ──────────────────")
        mpjpe_cls, mpjpe, accl = compute_recon_mpjpe(
            real_by_cls, vae, mean_t, std_t, DEVICE, procrustes=False)
        for cls in sorted(mpjpe_cls):
            print(f"  UPDRS {cls} {CLS_NAMES[cls]:<10}  {mpjpe_cls[cls]:.2f} mm")
        print(f"  MPJPE (mean): {mpjpe:.2f} mm  |  ACCL: {accl:.2f} mm/frame²")

        print("\n── VAE Reconstruction PA-MPJPE (Procrustes-Aligned) ──────────")
        pa_cls, pampjpe, _ = compute_recon_mpjpe(
            real_by_cls, vae, mean_t, std_t, DEVICE, procrustes=True)
        for cls in sorted(pa_cls):
            print(f"  UPDRS {cls} {CLS_NAMES[cls]:<10}  {pa_cls[cls]:.2f} mm")
        print(f"  PA-MPJPE (mean): {pampjpe:.2f} mm")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  AVE       = {ave:.5f}")
    print(f"  AAMD      = {aamd:.5f}")
    print(f"  ASMD      = {asmd:.5f}")
    print(f"  MPJPE     = {mpjpe:.2f} mm")
    print(f"  PA-MPJPE  = {pampjpe:.2f} mm")
    print(f"  ACCL      = {accl:.2f} mm/frame\u00b2")
    print("=" * 60)


if __name__ == "__main__":
    main()
