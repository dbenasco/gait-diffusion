"""
Biomechanical evaluation of generated gait (paper Table 1).

Per UPDRS class, averaged over the six sagittal leg joints (L/R hip, knee,
ankle):
  - ROM (real / synthetic), degrees
  - ROM error (%)
  - DTW (Sakoe-Chiba band = 10% of sequence length)

Also reports arm-swing amplitude and trunk inclination (real vs synthetic).
Synthetic samples are compared against VAE-reconstructed real data when a VAE
checkpoint is available, so both sides pass through the same decoder.

Usage:
    python -m evaluation.synthetic [--n_samples 300] [--gen_path PATH] [--vae_path PATH]
"""

import os
import sys
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    EVAL_DATA_PATH, EVAL_LABELS_PATH, GEN_OUTPUT_PATH,
    VAE_MODEL_PATH, NORM_PARAMS_PATH, EVAL_OUTPUT_DIR,
    N_CHANNELS, LATENT_CHANNELS, UPDRS_CLASSES,
)
from training.vae_updrs import GaitVAE
from h3d_bridge import (
    h3d_to_angles, h3d_to_positions22, arm_swing_range, trunk_inclination,
)

CLS_NAMES = {0: "Normal", 1: "Mild", 2: "Moderate", 3: "Severe"}
DTW_BAND = 0.10


def _dtw_1d(a, b, window):
    """Sakoe-Chiba DTW, normalized by path length. Lower = better."""
    n, m = len(a), len(b)
    window = max(window, abs(n - m))
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(max(1, i - window), min(m, i + window) + 1):
            c = abs(float(a[i - 1]) - float(b[j - 1]))
            cost[i, j] = c + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    return float(cost[n, m]) / (n + m)


try:
    from numba import njit

    @njit(cache=True)
    def _dtw_1d_kernel(a, b, window):
        n, m = len(a), len(b)
        window = max(window, abs(n - m))
        cost = np.full((n + 1, m + 1), np.inf)
        cost[0, 0] = 0.0
        for i in range(1, n + 1):
            for j in range(max(1, i - window), min(m, i + window) + 1):
                c = abs(a[i - 1] - b[j - 1])
                cost[i, j] = c + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
        return cost[n, m] / (n + m)

    _dtw_1d_kernel(np.zeros(4), np.zeros(4), 2)

    def _dtw_1d(a, b, window):
        return float(_dtw_1d_kernel(a.astype(np.float64), b.astype(np.float64), int(window)))
except ImportError:
    pass


def _to_tensor(x):
    return torch.from_numpy(np.ascontiguousarray(x).astype(np.float32))


@torch.no_grad()
def _vae_roundtrip(real_nct, vae, mean_t, std_t, batch=64):
    """Encode real (N, 263, T) through the VAE (μ) then decode; bridge to (N, 6, T) degrees."""
    x = torch.from_numpy(real_nct.astype(np.float32))
    x = torch.clamp((x - mean_t) / std_t, -4, 4)
    rec = [vae.decode(vae.encode(x[i:i + batch])[1]) for i in range(0, len(x), batch)]
    rec = torch.cat(rec, dim=0) * std_t + mean_t
    return h3d_to_angles(rec, sagittal_only=True).numpy()


def _rom(angles_nct):
    """(N, C, T) → (C,) mean per-sample range of motion."""
    return (angles_nct.max(axis=2) - angles_nct.min(axis=2)).mean(axis=0)


def _mean_dtw(synth_nct, real_nct):
    """(C,) DTW between synthetic and real class-mean waveforms."""
    n_ch, T = synth_nct.shape[1], synth_nct.shape[2]
    window = max(1, int(DTW_BAND * T))
    s_mean, r_mean = synth_nct.mean(axis=0), real_nct.mean(axis=0)
    return np.array([_dtw_1d(s_mean[c], r_mean[c], window) for c in range(n_ch)])


def _load_vae():
    if not (os.path.exists(VAE_MODEL_PATH) and os.path.exists(NORM_PARAMS_PATH)):
        print(f"WARNING: VAE/norm params not found — using raw real data as reference.")
        return None, None, None
    state = torch.load(VAE_MODEL_PATH, map_location="cpu")
    vae = GaitVAE(N_CHANNELS, LATENT_CHANNELS, UPDRS_CLASSES,
                  use_prototypes="prototypes" in state)
    vae.load_state_dict(state)
    vae.eval()
    norm = torch.load(NORM_PARAMS_PATH, map_location="cpu")
    return vae, norm["mean"].float(), norm["std"].float()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=300,
                        help="Max synthetic samples per class")
    parser.add_argument("--gen_path", type=str, default=None)
    parser.add_argument("--vae_path", type=str, default=None)
    args = parser.parse_args()

    global GEN_OUTPUT_PATH, VAE_MODEL_PATH
    if args.gen_path:
        GEN_OUTPUT_PATH = args.gen_path
    if args.vae_path:
        VAE_MODEL_PATH = args.vae_path

    synth = np.load(GEN_OUTPUT_PATH)
    synth_labels = np.load(GEN_OUTPUT_PATH.replace(".npy", "_labels.npy"))
    real = np.load(EVAL_DATA_PATH)
    real_labels = np.load(EVAL_LABELS_PATH)

    rng = np.random.default_rng(42)
    keep = np.concatenate([
        rng.choice(np.where(synth_labels == c)[0],
                   size=min(args.n_samples, int((synth_labels == c).sum())), replace=False)
        for c in np.unique(synth_labels)
    ])
    synth, synth_labels = synth[keep], synth_labels[keep]

    vae, mean_t, std_t = _load_vae()

    synth_ang = h3d_to_angles(_to_tensor(synth), sagittal_only=True).numpy()
    real_ang = h3d_to_angles(_to_tensor(real), sagittal_only=True).numpy()
    synth_pos = h3d_to_positions22(_to_tensor(synth)).numpy()
    real_pos = h3d_to_positions22(_to_tensor(real)).numpy()
    synth_arm, real_arm = arm_swing_range(synth_pos), arm_swing_range(real_pos)
    synth_trunk, real_trunk = trunk_inclination(synth_pos), trunk_inclination(real_pos)

    classes = sorted(np.unique(np.concatenate([synth_labels, real_labels])).tolist())

    rows = []
    for c in classes:
        s_mask, r_mask = synth_labels == c, real_labels == c
        s_ang = synth_ang[s_mask]
        r_ang_raw = real[r_mask]
        r_ang = _vae_roundtrip(r_ang_raw, vae, mean_t, std_t) if vae is not None else real_ang[r_mask]

        rom_s, rom_r = _rom(s_ang), _rom(r_ang)
        rom_err = (rom_s.mean() - rom_r.mean()) / rom_r.mean() * 100.0
        dtw = _mean_dtw(s_ang, r_ang).mean()
        rows.append((c, rom_r.mean(), rom_s.mean(), rom_err, dtw,
                     real_arm[r_mask].mean(), synth_arm[s_mask].mean(),
                     real_trunk[r_mask].mean(), synth_trunk[s_mask].mean()))
        print(f"UPDRS {c} ({CLS_NAMES[c]}): {int(s_mask.sum())} synth / {int(r_mask.sum())} real")

    print("\n" + "=" * 74)
    print("  Biomechanical metrics (mean over 6 sagittal leg joints)")
    print("=" * 74)
    print(f"  {'Class':<16} {'Real ROM':>9} {'Synth ROM':>10} {'ROM err %':>10} {'DTW':>8}")
    print("  " + "-" * 58)
    for c, rr, rs, re, dtw, *_ in rows:
        print(f"  UPDRS {c} {CLS_NAMES[c]:<8} {rr:>9.2f} {rs:>10.2f} {re:>10.1f} {dtw:>8.3f}")
    print("  " + "-" * 58)
    print(f"  {'Overall':<16} {np.mean([r[1] for r in rows]):>9.2f} "
          f"{np.mean([r[2] for r in rows]):>10.2f} "
          f"{np.mean([abs(r[3]) for r in rows]):>10.1f} "
          f"{np.mean([r[4] for r in rows]):>8.3f}")

    print("\n" + "=" * 74)
    print("  Biomechanical biomarkers")
    print("=" * 74)
    print(f"  {'Class':<16} {'Arm real':>9} {'Arm synth':>10} {'Trunk real':>11} {'Trunk synth':>12}")
    print("  " + "-" * 62)
    for c, _, _, _, _, ar, asy, tr, ts in rows:
        print(f"  UPDRS {c} {CLS_NAMES[c]:<8} {ar:>9.4f} {asy:>10.4f} {tr:>11.2f} {ts:>12.2f}")

    rom_ok = all(rows[i][2] > rows[i + 1][2] for i in range(len(rows) - 1))
    print(f"\n  Cross-class ROM ordering (monotonic decrease): {'PASS' if rom_ok else 'FAIL'}")

    os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(EVAL_OUTPUT_DIR, "biomechanical_metrics.txt")
    with open(out_path, "w") as f:
        f.write("Biomechanical metrics (mean over 6 sagittal leg joints)\n")
        f.write("class real_rom synth_rom rom_err_pct dtw arm_real arm_synth trunk_real trunk_synth\n")
        for c, rr, rs, re, dtw, ar, asy, tr, ts in rows:
            f.write(f"UPDRS{c} {rr:.2f} {rs:.2f} {re:.2f} {dtw:.4f} "
                    f"{ar:.4f} {asy:.4f} {tr:.2f} {ts:.2f}\n")
    print(f"\n  Report: {out_path}")


if __name__ == "__main__":
    main()
