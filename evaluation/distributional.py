"""
Distributional evaluation of generated gait (paper Table 2).

Per UPDRS class and pooled over all classes:
  - LS-FID      : Fréchet distance in mean-pooled 64-dim VAE latent space
  - MMD²        : Maximum Mean Discrepancy on 132-dim 3D joint features
  - Precision   : k-NN precision (Kynkäänniemi et al., 2019)
  - Recall      : k-NN recall
  - Diversity   : synth/real mean pairwise L2 distance ratio in latent space

FID/MMD/Precision/Recall are bootstrapped over real subsets (mean ± std).

Usage:
    python -m evaluation.distributional [--n_boots 10] [--k_nn 3] [--gen_path PATH]
"""

import os
import sys
import argparse

import numpy as np
import scipy.linalg
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    EVAL_DATA_PATH, EVAL_LABELS_PATH, GEN_OUTPUT_PATH, EVAL_OUTPUT_DIR,
    VAE_MODEL_PATH, NORM_PARAMS_PATH, N_CHANNELS, LATENT_CHANNELS,
    UPDRS_CLASSES, DEVICE,
)
from training.vae_updrs import GaitVAE
from h3d_bridge import h3d_to_positions22

CLS_NAMES = {0: "Normal", 1: "Mild", 2: "Moderate", 3: "Severe"}
FK_BATCH = 512


def _joint_features(h3d_nct):
    """(N, 263, T) → (N, 132) per-joint temporal mean + std over 22 joints (metres)."""
    t = torch.from_numpy(np.ascontiguousarray(h3d_nct).astype(np.float32))
    chunks = [h3d_to_positions22(t[i:i + FK_BATCH].to(DEVICE)).cpu()
              for i in range(0, len(t), FK_BATCH)]
    pos = torch.cat(chunks, dim=0).numpy()
    mean3d = pos.mean(axis=1).reshape(len(h3d_nct), -1)
    std3d = pos.std(axis=1).reshape(len(h3d_nct), -1)
    return np.concatenate([mean3d, std3d], axis=1).astype(np.float64)


@torch.no_grad()
def _latents(data_nct, vae, mean_t, std_t, batch=64):
    """(N, 263, T) → mean-pooled VAE μ (N, 64)."""
    m, s = mean_t.to(DEVICE), std_t.to(DEVICE)
    x = torch.from_numpy(data_nct.astype(np.float32))
    mus = [vae.encode(torch.clamp((x[i:i + batch].to(DEVICE) - m) / s, -4, 4))[1].cpu()
           for i in range(0, len(x), batch)]
    return torch.cat(mus, dim=0).mean(dim=-1).numpy().astype(np.float64)


def _fid(real, synth, eps=1e-6):
    mu1, mu2 = real.mean(axis=0), synth.mean(axis=0)
    s1 = np.cov(real, rowvar=False) + eps * np.eye(real.shape[1])
    s2 = np.cov(synth, rowvar=False) + eps * np.eye(synth.shape[1])
    covmean = scipy.linalg.sqrtm(s1 @ s2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float((mu1 - mu2) @ (mu1 - mu2) + np.trace(s1 + s2 - 2.0 * covmean))


def _rbf(X, Y, gamma):
    x_sq = (X ** 2).sum(axis=1, keepdims=True)
    y_sq = (Y ** 2).sum(axis=1, keepdims=True)
    return np.exp(-gamma * np.clip(x_sq + y_sq.T - 2.0 * (X @ Y.T), 0, None))


def _mmd(X, Y):
    all_data = np.vstack([X, Y])
    sq = (all_data ** 2).sum(axis=1, keepdims=True)
    sq_dists = sq + sq.T - 2.0 * (all_data @ all_data.T)
    med = np.median(sq_dists[sq_dists > 0])
    gamma = 1.0 / (2.0 * med) if med > 1e-10 else 1.0
    return float(_rbf(X, X, gamma).mean() - 2.0 * _rbf(X, Y, gamma).mean()
                 + _rbf(Y, Y, gamma).mean())


def _precision_recall(real, synth, k):
    def dists(A, B):
        a_sq = (A ** 2).sum(axis=1, keepdims=True)
        b_sq = (B ** 2).sum(axis=1, keepdims=True)
        return np.sqrt(np.clip(a_sq + b_sq.T - 2.0 * (A @ B.T), 0, None))

    rr, ss = dists(real, real), dists(synth, synth)
    np.fill_diagonal(rr, np.inf)
    np.fill_diagonal(ss, np.inf)
    real_radii = np.partition(rr, k - 1, axis=1)[:, k - 1]
    synth_radii = np.partition(ss, k - 1, axis=1)[:, k - 1]
    rs = dists(real, synth)
    return float((rs <= real_radii[:, None]).any(axis=0).mean()), \
           float((rs <= synth_radii[None, :]).any(axis=1).mean())


def _diversity(lats, n_pairs=300, rng=None):
    rng = rng or np.random.default_rng(42)
    n = len(lats)
    if n < 2:
        return 0.0
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    j[i == j] = (j[i == j] + 1) % n
    return float(np.linalg.norm(lats[i] - lats[j], axis=1).mean())


def _bootstrap(real_nct, synth_nct, vae, mean_t, std_t, n_boots, rng, k_nn):
    """Bootstrap real subsets of the synthetic sample size; return metric lists."""
    n_real, n_synth = len(real_nct), len(synth_nct)
    synth_feats = _joint_features(synth_nct)
    real_feats = _joint_features(real_nct)
    synth_lats = _latents(synth_nct, vae, mean_t, std_t)
    real_lats = _latents(real_nct, vae, mean_t, std_t)

    out = {key: [] for key in ("fid", "mmd", "prec", "rec")}
    size = min(n_synth, n_real)
    for _ in range(n_boots):
        idx = rng.choice(n_real, size=size, replace=False)
        out["fid"].append(_fid(real_lats[idx], synth_lats))
        out["mmd"].append(_mmd(real_feats[idx], synth_feats))
        prec, rec = _precision_recall(real_lats[idx], synth_lats, k_nn)
        out["prec"].append(prec)
        out["rec"].append(rec)

    div = _diversity(synth_lats, rng=rng) / max(_diversity(real_lats, rng=rng), 1e-12)
    return out, div


def _fmt(vals):
    return f"{np.mean(vals):.4f} ± {np.std(vals):.3f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_boots", type=int, default=10)
    parser.add_argument("--k_nn", type=int, default=3)
    parser.add_argument("--gen_path", type=str, default=None)
    args = parser.parse_args()

    gen_path = args.gen_path or GEN_OUTPUT_PATH

    state = torch.load(VAE_MODEL_PATH, map_location=DEVICE)
    vae = GaitVAE(N_CHANNELS, LATENT_CHANNELS, UPDRS_CLASSES,
                  use_prototypes="prototypes" in state).to(DEVICE)
    vae.load_state_dict(state)
    vae.eval()
    norm = torch.load(NORM_PARAMS_PATH, map_location="cpu")
    mean_t, std_t = norm["mean"].float(), norm["std"].float()

    real = np.load(EVAL_DATA_PATH)
    real_labels = np.load(EVAL_LABELS_PATH)
    synth = np.load(gen_path)
    synth_labels = np.load(gen_path.replace(".npy", "_labels.npy"))

    def _nct(a):
        return a if a.ndim == 3 and a.shape[1] == N_CHANNELS else a.transpose(0, 2, 1)

    real_nct, synth_nct = _nct(real), _nct(synth)
    rng = np.random.default_rng(42)
    classes = sorted(np.unique(np.concatenate([real_labels, synth_labels])).tolist())

    print("=" * 78)
    print(f"  Distributional metrics  (n_boots={args.n_boots}, k_nn={args.k_nn})")
    print("=" * 78)
    print(f"  {'Class':<16} {'LS-FID':>16} {'MMD²':>16} {'Precision':>10} {'Recall':>10} {'Div':>6}")
    print("  " + "-" * 78)

    rows = []
    for c in classes:
        r_cls, s_cls = real_nct[real_labels == c], synth_nct[synth_labels == c]
        if len(s_cls) == 0:
            continue
        res, div = _bootstrap(r_cls, s_cls, vae, mean_t, std_t, args.n_boots, rng, args.k_nn)
        rows.append((c, res["fid"], res["mmd"], res["prec"], res["rec"], div))
        print(f"  UPDRS {c} {CLS_NAMES[c]:<8} {_fmt(res['fid']):>16} {_fmt(res['mmd']):>16} "
              f"{_fmt(res['prec']):>10} {_fmt(res['rec']):>10} {div:>6.3f}")

    res_all, div_all = _bootstrap(real_nct, synth_nct, vae, mean_t, std_t,
                                  args.n_boots, rng, args.k_nn)
    print("  " + "-" * 78)
    print(f"  {'Overall':<16} {_fmt(res_all['fid']):>16} {_fmt(res_all['mmd']):>16} "
          f"{_fmt(res_all['prec']):>10} {_fmt(res_all['rec']):>10} {div_all:>6.3f}")

    os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(EVAL_OUTPUT_DIR, "distributional_metrics.txt")
    with open(out_path, "w") as f:
        f.write(f"Distributional metrics (n_boots={args.n_boots}, k_nn={args.k_nn})\n")
        for c, fid, mmd, prec, rec, div in rows:
            f.write(f"UPDRS{c} fid={np.mean(fid):.4f} mmd={np.mean(mmd):.4f} "
                    f"prec={np.mean(prec):.4f} rec={np.mean(rec):.4f} div={div:.3f}\n")
        f.write(f"Overall fid={np.mean(res_all['fid']):.4f} mmd={np.mean(res_all['mmd']):.4f} "
                f"prec={np.mean(res_all['prec']):.4f} rec={np.mean(res_all['rec']):.4f} "
                f"div={div_all:.3f}\n")
    print(f"\n  Report: {out_path}")


if __name__ == "__main__":
    main()
