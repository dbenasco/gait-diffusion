"""
Bootstrapped distributional metrics: FID, MMD², Precision, Recall.

For each UPDRS class, draws N_BOOTS random subsets of real data and computes
each metric against the fixed synthetic set, then reports mean ± std.
A real-vs-real baseline is computed the same way for comparison.

Usage:
    python -m evaluation.distributional [--n_boots 10]
"""
import argparse
import os
import sys

import numpy as np
import scipy.linalg
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    EVAL_DATA_PATH,
    EVAL_LABELS_PATH,
    GEN_OUTPUT_PATH,
    EVAL_OUTPUT_DIR,
    VAE_MODEL_PATH,
    NORM_PARAMS_PATH,
    N_CHANNELS,
    LATENT_CHANNELS,
    UPDRS_CLASSES as N_UPDRS_CLASSES,
)
from training.vae_updrs import GaitVAE
from h3d_bridge import h3d_to_positions22 as _h3d_to_positions22

_DEVICE   = None   # GPU device; set in main() after VAE load
_FK_BATCH = 512    # FK samples per GPU batch — limits peak VRAM to ~512×263×96×4 ≈ 50 MB


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_3d_features(h3d_nct):
    """(N, 263, T) H3D → (N, 132) 3D joint features.
    Per-joint temporal mean (66) + std (66) over all 22 SMPL joints in metres.
    Uniform units, all joints contribute equally — arms, trunk, root included."""
    device = _DEVICE or torch.device('cpu')
    t = torch.from_numpy(np.ascontiguousarray(h3d_nct).astype(np.float32))
    chunks = []
    for i in range(0, len(t), _FK_BATCH):
        chunks.append(_h3d_to_positions22(t[i:i + _FK_BATCH].to(device)).cpu())
    pos = torch.cat(chunks, dim=0).numpy()              # (N, T, 22, 3)
    mean3d = pos.mean(axis=1).reshape(len(h3d_nct), -1)
    std3d  = pos.std(axis=1).reshape(len(h3d_nct), -1)
    return np.concatenate([mean3d, std3d], axis=1).astype(np.float64)  # (N, 132)


@torch.no_grad()
def _vae_encode_latents(data_nct, vae_model, mean_t, std_t, batch=64):
    """(N, C, T) → mean-pooled VAE μ (N, C_lat).
    Mean-pooling over the time axis gives a compact domain-specific embedding
    (64-dim) suitable for FID — analogous to how MDM pools transformer tokens.
    Flattening to (N, 1536) is underdetermined with our sample sizes."""
    device = next(vae_model.parameters()).device
    m = mean_t.to(device)
    s = std_t.to(device)
    x_t = torch.from_numpy(data_nct.astype(np.float32))
    mus = []
    for i in range(0, len(x_t), batch):
        x = torch.clamp((x_t[i:i + batch].to(device) - m) / s, -4, 4)
        _, mu, _ = vae_model.encode(x)
        mus.append(mu.cpu())
    mu_all = torch.cat(mus, dim=0)                       # (N, 64, 24)
    return mu_all.mean(dim=-1).numpy().astype(np.float64)  # (N, 64)


def _compute_fid(real_feats, synth_feats, eps=1e-6):
    """Fréchet distance between two feature sets. Lower = better."""
    mu1, mu2 = real_feats.mean(axis=0), synth_feats.mean(axis=0)
    sigma1 = np.cov(real_feats, rowvar=False) + eps * np.eye(real_feats.shape[1])
    sigma2 = np.cov(synth_feats, rowvar=False) + eps * np.eye(synth_feats.shape[1])
    diff = mu1 - mu2
    covmean = scipy.linalg.sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2.0 * covmean))


def _rbf_kernel(X, Y, gamma):
    X_sq = (X ** 2).sum(axis=1, keepdims=True)
    Y_sq = (Y ** 2).sum(axis=1, keepdims=True)
    return np.exp(-gamma * np.clip(X_sq + Y_sq.T - 2.0 * (X @ Y.T), 0, None))


def _compute_mmd(X, Y):
    """MMD² with RBF kernel and median heuristic gamma. Lower = better."""
    all_data = np.vstack([X, Y])
    X_sq = (all_data ** 2).sum(axis=1, keepdims=True)
    sq_dists = X_sq + X_sq.T - 2.0 * (all_data @ all_data.T)
    median_sq = np.median(sq_dists[sq_dists > 0])
    gamma = 1.0 / (2.0 * median_sq) if median_sq > 1e-10 else 1.0
    XX = _rbf_kernel(X, X, gamma).mean()
    YY = _rbf_kernel(Y, Y, gamma).mean()
    XY = _rbf_kernel(X, Y, gamma).mean()
    return float(XX - 2.0 * XY + YY)


def _compute_diversity(lats, n_pairs=300, rng=None):
    """
    Standard diversity from motion generation literature (ACTOR, MDM):
    mean L2 distance over n_pairs randomly drawn pairs in latent space.
    Higher = more diverse, less mode collapse.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    N = len(lats)
    if N < 2:
        return 0.0
    i_idx = rng.integers(0, N, size=n_pairs)
    j_idx = rng.integers(0, N, size=n_pairs)
    collide = i_idx == j_idx
    j_idx[collide] = (j_idx[collide] + 1) % N
    return float(np.linalg.norm(lats[i_idx] - lats[j_idx], axis=1).mean())


def _compute_precision_recall(real_feats, synth_feats, k=3):
    """Kynkäänniemi et al. 2019 k-NN Precision and Recall. Higher = better."""
    def _dists(A, B):
        A_sq = (A ** 2).sum(axis=1, keepdims=True)
        B_sq = (B ** 2).sum(axis=1, keepdims=True)
        return np.sqrt(np.clip(A_sq + B_sq.T - 2.0 * (A @ B.T), 0, None))

    rr = _dists(real_feats,  real_feats)
    ss = _dists(synth_feats, synth_feats)
    np.fill_diagonal(rr, np.inf)
    np.fill_diagonal(ss, np.inf)
    real_radii  = np.partition(rr, k - 1, axis=1)[:, k - 1]
    synth_radii = np.partition(ss, k - 1, axis=1)[:, k - 1]
    rs = _dists(real_feats, synth_feats)
    precision = float((rs <= real_radii[:, None]).any(axis=0).mean())
    recall    = float((rs <= synth_radii[None, :]).any(axis=1).mean())
    return precision, recall


def _bootstrap_class(real_nct, synth_nct, vae_model, mean_t, std_t,
                     n_boots, rng, k_nn=3):
    """
    Run n_boots bootstrap iterations for one UPDRS class.
    Returns (results_dict, synth_lats, all_real_lats).
    All features/latents are extracted once up front; bootstrap loops only slice arrays.
    """
    n_real  = len(real_nct)
    # Cap synth at n_real//2 so the bootstrap can draw two non-overlapping real subsets
    # of equal size for the baseline. If n_synth > n_real//2 the bootstrap degenerates
    # (baseline compares real to itself → FID=0, std=0 everywhere).
    n_synth_full = len(synth_nct)
    cap = n_real // 2
    if n_synth_full > cap:
        idx_cap  = rng.choice(n_synth_full, size=cap, replace=False)
        synth_nct = synth_nct[idx_cap]
        print(f"  ⚠️  n_synth ({n_synth_full}) > n_real//2 ({cap}) — subsampling synth to {cap}")
    n_synth = len(synth_nct)

    # Extract everything once — bootstrap iterations just slice the cached arrays.
    synth_feats    = _extract_3d_features(synth_nct)
    all_real_feats = _extract_3d_features(real_nct)

    synth_lats    = None
    all_real_lats = None
    if vae_model is not None:
        synth_lats    = _vae_encode_latents(synth_nct, vae_model, mean_t, std_t)
        all_real_lats = _vae_encode_latents(real_nct,  vae_model, mean_t, std_t)

    results = {k: [] for k in ('fid', 'base_fid', 'mmd', 'base_mmd', 'prec', 'rec')}

    for _ in range(n_boots):
        perm = rng.permutation(n_real)
        if n_real >= 2 * n_synth:
            idx_a, idx_b = perm[:n_synth], perm[n_synth:2 * n_synth]
        else:
            idx_a = perm[:n_synth]
            idx_b = rng.permutation(n_real)[:n_synth]

        real_feats = all_real_feats[idx_a]
        base_feats = all_real_feats[idx_b]
        results['mmd'].append(_compute_mmd(real_feats, synth_feats))
        results['base_mmd'].append(_compute_mmd(real_feats, base_feats))

        if all_real_lats is not None:
            real_lats = all_real_lats[idx_a]
            base_lats = all_real_lats[idx_b]
            results['fid'].append(_compute_fid(real_lats, synth_lats))
            results['base_fid'].append(_compute_fid(real_lats, base_lats))
            p, r = _compute_precision_recall(real_lats, synth_lats, k=k_nn)
            results['prec'].append(p)
            results['rec'].append(r)

    return results, synth_lats, all_real_lats


def _fmt(vals):
    if not vals:
        return "N/A"
    return f"{np.mean(vals):.4f} ± {np.std(vals):.4f}"


def _pct_err(synth_vals, base_vals):
    if not synth_vals or not base_vals:
        return "—"
    b = np.mean(base_vals)
    if abs(b) < 1e-10:
        return "—"
    return f"{(np.mean(synth_vals) - b) / b * 100:+.1f}%"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_boots", type=int, default=10,
                        help="Number of bootstrap iterations per class (default 10)")
    parser.add_argument("--k_nn", type=int, default=3,
                        help="k for Precision/Recall k-NN (default 3)")
    parser.add_argument("--gen_path", type=str, default=None,
                        help="Path to generated .npy (overrides UPDRS_GEN_OUTPUT_PATH)")
    args = parser.parse_args()

    global GEN_OUTPUT_PATH
    if args.gen_path:
        os.environ["UPDRS_GEN_OUTPUT_PATH"] = args.gen_path
        GEN_OUTPUT_PATH = args.gen_path

    print("\n" + "=" * 64)
    print("  Bootstrapped Distributional Metrics")
    print(f"  n_boots={args.n_boots}  k_nn={args.k_nn}")
    print("=" * 64)

    # Load data
    real_data   = np.load(EVAL_DATA_PATH)
    real_labels = np.load(EVAL_LABELS_PATH)
    synth_data  = np.load(GEN_OUTPUT_PATH)
    synth_labels_path = GEN_OUTPUT_PATH.replace(".npy", "_labels.npy")
    synth_labels = np.load(synth_labels_path) if os.path.exists(synth_labels_path) \
                   else np.zeros(len(synth_data), dtype=np.int64)

    # Ensure (N, C, T)
    def _to_nct(arr):
        # Check axis-1 against the known channel count (works for both 6-ch and H3D-263)
        if arr.ndim == 3 and arr.shape[1] == N_CHANNELS:
            return arr  # already (N, C, T)
        return arr.transpose(0, 2, 1)

    real_nct  = _to_nct(real_data)
    synth_nct = _to_nct(synth_data)

    # Load VAE
    from config import DEVICE as _TORCH_DEVICE
    global _DEVICE
    vae_model = mean_t = std_t = None
    if os.path.exists(VAE_MODEL_PATH) and os.path.exists(NORM_PARAMS_PATH):
        _DEVICE   = _TORCH_DEVICE
        norm      = torch.load(NORM_PARAMS_PATH, map_location='cpu')
        mean_t    = norm['mean'].float()
        std_t     = norm['std'].float()
        _vae_state_dict = torch.load(VAE_MODEL_PATH, map_location=_DEVICE)
        vae_model = GaitVAE(N_CHANNELS, LATENT_CHANNELS, N_UPDRS_CLASSES,
                             use_prototypes="prototypes" in _vae_state_dict).to(_DEVICE)
        vae_model.load_state_dict(_vae_state_dict)
        vae_model.eval()
        print(f"VAE loaded on {_DEVICE} — FID computed in mean-pooled latent space ({LATENT_CHANNELS}-dim)")
    else:
        print("WARNING: VAE not found — FID/Precision/Recall will be skipped")

    rng = np.random.default_rng(42)
    cls_display = {0: "Normal", 1: "Mild", 2: "Moderate", 3: "Severe"}
    all_classes = sorted(np.union1d(np.unique(real_labels), np.unique(synth_labels)).tolist())

    lines = []
    all_real_lats_last, all_synth_lats_fixed = [], []

    for cls in all_classes:
        label     = cls_display.get(int(cls), f"UPDRS_{int(cls)}")
        r_mask    = real_labels  == cls
        s_mask    = synth_labels == cls
        real_cls  = real_nct[r_mask]
        synth_cls = synth_nct[s_mask]

        if len(synth_cls) == 0:
            print(f"\nSkipping UPDRS {cls} — no synthetic samples.")
            continue

        print(f"\nUPDRS {cls} ({label}): {len(synth_cls)} synth / {len(real_cls)} real  "
              f"— running {args.n_boots} bootstrap rounds...")

        res, synth_lats, all_real_lats_cls = _bootstrap_class(
            real_cls, synth_cls, vae_model, mean_t, std_t,
            args.n_boots, rng, k_nn=args.k_nn,
        )

        hdr = f"\n  UPDRS {cls} — {label}  ({len(synth_cls)} synth / {len(real_cls)} real)"
        print(hdr)
        print(f"  {'Metric':<22} {'Synth vs Real':>22}  {'Real vs Real (baseline)':>24}  {'% error':>8}")
        print("  " + "-" * 80)

        prec_lbl = f"Precision (k={args.k_nn})"
        rec_lbl  = f"Recall    (k={args.k_nn})"
        row_fid  = f"  {'FID (latent)':<22} {_fmt(res['fid']):>22}  {_fmt(res['base_fid']):>24}  {_pct_err(res['fid'], res['base_fid']):>8}"
        row_mmd  = f"  {'MMD² (feat)':<22} {_fmt(res['mmd']):>22}  {_fmt(res['base_mmd']):>24}  {_pct_err(res['mmd'], res['base_mmd']):>8}"
        row_prec = f"  {prec_lbl:<22} {_fmt(res['prec']):>22}  {'—':>24}  {'—':>8}"
        row_rec  = f"  {rec_lbl:<22} {_fmt(res['rec']):>22}  {'—':>24}  {'—':>8}"
        for row in (row_fid, row_mmd, row_prec, row_rec):
            print(row)
        lines += [hdr, row_fid, row_mmd, row_prec, row_rec]

        if synth_lats is not None and all_real_lats_cls is not None:
            perm = rng.permutation(len(all_real_lats_cls))
            last_real_lats = all_real_lats_cls[perm[:len(synth_lats)]]
            all_real_lats_last.append(last_real_lats)
            all_synth_lats_fixed.append(synth_lats)

            div_synth = _compute_diversity(synth_lats,     rng=rng)
            div_real  = _compute_diversity(last_real_lats, rng=rng)
            row_div = (f"  {'Diversity (L2 latent)':<22} "
                       f"synth={div_synth:.4f}   real={div_real:.4f}  "
                       f"(ratio={div_synth/div_real:.3f})")
            print(row_div)
            lines.append(row_div)

    # Overall — all classes pooled, full bootstrapped suite
    if vae_model is not None:
        n_synth_total = len(synth_nct)
        print(f"\nOVERALL (all classes pooled): {n_synth_total} synth / {len(real_nct)} real"
              f"  — running {args.n_boots} bootstrap rounds...")

        res_all, synth_lats_all, all_real_lats_all = _bootstrap_class(
            real_nct, synth_nct, vae_model, mean_t, std_t,
            args.n_boots, rng, k_nn=args.k_nn,
        )

        perm_all      = rng.permutation(len(all_real_lats_all))
        real_lats_all = all_real_lats_all[perm_all[:n_synth_total]]
        div_synth_all = _compute_diversity(synth_lats_all, rng=rng)
        div_real_all  = _compute_diversity(real_lats_all,  rng=rng)

        hdr2     = f"\n  OVERALL — all classes pooled  ({n_synth_total} synth / {len(real_nct)} real)"
        row_fid2 = f"  {'FID (latent)':<22} {_fmt(res_all['fid']):>22}  {_fmt(res_all['base_fid']):>24}  {_pct_err(res_all['fid'], res_all['base_fid']):>8}"
        row_mmd2 = f"  {'MMD² (feat)':<22} {_fmt(res_all['mmd']):>22}  {_fmt(res_all['base_mmd']):>24}  {_pct_err(res_all['mmd'], res_all['base_mmd']):>8}"
        row_pre2 = f"  {'Precision (k=' + str(args.k_nn) + ')':<22} {_fmt(res_all['prec']):>22}  {'—':>24}  {'—':>8}"
        row_rec2 = f"  {'Recall    (k=' + str(args.k_nn) + ')':<22} {_fmt(res_all['rec']):>22}  {'—':>24}  {'—':>8}"
        row_div2 = (f"  {'Diversity (L2 latent)':<22} "
                    f"synth={div_synth_all:.4f}   real={div_real_all:.4f}  "
                    f"(ratio={div_synth_all/div_real_all:.3f})")
        print(hdr2)
        print(f"  {'Metric':<22} {'Synth vs Real':>22}  {'Real vs Real (baseline)':>24}  {'% error':>8}")
        print("  " + "-" * 80)
        for row in (row_fid2, row_mmd2, row_pre2, row_rec2):
            print(row)
        print(row_div2)
        lines += [hdr2, row_fid2, row_mmd2, row_pre2, row_rec2, row_div2]

    # Save report
    os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(EVAL_OUTPUT_DIR, "distributional_metrics.txt")
    with open(out_path, "w") as f:
        f.write(f"Bootstrapped Distributional Metrics (n_boots={args.n_boots}, k_nn={args.k_nn})\n")
        f.write("\n".join(lines) + "\n")
    print(f"\nReport saved to {out_path}")


if __name__ == "__main__":
    main()
