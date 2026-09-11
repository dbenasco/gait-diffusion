"""
Classifier evaluation suite — TRTR, TSTR, TRTS.

All three run on one of three feature spaces (--features): VAE latent (mu,
default), raw flattened angle windows, or clinical (leg ROM + arm-swing ROM +
trunk inclination, 8-dim); and one of four classifiers (--classifier): k-NN
(default), Random Forest, Logistic Regression, or an ensemble (mean of RF +
LogReg probabilities) — the multi-classifier TSTR/TRTS approach standard in
the synthetic-data literature (Esteban et al. 2017 RCGAN, CTGAN), reported
alongside k-NN rather than replacing it.
⚠️  TRTS was circular in its original formulation (training on data the VAE had
    seen, testing on synth generated through that same VAE). As of 2026-07-16 it
    trains on the eval split (unseen by the VAE) to break that circularity.
    TSTR tests on eval-split real data and always has been clean.

  TRTR — Train on Real,      Test on Real      → ceiling (real class separability)
  TSTR — Train on Synthetic, Test on Real      → synthetic utility as training signal
  TRTS — Train on Real,      Test on Synthetic → conditioning correctness (circular)

Usage:
    python -m evaluation.generalization
    python -m evaluation.generalization --skip tstr
    python -m evaluation.generalization --k 5 --n_folds 10
    python -m evaluation.generalization --classes 0 2   # binary
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    PROCESSED_DATA_PATH, PROCESSED_LABELS_PATH,
    TRAIN_DATA_PATH, TRAIN_LABELS_PATH,
    EVAL_DATA_PATH, EVAL_LABELS_PATH,
    GEN_OUTPUT_PATH, UPDRS_CLASSES,
    VAE_MODEL_PATH, STATS_PATH,
    N_CHANNELS, LATENT_CHANNELS, LATENT_TIME,
)
from h3d_bridge import (
    h3d_to_angles, h3d_to_positions22, arm_swing_range_t, trunk_inclination_t,
)

CLS_NAMES = {0: "Normal", 1: "Mild", 2: "Moderate", 3: "Severe"}


# ── Shared helpers ────────────────────────────────────────────────────────────

_VAE_ENCODER  = None   # GaitVAE instance, loaded in main()
_NORM_MEAN    = None   # train-set normalization mean (C, 1) tensor
_NORM_STD     = None   # train-set normalization std  (C, 1) tensor
_ENCODE_BATCH = 512    # batch size for VAE encoding to avoid OOM
_FEATURE_MODE = "latent"  # "latent" (VAE mu), "raw" (normalized angle windows), or "clinical" (ROM/arm-swing/trunk)
_DEVICE       = None   # set in main() after VAE load; used by knn_predict for GPU distance
_CLASSIFIER   = "knn"  # "knn" (default), "rf", "logreg", or "ensemble" (mean of rf+logreg probabilities)
_CLINICAL_MEAN = None  # (1, 8) real-data mean, for z-scoring clinical features — set in main()
_CLINICAL_STD  = None  # (1, 8) real-data std


def _clinical_raw(data_nct):
    """(N, C, T) H3D window → (N, 8) unnormalized clinical vector:
    leg ROM (6, degrees) + arm-swing ROM (1, metres) + trunk inclination (1, degrees)."""
    feat = torch.from_numpy(data_nct).float()
    ang = h3d_to_angles(feat)                                     # (N, 6, T) sagittal degrees
    leg_rom = ang.max(dim=-1).values - ang.min(dim=-1).values     # (N, 6)
    pos = h3d_to_positions22(feat)                                # (N, T, 22, 3)
    arm_swing  = arm_swing_range_t(pos).unsqueeze(1)              # (N, 1) metres
    trunk_incl = trunk_inclination_t(pos).unsqueeze(1)            # (N, 1) degrees
    return torch.cat([leg_rom, arm_swing, trunk_incl], dim=1)     # (N, 8)


def extract_features(data_nct):
    """(N, C, T) → feature vector. "latent": VAE encoder mu (LATENT_CHANNELS*LATENT_TIME-dim).
    "raw": normalized angle window, flattened (C*T-dim).
    "clinical": leg ROM (6) + arm-swing ROM (1) + trunk inclination (1) = 8-dim,
    z-scored against real-data mean/std (_CLINICAL_MEAN/_CLINICAL_STD, set in main())
    so no single channel's raw unit scale (e.g. ROM in degrees vs. arm-swing in
    metres) dominates a Euclidean-distance classifier like k-NN."""
    if _FEATURE_MODE == "clinical":
        raw = _clinical_raw(data_nct)
        if _CLINICAL_MEAN is not None:
            raw = (raw - _CLINICAL_MEAN) / _CLINICAL_STD
        return raw.numpy()
    if _FEATURE_MODE == "raw":
        mean = _NORM_MEAN.numpy().reshape(1, -1, 1)
        std  = _NORM_STD.numpy().reshape(1, -1, 1)
        x = np.clip((data_nct - mean) / std, -4, 4)
        return x.reshape(len(data_nct), -1)
    device = next(_VAE_ENCODER.parameters()).device
    mean   = _NORM_MEAN.to(device)
    std    = _NORM_STD.to(device)
    all_z  = []
    for i in range(0, len(data_nct), _ENCODE_BATCH):
        x = torch.from_numpy(data_nct[i:i + _ENCODE_BATCH]).float().to(device)
        x = torch.clamp((x - mean) / std, -4, 4)
        with torch.no_grad():
            _, mu, _ = _VAE_ENCODER.encode(x)
        all_z.append(mu.cpu().numpy())
    return np.concatenate(all_z, axis=0).reshape(len(data_nct), -1)


def knn_predict(train_feats, train_labels, test_feats, k, train_weights=None):
    train_labels      = np.asarray(train_labels).ravel().astype(int)
    train_weights_arr = np.asarray(train_weights) if train_weights is not None else None

    device = _DEVICE or torch.device('cpu')
    tr = torch.from_numpy(train_feats).float().to(device)   # (N_train, D)
    te = torch.from_numpy(test_feats).float().to(device)    # (N_test,  D)

    # squared L2 via dot-product trick — avoids materialising (N_test, N_train, D)
    # peak GPU mem: N_test × N_train × 4 bytes ≈ 3476 × 9504 × 4 ≈ 132 MB — fine
    tr_sq    = (tr ** 2).sum(dim=1)
    te_sq    = (te ** 2).sum(dim=1)
    dists_sq = te_sq[:, None] + tr_sq[None, :] - 2.0 * (te @ tr.T)
    dists_sq.clamp_(min=0.0)

    nn_idx    = torch.topk(dists_sq, k, dim=1, largest=False).indices.cpu().numpy()
    nn_labels = train_labels[nn_idx]

    if train_weights_arr is not None:
        nn_w = train_weights_arr[nn_idx]
        return np.array([
            np.bincount(row.astype(int), weights=wrow, minlength=UPDRS_CLASSES).argmax()
            for row, wrow in zip(nn_labels, nn_w)
        ])
    return np.array([
        np.bincount(row.astype(int), minlength=UPDRS_CLASSES).argmax()
        for row in nn_labels
    ])


def _make_sklearn_clf(name):
    if name == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    raise ValueError(f"Unknown sklearn classifier: {name}")


def _sklearn_proba(name, train_feats, train_labels, test_feats, train_weights):
    """Fit + predict_proba, remapped to a full (N, UPDRS_CLASSES) array in case
    a class is missing from this particular training slice."""
    clf = _make_sklearn_clf(name)
    clf.fit(train_feats, train_labels, sample_weight=train_weights)
    proba = np.zeros((len(test_feats), UPDRS_CLASSES))
    proba[:, clf.classes_] = clf.predict_proba(test_feats)
    return proba


def _torch_logreg_proba(train_feats, train_labels, test_feats, train_weights=None,
                         max_iter=200, l2=1.0):
    """Multinomial logistic regression (linear + softmax, cross-entropy loss),
    fit via full-batch LBFGS on GPU. Same model sklearn's LogisticRegression
    fits — just run on _DEVICE instead of CPU, which matters at 1536-dim
    (VAE latent): sklearn's lbfgs solver was the slow path there, not the
    classifier itself. Features are standardized (train-fold mean/std) before
    fitting, both for faster convergence and to avoid one raw-scale feature
    (e.g. leg ROM in degrees vs. arm-swing in metres) dominating the others."""
    device = _DEVICE or torch.device('cpu')
    X  = torch.from_numpy(np.asarray(train_feats)).float().to(device)
    y  = torch.from_numpy(np.asarray(train_labels).ravel().astype(int)).long().to(device)
    Xt = torch.from_numpy(np.asarray(test_feats)).float().to(device)
    w  = (torch.from_numpy(np.asarray(train_weights)).float().to(device)
          if train_weights is not None else torch.ones(len(y), device=device))

    mean = X.mean(dim=0, keepdim=True)
    std  = X.std(dim=0, keepdim=True).clamp_min(1e-6)
    Xn, Xtn = (X - mean) / std, (Xt - mean) / std

    torch.manual_seed(42)
    linear = torch.nn.Linear(Xn.shape[1], UPDRS_CLASSES).to(device)
    opt = torch.optim.LBFGS(linear.parameters(), max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        logits = linear(Xn)
        loss = torch.nn.functional.cross_entropy(logits, y, reduction="none")
        loss = (loss * w).mean() + (l2 / len(y)) * (linear.weight ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)

    with torch.no_grad():
        return torch.softmax(linear(Xtn), dim=1).cpu().numpy()


def classify_predict(train_feats, train_labels, test_feats, k, train_weights=None):
    """Dispatches to whichever classifier _CLASSIFIER selects. k is only used by knn."""
    train_labels = np.asarray(train_labels).ravel().astype(int)
    if _CLASSIFIER == "knn":
        return knn_predict(train_feats, train_labels, test_feats, k, train_weights=train_weights)
    if _CLASSIFIER == "rf":
        clf = _make_sklearn_clf("rf")
        clf.fit(train_feats, train_labels, sample_weight=train_weights)
        return clf.predict(test_feats)
    if _CLASSIFIER == "logreg":
        proba = _torch_logreg_proba(train_feats, train_labels, test_feats, train_weights)
        return np.argmax(proba, axis=1)
    if _CLASSIFIER == "ensemble":
        proba_rf = _sklearn_proba("rf", train_feats, train_labels, test_feats, train_weights)
        proba_lr = _torch_logreg_proba(train_feats, train_labels, test_feats, train_weights)
        return np.argmax((proba_rf + proba_lr) / 2.0, axis=1)
    raise ValueError(f"Unknown classifier: {_CLASSIFIER}")


def compute_class_weights(labels):
    """Inverse-frequency per-sample weights: w_c = N_total / (n_classes * N_c). E[w]=1."""
    n = len(labels)
    counts = np.array([float((labels == c).sum()) for c in range(UPDRS_CLASSES)])
    return (n / (UPDRS_CLASSES * counts))[labels]


def balance(data, labels, n_per_class, rng):
    idx = np.concatenate([
        rng.choice(np.where(labels == c)[0], size=n_per_class, replace=False)
        for c in range(UPDRS_CLASSES)
    ])
    return data[idx], labels[idx]


def load_real(split="all"):
    def _load(path, lpath):
        data   = np.load(path)
        labels = np.load(lpath).astype(int)
        if data.ndim == 3 and data.shape[1] != N_CHANNELS:
            data = data.transpose(0, 2, 1)
        mask = labels < UPDRS_CLASSES
        return data[mask].astype(np.float32), labels[mask]

    if split == "all":
        if os.path.exists(PROCESSED_DATA_PATH):
            return _load(PROCESSED_DATA_PATH, PROCESSED_LABELS_PATH)
        d0, l0 = _load(TRAIN_DATA_PATH, TRAIN_LABELS_PATH)
        d1, l1 = _load(EVAL_DATA_PATH,  EVAL_LABELS_PATH)
        return np.concatenate([d0, d1]), np.concatenate([l0, l1])
    if split == "train":
        return _load(TRAIN_DATA_PATH, TRAIN_LABELS_PATH)
    return _load(EVAL_DATA_PATH, EVAL_LABELS_PATH)


def load_synth(n_per_class=None, seed=42):
    data   = np.load(GEN_OUTPUT_PATH)
    labels = np.load(GEN_OUTPUT_PATH.replace(".npy", "_labels.npy")).ravel().astype(int)
    if data.ndim == 3 and data.shape[1] != N_CHANNELS:
        data = data.transpose(0, 2, 1)
    mask   = labels < UPDRS_CLASSES
    data, labels = data[mask], labels[mask]
    if n_per_class is not None:
        rng = np.random.default_rng(seed)
        data, labels = balance(data, labels, n_per_class, rng)
    return data.astype(np.float32), labels


def filter_and_remap(data, labels, class_list):
    """Keep only samples in class_list and remap labels to 0-indexed."""
    mask = np.isin(labels, class_list)
    data, labels = data[mask], labels[mask]
    remap = {orig: new for new, orig in enumerate(class_list)}
    return data, np.array([remap[l] for l in labels], dtype=int)


def confusion_matrix(preds, targets):
    cm = np.zeros((UPDRS_CLASSES, UPDRS_CLASSES), dtype=int)
    for t, p in zip(targets, preds):
        cm[t, p] += 1
    return cm


def print_results(tag, preds, labels, min_n_train, note=""):
    cm  = confusion_matrix(preds, labels)
    acc = (preds == labels).mean() * 100

    true_pred = "True \\ Pred"
    header = f"  {true_pred:<14}" + "".join(
        f"  {('U'+str(c)+'('+CLS_NAMES[c][:3]+')'):>12}" for c in range(UPDRS_CLASSES))
    sep = "  " + "─" * (len(header) - 2)

    print(f"\n{'='*64}")
    print(f"{tag}" + (f"  —  {note}" if note else ""))
    print(f"{'='*64}")
    if min_n_train is not None:
        print(f"  Training set  : {min_n_train}/class (balanced)")
    else:
        print(f"  Training set  : all data (class-weighted voting)")
    counts = "/".join(str(int((labels == c).sum())) for c in range(UPDRS_CLASSES))
    clsids = "/".join(f"U{c}" for c in range(UPDRS_CLASSES))
    print(f"  Test set      : {counts}  ({clsids})")
    print(f"\n{header}\n{sep}")
    for i in range(UPDRS_CLASSES):
        total = cm[i].sum()
        cells = "".join(f"  {cm[i,j]:>6} ({cm[i,j]/max(total,1)*100:4.1f}%)" for j in range(UPDRS_CLASSES))
        print(f"  UPDRS {i} ({CLS_NAMES[i][:3]}): {cells}")
    print()
    for c in range(UPDRS_CLASSES):
        acc_c = cm[c, c] / max(cm[c].sum(), 1) * 100
        flag  = "✅" if acc_c >= 60 else ("⚠️ " if acc_c >= 40 else "❌")
        print(f"  UPDRS {c} ({CLS_NAMES[c]:8s}): {cm[c,c]:>5}/{cm[c].sum():<5}  {acc_c:.1f}%  {flag}")
    print(f"\n  Overall: {acc:.1f}%  (chance={100/UPDRS_CLASSES:.1f}%)")
    return acc, {c: cm[c,c] / max(cm[c].sum(), 1) * 100 for c in range(UPDRS_CLASSES)}


# ── TRTR ──────────────────────────────────────────────────────────────────────

def run_trtr(data, labels, k, n_folds, seed):
    rng     = np.random.default_rng(seed)
    weights = compute_class_weights(labels)
    n       = len(labels)
    preds   = np.zeros(n, dtype=int)

    print(f"  Encoding {n} samples once for {n_folds}-fold CV ...", end=" ", flush=True)
    all_feats = extract_features(data)   # encode full dataset once; slice per fold
    print("done.")

    class_idx = [rng.permutation(np.where(labels == c)[0]) for c in range(UPDRS_CLASSES)]
    for fold in range(n_folds):
        te_idx = np.concatenate([idx[fold::n_folds] for idx in class_idx])
        tr_idx = np.setdiff1d(np.arange(n), te_idx)
        preds[te_idx] = classify_predict(
            all_feats[tr_idx], labels[tr_idx],
            all_feats[te_idx], k,
            train_weights=weights[tr_idx],
        )

    return preds, labels, None


# ── TSTR ──────────────────────────────────────────────────────────────────────

def run_tstr(synth_data, synth_labels, real_test_data, real_test_labels, k, seed):
    tr_weights = compute_class_weights(synth_labels)
    preds = classify_predict(
        extract_features(synth_data), synth_labels,
        extract_features(real_test_data), k,
        train_weights=tr_weights,
    )
    return preds, real_test_labels, None


# ── TRTS ──────────────────────────────────────────────────────────────────────

def run_trts(real_data, real_labels, synth_data, synth_labels, k, seed):
    tr_weights = compute_class_weights(real_labels)
    preds = classify_predict(
        extract_features(real_data), real_labels,
        extract_features(synth_data), k,
        train_weights=tr_weights,
    )
    return preds, synth_labels, None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global UPDRS_CLASSES, CLS_NAMES
    parser = argparse.ArgumentParser()
    parser.add_argument("--k",        type=int,  default=5)
    parser.add_argument("--n_folds",  type=int,  default=10,
                        help="Folds for TRTR cross-validation (default 10)")
    parser.add_argument("--n_synth",  type=int,  default=None,
                        help="Synthetic samples per class (default: all available)")
    parser.add_argument("--seed",     type=int,  default=42)
    parser.add_argument("--split",    choices=["all", "train", "eval"], default="all",
                        help="Real data split for TRTR (default: all)")
    parser.add_argument("--trts_split", choices=["all", "train", "eval"], default="eval",
                        help="Real data split for TRTS training (default: eval — VAE-unseen, "
                             "avoids circularity). Use 'all' when eval-split sample counts "
                             "are too small for a reliable classifier (e.g. UPDRS 3).")
    parser.add_argument("--skip",     nargs="*", default=[],
                        choices=["trtr", "tstr", "trts"],
                        help="Skip one or more evaluations")
    parser.add_argument("--classes",  type=int,  nargs="+", default=None,
                        help="UPDRS classes to include (default: all). E.g. --classes 0 2")
    parser.add_argument("--features", choices=["latent", "raw", "clinical", "compare"], default="latent",
                        help="latent: VAE mu (default). raw: normalized flattened angle window. "
                             "clinical: leg ROM + arm-swing ROM + trunk inclination (8-dim). "
                             "compare: run TRTR under latent vs raw and print a side-by-side table.")
    parser.add_argument("--classifier", choices=["knn", "rf", "logreg", "ensemble"], default="knn",
                        help="knn: k-NN, k=--k (default). rf: Random Forest. logreg: Logistic Regression. "
                             "ensemble: mean of RF + LogReg predicted probabilities, then argmax — "
                             "the standard multi-classifier TSTR/TRTS approach in the synthetic-data "
                             "literature (Esteban et al. 2017, CTGAN), reported alongside k-NN rather "
                             "than replacing it.")
    parser.add_argument("--gen_path", type=str, default=None,
                        help="Path to generated .npy (overrides UPDRS_GEN_OUTPUT_PATH)")
    parser.add_argument("--vae_path", type=str, default=None,
                        help="Path to VAE checkpoint (overrides UPDRS_VAE_MODEL_PATH)")
    args = parser.parse_args()

    global _VAE_ENCODER, _NORM_MEAN, _NORM_STD, _FEATURE_MODE, _DEVICE, _CLASSIFIER
    global _CLINICAL_MEAN, _CLINICAL_STD
    global GEN_OUTPUT_PATH, VAE_MODEL_PATH
    _CLASSIFIER = args.classifier
    if args.gen_path:
        GEN_OUTPUT_PATH = args.gen_path
    if args.vae_path:
        VAE_MODEL_PATH = args.vae_path

    from training.vae_updrs import GaitVAE
    from config import DEVICE
    _vae_state_dict = torch.load(VAE_MODEL_PATH, map_location=DEVICE)
    _vae = GaitVAE(
        in_channels=N_CHANNELS,
        latent_channels=LATENT_CHANNELS,
        updrs_classes=UPDRS_CLASSES,
        use_prototypes="prototypes" in _vae_state_dict,
    ).to(DEVICE)
    _vae.load_state_dict(_vae_state_dict)
    _vae.eval()
    for p in _vae.parameters():
        p.requires_grad_(False)
    _VAE_ENCODER = _vae
    _DEVICE      = next(_vae.parameters()).device
    _norm = torch.load(STATS_PATH, map_location="cpu")
    _NORM_MEAN = _norm.get("mean", torch.zeros(N_CHANNELS, 1))
    _NORM_STD  = _norm["std"]

    if args.features != "compare":
        _FEATURE_MODE = args.features
    if _FEATURE_MODE == "latent":
        feat_desc = f"VAE latent mu ({LATENT_CHANNELS}×{LATENT_TIME}={LATENT_CHANNELS*LATENT_TIME}-dim)"
    elif _FEATURE_MODE == "clinical":
        feat_desc = "clinical (leg ROM×6 + arm-swing ROM + trunk inclination = 8-dim)"
    else:
        feat_desc = f"raw normalized angles ({N_CHANNELS}×T flattened)"
    clf_desc = {
        "knn":      f"k-NN (k={args.k})",
        "rf":       "Random Forest",
        "logreg":   "Logistic Regression",
        "ensemble": "Ensemble (mean of RF + LogReg probabilities)",
    }[args.classifier]
    print(f"Classifier: {clf_desc}  |  Features: {feat_desc}  ⚠️ TRTS is circular")

    skip = set(args.skip or [])

    # ── Load ──────────────────────────────────────────────────────────────────
    real_data, real_labels = load_real(args.split)
    real_eval_data, real_eval_labels = load_real("eval")
    real_trts_data, real_trts_labels = load_real(args.trts_split)

    synth_data = synth_labels = None
    if "tstr" not in skip or "trts" not in skip:
        synth_data, synth_labels = load_synth(n_per_class=args.n_synth, seed=args.seed)

    # ── Class filtering ───────────────────────────────────────────────────────
    active = sorted(set(args.classes)) if args.classes else list(range(UPDRS_CLASSES))
    if active != list(range(UPDRS_CLASSES)):
        real_data,      real_labels      = filter_and_remap(real_data,      real_labels,      active)
        real_eval_data, real_eval_labels = filter_and_remap(real_eval_data, real_eval_labels, active)
        real_trts_data, real_trts_labels = filter_and_remap(real_trts_data, real_trts_labels, active)
        if synth_data is not None:
            synth_data, synth_labels = filter_and_remap(synth_data, synth_labels, active)
        _base_names = dict(CLS_NAMES)
        CLS_NAMES     = {new: _base_names[orig] for new, orig in enumerate(active)}
        UPDRS_CLASSES = len(active)
        print(f"Class subset: {active}  →  remapped to 0..{UPDRS_CLASSES-1}  "
              f"({', '.join(f'{orig}={CLS_NAMES[new]}' for new, orig in enumerate(active))})\n")

    if _FEATURE_MODE == "clinical":
        with torch.no_grad():
            ref = _clinical_raw(real_data)
        _CLINICAL_MEAN = ref.mean(dim=0, keepdim=True)
        _CLINICAL_STD  = ref.std(dim=0, keepdim=True).clamp_min(1e-6)

    real_counts  = {c: int((real_labels  == c).sum()) for c in range(UPDRS_CLASSES)}
    synth_counts = ({c: int((synth_labels == c).sum()) for c in range(UPDRS_CLASSES)}
                   if synth_labels is not None else {})

    print(f"Real ({args.split})  : {real_counts}  (total {len(real_labels)})")
    if synth_counts:
        print(f"Synthetic     : {synth_counts}  (total {sum(synth_counts.values())})")
    print(f"TRTR uses {args.n_folds}-fold CV\n")

    # ── TRTR feature comparison (latent vs raw) ─────────────────────────────────
    if args.features == "compare":
        _FEATURE_MODE = "latent"
        preds_l, labels_b, min_n = run_trtr(real_data, real_labels,
                                             k=args.k, n_folds=args.n_folds, seed=args.seed)
        acc_l, per_l = print_results(
            "TRTR — VAE latent", preds_l, labels_b, min_n, note=f"{args.n_folds}-fold CV")

        _FEATURE_MODE = "raw"
        preds_r, labels_b, min_n = run_trtr(real_data, real_labels,
                                             k=args.k, n_folds=args.n_folds, seed=args.seed)
        acc_r, per_r = print_results(
            "TRTR — Raw angles", preds_r, labels_b, min_n, note=f"{args.n_folds}-fold CV")

        print(f"\n{'='*56}")
        print("COMPARISON — TRTR: VAE latent vs raw angles")
        print(f"{'='*56}")
        print(f"  {'':14}  {'Latent':>10}  {'Raw':>10}  {'Δ (lat-raw)':>12}")
        for c in range(UPDRS_CLASSES):
            print(f"  UPDRS {c} ({CLS_NAMES[c][:3]})  {per_l[c]:>9.1f}%  {per_r[c]:>9.1f}%  {per_l[c]-per_r[c]:>+11.1f}pp")
        print(f"  {'Overall':14}  {acc_l:>9.1f}%  {acc_r:>9.1f}%  {acc_l-acc_r:>+11.1f}pp")
        print(f"\n  Chance: {100/UPDRS_CLASSES:.1f}%")
        print("  Positive Δ → VAE latent improves class separability over raw angles.")
        return

    results = {}

    # ── TRTR ──────────────────────────────────────────────────────────────────
    if "trtr" not in skip:
        preds, labels_b, min_n = run_trtr(real_data, real_labels,
                                           k=args.k, n_folds=args.n_folds, seed=args.seed)
        acc, per_cls = print_results(
            "TRTR — Train on Real, Test on Real",
            preds, labels_b, min_n,
            note=f"{args.n_folds}-fold CV → ceiling",
        )
        results["trtr"] = (acc, per_cls)

    # ── TSTR ──────────────────────────────────────────────────────────────────
    if "tstr" not in skip:
        preds, labels_b, min_n = run_tstr(
            synth_data, synth_labels,
            real_eval_data, real_eval_labels,
            k=args.k, seed=args.seed,
        )
        acc, per_cls = print_results(
            "TSTR — Train on Synthetic, Test on Real",
            preds, labels_b, min_n,
            note="synthetic utility as training signal",
        )
        results["tstr"] = (acc, per_cls)

    # ── TRTS ──────────────────────────────────────────────────────────────────
    if "trts" not in skip:
        trts_counts = {c: int((real_trts_labels == c).sum()) for c in range(UPDRS_CLASSES)}
        counts_str = ", ".join(f"U{c}={trts_counts[c]}" for c in range(UPDRS_CLASSES))
        print(f"\n  TRTS training set ({args.trts_split} split): {counts_str}")
        preds, labels_b, min_n = run_trts(
            real_trts_data, real_trts_labels,
            synth_data, synth_labels,
            k=args.k, seed=args.seed,
        )
        acc, per_cls = print_results(
            "TRTS — Train on Real, Test on Synthetic",
            preds, labels_b, min_n,
            note=f"train split: {args.trts_split} ({counts_str})",
        )
        results["trts"] = (acc, per_cls)

    # ── Summary ───────────────────────────────────────────────────────────────
    if len(results) > 1:
        print(f"\n{'='*64}")
        print("SUMMARY")
        print(f"{'='*64}")
        col = 10
        tags = [t.upper() for t in ["trtr", "tstr", "trts"] if t in results]
        hdr  = f"  {'':12}" + "".join(f"  {t:>{col}}" for t in tags)
        print(hdr)
        print("  " + "─" * (len(hdr) - 2))
        for c in range(UPDRS_CLASSES):
            row = f"  UPDRS {c} ({CLS_NAMES[c][:3]})"
            for t in ["trtr", "tstr", "trts"]:
                if t in results:
                    row += f"  {results[t][1][c]:>{col}.1f}%"
            print(row)
        print("  " + "─" * (len(hdr) - 2))
        overall_row = f"  {'Overall':<12}"
        for t in ["trtr", "tstr", "trts"]:
            if t in results:
                overall_row += f"  {results[t][0]:>{col}.1f}%"
        print(overall_row)
        print(f"\n  Chance: {100/UPDRS_CLASSES:.1f}%")
        if "trtr" in results and "trts" in results:
            gap = results["trtr"][0] - results["trts"][0]
            print(f"  TRTR–TRTS gap: {gap:.1f}pp  "
                  f"({'model adds confusion beyond data overlap' if gap > 10 else 'model reproduces real overlap faithfully'})")


if __name__ == "__main__":
    main()
