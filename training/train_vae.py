"""
Stage 1: Train the GaitVAE with UPDRS surrogate.

Output:
  models/vae.pth   — full VAE (encoder + decoder + surrogate)
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.vae_updrs import GaitVAE, vae_loss
from config import (
    TRAIN_DATA_PATH, TRAIN_LABELS_PATH,
    EVAL_DATA_PATH, EVAL_LABELS_PATH,
    NORM_PARAMS_PATH, VAE_MODEL_PATH,
    N_CHANNELS, LATENT_CHANNELS, UPDRS_CLASSES,
    VAE_BETA, VAE_LAMBDA, VAE_EPOCHS, VAE_LR,
    PROTOTYPE_PROXIMITY_WEIGHT, PROTOTYPE_DISTANCE_WEIGHT, PROTOTYPE_MARGIN,
    BATCH_SIZE, DEVICE,
)


def _load_channels(path):
    return np.load(path)


def make_loader(data_path, labels_path, norm_mean, norm_std, shuffle):
    data   = torch.from_numpy(_load_channels(data_path)).float()
    labels = torch.from_numpy(np.load(labels_path)).long()
    data   = torch.clamp((data - norm_mean) / norm_std, -4, 4)
    return DataLoader(TensorDataset(data, labels), batch_size=BATCH_SIZE, shuffle=shuffle)


def train():
    print("=" * 60)
    print("GaitVAE + UPDRS Surrogate — Stage 1 Training")
    print(f"  SEQ_LEN        : from config  (must be 180)")
    print(f"  Latent channels: {LATENT_CHANNELS}  (compression r=4 in time)")
    print(f"  β (KL weight)  : {VAE_BETA}")
    print(f"  λ (surrogate)  : {VAE_LAMBDA}")
    print(f"  γ_proto        : {PROTOTYPE_PROXIMITY_WEIGHT}  (pooled KL toward N(prototype, I) — noisy, not discrete)")
    print(f"  γ_dist         : {PROTOTYPE_DISTANCE_WEIGHT}")
    print(f"  margin         : {PROTOTYPE_MARGIN}")
    print(f"  Epochs         : {VAE_EPOCHS}")
    print("=" * 60)

    if not os.path.exists(TRAIN_DATA_PATH):
        print(f"ERROR: {TRAIN_DATA_PATH} not found. Run preprocessing.preprocess_carepd_h3d first.")
        return

    train_raw = torch.from_numpy(_load_channels(TRAIN_DATA_PATH)).float()
    norm_mean = train_raw.mean(dim=(0, 2)).view(1, -1, 1)
    norm_std  = train_raw.std(dim=(0, 2)).view(1, -1, 1)
    norm_std[norm_std < 1e-6] = 1.0
    os.makedirs(os.path.dirname(NORM_PARAMS_PATH), exist_ok=True)
    torch.save({'mean': norm_mean.squeeze(-1).squeeze(0).view(-1, 1),
                'std':  norm_std.squeeze(-1).squeeze(0).view(-1, 1)}, NORM_PARAMS_PATH)

    train_loader = make_loader(TRAIN_DATA_PATH, TRAIN_LABELS_PATH, norm_mean, norm_std, shuffle=True)
    eval_loader  = make_loader(EVAL_DATA_PATH,  EVAL_LABELS_PATH,  norm_mean, norm_std, shuffle=False)

    train_labels_all = torch.from_numpy(np.load(TRAIN_LABELS_PATH)).long()
    eval_labels_all  = torch.from_numpy(np.load(EVAL_LABELS_PATH)).long()
    print(f"Train windows: {len(train_raw)}  |  Eval windows: {len(np.load(EVAL_DATA_PATH))}")
    for cls in range(UPDRS_CLASSES):
        nt = int((train_labels_all == cls).sum())
        ne = int((eval_labels_all  == cls).sum())
        print(f"  UPDRS {cls}: train={nt}, eval={ne}")

    train_counts = torch.bincount(train_labels_all, minlength=UPDRS_CLASSES).float()
    class_weights = (len(train_labels_all) / (UPDRS_CLASSES * train_counts)).to(DEVICE)
    print(f"  Surrogate class weights: {class_weights.tolist()}")

    model = GaitVAE(
        in_channels=N_CHANNELS,
        latent_channels=LATENT_CHANNELS,
        updrs_classes=UPDRS_CLASSES,
        use_prototypes=True,
    ).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel on {DEVICE}. Params: {total_params / 1e6:.3f}M")

    optimizer = optim.Adam(model.parameters(), lr=VAE_LR)

    _p = Path(VAE_MODEL_PATH)
    best_val_path = str(_p.with_name(_p.stem + '_best_val' + _p.suffix))

    best_eval_loss = float('inf')
    best_epoch = 0

    for epoch in range(VAE_EPOCHS):
        model.train()
        train_losses = {'total': [], 'recon': [], 'kl': [], 'surr': [], 'proto': [], 'dist': []}

        for x, updrs_labels in train_loader:
            x            = x.to(DEVICE)
            updrs_labels = updrs_labels.to(DEVICE)

            recon, mu, logvar, updrs_logits = model(x)
            total, recon_l, kl_l, surr_l, proto_l, dist_l = vae_loss(
                recon, x, mu, logvar, updrs_logits, updrs_labels,
                beta=VAE_BETA, lam=VAE_LAMBDA,
                prototypes=model.prototypes, margin=PROTOTYPE_MARGIN,
                gamma_proto=PROTOTYPE_PROXIMITY_WEIGHT, gamma_dist=PROTOTYPE_DISTANCE_WEIGHT,
                class_weights=class_weights,
            )

            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses['total'].append(total.item())
            train_losses['recon'].append(recon_l.item())
            train_losses['kl'].append(kl_l.item())
            train_losses['surr'].append(surr_l.item())
            train_losses['proto'].append(proto_l.item())
            train_losses['dist'].append(dist_l.item())

        if (epoch + 1) % 10 == 0:
            model.eval()
            eval_losses = {'total': [], 'recon': [], 'kl': [], 'surr': [], 'proto': [], 'dist': []}
            correct, total_samples = 0, 0

            with torch.no_grad():
                for x_v, updrs_v in eval_loader:
                    x_v      = x_v.to(DEVICE)
                    updrs_v  = updrs_v.to(DEVICE)

                    recon_v, mu_v, logvar_v, logits_v = model(x_v)
                    tot_v, rec_v, kl_v, sur_v, proto_v, dist_v = vae_loss(
                        recon_v, x_v, mu_v, logvar_v, logits_v, updrs_v,
                        beta=VAE_BETA, lam=VAE_LAMBDA,
                        prototypes=model.prototypes, margin=PROTOTYPE_MARGIN,
                        gamma_proto=PROTOTYPE_PROXIMITY_WEIGHT, gamma_dist=PROTOTYPE_DISTANCE_WEIGHT,
                        class_weights=class_weights,
                    )
                    eval_losses['total'].append(tot_v.item())
                    eval_losses['recon'].append(rec_v.item())
                    eval_losses['kl'].append(kl_v.item())
                    eval_losses['surr'].append(sur_v.item())
                    eval_losses['proto'].append(proto_v.item())
                    eval_losses['dist'].append(dist_v.item())

                    preds   = logits_v.argmax(dim=1)
                    correct += int((preds == updrs_v).sum())
                    total_samples += len(updrs_v)

            surr_acc = 100 * correct / total_samples
            tr = {k: np.mean(v) for k, v in train_losses.items()}
            ev = {k: np.mean(v) for k, v in eval_losses.items()}

            print(
                f"Epoch {epoch+1:4d}/{VAE_EPOCHS} | "
                f"Train total={tr['total']:.4f} recon={tr['recon']:.4f} kl={tr['kl']:.4f} surr={tr['surr']:.4f} "
                f"proto={tr['proto']:.4f} dist={tr['dist']:.4f} | "
                f"Eval  total={ev['total']:.4f} recon={ev['recon']:.4f} surr_acc={surr_acc:.1f}%"
            )

            if ev['total'] < best_eval_loss:
                best_eval_loss = ev['total']
                best_epoch = epoch + 1
                os.makedirs(os.path.dirname(best_val_path), exist_ok=True)
                torch.save(model.state_dict(), best_val_path)

        elif (epoch + 1) % 100 == 0:
            model.eval()
            per_class_correct  = {c: 0 for c in range(UPDRS_CLASSES)}
            per_class_total    = {c: 0 for c in range(UPDRS_CLASSES)}
            with torch.no_grad():
                for x_v, updrs_v in eval_loader:
                    x_v     = x_v.to(DEVICE)
                    updrs_v = updrs_v.to(DEVICE)
                    _, mu_v, _, logits_v = model(x_v)
                    preds = logits_v.argmax(dim=1)
                    for cls in range(UPDRS_CLASSES):
                        mask = updrs_v == cls
                        per_class_correct[cls] += int((preds[mask] == updrs_v[mask]).sum())
                        per_class_total[cls]   += int(mask.sum())
            print("  Surrogate accuracy per class:")
            for cls in range(UPDRS_CLASSES):
                n = per_class_total[cls]
                acc = 100 * per_class_correct[cls] / n if n > 0 else 0
                print(f"    UPDRS {cls}: {acc:.1f}%  ({n} samples)")

    os.makedirs(os.path.dirname(VAE_MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), VAE_MODEL_PATH)
    print(f"\nFinal model (last epoch) saved to {VAE_MODEL_PATH}")
    print(f"Best model (ep{best_epoch}, eval loss {best_eval_loss:.4f}) saved to {best_val_path}")
    print("\nNext step: run training.train_dit")


if __name__ == "__main__":
    train()
