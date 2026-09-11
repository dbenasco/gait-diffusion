import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.model_updrs_dit import DiffusionTransformerUPDRS
from training.vae_updrs import GaitVAE
from config import (
    TRAIN_DATA_PATH, TRAIN_LABELS_PATH,
    EVAL_DATA_PATH, EVAL_LABELS_PATH,
    NORM_PARAMS_PATH, MODEL_PATH, STATS_PATH,
    VAE_MODEL_PATH,
    N_CHANNELS, SEQ_LEN, LATENT_CHANNELS, LATENT_TIME,
    EMBED_DIM, N_HEADS, N_LAYERS, DROPOUT,
    UPDRS_CLASSES,
    BATCH_SIZE, TIMESTEPS, LEARNING_RATE, EPOCHS, DEVICE,
    CFG_DROPOUT_PROB,
    MSE_WEIGHT, VELOCITY_WEIGHT,
    ROM_WEIGHT, TARGET_ROM_DEG,
    RESUME_TRAINING, START_EPOCH,
    CLASS_WEIGHTED_LOSS,
)
from h3d_bridge import h3d_to_angles


# ============================================================
# Noise schedule (same as existing pipeline)
# ============================================================

def get_noise_schedule(beta_start=1e-4, beta_end=0.02, n_steps=500):
    return torch.linspace(beta_start, beta_end, n_steps)


# ============================================================
# Full-body ROM supervision (per-class batch-mean L1 vs real targets)
# ============================================================

def compute_rom_loss(pred_z0, updrs_cls, cond_mask, vae, dataset, device):
    """
    Decode pred_z0 → denormalized angle space → per-UPDRS-class batch-mean ROM
    vs TARGET_ROM_DEG (L1). Returns (weighted_loss, rom_loss); skips the decode
    entirely when ROM_WEIGHT == 0.
    """
    zero = torch.tensor(0.0, device=device)
    if ROM_WEIGHT == 0.0:
        return zero, zero

    x_pred = vae.decode(pred_z0)
    avg_std_d = dataset.avg_std.to(device).view(1, -1, 1)
    avg_mean_d = dataset.avg_mean.to(device).view(1, -1, 1)
    x_pred_deg = x_pred * avg_std_d + avg_mean_d

    leg_ang = h3d_to_angles(x_pred_deg, sagittal_only=True)             # (B, 6, T)
    pred_rom = leg_ang.max(dim=-1).values - leg_ang.min(dim=-1).values  # (B, 6)

    rom_loss = zero
    n_cls = 0
    for cls_idx in range(UPDRS_CLASSES):
        mask = (updrs_cls == cls_idx) & cond_mask
        if mask.sum() == 0:
            continue
        n_cls += 1
        rom_loss = rom_loss + F.l1_loss(pred_rom[mask].mean(dim=0),
                                        TARGET_ROM_DEG[cls_idx].to(device))
    if n_cls > 0:
        rom_loss = rom_loss / n_cls

    return ROM_WEIGHT * rom_loss, rom_loss


# ============================================================
# Dataset
# ============================================================

class GaitDatasetUPDRS(Dataset):
    def __init__(self, tensor_data, labels, norm_stats=None):
        """tensor_data: (N, C, T); labels: (N,) UPDRS labels; norm_stats: optional (mean, std)."""
        self.data = tensor_data
        self.updrs = labels

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
        normalized = (sample - self.avg_mean.to(sample.device)) / self.avg_std.to(sample.device)
        normalized = torch.clamp(normalized, -4, 4)
        return normalized, {'updrs': self.updrs[idx]}


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
        print("Run preprocessing.preprocess_carepd_h3d first.")
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
    train_set = GaitDatasetUPDRS(train_tensor, train_labels)
    val_set = GaitDatasetUPDRS(eval_tensor, eval_labels,
                               norm_stats=(train_set.avg_mean, train_set.avg_std))
    dataset = train_set  # reference for normalization stats

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    norm_params = {'std': dataset.avg_std, 'mean': dataset.avg_mean}
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
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model on {DEVICE}. Total Params: {total_params / 1e6:.2f}M")

    # Resume
    start_epoch = 0
    if RESUME_TRAINING and os.path.exists(MODEL_PATH):
        print(f"🔄 Resuming from: {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
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
    print(f"  CLASS_WEIGHTED_LOSS  : {CLASS_WEIGHTED_LOSS}")
    if CLASS_WEIGHTED_LOSS:
        for cls in range(UPDRS_CLASSES):
            print(f"    UPDRS {cls} : n={int(train_counts[cls])}  w={class_weights[cls].item():.3f}")
    print("=" * 60 + "\n")

    # 4. Training loop
    best_val_loss = float('inf')
    best_epoch    = start_epoch

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        batch_losses = []
        batch_rom_losses = []

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

            fb_loss, rom_loss = compute_rom_loss(
                pred_z0, updrs_cls, cond_mask, vae, dataset, DEVICE)

            total_loss = MSE_WEIGHT * mse_loss + VELOCITY_WEIGHT * vel_loss + fb_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_losses.append(total_loss.item())
            batch_rom_losses.append(rom_loss.item())

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
                fb_v, _ = compute_rom_loss(
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

        if (epoch + 1) % 500 == 0:
            _p = Path(MODEL_PATH)
            ckpt_path = str(_p.with_name(_p.stem + f'_ep{epoch + 1}' + _p.suffix))
            torch.save(model.state_dict(), ckpt_path)
            print(f"📌 Numbered checkpoint: {ckpt_path}")

    print(f"\n✅ Training complete. Best epoch: {best_epoch}  val_loss: {best_val_loss:.6f}")
    print(f"   Best model already saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()
