"""
1D Convolutional VAE with UPDRS surrogate for latent diffusion.

Architecture:
  Encoder: (B, 8, 180) → μ, logσ² each (B, 8, 45)   [compression r=4]
  Decoder: (B, 8, 45)  → (B, 8, 180)
  Surrogate: mean-pool z over time → MLP → UPDRS logits

Loss:
  L = MSE(recon, x) + β * KL + λ * CE(surrogate(z), updrs_label)
"""

import torch
import torch.nn as nn


# Inner conv widths (h1, h2). 6/8-channel angle inputs keep the original (32, 64)
# so existing checkpoints still load; the 263-dim HumanML3D input needs a much
# wider stem (a 263→32 first conv would be a severe early bottleneck).
def _hidden_widths(in_channels: int) -> tuple:
    return (32, 64) if in_channels <= 32 else (256, 256)


class GaitVAEEncoder(nn.Module):
    def __init__(self, in_channels=8, latent_channels=8, hidden=None):
        super().__init__()
        h1, h2 = hidden or _hidden_widths(in_channels)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, h1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(h1, h2, kernel_size=4, stride=2, padding=1),   # T → T/2
            nn.ReLU(),
            nn.Conv1d(h2, h2, kernel_size=4, stride=2, padding=1),   # T/2 → T/4
            nn.ReLU(),
        )
        self.to_mu     = nn.Conv1d(h2, latent_channels, kernel_size=1)
        self.to_logvar = nn.Conv1d(h2, latent_channels, kernel_size=1)

    def forward(self, x):
        h = self.net(x)
        # Clamp logvar: with beta this small there's little pressure keeping it
        # near 0, so over thousands of epochs it can drift up until exp()
        # overflows to inf/nan and poisons every weight from that step on.
        logvar = torch.clamp(self.to_logvar(h), min=-10.0, max=10.0)
        return self.to_mu(h), logvar


class GaitVAEDecoder(nn.Module):
    def __init__(self, latent_channels=8, out_channels=8, hidden=None):
        super().__init__()
        h1, h2 = hidden or _hidden_widths(out_channels)
        self.net = nn.Sequential(
            nn.Conv1d(latent_channels, h2, kernel_size=1),
            nn.ReLU(),
            nn.ConvTranspose1d(h2, h2, kernel_size=4, stride=2, padding=1),  # T/4 → T/2
            nn.ReLU(),
            nn.ConvTranspose1d(h2, h1, kernel_size=4, stride=2, padding=1),  # T/2 → T
            nn.ReLU(),
            nn.Conv1d(h1, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, z):
        return self.net(z)


class UPDRSSurrogate(nn.Module):
    """MLP on concat(mean-pool, max-pool, std-pool) of latent → UPDRS class logits."""
    def __init__(self, latent_channels=8, updrs_classes=3):
        super().__init__()
        in_dim = latent_channels * 3   # mean + max + std pooling
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, updrs_classes),
        )

    def forward(self, z):
        # z: (B, C, T)
        z_mean = z.mean(dim=2)
        z_max  = z.max(dim=2).values
        z_std  = z.std(dim=2)
        pooled = torch.cat([z_mean, z_max, z_std], dim=1)   # (B, 3C)
        return self.mlp(pooled)


class GaitVAE(nn.Module):
    def __init__(self, in_channels=8, latent_channels=8, updrs_classes=3, use_prototypes=False):
        super().__init__()
        self.encoder   = GaitVAEEncoder(in_channels, latent_channels)
        self.decoder   = GaitVAEDecoder(latent_channels, in_channels)
        self.surrogate = UPDRSSurrogate(latent_channels, updrs_classes)
        # Learnable per-class prototypes in mean-pooled mu space (MAD-VAE style,
        # see docs/superpowers/specs/2026-07-02-vae-class-prototype-loss-design.md).
        # Opt-in: existing checkpoints/scripts use the default GaitVAE(...) call
        # with no `prototypes` key, so they must keep loading with strict=True.
        self.prototypes = (
            nn.Parameter(torch.randn(updrs_classes, latent_channels) * 0.1)
            if use_prototypes else None
        )

    def load_state_dict(self, state_dict, strict=True):
        # Strip surrogate keys if the checkpoint's class count differs (e.g.
        # UPDRS_CLASSES=3 → 4). The surrogate is unused post-training.
        surrogate_keys = [k for k in state_dict if k.startswith("surrogate.")]
        if surrogate_keys:
            ckpt_shape = state_dict[surrogate_keys[0]].shape[0]
            model_shape = self.surrogate.mlp[-1].weight.shape[0]
            if ckpt_shape != model_shape:
                for k in surrogate_keys:
                    del state_dict[k]
        # Same for prototypes: checkpoint (3, 64) vs. model (4, 64).
        if "prototypes" in state_dict and self.prototypes is not None:
            ckpt_shape = state_dict["prototypes"].shape[0]
            model_shape = self.prototypes.shape[0]
            if ckpt_shape != model_shape:
                del state_dict["prototypes"]
        return super().load_state_dict(state_dict, strict=False)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std
        return mu

    def encode(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        recon = self.decode(z)
        updrs_logits = self.surrogate(z)
        return recon, mu, logvar, updrs_logits


def vae_loss(recon, x, mu, logvar, updrs_logits, updrs_labels, beta=0.001, lam=0.1,
             prototypes=None, margin=1.0, gamma_proto=0.0, gamma_dist=0.0,
             class_weights=None):
    recon_loss = nn.functional.mse_loss(recon, x)
    # Per-timestep KL to the standard N(0, I) prior — unchanged. This is what the
    # decoder is trained against, so touching it risks flattening the temporal
    # (within-window) dynamics the encoder needs to reconstruct gait. The
    # class-prototype pull below is a separate, additional term that only
    # constrains the pooled (per-sample, per-channel) summary, not every
    # individual timestep.
    kl_loss   = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    surr_loss = nn.functional.cross_entropy(updrs_logits, updrs_labels, weight=class_weights)

    if prototypes is not None and (gamma_proto > 0 or gamma_dist > 0):
        # Class-conditional KL on the POOLED distribution only:
        # KL(N(mu_pooled, exp(logvar_pooled)) || N(prototype[label], I)).
        # Because this has a learned variance (not just a mean pulled by MSE),
        # each class's pooled latent is regularized toward an actual Gaussian
        # cloud around its own prototype "slot" — it can't collapse onto a
        # single discrete point the way a bare distance/MSE loss could pull it.
        # See docs/superpowers/specs/2026-07-02-vae-class-prototype-loss-design.md.
        mu_pooled     = mu.mean(dim=2)                     # (B, C)
        logvar_pooled = logvar.mean(dim=2)                 # (B, C) — pooled log-variance
        target        = prototypes[updrs_labels]           # (B, C)
        proto_kl_loss = -0.5 * torch.mean(
            1 + logvar_pooled - (mu_pooled - target).pow(2) - logvar_pooled.exp()
        )

        n_classes = prototypes.shape[0]
        iu = torch.triu_indices(n_classes, n_classes, offset=1, device=prototypes.device)
        pair_dists = torch.norm(prototypes[iu[0]] - prototypes[iu[1]], dim=1)
        distance_loss = torch.relu(margin - pair_dists).pow(2).mean()
    else:
        proto_kl_loss = torch.tensor(0.0, device=recon.device)
        distance_loss = torch.tensor(0.0, device=recon.device)

    total = (
        recon_loss + beta * kl_loss + lam * surr_loss
        + gamma_proto * proto_kl_loss + gamma_dist * distance_loss
    )
    return total, recon_loss, kl_loss, surr_loss, proto_kl_loss, distance_loss
