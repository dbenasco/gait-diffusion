"""
1D Convolutional VAE with a UPDRS surrogate classifier.

Compresses a (B, 263, 96) gait window into a (B, 64, 24) latent and reconstructs
it, with a surrogate head on the mean-pooled latent and an optional
class-prototype loss that structures the latent by severity.

Loss:
  L = MSE(recon, x) + β * KL + λ * CE(surrogate(z), updrs_label) [+ prototype]
"""

import torch
import torch.nn as nn


def _hidden_widths(in_channels: int) -> tuple:
    return (32, 64) if in_channels <= 32 else (256, 256)


class GaitVAEEncoder(nn.Module):
    def __init__(self, in_channels=8, latent_channels=8, hidden=None):
        super().__init__()
        h1, h2 = hidden or _hidden_widths(in_channels)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, h1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(h1, h2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(h2, h2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.to_mu     = nn.Conv1d(h2, latent_channels, kernel_size=1)
        self.to_logvar = nn.Conv1d(h2, latent_channels, kernel_size=1)

    def forward(self, x):
        h = self.net(x)
        logvar = torch.clamp(self.to_logvar(h), min=-10.0, max=10.0)
        return self.to_mu(h), logvar


class GaitVAEDecoder(nn.Module):
    def __init__(self, latent_channels=8, out_channels=8, hidden=None):
        super().__init__()
        h1, h2 = hidden or _hidden_widths(out_channels)
        self.net = nn.Sequential(
            nn.Conv1d(latent_channels, h2, kernel_size=1),
            nn.ReLU(),
            nn.ConvTranspose1d(h2, h2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(h2, h1, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(h1, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, z):
        return self.net(z)


class UPDRSSurrogate(nn.Module):
    """MLP on concat(mean-pool, max-pool, std-pool) of latent → UPDRS class logits."""
    def __init__(self, latent_channels=8, updrs_classes=3):
        super().__init__()
        in_dim = latent_channels * 3
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
        z_mean = z.mean(dim=2)
        z_max  = z.max(dim=2).values
        z_std  = z.std(dim=2)
        pooled = torch.cat([z_mean, z_max, z_std], dim=1)
        return self.mlp(pooled)


class GaitVAE(nn.Module):
    def __init__(self, in_channels=8, latent_channels=8, updrs_classes=3, use_prototypes=False):
        super().__init__()
        self.encoder   = GaitVAEEncoder(in_channels, latent_channels)
        self.decoder   = GaitVAEDecoder(latent_channels, in_channels)
        self.surrogate = UPDRSSurrogate(latent_channels, updrs_classes)
        self.prototypes = (
            nn.Parameter(torch.randn(updrs_classes, latent_channels) * 0.1)
            if use_prototypes else None
        )

    def load_state_dict(self, state_dict, strict=True):
        surrogate_keys = [k for k in state_dict if k.startswith("surrogate.")]
        if surrogate_keys:
            ckpt_shape = state_dict[surrogate_keys[0]].shape[0]
            model_shape = self.surrogate.mlp[-1].weight.shape[0]
            if ckpt_shape != model_shape:
                for k in surrogate_keys:
                    del state_dict[k]
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
    kl_loss   = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    surr_loss = nn.functional.cross_entropy(updrs_logits, updrs_labels, weight=class_weights)

    if prototypes is not None and (gamma_proto > 0 or gamma_dist > 0):
        mu_pooled     = mu.mean(dim=2)
        logvar_pooled = logvar.mean(dim=2)
        target        = prototypes[updrs_labels]
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
