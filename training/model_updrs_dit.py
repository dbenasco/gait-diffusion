"""
Diffusion Transformer (DiT) for UPDRS-conditioned latent gait generation.

AdaLN-Zero conditioning on the UPDRS class embedding plus a null-condition
embedding for classifier-free guidance.
"""

import torch
import torch.nn as nn
import math


class SinusoidalPosEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(0)) + shift.unsqueeze(0)


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm (AdaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads, dropout=dropout
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)

        norm_x = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + gate_msa.unsqueeze(0) * self.drop(attn_out)

        norm_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp_out = self.mlp(norm_x)
        x = x + gate_mlp.unsqueeze(0) * self.drop(mlp_out)

        return x


class DiffusionTransformerUPDRS(nn.Module):
    """DiT for UPDRS-conditioned gait generation (AdaLN-Zero, CFG)."""

    def __init__(self, n_channels=10, seq_len=100, embed_dim=256, n_heads=4,
                 n_layers=4, dropout=0.1, updrs_classes=4):
        super().__init__()
        self.n_channels = n_channels
        self.seq_len = seq_len
        self.embed_dim = embed_dim

        self.input_proj = nn.Linear(n_channels, embed_dim)

        self.pos_embed = nn.Parameter(torch.zeros(seq_len, 1, embed_dim))

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmbed(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.updrs_emb = nn.Embedding(updrs_classes, embed_dim)

        self.null_cond_emb = nn.Parameter(torch.randn(1, embed_dim) * 0.02)

        self.blocks = nn.ModuleList([
            DiTBlock(embed_dim, n_heads, dropout=dropout) for _ in range(n_layers)
        ])

        self.final_layer = nn.Sequential(
            nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6),
            nn.Linear(embed_dim, n_channels)
        )
        self.adaLN_modulation_final = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, 2 * embed_dim, bias=True)
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer[-1].weight, 0)
        nn.init.constant_(self.final_layer[-1].bias, 0)
        nn.init.constant_(self.adaLN_modulation_final[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation_final[-1].bias, 0)

        nn.init.normal_(self.updrs_emb.weight, std=0.02)

    def forward(self, x, t, phys_cond=None, drop_phys=False):
        """
        x:         (batch, Channels, Time) — noisy latents
        t:         (Batch,) — diffusion timesteps
        phys_cond: dict with key 'updrs' LongTensor (B,) ∈ {0,1,2,3}
        drop_phys: bool or BoolTensor (B,) — replace conditioning with null embedding
        """
        B = x.shape[0]

        t_emb = self.time_mlp(t)

        if phys_cond is not None and not (isinstance(drop_phys, bool) and drop_phys):
            phys_emb = self.updrs_emb(phys_cond['updrs'])
            if isinstance(drop_phys, torch.Tensor):
                drop_mask = drop_phys.view(B, 1)
                phys_emb = torch.where(drop_mask, self.null_cond_emb.expand(B, -1), phys_emb)
        else:
            phys_emb = self.null_cond_emb.expand(B, -1)

        c = t_emb + phys_emb

        x = x.permute(2, 0, 1)
        x = self.input_proj(x)
        x = x + self.pos_embed

        for block in self.blocks:
            x = block(x, c)

        shift, scale = self.adaLN_modulation_final(c).chunk(2, dim=1)
        x = modulate(self.final_layer[0](x), shift, scale)
        x = self.final_layer[1](x)

        x = x.permute(1, 2, 0)

        return x
