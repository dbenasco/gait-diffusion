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
        # x: (Time, Batch, HiddenSize)
        # c: (Batch, HiddenSize)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)

        # 1. Attention
        norm_x = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + gate_msa.unsqueeze(0) * self.drop(attn_out)

        # 2. MLP
        norm_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp_out = self.mlp(norm_x)
        x = x + gate_mlp.unsqueeze(0) * self.drop(mlp_out)

        return x


class DiffusionTransformerUPDRS(nn.Module):
    """
    DiT model for UPDRS-conditioned gait generation.
    Same architecture as DiffusionTransformerDiT but:
    - No pulse conditioning channels (can be added later)
    - UPDRS embedding (4 classes) instead of step_len embedding (2 classes)

    Laterality conditioning was removed (2026-07-20). The `laterality_emb`
    parameter is kept as dead weight for strict checkpoint-loading backward
    compat with all previously trained models. ``forward`` does NOT route any
    signal through it — only `updrs_emb` is used. Retrain is required before
    evaluating.
    """
    def __init__(self, n_channels=10, seq_len=100, embed_dim=256, n_heads=4,
                 n_layers=4, dropout=0.1, updrs_classes=4, stepsize_bins=0):
        super().__init__()
        self.n_channels = n_channels
        self.seq_len = seq_len
        self.embed_dim = embed_dim

        # 1. Input Projection (joints only — no pulse channels)
        self.input_proj = nn.Linear(n_channels, embed_dim)

        # 2. Positional Encoding
        self.pos_embed = nn.Parameter(torch.zeros(seq_len, 1, embed_dim))

        # 3. Timestep Embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmbed(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # 4. UPDRS-gait Embedding (0=normal, 1=mild, 2=moderate, 3=severe)
        self.updrs_emb = nn.Embedding(updrs_classes, embed_dim)

        # 5. Laterality Embedding — dead weight, kept for strict-checkpoint
        #    backward compat. forward() does NOT route any signal through it.
        #    Newly-saved checkpoints will still contain this parameter but it
        #    receives no gradients in the current training setup.
        self.laterality_emb = nn.Embedding(3, embed_dim)

        # 5b. Stepsize Embedding (step length bin). Created only when
        #     stepsize_bins > 0, so old checkpoints load with strict=False.
        if stepsize_bins > 0:
            self.stepsize_emb = nn.Embedding(stepsize_bins, embed_dim)
        else:
            self.stepsize_emb = None

        # 6. CFG Null Embedding (drops the UPDRS conditioning axis)
        self.null_cond_emb = nn.Parameter(torch.randn(1, embed_dim) * 0.02)

        # 6. DiT Blocks
        self.blocks = nn.ModuleList([
            DiTBlock(embed_dim, n_heads, dropout=dropout) for _ in range(n_layers)
        ])

        # 7. Final Layer
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

        # Zero-init DiT block modulations (crucial for stability)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-init final layer
        nn.init.constant_(self.final_layer[-1].weight, 0)
        nn.init.constant_(self.final_layer[-1].bias, 0)
        nn.init.constant_(self.adaLN_modulation_final[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation_final[-1].bias, 0)

        # Normal init for conditioning embeddings
        nn.init.normal_(self.updrs_emb.weight,      std=0.02)
        nn.init.normal_(self.laterality_emb.weight, std=0.02)
        if self.stepsize_emb is not None:
            nn.init.normal_(self.stepsize_emb.weight, std=0.02)

    def forward(self, x, t, phys_cond=None, drop_phys=False):
        """
        x:         (batch, Channels, Time) — noisy joints
        t:         (Batch,) — diffusion timesteps
        phys_cond: dict with keys:
                     'updrs'  LongTensor (B,) ∈ {0,1,2,3}
        drop_phys: bool or BoolTensor (B,) — if True, replace phys_cond with null_cond_emb
        """
        B = x.shape[0]

        # 1. Timestep embedding
        t_emb = self.time_mlp(t)  # (B, EmbedDim)

        # 2. UPDRS conditioning (single axis — laterality removed)
        if phys_cond is not None and not (isinstance(drop_phys, bool) and drop_phys):
            updrs_e  = self.updrs_emb(phys_cond['updrs'])   # (B, D)
            phys_emb = updrs_e
            if self.stepsize_emb is not None and 'stepsize' in phys_cond:
                phys_emb = phys_emb + self.stepsize_emb(phys_cond['stepsize'])

            # Per-sample CFG dropout during training
            if isinstance(drop_phys, torch.Tensor):
                drop_mask = drop_phys.view(B, 1)
                phys_emb = torch.where(drop_mask, self.null_cond_emb.expand(B, -1), phys_emb)
        else:
            phys_emb = self.null_cond_emb.expand(B, -1)

        # Combined condition vector for AdaLN
        c = t_emb + phys_emb  # (B, EmbedDim)

        # 3. Prepare spatial inputs (no pulse concatenation)
        x = x.permute(2, 0, 1)  # (T, B, Channels)
        x = self.input_proj(x)   # (T, B, EmbedDim)
        x = x + self.pos_embed

        # 4. DiT Blocks
        for block in self.blocks:
            x = block(x, c)

        # 5. Final layer modulation and projection
        shift, scale = self.adaLN_modulation_final(c).chunk(2, dim=1)
        x = modulate(self.final_layer[0](x), shift, scale)
        x = self.final_layer[1](x)  # (T, B, n_channels)

        # Revert layout
        x = x.permute(1, 2, 0)  # (B, n_channels, T)

        return x
