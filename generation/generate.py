"""Generate UPDRS-conditioned gait with the latent Diffusion Transformer.

Paper settings: guidance scale 1.5, temperature 1.0 (config defaults).

Usage:
    python -m generation.generate [--n_samples 300] [--gen_path PATH]
                                  [--model PATH] [--vae PATH] [--output PATH]
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    N_CHANNELS, LATENT_CHANNELS, LATENT_TIME,
    EMBED_DIM, N_HEADS, N_LAYERS, DROPOUT, UPDRS_CLASSES,
    DEVICE, GUIDANCE_SCALE, TEMPERATURE,
    MODEL_PATH, VAE_MODEL_PATH, STATS_PATH, GEN_OUTPUT_PATH, GEN_N_PER_CLASS,
    LATENT_NORM_PARAMS_PATH,
)
from training.model_updrs_dit import DiffusionTransformerUPDRS
from training.vae_updrs import GaitVAE
from training.train_dit import generate_batch_updrs

CLS_NAMES = {0: "Normal", 1: "Mild", 2: "Moderate", 3: "Severe"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=None, help="Samples per class")
    parser.add_argument("--guidance_scale", type=float, default=None, help="CFG guidance scale")
    parser.add_argument("--temperature", type=float, default=None, help="DDPM noise temperature")
    parser.add_argument("--model", type=str, default=None, help="DiT checkpoint")
    parser.add_argument("--vae", type=str, default=None, help="VAE checkpoint")
    parser.add_argument("--output", type=str, default=None, help="Output .npy path")
    args = parser.parse_args()

    n_per_class = args.n_samples or int(os.environ.get("UPDRS_GEN_N_SAMPLES", GEN_N_PER_CLASS))
    guidance = args.guidance_scale or float(os.environ.get("UPDRS_GUIDANCE_SCALE", GUIDANCE_SCALE))
    temperature = args.temperature or float(os.environ.get("UPDRS_TEMPERATURE", TEMPERATURE))
    model_path = args.model or MODEL_PATH
    vae_path = args.vae or VAE_MODEL_PATH
    out_path = args.output or GEN_OUTPUT_PATH

    print("=" * 60)
    print("  UPDRS latent DiT — gait generation")
    print("=" * 60)
    print(f"  DiT model:      {model_path}")
    print(f"  VAE model:      {vae_path}")
    print(f"  Output:         {out_path}")
    print(f"  Samples/class:  {n_per_class}")
    print(f"  Guidance scale: {guidance}")
    print(f"  Temperature:    {temperature}")
    print("=" * 60 + "\n")

    for path in (model_path, vae_path, STATS_PATH):
        if not os.path.exists(path):
            print(f"ERROR: {path} not found.")
            return

    vae_state = torch.load(vae_path, map_location=DEVICE)
    vae = GaitVAE(in_channels=N_CHANNELS, latent_channels=LATENT_CHANNELS,
                  updrs_classes=UPDRS_CLASSES,
                  use_prototypes="prototypes" in vae_state).to(DEVICE)
    vae.load_state_dict(vae_state)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    model = DiffusionTransformerUPDRS(
        n_channels=LATENT_CHANNELS, seq_len=LATENT_TIME, embed_dim=EMBED_DIM,
        n_heads=N_HEADS, n_layers=N_LAYERS, dropout=DROPOUT,
        updrs_classes=UPDRS_CLASSES,
    ).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE), strict=False)
    model.eval()

    norm = torch.load(STATS_PATH, map_location=DEVICE)
    std = norm["std"].to(DEVICE)
    mean = norm.get("mean", torch.zeros_like(std)).to(DEVICE)

    lat_std = None
    if os.path.exists(LATENT_NORM_PARAMS_PATH):
        lat_std = torch.load(LATENT_NORM_PARAMS_PATH, map_location=DEVICE)["std"].to(DEVICE)

    all_synth, all_labels = [], []
    for cls in range(UPDRS_CLASSES):
        print(f"  UPDRS {cls} ({CLS_NAMES.get(cls, '?')}): {n_per_class} samples...")
        phys_cond = {"updrs": torch.full((n_per_class,), cls, dtype=torch.long, device=DEVICE)}
        synth = generate_batch_updrs(
            model, vae, n_per_class, DEVICE, phys_cond,
            scale=guidance, lat_mean=None, lat_std=lat_std, temperature=temperature,
        )
        all_synth.append((synth * std + mean).cpu().numpy())
        all_labels.append(np.full(n_per_class, cls, dtype=np.int64))

    gen_data = np.concatenate(all_synth, axis=0)
    gen_labels = np.concatenate(all_labels, axis=0)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, gen_data)
    np.save(out_path.replace(".npy", "_labels.npy"), gen_labels)
    print(f"\n  Saved {gen_data.shape} to {out_path}")


if __name__ == "__main__":
    main()
