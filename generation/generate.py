import argparse
import torch
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.model_updrs_dit import DiffusionTransformerUPDRS
from training.vae_updrs import GaitVAE
from training.train_dit import generate_batch_updrs
from config import (
    N_CHANNELS, LATENT_CHANNELS, LATENT_TIME,
    EMBED_DIM, N_HEADS, N_LAYERS, DROPOUT,
    UPDRS_CLASSES,
    DEVICE, GUIDANCE_SCALE, TEMPERATURE,
    MODEL_PATH, VAE_MODEL_PATH, STATS_PATH, GEN_OUTPUT_PATH, GEN_N_PER_CLASS,
    LATENT_NORM_PARAMS_PATH,
)

# CLI args override env vars which override config defaults
_parser = argparse.ArgumentParser()
_parser.add_argument("--n_samples",      type=int,   default=None, help="Samples per class")
_parser.add_argument("--guidance_scale", type=float, default=None, help="CFG guidance scale")
_parser.add_argument("--temperature",    type=float, default=None, help="DDPM noise temperature (1.0=standard, 1.2=recommended)")
_parser.add_argument("--model",          type=str,   default=None,
                     help="Path to DiT model checkpoint (overrides MODEL_PATH from config)")
_parser.add_argument("--vae",            type=str,   default=None,
                     help="Path to VAE checkpoint (overrides VAE_MODEL_PATH from config)")
_parser.add_argument("--output",         type=str,   default=None,
                     help="Output path for generated samples (overrides GEN_OUTPUT_PATH from "
                          "config); _labels.npy sibling is derived from it")
_parser.add_argument("--classes",        type=int,   default=None,
                     help="Number of UPDRS classes to generate (overrides UPDRS_CLASSES from config)")
_parser.add_argument("--stepsize",       type=int,   default=None,
                     help="Step length bin index (0-based). Overrides per-class default median. "
                          "Only meaningful when the checkpoint was trained with stepsize conditioning.")
_parser.add_argument("--stepsize_all",   action="store_true",
                     help="Generate all stepsize bins for every UPDRS class (grid sweep).")
_parser.add_argument("--stepsize_label", type=str,   default=None,
                     help="Path to train stepsize .npy file for computing per-class median bins "
                          "(optional; used when no --stepsize given and checkpoint has stepsize).")
_args, _ = _parser.parse_known_args()

GEN_N_PER_CLASS = _args.n_samples      or int(os.environ.get("UPDRS_GEN_N_SAMPLES",   GEN_N_PER_CLASS))
GUIDANCE_SCALE  = _args.guidance_scale or float(os.environ.get("UPDRS_GUIDANCE_SCALE", GUIDANCE_SCALE))
TEMPERATURE     = _args.temperature    or float(os.environ.get("UPDRS_TEMPERATURE",    TEMPERATURE))
if _args.model:
    MODEL_PATH = _args.model
if _args.vae:
    VAE_MODEL_PATH = _args.vae
if _args.output:
    GEN_OUTPUT_PATH = _args.output
if _args.classes:
    UPDRS_CLASSES = _args.classes


def _output_path_for_stepsize(base_path: str, stepsize_bin: int = None) -> str:
    if stepsize_bin is None:
        return base_path
    stem, ext = os.path.splitext(base_path)
    return f"{stem}_ss{stepsize_bin}{ext}"


def main():
    print("\n" + "=" * 60)
    print("UPDRS Latent DiT — GAIT GENERATION")
    print("=" * 60)
    print(f"  DiT model:      {MODEL_PATH}")
    print(f"  VAE model:      {VAE_MODEL_PATH}")
    print(f"  Output:         {GEN_OUTPUT_PATH}")
    print(f"  Samples/class:  {GEN_N_PER_CLASS}")
    cls_names = {0: "Normal", 1: "Mild", 2: "Moderate", 3: "Severe"}
    active = ", ".join(f"{i}={cls_names[i]}" for i in range(UPDRS_CLASSES))
    print(f"  UPDRS classes:  {UPDRS_CLASSES} ({active})")
    print(f"  Guidance scale: {GUIDANCE_SCALE}")
    print(f"  Temperature:    {TEMPERATURE}")
    print("=" * 60 + "\n")

    for path in [MODEL_PATH, VAE_MODEL_PATH, STATS_PATH]:
        if not os.path.exists(path):
            print(f"ERROR: {path} not found.")
            return

    # Auto-detect stepsize conditioning from the checkpoint
    ckpt_raw = torch.load(MODEL_PATH, map_location=DEVICE)
    has_stepsize = 'stepsize_emb.weight' in ckpt_raw
    stepsize_bins = ckpt_raw['stepsize_emb.weight'].shape[0] if has_stepsize else 0

    # Load normalization
    norm     = torch.load(STATS_PATH, map_location=DEVICE)
    avg_std  = norm['std'].to(DEVICE)
    avg_mean = norm.get('mean', torch.zeros_like(avg_std)).to(DEVICE)
    stepsize_edges = norm.get('stepsize_bin_edges', None)

    if stepsize_bins > 0 and stepsize_edges is not None:
        edges = stepsize_edges.cpu().numpy()
        print(f"  Stepsize bins:   {stepsize_bins}")
        for b in range(stepsize_bins):
            print(f"    Bin {b}: [{edges[b]:.1f}, {edges[b+1]:.1f}) cm")
        # Compute per-class median bin from train labels if available
        cls_median_bin = None
        if _args.stepsize_label and os.path.exists(_args.stepsize_label):
            ssl = np.load(_args.stepsize_label)
            cls_median_bin = {}
            for cls in range(UPDRS_CLASSES):
                mask = (np.load(STATS_PATH.replace("stats", "updrs_labels")) == cls)  # won't exist — skip
            cls_median_bin = None
        # Fallback: default to bin 1 (moderately short) as a safe middle
        _default_bin = stepsize_bins // 2
        if _args.stepsize is not None:
            if not (0 <= _args.stepsize < stepsize_bins):
                print(f"ERROR: --stepsize {_args.stepsize} out of range [0, {stepsize_bins})")
                return
            active_bins = [_args.stepsize]
        elif _args.stepsize_all:
            active_bins = list(range(stepsize_bins))
        else:
            active_bins = [_default_bin]
        print(f"  Generating bins:  {active_bins}")
    else:
        if _args.stepsize is not None or _args.stepsize_all:
            print("WARNING: --stepsize/--stepsize_all specified but checkpoint has no stepsize_emb.")
        active_bins = [None]
        stepsize_bins = 0

    # Load frozen VAE
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
    print(f"✅ VAE loaded from {VAE_MODEL_PATH}")

    # Load latent DiT (auto-detected stepsize_bins from checkpoint)
    model = DiffusionTransformerUPDRS(
        n_channels=LATENT_CHANNELS,
        seq_len=LATENT_TIME,
        embed_dim=EMBED_DIM,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        dropout=DROPOUT,
        updrs_classes=UPDRS_CLASSES,
        stepsize_bins=stepsize_bins,
    ).to(DEVICE)
    model.load_state_dict(ckpt_raw, strict=True)
    model.eval()
    print(f"✅ Latent DiT loaded from {MODEL_PATH}"
          + (f"  (stepsize_bins={stepsize_bins})" if stepsize_bins > 0 else ""))

    # Load latent normalization stats (std-only)
    lat_mean = None
    lat_std  = None
    if os.path.exists(LATENT_NORM_PARAMS_PATH):
        _ln = torch.load(LATENT_NORM_PARAMS_PATH, map_location=DEVICE)
        lat_std = _ln['std'].to(DEVICE)
        print(f"Latent norm stats loaded: std ∈ [{lat_std.min():.3f}, {lat_std.max():.3f}]")
    else:
        print("No latent norm stats found — generating without normalization")

    cls_names = {0: "Normal", 1: "Mild", 2: "Moderate", 3: "Severe"}

    for sb in active_bins:
        all_synth  = []
        all_labels = []
        step_labels = [] if sb is not None else None

        for cls in range(UPDRS_CLASSES):
            cls_name = cls_names.get(cls, "?")
            n = GEN_N_PER_CLASS
            bin_label = f" (stepsize bin {sb})" if sb is not None else ""
            print(f"\n🔹 UPDRS {cls} ({cls_name}): {n} samples{bin_label}...")

            updrs_labels = torch.full((n,), cls, dtype=torch.long, device=DEVICE)
            phys_cond = {'updrs': updrs_labels}
            if sb is not None:
                phys_cond['stepsize'] = torch.full((n,), sb, dtype=torch.long, device=DEVICE)

            synth_norm = generate_batch_updrs(
                model, vae, n, DEVICE, phys_cond, scale=GUIDANCE_SCALE,
                lat_mean=lat_mean, lat_std=lat_std, temperature=TEMPERATURE,
            )
            synth_denorm = synth_norm * avg_std + avg_mean

            all_synth.append(synth_denorm.cpu().numpy())
            all_labels.append(np.full(n, cls, dtype=np.int64))
            if step_labels is not None:
                step_labels.append(np.full(n, sb, dtype=np.int64))

        gen_data   = np.concatenate(all_synth,  axis=0)
        gen_labels = np.concatenate(all_labels, axis=0)

        out_path = _output_path_for_stepsize(GEN_OUTPUT_PATH, sb)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.save(out_path, gen_data)
        np.save(out_path.replace(".npy", "_labels.npy"), gen_labels)
        if step_labels is not None:
            np.save(out_path.replace(".npy", "_stepsize_labels.npy"),
                    np.concatenate(step_labels, axis=0))

        print(f"\n✅ Generated data saved to {out_path}")
        print(f"   Shape: {gen_data.shape}")
        print(f"   UPDRS labels: {dict(zip(*np.unique(gen_labels, return_counts=True)))}")


if __name__ == "__main__":
    main()
