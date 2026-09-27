#!/usr/bin/env python3
"""Tiny synthetic forward/backward/DDIM check for an explicitly chosen device."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from latent_diffusion import ConditionalDenoiser, GaussianDiffusion  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if args.threads < 1:
        raise ValueError("Thread count must be positive")
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    device = torch.device(args.device)
    model = ConditionalDenoiser(128, **config["model"]).to(device)
    diffusion = GaussianDiffusion(**config["diffusion"]).to(device)
    clean = torch.randn(2, 128, device=device)
    condition = torch.randn(2, 2, device=device)
    noise = torch.randn_like(clean)
    t = torch.tensor([0, diffusion.timesteps - 1], device=device)
    loss = diffusion.training_loss(model, clean, t, condition, noise=noise)
    loss.backward()
    gradients_finite = all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
                           for parameter in model.parameters())
    if not gradients_finite:
        raise FloatingPointError("Nonfinite or missing denoiser gradient")
    with torch.no_grad():
        sample = diffusion.ddim_sample(model.eval(), condition, noise, steps=20, guidance=1.0)
    if not bool(torch.isfinite(sample).all()):
        raise FloatingPointError("Nonfinite DDIM sample")
    print(json.dumps({
        "status": "passed",
        "device": args.device,
        "torch": torch.__version__,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "training_timesteps": diffusion.timesteps,
        "terminal_alpha_bar": float(diffusion.alpha_bars[-1]),
        "synthetic_loss": float(loss.detach()),
        "sample_shape": list(sample.shape),
        "sample_norms": sample.norm(dim=1).tolist(),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
