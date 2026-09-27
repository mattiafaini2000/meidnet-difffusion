"""Noise-prediction MLP; this module has no MEIDNet or materials dependency."""

import math

import torch
from torch import nn
from torch.nn import functional as F


def sinusoidal_timestep_embedding(t: torch.Tensor, width: int) -> torch.Tensor:
    """Embed 0-based diffusion timesteps with fixed sine/cosine frequencies."""
    if t.ndim != 1:
        raise ValueError("Timesteps must have shape [batch]")
    half = width // 2
    frequencies = torch.exp(
        -math.log(10_000) * torch.arange(half, device=t.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    phases = t.float()[:, None] * frequencies[None, :]
    embedding = torch.cat((phases.sin(), phases.cos()), dim=1)
    return F.pad(embedding, (0, width - embedding.shape[1]))


class ResidualBlock(nn.Module):
    """Apply timestep/property context through addition or featurewise scale/shift."""
    def __init__(self, width: int, expansion: int, context_mode: str):
        super().__init__()
        if context_mode not in ("additive", "film"):
            raise ValueError("context_mode must be 'additive' or 'film'")
        self.context_mode = context_mode
        self.norm = nn.LayerNorm(width)
        self.context = nn.Linear(width, width if context_mode == "additive" else 2 * width)
        self.up = nn.Linear(width, expansion * width)
        self.down = nn.Linear(expansion * width, width)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        if self.context_mode == "additive":
            h = h + self.context(context)
        else:
            scale, shift = self.context(context).chunk(2, dim=-1)
            h = h * (1 + scale) + shift
        return x + self.down(F.silu(self.up(F.silu(h))))


class ConditionalDenoiser(nn.Module):
    """Predict epsilon from a noisy latent, timestep, and optional two-property pair.

    `condition_present=False` masks both numerical properties and supplies a zero
    presence bit. It is therefore distinct from a present standardized [0, 0].
    """

    def __init__(
        self,
        latent_dim: int,
        width: int = 256,
        blocks: int = 4,
        expansion: int = 2,
        context_mode: str = "film",
    ):
        super().__init__()
        if min(latent_dim, width, blocks, expansion) < 1:
            raise ValueError("All model dimensions must be positive")
        self.latent_dim = latent_dim
        self.width = width
        self.blocks = blocks
        self.expansion = expansion
        self.context_mode = context_mode
        self.input = nn.Linear(latent_dim, width)
        self.time_mlp = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.condition_mlp = nn.Sequential(nn.Linear(3, width), nn.SiLU(), nn.Linear(width, width))
        self.residuals = nn.ModuleList(
            ResidualBlock(width, expansion, context_mode) for _ in range(blocks)
        )
        self.output_norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, latent_dim)

    def forward(
        self,
        u_t: torch.Tensor,
        t: torch.Tensor,
        conditions: torch.Tensor | None,
        condition_present: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return epsilon predictions with the same shape as ``u_t``."""
        if u_t.ndim != 2 or u_t.shape[1] != self.latent_dim:
            raise ValueError("Noisy latents must have shape [batch, latent_dim]")
        batch = u_t.shape[0]
        if t.shape != (batch,):
            raise ValueError("Timesteps must have shape [batch]")
        if conditions is None:
            if condition_present is not None and bool(condition_present.any()):
                raise ValueError("A present condition needs numerical values")
            conditions = torch.zeros(batch, 2, device=u_t.device, dtype=u_t.dtype)
            present = torch.zeros(batch, device=u_t.device, dtype=torch.bool)
        else:
            if conditions.shape != (batch, 2):
                raise ValueError("Conditions must have shape [batch, 2]")
            present = (
                torch.ones(batch, device=u_t.device, dtype=torch.bool)
                if condition_present is None else condition_present.to(device=u_t.device, dtype=torch.bool)
            )
            if present.shape != (batch,):
                raise ValueError("Condition presence must have shape [batch]")
        if not bool(torch.isfinite(u_t).all()) or not bool(torch.isfinite(conditions).all()):
            raise FloatingPointError("Nonfinite denoiser input")
        condition_input = torch.cat(
            (torch.where(present[:, None], conditions, 0.0), present[:, None].to(conditions.dtype)),
            dim=-1,
        )
        context = self.time_mlp(sinusoidal_timestep_embedding(t, self.width))
        context = context + self.condition_mlp(condition_input)
        h = self.input(u_t)
        for block in self.residuals:
            h = block(h, context)
        prediction = self.output(F.silu(self.output_norm(h)))
        if not bool(torch.isfinite(prediction).all()):
            raise FloatingPointError("Nonfinite denoiser output")
        return prediction
