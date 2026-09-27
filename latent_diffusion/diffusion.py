"""Gaussian forward diffusion and deterministic DDIM for continuous latents.

Ho et al., Denoising Diffusion Probabilistic Models (2020), eqs. 2 and 12:
q(u_t | u_0) and epsilon-prediction MSE. Song et al., Denoising Diffusion
Implicit Models (2020), eq. 12 with eta=0: the deterministic transition.
Timesteps are 0-based; index 0 is the first *noisy* training state.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class GaussianDiffusion(nn.Module):
    """Fixed linear-beta schedule for [batch, latent_dim] float tensors.

    Level zero is noisy; the clean endpoint of deterministic DDIM sampling is
    represented by ``previous_timestep=-1``.
    """
    def __init__(self, timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        super().__init__()
        if timesteps < 2 or not 0 < beta_start <= beta_end < 1:
            raise ValueError("Require >=2 timesteps and 0 < beta_start <= beta_end < 1")
        self.timesteps = timesteps
        betas64 = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)
        log_alpha_bars = torch.cumsum(torch.log1p(-betas64), dim=0)
        alpha_bars64 = log_alpha_bars.exp()
        if not bool((alpha_bars64 > 0).all()) or not bool((alpha_bars64[1:] < alpha_bars64[:-1]).all()):
            raise ArithmeticError("Invalid diffusion schedule")
        self.register_buffer("betas", betas64.float())
        self.register_buffer("alpha_bars", alpha_bars64.float())
        self.register_buffer("sqrt_alpha_bars", alpha_bars64.sqrt().float())
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(-torch.expm1(log_alpha_bars)).float())

    def _coefficient(self, values: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if t.ndim != 1 or t.shape[0] != x.shape[0] or t.dtype != torch.long:
            raise ValueError("Timesteps must be int64 with shape [batch]")
        if bool((t < 0).any()) or bool((t >= self.timesteps).any()):
            raise ValueError("Timestep outside the 0-based schedule")
        return values[t].to(device=x.device, dtype=x.dtype)[:, None]

    def q_sample(self, u0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        """sqrt(alpha_bar_t) u0 + sqrt(1-alpha_bar_t) epsilon."""
        if noise is None:
            noise = torch.randn_like(u0)
        if noise.shape != u0.shape:
            raise ValueError("Noise must match clean-latent shape")
        result = (self._coefficient(self.sqrt_alpha_bars, t, u0) * u0
                  + self._coefficient(self.sqrt_one_minus_alpha_bars, t, u0) * noise)
        if not bool(torch.isfinite(result).all()):
            raise FloatingPointError("Nonfinite forward-diffusion state")
        return result

    def predict_x0(self, u_t: torch.Tensor, t: torch.Tensor, epsilon: torch.Tensor) -> torch.Tensor:
        """DDIM's clean-state estimate from an epsilon prediction."""
        if epsilon.shape != u_t.shape:
            raise ValueError("Predicted noise must match the noisy latent")
        result = (u_t - self._coefficient(self.sqrt_one_minus_alpha_bars, t, u_t) * epsilon) \
            / self._coefficient(self.sqrt_alpha_bars, t, u_t)
        if not bool(torch.isfinite(result).all()):
            raise FloatingPointError("Nonfinite clean-latent estimate")
        return result

    def training_loss(
        self,
        model: nn.Module,
        u0: torch.Tensor,
        t: torch.Tensor,
        conditions: torch.Tensor,
        condition_present: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Epsilon MSE; the caller owns training RNG and condition dropout."""
        if noise is None:
            noise = torch.randn_like(u0)
        prediction = model(self.q_sample(u0, t, noise), t, conditions, condition_present)
        if prediction.shape != noise.shape:
            raise ValueError("Denoiser prediction and noise shapes differ")
        loss = F.mse_loss(prediction, noise)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite diffusion loss")
        return loss

    def ddim_timesteps(self, steps: int) -> torch.Tensor:
        """Strict descending trained levels, including T-1 and 0."""
        if not 2 <= steps <= self.timesteps:
            raise ValueError("DDIM steps must be between 2 and the training horizon")
        levels = torch.linspace(0, self.timesteps - 1, steps).round().long().flip(0)
        if levels[0] != self.timesteps - 1 or levels[-1] != 0 or not bool((levels[:-1] > levels[1:]).all()):
            raise ArithmeticError("DDIM levels must be unique with both endpoints")
        return levels

    def ddim_step(self, u_t: torch.Tensor, timestep: int, previous_timestep: int,
                  epsilon: torch.Tensor) -> torch.Tensor:
        """Eta=0 transition; previous_timestep=-1 is the final clean state."""
        if not 0 <= timestep < self.timesteps or not -1 <= previous_timestep < timestep:
            raise ValueError("Expected a descending timestep or clean endpoint -1")
        t = torch.full((u_t.shape[0],), timestep, device=u_t.device, dtype=torch.long)
        x0 = self.predict_x0(u_t, t, epsilon)
        if previous_timestep == -1:
            return x0  # alpha_bar_clean = 1, so the epsilon coefficient is zero.
        sqrt_alpha_prev = self.sqrt_alpha_bars[previous_timestep].to(device=u_t.device, dtype=u_t.dtype)
        sqrt_noise_prev = self.sqrt_one_minus_alpha_bars[previous_timestep].to(device=u_t.device, dtype=u_t.dtype)
        result = sqrt_alpha_prev * x0 + sqrt_noise_prev * epsilon
        if not bool(torch.isfinite(result).all()):
            raise FloatingPointError("Nonfinite DDIM state")
        return result

    def guided_epsilon(self, model: nn.Module, u_t: torch.Tensor, t: torch.Tensor,
                       conditions: torch.Tensor, guidance: float) -> torch.Tensor:
        """Classifier-free guidance: eps_u + s(eps_c - eps_u).

        Both passes use the same noisy latent and standardized property pair;
        the presence bit alone distinguishes conditional and unconditional.
        """
        if not math.isfinite(guidance) or guidance < 0:
            raise ValueError("Guidance must be finite and nonnegative")
        absent = torch.zeros(u_t.shape[0], device=u_t.device, dtype=torch.bool)
        present = torch.ones_like(absent)
        eps_uncond = model(u_t, t, conditions, absent)
        eps_cond = model(u_t, t, conditions, present)
        result = eps_uncond + guidance * (eps_cond - eps_uncond)
        if result.shape != u_t.shape or not bool(torch.isfinite(result).all()):
            raise FloatingPointError("Nonfinite or malformed guided prediction")
        return result

    @torch.no_grad()
    def ddim_sample(
        self,
        model: nn.Module,
        conditions: torch.Tensor,
        initial_noise: torch.Tensor,
        steps: int = 20,
        guidance: float = 1.0,
    ) -> torch.Tensor:
        """Map caller-supplied Gaussian noise and properties to a clean latent.

        No reference clean latent, CIF, geometry, or species enters this API.
        The caller owns the initial-noise RNG. The final step uses alpha_bar=1.
        """
        if initial_noise.ndim != 2 or conditions.shape != (initial_noise.shape[0], 2):
            raise ValueError("Expected noise [batch, latent_dim] and conditions [batch, 2]")
        if not bool(torch.isfinite(initial_noise).all()) or not bool(torch.isfinite(conditions).all()):
            raise FloatingPointError("Nonfinite DDIM input")
        sample = initial_noise.clone()
        levels = self.ddim_timesteps(steps).tolist()
        for index, timestep in enumerate(levels):
            t = torch.full((sample.shape[0],), timestep, device=sample.device, dtype=torch.long)
            epsilon = self.guided_epsilon(model, sample, t, conditions, guidance)
            previous = levels[index + 1] if index + 1 < len(levels) else -1
            sample = self.ddim_step(sample, timestep, previous, epsilon)
        return sample
