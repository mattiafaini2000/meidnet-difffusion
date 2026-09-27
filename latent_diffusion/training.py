"""Small, resumable epsilon-prediction trainer for cached MEIDNet joint latents.

The trainer never imports MEIDNet or reads CIFs. Its inputs are already encoded,
CPU float32 train/validation caches. A validation bank is generated once from a
separate random stream and kept fixed for every reported comparison.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from latent_diffusion.data import apply_scalers, fit_scalers, validate_scalers
from latent_diffusion.diffusion import GaussianDiffusion
from latent_diffusion.model import ConditionalDenoiser


CHECKPOINT_VERSION = 1
NOISE_LEVEL_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)


def _json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; no CPU fallback")
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Use an explicit cpu or cuda device")
    return device


def _numpy_state() -> dict[str, Any]:
    name, keys, position, has_gauss, gaussian = np.random.get_state()
    return {
        "name": name,
        "keys": keys.tolist(),
        "position": position,
        "has_gauss": has_gauss,
        "gaussian": gaussian,
    }


def _restore_numpy_state(state: dict[str, Any]) -> None:
    np.random.set_state(
        (
            state["name"],
            np.asarray(state["keys"], dtype=np.uint32),
            state["position"],
            state["has_gauss"],
            state["gaussian"],
        )
    )


def _cpu_copy(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    return value


def _move_optimizer(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(_cpu_copy(payload), temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _check_config(config: dict[str, Any], latent_dim: int) -> None:
    model = config["model"]
    diffusion = config["diffusion"]
    train = config["training"]
    if "latent_dim" in model and model["latent_dim"] != latent_dim:
        raise ValueError("Configured latent dimension disagrees with the cache")
    if latent_dim < 1 or train["batch_size"] < 1 or train["max_steps"] < 1:
        raise ValueError("Latent dimension, batch size and maximum steps must be positive")
    if not 0 <= train["condition_dropout"] < 1:
        raise ValueError("Condition dropout must be in [0, 1)")
    if train["num_workers"] != 0:
        raise ValueError("This entry point uses num_workers=0")
    if train["validation_frequency"] < 1 or train["validation_bank_size"] < 1:
        raise ValueError("Validation frequency and bank size must be positive")
    if train["lr"] <= 0 or train["weight_decay"] < 0 or train["gradient_clip"] <= 0:
        raise ValueError("Invalid optimizer hyperparameters")
    if diffusion["timesteps"] < 2:
        raise ValueError("Diffusion needs at least two timesteps")


def _compatibility(config: dict[str, Any], train_cache: dict, val_cache: dict, scalers: dict, mode: str) -> dict:
    return {
        "model": dict(config["model"]) | {"latent_dim": train_cache["latents"].shape[1]},
        "diffusion": dict(config["diffusion"]),
        "training": dict(config["training"]),
        "mode": mode,
        "train_cache_fingerprint": train_cache["manifest"]["fingerprint"],
        "val_cache_fingerprint": val_cache["manifest"]["fingerprint"],
        "train_selected_ids_sha256": train_cache["manifest"]["selected_ids_sha256"],
        "val_selected_ids_sha256": val_cache["manifest"]["selected_ids_sha256"],
        "scaler_hash": scalers["hash"],
        "train_cache_manifest": train_cache["manifest"],
        "val_cache_manifest": val_cache["manifest"],
    }


def _validation_bank(
    u: torch.Tensor,
    c: torch.Tensor,
    raw_conditions: torch.Tensor,
    diffusion: GaussianDiffusion,
    size: int,
    generator: torch.Generator,
) -> dict[str, torch.Tensor]:
    if len(u) == 0:
        raise ValueError("Validation cache is empty")
    chosen = torch.randperm(len(u), device=u.device, generator=generator)[: min(size, len(u))]
    levels = torch.tensor(
        [round(fraction * (diffusion.timesteps - 1)) for fraction in NOISE_LEVEL_FRACTIONS],
        dtype=torch.long,
        device=u.device,
    )
    indices = chosen.repeat_interleave(len(levels))
    timesteps = levels.repeat(len(chosen))
    clean = u[indices]
    noise = torch.randn(clean.shape, device=u.device, dtype=clean.dtype, generator=generator)
    return {
        "u0": clean,
        "c": c[indices],
        "raw_gap_zero": raw_conditions[indices, 1] == 0,
        "t": timesteps,
        "noise": noise,
    }


def fixed_validation_metrics(
    model: ConditionalDenoiser,
    diffusion: GaussianDiffusion,
    bank: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Evaluate one fixed corruption bank, including zero-gap strata and baselines."""
    was_training = model.training
    model.eval()
    with torch.no_grad():
        noisy = diffusion.q_sample(bank["u0"], bank["t"], bank["noise"])
        predicted = model(noisy, bank["t"], bank["c"])
        if not torch.isfinite(predicted).all():
            raise FloatingPointError("Non-finite validation prediction")
        errors = (predicted - bank["noise"]).square().mean(dim=1)
        zero_baseline = bank["noise"].square().mean(dim=1)
        noisy_baseline = (noisy - bank["noise"]).square().mean(dim=1)
        by_level = []
        for timestep in sorted(set(bank["t"].tolist())):
            for gap_group, mask in (
                ("zero", bank["raw_gap_zero"]),
                ("positive", ~bank["raw_gap_zero"]),
            ):
                selected = (bank["t"] == timestep) & mask
                count = int(selected.sum().item())
                by_level.append(
                    {
                        "timestep": timestep,
                        "gap_group": gap_group,
                        "count": count,
                        "mse": float(errors[selected].mean().item()) if count else None,
                        "zero_predictor_mse": float(zero_baseline[selected].mean().item()) if count else None,
                        "noisy_latent_predictor_mse": float(noisy_baseline[selected].mean().item()) if count else None,
                    }
                )
        result = {
            "count": len(bank["u0"]),
            "mse": float(errors.mean().item()),
            "zero_predictor_mse": float(zero_baseline.mean().item()),
            "noisy_latent_predictor_mse": float(noisy_baseline.mean().item()),
            "by_level_and_gap": by_level,
        }
    model.train(was_training)
    return result


class DenoiserTrainer:
    """One run, with isolated batch/corruption/validation/sampling RNG streams."""

    def __init__(
        self,
        train_cache: dict,
        val_cache: dict,
        config: dict[str, Any],
        *,
        mode: str = "smoke",
        device: str = "cpu",
    ) -> None:
        if mode not in ("smoke", "overfit"):
            raise ValueError("mode must be smoke or overfit")
        self.device = _device(device)
        if train_cache["latents"].ndim != 2 or val_cache["latents"].ndim != 2:
            raise ValueError("Caches need [N, D] latent arrays")
        latent_dim = train_cache["latents"].shape[1]
        if val_cache["latents"].shape[1] != latent_dim:
            raise ValueError("Train and validation latent dimensions differ")
        if not train_cache["ids"] or not val_cache["ids"]:
            raise ValueError("Both caches must contain examples")
        if train_cache["manifest"].get("split") != "train" or val_cache["manifest"].get("split") != "val":
            raise ValueError("Training requires a train cache and a validation cache")
        if train_cache["manifest"].get("status") != "complete" or val_cache["manifest"].get("status") != "complete":
            raise ValueError("Finish the selected train/validation caches before training")
        if set(train_cache["ids"]) & set(val_cache["ids"]):
            raise ValueError("Train and validation cache IDs overlap")
        if any("test" in str(cache["manifest"].get("split", "")).lower() for cache in (train_cache, val_cache)):
            raise ValueError("Test-split caches are prohibited for this milestone")
        _check_config(config, latent_dim)
        self.config = config
        self.mode = mode
        self.scalers = fit_scalers(train_cache)
        validate_scalers(self.scalers)
        self.compatibility = _compatibility(config, train_cache, val_cache, self.scalers, mode)
        self.compatibility_hash = _json_hash(self.compatibility)
        seed = int(config["training"]["seed"])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.model = ConditionalDenoiser(latent_dim=latent_dim, **{key: value for key, value in config["model"].items() if key != "latent_dim"}).to(self.device)
        self.diffusion = GaussianDiffusion(**config["diffusion"]).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config["training"]["lr"],
            weight_decay=config["training"]["weight_decay"],
        )
        self.generators = {
            name: torch.Generator(device=self.device).manual_seed(seed + offset)
            for name, offset in (("batch", 11), ("train", 23), ("validation", 37), ("sampling", 53), ("categorical", 71))
        }
        self.train_u, self.train_c = (tensor.to(self.device) for tensor in apply_scalers(train_cache["latents"], train_cache["conditions"], self.scalers))
        val_u, val_c = (tensor.to(self.device) for tensor in apply_scalers(val_cache["latents"], val_cache["conditions"], self.scalers))
        self.validation_bank = _validation_bank(
            val_u,
            val_c,
            val_cache["conditions"].to(self.device),
            self.diffusion,
            config["training"]["validation_bank_size"],
            self.generators["validation"],
        )
        self.overfit_fixture = None
        if mode == "overfit":
            count = min(16, len(self.train_u))
            fixture_generator = torch.Generator(device=self.device).manual_seed(seed + 89)
            t = torch.randint(self.diffusion.timesteps, (count,), generator=fixture_generator, device=self.device)
            noise = torch.randn((count, latent_dim), generator=fixture_generator, device=self.device)
            self.overfit_fixture = {"u0": self.train_u[:count], "c": self.train_c[:count], "t": t, "noise": noise}
        self.step_number = 0
        self.history: list[dict[str, Any]] = []
        self.validation_history: list[dict[str, Any]] = []

    def fixed_loss(self) -> float:
        """Measure the unchanged, at-most-16-row overfit corruption fixture."""
        if self.overfit_fixture is None:
            raise ValueError("Fixed-corruption loss is only defined for overfit mode")
        fixture = self.overfit_fixture
        with torch.no_grad():
            loss = self.diffusion.training_loss(
                self.model, fixture["u0"], fixture["t"], fixture["c"],
                condition_present=torch.ones(len(fixture["u0"]), dtype=torch.bool, device=self.device),
                noise=fixture["noise"],
            )
        return float(loss.item())

    def update(self) -> float:
        """Take one optimizer step using the mode's dedicated RNG stream."""
        self.model.train()
        training_config = self.config["training"]
        if self.mode == "overfit":
            assert self.overfit_fixture is not None
            fixture = self.overfit_fixture
            clean, conditions, t, noise = (fixture[key] for key in ("u0", "c", "t", "noise"))
            present = torch.ones(len(clean), dtype=torch.bool, device=self.device)
        else:
            batch_indices = torch.randint(len(self.train_u), (training_config["batch_size"],), device=self.device, generator=self.generators["batch"])
            clean, conditions = self.train_u[batch_indices], self.train_c[batch_indices]
            t = torch.randint(self.diffusion.timesteps, (len(clean),), device=self.device, generator=self.generators["train"])
            noise = torch.randn(clean.shape, device=self.device, generator=self.generators["train"])
            present = torch.rand(len(clean), device=self.device, generator=self.generators["train"]) >= training_config["condition_dropout"]
        self.optimizer.zero_grad(set_to_none=True)
        loss = self.diffusion.training_loss(self.model, clean, t, conditions, condition_present=present, noise=noise)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), training_config["gradient_clip"], error_if_nonfinite=True)
        self.optimizer.step()
        self.step_number += 1
        value = float(loss.item())
        self.history.append({"step": self.step_number, "train_mse": value, "pre_clip_gradient_norm": float(gradient_norm.item())})
        return value

    def evaluate(self) -> dict[str, Any]:
        result = fixed_validation_metrics(self.model, self.diffusion, self.validation_bank)
        result["step"] = self.step_number
        return result

    def checkpoint_payload(self) -> dict[str, Any]:
        """Capture weights, optimizer, scaler, validation bank, and RNG states."""
        return {
            "format_version": CHECKPOINT_VERSION,
            "compatibility": self.compatibility,
            "compatibility_hash": self.compatibility_hash,
            "model_state": self.model.state_dict(),
            "diffusion_state": self.diffusion.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scalers": self.scalers,
            "step": self.step_number,
            "history": self.history,
            "validation_history": self.validation_history,
            "validation_bank": self.validation_bank,
            "overfit_fixture": self.overfit_fixture,
            "rng": {
                "streams": {name: generator.get_state() for name, generator in self.generators.items()},
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else [],
                "python": random.getstate(),
                "numpy": _numpy_state(),
            },
            "saved_device_type": self.device.type,
        }

    def save(self, path: Path) -> None:
        _atomic_torch_save(self.checkpoint_payload(), path)

    def resume(self, path: Path) -> None:
        """Resume only when cache, model, schedule, scaler, and device agree."""
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["format_version"] != CHECKPOINT_VERSION:
            raise ValueError("Unsupported training checkpoint version")
        if payload["compatibility_hash"] != _json_hash(payload["compatibility"]):
            raise ValueError("Checkpoint compatibility metadata is corrupt")
        if payload["compatibility"] != self.compatibility:
            raise ValueError("Cache, scaler, model, or training configuration changed")
        if payload["saved_device_type"] != self.device.type:
            raise ValueError("Cross-device training continuation is unvalidated; initialize a fresh run")
        validate_scalers(payload["scalers"])
        if payload["scalers"]["hash"] != self.scalers["hash"]:
            raise ValueError("Checkpoint scaler differs from current train cache")
        self.model.load_state_dict(payload["model_state"], strict=True)
        self.diffusion.load_state_dict(payload["diffusion_state"], strict=True)
        for name, value in GaussianDiffusion(**self.config["diffusion"]).state_dict().items():
            if not torch.equal(payload["diffusion_state"][name], value):
                raise ValueError("Checkpoint noise schedule disagrees with its configuration")
        self.optimizer.load_state_dict(payload["optimizer_state"])
        _move_optimizer(self.optimizer, self.device)
        self.step_number = int(payload["step"])
        self.history = payload["history"]
        self.validation_history = payload["validation_history"]
        self.validation_bank = {key: value.to(self.device) for key, value in payload["validation_bank"].items()}
        self.overfit_fixture = (
            None if payload["overfit_fixture"] is None else {key: value.to(self.device) for key, value in payload["overfit_fixture"].items()}
        )
        for name, generator in self.generators.items():
            generator.set_state(payload["rng"]["streams"][name])
        torch.set_rng_state(payload["rng"]["torch_cpu"])
        if self.device.type == "cuda":
            torch.cuda.set_rng_state_all(payload["rng"]["torch_cuda"])
        random.setstate(payload["rng"]["python"])
        _restore_numpy_state(payload["rng"]["numpy"])

    def run(self, target_step: int, checkpoint_path: Path) -> dict[str, Any]:
        """Advance to an absolute update count and save resumable checkpoints."""
        cap = min(self.config["training"]["max_steps"], 200 if self.mode == "overfit" else math.inf)
        if not self.step_number <= target_step <= cap:
            raise ValueError("Target step must be between the current step and configured cap")
        initial_step = self.step_number
        started = time.perf_counter()
        before = self.fixed_loss() if self.mode == "overfit" and initial_step == 0 else None
        if self.mode == "smoke" and initial_step == 0:
            self.validation_history.append(self.evaluate())
        frequency = self.config["training"]["validation_frequency"]
        for _ in range(initial_step, target_step):
            self.update()
            if self.mode == "smoke" and (self.step_number % frequency == 0 or self.step_number == target_step):
                self.validation_history.append(self.evaluate())
                self.save(checkpoint_path)
        self.save(checkpoint_path)
        elapsed = time.perf_counter() - started
        return {
            "mode": self.mode,
            "completed_steps": self.step_number,
            "steps_this_process": self.step_number - initial_step,
            "elapsed_seconds": elapsed,
            "updates_per_second": (self.step_number - initial_step) / elapsed if elapsed else None,
            "fixed_corruption_mse_before": before,
            "fixed_corruption_mse_after": self.fixed_loss() if self.mode == "overfit" else None,
            "validation": self.validation_history,
            "last_training_step": self.history[-1] if self.history else None,
            "parameter_count": sum(parameter.numel() for parameter in self.model.parameters()),
            "latent_dim": self.train_u.shape[1],
            "training_tensor_shape": list(self.train_u.shape),
            "validation_bank_shape": list(self.validation_bank["u0"].shape),
            "terminal_alpha_bar": float(self.diffusion.alpha_bars[-1].item()),
            "compatibility_hash": self.compatibility_hash,
            "scaler_hash": self.scalers["hash"],
        }


def load_frozen_model(checkpoint_path: Path, device: str = "cpu") -> tuple[ConditionalDenoiser, GaussianDiffusion, dict, dict]:
    """Safely load inference weights; caller must verify source/policy hashes."""
    target = _device(device)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if payload["format_version"] != CHECKPOINT_VERSION:
        raise ValueError("Unsupported checkpoint version")
    compatibility = payload["compatibility"]
    if payload["compatibility_hash"] != _json_hash(compatibility):
        raise ValueError("Checkpoint metadata hash mismatch")
    scalers = payload["scalers"]
    validate_scalers(scalers)
    if compatibility["scaler_hash"] != scalers["hash"]:
        raise ValueError("Checkpoint scaler hash mismatch")
    model = ConditionalDenoiser(**compatibility["model"])
    diffusion = GaussianDiffusion(**compatibility["diffusion"])
    model.load_state_dict(payload["model_state"], strict=True)
    for name, value in diffusion.state_dict().items():
        if not torch.equal(payload["diffusion_state"][name], value):
            raise ValueError("Checkpoint noise schedule disagrees with its configuration")
    diffusion.load_state_dict(payload["diffusion_state"], strict=True)
    model.to(target).eval().requires_grad_(False)
    diffusion.to(target).eval()
    return model, diffusion, scalers, {
        "step": int(payload["step"]),
        "compatibility": compatibility,
        "compatibility_hash": payload["compatibility_hash"],
    }
