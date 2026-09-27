"""Asset-free CPU checks for training, fixed validation, and safe continuation."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from latent_diffusion.training import DenoiserTrainer, load_frozen_model


def _cache(split: str, count: int = 12, dim: int = 8) -> dict:
    generator = torch.Generator().manual_seed(103 if split == "train" else 107)
    latents = torch.randn((count, dim), generator=generator)
    conditions = torch.stack((torch.linspace(-0.2, 1.2, count), torch.arange(count).float() % 3), dim=1)
    ids = [f"{split}_{index:03d}" for index in range(count)]
    fingerprint = hashlib.sha256(f"{split}:{count}:{dim}".encode()).hexdigest()
    return {
        "latents": latents,
        "conditions": conditions,
        "ids": ids,
        "manifest": {
            "split": split,
            "fingerprint": fingerprint,
            "selected_ids_sha256": hashlib.sha256("|".join(ids).encode()).hexdigest(),
            "status": "complete",
        },
    }


def _config(max_steps: int = 4) -> dict:
    return {
        "model": {"width": 16, "blocks": 2, "expansion": 2, "context_mode": "film"},
        "diffusion": {"timesteps": 20, "beta_start": 0.0001, "beta_end": 0.02},
        "training": {
            "batch_size": 4,
            "seed": 0,
            "max_steps": max_steps,
            "lr": 0.001,
            "weight_decay": 0.0001,
            "gradient_clip": 1.0,
            "condition_dropout": 0.1,
            "validation_frequency": 2,
            "validation_bank_size": 6,
            "num_workers": 0,
        },
    }


def test_cpu_resume_matches_uninterrupted_updates(tmp_path: Path) -> None:
    torch.set_num_threads(1)
    train, val, config = _cache("train"), _cache("val"), _config()
    straight = DenoiserTrainer(train, val, config)
    straight.run(4, tmp_path / "straight.pt")

    interrupted = DenoiserTrainer(train, val, config)
    interrupted.run(2, tmp_path / "interrupted.pt")
    resumed = DenoiserTrainer(train, val, config)
    resumed.resume(tmp_path / "interrupted.pt")
    resumed.run(4, tmp_path / "resumed.pt")

    assert resumed.step_number == straight.step_number == 4
    assert resumed.history == straight.history
    assert resumed.validation_history == straight.validation_history
    for name, tensor in straight.model.state_dict().items():
        assert torch.equal(tensor, resumed.model.state_dict()[name]), name
    for name in straight.generators:
        assert torch.equal(straight.generators[name].get_state(), resumed.generators[name].get_state())
    left_optimizer = straight.optimizer.state_dict()
    right_optimizer = resumed.optimizer.state_dict()
    assert left_optimizer["param_groups"] == right_optimizer["param_groups"]
    for key in left_optimizer["state"]:
        for field, value in left_optimizer["state"][key].items():
            other = right_optimizer["state"][key][field]
            assert torch.equal(value, other) if isinstance(value, torch.Tensor) else value == other
    assert torch.load(tmp_path / "resumed.pt", weights_only=True)["step"] == 4


def test_fixed_corruption_overfit_decreases_and_uses_no_dropout(tmp_path: Path) -> None:
    torch.set_num_threads(1)
    trainer = DenoiserTrainer(_cache("train", 4), _cache("val", 4), _config(60), mode="overfit")
    result = trainer.run(60, tmp_path / "overfit.pt")
    assert result["training_tensor_shape"] == [4, 8]
    assert result["fixed_corruption_mse_after"] < result["fixed_corruption_mse_before"]
    assert result["completed_steps"] == 60


def test_fixed_validation_bank_and_gap_groups_are_explicit(tmp_path: Path) -> None:
    trainer = DenoiserTrainer(_cache("train"), _cache("val"), _config())
    before = trainer.evaluate()
    trainer.update()
    after = trainer.evaluate()
    assert before["count"] == 30  # six examples, each at five prescribed noise levels
    assert before["zero_predictor_mse"] > 0
    assert len(before["by_level_and_gap"]) == 10
    assert before["mse"] != after["mse"]
    assert torch.equal(trainer.validation_bank["noise"], trainer.validation_bank["noise"].clone())


def test_validation_does_not_advance_training_rng() -> None:
    train, val, config = _cache("train"), _cache("val"), _config()
    inspected = DenoiserTrainer(train, val, config)
    inspected.update()
    inspected.evaluate()
    inspected.update()
    uninspected = DenoiserTrainer(train, val, config)
    uninspected.update()
    uninspected.update()
    assert inspected.history == uninspected.history
    for name, tensor in inspected.model.state_dict().items():
        assert torch.equal(tensor, uninspected.model.state_dict()[name]), name


def test_checkpoint_rejects_changed_cache_or_model_and_inference_is_frozen(tmp_path: Path) -> None:
    train, val, config = _cache("train"), _cache("val"), _config()
    original = DenoiserTrainer(train, val, config)
    checkpoint = tmp_path / "checkpoint.pt"
    original.run(1, checkpoint)
    frozen, diffusion, scalers, metadata = load_frozen_model(checkpoint)
    assert not frozen.training
    assert not any(parameter.requires_grad for parameter in frozen.parameters())
    assert diffusion.timesteps == 20
    assert metadata["compatibility"]["scaler_hash"] == scalers["hash"]
    probe = torch.randn((2, 8))
    t = torch.tensor([0, 10], dtype=torch.long)
    conditions = torch.zeros((2, 2))
    original.model.eval()
    with torch.no_grad():
        assert torch.equal(original.model(probe, t, conditions), frozen(probe, t, conditions))
    changed_val = _cache("val")
    changed_val["manifest"]["fingerprint"] = "different"
    with pytest.raises(ValueError, match="configuration changed"):
        DenoiserTrainer(train, changed_val, config).resume(checkpoint)
    changed_config = _config()
    changed_config["model"]["width"] = 32
    with pytest.raises(ValueError, match="configuration changed"):
        DenoiserTrainer(train, val, changed_config).resume(checkpoint)


def test_test_cache_and_partial_cache_are_rejected() -> None:
    with pytest.raises(ValueError, match="train cache and a validation cache"):
        DenoiserTrainer(_cache("train"), _cache("test"), _config())
    partial = _cache("val")
    partial["manifest"]["status"] = "partial"
    with pytest.raises(ValueError, match="Finish the selected"):
        DenoiserTrainer(_cache("train"), partial, _config())


def test_frozen_load_rejects_tampered_scaler_or_noise_schedule(tmp_path: Path) -> None:
    trainer = DenoiserTrainer(_cache("train"), _cache("val"), _config())
    original = tmp_path / "original.pt"
    trainer.save(original)
    payload = torch.load(original, map_location="cpu", weights_only=True)
    payload["scalers"]["latent"]["mean"][0] += 1
    changed_scaler = tmp_path / "changed_scaler.pt"
    torch.save(payload, changed_scaler)
    with pytest.raises(ValueError, match="Scaler hash mismatch"):
        load_frozen_model(changed_scaler)
    payload = torch.load(original, map_location="cpu", weights_only=True)
    payload["diffusion_state"]["betas"][0] *= 2
    changed_schedule = tmp_path / "changed_schedule.pt"
    torch.save(payload, changed_schedule)
    with pytest.raises(ValueError, match="noise schedule"):
        load_frozen_model(changed_schedule)
