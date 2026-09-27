from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "smoke_forward.py"
AUDIT_CSV = PROJECT_ROOT / "data" / "processed" / "cmr_reconstructed" / "audit.csv"
CIF_ROOT = PROJECT_ROOT / "data" / "processed" / "cmr_reconstructed" / "cifs" / "val"
CHECKPOINT = (
    PROJECT_ROOT
    / "MEIDNet-main"
    / "checkpoints"
    / "dual_autoencoder_clip_earlyfusion_propertyaware_2k.pth"
)


def load_module():
    spec = importlib.util.spec_from_file_location("smoke_forward", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


smoke_forward = load_module()


def test_reference_free_inputs_exclude_reference_coordinates_and_center():
    coordinates = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
    center = torch.ones(2, 3)

    zero_coordinates, zero_center = smoke_forward.reference_free_inputs(coordinates, center)

    assert zero_coordinates.shape == coordinates.shape
    assert zero_center.shape == center.shape
    assert torch.count_nonzero(zero_coordinates) == 0
    assert torch.count_nonzero(zero_center) == 0
    assert zero_coordinates.data_ptr() != coordinates.data_ptr()
    assert zero_center.data_ptr() != center.data_ptr()

    captured = {}

    def decoder(z_joint, **kwargs):
        captured.update(kwargs)
        return z_joint

    latent = torch.ones(2, 8)
    output, passed_coordinates, passed_center = smoke_forward.decode_reference_free(
        decoder, latent, coordinates, center
    )
    assert output is latent
    assert set(captured) == {"input_coords", "center"}
    assert captured["input_coords"] is passed_coordinates
    assert captured["center"] is passed_center
    assert torch.count_nonzero(passed_coordinates) == 0
    assert torch.count_nonzero(passed_center) == 0


def test_difference_summary_requires_equal_shapes_and_measures_changes():
    left = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    right = torch.tensor([[0.0, 2.0], [2.0, 1.0]])

    result = smoke_forward.difference_summary(left, right)

    assert result["shape_compatible"] is True
    assert result["exactly_equal"] is False
    assert result["all_finite"] is True
    assert result["max_abs"] == 2.0
    assert result["mean_abs"] == 0.75

    mismatch = smoke_forward.difference_summary(left, right[:, :1])
    assert mismatch["shape_compatible"] is False


@pytest.mark.skipif(
    not (AUDIT_CSV.is_file() and CIF_ROOT.is_dir() and CHECKPOINT.is_file()),
    reason="local audit CSV/CIFs or official checkpoint are absent",
)
def test_one_real_example_has_finite_paired_forward_shapes():
    report = smoke_forward.analyse_forward(AUDIT_CSV, CIF_ROOT, CHECKPOINT, batch_size=1)

    assert report["execution_status"] == "passed_frozen_cpu_forward"
    assert report["inputs"]["batch_size"] == 1
    assert report["usual_full_forward"]["all_outputs_finite"] is True
    assert report["latents"]["z_crystal_raw"]["shape"] == [1, 128]
    assert report["latents"]["z_crystal_common"]["shape"] == [1, 128]
    assert report["latents"]["z_joint_not_renormalized"]["shape"] == [1, 128]
    assert report["usual_full_forward"]["outputs"]["species_logits"]["shape"] == [1, 20, 118]
    assert report["paired_decoder_interface"]["B_output_summaries"]["coordinates"]["shape"] == [1, 20, 3]
    assert report["paired_decoder_interface"]["B_output_summaries"]["coordinates"]["all_finite"] is True
    assert report["paired_decoder_interface"]["per_example"][0]["material_id"]
    assert report["actual_generation_implementation"]["strict_checkpoint_load"] is True
