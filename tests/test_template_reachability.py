from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
import torch


WORKSPACE = Path(__file__).resolve().parents[1]
SCRIPT = WORKSPACE / "scripts" / "audit_template_reachability.py"
DATA_ROOT = WORKSPACE / "data" / "processed" / "cmr_reconstructed"
AUDIT_CSV = DATA_ROOT / "audit.csv"
CIF_ROOT = DATA_ROOT / "cifs" / "val"
TRAIN_CSV = DATA_ROOT / "train.csv"
TRAIN_CIF_ROOT = DATA_ROOT / "cifs" / "train"
CHECKPOINT = (
    WORKSPACE
    / "MEIDNet-main"
    / "checkpoints"
    / "dual_autoencoder_clip_earlyfusion_propertyaware_2k.pth"
)

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(os.environ.get("TEMP", ".")) / "meidnet-template-pytest"),
)


def load_module():
    spec = importlib.util.spec_from_file_location("audit_template_reachability", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reachability = load_module()
HAS_ASSETS = (
    AUDIT_CSV.is_file()
    and CIF_ROOT.is_dir()
    and TRAIN_CSV.is_file()
    and TRAIN_CIF_ROOT.is_dir()
    and CHECKPOINT.is_file()
)


class CapturingDecoder:
    def __init__(self) -> None:
        self.calls: list[dict[str, torch.Tensor | None]] = []

    def __call__(self, latent, input_coords=None, center=None, species_mask=None):
        self.calls.append(
            {
                "input_coords": input_coords.detach().clone(),
                "center": center.detach().clone(),
                "species_mask": species_mask,
            }
        )
        batch = latent.shape[0]
        return (
            torch.zeros(batch, 6),
            torch.ones(batch, 20, 20),
            torch.zeros(batch, 20, 118),
            torch.zeros(batch, 20, 3),
        )


class CapturingModel:
    def __init__(self) -> None:
        self.crystal_decoder = CapturingDecoder()
        self.property_decoder = lambda latent: torch.zeros(latent.shape[0], 2)


def fixed_template() -> tuple[torch.Tensor, torch.Tensor]:
    coordinates = torch.zeros(20, 3)
    coordinates[:5] = reachability.generation.TEMPLATE[[0, 1, 3, 4, 2]]
    return coordinates, coordinates[:5].mean(dim=0)


def test_fixed_template_has_training_slot_order_padding_and_occupied_center() -> None:
    coordinates, center = fixed_template()

    assert coordinates.shape == (20, 3)
    assert coordinates[:5].tolist() == [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
        [0.0, 0.5, 0.5],
    ]
    assert torch.count_nonzero(coordinates[5:]) == 0
    assert center.tolist() == pytest.approx([0.3, 0.3, 0.3])
    assert coordinates.mean(dim=0).tolist() == pytest.approx([0.075, 0.075, 0.075])
    assert not torch.allclose(center, coordinates.mean(dim=0))


def test_adapter_zero_mode_delegates_and_template_mode_uses_one_global_tensor() -> None:
    model = CapturingModel()
    coordinates, center = fixed_template()
    latent = torch.randn(2, 128)
    mask = torch.ones(20, 118, dtype=torch.bool)

    zero = reachability.GeometryInputAdapter(model, "zero", coordinates, center)
    template = reachability.GeometryInputAdapter(model, "template", coordinates, center)
    reachability.audit.zero_reference_decode(zero, latent, mask)
    reachability.audit.zero_reference_decode(template, latent, mask)

    zero_call, template_call = model.crystal_decoder.calls
    assert torch.count_nonzero(zero_call["input_coords"]) == 0
    assert torch.count_nonzero(zero_call["center"]) == 0
    assert torch.equal(template_call["input_coords"][0], coordinates)
    assert torch.equal(template_call["input_coords"][0], template_call["input_coords"][1])
    assert torch.equal(template_call["center"][0], center)
    assert torch.equal(template_call["center"][0], template_call["center"][1])
    assert template_call["species_mask"] is mask


def test_repository_charge_tables_distinguish_neutrality_from_later_filters() -> None:
    sodium_vanadate = reachability.permitted_charge_tuples("Na", "V", "O")
    calcium_lead = reachability.permitted_charge_tuples("Ca", "Pb", "O")
    cesium_chromium_nitride = reachability.permitted_charge_tuples("Cs", "Cr", "N")

    assert sodium_vanadate["existential_neutrality_from_tables"] is True
    assert calcium_lead["existential_neutrality_from_tables"] is True
    assert cesium_chromium_nitride["existential_neutrality_from_tables"] is False

    calcium_oracle, _ = reachability.oracle_species_record("Ca", "Pb", "O", refine=True)
    assert calcium_oracle["existential_charge_neutrality_from_tables"] is True
    assert calcium_oracle["charge_filter_passed"] is True
    assert calcium_oracle["tolerance_mu_filter_passed"] is False
    assert calcium_oracle["deterministically_exportable"] is False


def test_sampling_reachability_keeps_raw_and_topk_probabilities_separate() -> None:
    context = {"A": "Na", "B": "V", "X": "O", "family": "oxide"}
    unmasked = torch.zeros(1, 20, 118)
    masked = unmasked.clone()
    b_pool = reachability.generation.target_conditioned_B_set(
        0.0,
        "oxide",
        use_target_filter=False,
    )
    for rank, symbol in enumerate(b_pool):
        index = reachability.generation.SPECIES_LIST.index(symbol)
        unmasked[0, 1, index] = float(rank)
        masked[0, 1, index] = float(rank)

    result = reachability.sampling_reachability(
        unmasked,
        masked,
        context,
        temperature=1.25,
        topk=1,
    )

    assert result["source_B_raw_all_species_probability"] > 0
    assert result["source_B_in_actual_topk"] is False
    assert result["source_B_actual_post_topk_probability"] == 0.0
    assert 0.0 <= result["B_actual_post_topk_mass_without_compatible_A"] <= 1.0


@pytest.mark.skipif(not HAS_ASSETS, reason="local training/audit assets are absent")
def test_training_only_template_selection_is_deterministic() -> None:
    coordinates, convention = reachability.derive_training_template_convention(
        TRAIN_CSV,
        TRAIN_CIF_ROOT,
    )

    assert convention["eligible_training_rows"] == 224
    assert convention["role_order_A_B_X_X_X_count"] == 224
    assert convention["selected_slot_from_generation_template"] == [0, 1, 3, 4, 2]
    assert torch.equal(coordinates[:5], reachability.generation.TEMPLATE[[0, 1, 3, 4, 2]])


@pytest.mark.skipif(not HAS_ASSETS, reason="local audit assets or checkpoint are absent")
def test_real_zero_adapter_trace_and_template_baseline_parity() -> None:
    frame, loaded = reachability.audit.load_audit_inputs(AUDIT_CSV, CIF_ROOT)
    checkpoint, _ = reachability.audit.checkpoint_summary(CHECKPOINT)
    main_model, _ = reachability.audit.instantiate_model(checkpoint)
    generation_model, _ = reachability.audit.load_generation_model(checkpoint)
    row = frame.iloc[0]
    material_id = str(row["material_id"])
    sample = loaded[material_id]
    paths, _, _, _, _ = reachability.audit.latent_bundle(
        main_model,
        sample["crystal_vec"].unsqueeze(0),
        sample["heat_all"].reshape(1),
        sample["dir_gap"].reshape(1),
        material_id,
    )
    context = reachability.audit.source_context(row)
    mask = reachability.generation.build_species_mask(context["allowed_anions"])
    coordinates, center = fixed_template()
    zero_adapter = reachability.GeometryInputAdapter(
        generation_model,
        "zero",
        coordinates,
        center,
    )
    template_adapter = reachability.GeometryInputAdapter(
        generation_model,
        "template",
        coordinates,
        center,
    )
    seed = reachability.generation.seed_from_target(
        0,
        float(sample["dir_gap"]),
        float(sample["heat_all"]),
    )

    original_structure, original_trace = reachability.audit.traced_generation_decode(
        generation_model,
        paths["joint"],
        context["allowed_anions"],
        mask,
        np.random.RandomState(seed),
        float(sample["dir_gap"]),
        context["family"],
        1.25,
        12,
        2,
        refine=False,
    )
    adapter_structure, adapter_trace = reachability.audit.traced_generation_decode(
        zero_adapter,
        paths["joint"],
        context["allowed_anions"],
        mask,
        np.random.RandomState(seed),
        float(sample["dir_gap"]),
        context["family"],
        1.25,
        12,
        2,
        refine=False,
    )
    assert reachability.trace_equal(
        original_structure,
        original_trace,
        adapter_structure,
        adapter_trace,
    )

    template_structure, template_trace = reachability.audit.traced_generation_decode(
        template_adapter,
        paths["joint"],
        context["allowed_anions"],
        mask,
        np.random.RandomState(seed),
        float(sample["dir_gap"]),
        context["family"],
        1.25,
        12,
        2,
        refine=False,
    )
    baseline_structure, baseline_info, gap, heat = reachability.audit.baseline_generation_decode(
        template_adapter,
        paths["joint"],
        context["allowed_anions"],
        mask,
        seed,
        float(sample["dir_gap"]),
        context["family"],
        1.25,
        12,
        2,
        refine=False,
    )
    assert reachability.audit.endpoint_matches_baseline(
        template_structure,
        template_trace,
        baseline_structure,
        baseline_info,
        gap,
        heat,
    )
