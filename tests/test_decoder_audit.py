from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch


WORKSPACE = Path(__file__).resolve().parents[1]
SCRIPT = WORKSPACE / "scripts" / "audit_decoder.py"
DATA_ROOT = WORKSPACE / "data" / "processed" / "cmr_reconstructed"
AUDIT_CSV = DATA_ROOT / "audit.csv"
CIF_ROOT = DATA_ROOT / "cifs" / "val"
CHECKPOINT = (
    WORKSPACE
    / "MEIDNet-main"
    / "checkpoints"
    / "dual_autoencoder_clip_earlyfusion_propertyaware_2k.pth"
)
SUMMARY_JSON = WORKSPACE / "reports" / "decoder_audit_summary.json"

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TEMP", ".")) / "meidnet-pytest-mpl"))


def load_module():
    spec = importlib.util.spec_from_file_location("audit_decoder", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit_decoder = load_module()
HAS_ASSETS = AUDIT_CSV.is_file() and CIF_ROOT.is_dir() and CHECKPOINT.is_file()


class CapturingDecoder:
    def __init__(self) -> None:
        self.kwargs = None

    def __call__(self, latent, **kwargs):
        self.kwargs = kwargs
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


def test_reference_free_decoder_accepts_no_per_example_reference_inputs() -> None:
    model = CapturingModel()
    latent = torch.randn(2, 128)
    mask = torch.ones(20, 118, dtype=torch.bool)

    outputs = audit_decoder.zero_reference_decode(model, latent, mask)

    assert [list(output.shape) for output in outputs] == [
        [2, 6],
        [2, 20, 20],
        [2, 20, 118],
        [2, 20, 3],
    ]
    assert model.crystal_decoder.kwargs is not None
    assert set(model.crystal_decoder.kwargs) == {"input_coords", "center", "species_mask"}
    assert torch.count_nonzero(model.crystal_decoder.kwargs["input_coords"]) == 0
    assert torch.count_nonzero(model.crystal_decoder.kwargs["center"]) == 0
    assert model.crystal_decoder.kwargs["species_mask"] is mask


def test_paired_difference_and_boolean_helpers() -> None:
    tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    same = audit_decoder.difference_stats(tensor, tensor.clone())
    changed = audit_decoder.difference_stats(tensor, tensor + 1)

    assert same["exactly_equal"] is True
    assert same["maximum_absolute"] == 0.0
    assert changed["exactly_equal"] is False
    assert changed["maximum_absolute"] == 1.0
    assert audit_decoder.parse_bool("False") is False
    assert audit_decoder.parse_bool("true") is True
    with pytest.raises(ValueError, match="Invalid Boolean"):
        audit_decoder.parse_bool("unknown")


def test_repeat_summary_tracks_each_head_with_declared_tolerance() -> None:
    first = (
        torch.ones(1, 6),
        torch.ones(1, 20, 20),
        torch.ones(1, 20, 118),
        torch.ones(1, 20, 3),
    )
    within = tuple(tensor + 5e-7 for tensor in first)
    outside = list(within)
    outside[2] = first[2] + 1e-3
    summary = audit_decoder.new_repeat_summary()

    audit_decoder.update_repeat_summary(summary, first, within)

    assert summary["call_pairs"] == 1
    assert summary["all_heads_within_tolerance"] is True
    assert all(item["within_tolerance"] for item in summary["heads"].values())

    audit_decoder.update_repeat_summary(summary, first, tuple(outside))

    assert summary["call_pairs"] == 2
    assert summary["all_heads_within_tolerance"] is False
    assert summary["heads"]["species_logits"]["within_tolerance"] is False
    assert summary["heads"]["species_logits"]["maximum_absolute_difference"] == pytest.approx(
        1e-3,
        abs=1e-6,
    )


@pytest.mark.skipif(not SUMMARY_JSON.is_file(), reason="decoder-audit summary is absent")
def test_markdown_renderer_accepts_the_current_summary_schema() -> None:
    summary = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))

    rendered = audit_decoder.markdown_report(summary)

    assert "## Separate conclusions" in rendered
    assert "Stage-C generation implementation by latent arm" in rendered
    assert summary["readiness"]["numerical_execution"] in rendered


def test_information_metrics_exclude_padding_and_use_stable_probabilities() -> None:
    crystal = torch.zeros(1, 6 + 20 * 20 + 20 * 118 + 20 * 3)
    species_offset = 6 + 20 * 20
    true_indices = [
        audit_decoder.SPECIES_LIST.index("K"),
        audit_decoder.SPECIES_LIST.index("W"),
        *[audit_decoder.SPECIES_LIST.index("O")] * 3,
    ]
    for slot, index in enumerate(true_indices):
        crystal[0, species_offset + slot * 118 + index] = 1
    logits = torch.zeros(1, 20, 118)
    for slot, index in enumerate(true_indices):
        logits[0, slot, index] = 2
    logits[0, 5:, :] = float("nan")

    result, sites = audit_decoder.species_information_metrics(logits, crystal)

    expected_nll = -torch.log_softmax(torch.tensor([2.0] + [0.0] * 117), dim=0)[0]
    assert result["site_count"] == 5
    assert result["nll_nats_per_site"] == pytest.approx(float(expected_nll))
    assert result["nll_status"] == "finite"
    assert result["top1_accuracy"] == 1.0
    assert len(sites) == 5
    assert len(json.loads(sites[0]["logits_json"])) == 118
    assert len(json.loads(sites[0]["probabilities_json"])) == 118
    assert sum(json.loads(sites[0]["probabilities_json"])) == pytest.approx(1.0)
    assert sites[0]["negative_log_likelihood_status"] == "finite"

    mask = torch.ones(20, 118, dtype=torch.bool)
    mask[0, true_indices[0]] = False
    masked, masked_sites = audit_decoder.species_information_metrics(
        logits.nan_to_num(),
        crystal,
        mask,
        scored_slots=[0, 1],
    )
    assert masked["site_count"] == 2
    assert masked["nll_nats_per_site"] is None
    assert masked["nll_status"] == "positive_infinity_restriction_failure"
    assert masked["nll_is_infinite_due_to_restriction"] is True
    assert masked["restriction_failure_sites"] == 1
    assert masked_sites[0]["negative_log_likelihood"] is None
    assert (
        masked_sites[0]["negative_log_likelihood_status"]
        == "positive_infinity_restriction_failure"
    )
    assert json.loads(masked_sites[0]["probabilities_json"])[true_indices[0]] == 0.0


def test_total_variation_ignores_common_logit_offsets() -> None:
    crystal = torch.zeros(1, 6 + 20 * 20 + 20 * 118 + 20 * 3)
    species_offset = 6 + 20 * 20
    for slot in range(5):
        crystal[0, species_offset + slot * 118 + slot] = 1
    logits = torch.randn(1, 20, 118)

    unchanged = audit_decoder.occupied_site_total_variation(logits, logits + 9.0, crystal)
    changed_logits = logits.clone()
    changed_logits[0, 0, 0] += 5
    changed = audit_decoder.occupied_site_total_variation(logits, changed_logits, crystal)
    ab_only = audit_decoder.occupied_site_total_variation(
        logits,
        changed_logits,
        crystal,
        scored_slots=[0, 1],
    )

    assert unchanged["maximum"] == pytest.approx(0.0, abs=1e-6)
    assert changed["mean"] > 0
    assert len(changed["per_site"]) == 5
    assert len(ab_only["per_site"]) == 2
    assert ab_only["per_site"][0] > 0


def test_final_yield_uses_only_accepted_ab_and_excludes_forced_x_diversity() -> None:
    def result_row(
        material_id: str,
        path: str,
        sampled_a: str,
        sampled_b: str,
        formula: str | None,
        source_ab: bool,
        attempts: int = 2,
    ) -> dict:
        accepted = formula is not None
        return {
            "material_id": material_id,
            "composition_group": material_id,
            "latent_path": path,
            "seed": 0,
            "attempts_used": attempts,
            "constructed_count": int(accepted),
            "accepted": accepted,
            "sampled_A": sampled_a if accepted else None,
            "sampled_B": sampled_b if accepted else None,
            "formula": formula,
            "structure_signature": formula,
            "matcher_structure_cluster": formula,
            "attempt_source_AB_recovery_count": int(source_ab),
            "attempt_source_AB_recovery_rate": int(source_ab) / attempts,
            "passed_source_AB_recovery_count": int(accepted and source_ab),
            "passed_source_AB_recovery_rate": int(accepted and source_ab) / attempts,
            "attempt_source_exact_composition_recovery_count": 0,
            "source_A_recovered": accepted and source_ab,
            "source_B_recovered": accepted and source_ab,
            "source_AB_recovered": accepted and source_ab,
            "source_exact_composition_recovered": False,
            "rejection_stage_counts": "{}",
            "endpoint_matches_baseline": True,
        }

    joint = [
        result_row("m1", "joint", "K", "W", "KWO3", True),
        result_row("m2", "joint", "K", "W", "KWN3", True),
        result_row("m3", "joint", "Na", "Nb", "NaNbO3", True, attempts=4),
    ]
    controls = [
        result_row("m1", "fixed_real", "Rb", "Ta", "RbTaO3", False),
        result_row("m2", "fixed_real", "Rb", "Ta", "RbTaN3", False),
        result_row("m3", "fixed_real", "Rb", "Ta", None, False, attempts=4),
    ]

    summary = audit_decoder.yield_summary(joint)
    assert summary["N_unique_composition"] == 3
    assert summary["N_unique_accepted_AB"] == 2
    assert summary["dominant_accepted_AB_fraction"] == pytest.approx(2 / 3)
    assert summary["passed_source_AB_recovery_count"] == 3
    assert summary["passed_source_AB_recovery_rate"] == pytest.approx(3 / 8)

    by_condition = audit_decoder.condition_level_ab_summary(joint, "joint")
    assert by_condition["conditions_with_accepted_output"] == 3
    assert by_condition["conditions_with_single_accepted_AB"] == 3
    assert by_condition["conditions_with_multiple_accepted_AB"] == 0

    paired = audit_decoder.paired_final_summary(joint + controls, "fixed_real")
    accepted_effect = paired["joint_minus_control_passed_source_AB_recovery_rate"]
    assert accepted_effect["mean"] == pytest.approx((0.5 + 0.5 + 0.25) / 3)
    assert paired["passed_source_AB_recovery_rate_difference_by_seed"]["0"] == pytest.approx(
        accepted_effect["mean"]
    )


@pytest.mark.skipif(not HAS_ASSETS, reason="local audit CSV/CIFs or official checkpoint are absent")
def test_one_real_trace_matches_the_unmodified_generation_function() -> None:
    frame, loaded = audit_decoder.load_audit_inputs(AUDIT_CSV, CIF_ROOT)
    checkpoint, _ = audit_decoder.checkpoint_summary(CHECKPOINT)
    main_model, _ = audit_decoder.instantiate_model(checkpoint)
    generation_model, report = audit_decoder.load_generation_model(checkpoint)
    row = frame.iloc[0]
    material_id = str(row["material_id"])
    sample = loaded[material_id]
    paths, _, _, _, _ = audit_decoder.latent_bundle(
        main_model,
        sample["crystal_vec"].unsqueeze(0),
        sample["heat_all"].reshape(1),
        sample["dir_gap"].reshape(1),
        material_id,
    )
    joint = paths["joint"]
    delta_small = paths["joint_perturb_0p01"] - joint
    delta_large = paths["joint_perturb_0p05"] - joint
    assert torch.linalg.vector_norm(delta_small) / torch.linalg.vector_norm(joint) == pytest.approx(
        0.01, rel=1e-5
    )
    assert torch.linalg.vector_norm(delta_large) / torch.linalg.vector_norm(joint) == pytest.approx(
        0.05, rel=1e-5
    )
    assert torch.nn.functional.cosine_similarity(delta_small, delta_large).item() == pytest.approx(
        1.0, abs=1e-6
    )
    context = audit_decoder.source_context(row)
    mask = audit_decoder.generation.build_species_mask(context["allowed_anions"])
    resolved_seed = audit_decoder.generation.seed_from_target(
        0,
        float(sample["dir_gap"]),
        float(sample["heat_all"]),
    )

    traced_structure, traced = audit_decoder.traced_generation_decode(
        generation_model,
        paths["joint"],
        context["allowed_anions"],
        mask,
        np.random.RandomState(resolved_seed),
        float(sample["dir_gap"]),
        context["family"],
        1.25,
        12,
        12,
        refine=False,
    )
    baseline_structure, baseline_info, baseline_gap, baseline_heat = audit_decoder.baseline_generation_decode(
        generation_model,
        paths["joint"],
        context["allowed_anions"],
        mask,
        resolved_seed,
        float(sample["dir_gap"]),
        context["family"],
        1.25,
        12,
        12,
        refine=False,
    )

    assert report["strict_state_dict_load"] is True
    assert audit_decoder.endpoint_matches_baseline(
        traced_structure,
        traced,
        baseline_structure,
        baseline_info,
        baseline_gap,
        baseline_heat,
    )
    assert traced["attempts_used"] <= 12
    assert len(traced["attempt_records"]) == traced["attempts_used"]
    assert traced["constructed_count"] == sum(
        record["constructed"] for record in traced["attempt_records"]
    )
    assert traced["accepted"] is (traced_structure is not None)
