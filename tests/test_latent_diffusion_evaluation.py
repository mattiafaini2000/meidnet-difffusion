"""Asset-free tests for bounded target selection and attempt accounting."""

import json

import pytest
import torch

from latent_diffusion.evaluation import (
    choose_training_targets,
    full_composition_key,
    nearest_condition_indices,
    stable_seed,
    summarize_attempts,
)


def test_targets_cover_available_anion_keys_and_positive_gap_without_duplicate_pairs():
    rows = [
        {"material_id": f"a{i}", "source_anion_key": key, "heat_all": float(i), "dir_gap": 0.0}
        for i, key in enumerate(("O3", "N3", "O2N", "O2F"))
    ]
    rows += [
        {"material_id": "p", "source_anion_key": "N3", "heat_all": 0.3, "dir_gap": 1.2},
        {"material_id": "duplicate", "source_anion_key": "O3", "heat_all": 0.0, "dir_gap": 0.0},
    ]
    selected = choose_training_targets(list(reversed(rows)), limit=5)
    assert len(selected) == 5
    assert {row["source_anion_key"] for row in selected} == {"O3", "N3", "O2N", "O2F"}
    assert any(row["dir_gap"] > 0 for row in selected)
    assert len({(row["heat_all"], row["dir_gap"]) for row in selected}) == 5
    assert choose_training_targets(rows, limit=5) == selected


def test_nearest_ties_use_ids_and_full_composition_keeps_anions():
    values = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    neighbors = nearest_condition_indices(values, torch.tensor([1.0, 0.0]),
                                          torch.ones(2),
                                          ["z", "b", "a"], k=3)
    assert [index for index, _ in neighbors] == [2, 1, 0]
    assert full_composition_key(("Na", "V", "O", "O", "N")) != full_composition_key(
        ("Na", "V", "O", "O", "O"))
    assert stable_seed("target", 1) == stable_seed("target", 1)
    assert stable_seed("target", 1) != stable_seed("target", 2)


def test_attempt_summary_keeps_rejections_and_duplicates():
    rows = [
        {"target_id": "x", "method": "diffusion", "accepted": True,
         "full_composition": "N:3|Na:1|V:1", "rejection_reason": None},
        {"target_id": "x", "method": "diffusion", "accepted": True,
         "full_composition": "N:3|Na:1|V:1", "rejection_reason": None},
        {"target_id": "x", "method": "diffusion", "accepted": False,
         "full_composition": None, "rejection_reason": "INVALID_LEARNED_LATTICE"},
    ]
    result = summarize_attempts(rows)[0]
    assert result["attempts"] == 3
    assert result["accepted"] == 2
    assert result["unique_accepted_compositions"] == 1
    assert result["largest_composition_share"] == 1.0
    assert result["rejection_reasons"] == {"INVALID_LEARNED_LATTICE": 1}


def test_generation_rejects_test_manifest_before_loading_tensor_snapshot(tmp_path):
    from scripts.evaluate_latent_diffusion import require_cache_split

    (tmp_path / "manifest.json").write_text(json.dumps({"split": "test"}), encoding="utf-8")
    with pytest.raises(ValueError, match="no test data opened"):
        require_cache_split(tmp_path, "val")
