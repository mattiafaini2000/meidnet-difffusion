"""Small synthetic checks for the multifamily audit's accounting interfaces."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_multifamily_decoder.py"
spec = importlib.util.spec_from_file_location("audit_multifamily_decoder", SCRIPT)
assert spec and spec.loader
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_mixed_x_multiset_sums_distinct_permutations():
    heads = [
        {"O": 0.6, "N": 0.3, "F": 0.1},
        {"O": 0.2, "N": 0.5, "F": 0.3},
        {"O": 0.4, "N": 0.1, "F": 0.5},
    ]
    expected_oon = 0.6 * 0.2 * 0.1 + 0.6 * 0.5 * 0.4 + 0.3 * 0.2 * 0.4
    assert audit.multiset_probability(heads, ("O", "O", "N")) == pytest.approx(expected_oon)
    assert audit.multiset_probability(heads, ("N", "O", "O")) == pytest.approx(expected_oon)
    assert audit.multiset_probability(heads, ("O", "N", "F")) == pytest.approx(
        sum(
            heads[0][x] * heads[1][y] * heads[2][z]
            for x, y, z in {("O", "N", "F"), ("O", "F", "N"),
                            ("N", "O", "F"), ("N", "F", "O"),
                            ("F", "O", "N"), ("F", "N", "O")}
        )
    )


def test_validation_panel_and_loader_reject_test_ids(tmp_path):
    with pytest.raises(ValueError, match="non-validation"):
        audit.panel.select_validation_panels([{
            "material_id": "cmr_test", "composition_group": "O:3|Na:1|V:1",
            "anion_key": "O3", "dir_gap": 0.0, "split": "test",
        }])

    # load_selected checks membership before invoking the CIF loader.
    (tmp_path / "val.csv").write_text(
        "material_id,composition_group,dir_gap,heat_all\ncmr_val,g,0,0\n", encoding="utf-8"
    )
    with pytest.raises(AssertionError, match="Selected validation IDs missing"):
        audit.load_selected(tmp_path, ["cmr_test"])


def test_universal_input_hash_is_frozen(tmp_path, monkeypatch):
    coordinates, center, manifest = audit.template_input()
    assert tuple(coordinates.shape) == (20, 3)
    assert tuple(center.shape) == (3,)
    assert manifest["coordinates_sha256"] == audit.EXPECTED_COORD_HASH
    assert manifest["center_sha256"] == audit.EXPECTED_CENTER_HASH

    modified = json.loads(audit.INPUT_MANIFEST.read_text(encoding="utf-8"))
    modified["coordinates"][0][0] = 0.125
    path = tmp_path / "altered_template.json"
    path.write_text(json.dumps(modified), encoding="utf-8")
    monkeypatch.setattr(audit, "INPUT_MANIFEST", path)
    with pytest.raises(AssertionError):
        audit.template_input()


def ledger_rows(material_id="cmr_a", eligible=True):
    rows = []
    for policy in ("MF_P1", "MF_P2"):
        for attempt in range(1, 13):
            rows.append({
                "material_id": material_id,
                "composition_group": f"group_{material_id}",
                "latent_arm": "joint",
                "seed": 0,
                "attempt": attempt,
                "policy": policy,
                "anion_setting": "DECLARED_DOMAIN",
                "source_in_policy_scope": eligible,
                "sampled_roles": ("Na", "V", "O", "O", "O"),
                "policy_accepted": attempt == 1 and policy == "MF_P2",
                "source_structure_match_scale_false": False,
            })
    return rows


def test_p1_p2_replay_and_fixed_twelve_draw_budget():
    rows = ledger_rows()
    result = audit.assert_replay_and_budget(rows)
    assert result["P1_P2_paired_proposals"] == 12
    assert result["every_budget_has_12_draws"]

    altered = [dict(row) for row in rows]
    altered[-1]["sampled_roles"] = ("K", "V", "O", "O", "O")
    with pytest.raises(AssertionError, match="Proposal replay"):
        audit.assert_replay_and_budget(altered)
    with pytest.raises(AssertionError, match="Proposal replay/budget"):
        audit.assert_replay_and_budget(rows[:-1])


def test_policy_pair_uses_shared_eligible_references_and_undefined_zero_denominator():
    eligible = ledger_rows("cmr_a")
    ineligible = ledger_rows("cmr_b")
    for row in ineligible:
        if row["policy"] == "MF_P2":
            row["source_in_policy_scope"] = False
    compared = audit.policy_pair_comparisons(eligible + ineligible)
    pair = next(row for row in compared if row["comparison"] == "same_proposal_output_cell"
                and row["latent_arm"] == "joint")
    assert pair["N_references"] == 1
    assert pair["N_paired_seed_conditions"] == 1
    assert pair["right_minus_left_accepted_per_attempt"] == pytest.approx(1 / 12)

    empty = audit.policy_pair_comparisons(ineligible)
    pair = next(row for row in empty if row["comparison"] == "same_proposal_output_cell"
                and row["latent_arm"] == "joint")
    assert pair["N_references"] == 0
    assert pair["right_minus_left_accepted_per_attempt"] is None
    assert pair["bootstrap_95_group_interval"] is None


def test_full_composition_keys_keep_all_five_species():
    first = ("Na", "V", "O", "O", "N")
    rearranged = ("V", "Na", "N", "O", "O")
    different_x = ("Na", "V", "O", "F", "N")
    same_ab_different_x = ("Na", "V", "O", "O", "O")
    assert audit.counter_key(first) == audit.counter_key(rearranged)
    assert audit.counter_key(first) != audit.counter_key(different_x)
    assert audit.counter_key(first) != audit.counter_key(same_ab_different_x)
    assert audit.counter_key(("Na", "Na", "O", "O", "N")) == "N:1|Na:2|O:2"


def test_generated_training_presence_is_not_a_filter_or_novelty_claim():
    training = {
        "anion_counts": {"O2N": 3},
        "composition_counts": {"N:1|Na:1|O:2|V:1": 1},
    }
    assert audit.sampled_training_presence(("Na", "V", "N", "O", "O"), training) == (True, True)
    assert audit.sampled_training_presence(("K", "V", "N", "O", "O"), training) == (True, False)
    assert audit.sampled_training_presence(("K", "V", "N", "N", "O"), training) == (False, False)
    assert audit.sampled_training_presence(None, training) == (None, None)
