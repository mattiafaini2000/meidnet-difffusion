"""Focused checks for the additive multifamily policy helpers."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "multifamily_policies.py"
spec = importlib.util.spec_from_file_location("multifamily_policies", MODULE_PATH)
assert spec is not None and spec.loader is not None
policies = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = policies
spec.loader.exec_module(policies)


def heads() -> np.ndarray:
    return np.full((20, 118), -100.0)


def prefer(values: np.ndarray, slot: int, symbol: str, value: float = 100.0) -> None:
    values[slot, policies.legacy.SPECIES_LIST.index(symbol)] = value


def test_legacy_scope_is_explicit_and_p1_p2_replay_without_charge_or_topk() -> None:
    vocab = {"A": (), "B": (), "X": ()}  # expanded vocabulary must not affect legacy pools
    p1 = policies.PolicyConfig("MF_P1", family="oxide")
    p2 = policies.PolicyConfig("MF_P2", family="oxide")
    allowed = policies.allowed_roles(p1, vocab)
    assert set(allowed["A"]) == policies.legacy.A_CATIONS
    assert set(allowed["B"]) == set(policies.legacy.VALENCE_B_BY_FAMILY["oxide"])
    assert allowed["X"] == ("O",)
    values = heads()
    prefer(values, 0, "Cs")
    prefer(values, 1, "Ni")
    for slot in range(2, 5):
        prefer(values, slot, "O")
    first = policies.sample_roles(values, p1, vocab, seed=17)
    second = policies.sample_roles(values, p2, vocab, seed=17)
    assert first["roles"] == second["roles"] == ("Cs", "Ni", "O", "O", "O")
    assert first["sampled"] is True
    assert first["forced_x"] is True
    assert len(first["slot_probabilities"][0]) == len(allowed["A"])
    assert len(first["slot_probabilities"][1]) == len(allowed["B"])
    assert first["selected_probabilities"][0] > 0.99
    with pytest.raises(ValueError, match="supported family"):
        policies.PolicyConfig("MF_P1", family="selenide")
    with pytest.raises(NotImplementedError, match="unchanged legacy"):
        policies.sample_roles(values, policies.PolicyConfig("MF_P0", family="oxide"), vocab, 17)


def test_training_only_global_vocabulary_allows_overlapping_roles_and_mixed_x() -> None:
    train_rows = [("Ti", "Ti", "O", "N", "O"), ("Na", "Ti", "F", "F", "O")]
    vocab = policies.build_training_vocabulary(train_rows)
    assert vocab["A"] == ("Na", "Ti")  # order follows checkpoint element mapping
    assert set(vocab["B"]) == {"Ti"}
    assert set(vocab["X"]) == {"N", "O", "F"}
    assert vocab["counts"]["X"]["O"] == 3
    restricted = policies.build_training_vocabulary(train_rows, species_list=("Ti", "O"))
    assert restricted["A"] == ("Ti",)
    assert restricted["B"] == ("Ti",)
    assert restricted["X"] == ("O",)
    assert len(restricted["unsupported_training_roles"]) == 4
    p3 = policies.PolicyConfig("MF_P3")
    p4 = policies.PolicyConfig("MF_P4")
    mixed = ("Ti", "Ti", "O", "N", "O")
    assert policies.source_scope(mixed, p3, vocab)["eligible"] is False
    assert "OUTSIDE_HOMOGENEOUS_X_SCOPE" in policies.source_scope(mixed, p3, vocab)["reasons"]
    assert policies.source_scope(mixed, p4, vocab)["eligible"] is True
    assert policies.source_scope(("Xe", "Ti", "O", "N", "O"), p4, vocab)["eligible"] is False
    with pytest.raises(ValueError, match="cannot receive"):
        policies.allowed_roles(p4, vocab, declared_anions=("O",))
    domain = policies.PolicyConfig("MF_P4", anion_setting="DECLARED_DOMAIN")
    with pytest.raises(ValueError, match="explicit X support"):
        policies.allowed_roles(domain, vocab)
    assert policies.allowed_roles(domain, vocab, declared_anions=("O",))["X"] == ("O",)


def test_expanded_sampler_keeps_a_equals_b_and_independent_mixed_anions() -> None:
    vocab = {"A": ("Ti",), "B": ("Ti",), "X": ("O", "N", "F")}
    values = heads()
    prefer(values, 0, "Ti")
    prefer(values, 1, "Ti")
    for slot, symbol in zip(range(2, 5), ("O", "N", "F")):
        prefer(values, slot, symbol)
    p4 = policies.PolicyConfig("MF_P4")
    sampled = policies.sample_roles(values, p4, vocab, seed=23)
    assert sampled["roles"] == ("Ti", "Ti", "O", "N", "F")
    assert sampled["forced_x"] is False
    assert all(math.isclose(sum(prob.values()), 1.0) for prob in sampled["slot_probabilities"])
    assert sampled["slot_probabilities"][2] != sampled["slot_probabilities"][3]
    p3 = policies.PolicyConfig("MF_P3")
    homogeneous = policies.sample_roles(values, p3, vocab, seed=23)
    assert homogeneous["roles"][0:2] == ("Ti", "Ti")
    assert len(set(homogeneous["roles"][2:])) == 1


def test_full_support_sampling_reports_mask_and_underflow_without_rescue() -> None:
    vocab = {"A": ("Na", "K"), "B": ("Ti",), "X": ("O",)}
    values = heads()
    prefer(values, 0, "Na", 0.0)
    prefer(values, 1, "Ti", 0.0)
    for slot in range(2, 5):
        prefer(values, slot, "O", 0.0)
    values[0, policies.legacy.SPECIES_LIST.index("K")] = -math.inf
    result = policies.sample_roles(values, policies.PolicyConfig("MF_P3"), vocab, 0)
    assert result["sampled"] is True
    assert result["slot_probabilities"][0]["K"] == 0.0
    assert result["distribution_diagnostics"][0]["masked_minus_infinity"] == 1
    values[0, policies.legacy.SPECIES_LIST.index("Na")] = math.nan
    failed = policies.sample_roles(values, policies.PolicyConfig("MF_P3"), vocab, 0)
    assert failed["sampled"] is False
    assert failed["reason"] == "INVALID_LOGITS_SLOT_0"


def test_output_template_is_legacy_order_and_learned_cell_uses_training_inverse() -> None:
    assert policies.TEMPLATE_FRACTIONAL == tuple(
        tuple(row) for row in policies.legacy.TEMPLATE.tolist()
    )
    assert policies.INPUT_TEMPLATE_20_SLOT_SHA256 != policies.OUTPUT_TEMPLATE_SHA256
    roles = ("Na", "Ti", "O", "F", "F")
    learned = policies.PolicyConfig("MF_P4")
    candidate, details = policies.construct_candidate(
        roles, learned, [0.20, 0.25, 0.30, 0.4, 0.5, 0.6]
    )
    assert details["constructed"] is True
    assert details["raw_lengths_angstrom"] == pytest.approx((4.0, 5.0, 6.0))
    assert details["raw_angles_degrees"] == pytest.approx((72.0, 90.0, 108.0))
    assert details["projected_cubic_length_angstrom"] == pytest.approx(5.0)
    assert candidate is not None
    assert [site.specie.symbol for site in candidate] == list(roles)
    assert candidate.lattice.angles == pytest.approx((90.0, 90.0, 90.0))
    assert np.allclose(candidate.frac_coords, policies.TEMPLATE_FRACTIONAL)
    assert candidate.get_space_group_info()[0] != "Pm-3m"  # mixed decoration breaks chemical cubic symmetry
    invalid, detail = policies.construct_candidate(roles, learned, [-0.1, 0.2, 0.3, .5, .5, .5])
    assert invalid is None and detail["reason"] == "INVALID_LEARNED_LATTICE"
    with pytest.raises(ValueError, match="Unknown policy"):
        policies.PolicyConfig("MF_P5")


def test_radius_control_is_separate_and_minimal_check_uses_explicit_roles() -> None:
    roles = ("Cs", "Ni", "O", "O", "O")
    p1 = policies.PolicyConfig("MF_P1", family="oxide")
    candidate, details = policies.construct_candidate(roles, p1, [math.nan] * 6)
    assert candidate is not None
    assert details["cell_source"] == "legacy_radius_derived"
    support = policies.allowed_roles(p1, {"A": (), "B": (), "X": ()})
    assert policies.minimal_check(candidate, roles, support)["valid"] is True
    wrong = {"A": ("Na",), "B": ("Ni",), "X": ("O",)}
    assert "OUT_OF_SUPPORT_A" in policies.minimal_check(candidate, roles, wrong)["failure_reasons"]


def test_periodic_check_includes_boundary_neighbors_and_nonzero_self_images() -> None:
    roles = ("Na", "Ti", "O", "F", "F")
    support = {"A": ("Na",), "B": ("Ti",), "X": ("O", "F")}
    boundary = Structure(
        Lattice.cubic(5.0), roles,
        [(0.02, 0, 0), (0.98, 0, 0), (0.5, 0.5, 0), (0, 0.5, 0.5), (0.5, 0, 0.5)],
    )
    checked = policies.minimal_check(boundary, roles, support)
    assert checked["minimum_periodic_distance_angstrom"] == pytest.approx(0.2)
    assert "PERIODIC_OVERLAP_BELOW_0P8_ANGSTROM" in checked["failure_reasons"]
    short_axis = Structure(
        Lattice.orthorhombic(5.0, 5.0, 0.7), roles,
        [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0), (0, 0.5, 0), (0.25, 0.25, 0)],
    )
    checked = policies.minimal_check(short_axis, roles, support)
    assert checked["minimum_periodic_distance_angstrom"] == pytest.approx(0.7)
    assert checked["valid"] is False


def test_mixed_multiset_sums_distinct_orders_and_duplicate_key_uses_all_species() -> None:
    slots = [
        {"O": .7, "N": .3},
        {"O": .2, "N": .8},
        {"O": .5, "N": .5},
    ]
    expected = .7 * .2 * .5 + .7 * .8 * .5 + .3 * .2 * .5
    assert policies.mixed_multiset_probability(slots, ("O", "O", "N")) == pytest.approx(expected)
    assert policies.mixed_multiset_log_probability(slots, ("O", "O", "N")) == pytest.approx(math.log(expected))
    assert policies.mixed_multiset_log_probability(slots, ("O", "O", "F")) == -math.inf
    assert policies.full_composition_key(("Na", "Ti", "O", "O", "N")) != policies.full_composition_key(
        ("Na", "Ti", "O", "O", "O")
    )
    assert policies.full_composition_key(("Na", "Ti", "O", "O", "N")) == policies.full_composition_key(
        ("Na", "Ti", "N", "O", "O")
    )
    assert policies.proposal_seed("source", "joint", 0, 1) == policies.proposal_seed(
        "source", "joint", 0, 1
    )
