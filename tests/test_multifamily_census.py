from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from ase import Atoms


ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "scripts"))

import multifamily_census as census  # noqa: E402


def row(split: str, symbols: list[str], source_anion: str) -> dict[str, str]:
    return {
        "split": split,
        "material_id": "fixture_" + split + "_" + source_anion,
        "source_id": "1",
        "source_unique_id": "fixture",
        "composition_group": "fixture-group",
        "source_formula": "fixture",
        "source_A_ion": symbols[0],
        "source_B_ion": symbols[1],
        "source_anion": source_anion,
        "source_site_symbols": json.dumps(symbols),
        "meidnet_site_symbols": json.dumps(symbols),
        "site_permutation": json.dumps([0, 1, 2, 3, 4]),
        "heat_all": "0.0",
        "dir_gap": "0.0",
    }


def atoms(symbols: list[str], positions: list[tuple[float, float, float]] | None = None) -> Atoms:
    if positions is None:
        positions = [
            (0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0, 0.5),
            (0.5, 0.5, 0), (0, 0.5, 0.5),
        ]
    return Atoms(symbols, scaled_positions=positions, cell=np.eye(3) * 4, pbc=True)


def test_family_categories_and_exact_keys_do_not_collapse_mixed_sites() -> None:
    assert census.classify_family(("O", "O", "O"))[0] == "oxide"
    assert census.classify_family(("O", "N", "O"))[0] == "oxynitride"
    assert census.classify_family(("O", "F", "N"))[0] == "other_mixed"
    assert census.classify_family(("F", "Cl", "F"))[0] == "mixed_halide"
    assert census.anion_key(("N", "O", "O")) == "O2N"
    assert census.anion_key(("O", "N", "O")) == "O2N"


def test_training_roles_only_define_global_vocabulary() -> None:
    training = row("train", ["Ti", "Ti", "O", "O", "N"], "O2N")
    validation = row("val", ["Se", "Ti", "Se", "Se", "Se"], "Se3")
    support = census.training_support([training, validation])
    assert support["V_A"] == {"Ti"}
    assert support["V_B"] == {"Ti"}
    assert support["V_X"] == {"O", "N"}
    assert "Se" not in support["V_A"] | support["V_X"]
    assert support["anion_counts"] == {"O2N": 1}


def test_verified_mixed_roles_and_A_equals_B_are_in_expanded_scope() -> None:
    training = row("train", ["Ti", "Ti", "O", "O", "N"], "O2N")
    support = census.training_support([training])
    roles = census.source_roles(training)
    assert roles["role_status"] == "VERIFIED"
    assert roles["X"] == ("O", "O", "N")
    finding = census.census_entry(training, atoms(["Ti", "Ti", "O", "O", "N"]), support, {})
    assert finding["A_equals_B"]
    assert finding["MF_P4_source_eligible"]
    assert not finding["MF_P3_source_eligible"]
    assert finding["legacy_oracle_status"] == "OUTSIDE_LEGACY_SCOPE"
    assert finding["MF_P2_to_P4_learned_exportability"] == "NOT_EVALUATED"
    assert finding["MF_P3_waterfall_status"] == "shared_X_required"
    assert finding["MF_P4_waterfall_status"] == "eligible_source"


def test_minimum_source_distance_includes_periodic_boundary() -> None:
    structure = atoms(
        ["Ba", "Ti", "O", "O", "O"],
        [(0.01, 0, 0), (0.99, 0, 0), (0.5, 0, 0.5), (0.5, 0.5, 0), (0, 0.5, 0.5)],
    )
    result = census.source_geometry(structure)
    assert np.isclose(result["source_min_periodic_distance_angstrom"], 0.08)
    assert not result["source_minimal_valid"]
    assert "periodic_distance_below_0p8_angstrom" in result["source_geometry_reasons"]


def test_role_ambiguity_remains_visible() -> None:
    entry = row("train", ["Ba", "Ti", "O", "O", "O"], "O3")
    entry["site_permutation"] = json.dumps([1, 0, 2, 3, 4])
    roles = census.source_roles(entry)
    assert roles["role_status"] == "AMBIGUOUS"
    assert "parser_species_mismatch" in roles["role_reason"]


def test_missing_label_precedes_scope_in_fixed_waterfall() -> None:
    training = row("train", ["Ti", "Ti", "O", "O", "N"], "O2N")
    missing = dict(training, split="val", material_id="fixture_missing", dir_gap="")
    finding = census.census_entry(
        missing, atoms(["Ti", "Ti", "O", "O", "N"]), census.training_support([training]), {}
    )
    assert not finding["paired_labels_available"]
    assert not finding["MF_P4_source_eligible"]
    assert finding["MF_P0_waterfall_status"] == "missing_paired_labels"
    assert finding["MF_P4_waterfall_status"] == "missing_paired_labels"
