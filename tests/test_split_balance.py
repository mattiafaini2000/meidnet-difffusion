from __future__ import annotations

import sys
from pathlib import Path

import pytest


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(WORKSPACE_ROOT / "scripts"))

import audit_split_balance as balance


DATASET_DIR = WORKSPACE_ROOT / "data" / "processed" / "cmr_reconstructed"
HAS_ASSETS = all((DATASET_DIR / name).is_file() for name in balance.EXPECTED_HASHES)


def test_boolean_parser_does_not_treat_false_string_as_true() -> None:
    assert balance.parse_boolean("True") is True
    assert balance.parse_boolean(" true ") is True
    assert balance.parse_boolean("1") is True
    assert balance.parse_boolean("False") is False
    assert balance.parse_boolean(" false ") is False
    assert balance.parse_boolean("0") is False
    with pytest.raises(ValueError, match="explicit Boolean"):
        balance.parse_boolean("")
    with pytest.raises(ValueError, match="explicit Boolean"):
        balance.parse_boolean("not-a-boolean")


def test_fixed_property_bin_boundaries() -> None:
    assert [balance.gap_bin(value) for value in (0, 0.1, 1, 1.1, 2, 2.1, 3, 3.1)] == [
        "zero",
        "(0,1]",
        "(0,1]",
        "(1,2]",
        "(1,2]",
        "(2,3]",
        "(2,3]",
        ">3",
    ]
    with pytest.raises(ValueError, match="cannot be negative"):
        balance.gap_bin(-0.1)
    assert [
        balance.heat_bin(value)
        for value in (-0.5, -0.49, -0.25, -0.01, 0, 0.5, 1, 2, 2.1)
    ] == [
        "<=-0.50",
        "(-0.50,-0.25]",
        "(-0.50,-0.25]",
        "(-0.25,0)",
        "zero",
        "(0,0.50]",
        "(0.50,1]",
        "(1,2]",
        ">2",
    ]


@pytest.mark.skipif(not HAS_ASSETS, reason="reconstructed CMR CSV assets are absent")
def test_existing_split_and_audit_reproduce_exactly() -> None:
    result = balance.audit_dataset(DATASET_DIR)
    grouping = result["verification"]["grouping"]
    audit = result["verification"]["audit"]
    assert grouping["assignment_mismatches"] == 0
    assert grouping["recalculated_group_mismatches"] == 0
    assert all(
        value == 0
        for pair in grouping["overlaps"].values()
        for value in pair.values()
    )
    assert (audit["audit_rows"], audit["audit_zero_gap_rows"], audit["audit_positive_gap_rows"]) == (
        20,
        19,
        1,
    )
    assert (
        audit["eligible_validation_rows"],
        audit["eligible_validation_zero_gap_rows"],
        audit["eligible_validation_positive_gap_rows"],
    ) == (32, 31, 1)
    assert all(row["matches_setup_snapshot"] for row in result["input_hashes"])

    tables = result["tables"]
    split_counts = {
        row["split"]: row["rows"] for row in tables["split_summary"]
    }
    assert split_counts == {"train": 15143, "val": 1893, "test": 1892, "all": 18928}
    for scope in ("all", "generation_scope"):
        for split in ("train", "val", "test", "all"):
            gap_rows = [
                row
                for row in tables["gap_bins"]
                if row["scope"] == scope and row["split"] == split
            ]
            occupancy_rows = [
                row
                for row in tables["two_property_occupancy"]
                if row["scope"] == scope and row["split"] == split
            ]
            expected = sum(row["rows"] for row in gap_rows)
            assert sum(row["rows"] for row in occupancy_rows) == expected
