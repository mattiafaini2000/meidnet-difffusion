from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.db import connect


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(WORKSPACE_ROOT / "scripts"))

import prepare_cmr_dataset as preparation


def fake_id(number: int) -> str:
    return f"cmr_{number:032x}"


def make_record(material_id: str, group: str) -> dict:
    record = {column: "" for column in preparation.CSV_COLUMNS}
    record.update(
        {
            "material_id": material_id,
            "heat_all": 0.0,
            "dir_gap": 0.0,
            "source_id": int(material_id.split("_")[1], 16),
            "source_unique_id": f"source-{material_id}",
            "source_formula": "BaTiO3",
            "source_combination": "ABO3",
            "source_A_ion": "Ba",
            "source_B_ion": "Ti",
            "source_anion": "O3",
            "heat_of_formation_all_raw": 0.0,
            "gllbsc_dir_gap_raw": 0.0,
            "gllbsc_ind_gap_raw": 0.0,
            "reduced_composition": "BaTiO3",
            "composition_group": group,
            "mixed_anion": False,
            "single_anion": True,
            "generation_scope_compatible": True,
            "cif_filename": f"{material_id}.cif",
            "site_order_preserved": True,
            "site_permutation": "[0, 1, 2, 3, 4]",
        }
    )
    return record


def test_missing_and_zero_labels_are_distinct() -> None:
    heat, direct, indirect = preparation.map_required_labels(
        {
            "heat_of_formation_all": -0.64,
            "gllbsc_dir_gap": 0.0,
            "gllbsc_ind_gap": 1.25,
        }
    )
    assert (heat, direct, indirect) == (-0.64, 0.0, 1.25)
    with pytest.raises(preparation.RecordError, match="missing_gllbsc_dir_gap"):
        preparation.map_required_labels(
            {"heat_of_formation_all": 0.0, "gllbsc_ind_gap": 0.0}
        )
    with pytest.raises(preparation.RecordError, match="nonfinite_gllbsc_dir_gap"):
        preparation.map_required_labels(
            {
                "heat_of_formation_all": 0.0,
                "gllbsc_dir_gap": float("nan"),
                "gllbsc_ind_gap": 0.0,
            }
        )


def test_stable_ids_do_not_trigger_integer_filename_coercion() -> None:
    first = "0123456789abcdef0123456789abcdef"
    second = "1123456789abcdef0123456789abcdef"
    assert preparation.stable_material_id(first) == f"cmr_{first}"
    assert preparation.stable_material_id(first) != preparation.stable_material_id(second)
    with pytest.raises(ValueError, match="Unexpected ASE unique_id"):
        preparation.stable_material_id("7")


def test_mixed_anions_are_retained_but_not_generation_compatible() -> None:
    single = preparation.classify_anion_scope("Ba", "Ti", "O3", ["Ba", "Ti", "O", "O", "O"])
    mixed = preparation.classify_anion_scope("Ba", "Ti", "O2N", ["Ba", "Ti", "O", "O", "N"])
    allowed_nitride = preparation.classify_anion_scope(
        "K", "Ta", "N3", ["K", "Ta", "N", "N", "N"]
    )
    unsupported_nitride_b = preparation.classify_anion_scope(
        "Na", "Zn", "N3", ["Na", "Zn", "N", "N", "N"]
    )
    assert single == (True, False, True)
    assert mixed == (False, True, False)
    assert allowed_nitride == (True, False, True)
    assert unsupported_nitride_b == (True, False, False)


def test_grouped_split_is_deterministic_and_keeps_groups_together() -> None:
    records = [
        make_record(fake_id(1), "Ba:1|O:3|Ti:1"),
        make_record(fake_id(2), "Ba:1|O:3|Ti:1"),
        make_record(fake_id(3), "Ca:1|O:3|Ti:1"),
        make_record(fake_id(4), "N:3|Rb:1|Ta:1"),
        make_record(fake_id(5), "K:1|O:3|Ta:1"),
    ]
    first = preparation.grouped_split(records, seed=42)
    second = preparation.grouped_split(list(reversed(records)), seed=42)
    assert first == second
    assert first[fake_id(1)] == first[fake_id(2)]


def test_cif_round_trip_uses_real_meidnet_parser(tmp_path: Path) -> None:
    atoms = Atoms(
        symbols=["Ba", "Ti", "O", "O", "O"],
        scaled_positions=[
            (0, 0, 0),
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0),
            (0.5, 0, 0.5),
            (0, 0.5, 0.5),
        ],
        cell=np.eye(3) * 4.0,
        pbc=True,
    )
    result = preparation.write_and_validate_cif(atoms, tmp_path / "fixture.cif")
    assert result["site_order_preserved"] is True
    assert result["site_permutation"] == [0, 1, 2, 3, 4]
    assert (
        result["maximum_periodic_position_error_angstrom"]
        < preparation.ROUNDTRIP_POSITION_TOLERANCE_ANGSTROM
    )


def test_existing_raw_db_uses_local_file_route(tmp_path: Path) -> None:
    raw_db = tmp_path / "local.db"
    database = connect(str(raw_db))
    database.write(Atoms("H", positions=[[0, 0, 0]], cell=np.eye(3), pbc=True))
    result = preparation.obtain_raw_database(
        raw_db,
        url="https://invalid.example.invalid/should-not-be-requested",
        retries=1,
    )
    assert result["source_kind"] == "locally_supplied_database"
    assert result["database_rows"] == 1
    assert result["cache_hit"] is True


def test_csv_and_cif_correspondence(tmp_path: Path) -> None:
    records_by_split = {
        "train": [make_record(fake_id(1), "Ba:1|O:3|Ti:1")],
        "val": [make_record(fake_id(2), "Ca:1|O:3|Ti:1")],
        "test": [make_record(fake_id(3), "K:1|O:3|Ta:1")],
    }
    for split, records in records_by_split.items():
        cif_dir = tmp_path / "cifs" / split
        cif_dir.mkdir(parents=True)
        for record in records:
            (cif_dir / record["cif_filename"]).write_text(
                "data_fixture\n_cell_length_a 4.0\n", encoding="utf-8"
            )
        preparation.write_csv(tmp_path / f"{split}.csv", records, cif_dir)

    preparation.validate_exports(
        tmp_path,
        records_by_split,
        excluded_count=0,
        raw_count=3,
    )

    with (tmp_path / "train.csv").open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["material_id"] == fake_id(1)
    assert row["cif"] == (tmp_path / "cifs" / "train" / f"{fake_id(1)}.cif").read_text()
