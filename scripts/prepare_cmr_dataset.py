"""Download and convert the DTU CMR cubic-perovskite database for MEIDNet.

The original MEIDNet split CSVs are not public.  This script therefore creates
new composition-grouped splits while preserving the CMR structures and the raw
``heat_of_formation_all`` and ``gllbsc_dir_gap`` labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import shutil
import sqlite3
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from ase.db import connect
from pymatgen.core import Composition
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.cif import CifWriter
from scipy.optimize import linear_sum_assignment


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
MEIDNET_SOURCE = WORKSPACE_ROOT / "MEIDNet-main"
CMR_PAGE = "https://cmr.fysik.dtu.dk/cubic_perovskites/cubic_perovskites.html"
CMR_URL = "https://wiki.fysik.dtu.dk/cmr-files/cubic_perovskites.db"
CMR_LICENSE = "CC BY-SA 4.0"
CMR_LICENSE_URL = "https://creativecommons.org/licenses/by-sa/4.0/"
ROUNDTRIP_POSITION_TOLERANCE_ANGSTROM = 1e-3

A_CATIONS = {
    "Ba", "Sr", "Ca", "Na", "K", "Rb", "Cs",
    "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho",
    "Er", "Tm", "Yb", "Lu",
}
GENERATION_ANIONS = {"O", "F", "Cl", "Br", "I", "S", "Se", "Te", "N"}
# Source single-anion records are oxides and nitrides. These sets mirror the
# family-specific B-site tables used by MEIDNet's generation code; they are a
# scope annotation, not a stability or charge-neutrality filter.
GENERATION_B_BY_ANION = {
    "O": {
        "Ti", "Zr", "Hf", "V", "Nb", "Ta", "Cr", "Mn", "Fe", "Co",
        "Ni", "Cu", "Zn", "Sc", "Y", "Al", "Ga", "In", "Ge", "Sn",
        "Pb", "W", "Mo",
    },
    "N": {"W", "Mo", "Ta", "Nb", "Ti", "Zr", "Hf", "Cr", "Mn", "Fe", "Co", "Ni"},
}

CSV_COLUMNS = [
    "material_id",
    "heat_all",
    "dir_gap",
    "cif",
    "source_id",
    "source_unique_id",
    "source_project",
    "source_formula",
    "source_combination",
    "source_A_ion",
    "source_B_ion",
    "source_anion",
    "heat_of_formation_all_raw",
    "gllbsc_dir_gap_raw",
    "gllbsc_ind_gap_raw",
    "source_standard_energy",
    "source_total_energy",
    "source_CB_dir",
    "source_CB_ind",
    "source_VB_dir",
    "source_VB_ind",
    "reduced_composition",
    "composition_group",
    "mixed_anion",
    "single_anion",
    "generation_scope_compatible",
    "cif_filename",
    "source_site_symbols",
    "meidnet_site_symbols",
    "positions_wrapped_by_meidnet_parser",
    "site_order_preserved",
    "site_permutation",
]


class RecordError(ValueError):
    """A source row cannot be exported without changing its meaning."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def validate_sqlite_database(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"Raw database does not exist: {path}")
    if path.stat().st_size < 4_096:
        raise ValueError(f"Raw database is unexpectedly small: {path.stat().st_size} bytes")

    with path.open("rb") as handle:
        prefix = handle.read(32)
    if prefix.lstrip().lower().startswith((b"<!doctype html", b"<html")):
        raise ValueError("Downloaded file is HTML, not an ASE database")
    if not prefix.startswith(b"SQLite format 3\x00"):
        raise ValueError("Downloaded file does not have a SQLite database header")

    uri = path.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        result = connection.execute("PRAGMA quick_check").fetchone()[0]
        if result != "ok":
            raise ValueError(f"SQLite quick_check failed: {result}")
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        required = {"systems", "species", "keys", "information"}
        if not required.issubset(tables):
            missing = sorted(required - tables)
            raise ValueError(f"Not an ASE SQLite database; missing tables: {missing}")
        row_count = connection.execute("SELECT COUNT(*) FROM systems").fetchone()[0]
        version_row = connection.execute(
            "SELECT value FROM information WHERE name='version'"
        ).fetchone()

    return {
        "sqlite_quick_check": result,
        "ase_schema_version": int(version_row[0]) if version_row else None,
        "database_rows": int(row_count),
    }


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def obtain_raw_database(
    raw_db: Path,
    url: str = CMR_URL,
    timeout: float = 60.0,
    retries: int = 3,
) -> dict[str, Any]:
    """Use a valid local database or download it with bounded atomic retries."""
    provenance_path = raw_db.with_suffix(raw_db.suffix + ".provenance.json")
    if raw_db.exists():
        validation = validate_sqlite_database(raw_db)
        saved = load_json(provenance_path)
        if saved and saved.get("sha256") == sha256_file(raw_db):
            return {**saved, **validation, "cache_hit": True}
        return {
            "source_kind": "locally_supplied_database",
            "requested_url": None,
            "resolved_url": None,
            "retrieved_at_utc": None,
            "sha256": sha256_file(raw_db),
            "size_bytes": raw_db.stat().st_size,
            "cache_hit": True,
            **validation,
        }

    raw_db.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        temporary: Path | None = None
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "MEIDNet-dataset-preparation/1.0"},
            )
            retrieved_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
            with urllib.request.urlopen(request, timeout=timeout) as response:
                with tempfile.NamedTemporaryFile(
                    "wb", dir=raw_db.parent, prefix=".cmr-download-", suffix=".part", delete=False
                ) as handle:
                    temporary = Path(handle.name)
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
                    handle.flush()
                    os.fsync(handle.fileno())
                response_metadata = {
                    "source_kind": "downloaded_cmr_database",
                    "requested_url": url,
                    "resolved_url": response.geturl(),
                    "retrieved_at_utc": retrieved_at,
                    "http_last_modified": response.headers.get("Last-Modified"),
                    "http_etag": response.headers.get("ETag"),
                    "http_content_length": response.headers.get("Content-Length"),
                    "http_content_type": response.headers.get("Content-Type"),
                }

            validation = validate_sqlite_database(temporary)
            result = {
                **response_metadata,
                "sha256": sha256_file(temporary),
                "size_bytes": temporary.stat().st_size,
                "cache_hit": False,
                **validation,
            }
            os.replace(temporary, raw_db)
            atomic_json(provenance_path, result)
            return result
        except (OSError, TimeoutError, urllib.error.URLError, ValueError) as error:
            last_error = error
            if temporary and temporary.exists():
                temporary.unlink()
            if attempt < retries:
                time.sleep(2 ** (attempt - 1))

    raise RuntimeError(
        f"Failed to download and validate {url} after {retries} attempts: {last_error}"
    ) from last_error


def required_finite(values: dict[str, Any], key: str) -> float:
    if key not in values:
        raise RecordError(f"missing_{key}")
    value = float(values[key])
    if not math.isfinite(value):
        raise RecordError(f"nonfinite_{key}")
    return value


def map_required_labels(values: dict[str, Any]) -> tuple[float, float, float]:
    """Copy the three source labels without normalization or imputation."""
    return (
        required_finite(values, "heat_of_formation_all"),
        required_finite(values, "gllbsc_dir_gap"),
        required_finite(values, "gllbsc_ind_gap"),
    )


def stable_material_id(source_unique_id: str) -> str:
    token = str(source_unique_id).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{32}", token):
        raise ValueError(f"Unexpected ASE unique_id: {source_unique_id!r}")
    return f"cmr_{token}"


def composition_group(symbols: Iterable[str]) -> tuple[str, str]:
    composition = Composition(Counter(symbols))
    reduced, _ = composition.get_reduced_composition_and_factor()
    pieces = []
    for symbol, amount in sorted(reduced.get_el_amt_dict().items()):
        rounded = round(float(amount))
        amount_text = str(rounded) if abs(amount - rounded) < 1e-10 else f"{amount:.12g}"
        pieces.append(f"{symbol}:{amount_text}")
    return reduced.reduced_formula, "|".join(pieces)


def classify_anion_scope(
    a_ion: str,
    b_ion: str,
    source_anion: str,
    symbols: Iterable[str],
) -> tuple[bool, bool, bool]:
    anion_elements = {str(element) for element in Composition(source_anion).elements}
    single_anion = len(anion_elements) == 1
    mixed_anion = len(anion_elements) > 1
    expected_symbols = [a_ion, b_ion]
    if single_anion:
        anion = next(iter(anion_elements))
        expected_symbols.extend([anion] * 3)
    else:
        anion = ""
    generation_compatible = (
        single_anion
        and a_ion in A_CATIONS
        and b_ion in GENERATION_B_BY_ANION.get(anion, set())
        and anion_elements.issubset(GENERATION_ANIONS)
        and Counter(symbols) == Counter(expected_symbols)
    )
    return single_anion, mixed_anion, generation_compatible


def periodic_assignment(
    source_symbols: list[str],
    source_fractional: np.ndarray,
    parsed_symbols: list[str],
    parsed_fractional: np.ndarray,
    lattice: np.ndarray,
) -> tuple[list[int], float]:
    count = len(source_symbols)
    if len(parsed_symbols) != count:
        raise RecordError("roundtrip_atom_count_changed")

    costs = np.full((count, count), 1e9, dtype=float)
    for i, source_symbol in enumerate(source_symbols):
        for j, parsed_symbol in enumerate(parsed_symbols):
            if source_symbol != parsed_symbol:
                continue
            delta = source_fractional[i] - parsed_fractional[j]
            delta -= np.round(delta)
            costs[i, j] = np.linalg.norm(delta @ lattice)

    source_indices, parsed_indices = linear_sum_assignment(costs)
    if not np.array_equal(source_indices, np.arange(count)):
        raise RecordError("roundtrip_assignment_failed")
    maximum_distance = float(costs[source_indices, parsed_indices].max())
    if maximum_distance > ROUNDTRIP_POSITION_TOLERANCE_ANGSTROM:
        raise RecordError(f"roundtrip_position_mismatch_{maximum_distance:.3e}_angstrom")
    return parsed_indices.astype(int).tolist(), maximum_distance


def write_and_validate_cif(atoms: Any, path: Path) -> dict[str, Any]:
    """Write without refinement and round-trip through MEIDNet's real parser."""
    source_symbols = atoms.get_chemical_symbols()
    source_fractional = atoms.get_scaled_positions(wrap=False)
    source_lattice = np.asarray(atoms.cell.array, dtype=float)
    source_cell_parameters = np.asarray(atoms.cell.cellpar(), dtype=float)

    structure = AseAtomsAdaptor.get_structure(atoms)
    writer = CifWriter(
        structure,
        symprec=None,
        significant_figures=12,
        refine_struct=False,
    )
    cif_text = str(writer)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(cif_text, encoding="utf-8", newline="\n")

    sys.dont_write_bytecode = True
    source_path = str(MEIDNET_SOURCE.resolve())
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
    from meidnet.model import MAX_SITES, NUM_SPECIES, SPECIES_LIST, parse_cif_to_dense

    dense = parse_cif_to_dense(str(path), max_sites=MAX_SITES)
    if not np.isfinite(dense).all():
        raise RecordError("meidnet_dense_vector_nonfinite")

    species_offset = 6 + MAX_SITES * MAX_SITES
    species_length = MAX_SITES * NUM_SPECIES
    species = dense[species_offset : species_offset + species_length].reshape(
        MAX_SITES, NUM_SPECIES
    )
    coordinates = dense[species_offset + species_length :].reshape(MAX_SITES, 3)
    occupied = np.flatnonzero(species.sum(axis=1) > 0.5)
    if len(occupied) != len(source_symbols):
        raise RecordError("meidnet_roundtrip_atom_count_changed")
    parsed_symbols = [SPECIES_LIST[int(species[index].argmax())] for index in occupied]
    parsed_fractional = coordinates[occupied]

    if Counter(parsed_symbols) != Counter(source_symbols):
        raise RecordError("meidnet_roundtrip_composition_changed")
    parsed_cell_parameters = np.concatenate((dense[:3] * 20.0, dense[3:6] * 180.0))
    if not np.allclose(parsed_cell_parameters, source_cell_parameters, atol=2e-5, rtol=0):
        raise RecordError("meidnet_roundtrip_lattice_changed")

    permutation, maximum_distance = periodic_assignment(
        source_symbols,
        source_fractional,
        parsed_symbols,
        parsed_fractional,
        source_lattice,
    )
    return {
        "cif_text": cif_text,
        "site_permutation": permutation,
        "site_order_preserved": permutation == list(range(len(source_symbols))),
        "parsed_symbols": parsed_symbols,
        "positions_wrapped": bool(
            np.any(source_fractional < 0.0) or np.any(source_fractional >= 1.0)
        ),
        "maximum_periodic_position_error_angstrom": maximum_distance,
    }


def row_values(row: Any) -> dict[str, Any]:
    return dict(row.key_value_pairs)


def convert_database(raw_db: Path, all_cifs: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    database = connect(str(raw_db))
    retained: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []

    for row in database.select():
        source_id = int(row.id)
        unique_id = str(row.unique_id)
        values = row_values(row)
        cif_path: Path | None = None
        try:
            if "reference" in values:
                raise RecordError("reference_record")

            heat, direct_gap, indirect_gap = map_required_labels(values)
            atoms = row.toatoms()
            if len(atoms) != 5:
                raise RecordError(f"atom_count_{len(atoms)}_not_five")
            if not bool(np.asarray(atoms.pbc, dtype=bool).all()):
                raise RecordError("cell_not_fully_periodic")
            lattice = np.asarray(atoms.cell.array, dtype=float)
            fractional = atoms.get_scaled_positions(wrap=False)
            if not np.isfinite(lattice).all() or abs(np.linalg.det(lattice)) < 1e-8:
                raise RecordError("invalid_periodic_cell")
            if not np.isfinite(fractional).all():
                raise RecordError("nonfinite_fractional_positions")
            symbols = atoms.get_chemical_symbols()
            if any(atoms.numbers < 1) or any(atoms.numbers > 118):
                raise RecordError("species_not_representable_by_meidnet")

            material_id = stable_material_id(unique_id)
            cif_path = all_cifs / f"{material_id}.cif"
            roundtrip = write_and_validate_cif(atoms, cif_path)
            reduced_formula, group_key = composition_group(symbols)

            a_ion = str(values.get("A_ion", ""))
            b_ion = str(values.get("B_ion", ""))
            source_anion = str(values.get("anion", ""))
            single_anion, mixed_anion, generation_compatible = classify_anion_scope(
                a_ion,
                b_ion,
                source_anion,
                symbols,
            )

            retained.append(
                {
                    "material_id": material_id,
                    "heat_all": heat,
                    "dir_gap": direct_gap,
                    "source_id": source_id,
                    "source_unique_id": unique_id,
                    "source_project": str(values.get("project", "")),
                    "source_formula": str(row.formula),
                    "source_combination": str(values.get("combination", "")),
                    "source_A_ion": a_ion,
                    "source_B_ion": b_ion,
                    "source_anion": source_anion,
                    "heat_of_formation_all_raw": heat,
                    "gllbsc_dir_gap_raw": direct_gap,
                    "gllbsc_ind_gap_raw": indirect_gap,
                    "source_standard_energy": float(values["standard_energy"]),
                    "source_total_energy": float(row.energy),
                    "source_CB_dir": float(values["CB_dir"]),
                    "source_CB_ind": float(values["CB_ind"]),
                    "source_VB_dir": float(values["VB_dir"]),
                    "source_VB_ind": float(values["VB_ind"]),
                    "reduced_composition": reduced_formula,
                    "composition_group": group_key,
                    "mixed_anion": mixed_anion,
                    "single_anion": single_anion,
                    "generation_scope_compatible": generation_compatible,
                    "cif_filename": f"{material_id}.cif",
                    "source_site_symbols": json.dumps(symbols),
                    "meidnet_site_symbols": json.dumps(roundtrip["parsed_symbols"]),
                    "positions_wrapped_by_meidnet_parser": roundtrip["positions_wrapped"],
                    "site_order_preserved": roundtrip["site_order_preserved"],
                    "site_permutation": json.dumps(roundtrip["site_permutation"]),
                    "maximum_periodic_position_error_angstrom": roundtrip[
                        "maximum_periodic_position_error_angstrom"
                    ],
                }
            )
        except Exception as error:
            if cif_path is not None and cif_path.exists():
                cif_path.unlink()
            reason = str(error) if isinstance(error, RecordError) else f"{type(error).__name__}: {error}"
            excluded.append(
                {
                    "source_id": source_id,
                    "source_unique_id": unique_id,
                    "reason": reason,
                }
            )

        if source_id % 1000 == 0:
            print(f"  inspected {source_id} source rows", flush=True)

    retained.sort(key=lambda record: record["material_id"])
    excluded.sort(key=lambda record: record["source_id"])
    return retained, excluded


def grouped_split(
    records: list[dict[str, Any]],
    seed: int = 42,
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> dict[str, str]:
    if len(fractions) != 3 or any(fraction < 0 for fraction in fractions):
        raise ValueError("Expected three non-negative split fractions")
    if not math.isclose(sum(fractions), 1.0, abs_tol=1e-12):
        raise ValueError("Split fractions must sum to one")

    groups: dict[str, list[str]] = defaultdict(list)
    for record in sorted(records, key=lambda item: item["material_id"]):
        groups[record["composition_group"]].append(record["material_id"])

    group_keys = sorted(groups)
    random.Random(seed).shuffle(group_keys)
    names = ("train", "val", "test")
    targets = dict(zip(names, (len(records) * fraction for fraction in fractions)))
    counts = {name: 0 for name in names}
    assignments: dict[str, str] = {}

    for group_key in group_keys:
        split = max(names, key=lambda name: targets[name] - counts[name])
        for material_id in groups[group_key]:
            assignments[material_id] = split
        counts[split] += len(groups[group_key])
    return assignments


def select_audit(records: list[dict[str, Any]], size: int) -> list[dict[str, Any]]:
    candidates = sorted(
        (record for record in records if record["generation_scope_compatible"]),
        key=lambda record: record["material_id"],
    )
    if size <= 0 or not candidates:
        return []
    if len(candidates) <= size:
        return candidates

    properties = np.asarray(
        [[record["heat_all"], record["dir_gap"]] for record in candidates], dtype=float
    )
    minima = properties.min(axis=0)
    ranges = properties.max(axis=0) - minima
    ranges[ranges == 0] = 1.0
    scaled = (properties - minima) / ranges

    extreme_indices = []
    for column in range(2):
        extreme_indices.extend([int(np.argmin(scaled[:, column])), int(np.argmax(scaled[:, column]))])
    selected = list(dict.fromkeys(extreme_indices))

    while len(selected) < size:
        remaining = [index for index in range(len(candidates)) if index not in selected]
        distances = []
        for index in remaining:
            closest = min(np.linalg.norm(scaled[index] - scaled[chosen]) for chosen in selected)
            distances.append((float(closest), candidates[index]["material_id"], index))
        best_distance = max(item[0] for item in distances)
        best = min(
            (item for item in distances if math.isclose(item[0], best_distance, abs_tol=1e-15)),
            key=lambda item: item[1],
        )
        selected.append(best[2])

    return sorted((candidates[index] for index in selected[:size]), key=lambda item: item["material_id"])


def write_csv(path: Path, records: list[dict[str, Any]], cifs_dir: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for record in sorted(records, key=lambda item: item["material_id"]):
            row = {column: record[column] for column in CSV_COLUMNS if column != "cif"}
            row["cif"] = (cifs_dir / record["cif_filename"]).read_text(encoding="utf-8")
            writer.writerow(row)


def validate_exports(
    output_dir: Path,
    records_by_split: dict[str, list[dict[str, Any]]],
    excluded_count: int,
    raw_count: int,
) -> None:
    id_sets = {
        split: {record["material_id"] for record in records}
        for split, records in records_by_split.items()
    }
    group_sets = {
        split: {record["composition_group"] for record in records}
        for split, records in records_by_split.items()
    }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if id_sets[left] & id_sets[right]:
            raise AssertionError(f"IDs overlap between {left} and {right}")
        if group_sets[left] & group_sets[right]:
            raise AssertionError(f"Composition groups overlap between {left} and {right}")

    retained_count = sum(len(records) for records in records_by_split.values())
    if retained_count + excluded_count != raw_count:
        raise AssertionError("Retained and excluded rows do not account for the source database")
    all_ids = set().union(*id_sets.values())
    if len(all_ids) != retained_count:
        raise AssertionError("Duplicate material IDs were generated")
    if len({material_id.casefold() for material_id in all_ids}) != retained_count:
        raise AssertionError("Case-insensitive material ID or filename collision")
    if any(not re.fullmatch(r"cmr_[0-9a-f]{32}", material_id) for material_id in all_ids):
        raise AssertionError("A material ID is not a source-derived CMR unique_id")

    for split, records in records_by_split.items():
        csv_path = output_dir / f"{split}.csv"
        frame = pd.read_csv(csv_path, dtype={"material_id": "string"})
        csv_ids = set(frame["material_id"].astype(str))
        cif_dir = output_dir / "cifs" / split
        file_ids = {path.stem for path in cif_dir.glob("*.cif")}
        if csv_ids != id_sets[split] or file_ids != id_sets[split]:
            raise AssertionError(f"CSV/CIF correspondence failed for {split}")
        if frame["material_id"].duplicated().any():
            raise AssertionError(f"Duplicate material IDs in {csv_path}")
        values = frame[["heat_all", "dir_gap"]].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise AssertionError(f"Non-finite required values in {csv_path}")
        for row in frame.itertuples(index=False):
            text = (cif_dir / f"{row.material_id}.cif").read_text(encoding="utf-8")
            if row.cif != text:
                raise AssertionError(f"Embedded/file CIF mismatch for {row.material_id}")


def directory_digest(path: Path) -> str:
    digest = hashlib.sha256()
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = file_path.relative_to(path).as_posix().encode("utf-8")
        contents_hash = sha256_file(file_path).encode("ascii")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(contents_hash)
    return digest.hexdigest()


def remove_staging_directory(path: Path, staging_parent: Path) -> None:
    """Remove only a temporary directory created by this preparation script."""
    resolved_path = path.resolve()
    resolved_parent = staging_parent.resolve()
    if (
        resolved_path.parent != resolved_parent
        or not resolved_path.name.startswith("cmr_reconstructed-")
    ):
        raise RuntimeError(f"Refusing to remove unexpected staging path: {resolved_path}")
    shutil.rmtree(resolved_path)


def prepare(
    raw_db: Path,
    output_dir: Path,
    raw_provenance: dict[str, Any],
    seed: int,
    fractions: tuple[float, float, float],
    audit_size: int,
    check_determinism: bool,
) -> dict[str, Any]:
    if output_dir.exists():
        manifest = load_json(output_dir / "manifest.json")
        if not manifest:
            raise FileExistsError(f"Refusing to overwrite existing output without a manifest: {output_dir}")
        if manifest.get("raw_source", {}).get("sha256") != raw_provenance["sha256"]:
            raise FileExistsError("Existing output was made from a different raw database")
        expected_fractions = dict(zip(("train", "val", "test"), fractions))
        existing_split = manifest.get("split", {})
        existing_audit = manifest.get("audit", {})
        if (
            existing_split.get("seed") != seed
            or existing_split.get("requested_fractions") != expected_fractions
            or existing_audit.get("requested_size") != audit_size
        ):
            raise FileExistsError(
                "Existing output uses a different seed, split fraction, or audit size; "
                "choose a new --output-dir instead of overwriting it"
            )
        if not check_determinism:
            print(f"Using cached output with matching source and configuration: {output_dir}")
            return manifest

    staging_parent = WORKSPACE_ROOT / "data" / ".staging"
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="cmr_reconstructed-", dir=staging_parent))
    try:
        all_cifs = staging / "_all_cifs"
        all_cifs.mkdir()
        retained, excluded = convert_database(raw_db, all_cifs)
        assignments = grouped_split(retained, seed=seed, fractions=fractions)
        records_by_split = {name: [] for name in ("train", "val", "test")}
        for record in retained:
            split = assignments[record["material_id"]]
            record["split"] = split
            records_by_split[split].append(record)

        for split, records in records_by_split.items():
            cif_dir = staging / "cifs" / split
            cif_dir.mkdir(parents=True)
            for record in records:
                (all_cifs / record["cif_filename"]).replace(cif_dir / record["cif_filename"])
            write_csv(staging / f"{split}.csv", records, cif_dir)
        all_cifs.rmdir()

        validation_records = records_by_split["val"]
        audit = select_audit(validation_records, audit_size)
        write_csv(staging / "audit.csv", audit, staging / "cifs" / "val")

        with (staging / "id_mapping.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["material_id", "source_id", "source_unique_id", "split", "cif_filename"],
                lineterminator="\n",
            )
            writer.writeheader()
            for record in retained:
                writer.writerow({key: record[key] for key in writer.fieldnames})

        with (staging / "exclusions.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["source_id", "source_unique_id", "reason"],
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(excluded)

        validate_exports(staging, records_by_split, len(excluded), raw_provenance["database_rows"])
        split_counts = {name: len(records) for name, records in records_by_split.items()}
        retained_count = len(retained)
        manifest = {
            "original_splits_recovered": False,
            "dataset_status": "CMR-derived MEIDNet-format dataset; newly generated splits",
            "pretrained_training_overlap": "unknown",
            "property_checkpoint_compatibility": "provisional",
            "property_conditioned_checkpoint_evaluation": "blocked pending original-label compatibility evidence",
            "raw_source": raw_provenance,
            "source_page": CMR_PAGE,
            "license": CMR_LICENSE,
            "license_url": CMR_LICENSE_URL,
            "label_mapping": {
                "dir_gap": "gllbsc_dir_gap, direct GLLB-SC band gap, eV; zero is retained",
                "heat_all": "raw heat_of_formation_all, eV/atom; finite-pool DFT formation/stability energy",
                "heat_reference_note": (
                    "Uses the CMR reference pool, including H2O(g)-H2(g) for oxygen. "
                    "It is not energy above hull. Raw 0.02 eV/atom quantization is preserved."
                ),
            },
            "selection": {
                "rule": "non-reference, five-atom, finite labels, representable ordered species, valid periodic cell, MEIDNet CIF round trip",
                "raw_rows": raw_provenance["database_rows"],
                "retained_rows": retained_count,
                "excluded_rows": len(excluded),
                "exclusion_reasons": dict(Counter(item["reason"] for item in excluded)),
                "mixed_anion_rows": sum(record["mixed_anion"] for record in retained),
                "single_anion_rows": sum(record["single_anion"] for record in retained),
                "generation_scope_compatible_rows": sum(
                    record["generation_scope_compatible"] for record in retained
                ),
                "generation_scope_note": (
                    "Single anion plus MEIDNet A-site and family-specific B-site species; "
                    "no charge, geometry, or stability success filter was applied."
                ),
                "site_order_preserved_rows": sum(record["site_order_preserved"] for record in retained),
                "site_reordered_rows": sum(not record["site_order_preserved"] for record in retained),
                "maximum_roundtrip_position_error_angstrom": max(
                    record["maximum_periodic_position_error_angstrom"] for record in retained
                ),
                "roundtrip_position_tolerance_angstrom": ROUNDTRIP_POSITION_TOLERANCE_ANGSTROM,
            },
            "split": {
                "algorithm": "sort material IDs; group by reduced composition; seeded group shuffle; greedy remaining-capacity assignment",
                "seed": seed,
                "requested_fractions": dict(zip(("train", "val", "test"), fractions)),
                "counts": split_counts,
                "actual_fractions": {
                    name: count / retained_count for name, count in split_counts.items()
                },
                "composition_groups": {
                    name: len({record["composition_group"] for record in records})
                    for name, records in records_by_split.items()
                },
            },
            "audit": {
                "requested_size": audit_size,
                "actual_size": len(audit),
                "source_split": "val",
                "selection": "deterministic two-property maximin over generation-scope-compatible validation rows",
                "material_ids": [record["material_id"] for record in audit],
                "heat_all_range": [
                    min(record["heat_all"] for record in audit) if audit else None,
                    max(record["heat_all"] for record in audit) if audit else None,
                ],
                "dir_gap_range": [
                    min(record["dir_gap"] for record in audit) if audit else None,
                    max(record["dir_gap"] for record in audit) if audit else None,
                ],
            },
        }
        atomic_json(staging / "manifest.json", manifest)
        manifest["export_sha256_before_manifest_hash_field"] = directory_digest(staging)
        atomic_json(staging / "manifest.json", manifest)

        if output_dir.exists():
            existing_digest = directory_digest(output_dir)
            rebuilt_digest = directory_digest(staging)
            if existing_digest != rebuilt_digest:
                raise AssertionError(
                    "Determinism check failed: rebuilt output differs from the existing output"
                )
            print(f"Determinism check passed: {rebuilt_digest}")
            return load_json(output_dir / "manifest.json") or manifest

        output_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, output_dir)
        print(f"Prepared dataset: {output_dir}")
        return manifest
    finally:
        if staging.exists():
            remove_staging_directory(staging, staging_parent)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-db",
        type=Path,
        default=WORKSPACE_ROOT / "data" / "raw" / "cmr" / "cubic_perovskites.db",
        help="Cached download destination or an existing locally supplied ASE database",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WORKSPACE_ROOT / "data" / "processed" / "cmr_reconstructed",
    )
    parser.add_argument("--download-url", default=CMR_URL)
    parser.add_argument("--download-timeout", type=float, default=60.0)
    parser.add_argument("--download-retries", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--audit-size", type=int, default=20)
    parser.add_argument(
        "--check-determinism",
        action="store_true",
        help="Rebuild in staging and require byte-identical output when output already exists",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_db = args.raw_db.resolve()
    output_dir = args.output_dir.resolve()
    provenance = obtain_raw_database(
        raw_db,
        url=args.download_url,
        timeout=args.download_timeout,
        retries=args.download_retries,
    )
    print(
        f"Raw database: {raw_db} ({provenance['size_bytes']} bytes, "
        f"sha256={provenance['sha256']})",
        flush=True,
    )
    manifest = prepare(
        raw_db=raw_db,
        output_dir=output_dir,
        raw_provenance=provenance,
        seed=args.seed,
        fractions=(args.train_fraction, args.val_fraction, args.test_fraction),
        audit_size=args.audit_size,
        check_determinism=args.check_determinism,
    )
    print(json.dumps({"selection": manifest["selection"], "split": manifest["split"], "audit": manifest["audit"]}, indent=2))


if __name__ == "__main__":
    main()
